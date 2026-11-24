from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from arx_shared.config import load_yaml
from arx_shared.files import ensure_dir, new_run_dir
from arx_shared.schemas import (
    CaptureResult,
    EndPoseRequest,
    ExecutionResult,
    GraspPlanRequest,
    GraspPredictRequest,
    RunGraspRequest,
    RunGraspResult,
    SegmentationResult,
    ZeroGraspInputRequest,
)

from .grasp import prepare_zerograsp_input
from .model import predict_grasp
from .planning import plan_grasp


def _dump_model(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _safe_name(text: str) -> str:
    name = text.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "target"


def _parse_object_index(path: Path) -> int | None:
    stem = path.stem
    if stem.endswith(".grasp"):
        stem = stem[: -len(".grasp")]
    maybe_idx = stem.split("_")[-1]
    return int(maybe_idx) if maybe_idx.isdigit() else None


def _candidate_from_row(row: np.ndarray, source_file: str, object_index: int | None, row_index: int) -> dict[str, Any]:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.shape[0] < 16:
        raise ValueError(f"grasp row must contain at least 16 values, got {values.shape[0]}")
    return {
        "source_file": source_file,
        "object_index": object_index,
        "row_index": int(row_index),
        "score": float(values[0]),
        "width_m": float(values[1]),
        "height_m": float(values[2]),
        "depth_m": float(values[3]),
        "rotation_matrix_camera": values[4:13].reshape(3, 3).tolist(),
        "translation_m_camera": values[13:16].tolist(),
        "object_id": int(values[16]) if values.shape[0] > 16 else None,
    }


def _grasp_candidate_paths(prediction_output_dir: str | None, raw_report: dict[str, Any] | None) -> list[Path]:
    raw_outputs_dir = None
    if isinstance(raw_report, dict):
        raw_outputs_dir = raw_report.get("raw_outputs_dir")
    if raw_outputs_dir:
        root = Path(str(raw_outputs_dir))
    elif prediction_output_dir:
        root = Path(prediction_output_dir) / "raw_outputs"
    else:
        return []
    return sorted(root.glob("*.grasp.npy"))


def _iter_grasp_candidates(prediction) -> list[dict[str, Any]]:
    raw_report = prediction.raw.get("run_report") if isinstance(prediction.raw, dict) else None
    paths = _grasp_candidate_paths(prediction.output_dir, raw_report)
    candidates: list[dict[str, Any]] = []
    for grasp_path in paths:
        grasps = np.asarray(np.load(grasp_path), dtype=np.float64)
        if grasps.size == 0:
            continue
        grasps = grasps.reshape(1, -1) if grasps.ndim == 1 else grasps
        object_index = _parse_object_index(grasp_path)
        for row_index, row in enumerate(grasps):
            candidates.append(_candidate_from_row(row, grasp_path.name, object_index, row_index))
    return candidates


def _apply_target_x_filter(
    *,
    prediction,
    current_end_pos: list[float],
    request: RunGraspRequest,
    max_target_x_m: float | None,
) -> dict[str, Any]:
    original_top1 = prediction.raw.get("recommended_grasp_top1") if isinstance(prediction.raw, dict) else None
    if max_target_x_m is None:
        return original_top1
    max_x = float(max_target_x_m)
    if max_x <= 0.0:
        return original_top1

    candidates = _iter_grasp_candidates(prediction)
    if not candidates:
        raise RuntimeError("target-x filter enabled but no raw ZeroGrasp candidates were found")

    records: list[dict[str, Any]] = []
    accepted: list[tuple[float, dict[str, Any], GraspPlanRequest]] = []
    for candidate in candidates:
        record = {
            "source_file": candidate.get("source_file"),
            "row_index": candidate.get("row_index"),
            "score": candidate.get("score"),
        }
        try:
            candidate_plan = plan_grasp(GraspPlanRequest(
                arm_id=request.arm_id,
                prediction=candidate,
                current_end_pos=current_end_pos,
                allow_debug_zero_pose=request.allow_debug_zero_pose,
                execute=request.execute,
            ))
            target_pose = candidate_plan.target_pose or []
            target_x = float(target_pose[0]) if len(target_pose) >= 1 else float("nan")
            record.update({
                "target_x_m": target_x,
                "safety_passed": candidate_plan.safety_passed,
                "accepted": bool(candidate_plan.safety_passed and target_x <= max_x),
            })
            if candidate_plan.safety_failures:
                record["safety_failures"] = candidate_plan.safety_failures
            if record["accepted"]:
                accepted.append((float(candidate.get("score", 0.0)), candidate, GraspPlanRequest(
                    arm_id=request.arm_id,
                    prediction=candidate,
                    current_end_pos=current_end_pos,
                    allow_debug_zero_pose=request.allow_debug_zero_pose,
                    execute=request.execute,
                )))
        except Exception as exc:
            record.update({"accepted": False, "error": str(exc)})
        records.append(record)

    filter_report = {
        "enabled": True,
        "max_target_x_m": max_x,
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "original_recommended_grasp_top1": original_top1,
        "candidates": records,
    }
    prediction.raw["target_x_filter"] = filter_report

    if not accepted:
        prediction.raw["recommended_grasp_top1"] = None
        raise RuntimeError(f"no ZeroGrasp candidates satisfy target_pose.x <= {max_x:.3f} m")

    _, selected, _ = max(accepted, key=lambda item: item[0])
    filter_report["selected"] = {
        "source_file": selected.get("source_file"),
        "row_index": selected.get("row_index"),
        "score": selected.get("score"),
    }
    prediction.raw["recommended_grasp_top1"] = selected
    if prediction.output_dir:
        selected_path = Path(prediction.output_dir) / "recommended_grasp_target_x_filtered.json"
        selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        prediction.grasp_path = str(selected_path)
    return selected


def _project_camera_point(point: np.ndarray, camera_matrix: np.ndarray) -> tuple[int, int] | None:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    x, y, z = point.tolist()
    if z <= 0.0:
        return None
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    return (u, v)


def _unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return arr / norm


def _draw_arrow(draw, start: tuple[int, int], end: tuple[int, int], fill: tuple[int, int, int], width: int = 3) -> None:
    import math

    draw.line([start, end], fill=fill, width=width)
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    ux, uy = dx / length, dy / length
    size = 12.0
    left = (end[0] - ux * size - uy * size * 0.45, end[1] - uy * size + ux * size * 0.45)
    right = (end[0] - ux * size + uy * size * 0.45, end[1] - uy * size - ux * size * 0.45)
    draw.polygon([end, left, right], fill=fill)


def _draw_cross(draw, center: tuple[int, int], radius: int, fill: tuple[int, int, int]) -> None:
    x, y = center
    draw.line([(x - radius, y), (x + radius, y)], fill=fill, width=3)
    draw.line([(x, y - radius), (x, y + radius)], fill=fill, width=3)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 0, 0), width=2)


def _draw_grasp_projection(
    *,
    rgb_path: str,
    camera_info_path: str,
    grasp: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    camera_info = json.loads(Path(camera_info_path).read_text(encoding="utf-8"))
    K = np.asarray(camera_info["cam_K"], dtype=np.float64).reshape(3, 3)
    position = np.asarray(grasp["translation_m_camera"], dtype=np.float64).reshape(3)
    rotation = np.asarray(grasp["rotation_matrix_camera"], dtype=np.float64).reshape(3, 3)
    width_m = float(grasp.get("width_m", 0.04) or 0.04)
    depth_m = float(grasp.get("depth_m", 0.02) or 0.02)

    center = _project_camera_point(position, K)
    if center is None:
        raise ValueError(f"cannot project grasp center behind camera: {position.tolist()}")

    # Same convention as vendor/maniskill_codex/grasp_axes.py: width=-R[:,1], approach=-R[:,0].
    width_axis = _unit_vector(-rotation[:, 1], fallback=np.array([0.0, -1.0, 0.0]))
    approach_axis = _unit_vector(-rotation[:, 0], fallback=np.array([0.0, 0.0, -1.0]))
    width_a = _project_camera_point(position + width_axis * (width_m / 2.0), K)
    width_b = _project_camera_point(position - width_axis * (width_m / 2.0), K)
    approach_tip = _project_camera_point(position - approach_axis * max(0.06, depth_m * 2.5), K)

    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    green = (0, 255, 0)
    yellow = (255, 220, 0)
    black = (0, 0, 0)

    if width_a is not None and width_b is not None:
        draw.line([width_a, width_b], fill=green, width=4)
        draw.ellipse((width_a[0] - 4, width_a[1] - 4, width_a[0] + 4, width_a[1] + 4), fill=yellow, outline=black)
        draw.ellipse((width_b[0] - 4, width_b[1] - 4, width_b[0] + 4, width_b[1] + 4), fill=yellow, outline=black)
    if approach_tip is not None:
        _draw_arrow(draw, center, approach_tip, fill=yellow, width=3)
    _draw_cross(draw, center, radius=7, fill=green)

    label = f"score={float(grasp.get('score', 0.0) or 0.0):.3f} row={grasp.get('row_index', 'top1')}"
    text_xy = (center[0] + 12, center[1] - 16)
    draw.text((text_xy[0] + 1, text_xy[1] + 1), label, fill=black)
    draw.text(text_xy, label, fill=green)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    direction_angle_deg = None
    if approach_tip is not None:
        direction_angle_deg = float(np.degrees(np.arctan2(approach_tip[1] - center[1], approach_tip[0] - center[0])))
    return {
        "path": str(output_path),
        "center_px": [int(center[0]), int(center[1])],
        "approach_tip_px": [int(approach_tip[0]), int(approach_tip[1])] if approach_tip else None,
        "width_endpoints_px": [list(width_a), list(width_b)] if width_a and width_b else None,
        "approach_angle_deg_image": direction_angle_deg,
    }


def _service_url(name: str) -> str:
    return str(load_yaml("services.yaml")[name]).rstrip("/")


def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_json(url: str, timeout: float = 10.0) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _run_dir(request: RunGraspRequest) -> Path:
    if request.output_dir:
        base = ensure_dir(request.output_dir)
    else:
        base = new_run_dir("zerograsp_task")
    if request.run_tag:
        tagged = base.parent / f"{base.name}_{_safe_name(request.run_tag)}"
        return ensure_dir(tagged)
    return base


def _camera_id_for_arm(arm_id: str) -> str:
    return "left_hand" if arm_id == "left" else "right_hand"


def _capture_rgbd(services: dict[str, Any], output_dir: Path, camera_id: str) -> CaptureResult:
    return CaptureResult(**_post_json(
        f"{services['arx_client_url'].rstrip('/')}/camera/capture-rgbd",
        {"output_dir": str(output_dir), "camera_id": camera_id},
        timeout=120.0,
    ))


def _intermediate_waypoint_pose(
    *,
    current_pose: list[float],
    target_pose: list[float],
    height_offset_m: float,
    target_ratio: float,
) -> list[float]:
    if len(current_pose) != 6 or len(target_pose) != 6:
        raise ValueError(f"current_pose and target_pose must be 6D, got {current_pose}, {target_pose}")
    ratio = min(1.0, max(0.0, float(target_ratio)))
    waypoint = list(target_pose)
    waypoint[0] = float(current_pose[0]) + (float(target_pose[0]) - float(current_pose[0])) * ratio
    waypoint[1] = float(current_pose[1]) + (float(target_pose[1]) - float(current_pose[1])) * ratio
    waypoint[2] = float(target_pose[2]) + float(height_offset_m)
    return waypoint


def _capture_depth_burst(
    *,
    services: dict[str, Any],
    camera_dir: Path,
    task_cfg: dict[str, Any],
    camera_id: str,
) -> tuple[CaptureResult, list[str], dict[str, Any]]:
    enabled = bool(task_cfg.get("depth_fusion_enabled", True))
    frame_count = int(task_cfg.get("depth_burst_frames", 3) or 1)
    frame_count = max(1, frame_count)
    if not enabled or frame_count <= 1:
        capture = _capture_rgbd(services, camera_dir, camera_id)
        return capture, [capture.depth_path], {
            "enabled": False,
            "frame_count": 1,
            "captures": [{"frame_index": 0, "depth_path": capture.depth_path, "color_path": capture.color_path}],
        }

    window_sec = max(0.0, float(task_cfg.get("depth_burst_window_sec", 1.0) or 0.0))
    interval_sec = window_sec / max(1, frame_count - 1) if frame_count > 1 else 0.0
    captures: list[CaptureResult] = []
    start = time.monotonic()
    for index in range(frame_count):
        frame_dir = camera_dir if index == 0 else ensure_dir(camera_dir / f"frame_{index}")
        captures.append(_capture_rgbd(services, frame_dir, camera_id))
        if index < frame_count - 1 and interval_sec > 0.0:
            next_time = start + interval_sec * (index + 1)
            sleep_sec = next_time - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

    report = {
        "enabled": True,
        "frame_count": len(captures),
        "target_window_sec": window_sec,
        "actual_duration_sec": time.monotonic() - start,
        "captures": [
            {"frame_index": index, "depth_path": capture.depth_path, "color_path": capture.color_path}
            for index, capture in enumerate(captures)
        ],
    }
    return captures[0], [capture.depth_path for capture in captures], report


def run_grasp_task(request: RunGraspRequest) -> RunGraspResult:
    services = load_yaml("services.yaml")
    task_cfg = load_yaml("zerograsp.yaml").get("task", {})
    arm_id = request.arm_id or "right"
    camera_id = _camera_id_for_arm(arm_id)
    run_dir = _run_dir(request)
    camera_dir = ensure_dir(run_dir / "camera")
    label = _safe_name(request.mask_target or request.prompt.rstrip("."))
    mask_name = f"{label}_mask.png"
    overlay_name = f"{label}_overlay.png"
    info_name = f"{label}_mask.json"

    if request.reuse_camera_dir and request.reuse_mask_path:
        reuse_camera_dir = Path(request.reuse_camera_dir)
        capture = CaptureResult(
            color_path=str(reuse_camera_dir / "color.png"),
            depth_path=str(reuse_camera_dir / "depth.png"),
            params_path=str(reuse_camera_dir / "params.json"),
        )
        depth_image_paths = [capture.depth_path]
        capture_burst = {
            "enabled": False,
            "source": "reused_observe_capture",
            "captures": [{"frame_index": 0, "depth_path": capture.depth_path, "color_path": capture.color_path}],
        }
    else:
        capture, depth_image_paths, capture_burst = _capture_depth_burst(
            services=services,
            camera_dir=camera_dir,
            task_cfg=task_cfg,
            camera_id=camera_id,
        )

    status_data = None
    current_pose_source = "arm_status"
    try:
        status_data = _get_json(f"{services['arx_client_url'].rstrip('/')}/arm/status?arm_id={arm_id}", timeout=10.0)
        current_end_pos = status_data.get("end_pos")
        if not isinstance(current_end_pos, list) or len(current_end_pos) != 6:
            raise ValueError(f"invalid end_pos: {current_end_pos}")
    except Exception:
        if not request.allow_debug_zero_pose:
            raise
        current_pose_source = "debug_zero_pose"
        current_end_pos = [0.0] * 6

    if request.reuse_camera_dir and request.reuse_mask_path:
        reuse_segmentation = dict(request.reuse_segmentation or {})
        segmentation = SegmentationResult(
            mask_path=request.reuse_mask_path,
            overlay_path=reuse_segmentation.get("overlay_path"),
            info_path=reuse_segmentation.get("info_path"),
            bbox=reuse_segmentation.get("bbox"),
            centroid=reuse_segmentation.get("centroid"),
            score=reuse_segmentation.get("score"),
            label=request.prompt,
            found=True,
            area=reuse_segmentation.get("area"),
            raw={"mode": "reused_observe_segmentation", "source": reuse_segmentation},
        )
        segmentation_camera_dir = Path(request.reuse_camera_dir)
    else:
        segmentation = SegmentationResult(**_post_json(
            f"{services['langsam_url'].rstrip('/')}/segment",
            {
                "image_path": capture.color_path,
                "prompt": request.prompt,
                "points": [
                    point.model_dump() if hasattr(point, "model_dump") else point.dict()
                    for point in request.points
                ],
                "output_dir": str(camera_dir),
                "method": request.mask_method,
                "mask_name": mask_name,
                "overlay_name": overlay_name,
                "info_name": info_name,
                "min_area": int(task_cfg.get("mask_min_area", 50)),
                "keep_largest": True,
            },
            timeout=float(task_cfg.get("langsam_timeout_sec", 240.0)),
        ))
        segmentation_camera_dir = camera_dir
    if not segmentation.found or not segmentation.mask_path:
        raise RuntimeError(f"segmentation failed: {segmentation.raw}")

    zg_input = prepare_zerograsp_input(ZeroGraspInputRequest(
        camera_dir=str(segmentation_camera_dir),
        mask_path=segmentation.mask_path,
        label=label,
        output_dir=str(run_dir / "zg_input"),
        depth_image_paths=depth_image_paths,
        depth_fusion_enabled=bool(task_cfg.get("depth_fusion_enabled", True)),
        depth_fusion_drop_worst=bool(task_cfg.get("depth_fusion_drop_worst", True)),
    ))
    depth_fusion = dict(zg_input.depth_fusion or {})
    depth_fusion["capture_burst"] = capture_burst
    prediction = predict_grasp(GraspPredictRequest(
        input_dir=zg_input.input_dir,
        output_dir=str(run_dir / "zg_output"),
        timeout_sec=request.timeout_sec,
    ))
    if not prediction.grasp_path:
        raise RuntimeError(f"ZeroGrasp did not produce recommended_grasp_top1.json: {prediction.raw}")

    max_target_x_m = (
        request.max_target_x_m
        if request.max_target_x_m is not None
        else task_cfg.get("max_target_x_m", 0.38)
    )
    top1 = _apply_target_x_filter(
        prediction=prediction,
        current_end_pos=current_end_pos,
        request=request,
        max_target_x_m=max_target_x_m,
    )
    plan = plan_grasp(GraspPlanRequest(
        arm_id=arm_id,
        prediction=top1,
        prediction_path=prediction.grasp_path,
        current_end_pos=current_end_pos,
        allow_debug_zero_pose=request.allow_debug_zero_pose,
        execute=request.execute,
    ))
    if not plan.safety_passed:
        raise RuntimeError("safety check failed: " + "; ".join(plan.safety_failures))

    grasp_projection_path = None
    if not request.skip_projection:
        try:
            projection = _draw_grasp_projection(
                rgb_path=capture.color_path,
                camera_info_path=zg_input.camera_info_path,
                grasp=top1,
                output_path=Path(prediction.output_dir or run_dir / "zg_output") / "grasp_projection.png",
            )
            grasp_projection_path = projection["path"]
            prediction.raw["grasp_projection"] = projection
        except Exception as exc:
            prediction.raw["grasp_projection_error"] = str(exc)

    intermediate_waypoint: dict[str, Any] = {}
    intermediate_execution = None
    execution = None
    mode = "plan_only"
    if request.execute:
        mode = "execute"
        final_speed = float(task_cfg.get("final_speed", 1.3))
        close_delay = float(task_cfg.get("close_delay", 0.5))
        motion_timeout = float(task_cfg.get("motion_timeout_sec", 120.0))
        execute_duration = request.execute_duration_sec
        if execute_duration is not None:
            execute_duration = max(0.1, float(execute_duration))
            motion_timeout = min(motion_timeout, execute_duration + 10.0)

        waypoint_enabled = bool(task_cfg.get("intermediate_waypoint_enabled", True))
        if waypoint_enabled:
            waypoint_pose = _intermediate_waypoint_pose(
                current_pose=current_end_pos,
                target_pose=plan.target_pose or [],
                height_offset_m=float(task_cfg.get("intermediate_waypoint_height_offset_m", 0.10)),
                target_ratio=float(task_cfg.get("intermediate_waypoint_target_ratio", 2.0 / 3.0)),
            )
            waypoint_timeout = float(task_cfg.get("intermediate_waypoint_timeout_sec", 10.0))
            intermediate_waypoint = {
                "enabled": True,
                "pose": waypoint_pose,
                "height_offset_m": float(task_cfg.get("intermediate_waypoint_height_offset_m", 0.10)),
                "target_ratio": float(task_cfg.get("intermediate_waypoint_target_ratio", 2.0 / 3.0)),
                "timeout_sec": waypoint_timeout,
                "target_gripper": plan.target_gripper,
                "hold_gripper": False,
                "continue_to_target_on_failure": True,
                "source": "xy=current_to_target_ratio,z=target_z_plus_offset,rpy=target_rpy",
            }
            waypoint_request = EndPoseRequest(
                arm_id=arm_id,
                target_pose=waypoint_pose,
                target_gripper=plan.target_gripper,
                hold_gripper=False,
                speed=final_speed,
                close_delay=close_delay,
                exit_on_pose=True,
                timeout_sec=waypoint_timeout,
            )
            waypoint_payload = _dump_model(waypoint_request)
            try:
                intermediate_execution = ExecutionResult(**_post_json(
                    f"{services['arx_client_url'].rstrip('/')}/arm/move-end-pose",
                    waypoint_payload,
                    timeout=waypoint_timeout + 10.0,
                ))
            except Exception as exc:
                intermediate_execution = ExecutionResult(
                    ok=False,
                    message=f"intermediate waypoint request failed; continuing to final target pose: {exc}",
                    data={"request": waypoint_payload},
                )
        else:
            intermediate_waypoint = {"enabled": False}

        execution = ExecutionResult(**_post_json(
            f"{services['arx_client_url'].rstrip('/')}/arm/move-end-pose",
            _dump_model(EndPoseRequest(
                arm_id=arm_id,
                target_pose=plan.target_pose or [],
                target_gripper=plan.target_gripper,
                speed=final_speed,
                close_delay=close_delay,
                exit_on_pose=False,
                exit_after_sec=execute_duration,
                timeout_sec=motion_timeout,
            )),
            timeout=motion_timeout + 10.0,
        ))

    result = RunGraspResult(
        ok=True,
        mode=mode,
        run_dir=str(run_dir),
        current_pose_source=current_pose_source,
        capture=capture,
        segmentation=segmentation,
        zg_input=zg_input,
        prediction=prediction,
        plan=plan,
        intermediate_waypoint=intermediate_waypoint,
        intermediate_execution=intermediate_execution,
        execution=execution,
        grasp_projection_path=grasp_projection_path,
        depth_fusion=depth_fusion,
        report_path=str(run_dir / "run_report.json"),
    )
    report = {
        "created_unix": time.time(),
        "request": _dump_model(request),
        "services": services,
        "arm_id": arm_id,
        "camera_id": camera_id,
        "arm_status": status_data,
        "result": _dump_model(result),
    }
    Path(result.report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
