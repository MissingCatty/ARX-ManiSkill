from __future__ import annotations

import json
from pathlib import Path

from arx_shared.config import load_yaml
from arx_shared.schemas import GraspPredictRequest, ZeroGraspInputRequest, ZeroGraspInputResult

from .model import predict_grasp
from .storage import new_zerograsp_run_dir


def _depth_to_mm(raw_depth, depth_scale: float):
    import numpy as np

    return (raw_depth.astype(np.float64) * depth_scale * 1000.0).astype(np.uint16)


def _mask_hole_ratio(depth_mm, object_pixels) -> tuple[float, int, int]:
    import numpy as np

    mask_count = int(np.count_nonzero(object_pixels))
    if mask_count <= 0:
        return 0.0, 0, 0
    holes = object_pixels & (depth_mm == 0)
    hole_count = int(np.count_nonzero(holes))
    return float(hole_count / mask_count), hole_count, mask_count


def _fuse_depths_in_mask(depths_mm, object_pixels, *, drop_worst: bool) -> tuple[object, dict]:
    import numpy as np

    hole_stats = []
    for index, depth_mm in enumerate(depths_mm):
        ratio, holes, pixels = _mask_hole_ratio(depth_mm, object_pixels)
        hole_stats.append({
            "frame_index": index,
            "hole_ratio": ratio,
            "hole_pixels": holes,
            "mask_pixels": pixels,
        })

    selected_indices = list(range(len(depths_mm)))
    dropped_index = None
    if drop_worst and len(depths_mm) > 1:
        worst = max(hole_stats, key=lambda item: (item["hole_ratio"], item["hole_pixels"], item["frame_index"]))
        dropped_index = int(worst["frame_index"])
        selected_indices = [index for index in selected_indices if index != dropped_index]

    base_index = 0 if 0 in selected_indices else selected_indices[0]
    fused = depths_mm[base_index].copy()
    selected_stack = np.stack([depths_mm[index] for index in selected_indices], axis=0)
    selected_mask_values = selected_stack[:, object_pixels]
    valid = selected_mask_values > 0
    fused_values = np.zeros(selected_mask_values.shape[1], dtype=np.uint16)
    for column in range(selected_mask_values.shape[1]):
        values = selected_mask_values[:, column]
        valid_values = values[valid[:, column]]
        if valid_values.size:
            fused_values[column] = np.uint16(int(np.median(valid_values)))

    before_holes = object_pixels & (fused == 0)
    fused[object_pixels] = fused_values
    after_peer_holes = object_pixels & (fused == 0)
    filled_from_peer = int(np.count_nonzero(before_holes & ~after_peer_holes))

    filled_by_median = 0
    valid_object_depth = object_pixels & (fused > 0)
    if np.any(valid_object_depth) and np.any(after_peer_holes):
        fill_depth = int(np.median(fused[valid_object_depth]))
        fused[after_peer_holes] = np.uint16(fill_depth)
        filled_by_median = int(np.count_nonzero(after_peer_holes))

    final_ratio, final_holes, mask_pixels = _mask_hole_ratio(fused, object_pixels)
    report = {
        "enabled": True,
        "frame_count": len(depths_mm),
        "drop_worst": bool(drop_worst),
        "dropped_frame_index": dropped_index,
        "selected_frame_indices": selected_indices,
        "base_frame_index": base_index,
        "hole_stats": hole_stats,
        "holes_filled_from_peer_frames": filled_from_peer,
        "holes_filled_by_median": filled_by_median,
        "final_hole_ratio": final_ratio,
        "final_hole_pixels": final_holes,
        "mask_pixels": mask_pixels,
    }
    return fused, report


def prepare_zerograsp_input(request: ZeroGraspInputRequest) -> ZeroGraspInputResult:
    import cv2
    import numpy as np

    camera_dir = Path(request.camera_dir)
    mask_path = Path(request.mask_path)
    output_dir = Path(request.output_dir) if request.output_dir else new_zerograsp_run_dir() / "zg_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads((camera_dir / "params.json").read_text(encoding="utf-8"))
    intrinsics = params["intrinsics"]
    depth_scale = float(params["depth_scale"])

    color_image = cv2.imread(str(camera_dir / "color.png"), cv2.IMREAD_COLOR)
    if color_image is None:
        raise FileNotFoundError(f"cannot read color image: {camera_dir / 'color.png'}")
    raw_depth = cv2.imread(str(camera_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise FileNotFoundError(f"cannot read depth image: {camera_dir / 'depth.png'}")
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise FileNotFoundError(f"cannot read mask image: {mask_path}")

    instance_mask = np.where(raw_mask > 0, np.uint8(1), np.uint8(0))
    object_pixels = instance_mask > 0
    depth_paths = [camera_dir / "depth.png"]
    if request.depth_fusion_enabled:
        for depth_path in request.depth_image_paths:
            path = Path(depth_path)
            if path not in depth_paths:
                depth_paths.append(path)

    depths_mm = []
    for depth_path in depth_paths:
        frame_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if frame_depth is None:
            raise FileNotFoundError(f"cannot read depth image: {depth_path}")
        if frame_depth.shape != raw_depth.shape:
            raise ValueError(f"depth shape mismatch: {depth_path} has {frame_depth.shape}, expected {raw_depth.shape}")
        depths_mm.append(_depth_to_mm(frame_depth, depth_scale))

    if request.depth_fusion_enabled and len(depths_mm) > 1:
        depth_mm, depth_fusion = _fuse_depths_in_mask(
            depths_mm,
            object_pixels,
            drop_worst=request.depth_fusion_drop_worst,
        )
        filled_depth_holes = int(
            depth_fusion.get("holes_filled_from_peer_frames", 0)
            + depth_fusion.get("holes_filled_by_median", 0)
        )
    else:
        depth_mm = depths_mm[0]
        valid_object_depth = object_pixels & (depth_mm > 0)
        object_depth_holes = object_pixels & (depth_mm == 0)
        ratio, holes, mask_pixels = _mask_hole_ratio(depth_mm, object_pixels)
        filled_depth_holes = 0
        if np.any(valid_object_depth) and np.any(object_depth_holes):
            fill_depth = int(np.median(depth_mm[valid_object_depth]))
            depth_mm[object_depth_holes] = np.uint16(fill_depth)
            filled_depth_holes = int(np.count_nonzero(object_depth_holes))
        final_ratio, final_holes, _ = _mask_hole_ratio(depth_mm, object_pixels)
        depth_fusion = {
            "enabled": False,
            "frame_count": 1,
            "hole_stats": [{
                "frame_index": 0,
                "hole_ratio": ratio,
                "hole_pixels": holes,
                "mask_pixels": mask_pixels,
            }],
            "holes_filled_by_median": filled_depth_holes,
            "final_hole_ratio": final_ratio,
            "final_hole_pixels": final_holes,
            "mask_pixels": mask_pixels,
        }

    rgb_path = output_dir / "rgb.png"
    depth_path = output_dir / "depth.png"
    zg_mask_path = output_dir / "mask.png"
    camera_info_path = output_dir / "camera.json"
    cv2.imwrite(str(rgb_path), color_image)
    cv2.imwrite(str(depth_path), depth_mm)
    cv2.imwrite(str(zg_mask_path), instance_mask)

    arx_cfg = load_yaml("arx.yaml")
    camera_json = {
        "cam_K": [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        "depth_scale": 1.0,
        "camera_name": arx_cfg.get("camera", {}).get("name", "right_arm"),
        "mask_mode": request.label,
        "objects": [{"label": 1, "display_name": request.label}],
    }
    camera_info_path.write_text(json.dumps(camera_json, ensure_ascii=False, indent=2), encoding="utf-8")

    return ZeroGraspInputResult(
        input_dir=str(output_dir),
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        mask_path=str(zg_mask_path),
        camera_info_path=str(camera_info_path),
        filled_depth_holes=filled_depth_holes,
        depth_fusion=depth_fusion,
    )


def run_predict_only(request: GraspPredictRequest):
    return predict_grasp(request)

