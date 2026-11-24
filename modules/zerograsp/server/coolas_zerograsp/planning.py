from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from arx_shared.config import load_yaml
from arx_shared.geometry import identity4, inverse_rigid4, matmul4, matrix_to_pose6, pose6_to_matrix, validate_matrix4
from arx_shared.schemas import GraspPlan, GraspPlanRequest


def load_calibration() -> dict:
    return load_yaml("calibration.yaml")


def _translation_matrix(x: float, y: float, z: float) -> list[list[float]]:
    T = identity4()
    T[0][3] = float(x)
    T[1][3] = float(y)
    T[2][3] = float(z)
    return T


def _prediction_from_request(request: GraspPlanRequest) -> dict[str, Any]:
    if request.prediction is not None:
        return request.prediction
    if not request.prediction_path:
        raise ValueError("prediction or prediction_path is required")
    path = Path(request.prediction_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_from_grasp(prediction: dict[str, Any]) -> list[list[float]]:
    rotation = prediction.get("rotation_matrix_camera")
    translation = prediction.get("translation_m_camera")
    if rotation is None or translation is None:
        raise ValueError("prediction must contain rotation_matrix_camera and translation_m_camera")
    rows = validate_matrix4([
        [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return rows


def _ee_camera_matrix(calibration: dict, arm_id: str) -> list[list[float]]:
    key = f"{arm_id}_ee_camera"
    if key not in calibration:
        raise ValueError(f"calibration.{key} is required for arm_id={arm_id}")
    return validate_matrix4(calibration[key]["matrix"])


def _tcp_gripper_matrix(calibration: dict, arm_id: str = "right") -> list[list[float]]:
    tcp = calibration.get(f"{arm_id}_tcp_gripper") or calibration.get("tcp_gripper", {})
    xyz = tcp.get("xyz_m", [0.16, 0.0, 0.0])
    return _translation_matrix(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _target_gripper(prediction: dict[str, Any]) -> tuple[float, bool, float]:
    cfg = load_yaml("zerograsp.yaml").get("task", {})
    fixed_open = bool(cfg.get("fixed_open_gripper", True))
    fixed_value = float(cfg.get("fixed_open_gripper_value", -3.2))
    zg_value = -float(prediction.get("width_m", 0.04) or 0.04) * 60.0
    return (fixed_value if fixed_open else zg_value), fixed_open, zg_value


def _safety_check(current: list[float], target: list[float]) -> tuple[bool, list[str], dict[str, Any]]:
    motion = load_yaml("arx.yaml").get("motion", {})
    task = load_yaml("zerograsp.yaml").get("task", {})
    axis_limits = task.get("safety_axis_limits_m", {"x": 0.56, "y": 0.30, "z": 0.30})
    translation_limit = float(task.get("safety_translation_limit_m", 0.65))

    dx = float(target[0]) - float(current[0])
    dy = float(target[1]) - float(current[1])
    dz = float(target[2]) - float(current[2])
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    failures = []
    for key, delta in [("x", abs(dx)), ("y", abs(dy)), ("z", abs(dz))]:
        limit = float(axis_limits.get(key, 999.0))
        if delta > limit:
            failures.append(f"{key} displacement {delta:.4f}m exceeds limit {limit:.4f}m")
    if dist > translation_limit:
        failures.append(f"translation distance {dist:.4f}m exceeds limit {translation_limit:.4f}m")

    for key, value in [("x", target[0]), ("y", target[1]), ("z", target[2])]:
        low, high = [float(v) for v in motion.get(f"{key}_range", [-999.0, 999.0])]
        if not (low <= float(value) <= high):
            failures.append(f"target {key}={float(value):.4f} outside workspace [{low:.4f}, {high:.4f}]")

    return not failures, failures, {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "translation_distance_m": dist,
        "translation_limit_m": translation_limit,
        "axis_limits_m": axis_limits,
    }


def plan_grasp(request: GraspPlanRequest) -> GraspPlan:
    prediction = _prediction_from_request(request)
    current = request.current_end_pos
    if current is None:
        if not request.allow_debug_zero_pose:
            raise ValueError("current_end_pos is required; use arx-ops /arm/status before planning")
        current = [0.0] * 6
    if len(current) != 6:
        raise ValueError("current_end_pos must contain 6 values")

    arm_id = request.arm_id or "right"
    calibration = load_calibration()
    T_ee_camera = _ee_camera_matrix(calibration, arm_id)
    T_ee_gripper = _tcp_gripper_matrix(calibration, arm_id)
    T_camera_grasp = _matrix_from_grasp(prediction)
    T_arm_ee_current = pose6_to_matrix(current)

    T_ee_grasp_target = matmul4(T_ee_camera, T_camera_grasp)
    T_E0_Etarget = matmul4(T_ee_grasp_target, inverse_rigid4(T_ee_gripper))
    T_arm_ee_goal_pre_depth = matmul4(T_arm_ee_current, T_E0_Etarget)

    grasp_depth_m = max(0.0, float(prediction.get("depth_m", 0.0) or 0.0))
    T_depth_advance = _translation_matrix(grasp_depth_m, 0.0, 0.0)
    T_arm_ee_goal = matmul4(T_arm_ee_goal_pre_depth, T_depth_advance)

    approach_pose = matrix_to_pose6(T_arm_ee_goal_pre_depth)
    target_pose = matrix_to_pose6(T_arm_ee_goal)
    target_gripper, fixed_open, zg_width_gripper = _target_gripper(prediction)
    safety_passed, safety_failures, safety = _safety_check(current, target_pose)

    return GraspPlan(
        approach_pose=approach_pose,
        target_pose=target_pose,
        target_gripper=target_gripper,
        advance_depth_m=grasp_depth_m,
        safety_passed=safety_passed,
        safety_failures=safety_failures,
        metadata={
            "phase": "phase2",
            "arm_id": arm_id,
            "calibration_keys": {
                "ee_camera": f"{arm_id}_ee_camera",
                "tcp_gripper": f"{arm_id}_tcp_gripper",
            },
            "current_end_pos": current,
            "zero_grasp": {
                "score": prediction.get("score"),
                "width_m": prediction.get("width_m"),
                "depth_m": prediction.get("depth_m"),
                "translation_m_camera": prediction.get("translation_m_camera"),
                "rotation_matrix_camera": prediction.get("rotation_matrix_camera"),
            },
            "fixed_open_gripper": fixed_open,
            "zg_width_gripper": zg_width_gripper,
            "transforms": {
                "T_ee_camera": T_ee_camera,
                "T_E_G": T_ee_gripper,
                "T_camera_grasp": T_camera_grasp,
                "T_ee_grasp_target": T_ee_grasp_target,
                "T_E0_Etarget": T_E0_Etarget,
                "T_arm_ee_current": T_arm_ee_current,
                "T_depth_advance": T_depth_advance,
                "T_arm_ee_goal_pre_depth": T_arm_ee_goal_pre_depth,
                "T_arm_ee_goal": T_arm_ee_goal,
            },
            "safety": safety,
        },
    )

