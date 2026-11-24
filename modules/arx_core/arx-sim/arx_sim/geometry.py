from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def quaternion_wxyz_to_matrix(value: Iterable[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in value)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25*s, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0,0] - r[1,1] - r[2,2]) * 2
            q = np.array([(r[2,1]-r[1,2])/s, 0.25*s, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s])
        elif i == 1:
            s = math.sqrt(1.0 + r[1,1] - r[0,0] - r[2,2]) * 2
            q = np.array([(r[0,2]-r[2,0])/s, (r[0,1]+r[1,0])/s, 0.25*s, (r[1,2]+r[2,1])/s])
        else:
            s = math.sqrt(1.0 + r[2,2] - r[0,0] - r[1,1]) * 2
            q = np.array([(r[1,0]-r[0,1])/s, (r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, 0.25*s])
    return q / np.linalg.norm(q)


def pose_matrix(position: Iterable[float], orientation_wxyz: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(orientation_wxyz)
    matrix[:3, 3] = np.asarray(list(position), dtype=float)
    return matrix


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(rotation)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-6:
        return math.atan2(r[2,1], r[2,2]), pitch, math.atan2(r[1,0], r[0,0])
    return 0.0, pitch, math.atan2(-r[0,1], r[1,1])


def transform_pose_to_local(parent_world: np.ndarray, child_world: np.ndarray) -> list[float]:
    local = np.linalg.inv(parent_world) @ child_world
    return [*local[:3, 3].tolist(), *matrix_to_rpy(local[:3, :3])]


def pose6_local_to_world(parent_world: np.ndarray, pose: Iterable[float]) -> np.ndarray:
    values = [float(v) for v in pose]
    if len(values) != 6:
        raise ValueError("pose must have six values")
    local = np.eye(4)
    local[:3, :3] = rpy_to_matrix(*values[3:])
    local[:3, 3] = values[:3]
    return parent_world @ local


def rotation_vector_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([delta[2,1]-delta[1,2], delta[0,2]-delta[2,0], delta[1,0]-delta[0,1]])
    axis /= max(2.0 * math.sin(angle), 1e-8)
    return axis * angle


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=float)
    if np.any(points[:, 2] <= 0):
        raise ValueError("camera-space z must be positive")
    pixels = (intrinsics @ points.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def unproject_pixels(pixels: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    uv = np.asarray(pixels, dtype=float)
    z = np.asarray(depth_m, dtype=float).reshape(-1, 1)
    homogeneous = np.column_stack([uv, np.ones(len(uv))])
    return (np.linalg.inv(intrinsics) @ homogeneous.T).T * z
