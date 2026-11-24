from __future__ import annotations

import math
from typing import Iterable


Matrix4 = list[list[float]]


def identity4() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def validate_matrix4(matrix: Iterable[Iterable[float]]) -> Matrix4:
    rows = [[float(v) for v in row] for row in matrix]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("expected a 4x4 matrix")
    return rows


def matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def inverse_rigid4(T: Matrix4) -> Matrix4:
    T = validate_matrix4(T)
    R = [[T[i][j] for j in range(3)] for i in range(3)]
    t = [T[i][3] for i in range(3)]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    out = identity4()
    for i in range(3):
        for j in range(3):
            out[i][j] = Rt[i][j]
        out[i][3] = -sum(Rt[i][k] * t[k] for k in range(3))
    return out


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> Matrix4:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pose6_to_matrix(pose: Iterable[float]) -> Matrix4:
    values = [float(v) for v in pose]
    if len(values) != 6:
        raise ValueError("pose must be [x, y, z, roll, pitch, yaw]")
    x, y, z, roll, pitch, yaw = values
    matrix = rpy_to_matrix(roll, pitch, yaw)
    matrix[0][3] = x
    matrix[1][3] = y
    matrix[2][3] = z
    return matrix


def matrix_to_pose6(T: Matrix4) -> list[float]:
    T = validate_matrix4(T)
    x, y, z = T[0][3], T[1][3], T[2][3]
    pitch = math.asin(max(-1.0, min(1.0, -T[2][0])))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(T[2][1], T[2][2])
        yaw = math.atan2(T[1][0], T[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-T[0][1], T[1][1])
    return [x, y, z, roll, pitch, yaw]

