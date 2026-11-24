"""Rigid action-frame geometry shared by robot diagnostics."""
import numpy as np
from scipy.spatial.transform import Rotation

EEF_TCP = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                    [0., 0., 1., .043], [0., 0., 0., 1.]])


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Expected a finite rigid transform')
    return t


def action_frame(eef_tcp=None):
    """Resolve a software action origin without changing the authored EEF frame."""
    frame = rigid(EEF_TCP if eef_tcp is None else eef_tcp).copy()
    if (not np.allclose(frame[:3, :3], EEF_TCP[:3, :3], atol=1e-12, rtol=0)
            or not np.allclose(frame[:2, 3], 0., atol=1e-12, rtol=0)
            or frame[2, 3] <= 0):
        raise ValueError('Action origin may move only along EEF +Z; action axes remain fixed')
    return frame


def target_pose(ee_world, action, eef_tcp=None):
    """Compose a local TCP increment, then recover EEF without moving the pivot."""
    frame = action_frame(eef_tcp)
    current = rigid(ee_world) @ frame
    target = current.copy()
    if action.translation.distance_cm:
        unit = action.translation.direction / np.linalg.norm(action.translation.direction)
        target[:3, 3] += current[:3, :3] @ unit * action.translation.distance_cm / 100.
    if action.rotation.angle_deg:
        axis = action.rotation.axis / np.linalg.norm(action.rotation.axis)
        target[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(
            axis * np.deg2rad(action.rotation.angle_deg)).as_matrix()
    return target @ np.linalg.inv(frame)
