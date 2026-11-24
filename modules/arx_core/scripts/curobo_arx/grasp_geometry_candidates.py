"""Candidate-level pose construction under the frozen ARX pad calibration."""
import numpy as np
from scipy.spatial.transform import Rotation

from sphere_refinement import PAD_CENTER_EEF


def construct(grasp, contact, sphere_fit, height_shift_m=0.):
    if height_shift_m not in (0., .003, .006, .009, .012):
        raise ValueError("Pose-height search outside declared algorithm grid")
    original = np.asarray(grasp)
    theta = float(np.arctan2(original[2, 1], original[2, 0]))
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    rotation = original[:3, :3] @ Rotation.from_rotvec([0., 0., theta]).as_matrix()
    if sphere_fit['accepted']:
        pad_target = np.asarray(sphere_fit['center_world']) + [0., 0., .010]
        mode = 'Observed sphere centre plus10mm'
    else:
        pad_target = np.asarray(contact) + [0., 0., height_shift_m]
        mode = 'Original ZeroGrasp/height-prior contact, interpreted as pad centre'
    refined = np.eye(4)
    refined[:3, :3] = rotation
    refined[:3, 3] = pad_target - rotation @ PAD_CENTER_EEF
    pre = refined.copy(); pre[:3, 3] -= .08 * rotation[:, 2]
    lift = refined.copy(); lift[2, 3] += .12
    return refined, pre, lift, {'hypothesis': 'H009', 'mode': mode,
        'rotation_about_original_approach_rad': theta,
        'closing_axis_z_before': float(original[2, 1]),
        'closing_axis_z_after': float(rotation[2, 1]),
        'pad_center_eef_m_frozen': PAD_CENTER_EEF.tolist(),
        'pad_target_world': pad_target.tolist(),
        'world_z_pose_search_offset_m': height_shift_m,
        'robot_configuration_changed': False}
