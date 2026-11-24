"""Classify measured tabletop contacts from raw points and the authored TCP."""
import numpy as np

from .geometry import FINGERS
from modules.arx_core.arms import get_arm


def classify_contact(profile, *, finger, is_table, stage, points, eef, table, tip_plane, arm='right'):
    result = {'allowed': False, 'contact_depth_m': 0., 'reason': 'nonlocal_table_contact'}
    if not profile.enabled or finger not in get_arm(arm).finger_links or not is_table or stage not in ['approach', 'close', 'lift']:
        return result
    eef = np.asarray(eef, dtype=float)
    center, dims = np.asarray(table['center']), np.asarray(table['dims'])
    if eef.shape != (4, 4) or center.shape != (3,) or dims.shape != (3,) or not (dims > 0).all():
        return {**result, 'reason': 'invalid_contact_geometry'}
    if not np.isfinite(eef).all() or not np.isfinite([*center, *dims, tip_plane]).all():
        return {**result, 'reason': 'invalid_contact_geometry'}
    if np.max(np.abs(eef[:3, :3].T @ eef[:3, :3] - np.eye(3))) > 1e-5 or abs(np.linalg.det(eef[:3, :3])-1) > 1e-5:
        return {**result, 'reason': 'invalid_contact_frame'}
    if not np.allclose(eef[3], [0., 0., 0., 1.], atol=1e-8, rtol=0):
        return {**result, 'reason': 'invalid_contact_frame'}
    for point in points:
        if not np.isfinite([*point['world'], *point['normal'], point['separation_m'], point['impulse_ns']]).all():
            return {**result, 'reason': 'nonfinite_contact_point'}
    active = [p for p in points if p['impulse_ns'] > 1e-8]
    if not active:
        return {**result, 'reason': 'missing_contact_points'}
    top = float(center[2] + dims[2]/2)
    depth = 0.
    for point in active:
        xyz, normal = np.asarray(point['world']), np.asarray(point['normal'])
        separation = float(point['separation_m'])
        if xyz.shape != (3,) or normal.shape != (3,) or not np.isfinite([*xyz, *normal, separation, point['impulse_ns']]).all():
            return {**result, 'reason': 'nonfinite_contact_point'}
        local = (xyz - eef[:3, 3]) @ eef[:3, :3]
        if local[2] < tip_plane - profile.distal_length_m - 1e-6 or local[2] > tip_plane + profile.maximum_penetration_m:
            return {**result, 'reason': 'finger_body_contact'}
        if abs(np.linalg.norm(normal)-1.) > 1e-4:
            return {**result, 'reason': 'invalid_contact_normal'}
        if abs(normal[2]) < profile.minimum_normal_z or np.any(np.abs(xyz[:2]-center[:2]) > dims[:2]/2):
            return {**result, 'reason': 'table_side_contact'}
        if abs(xyz[2]-top) > profile.maximum_penetration_m + .001 + profile.surface_position_noise_m:
            return {**result, 'reason': 'contact_not_on_table_top'}
        depth = max(depth, max(0., -separation))
    result['contact_depth_m'] = depth
    if depth > profile.maximum_penetration_m:
        return {**result, 'reason': 'contact_depth_exceeds_tolerance'}
    return {**result, 'allowed': True, 'reason': 'bounded_distal_table_contact'}


class ReleaseEnvelope:
    """Track a nonincreasing contact ceiling independently for each path/frame stream."""
    def __init__(self, profile):
        self.profile = profile
        self.ceiling = None

    def anchor(self, depth):
        depth = float(depth)
        if not np.isfinite(depth) or not 0 <= depth <= self.profile.maximum_penetration_m:
            raise ValueError('Invalid measured release anchor')
        self.ceiling = depth

    def accept(self, depth):
        depth = float(depth)
        if self.ceiling is None or not np.isfinite(depth) or depth < 0:
            return False
        valid = depth <= self.profile.maximum_penetration_m and depth <= self.ceiling + self.profile.release_depth_noise_m
        if valid:
            self.ceiling = min(self.ceiling, depth)
        return valid
