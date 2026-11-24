"""Native visible surfaces plus a bounded gravity-column closure hypothesis.

No coarse goal, object pose, mesh completion, or model input changes. The
inferred column is admission evidence only, followed by original closed-model
validation and independent contact/hold evaluation.
"""
import numpy as np
from .advisor32_rules import rigid, points_in_frame, goal_from_delta, parse_prediction, local_admission
from .advisor33_geometry import PAD, corridor, predicted_saturation, observe as visible_observe

VERSION = 'arx_advisor34_gravity_column_near335_v1'
NEAR_PLANE_M = .0335
SHALLOW_M = .0365
COLUMN_M = .025
COLUMN_Z_M = (.020, .048)


def column_intersections(points, down, box, *, include_z=False):
    """Exact segment/AABB intersection, without voxelization or sample inflation."""
    p = np.asarray(points, dtype=float); g = np.asarray(down, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or g.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Invalid gravity-column geometry')
    if not np.isclose(np.linalg.norm(g), 1., atol=1e-8, rtol=0):
        raise ValueError('Gravity direction must be a unit vector')
    enter = np.zeros(len(p)); leave = np.full(len(p), COLUMN_M)
    for i, name in enumerate(('x_EE_m', 'y_EE_m', 'z_EE_m')[:3 if include_z else 2]):
        lo, hi = COLUMN_Z_M if i == 2 else box[name]
        if abs(g[i]) < 1e-12:
            leave[(p[:, i] < lo) | (p[:, i] > hi)] = -1.
        else:
            a, b = (lo-p[:, i])/g[i], (hi-p[:, i])/g[i]
            enter = np.maximum(enter, np.minimum(a, b)); leave = np.minimum(leave, np.maximum(a, b))
    keep = enter <= leave+1e-12
    return keep, p[keep]+enter[keep, None]*g


def surface(points, box, down):
    points = np.asarray(points, dtype=float)
    selected = points[corridor(points, box)]
    center = float(np.median(points[:, 1])) if len(points) else None
    centered = center is not None and box['y_EE_m'][0] <= center <= box['y_EE_m'][1]
    origin = 'direct_visible_corridor'
    if not len(selected) and centered:
        _, selected = column_intersections(points, down, box)
        origin = 'gravity_column_corridor'
    return {'median_z_EE_m': float(np.median(selected[:, 2])) if len(selected) else None,
            'points': len(selected), 'origin': origin, 'lateral_center_EE_m': center,
            'center_between_jaws': bool(centered)}


def observe(projector, asset, row, depth):
    old, masks, points = visible_observe(projector, asset, row, depth)
    ee = rigid(row['snapshot']['link_world']['right_eef_link'])
    down = ee[:3, :3].T @ np.array([0., 0., -1.])
    box = old['box']; metrics = surface(points, box, down)
    extended, _ = column_intersections(points, down, box, include_z=True)
    occluded = bool(extended.any() and metrics['center_between_jaws'] and old['projection']['target_visibility_pass'])
    gate = {**old, 'version': VERSION, 'direct_visible_pass': old['pass'],
            'pass': bool(old['pass'] or occluded), 'gravity_down_EE': down.tolist(),
            'occlusion_envelope': {'pass': occluded, 'source_points': int(extended.sum()),
                'maximum_downward_extent_m': COLUMN_M, 'axial_band_EE_m': list(COLUMN_Z_M),
                'center_between_jaws': metrics['center_between_jaws'],
                'inferred_volume_is_not_observed_surface': True},
            'near_surface': metrics,
            'gate_origin': 'direct_visible' if old['pass'] else 'gravity_column' if occluded else None}
    # Preserve the actual visible-point counts. Inferred points are never added
    # to the native mask, neural image, or direct in-hand count.
    return gate, masks, points


def first_plane(current, prediction, points_EE, box):
    p, r = parse_prediction('D', prediction)
    lateral = p.copy(); lateral[2] = 0.
    candidate = goal_from_delta(current, lateral, r); current = rigid(current)
    transformed = points_in_frame(points_EE, current, candidate)
    down = candidate[:3, :3].T @ np.array([0., 0., -1.])
    metrics = surface(transformed, box, down); front = metrics['median_z_EE_m']
    if front is None: return {'pass': False, 'reason': 'first_plane_has_no_observed_or_bounded_column_surface'}
    slope = float(candidate[:3, 2] @ current[:3, 2])
    if slope <= 0: return {'pass': False, 'reason': 'first_plane_nonforward_axis'}
    desired = max(0., (front-NEAR_PLANE_M)/slope)
    if np.linalg.norm(p[:2]) > .065: return {'pass': False, 'reason': 'model_lateral_exceeds_65mm'}
    allowance = float(np.sqrt(max(0., .065**2-float(p[:2] @ p[:2]))))
    used = p.copy(); used[2] = min(desired, allowance)
    goal = goal_from_delta(current, used, r); admission = local_admission(current, goal)
    return {'pass': admission['pass'], 'reason': None if admission['pass'] else 'local_rotation_or_translation_bound',
            'goal_world': goal.tolist(), 'admission': admission, 'raw_translation_m': p.tolist(),
            'used_translation_m': used.tolist(), 'rotation_rad': r.tolist(), 'surface': metrics,
            'observed_surface_z_EE_m': front, 'near_plane_z_EE_m': NEAR_PLANE_M,
            'desired_axial_advance_m': desired, 'axial_cap_active': desired > allowance,
            'controller_generated_axial': True, 'model_generated_XY_and_rotation': True,
            'remaining_plane_gap_m': max(0., (desired-used[2])*slope)}


def compensate(current, dp, dr, points_EE, box):
    """Preserve given XY/rotation; add axial advance only if median Z >36.5mm."""
    current = rigid(current); p = np.asarray(dp, dtype=float).copy(); r = np.asarray(dr, dtype=float)
    proposed = goal_from_delta(current, p, r)
    transformed = points_in_frame(points_EE, current, proposed)
    down = proposed[:3, :3].T @ np.array([0., 0., -1.])
    metrics = surface(transformed, box, down); front = metrics['median_z_EE_m']
    result = {'pass': False, 'reason': None, 'raw_translation_m': p.tolist(),
              'rotation_rad': r.tolist(), 'predicted_surface': metrics, 'applied': False,
              'target_z_EE_m': NEAR_PLANE_M, 'trigger_z_EE_m': SHALLOW_M}
    if front is None: return {**result, 'reason': 'no_surface_for_preclosure_axial_alignment'}
    advance = 0.
    if front > SHALLOW_M:
        slope = float(proposed[:3, 2] @ current[:3, 2])
        if slope <= 0: return {**result, 'reason': 'preclosure_nonforward_axis'}
        advance = (front-NEAR_PLANE_M)/slope; p[2] += advance
    goal = goal_from_delta(current, p, r); admission = local_admission(current, goal)
    return {**result, 'pass': admission['pass'], 'reason': None if admission['pass'] else 'axial_alignment_exceeds_local_bound',
            'applied': advance > 0., 'additional_axial_advance_m': advance,
            'used_translation_m': p.tolist(), 'goal_world': goal.tolist(), 'admission': admission,
            'XY_and_rotation_preserved': True}
