"""ARX finger corridor from frozen meshes and current RGB-D; no grasp prior."""
import numpy as np
from .advisor32_rules import backproject, points_in_frame, rigid, parse_prediction, goal_from_delta, local_admission

VERSION = 'arx_advisor33_finger_box_v1'
PAD = np.array([-.006, 0., .0375])


def finger_box(projector, row):
    ee = row['snapshot']['link_world']['right_eef_link']; patches = []; tips = []
    for name, (vertices, faces) in projector.fingers.items():
        v = points_in_frame(vertices, row['snapshot']['link_world'][name], ee)
        tri = v[faces]; center = tri.mean(axis=1)
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        # Extract the calibrated distal flat pad, excluding proximal linkage.
        mask = (np.abs(normal[:, 1]) > .999) & (center[:, 2] > .015) & (abs(center[:, 0]-PAD[0]) < .030)
        if not mask.any(): raise ValueError('Frozen distal finger patch not identifiable')
        p = tri[mask].reshape(-1, 3)
        patches.append({'name': name, 'min': p.min(0), 'max': p.max(0), 'y': float(np.median(p[:, 1]))})
        tips.append(float(v[:, 2].max()))
    x = [max(p['min'][0] for p in patches), min(p['max'][0] for p in patches)]
    y = sorted(p['y'] for p in patches)
    # EEF origin is the proximal cross-section; distal bound is the mesh tip.
    # Use the graspable finger envelope, not only its small flat distal patch:
    # successful insertions can put the visible top surface behind that patch.
    z = [0., min(tips)]
    if x[0] >= x[1] or y[0] >= y[1] or z[1] <= z[0]: raise ValueError('Invalid actual finger corridor')
    return {'version': VERSION, 'x_EE_m': list(map(float, x)), 'y_EE_m': list(map(float, y)),
            'z_EE_m': list(map(float, z)), 'pad_center_EE_m': PAD.tolist(),
            'z_relative_pad_m': [float(a-PAD[2]) for a in z],
            'basis': 'Distal flat-pad X envelope, measured jaw separation, calibrated EEF origin to frozen fingertip',
            'physics_or_calibration_changed': False}


def corridor(points, box, *, include_z=False):
    p = np.asarray(points); keep = np.ones(len(p), dtype=bool)
    for i, name in enumerate(('x_EE_m', 'y_EE_m', 'z_EE_m')[:3 if include_z else 2]):
        lo, hi = box[name]; keep &= (p[:, i] >= lo) & (p[:, i] <= hi)
    return keep


def observe(projector, asset, row, depth):
    masks, projection, _ = projector.project(asset, row, depth)
    box = finger_box(projector, row)
    valid = masks['target'] & np.isfinite(depth) & (depth > 0)
    points = backproject(depth, valid, row['intrinsics']) if valid.any() else np.empty((0, 3))
    local = points_in_frame(points, row['camera_world_optical'], row['snapshot']['link_world']['right_eef_link'])
    lateral = corridor(local, box); inside = corridor(local, box, include_z=True)
    p = local[lateral]
    info = {'version': VERSION, 'box': box, 'projection': projection,
            'target_points': len(local), 'corridor_points': int(lateral.sum()), 'in_hand_points': int(inside.sum()),
            'pass': bool(inside.any() and projection['target_visibility_pass']),
            'corridor_median_z_EE_m': float(np.median(p[:, 2])) if len(p) else None,
            'corridor_median_z_relative_pad_m': float(np.median(p[:, 2])-PAD[2]) if len(p) else None,
            'target_direction_EE_xy': (np.median(local[:, :2], axis=0)-PAD[:2]).tolist() if len(local) else [0., 0.],
            'current_optical_median_m': float(np.median(points[:, 2])) if len(points) else None,
            'closure_collision_check_required': True}
    return info, masks, local


def first_plane(current, prediction, points_EE, box):
    p, r = parse_prediction('D', prediction)
    lateral = p.copy(); lateral[2] = 0.
    without_axial = goal_from_delta(current, lateral, r)
    # Old EE point coordinates become coordinates of the rotated/translating
    # prospective hand. No numerical object/goal pose is consumed.
    transformed = points_in_frame(points_EE, current, without_axial)
    selected = transformed[corridor(transformed, box)]
    if not len(selected): return {'pass': False, 'reason': 'first_plane_has_no_observed_corridor_surface'}
    current, without_axial = rigid(current), rigid(without_axial)
    slope = float(without_axial[:3, 2] @ current[:3, 2])
    if slope <= 0: return {'pass': False, 'reason': 'first_plane_nonforward_axis'}
    front = float(np.median(selected[:, 2])); plane = box['z_EE_m'][1]
    desired = max(0., (front-plane)/slope)
    if np.linalg.norm(p[:2]) > .065:
        return {'pass': False, 'reason': 'model_lateral_exceeds_65mm'}
    allowance = float(np.sqrt(max(0., .065**2-float(p[:2] @ p[:2]))))
    used = p.copy(); used[2] = min(desired, allowance)
    goal = goal_from_delta(current, used, r); guard = local_admission(current, goal)
    return {'pass': guard['pass'], 'reason': None if guard['pass'] else 'local_rotation_or_translation_bound',
            'goal_world': goal.tolist(), 'admission': guard, 'raw_translation_m': p.tolist(),
            'used_translation_m': used.tolist(), 'rotation_rad': r.tolist(),
            'observed_surface_z_EE_m': front, 'near_plane_z_EE_m': float(plane),
            'desired_axial_advance_m': desired, 'axial_cap_active': desired > allowance,
            'controller_generated_axial': True, 'model_generated_XY_and_rotation': True,
            'remaining_plane_gap_m': max(0., (desired-used[2])*slope)}


def predicted_saturation(row, local_points, goal, near_m):
    ee = rigid(row['snapshot']['link_world']['right_eef_link'])
    camera = rigid(row['camera_world_optical']); future = rigid(goal) @ np.linalg.inv(ee) @ camera
    points = points_in_frame(local_points, ee, future)
    return bool(len(points) and np.percentile(points[:, 2], 1) < near_m)
