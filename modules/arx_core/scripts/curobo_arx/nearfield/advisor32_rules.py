"""Prospective local control rules, independent of any coarse or teacher goal.

This module deliberately has no registration, trial, seed, or G0 argument. It
is not installed into an existing runtime. The camera-distance contract must
be resolved before a new physical campaign can enable the first-plane rule.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


VERSION = 'arx_advisor32_local_rules_v1'
LIMITS = {'translation_m': .065, 'rotation_degrees': 25.,
          'small_translation_m': .005, 'small_rotation_degrees': 4.,
          'retreat_m': -.003, 'opposed_lateral_m': .005,
          'reported_speed_rad_s': .01, 'window_seconds': 2.,
          'window_minimum_samples': 10, 'position_range_m': .0005,
          'differential_speed_rad_s': .01, 'wait_wall_seconds': 10.}


def rigid(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(a[:3, :3])-1) > 1e-5):
        raise ValueError('Expected a finite proper rigid transform')
    return a.copy()


def parse_prediction(method, text):
    if method not in ('C', 'D') or not isinstance(text, str):
        raise ValueError('Expected a registered C/D text prediction')
    parts = text.strip().split()
    if parts == ['GRASP']:
        return None
    if len(parts) != 7 or parts[0] != ('REFINE' if method == 'D' else 'STEP'):
        raise ValueError('Expected six local coordinates or GRASP')
    values = np.asarray([float(x) for x in parts[1:]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction')
    return values[:3]/1000., np.deg2rad(values[3:])


def goal_from_delta(current, translation, rotation):
    ee = rigid(current)
    p, r = np.asarray(translation, dtype=float), np.asarray(rotation, dtype=float)
    if p.shape != (3,) or r.shape != (3,) or not np.isfinite(np.r_[p, r]).all():
        raise ValueError('Expected finite three-vector translation and rotation')
    goal = ee.copy()
    goal[:3, 3] += ee[:3, :3] @ p
    goal[:3, :3] = ee[:3, :3] @ Rotation.from_rotvec(r).as_matrix()
    return rigid(goal)


def local_admission(current, goal):
    a, b = rigid(current), rigid(goal)
    distance = float(np.linalg.norm(b[:3, 3]-a[:3, 3]))
    angle = float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))
    return {'pass': distance <= LIMITS['translation_m']+1e-12
                    and angle <= LIMITS['rotation_degrees']+1e-9,
            'translation_m': distance, 'rotation_degrees': angle,
            'IK_and_collision_checks_still_required': True,
            'physical_execution_proven': False}


def compose_first_goal(current, prediction, axial_advance_m, method='D'):
    """Keep the model's XY/rotation; axial advance must come from perception.

    Admission rejects an excessive composed step rather than clipping away the
    model's lateral correction. The caller must record how depth was converted.
    """
    parts = parse_prediction(method, prediction)
    if parts is None:
        raise ValueError('GRASP does not define a first-plane translation')
    p, r = parts
    if not math.isfinite(axial_advance_m) or axial_advance_m < 0:
        raise ValueError('An axial insertion must be finite and nonnegative')
    used = p.copy(); used[2] = axial_advance_m
    goal = goal_from_delta(current, used, r)
    return {'goal_world': goal.tolist(), 'admission': local_admission(current, goal),
            'raw_translation_m': p.tolist(), 'used_translation_m': used.tolist(),
            'rotation_rad': r.tolist(), 'model_axial_prediction_overridden': True,
            'controller_axial_source': 'Explicitly bound current RGB-D distance contract'}


def backproject(depth, mask, intrinsics):
    z, m, k = np.asarray(depth), np.asarray(mask), np.asarray(intrinsics, dtype=float)
    if z.ndim != 2 or m.shape != z.shape or m.dtype != np.bool_ or k.shape != (3, 3):
        raise ValueError('Invalid depth/mask/intrinsics')
    if not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError('Noninvertible intrinsics')
    rows, cols = np.nonzero(m & np.isfinite(z) & (z > 0))
    if not len(rows):
        raise ValueError('No valid visible target depth')
    rays = np.c_[cols, rows, np.ones(len(rows))] @ np.linalg.inv(k).T
    return rays * z[rows, cols, None]


def points_in_frame(points, source_world, destination_world):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Invalid observed points')
    t = np.linalg.inv(rigid(destination_world)) @ rigid(source_world)
    return p @ t[:3, :3].T + t[:3, 3]


def optical_first_advance(current, camera, points_camera, prediction, target_depth_m=.0235):
    """Evaluate the advisor's literal camera-Z rule without executing it.

    The mounted optical axis differs from local EE +Z. Account for both the
    model's rotation/lateral translation and the full camera extrinsics.
    """
    ee, cam = rigid(current), rigid(camera)
    p, r = parse_prediction('D', prediction)
    lateral = p.copy(); lateral[2] = 0.
    candidate = goal_from_delta(ee, lateral, r)
    future_camera = candidate @ np.linalg.inv(ee) @ cam
    future_points = points_in_frame(points_camera, cam, future_camera)
    if not len(future_points) or not np.isfinite(target_depth_m) or target_depth_m <= 0:
        raise ValueError('Missing target surface or invalid near plane')
    slope = float(future_camera[:3, 2] @ ee[:3, 2])
    if slope <= 1e-6:
        raise ValueError('EE insertion does not reduce camera optical depth')
    median = float(np.median(future_points[:, 2]))
    advance = max(0., (median-target_depth_m)/slope)
    composed = compose_first_goal(ee, prediction, advance)
    return {**composed, 'axial_advance_m': advance,
            'camera_depth_before_insertion_m': median,
            'EE_axial_to_camera_optical_slope': slope,
            'target_optical_depth_m': target_depth_m,
            'diagnostic_only': True}


def second_decision(prediction, visual, method='D'):
    """Interpret a second prediction using measured visual facts only.

    `visual` is a closed schema. No object pose, goal, trial or previous target
    can be smuggled through this API. An affirmative result is a request for
    independent closed-model validation, never physical grasp success.
    """
    required = {'target_pixels', 'between_fingers', 'depth_gate_pass',
                'target_direction_EE_xy', 'predicted_near_saturation'}
    if set(visual) != required:
        raise ValueError('Unexpected or missing visual gate fields')
    direction = np.asarray(visual['target_direction_EE_xy'], dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError('Invalid target direction')
    in_hand = (type(visual['target_pixels']) is int and visual['target_pixels'] > 0
               and visual['between_fingers'] is True and visual['depth_gate_pass'] is True)
    parts = parse_prediction(method, prediction)
    reasons = []
    if parts is None:
        reasons.append('model_GRASP')
    else:
        p, r = parts
        if np.linalg.norm(p) <= .005+1e-12 and np.linalg.norm(r) <= math.radians(4)+1e-12:
            reasons.append('small_visual_residual')
        if visual['between_fingers']:
            if p[2] < -.003:
                reasons.append('backward_motion')
            if np.linalg.norm(p[:2]) > .005 and float(p[:2] @ direction) < 0:
                reasons.append('lateral_motion_opposes_target')
            if visual['predicted_near_saturation']:
                reasons.append('predicted_near_depth_saturation')
    return {'request_closure': bool(reasons and in_hand), 'reasons': reasons,
            'visual_closure_admission': bool(in_hand),
            'independent_closure_check_required': True,
            'model_explicit_GRASP': parts is None,
            'motion_is_not_automatically_valid': True}


def lift_goals(measured_closed, neutral_rotation=None):
    """Measured grasp + world Z. Optional neutral comes from robot convention.

    Keep-orientation candidates precede every neutral-rotation fallback. A
    caller may not reuse a preclosure target as the measured closed state.
    """
    grasp = rigid(measured_closed)
    turns = [np.zeros(3)]
    if neutral_rotation is not None:
        neutral = np.eye(4); neutral[:3, :3] = np.asarray(neutral_rotation)
        neutral = rigid(neutral)
        rv = Rotation.from_matrix(grasp[:3, :3].T @ neutral[:3, :3]).as_rotvec()
        length = float(np.linalg.norm(rv))
        if length > 1e-12:
            rv *= min(1., math.radians(10)/length)
            turns.extend([.5*rv, rv])
    candidates = []
    for rv in turns:
        for height in (.12, .10):
            goal = grasp.copy(); goal[2, 3] += height
            goal[:3, :3] = grasp[:3, :3] @ Rotation.from_rotvec(rv).as_matrix()
            candidates.append({'index': len(candidates), 'height_m': height,
                'rotation_degrees': float(np.rad2deg(np.linalg.norm(rv))),
                'goal_world': goal.tolist(), 'basis': 'measured closed hand + world Z'})
    return candidates


@dataclass(frozen=True)
class StopSample:
    sim_time: float
    position: tuple
    joints: tuple
    velocities: tuple


def settle(samples, *, active, fault, elapsed_wall_s):
    """Two independent measured stop routes; do not fabricate low velocities.

    At 5 Hz ten samples span only 1.8 s. Require BOTH >=10 samples AND >=2 s.
    Position-window acceptance is reported separately from commanded arrival.
    """
    if fault:
        raise RuntimeError('Controller fault: no continuation')
    if not math.isfinite(elapsed_wall_s) or elapsed_wall_s < 0:
        raise ValueError('Invalid stop wait time')
    result = {'pass': False, 'origin': None, 'controller_active': bool(active),
              'timeout': elapsed_wall_s >= LIMITS['wait_wall_seconds'],
              'window_applicable': False, 'commanded_arrival_proven': False}
    if result['timeout'] or active or not samples:
        return result
    t = np.asarray([s.sim_time for s in samples], dtype=float)
    p = np.asarray([s.position for s in samples], dtype=float)
    q = np.asarray([s.joints for s in samples], dtype=float)
    v = np.asarray([s.velocities for s in samples], dtype=float)
    if (p.shape != (len(samples), 3) or q.ndim != 2 or not q.shape[1] or v.shape != q.shape
            or not all(np.isfinite(x).all() for x in (t, p, q, v))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Missing/nonfinite/nonmonotonic measured stop samples')
    reported = float(np.max(np.abs(v[-1])))
    result['reported_maximum_speed_rad_s'] = reported
    if reported <= LIMITS['reported_speed_rad_s']:
        result.update(origin='settled_by_reported_velocity')
        result['pass'] = True
        return result
    # Include the sample just before the two-second boundary, rather than
    # accidentally accepting a short window or aggregating an entire motion.
    starts = np.flatnonzero(t <= t[-1]-LIMITS['window_seconds']+1e-9)
    if not len(starts):
        return result
    first = int(starts[-1]); times, positions, joints = t[first:], p[first:], q[first:]
    if len(times) < LIMITS['window_minimum_samples'] or np.max(np.diff(times)) > .25+1e-6:
        return result
    distances = np.linalg.norm(positions[:, None, :]-positions[None, :, :], axis=2)
    diameter = float(np.max(distances))
    speed = float(np.max(np.abs(np.diff(joints, axis=0)/np.diff(times)[:, None])))
    result.update(window_applicable=True, window_samples=len(times),
                  window_duration_s=float(times[-1]-times[0]),
                  maximum_TCP_position_range_m=diameter,
                  differential_joint_speed_rad_s=speed)
    if diameter <= LIMITS['position_range_m']+1e-12 and speed <= LIMITS['differential_speed_rad_s']+1e-12:
        result.update(origin='settled_by_position_window')
        result['pass'] = True
    return result


def continuation_after_slot(*, record_intact, configuration_intact, services_healthy,
                            old_world_absent, new_world_reset_verified):
    """A terminal slot does not erase infrastructure/integrity failures.

    An isolated timeout may continue only after teardown and a verified fresh
    reset. This is not a retry of the failed slot and does not delete its STOP.
    """
    fields = {'record_intact': record_intact, 'configuration_intact': configuration_intact,
              'services_healthy': services_healthy, 'old_world_absent': old_world_absent,
              'new_world_reset_verified': new_world_reset_verified}
    if any(type(v) is not bool for v in fields.values()):
        raise ValueError('Continuation evidence must be explicit booleans')
    return {'continue_next_slot': all(fields.values()), 'evidence': fields,
            'retry_failed_slot': False}
