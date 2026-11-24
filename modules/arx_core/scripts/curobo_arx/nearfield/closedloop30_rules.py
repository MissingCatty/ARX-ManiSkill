"""Prospective advisor30 rules; old receipts and the original evaluator stay intact."""
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, sha, digest, verify_files
from scripts.curobo_arx.settling import assess_settling
from training.show_harness_arx.contract import rigid
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx import closure_recovery as original
from .waypoint_continuation import forbidden_contacts
from .closure import observation_timing, bind as original_bind, validate_reply as original_validate

VERSION = 'arx_closedloop_advisor30_v1'
POLICY = {'version': VERSION, 'D_axial_remaining_m': .025,
          'D_nearfield_radius_m': .035, 'D_core_radius_m': .018,
          'D_outward_exit_m': .035, 'D_outward_min_increase_m': .010,
          'maximum_arm_speed_rad_s': .01,
          'entry_backoffs_m': [.08, .07, .06, .05, .04],
          'entry_fallback_object': 'household_ycb_spam_can',
          'entry_fallback_conditions': ['entry_offset', 'joint_prior_entry_offset'],
          'intermediate_tracking_error_is_diagnostic': True,
          'closure_hold': 'current measured idle joints, independently checked before dispatch',
          'original_settling_receipts_preserved': True,
          'physical_success': 'all original physical checks except certified open waypoint convergence'}


def waypoint(label):
    return label == 'd_entry_axis' or label.startswith(('c_step_', 'd_refine_'))


def entry_candidates(trial):
    nominal, g0 = rigid(trial['entry_world']), rigid(trial['G0_world'])
    adaptive = (trial['object_id'] == POLICY['entry_fallback_object'] and
                trial['condition'] in POLICY['entry_fallback_conditions'])
    rows = []
    for backoff in POLICY['entry_backoffs_m'] if adaptive else [.08]:
        goal = nominal.copy()
        goal[:3, 3] += g0[:3, 2] * (.08-backoff)
        rows.append({'backoff_m': backoff, 'goal_world': goal.tolist()})
    return rows


def axial_goal(current, g0):
    current, g0 = rigid(current), rigid(g0)
    axial = float((g0[:3, 3]-current[:3, 3]) @ g0[:3, 2])
    distance = max(0., axial-POLICY['D_axial_remaining_m'])
    if distance > .10:
        raise ValueError('Axial preparation exceeds the registered entrance range')
    goal = current.copy(); goal[:3, 3] += distance*g0[:3, 2]
    return {'goal_world': goal.tolist(), 'distance_m': distance,
            'current_axial_distance_m': axial, 'Gstar_used': False,
            'preserves_measured_lateral_position_and_orientation': True}


def interpret(method, token, inputs, executed):
    result = original.closure_trigger(method, token, inputs, executed)
    result = {**result, 'interpreter_version': VERSION, 'anti_regression': None}
    parts = original.raw_parts(method, token)
    if method != 'D' or executed != 1 or parts is None or result['request_closure']:
        return result
    current, g0 = rigid(inputs['eef_world']), rigid(inputs['G0_world'])
    goal = current.copy(); goal[:3, 3] += current[:3, :3] @ parts[0]
    goal[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(parts[1]).as_matrix()
    before, after = pose_error(current, g0), pose_error(goal, g0)
    stop = (before['translation_m'] < POLICY['D_core_radius_m'] and
            before['rotation_rad'] <= math.radians(original.POLICY['closure_region_rotation_degrees']) and
            after['translation_m'] > POLICY['D_outward_exit_m'] and
            after['translation_m']-before['translation_m'] >= POLICY['D_outward_min_increase_m'])
    result['anti_regression'] = {'intercept': bool(stop), 'current_to_G0': before,
                                'predicted_to_G0': after, 'predicted_goal_world': goal.tolist()}
    if stop:
        result.update(request_closure=True, origin='D_second_step_outward_intercept',
                      in_place_without_extra_motion=True)
    return result


def stationary(execution, snapshot, *, active, fault, label):
    if active or fault:
        raise RuntimeError('Waypoint/closure requires a confirmed healthy idle controller')
    if not waypoint(label):
        raise ValueError('Only registered open policy waypoints have continuation semantics')
    records = execution['records']
    times = np.asarray([r['time'] for r in records], float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Missing or invalid actual movement chronology')
    names = snapshot['joint_names']; arm = snapshot['right_arm']
    if len(names) != len(set(names)) or arm != execution['final']['right_arm']:
        raise ValueError('ARX joint identity changed')
    q = np.asarray([snapshot['joint_positions'][names.index(n)] for n in arm], float)
    v = np.asarray([snapshot['joint_velocities'][names.index(n)] for n in arm], float)
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite measured arm state')
    if snapshot['sim_time'] < execution['final']['sim_time']-1e-5:
        raise ValueError('Idle snapshot precedes movement completion')
    contacts = forbidden_contacts(execution)
    speed = float(np.max(np.abs(v)))
    return {'version': VERSION, 'pass': bool(speed <= .01 and not contacts),
            'label': label, 'maximum_arm_speed_rad_s': speed,
            'forbidden_contact_events': len(contacts), 'controller_active': False,
            'controller_fault': None, 'snapshot': snapshot,
            'scope': 'Measured stop and next observation permission; not commanded-pose convergence'}


def bind_waypoint(episode, label, execution, snapshot, desired, *, active, fault):
    episode = Path(episode)
    paths = [episode/(label+s+'.json') for s in ('_command', '_execution', '_settling')]
    files = {str(p): sha(p) for p in paths}
    command, stored, settling = [read(p) for p in paths]
    requests = read(episode/'requests.json')
    if not requests or requests[-1]['id'] != command['id'] or requests[-1]['label'] != label:
        raise RuntimeError('Waypoint proof is not bound to the last dispatched request')
    if digest(stored) != digest(execution):
        raise RuntimeError('Execution receipt changed')
    expected = assess_settling(execution, desired)
    from training.nearfield_arx.receipt_numbers import same_settling_diagnostics
    if not same_settling_diagnostics(expected, settling):
        raise RuntimeError('Original settling diagnosis changed')
    return {**stationary(execution, snapshot, active=active, fault=fault, label=label),
            'files': files, 'command_id': command['id'], 'desired_world': np.asarray(desired).tolist(),
            'original_arrival_accepted': bool(settling['accepted']), 'original_settling': settling}


def bind_closure(root, snapshot, observation, command, desired, *, active, fault,
                 request_id, configuration_identity, sample_seconds=.2):
    if command is None or not waypoint(command['label']):
        return original_bind(root, snapshot, observation, command, desired, active=active, fault=fault,
            request_id=request_id, configuration_identity=configuration_identity, sample_seconds=sample_seconds)
    root = Path(root); label = command['label']
    age = snapshot['sim_time']-observation['frame_time']
    if not math.isfinite(age) or age < -.00001 or age > sample_seconds+.01:
        raise RuntimeError('Closure observation is stale')
    timing = observation_timing(observation)
    if not observation.get('visual_geometry', {}).get('pass'):
        return {'pass': False, 'rejected': 'current_visual_geometry_rejected'}
    execution = read(root/(label+'_execution.json'))
    proof = bind_waypoint(root, label, execution, snapshot, desired, active=active, fault=fault)
    if read(root/(label+'_command.json')) != command:
        raise RuntimeError('Closure previous command changed')
    if any(r['label'] == 'close' for r in read(root/'requests.json')):
        raise RuntimeError('Duplicate closure')
    if not proof['pass']:
        return {'pass': False, 'rejected': 'measured_closure_not_stationary', 'stationary': proof}
    if not stationary(execution, observation['snapshot'], active=active, fault=fault, label=label)['pass']:
        return {'pass': False, 'rejected': 'closure_observation_not_stationary'}
    names = snapshot['right_arm']
    if command['joint_names'] != names:
        raise RuntimeError('Closure ARX joint order changed')
    q = [float(snapshot['joint_positions'][snapshot['joint_names'].index(n)]) for n in names]
    limits = np.asarray([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names])
    if np.max(np.maximum(limits[:, 0]-q, q-limits[:, 1])) > 1e-6:
        raise RuntimeError('Measured hold target exceeds frozen joint limits')
    return {'pass': True, 'version': VERSION, 'grasp_request_id': request_id,
            'attempt_id': observation['attempt_id'], 'observation_time_s': observation['frame_time'],
            'observation_sha256': digest(observation), 'observation_timing': timing,
            'configuration_identity': configuration_identity, 'previous_command_id': command['id'],
            'previous_command_label': label, 'hold_joint_names': names, 'hold_q': q,
            'snapshot_sha256': digest(snapshot), 'measured_hold_snapshot': snapshot,
            'arrival': proof, 'files': proof['files'], 'motion_dispatched': False,
            'hold_basis': 'fresh measured idle joints; original target error remains diagnostic'}


def validate_closure(binding, reply, command):
    if binding.get('version') != VERSION:
        return original_validate(binding, reply, command)
    if not reply.get('pass'):
        return False
    verify_files(binding['files'])
    if reply.get('binding_sha256') != digest(binding) or reply['hold_q'] != binding['hold_q']:
        raise RuntimeError('Closure certificate is not bound to the measured hold state')
    snap = binding['measured_hold_snapshot']
    expected = [float(snap['joint_positions'][snap['joint_names'].index(n)]) for n in snap['right_arm']]
    if expected != binding['hold_q'] or digest(snap) != binding['snapshot_sha256']:
        raise RuntimeError('Measured closure state changed')
    if command['id'] != binding['previous_command_id'] or command['label'] != binding['previous_command_label']:
        raise RuntimeError('Closure preceding command changed')
    return True
