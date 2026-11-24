"""Prospective contact/near-field rules; no robot or physical parameter edits."""
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, sha, digest, verify_files
from training.nearfield_arx import closure_recovery as policy
from training.nearfield_arx.contract import pose_error
from . import closedloop30_rules as old

VERSION = 'arx_advisor31_contact_visual_v1'
POLICY = {'version': VERSION, 'D_current_radius_m': .025,
          'D_small_prediction_m': .005, 'D_outward_increase_m': .003,
          'C_target_displacement_m': .003, 'contact_recent_seconds': .2,
          'nonempty_mean_finger_margin_m': .001, 'arm_stationary_rad_s': .01,
          'close_tracking_is_diagnostic': True, 'success_bilateral_still_required': True}


def fingers_touching(execution, snapshot=None):
    snap = snapshot or execution['final']
    target = snap['objects']['test_object']['path']
    end = float(snap['sim_time'])
    return sorted({n for c in execution['contacts']
                   if c['num_contacts'] > 0 and target in c['a']+c['b']
                   and end-POLICY['contact_recent_seconds']-1e-6 <= c['sim_time'] <= end+1e-6
                   for n in ('right_link27', 'right_link28') if n in c['a']+c['b']})


def near(current, g0, radius):
    error = pose_error(current, g0)
    return (error['translation_m'] <= radius and error['rotation_rad'] <= math.radians(30)), error


def interpret(method, token, inputs, executed, depth_proof=None, contact_proof=None):
    result = old.interpret(method, token, inputs, executed)
    result.update(interpreter_version=VERSION, advisor31=None)
    if contact_proof is not None and contact_proof['trigger']:
        if method != 'C' or token != 'GRASP':
            raise ValueError('Contact controller must issue an explicit labelled C closure request')
        result.update(request_closure=True, origin='C_contact_or_target_displacement',
                      in_place_without_extra_motion=True, advisor31={'contact': contact_proof})
        return result
    parts = policy.raw_parts(method, token)
    if method != 'D' or executed != 1 or parts is None or result['request_closure']:
        return result
    current = np.asarray(inputs['eef_world']); g0 = inputs['G0_world']
    goal = current.copy(); goal[:3, 3] += current[:3, :3] @ parts[0]
    goal[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(parts[1]).as_matrix()
    admitted, before = near(current, g0, POLICY['D_current_radius_m'])
    after = pose_error(goal, g0)
    saturated = False
    if depth_proof is not None:
        if depth_proof['goal_sha256'] != digest(goal.tolist()):
            raise ValueError('Depth prediction is not bound to the raw D target')
        saturated = not depth_proof['pass']
    flags = {'small_translation': math.hypot(*parts[0]) <= POLICY['D_small_prediction_m'],
             'outward': after['translation_m']-before['translation_m'] > POLICY['D_outward_increase_m'],
             'near_depth_saturation': bool(saturated)}
    trigger = admitted and any(flags.values())
    result['advisor31'] = {'near': admitted, 'intercept': bool(trigger), 'reasons': flags,
                           'current_to_G0': before, 'predicted_to_G0': after,
                           'predicted_goal_world': goal.tolist(),
                           'translation_norm_m': math.hypot(*parts[0]),
                           'depth': depth_proof}
    if trigger:
        result.update(request_closure=True, origin='D_second_step_nearfield_guard',
                      in_place_without_extra_motion=True)
    return result


def contact_from_request(request):
    row = read(request['observation']); ep = Path(request['observation']).parent.parent
    command = row.get('teacher_command') or {}; label = command.get('label', '')
    proof = {'version': VERSION, 'trigger': False, 'applicable': False,
             'source': 'Measured ARX contact/target state, not a model prediction', 'files': {}}
    if request['method'] != 'C' or not label.startswith('c_step_'):
        return proof
    paths = [ep/(label+s+'.json') for s in ('_command', '_execution', '_waypoint_stop')]
    paths += [ep/'capture_snapshot.json', Path(request['observation'])]
    stored, execution, stopped, initial, _ = [read(p) for p in paths]
    if stored['id'] != command['id'] or stored['label'] != label or not stopped['pass']:
        raise ValueError('C contact gate lacks a bound completed stationary step')
    verify_files(stopped['files'])
    if old.forbidden_contacts(execution):
        raise RuntimeError('C contact gate cannot continue after forbidden contact')
    snap = row['snapshot']
    if snap['sim_time'] <= execution['final']['sim_time']:
        raise ValueError('C contact observation must follow the completed motion')
    admissible, error = near(snap['link_world']['right_eef_link'], request['inputs']['G0_world'], .035)
    # The synchronized observation follows the completed step by one render
    # interval. Use contacts measured in that step's final 0.2 s, not a shifted
    # window that would discard every contact before the new rendered frame.
    touching = fingers_touching(execution)
    moved = float(np.linalg.norm(np.asarray(snap['objects']['test_object']['world'])[:3, 3]-
                                 np.asarray(initial['objects']['test_object']['world'])[:3, 3]))
    proof.update(applicable=True, trigger=bool(admissible and (touching or moved > POLICY['C_target_displacement_m'])),
                 current_to_G0=error, near=admissible, touching_fingers=touching,
                 target_displacement_m=moved, previous_command_id=stored['id'],
                 previous_command_label=label,
                 contact_observation_age_s=float(snap['sim_time']-execution['final']['sim_time']),
                 files={str(p): sha(p) for p in paths})
    return proof


def post_close(execution, snapshot, command, closed_m, *, active, fault):
    if command['label'] != 'close' or command['time_from_start'] != [0., 3.]:
        raise ValueError('Only original completed three-second close may use contact admission')
    if active or fault:
        raise RuntimeError('Post-close admission requires a healthy idle controller')
    if not execution['records'] or execution['records'][-1]['time'] < 3.-1e-5:
        raise ValueError('Incomplete close execution')
    if snapshot['sim_time'] < execution['final']['sim_time']-1e-6:
        raise ValueError('Post-close observation predates execution')
    names = snapshot['joint_names']; indices = [names.index(n) for n in snapshot['right_arm']]
    q = np.asarray(snapshot['joint_positions'])[indices]
    v = np.asarray(snapshot['joint_velocities'])[indices]
    limits = np.asarray(snapshot['joint_limits'])[indices]
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite post-close arm state')
    exceed = float(np.max(np.maximum(limits[:, 0]-q, q-limits[:, 1])))
    if exceed > 1e-6:
        raise RuntimeError('Actual arm exceeds frozen limits')
    positions = [float(snapshot['joint_positions'][names.index(n)]) for n in ('right_joint27', 'right_joint28')]
    if not np.isfinite(positions).all() or not math.isfinite(closed_m):
        raise ValueError('Nonfinite finger state')
    contacts = fingers_touching(execution, snapshot); forbidden = old.forbidden_contacts(execution)
    nonempty = float(np.mean(positions))-closed_m > POLICY['nonempty_mean_finger_margin_m']
    stopped = float(np.max(np.abs(v))) <= POLICY['arm_stationary_rad_s']
    return {'version': VERSION, 'pass': bool(stopped and not forbidden and (len(contacts)==2 or nonempty)),
            'stationary': bool(stopped), 'maximum_arm_speed_rad_s': float(np.max(np.abs(v))),
            'maximum_limit_exceed_rad': max(0., exceed), 'recent_fingers': contacts,
            'nonempty_finger_opening': bool(nonempty), 'finger_positions_m': positions,
            'frozen_closed_position_m': closed_m, 'forbidden_contacts': forbidden,
            'command_id': command['id'], 'snapshot': snapshot, 'controller_active': False,
            'controller_fault': None, 'close_tracking_error_used_for_admission': False,
            'physical_success_claimed': False}
