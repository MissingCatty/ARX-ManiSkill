"""C-only scheduling permission; never converts failed arrival into convergence."""
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, sha, digest
from scripts.curobo_arx.settling import assess_settling

VERSION = 'arx_stationary_C_waypoint_v1'
POLICY = {'version': VERSION, 'maximum_arm_speed_rad_s': .01,
          'tail_seconds': 1., 'minimum_tail_span_s': .75, 'maximum_joint_drift_rad': .0005,
          'closure_arrival_and_final_evaluator_unchanged': True}


def forbidden_contacts(execution):
    target = execution['final']['objects']['test_object']['path']
    return [c for c in execution['contacts'] if c['num_contacts'] > 0
            and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])
            and not (target in c['a'] + c['b'] and any(n in c['a'] + c['b'] for n in ('right_link27', 'right_link28')))]


def assess(method, label, execution, snapshot, *, active, fault):
    if active or fault:
        raise RuntimeError('Waypoint continuation requires confirmed idle healthy controller')
    if method != 'C' or not label.startswith('c_step_'):
        raise ValueError('Only a non-closure C waypoint may use this scheduling rule')
    records = execution['records']
    if not records:
        raise ValueError('Missing real movement records')
    times = np.asarray([r['time'] for r in records], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid movement record clock')
    tail = [r for r in records if r['time'] >= times[-1] - POLICY['tail_seconds']]
    positions = np.asarray([r['position'] for r in tail], dtype=float)
    arm = snapshot['right_arm']
    if arm != execution['final']['right_arm'] or positions.ndim != 2 or positions.shape[1] != len(arm):
        raise ValueError('ARX arm/record shape changed')
    speeds = dict(zip(snapshot['joint_names'], snapshot['joint_velocities']))
    current = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
    speed = float(np.max(np.abs([speeds[n] for n in arm])))
    drift = float(np.max(np.ptp(positions, axis=0)))
    current_drift = float(np.max(np.abs(np.asarray([current[n] for n in arm]) - positions[-1])))
    span = float(times[-1] - tail[0]['time'])
    if not np.isfinite(positions).all() or not np.isfinite([speed, drift, current_drift, span]).all():
        raise ValueError('Nonfinite measured waypoint state')
    contact_count = len(forbidden_contacts(execution))
    passed = (speed <= POLICY['maximum_arm_speed_rad_s'] and span >= POLICY['minimum_tail_span_s']
              and drift <= POLICY['maximum_joint_drift_rad'] and current_drift <= POLICY['maximum_joint_drift_rad']
              and not contact_count)
    return {'version': VERSION, 'pass': bool(passed), 'method': method, 'label': label,
            'maximum_arm_speed_rad_s': speed, 'tail_span_s': span, 'tail_joint_drift_rad': drift,
            'fresh_joint_drift_rad': current_drift, 'forbidden_contact_events': contact_count,
            'snapshot': snapshot, 'scope': 'Next C observation only; not arrival or grasp success'}


def bind(episode, label, execution, snapshot, desired, *, active, fault):
    episode = Path(episode)
    files = {str(episode / (label + suffix + '.json')): sha(episode / (label + suffix + '.json'))
             for suffix in ('_command', '_execution', '_settling')}
    command = read(episode / (label + '_command.json'))
    requests = read(episode / 'requests.json')
    if not requests or requests[-1]['id'] != command['id'] or requests[-1]['label'] != label:
        raise RuntimeError('Waypoint is not the last dispatched command')
    if digest(read(episode / (label + '_execution.json'))) != digest(execution):
        raise RuntimeError('Waypoint execution differs from recorded receipt')
    original = assess_settling(execution, desired)
    if digest(read(episode / (label + '_settling.json'))) != digest(original):
        raise RuntimeError('Original convergence record changed')
    return {**assess('C', label, execution, snapshot, active=active, fault=fault),
            'files': files, 'command_id': command['id'], 'desired_world': np.asarray(desired).tolist(),
            'original_arrival_accepted': original['accepted'], 'original_settling': original}
