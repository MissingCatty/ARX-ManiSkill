"""Seed provenance and additional eligibility checks, never relaxed task gates."""
import copy
from pathlib import Path
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, sha, verify_files, digest

POLICY = {'version': 'success_seed_join_v1', 'joint_match_rad': .005,
          'arm_velocity_match_rad_s': .01, 'finger_match_rad': .002,
          'object_position_match_m': .001, 'object_rotation_match_rad': .01,
          'tail_start_match_rad': .01}
TAIL = ('approach', 'close', 'lift', 'hold')
ANCHOR_WAIT = {'minimum_sim_seconds': 1.0, 'maximum_sim_seconds': 2.0}


def controls_ready(seed_id, results):
    controls=[r for r in results if r['seed_id']==seed_id and r['policy']=='control']
    return (len(controls)==2 and len({r['trial_id'] for r in controls})==2
            and all(r.get('completed') and r.get('admitted') for r in controls))


def joint_values(snapshot, names, field='joint_positions'):
    values = dict(zip(snapshot['joint_names'], snapshot[field]))
    return np.array([values[name] for name in names], dtype=float)


def command_payload(command):
    """Only the per-execution nonce may differ from the successful source."""
    return {k: v for k, v in command.items() if k != 'id'}


def replay_command(seed, phase):
    path = Path(seed['episode'])/(phase+'_command.json')
    if sha(path) != seed['files'][str(path)]:
        raise RuntimeError('Registered successful command changed')
    command = copy.deepcopy(read(path)); command['id'] = uuid.uuid4().hex
    return command


def source_seed(row):
    root = Path(row['episode'])
    required = ['episode.json', 'actual_entry.json', 'selected_grasp.json', 'initial_conditions.json',
                'capture_snapshot.json', 'quality_receipt.json', 'physical_receipt.json', 'visual_receipt.json',
                'approach_scene.json', 'lift_scene.json']
    required += [phase+suffix for phase in TAIL
                 for suffix in ('_command.json', '_execution.json', '_settling.json')]
    required += [phase+'_optimization.json' for phase in ('approach', 'lift')]
    paths = [root/name for name in required]
    if any(not path.is_file() for path in paths):
        raise ValueError('Successful seed is missing command/execution or provenance files')
    for name in ('quality_receipt.json', 'physical_receipt.json', 'visual_receipt.json'):
        if read(root/name)['pass'] is not True:
            raise ValueError('Seed must pass physical, image and record admission')
    ledger_receipt = row['receipt']
    if isinstance(ledger_receipt, str):
        import json
        ledger_receipt = json.loads(ledger_receipt)
    if sha(root/'quality_receipt.json') != ledger_receipt['quality_sha256'] or sha(root/'physical_receipt.json') != ledger_receipt['physical_sha256']:
        raise ValueError('Seed ledger acceptance hashes do not match')
    meta = read(root/'episode.json'); entry = read(root/'actual_entry.json')['snapshot']
    command = read(root/'approach_command.json')
    if meta['attempt_id'] != row['id'] or meta['object_id'] != row['object_id'] or not meta['write_complete']:
        raise ValueError('Seed identity or durable completion mismatch')
    if command['joint_names'] != entry['right_arm']:
        raise ValueError('Seed ARX joint order mismatch')
    q0 = np.array(command['position'][0])
    if np.max(np.abs(joint_values(entry, entry['right_arm'])-q0)) > .001:
        raise ValueError('Seed approach start differs from actual source entry')
    if np.max(np.abs(joint_values(entry, entry['right_arm'], 'joint_velocities'))) > .01:
        raise ValueError('Choose a measured stationary source entry')
    return {'seed_id': row['id'], 'object_id': row['object_id'], 'episode': str(root),
            'renderer': meta['renderer'], 'placement': meta['collection_placement'],
            'asset_sha256': meta['asset_sha256'], 'goal': read(root/'selected_grasp.json'),
            'anchor_world': entry['link_world']['right_eef_link'],
            'anchor_q': q0.tolist(), 'anchor_snapshot': entry,
            'files': {str(path): sha(path) for path in paths}}


def perturbed_entry(seed, translation, rotation_degrees):
    delta = np.asarray(translation, dtype=float); omega = np.asarray(rotation_degrees, dtype=float)
    if delta.shape != (3,) or omega.shape != (3,) or not np.isfinite([delta, omega]).all():
        raise ValueError('Invalid registered entry perturbation')
    if np.linalg.norm(delta) > .025+1e-12 or np.linalg.norm(omega) > 10+1e-12:
        raise ValueError('Entry perturbation exceeds pilot bounds')
    offset = np.eye(4); offset[:3, 3] = delta
    offset[:3, :3] = Rotation.from_rotvec(np.deg2rad(omega)).as_matrix()
    return (np.asarray(seed['anchor_world'])@offset).tolist()


def join_check(seed, snapshot, policy=POLICY):
    if policy != POLICY:
        raise ValueError('Unregistered join eligibility policy')
    source = seed['anchor_snapshot']; names = source['right_arm']
    if names != snapshot['right_arm'] or source['joint_names'] != snapshot['joint_names']:
        raise RuntimeError('ARX snapshot joint identity changed')
    measured = joint_values(snapshot, names)
    finger_names = ['right_joint27', 'right_joint28']
    obj = np.asarray(snapshot['objects']['test_object']['world'])
    expected = np.asarray(source['objects']['test_object']['world'])
    metrics = {
        'joint_match_rad': float(np.max(np.abs(measured-np.array(seed['anchor_q'])))),
        'arm_velocity_match_rad_s': float(np.max(np.abs(joint_values(snapshot, names, 'joint_velocities')-
                                                       joint_values(source, names, 'joint_velocities')))),
        'finger_match_rad': float(np.max(np.abs(joint_values(snapshot, finger_names)-joint_values(source, finger_names)))),
        'object_position_match_m': float(np.linalg.norm(obj[:3, 3]-expected[:3, 3])),
        'object_rotation_match_rad': float(Rotation.from_matrix(obj[:3, :3]@expected[:3, :3].T).magnitude()),
    }
    return {'pass': all(np.isfinite(value) and value <= policy[key] for key, value in metrics.items()),
            'metrics': metrics, 'policy': policy, 'snapshot': snapshot,
            'scope': 'Additional seed-replay eligibility; original convergence and physical success gates still required'}


def anchor_readiness(seed, snapshot, elapsed_sim_seconds):
    if not np.isfinite(elapsed_sim_seconds) or elapsed_sim_seconds < -1e-6:
        raise RuntimeError('Invalid anchor settling clock')
    result = join_check(seed, snapshot)
    ready = elapsed_sim_seconds >= ANCHOR_WAIT['minimum_sim_seconds']-1e-6 and result['pass']
    terminal = ready or elapsed_sim_seconds >= ANCHOR_WAIT['maximum_sim_seconds']-1e-6
    result.update(anchor_wait={'policy': ANCHOR_WAIT,
        'elapsed_sim_seconds': elapsed_sim_seconds,
        'status': 'ready' if ready else ('rejected' if terminal else 'waiting'),
        'preceding_accepted_position_target_held': True})
    result['point_state_matches'] = result['pass']; result['pass'] = ready
    return result


def audit_episode(episode, seed, mode):
    """Audit all occurred actions, including a connector omitted by no file glob."""
    root = Path(episode); verify_files(seed['files'])
    requests = read(root/'requests.json')
    requested = {value['label'] for value in requests}
    executed = {path.name.removesuffix('_execution.json') for path in root.glob('*_execution.json')}
    for label in requested | executed:
        for suffix in ('_command.json', '_execution.json', '_settling.json'):
            if not (root/(label+suffix)).is_file():
                raise RuntimeError('Occurred/requested action has missing evidence: '+label+suffix)
        if label in ('pregrasp', 'approach_connect', 'approach', 'lift'):
            cert = read(root/(label+'_optimization.json'))
            # Rejections are ordinary task failures, but an executed path must
            # always have a positive current-scene planning certificate.
            if not cert.get('success') or cert['invalid_samples'] or cert['max_limit_exceed_rad'] > 1e-6:
                raise RuntimeError('Executed path lacks original planning validity')
    if executed-requested:
        raise RuntimeError('Execution receipt exists outside the command request ledger')
    replayed = []
    if mode in ('control', 'stitch'):
        for phase in TAIL:
            if phase not in requested:
                continue
            source = read(Path(seed['episode'])/(phase+'_command.json'))
            actual = read(root/(phase+'_command.json'))
            if command_payload(source) != command_payload(actual):
                raise RuntimeError('Successful tail command changed beyond its execution nonce')
            replayed.append(phase)
        if 'approach' in requested and not (root/'join_state_check.json').is_file():
            raise RuntimeError('Source tail dispatched without a join-state receipt')
        if 'approach' in requested and not read(root/'join_state_check.json')['pass']:
            raise RuntimeError('Source tail dispatched despite a rejected join')
        if 'approach' in requested:
            wait = read(root/'join_state_check.json').get('anchor_wait', {})
            if wait.get('policy') != ANCHOR_WAIT or wait.get('status') != 'ready':
                raise RuntimeError('Source tail dispatched without its registered anchor settling interval')
    return {'pass': True, 'seed_id': seed['seed_id'], 'source_files': seed['files'],
            'mode': mode, 'requested_actions': sorted(requested), 'executed_actions': sorted(executed),
            'replayed_unchanged_tail_phases': replayed,
            'join_receipt_sha256': sha(root/'join_state_check.json') if (root/'join_state_check.json').exists() else None,
            'connector_training_phase': 'approach', 'connector_actual_command_label': 'approach_connect',
            'fresh_physical_execution_and_camera_required': True, 'old_images_or_outcomes_copied': False}
