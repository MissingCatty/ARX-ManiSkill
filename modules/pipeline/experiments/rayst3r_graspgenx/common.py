"""Contracts for the head-camera RaySt3R -> GraspGenX physical benchmark."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parent
MS = WORKSPACE / 'maniskill'
PRIOR = MS / 'outputs/arx_graspcl_compat160_20261120_01/fixed_zerograsp'
DEFAULT = HERE / 'runs/fixed452_160_20261122_01'
GPUS = {'sam3': 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169',
        'rayst3r': 'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b',
        'graspgenx': 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169',
        'curobo': 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'}
PROTOCOL = 'head_sam3_rayst3r_graspgenx_tcp43_fixed452_lift30_hold10_v1'


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    from arx_project.layout import read_json
    return read_json(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024): h.update(block)
    return h.hexdigest()


def paths():
    sys.path[:0] = [str(MS), str(MS / 'arx_pipeline'), str(HERE),
                   str(PROJECT / 'experiments/pointcloud_benchmark')]


def check_sources(reg):
    for name, expected in reg['source_hashes'].items():
        if sha(name) != expected: raise RuntimeError('Frozen source changed: ' + name)


def make_candidates(directory):
    """Generator origin is EEF; represent its unchanged pose at TCP43 once."""
    paths()
    from graspgenx_partial_geometry import generator_to_contact
    import graspcl_pipeline as display
    directory = Path(directory)
    raw = directory / 'graspgenx_raw'
    poses, scores = np.load(raw / 'grasps_generator_world.npy'), np.load(raw / 'scores.npy')
    tags = read(raw / 'inference.json')['branch_tags']
    assert poses.shape == (len(scores), 4, 4) and len(tags) == len(scores)
    assert np.isfinite(poses).all() and np.isfinite(scores).all()
    candidates = []
    for i in np.argsort(-scores, kind='stable'):
        pose = generator_to_contact(poses[i])
        candidates.append({'pose': pose, 'width': .088, 'score': float(scores[i]),
            'point1': pose[:3, 3] - .044 * pose[:3, 1],
            'point2': pose[:3, 3] + .044 * pose[:3, 1],
            'metadata': {'origin': 'graspgenx', 'source_generator_index': int(i),
                'source_branch': tags[i], 'source_model_score': float(scores[i]),
                'source_generator_pose_world': poses[i], 'storage_tcp_m': .043,
                'pose_refinement_applied': False, 'source_network_depth_m': 0.}})
    dest = directory / 'candidates'
    dest.mkdir()
    (directory / 'graspcl').symlink_to(dest, target_is_directory=True)
    save(dest / 'candidate_pool.json', {'candidates': candidates, 'protocol': PROTOCOL,
        'raw_count': len(poses), 'retained_count': len(candidates),
        'graspcl_invoked': False, 'topdown_filter': False, 'orientation_changes': 0,
        'position_changes': 0, 'native_score_order': True, 'additional_symmetries': False})
    camera = np.asarray(read(directory / 'state.json')['camera_world_optical'])
    rows = np.asarray([display.display_row(c, camera) for c in candidates]).reshape(-1, 17)
    np.save(directory / 'raw_grasps.npy', rows, allow_pickle=False)
    np.save(dest / 'raw_grasps.npy', rows, allow_pickle=False)
    save(dest / 'network_candidates.json', {'model_name': 'RaySt3R + GraspGenX',
        'score_label': 'GraspGenX native score', 'count': len(candidates),
        'compatibility_directory_only': 'graspcl', 'graspcl_invoked': False})
    return len(candidates)


def aggregate(root, phase):
    reg = read(root / 'registration.json')
    rows, claimed = [], []
    for p in sorted((root / 'claims').glob('*.json')):
        claim = read(p)
        if claim['state'] == 'complete':
            trial = root / 'trials' / p.stem
            assert sha(trial / 'result.json') == claim['result_sha256']
            rows.append(read(trial / 'result.json'))
        else: claimed.append(p.stem)
    n = len(reg['schedule'])
    summary = {'phase': phase, 'registered': n, 'completed': len(rows),
        'not_executed': n - len(rows), 'claimed_without_terminal_receipt': claimed,
        'planning_passes': sum(bool(x['planning_success']) for x in rows),
        'physical_attempts': sum(bool(x['physical_attempt']) for x in rows),
        'successes': sum(bool(x['success']) for x in rows),
        'failure_reasons': dict(Counter(x['reason'] for x in rows if not x['success'])),
        'updated_at': now()}
    summary['planning_rate_registered'] = summary['planning_passes'] / n
    summary['success_rate_registered'] = summary['successes'] / n
    save(root / 'progress.json', summary)
    return summary
