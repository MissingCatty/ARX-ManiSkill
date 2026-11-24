"""Model-only exhaustive ZeroGrasp raw-candidate timing on the frozen ARX model."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'arx_pipeline'))
sys.path.insert(0, str(CODE))
import planner as frozen
from graspcl_planner import GraspCLPlanner


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main(out, source):
    out.mkdir(parents=True, exist_ok=False)
    state = json.loads((source / 'state.json').read_text())
    rows = np.load(source / 'raw_grasps.npy', allow_pickle=False)
    rows = rows[np.argsort(-rows[:, 0], kind='stable')]
    camera = np.array(state['camera_world_optical'])
    axes = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    save(out / 'registration.json', {
        'created_at': datetime.now().astimezone().isoformat(), 'source': str(source),
        'source_sha256': {str(source / f): hashlib.sha256((source / f).read_bytes()).hexdigest() for f in ['state.json', 'raw_grasps.npy']},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_candidates': len(rows), 'grasp_ik_query_budget': 2 * len(rows),
        'warmup_ik_queries': 1, 'warmup_prm_queries': 1, 'warmup_trajectory_optimizations': 1,
        'physical_attempt_budget': 0,
        'exhaustive_scope': 'Every raw row, two finger-symmetric orientations; original ARX IK including open-scene collision, then nominal-closed collision. No GT center/height, width, approach or score cutoff in exhaustive phase. Not full trajectories for every pose.',
        'first_full_plan_scope': 'Separately call the existing GraspCL selector with ALL raw rows plus finger symmetry in original ZeroGrasp score order, retaining its downward-approach gate, no demonstration candidates or reranking. This measures until first full accepted plan or exhaustion.',
        'gpu': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        'wall_budget_seconds_exclusive_of_startup': 1800,
    })
    started = time.perf_counter()
    p = GraspCLPlanner(out / 'model_checks', state)
    torch.cuda.synchronize()
    construction = time.perf_counter() - started
    start = p.measured(state)
    warm_start = time.perf_counter()
    warm = p.ik(np.asarray(state['links']['right_eef_link']), start)
    torch.cuda.synchronize()
    if warm is None:
        raise RuntimeError('Measured start cannot pass unchanged IK/collision checks')
    warm_seconds = time.perf_counter() - warm_start
    extra_warm_start = time.perf_counter()
    warm_path = p.path(start, start)
    if warm_path is None:
        raise RuntimeError('Stationary measured-state PRM warmup failed')
    p.optimize('timing_warmup', warm_path, 6.)
    torch.cuda.synchronize()
    extra_warm_seconds = time.perf_counter() - extra_warm_start
    p.planner.reset_seed()
    candidates, measurements = [], []
    counts = {'tested': 0, 'grasp_ik_and_open_collision_valid': 0, 'also_nominal_closed_valid': 0}
    t0 = time.perf_counter()
    with (out / 'candidate_timings.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            r = camera[:3, :3] @ row[4:13].reshape(3, 3)
            contact = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3] * r[:, 0]
            for flip in [False, True]:
                pose = np.eye(4)
                pose[:3, :3] = r @ axes
                if flip:
                    pose[:3, :3] = pose[:3, :3] @ np.diag([-1., -1., 1.])
                pose[:3, 3] = contact
                candidates.append({'pose': pose.tolist(), 'score': float(row[0]), 'width': float(row[1]),
                    'metadata': {'origin': 'zerograsp', 'raw_rank': index + 1, 'finger_flip': flip}})
                grasp = pose.copy()
                grasp[:3, 3] -= .040 * pose[:3, 2]
                call_start = time.perf_counter()
                qg = p.ik(grasp, start)
                torch.cuda.synchronize()
                ik_seconds = time.perf_counter() - call_start
                closed_seconds = 0.
                closed_valid = False
                if qg is not None:
                    call_start = time.perf_counter()
                    closed_valid = bool(p.closed_checker.validate(qg[None, None]).all())
                    torch.cuda.synchronize()
                    closed_seconds = time.perf_counter() - call_start
                record = {'raw_rank': index + 1, 'finger_flip': flip, 'ik_seconds': ik_seconds,
                    'closed_collision_seconds': closed_seconds, 'grasp_ik_and_open_valid': qg is not None,
                    'also_nominal_closed_valid': closed_valid}
                stream.write(json.dumps(record) + '\n')
                measurements.append(record)
                counts['tested'] += 1
                counts['grasp_ik_and_open_collision_valid'] += int(qg is not None)
                counts['also_nominal_closed_valid'] += int(closed_valid)
            if (index + 1) % 100 == 0:
                save(out / 'progress.json', {'phase': 'exhaustive_grasp_ik', 'rows_completed': index + 1,
                    'raw_rows': len(rows), **counts, 'elapsed_seconds': time.perf_counter() - t0})
                print(index + 1, counts, time.perf_counter() - t0, flush=True)
            if time.perf_counter() - t0 > 1800:
                raise RuntimeError('Model-only wall budget exhausted; unexamined rows remain explicit')
    exhaustive_seconds = time.perf_counter() - t0
    (out / 'graspcl').mkdir()
    save(out / 'graspcl/candidate_pool.json', {'candidates': candidates})
    # The selector uses self.out.parent to locate its canonical candidate pool.
    p.out = out / 'full_plan'
    p.out.mkdir()
    p.planner.reset_seed()
    full_start = time.perf_counter()
    try:
        result = p.select(state, rows, camera)
    except Exception:
        result = {'success': False, 'reason': 'planner_error', 'error': traceback.format_exc()}
    frozen.flush_planning_artifacts()
    full_seconds = time.perf_counter() - full_start
    save(out / 'first_full_plan_result.json', result)
    ik_times = np.array([m['ik_seconds'] for m in measurements])
    save(out / 'summary.json', {
        'raw_candidates': len(rows), 'symmetric_grasp_poses_tested': len(measurements),
        'model_construction_seconds': construction, 'first_measured_pose_ik_warmup_seconds': warm_seconds,
        'prm_and_trajectory_warmup_seconds': extra_warm_seconds,
        'exhaustive_ik_and_collision_wall_seconds': exhaustive_seconds,
        'summed_ik_call_seconds': float(ik_times.sum()), 'median_ik_call_ms': float(np.median(ik_times) * 1000),
        'mean_ik_call_ms': float(ik_times.mean() * 1000),
        'summed_nominal_closed_collision_seconds': sum(m['closed_collision_seconds'] for m in measurements),
        'counts': counts, 'first_full_plan_seconds_warm': full_seconds,
        'first_full_plan_success': result['success'],
        'first_full_plan_selected': result.get('selected', {}).get('candidate_metadata'),
        'first_full_plan_rejected': result.get('rejected'),
        'physical_attempts': 0, 'all_raw_rows_tested': counts['tested'] == 2 * len(rows),
    })
    save(out / 'progress.json', {'phase': 'complete', 'raw_rows': len(rows), **counts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve(), args.source.resolve())
