"""Compare broader map seeds on frozen original endpoints, without motion."""
import argparse
from pathlib import Path
import shutil
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.assets import resource_path
from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.indexed_planning.index_reuse.lookup import FullPoseIndex
from .replay import verify_hashes

MODES = {'control': None, 'map3': 3., 'map5': 5., 'map10': 10.}


def prepare(root, diagnosis, gpu):
    if root.exists():
        raise ValueError('Keep previous registrations; use a fresh seed probe')
    accepted = read_json(diagnosis/'acceptance.json')
    if not accepted['passed'] or not accepted['task_complete']:
        raise ValueError('The map diagnosis must finish first')
    original = read_json(diagnosis/'registration.json')
    fixtures = []; inputs = {}
    for case in original['cases']:
        directory = Path(case['source_directory'])
        state = read_json(directory/'state.json')
        candidates = read_json(directory/'candidates/candidate_pool.json')['candidates']
        inputs.update({str(p): sha256(p) for p in [directory/'state.json',
            directory/'candidates/candidate_pool.json']})
        body = np.asarray(state['links']['body4'])
        for side in ['right', 'left']:
            indices = case['checked_candidate_indices'][side]
            # Exactly the same first/last fixtures chosen before the hit-rate
            # measurement. Do not select fixtures because a map query hits.
            chosen = sorted(set([indices[0], indices[-1]])) if indices else []
            canonical = body.copy()
            if side == 'left':
                shift = np.eye(4); shift[1, 3] = .5
                canonical = body@shift
            names = get_arm(side).joints
            current = np.array([state['qpos'][state['joint_names'].index(n)] for n in names])
            bounds = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in names])
            current = np.maximum(np.minimum(current, bounds[:, 1]-1e-4), bounds[:, 0]+1e-4).astype(np.float32)
            for index in chosen:
                eef = np.asarray(candidates[index]['pose']).copy()
                eef[:3, 3] -= .043*eef[:3, 2]
                fixtures.append({'id': case['id'], 'arm': side, 'candidate_index': index,
                    'eef_body': (np.linalg.inv(canonical)@eef).tolist(), 'current_q': current.tolist()})
    if len(fixtures) != accepted['seed_fixture_queries']:
        raise ValueError('The independently chosen fixture count changed')
    model = Path(original['source_run'])/'solver_prewarm/binding/arx.yml'
    sources = [Path(__file__), ROOT/'modules/curobo/indexed_planning/index_reuse/lookup.py',
        ROOT/'modules/curobo/indexed_planning/geometric/kinematics.py',
        ROOT/'modules/curobo/dual_arm/model.py', resource_path('arx_robot')/'robot.urdf',
        ROOT/'configs/policies/fingertip_table.json']
    sources += [ROOT/'modules/curobo/source/curobo/_src/solver'/name for name in
        ['solver_ik.py', 'solver_ik_cfg.py', 'seed_ik/seed_ik_solver.py']]
    inputs[str(model)] = sha256(model)
    inputs[str(diagnosis/'acceptance.json')] = sha256(diagnosis/'acceptance.json')
    inputs.update({k: v for k, v in original['input_hashes'].items() if Path(k).parent == Path(original['map_directory'])})
    root.mkdir(parents=True)
    for path in sources:
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write_json(root/'registration.json', {'at': now(), 'experiment': 'index_seed_probe',
        'authorization': 'V8继续；比较3/5/10度地图初值，保持原目标和最终精度。',
        'diagnosis_run': str(diagnosis), 'map_directory': original['map_directory'], 'model': str(model),
        'gpu': gpu, 'minimum_startup_free_mib': 10000, 'minimum_request_free_mib': 1536,
        'fixtures': fixtures, 'cases': original['cases'], 'variants': MODES, 'repeats': 3,
        'IK_calls': len(fixtures)*len(MODES)*3, 'planning_requests': 0,
        'physical_grasps': 0, 'simulation_instances': 0, 'perception_calls': 0,
        'seed_position_radius_m': .02, 'maximum_map_seeds': 8, 'total_LM_seeds': 32,
        'returned_IK_branches': 8, 'final_position_tolerance_m': .002,
        'final_rotation_tolerance_rad': .03, 'random_seed': 20261126,
        'automatic_retries': False, 'source_hashes': {str(p): sha256(p) for p in sources},
        'input_hashes': inputs})
    write_json(root/'progress.json', {'at': now(), 'phase': 'registered',
        'completed_IK_calls': 0, 'total_IK_calls': len(fixtures)*len(MODES)*3,
        'completed_requests': 0, 'total_requests': 0})


def run(root):
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('Never repeat an already started probe')
    reg = read_json(root/'registration.json'); verify_hashes(reg)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu']
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, JointState
    from modules.curobo.fingertip_contact.profile import load_profile
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]/1024**2 < reg['minimum_startup_free_mib']:
        raise RuntimeError('The registered device lacks the original startup margin')
    torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    write_json(root/'started.json', {'at': now(), 'GPU': reg['gpu']})
    index = FullPoseIndex(Path(reg['map_directory']))
    profile = load_profile()
    cfg = InverseKinematicsCfg.create(robot=reg['model'], scene_model=None,
        optimizer_configs=['ik/lbfgs_ik.yml'], self_collision_check=False,
        num_seeds=profile.ik_fallback_seeds, seed_solver_num_seeds=profile.ik_fallback_seeds,
        position_tolerance=profile.position_tolerance_m, orientation_tolerance=profile.rotation_tolerance_rad,
        use_cuda_graph=True, max_batch_size=1, load_collision_spheres=False)
    for rollout in [cfg.core_cfg.metrics_rollout_config, *cfg.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs(): cost.disable_self_collision()
    solver = InverseKinematics(cfg)
    names = list(index.chain.arm.joints)

    def solve(eef, current, seeds):
        quat = Rotation.from_matrix(eef[:3, :3]).as_quat()[[3, 0, 1, 2]]
        goal = GoalToolPose(tool_frames=['right_eef_link'],
            position=torch.tensor(eef[:3, 3], device='cuda', dtype=torch.float32).reshape(1, 1, 1, 1, 3),
            quaternion=torch.tensor(quat, device='cuda', dtype=torch.float32).reshape(1, 1, 1, 1, 4))
        state = JointState.from_position(torch.tensor(current, device='cuda', dtype=torch.float32)[None], joint_names=names)
        seed = torch.tensor(seeds, device='cuda', dtype=torch.float32)[None]
        traces = []; original = solver.seed_ik_solver._generate_seed_configs

        def traced(batch_size, seed_config=None):
            value = original(batch_size, seed_config)
            traces.append((value.detach().clone(), seed_config.detach().clone()))
            return value

        solver.reset_seed()
        with patch.object(solver.seed_ik_solver, '_generate_seed_configs', side_effect=traced):
            result = solver.solve_pose(goal, current_state=state, seed_config=seed, return_seeds=8)
        torch.cuda.synchronize()
        if len(traces) != 1 or traces[0][0].shape != (1, 32, 6):
            raise RuntimeError('The original LM seed budget was changed or the hook was bypassed')
        generated, received = traces[0]
        if not torch.equal(received, seed) or not torch.equal(generated[:, :seed.shape[1]], seed):
            raise RuntimeError('Map initial configurations did not actually enter the native LM solver')
        q = result.solution.reshape(-1, 6).detach().cpu().numpy()
        valid = result.success.reshape(-1).detach().cpu().numpy().astype(bool)
        return q[valid], generated.cpu().numpy().tolist()

    warmup = time.perf_counter()
    current = np.asarray(reg['fixtures'][0]['current_q'])
    measured = index.chain.fk(current)[0]
    positive, _ = solve(measured, current, [current.tolist()])
    far = measured.copy(); far[0, 3] += 10
    negative, _ = solve(far, current, [current.tolist()])
    if not len(positive) or len(negative): raise RuntimeError('Positive/negative native IK warmup failed')
    write_json(root/'startup.json', {'at': now(), 'passed': True,
        'model_warmup_seconds': time.perf_counter()-warmup, 'map_load_seconds': index.load_seconds,
        'positive_branches': len(positive), 'negative_branches': len(negative)})
    all_rows = []
    for ordinal, fixture in enumerate(reg['fixtures']):
        if torch.cuda.mem_get_info()[0]/1024**2 < reg['minimum_request_free_mib']:
            raise RuntimeError('The original per-request device margin is unavailable')
        eef, current = np.asarray(fixture['eef_body']), np.asarray(fixture['current_q'])
        rows = []
        for repeat in range(reg['repeats']):
            order = list(MODES)
            offset = (ordinal+repeat) % len(order); order = order[offset:]+order[:offset]
            if repeat % 2: order.reverse()
            for mode in order:
                torch.cuda.synchronize(); began = time.perf_counter()
                nearby = None if mode == 'control' else index.neighborhood(eef, current,
                    radius=reg['seed_position_radius_m'], angle=np.deg2rad(MODES[mode]), count=9)
                seeds = [current.tolist()] if nearby is None else nearby['seeds']
                query_seconds = 0. if nearby is None else nearby['seconds']
                ik_start = time.perf_counter()
                q, generated = solve(eef, current, seeds)
                elapsed = time.perf_counter()-began
                ik_seconds = time.perf_counter()-ik_start
                actual = index.chain.fk(q) if len(q) else np.empty((0, 4, 4))
                pe = np.linalg.norm(actual[:, :3, 3]-eef[:3, 3], axis=1)
                re = Rotation.from_matrix(actual[:, :3, :3]@eef[:3, :3].T).magnitude() if len(q) else np.empty(0)
                if (not np.isfinite(q).all() or (q < index.chain.limits[:, 0]).any()
                        or (q > index.chain.limits[:, 1]).any() or (pe > .002).any() or (re > .03).any()):
                    raise RuntimeError('An accepted IK branch failed independent physical FK or joint bounds')
                rows.append({'repeat': repeat, 'mode': mode, 'wall_seconds': elapsed,
                    'query_seconds': query_seconds, 'IK_and_transfer_seconds': ik_seconds,
                    'success': bool(len(q)), 'accepted_branches': len(q), 'solutions': q.tolist(),
                    'position_errors_m': pe.tolist(), 'rotation_errors_rad': re.tolist(),
                    'supplied_seed_count': len(seeds), 'map_rows': None if nearby is None else nearby['bank_rows'],
                    'actual_LM_initial_seeds': generated, 'target_pose_changed': False})
        value = {**fixture, 'runs': rows}
        write_json(root/'fixtures'/f'{ordinal:03d}.json', value); all_rows.append(value)
        write_json(root/'progress.json', {'at': now(), 'phase': 'probing',
            'completed_IK_calls': len(all_rows)*len(MODES)*reg['repeats'], 'total_IK_calls': reg['IK_calls'],
            'completed_fixtures': len(all_rows), 'total_fixtures': len(reg['fixtures']),
            'completed_requests': 0, 'total_requests': 0, 'current': fixture['id']})
        print('SEED_PROBE', ordinal+1, fixture['id'], fixture['arm'], flush=True)
    verify_hashes(reg)
    summaries = {}
    for mode in MODES:
        values = [r for row in all_rows for r in row['runs'] if r['mode'] == mode]
        seconds = [r['wall_seconds'] for r in values]
        summaries[mode] = {'IK_calls': len(values), 'successes': sum(r['success'] for r in values),
            'map_seed_calls': sum(r['supplied_seed_count'] > 1 for r in values),
            'median_ms': float(np.median(seconds)*1000), 'p90_ms': float(np.percentile(seconds, 90)*1000),
            'total_seconds': float(np.sum(seconds)),
            'total_query_seconds': sum(r['query_seconds'] for r in values)}
    retained = {mode: all(not base['success'] or other['success'] for row in all_rows
        for base in row['runs'] if base['mode'] == 'control'
        for other in row['runs'] if other['mode'] == mode and base['repeat'] == other['repeat'])
        for mode in MODES if mode != 'control'}
    write_json(root/'acceptance.json', {'at': now(), 'passed': True, 'task_complete': True,
        'record_integrity': True, 'fixtures': len(all_rows), 'IK_calls': reg['IK_calls'],
        'summaries': summaries, 'control_successes_retained': retained,
        'all_returned_branches_independently_verified': True,
        'map_seeds_actually_received_by_native_LM': True, 'total_seeds_per_target': 32,
        'planning_requests': 0, 'physical_grasps': 0, 'scene_collision_checked': False,
        'target_pose_changed': False})
    write_json(root/'progress.json', {'at': now(), 'phase': 'complete',
        'completed_IK_calls': reg['IK_calls'], 'total_IK_calls': reg['IK_calls'],
        'completed_requests': 0, 'total_requests': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--diagnosis', type=Path)
    parser.add_argument('--gpu')
    args = parser.parse_args(); root = ensure_run_path(args.root)
    if args.stage == 'prepare': prepare(root, ensure_run_path(args.diagnosis), args.gpu)
    else:
        try: run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc()}); raise
