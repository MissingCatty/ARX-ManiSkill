"""Paired translation of saved grasps: table-edge distance versus planning."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MS = PROJECT.parent / 'maniskill'
SOURCE = PROJECT / 'experiments/repaired_grasp_benchmark/runs/geometry200_repaired_20261123_01'
ROOT = HERE / 'runs/three_objects_20261123_01'
GPU = 'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
SLOTS = [('D0007_S1_open_banana', 'banana', '香蕉'),
         ('D0018_S1_open_boxed_food', 'boxed_food', '食品盒'),
         ('D0035_S1_open_donut', 'donut', '甜甜圈')]
GAPS = [.45, .35, .25, .15, .08]
DOC = PROJECT / 'docs/diverse_grasp_benchmark/debug_20261123'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    def encode(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    tmp = p.with_suffix(p.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode) + '\n')
    tmp.replace(p)


def guard(reg):
    for p, digest in {**reg['source_sha256'], **reg['input_sha256']}.items():
        assert sha(p) == digest, 'Changed registered file: ' + p


def prepare():
    assert not ROOT.exists()
    ROOT.mkdir(parents=True)
    sys.path[:0] = [str(PROJECT / 'experiments/diverse_object_scenes')]
    from workspace_geometry import world_corners, TABLE_MIN, TABLE_MAX
    catalog = load_object_catalog(SOURCE / 'object_catalog.json')['models']
    inputs = {str(SOURCE / 'object_catalog.json'): sha(SOURCE / 'object_catalog.json')}
    schedule = []
    for slot, slug, name in SLOTS:
        trial = SOURCE / 'trials' / slot
        entry, state = read(trial / 'entry.json'), read(trial / 'state.json')
        receipt = read(SOURCE / 'claims' / (slot + '.json'))
        assert receipt['state'] == 'complete' and receipt['result_sha256'] == sha(trial / 'result.json')
        assert not read(trial / 'plan_result.json')['success']
        assert len(entry['scene']['actors']) == 1
        actor = deepcopy(entry['scene']['actors'][0])
        actor['pose'] = deepcopy(state['objects']['target']['pose'])
        original_gap = float(world_corners(actor, catalog)[:, 0].min() - TABLE_MIN[0])
        assert original_gap > max(GAPS)
        pool = read(trial / 'candidates/candidate_pool.json')
        for n in ['state.json', 'entry.json', 'result.json', 'plan_result.json',
                  'candidates/candidate_pool.json', 'plan/arx.yml', 'raw_grasps.npy']:
            inputs[str(trial / n)] = sha(trial / n)
        for level, gap in enumerate([original_gap] + GAPS):
            case = f'{slug}_d{level}'
            out = ROOT / 'cases' / case
            out.mkdir(parents=True)
            delta = np.array([gap - original_gap, 0., 0.])
            shifted = deepcopy(state)
            shifted['object_world'] = np.asarray(shifted['object_world'])
            shifted['object_world'][:3, 3] += delta
            shifted['object_center'] = np.asarray(shifted['object_center']) + delta
            shifted['objects']['target']['pose'][:3] = (np.asarray(actor['pose'][:3]) + delta).tolist()
            shifted['scene_meshes']['target']['world'] = np.asarray(shifted['scene_meshes']['target']['world'])
            shifted['scene_meshes']['target']['world'][:3, 3] += delta
            new_entry = deepcopy(entry)
            new_entry['id'] = case
            new_entry['scene']['actors'][0]['pose'] = deepcopy(shifted['objects']['target']['pose'])
            new_entry['controlled_translation_m'] = delta
            new_entry['source_trial'] = str(trial)
            new_entry['experiment_arm'] = 'paired_edge_distance_planning_only'
            new_pool = deepcopy(pool)
            for old, item in zip(pool['candidates'], new_pool['candidates']):
                for field in ['pose', 'point1', 'point2']:
                    value = np.asarray(item[field])
                    if field == 'pose': value[:3, 3] += delta
                    else: value += delta
                    item[field] = value
                meta = item['metadata']
                for field in ['source_generator_pose_world', 'source_eef_pose_world']:
                    if field in meta:
                        value = np.asarray(meta[field]); value[:3, 3] += delta; meta[field] = value
                meta['diagnostic_translation_m'] = delta
                relative_old = np.linalg.inv(np.asarray(state['object_world'])) @ np.asarray(old['pose'])
                relative_new = np.linalg.inv(np.asarray(shifted['object_world'])) @ np.asarray(item['pose'])
                assert np.max(np.abs(relative_new - relative_old)) < 1e-12
            new_pool['experiment'] = 'same grasp identities, scores and object-relative poses; translation only'
            save(out / 'state.json', shifted)
            save(out / 'entry.json', new_entry)
            save(out / 'candidates/candidate_pool.json', new_pool)
            (out / 'graspcl').symlink_to(out / 'candidates', target_is_directory=True)
            # The native selector reads the JSON poses; rows carry the unchanged
            # legacy input shape and score order, then are regenerated below.
            sys.path[:0] = [str(MS), str(MS / 'arx_pipeline')]
            import graspcl_pipeline as display
            rows = np.asarray([display.display_row(c, np.asarray(shifted['camera_world_optical']))
                               for c in new_pool['candidates']])
            np.save(out / 'raw_grasps.npy', rows, allow_pickle=False)
            bounds = world_corners(new_entry['scene']['actors'][0], catalog)
            assert (bounds[:, :2].min(0) >= TABLE_MIN + .012).all()
            assert (bounds[:, :2].max(0) <= TABLE_MAX - .012).all()
            assert abs(bounds[:, 0].min() - TABLE_MIN[0] - gap) < 1e-12
            schedule.append({'id': case, 'source_id': slot, 'name': name, 'slug': slug,
                'level': level, 'edge_gap_m': gap, 'translation_m': delta,
                'candidates': len(new_pool['candidates']), 'object_pose': shifted['objects']['target']['pose']})
    for row in schedule:
        out = ROOT / 'cases' / row['id']
        for n in ['state.json', 'entry.json', 'candidates/candidate_pool.json', 'raw_grasps.npy']:
            inputs[str(out / n)] = sha(out / n)
    deps = [Path(__file__), MS / 'short_lift_planner.py', MS / 'resident_planner.py',
            MS / 'fast_resident_planner.py', MS / 'planning_rejection_audit.py', MS / 'ik_workspace_audit.py',
            PROJECT / 'experiments/repaired_grasp_benchmark/repair_contract.py',
            PROJECT / 'experiments/rayst3r_graspgenx/services.py']
    parent_sources = read(SOURCE / 'registration.json')['source_hashes']
    for p, digest in parent_sources.items(): assert sha(p) == digest, p
    save(ROOT / 'registration.json', {'created_at': datetime.now().astimezone().isoformat(),
        'source_run': str(SOURCE), 'gpu': GPU, 'schedule': schedule, 'registered': len(schedule),
        'source_sha256': {**parent_sources, **{str(p): sha(p) for p in deps}}, 'input_sha256': inputs,
        'planner_warmup_trial': read(SOURCE / 'registration.json')['planner_warmup_trial'],
        'method': 'paired rigid translation of saved native GraspGenX candidates with each object',
        'independent_variable': 'oriented object bounding box nearest-X face to robot-facing table edge',
        'table_near_edge_world_x_m': float(TABLE_MIN[0]), 'physical_attempts': 0,
        'fresh_neural_inference': False, 'candidate_cap': None, 'automatic_retries': 0,
        'pure_ik_seeds': 128, 'pure_ik_returned_solutions': 32,
        'position_tolerance_m': .002, 'orientation_tolerance_rad': .03,
        'robot_dof': 6, 'joint4_m': .452, 'actual_tcp_m': .043,
        'full_path_test': 'existing cuRobo selector, unchanged reach/approach and lift-IK gates',
        'original_dataset_modified': False})
    debug = ROOT / 'debug'
    debug.mkdir()
    (debug / '假设.md').write_text('# 距离对照假设\n\nH1：原摆放太远使大部分抓姿无法求出IK；只向近侧移动会增加无碰撞可行抓姿。\n\nH2：抓姿方向及桌面碰撞是主要限制；靠近后仍可能全部失败，或出现中间距离更好的情况。\n\n18个位置提前登记，全部候选保留；同一物体的相对抓姿和分数严格不变。\n')
    (debug / '已排除.md').write_text('# 已排除的解释\n\n待实验结果。\n')
    (debug / '定位记录.md').write_text('# 症状\n\n香蕉、食品盒、甜甜圈原位置的全部候选在抓取终点IK/碰撞门控被拦住。\n\n本轮分别比较原位置、45/35/25/15/8厘米桌沿间距。\n')
    save(ROOT / 'progress.json', {'phase': 'registered', 'registered': 18, 'completed': 0})
    print('REGISTERED', len(schedule), [(x['slug'], round(x['edge_gap_m'], 4)) for x in schedule], flush=True)


def capture():
    reg = read(ROOT / 'registration.json'); guard(reg)
    assert not (ROOT / 'capture_complete.json').exists()
    sys.path[:0] = [str(PROJECT / 'experiments/diverse_object_scenes'),
                   str(PROJECT / 'experiments/repaired_grasp_benchmark')]
    from pipeline import runtime
    from repair_contract import simulation_config
    catalog = load_object_catalog(SOURCE / 'object_catalog.json')['models']
    environment, baseline, run, sapien, torch, make, *_ = runtime(catalog)
    original = environment.ARXPickYCB._default_sim_config
    environment.ARXPickYCB._default_sim_config = property(lambda self: simulation_config(original.fget(self)))
    receipts = []
    for slot, slug, _ in SLOTS:
        rows = [r for r in reg['schedule'] if r['slug'] == slug]
        first = read(ROOT / 'cases' / rows[0]['id'] / 'entry.json')
        env = make(first['model_id'], first['scene'])
        try:
            env.reset(seed=first['seed'])
            baseline.settle(env)
            for row in rows:
                out = ROOT / 'cases' / row['id']
                expected = read(out / 'state.json')
                q = torch.tensor([expected['qpos']], device=env.device)
                env.agent.robot.set_qpos(q)
                env.agent.robot.set_qvel(torch.zeros_like(q))
                pose = row['object_pose']
                env.obj.set_pose(sapien.Pose(pose[:3], pose[3:]))
                env.obj._bodies[0].linear_velocity = np.zeros(3)
                env.obj._bodies[0].angular_velocity = np.zeros(3)
                obs = env.get_obs()
                actual = run.snapshot(env, obs)
                assert np.max(np.abs(np.asarray(actual['object_world']) - expected['object_world'])) < 2e-6
                assert np.max(np.abs(np.asarray(actual['qpos']) - expected['qpos'])) < 2e-6
                for key in actual['links']:
                    assert np.max(np.abs(np.asarray(actual['links'][key]) - expected['links'][key])) < 2e-5
                save(out / 'capture_state.json', actual)
                data = obs['sensor_data']['head']
                from PIL import Image
                Image.fromarray(data['rgb'][0].cpu().numpy()).save(out / 'head.png')
                Image.fromarray(env.render()[0].cpu().numpy()).save(out / 'overview.png')
                mask = data['segmentation'][0, :, :, 0].cpu().numpy() == env.obj._objs[0].per_scene_id
                Image.fromarray((mask * 255).astype(np.uint8)).save(out / 'target_mask.png')
                result = {'id': row['id'], 'passed': True, 'target_visible_pixels': int(mask.sum()),
                    'contact_offset_m': env.sim_config.scene_config.contact_offset,
                    'physical_attempts': 0, 'robot_motion_commands': 0,
                    'object_pose_error_m': float(np.linalg.norm(np.asarray(actual['object_world'])[:3, 3] - np.asarray(expected['object_world'])[:3, 3])),
                    'render_only_at_exact_registered_pose': True}
                save(out / 'capture.json', result); receipts.append(result)
                print('CAPTURE', row['id'], result['target_visible_pixels'], flush=True)
        finally:
            env.close()
    guard(reg)
    save(ROOT / 'capture_complete.json', {'passed': True, 'cases': receipts, 'physical_attempts': 0})


def plan():
    reg = read(ROOT / 'registration.json'); guard(reg)
    assert read(ROOT / 'capture_complete.json')['passed']
    assert not (ROOT / 'planning_started.json').exists()
    save(ROOT / 'planning_started.json', {'pid': os.getpid(), 'gpu': os.getenv('CUDA_VISIBLE_DEVICES')})
    sys.path[:0] = [str(PROJECT / 'experiments/rayst3r_graspgenx'), str(MS), str(MS / 'arx_pipeline')]
    from services import planner
    from planning_rejection_audit import make_pure
    from ik_workspace_audit import solve, collision_details
    from scene_tools import ArmFK
    from scipy.spatial.transform import Rotation
    import torch
    args = SimpleNamespace(root=ROOT, warmup_trial=Path(reg['planner_warmup_trial']))
    pool = planner(args)
    first = ROOT / 'cases' / reg['schedule'][0]['id']
    context = pool.bind(ROOT / 'endpoint_warmup', read(first / 'state.json'))
    pure = make_pure(context.model)
    fk = ArmFK()
    for row in reg['schedule']:
        directory = ROOT / 'cases' / row['id']
        assert not (directory / 'result.json').exists()
        begin = time.perf_counter(); state = read(directory / 'state.json')
        context = pool.bind(directory / 'endpoint', state)
        start = context.measured(state)
        candidates = read(directory / 'candidates/candidate_pool.json')['candidates']
        goals = []
        for item in candidates:
            t = np.array(item['pose']); t[:3, 3] -= .043 * t[:3, 2]; goals.append(t)
        pure.reset_seed()
        pq, pok, _, _ = solve(pure, context, [np.asarray(state['links']['right_eef_link'])], start)
        assert bool(pok.any()), 'Measured-pose IK warmup failed'
        pure.reset_seed()
        q, ok, pe, re = solve(pure, context, goals, start)
        ok &= (pe <= .002001) & (re <= .030001)
        ok &= ((q >= context.limits[:, 0] - 1e-6) & (q <= context.limits[:, 1] + 1e-6)).all(-1)
        clear = torch.zeros_like(ok); table = torch.zeros_like(ok); ground = torch.zeros_like(ok)
        locations = ok.nonzero(); solutions = q[ok]
        witnesses = []
        for offset in range(0, len(solutions), 1024):
            part = solutions[offset:offset + 1024]
            detail = collision_details(context, state, part)
            loc = locations[offset:offset + len(part)]
            clear[loc[:, 0], loc[:, 1]] = detail['full_scene_valid']
            table[loc[:, 0], loc[:, 1]] = detail['table_penetration'] > 0
            ground[loc[:, 0], loc[:, 1]] = detail['ground_penetration'] > 0
            good = detail['full_scene_valid'].nonzero().flatten()
            if len(good) and not witnesses:
                k = int(good[0]); index = int(loc[k, 0]); joints = part[k].cpu().numpy()
                actual = fk.fk(joints); target = context.invbase @ goals[index]
                err = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                rot = float(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude())
                assert err < .00201 and rot < .03001
                witnesses.append({'candidate': index, 'q': joints, 'cpu_position_error_m': err, 'cpu_rotation_error_rad': rot})
        np.savez_compressed(directory / 'endpoint_ik.npz',
            solved=ok.cpu().numpy(), scene_clear=clear.cpu().numpy(),
            table_collision=table.cpu().numpy(), ground_collision=ground.cpu().numpy())
        endpoint = {'candidates': len(goals), 'pure_ik_candidates': int(ok.any(-1).sum()),
            'scene_clear_candidates': int(clear.any(-1).sum()),
            'table_colliding_candidates': int(table.any(-1).sum()), 'witnesses': witnesses,
            'wall_seconds': time.perf_counter() - begin}
        save(directory / 'endpoint_result.json', endpoint)
        # Run the identical production selector and PRM/trajectory optimization.
        # It stops at the first complete feasible native candidate as usual.
        full = pool.request(directory)
        result = {**row, 'endpoint': endpoint, 'planning_success': bool(full['success']),
            'selected_candidate': full.get('selected', {}).get('index'),
            'rejected': full.get('rejected', {}), 'physical_attempts': 0,
            'wall_seconds': time.perf_counter() - begin}
        save(directory / 'result.json', result)
        completed = [read(p) for p in (ROOT / 'cases').glob('*/result.json')]
        save(ROOT / 'progress.json', {'phase': 'planning', 'completed': len(completed),
            'registered': len(reg['schedule']), 'planning_successes': sum(r['planning_success'] for r in completed), 'last': result})
        print('RESULT', row['id'], endpoint['pure_ik_candidates'], endpoint['scene_clear_candidates'],
              full['success'], full.get('rejected'), flush=True)
        guard(reg)
    rows = [read(ROOT / 'cases' / r['id'] / 'result.json') for r in reg['schedule']]
    save(ROOT / 'summary.json', {'completed': len(rows), 'registered': len(rows), 'cases': rows,
        'planning_successes': sum(r['planning_success'] for r in rows),
        'physical_attempts': 0, 'source_inputs_unchanged': True, 'model_loads': pool.loads})
    save(ROOT / 'progress.json', {'phase': 'complete', 'completed': len(rows), 'registered': len(rows),
        'planning_successes': sum(r['planning_success'] for r in rows), 'physical_attempts': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'capture', 'plan'])
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except BaseException:
        import traceback
        if ROOT.exists():
            save(ROOT / (args.stage + '_error.json'), {'error': traceback.format_exc(), 'pid': os.getpid()})
        raise
