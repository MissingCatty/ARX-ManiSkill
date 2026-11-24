"""Decompose every failed grasp endpoint, then sweep scene distance."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MS = PROJECT.parent / 'maniskill'
SOURCE = PROJECT / 'experiments/repaired_grasp_benchmark/runs/geometry200_repaired_20261123_01'
ROOT = HERE / 'runs/all53_20261123_01'
FRACTIONS = [0., .2, .4, .6, .8, 1.]
MIN_EDGE_GAP_M = .012
MAX_TRANSLATION_M = .45


def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    tmp = p.with_suffix(p.suffix + '.pending')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, default=encode) + '\n'); tmp.replace(p)


def guard(reg):
    if (ROOT / 'STOP').exists(): raise RuntimeError('STOP present')
    for p, h in {**reg['source_sha256'], **reg['input_sha256']}.items():
        assert sha(p) == h, 'Registered input/source changed: ' + p


def prepare():
    assert not ROOT.exists()
    final = read(SOURCE / 'final_summary.json')
    assert read(SOURCE / 'completion_verification.json')['passed']
    assert final['phase'] == 'complete' and final['completed'] == 200
    catalog_path = SOURCE / 'object_catalog.json'
    catalog = read(catalog_path)['models']
    sys.path.insert(0, str(PROJECT / 'experiments/diverse_object_scenes'))
    from workspace_geometry import world_corners, TABLE_MIN, TABLE_MAX
    parent = read(SOURCE / 'registration.json')
    for p, h in parent['source_hashes'].items(): assert sha(p) == h, p
    inputs = {str(catalog_path): sha(catalog_path), str(SOURCE / 'final_summary.json'): sha(SOURCE / 'final_summary.json')}
    sources, schedule = [], []
    ROOT.mkdir(parents=True)
    for entry in parent['schedule']:
        directory = SOURCE / 'trials' / entry['id']
        plan_file = directory / 'plan_result.json'
        if not plan_file.exists(): continue
        original_plan = read(plan_file)
        if original_plan['success']: continue
        claim = read(SOURCE / 'claims' / (entry['id'] + '.json'))
        assert claim['state'] == 'complete' and claim['result_sha256'] == sha(directory / 'result.json')
        assert original_plan['count'] == original_plan['rejected']['grasp_ik']
        state = read(directory / 'state.json')
        actors = [{'name': n, 'model_id': v['model_id'], 'pose': v['pose']} for n, v in state['objects'].items()]
        bounds = {s['name']: world_corners(s, catalog) for s in actors}
        target_gap = float(bounds['target'][:, 0].min() - TABLE_MIN[0])
        nearest_gap = float(min(b[:, 0].min() for b in bounds.values()) - TABLE_MIN[0])
        distance = max(0., min(MAX_TRANSLATION_M, nearest_gap - MIN_EDGE_GAP_M))
        sources.append({'id': entry['id'], 'source_trial': str(directory), 'model_id': entry['model_id'],
            'semantic_class': entry['semantic_class'], 'layout': entry['layout'],
            'candidates': original_plan['count'], 'target_edge_gap_m': target_gap,
            'nearest_actor_edge_gap_m': nearest_gap, 'maximum_translation_m': distance,
            'translation_limited_by': min(bounds, key=lambda n: bounds[n][:, 0].min()),
            'source_rejected': original_plan['rejected']})
        for n in ['state.json', 'entry.json', 'result.json', 'plan_result.json',
                  'candidates/candidate_pool.json', 'plan/arx.yml']:
            inputs[str(directory / n)] = sha(directory / n)
        for g in state['scene_meshes'].values():
            inputs[g['file_path']] = g['file_sha256']
        for level, fraction in enumerate(FRACTIONS):
            slot = entry['id'] + f'_d{level}'
            out = ROOT / 'cases' / slot
            delta = np.array([-distance * fraction, 0., 0.])
            changed = deepcopy(state)
            changed['object_world'] = np.asarray(changed['object_world']); changed['object_world'][:3, 3] += delta
            changed['object_center'] = np.asarray(changed['object_center']) + delta
            for actor in changed['objects'].values(): actor['pose'][:3] = (np.asarray(actor['pose'][:3]) + delta).tolist()
            for g in changed['scene_meshes'].values():
                g['world'] = np.asarray(g['world']); g['world'][:3, 3] += delta
            new_entry = deepcopy(entry)
            new_entry['id'] = slot
            new_entry['scene']['actors'] = [{'name': n, 'model_id': g['model_id'], 'pose': g['pose']}
                                            for n, g in changed['objects'].items()]
            new_entry['source_trial'] = str(directory)
            for actor in new_entry['scene']['actors']:
                b = world_corners(actor, catalog)
                # Baseline rounding can differ by micrometres; retain baseline,
                # and every translated scene has at least 12 mm support margin.
                assert (b[:, :2].min(0) >= TABLE_MIN + .0119).all(), (slot, actor['name'])
                assert (b[:, :2].max(0) <= TABLE_MAX - .0119).all(), (slot, actor['name'])
            save(out / 'state.json', changed); save(out / 'entry.json', new_entry)
            row = {'id': slot, 'source_id': entry['id'], 'source_trial': str(directory),
                'level': level, 'fraction_of_legal_translation': fraction, 'translation_m': delta,
                'target_edge_gap_m': target_gap + float(delta[0]),
                'group_nearest_edge_gap_m': nearest_gap + float(delta[0]),
                'candidates': original_plan['count'], 'layout': entry['layout'],
                'semantic_class': entry['semantic_class'], 'zero_translation': distance < 1e-6}
            save(out / 'translation.json', row); schedule.append(row)
            for n in ['state.json', 'entry.json', 'translation.json']: inputs[str(out / n)] = sha(out / n)
    assert len(sources) == final['planning_rejections'] == 53
    assert len(schedule) == 318
    reg = {'created_at': now(), 'source_run': str(SOURCE), 'sources': sources, 'schedule': schedule,
        'registered_source_failures': len(sources), 'registered_positions': len(schedule),
        'source_sha256': {**parent['source_hashes'], **{str(p): sha(p) for p in [HERE/'probe.py', HERE/'run_pipeline.py']}},
        'input_sha256': inputs, 'planner_warmup_trial': parent['planner_warmup_trial'],
        'fractions': FRACTIONS, 'minimum_actor_edge_gap_m': MIN_EDGE_GAP_M,
        'maximum_translation_m': MAX_TRANSLATION_M,
        'geometry_policy': 'Rigidly translate all objects together; unchanged relative geometry and all native grasps; table and robot fixed',
        'candidate_cap': None, 'physical_attempts': 0, 'fresh_neural_inference': False,
        'pure_ik_seeds': 128, 'returned_solutions': 32, 'position_tolerance_m': .002,
        'orientation_tolerance_rad': .03, 'planning': 'Original PRM and trajectory optimizer; same 128-seed diagnostic search at every distance',
        'original_search_miss_rule': 'If the unchanged baseline is recovered by stronger IK, label a search miss and never credit it to distance',
        'worker_count': 2, 'automatic_retries': 0}
    save(ROOT / 'registration.json', reg)
    save(ROOT / 'progress.json', {'phase': 'registered', 'completed': 0, 'registered': 318, 'source_failures': 53})
    save(HERE / 'active_run.json', {'root': str(ROOT), 'registered_sources': 53, 'registered_positions': 318, 'physical_grasp_budget': 0})
    debug = ROOT / 'debug'; debug.mkdir()
    (debug / '假设.md').write_text('# 全量失败假设\n\nH1：部分场景主要受运动学可达性限制，整组靠近可恢复。\n\nH2：部分场景主要受桌面或邻物碰撞限制，只靠近无法解决。\n\nH3：原单候选IK搜索可能漏解，在原位加强搜索即可恢复；该恢复不计入距离收益。\n')
    (debug / '已排除.md').write_text('# 已排除的解释\n\n待全量结果。\n')
    (debug / '定位记录.md').write_text('# 初始症状\n\n完整200例已完成，53例规划不出来。53例原日志全部为抓取终点IK/碰撞合并计数，未进入预抓取或路径搜索。\n\n本轮53×6个距离位置，逐个拆分运动学、桌面、地面、邻物碰撞，再核查预抓取、抬升与完整轨迹。\n')
    print('REGISTERED', len(sources), len(schedule), 'source candidates', sum(s['candidates'] for s in sources), flush=True)


def capture():
    reg = read(ROOT / 'registration.json'); guard(reg)
    assert not (ROOT / 'capture_started.json').exists()
    save(ROOT / 'capture_started.json', {'at': now(), 'pid': os.getpid(), 'gpu': os.getenv('CUDA_VISIBLE_DEVICES')})
    sys.path[:0] = [str(PROJECT / 'experiments/diverse_object_scenes'), str(PROJECT / 'experiments/repaired_grasp_benchmark')]
    from pipeline import runtime
    from repair_contract import simulation_config
    environment, baseline, run, sapien, torch, make, *_ = runtime(load_object_catalog(SOURCE / 'object_catalog.json')['models'])
    original = environment.ARXPickYCB._default_sim_config
    environment.ARXPickYCB._default_sim_config = property(lambda self: simulation_config(original.fget(self)))
    rows = []
    for source in reg['sources']:
        if (ROOT / 'STOP').exists(): raise RuntimeError('STOP present')
        levels = [r for r in reg['schedule'] if r['source_id'] == source['id']]
        first = read(ROOT / 'cases' / levels[0]['id'] / 'entry.json')
        env = make(first['model_id'], first['scene'])
        try:
            env.reset(seed=first['seed']); baseline.settle(env)
            for row in levels:
                out = ROOT / 'cases' / row['id']; state = read(out / 'state.json')
                q = torch.tensor([state['qpos']], device=env.device)
                env.agent.robot.set_qpos(q); env.agent.robot.set_qvel(torch.zeros_like(q))
                for n, info in state['objects'].items():
                    p = info['pose']; actor = env.objects[n]
                    actor.set_pose(sapien.Pose(p[:3], p[3:]))
                    actor._bodies[0].linear_velocity = np.zeros(3); actor._bodies[0].angular_velocity = np.zeros(3)
                obs = env.get_obs(); actual = run.snapshot(env, obs)
                for n, g in actual['objects'].items():
                    assert np.max(np.abs(np.asarray(g['pose']) - state['objects'][n]['pose'])) < 2e-6
                for n, t in actual['links'].items(): assert np.max(np.abs(np.asarray(t) - state['links'][n])) < 2e-5
                from PIL import Image
                data = obs['sensor_data']['head']
                Image.fromarray(data['rgb'][0].cpu().numpy()).save(out / 'head.jpg', quality=88)
                Image.fromarray(env.render()[0].cpu().numpy()).save(out / 'overview.jpg', quality=88)
                mask = data['segmentation'][0, :, :, 0].cpu().numpy() == env.obj._objs[0].per_scene_id
                result = {'id': row['id'], 'passed': True, 'visible_target_pixels': int(mask.sum()), 'physical_grasps': 0}
                save(out / 'capture.json', result); rows.append(result)
            save(ROOT / 'capture_progress.json', {'completed': len(rows), 'registered': 318, 'last_source': source['id']})
            print('CAPTURE', len(rows), source['id'], flush=True)
        finally: env.close()
    guard(reg); save(ROOT / 'capture_complete.json', {'passed': True, 'completed': len(rows), 'registered': 318})


def run_worker(worker):
    reg = read(ROOT / 'registration.json'); guard(reg)
    worker_root = ROOT / 'workers' / str(worker)
    assert not (worker_root / 'started.json').exists()
    save(worker_root / 'started.json', {'pid': os.getpid(), 'at': now(), 'gpu': os.getenv('CUDA_VISIBLE_DEVICES')})
    sys.path[:0] = [str(PROJECT / 'experiments/rayst3r_graspgenx'), str(MS), str(MS / 'arx_pipeline')]
    from services import planner
    from planning_rejection_audit import make_pure
    from ik_workspace_audit import solve, collision_details
    from resident_planner import all_kinematics_params, collision_scene_config
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from scene_tools import ArmFK
    from scipy.spatial.transform import Rotation
    import torch
    args = SimpleNamespace(root=worker_root, warmup_trial=Path(reg['planner_warmup_trial']))
    pool = planner(args)
    model = pool.contexts['open'].model
    pure = make_pure(model); position = make_pure(model, True)
    aux = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),
        scene_model={'cuboid': {}}, n_meshes=8, n_cuboids=8, collision_activation_distance=0.))
    aux.self_collision_cost = None
    aux_params = all_kinematics_params(aux)
    fk = ArmFK()
    kp = pool.contexts['open'].checker.kinematics.kinematics_config
    names = {int(v): k for k, v in kp.link_name_to_idx_map.items()}
    sphere_names = [names[int(i)] for i in kp.link_sphere_idx_map.cpu().tolist()]
    selected_sources = reg['sources'][worker::reg['worker_count']]
    assigned = {s['id'] for s in selected_sources}
    completed = []
    save(worker_root / 'warmup_complete.json', {'at': now(), 'source_count': len(assigned), 'model_loads': pool.loads})
    for row in reg['schedule']:
        if row['source_id'] not in assigned: continue
        if (ROOT / 'STOP').exists(): raise RuntimeError('STOP present')
        out = ROOT / 'cases' / row['id']; state = read(out / 'state.json')
        assert not (out / 'result.json').exists()
        begin = time.perf_counter(); start_collision = False
        try: context = pool.bind(out / 'plan', state)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model': raise
            context = pool.contexts['open']; start_collision = True
        for p in aux_params: p.copy_(context.checker.kinematics.kinematics_config)
        meshes = {n: g for n, g in read(out / 'plan/scene.json').get('mesh', {}).items()}
        if meshes:
            aux.clear_scene_cache(); aux.update_world(collision_scene_config({'mesh': meshes}))
        start = context.measured(state)
        source_pool = read(Path(row['source_trial']) / 'candidates/candidate_pool.json')['candidates']
        contacts = np.array([c['pose'] for c in source_pool]); contacts[:, :3, 3] += row['translation_m']
        goals = contacts.copy(); goals[:, :3, 3] -= .043 * goals[:, :3, 2]

        def assess(worlds, pos_only=False, detailed=False):
            solver = position if pos_only else pure
            solver.reset_seed()
            q, ok, pe, re = solve(solver, context, worlds, start)
            ok &= pe <= .002001
            if not pos_only: ok &= re <= .030001
            ok &= ((q >= context.limits[:, 0]-1e-6) & (q <= context.limits[:, 1]+1e-6)).all(-1)
            clear = torch.zeros_like(ok); table = torch.zeros_like(ok); ground = torch.zeros_like(ok); obstacle = torch.zeros_like(ok)
            locations = ok.nonzero(); solutions = q[ok]
            table_links = Counter(); collision_witnesses = []
            for offset in range(0, len(solutions), 1024):
                part = solutions[offset:offset+1024]; loc = locations[offset:offset+len(part)]
                d = collision_details(context, state, part)
                clear[loc[:, 0], loc[:, 1]] = d['full_scene_valid']
                table[loc[:, 0], loc[:, 1]] = d['table_penetration'] > 0
                ground[loc[:, 0], loc[:, 1]] = d['ground_penetration'] > 0
                if meshes:
                    obstacle[loc[:, 0], loc[:, 1]] = ~aux.validate(part[:, None]).reshape(-1)
                table_bad = d['table_penetration'] > 0
                table_links.update(sphere_names[i] for i in d['worst_table_sphere'][table_bad].cpu().tolist())
                bad = (~d['full_scene_valid']).nonzero().flatten()
                if detailed and len(bad) and not collision_witnesses:
                    k = int(bad[0]); index = int(loc[k, 0]); joints = part[k].cpu().numpy()
                    actual = fk.fk(joints); target = context.invbase @ worlds[index]
                    ep = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
                    er = float(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude())
                    assert ep < .00201 and (pos_only or er < .03001)
                    collision_witnesses.append({'candidate': index, 'q': joints,
                        'table_penetration_m': float(d['table_penetration'][k]),
                        'worst_table_link': sphere_names[int(d['worst_table_sphere'][k])],
                        'obstacle_collision': bool(obstacle[loc[k, 0], loc[k, 1]]),
                        'cpu_position_error_m': ep, 'cpu_rotation_error_rad': er})
            values = {'pure_ik_candidates': int(ok.any(-1).sum()), 'scene_clear_candidates': int(clear.any(-1).sum()),
                'table_collision_candidates': int(table.any(-1).sum()), 'ground_collision_candidates': int(ground.any(-1).sum()),
                'obstacle_collision_candidates': int(obstacle.any(-1).sum()),
                'table_collision_links': dict(table_links), 'collision_witnesses': collision_witnesses}
            if detailed:
                np.savez_compressed(out / 'endpoint_ik.npz', solved=ok.cpu().numpy(), scene_clear=clear.cpu().numpy(),
                    table_collision=table.cpu().numpy(), ground_collision=ground.cpu().numpy(), obstacle_collision=obstacle.cpu().numpy(),
                    solved_indices=locations.cpu().numpy(), solved_q=solutions.cpu().numpy())
            return values, q, clear

        # Independent control: the same saved measured EEF must be recoverable.
        _, _, control = assess([np.asarray(state['links']['right_eef_link'])])
        assert bool(control.any()) or start_collision, 'Recorded measured-pose control failed'
        endpoint, qs, clear = assess(goals, detailed=True)
        position_diagnostic = None
        if endpoint['pure_ik_candidates'] == 0:
            position_diagnostic = assess(goals, pos_only=True)[0]
        candidate_clear = clear.any(-1).cpu().numpy()
        witnesses = []
        valid_indices = np.flatnonzero(candidate_clear)
        if len(valid_indices):
            i = int(valid_indices[0]); branches = qs[i][clear[i]]
            k = int((branches-start).square().sum(-1).argmin()); q = branches[k].clone()
            actual = fk.fk(q.cpu().numpy()); target = context.invbase @ goals[i]
            ep = float(np.linalg.norm(actual[:3, 3]-target[:3, 3])); er = float(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude())
            assert ep < .00201 and er < .03001
            witnesses.append({'candidate': i, 'q': q.cpu().numpy(), 'cpu_position_error_m': ep, 'cpu_rotation_error_rad': er})
        counts = Counter(); selected = None; paths = None

        def auxiliary(world, seed, label):
            q = context.ik(world, seed)
            if q is not None: return q
            a, solutions, valid = assess([world])
            if a['scene_clear_candidates']:
                choices = solutions[0][valid[0]]
                counts[label+'_native_search_miss'] += 1
                return choices[(choices-seed).square().sum(-1).argmin()].clone()
            counts[label+('_no_ik' if not a['pure_ik_candidates'] else '_collision')] += 1
            return None

        if not start_collision:
            for index in valid_indices:
                index = int(index)
                qg = context.ik(goals[index], start)
                recovered = qg is None
                if recovered:
                    choices = qs[index][clear[index]]
                    qg = choices[(choices-start).square().sum(-1).argmin()].clone()
                    counts['grasp_native_search_miss'] += 1
                pre, lift = goals[index].copy(), goals[index].copy()
                pre[:3, 3] -= .08 * pre[:3, 2]; lift[2, 3] += .03
                qp = auxiliary(pre, qg, 'pregrasp'); ql = auxiliary(lift, qg, 'lift')
                if qp is None or ql is None: continue
                reach, approach = context.path(start, qp), context.path(qp, qg)
                if reach is None: counts['initial_to_pregrasp_path_failed'] += 1
                if approach is None: counts['approach_path_failed'] += 1
                if reach is None or approach is None: continue
                paths = {'pregrasp': context.optimize('pregrasp', reach, 6), 'approach': context.optimize('approach', approach, 5)}
                selected = {'index': index, 'source_score': source_pool[index]['score'],
                    'grasp_world': goals[index], 'pregrasp_world': pre, 'lift_world': lift,
                    'q_grasp': qg.cpu().numpy(), 'q_pre': qp.cpu().numpy(), 'q_lift': ql.cpu().numpy(),
                    'grasp_used_recovered_ik': recovered}
                break
        if start_collision: diagnosis = 'initial_state_collision'
        elif endpoint['pure_ik_candidates'] == 0:
            diagnosis = 'fixed_grasp_orientation_ik_not_found' if position_diagnostic['pure_ik_candidates'] else 'position_ik_not_found'
        elif endpoint['scene_clear_candidates'] == 0:
            table_hit = endpoint['table_collision_candidates'] > 0
            obstacle_hit = endpoint['obstacle_collision_candidates'] > 0
            diagnosis = 'table_and_obstacle_collision' if table_hit and obstacle_hit else 'table_collision' if table_hit else 'obstacle_collision' if obstacle_hit else 'ground_or_other_collision'
        elif selected is not None: diagnosis = 'planning_passed'
        elif counts['pregrasp_no_ik'] or counts['pregrasp_collision'] or counts['lift_no_ik'] or counts['lift_collision']:
            diagnosis = 'pregrasp_or_lift_endpoint_failed'
        else: diagnosis = 'path_not_found'
        result = {**row, 'endpoint': endpoint, 'position_only': position_diagnostic,
            'planning_success': selected is not None, 'selected': selected, 'stage_failures': dict(counts),
            'start_collision': start_collision, 'diagnosis': diagnosis, 'witnesses': witnesses,
            'original_baseline_recovered': row['level'] == 0 and selected is not None,
            'physical_attempts': 0, 'wall_seconds': time.perf_counter()-begin}
        save(out / 'plan_result.json', {'success': selected is not None, 'selected': selected, 'paths': paths,
            'diagnostic_ik_seeds': 128, 'same_frozen_robot_and_collision_gates': True})
        save(out / 'result.json', result); completed.append(row['id'])
        save(worker_root / 'progress.json', {'completed': len(completed), 'registered': len(assigned)*6, 'last': result, 'at': now()})
        print('RESULT', row['id'], endpoint, diagnosis, dict(counts), flush=True)
    guard(reg)
    save(worker_root / 'complete.json', {'passed': True, 'completed': len(completed), 'source_inputs_unchanged': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'capture', 'plan']); p.add_argument('--worker', type=int, default=0)
    args = p.parse_args()
    try:
        if args.stage == 'prepare': prepare()
        elif args.stage == 'capture': capture()
        else: run_worker(args.worker)
    except BaseException:
        import traceback
        if ROOT.exists():
            save(ROOT / (args.stage + f'_error_{args.worker}.json'), {'error': traceback.format_exc(), 'at': now()})
            (ROOT / 'STOP').write_text(traceback.format_exc())
        raise
