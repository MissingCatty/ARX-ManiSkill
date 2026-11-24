"""Saved-state-only separation of IK search and collision rejections; no simulator."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
import types
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT.parent / 'maniskill'
sys.path[:0] = [str(MS), str(MS / 'arx_pipeline'), str(MS / 'tmp')]
BASE = ROOT / 'docs/06_做配对验证/resources'
DEST = ROOT / 'docs/03_夹走并放下挡物/规划失败核查/resources/20261119_01'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cases():
    rows = read(BASE / 'S5_固定挡物接续01/配对合并验收.json')['rows']
    wanted = [r for r in rows if r['reason'].startswith('抓取规划:') or r['case'] in ['I0167', 'I0190', 'I0069']]
    order = ['I0069', 'I0038', 'I0018']
    wanted.sort(key=lambda r: (order.index(r['case']) if r['case'] in order else 3, r['case']))
    for row in wanted:
        row['directory'] = str(BASE / row['run_directory'] / '01_抓挡')
        row['control'] = row['case'] == 'I0069'
    return wanted


def static_check(case):
    directory = Path(case['directory'])
    state = read(directory / 'state.json')
    pool = read(directory / 'candidates/candidate_pool.json')
    original = read(directory / 'candidates/unrefined_candidate_pool.json')['candidates']
    camera = np.asarray(state['camera_world_optical'])
    axes = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    errors, angles = [], []
    for candidate in pool['candidates']:
        m = candidate['metadata']
        row = np.asarray(m['source_postprocessed_row'])
        rotation = camera[:3, :3] @ row[4:13].reshape(3, 3)
        contact = np.eye(4)
        contact[:3, :3] = rotation @ axes
        contact[:3, 3] = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3] * rotation[:, 0]
        source_error = float(np.abs(contact - np.asarray(original[m['source_pool_index']]['pose'])).max())
        contact[:3, 3] -= m['insertion_retraction_m'] * contact[:3, 2]
        errors.append(max(source_error, float(np.abs(contact - np.asarray(candidate['pose'])).max())))
        angles.append(float(m['orientation_change_rad']))
    assert max(errors, default=0) < 1e-7
    assert max(angles, default=0) < 1e-7
    marked = [name for name, item in state['scene_meshes'].items() if item['target']]
    assert marked == ['distractor_00']
    planned_scene = read(directory / 'plan/scene.json')
    # Geometry cache keys can include a digest; the archived scene uses original names.
    expected = {name for name, item in state['scene_meshes'].items() if not item['target']}
    assert set(planned_scene['mesh']) == expected
    report = {'case': case['case'], 'retained': len(pool['candidates']),
              'max_transform_reconstruction_error': max(errors, default=0),
              'maximum_orientation_change_rad': max(angles, default=0),
              'temporary_target': marked, 'obstacles': sorted(expected),
              'historical_fk_validation': read(directory / 'plan/fk_validation.json'),
              'historical_rejections': read(directory / 'plan_result.json')['rejected']}
    return report


def register():
    selected = cases()
    DEST.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), MS / 'arx_pipeline/planner.py', MS / 'arx_pipeline/scene_tools.py',
             MS / 'resident_planner.py', MS / 'fast_resident_planner.py', MS / 'ik_workspace_audit.py',
             MS / 'planning_rejection_audit.py', MS / 'tmp/physical_table_pair_20261118_01.py',
             MS / 'assets/arx/robot.urdf',
             ROOT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf',
             ROOT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.yml']
    for case in selected:
        directory = Path(case['directory'])
        paths += [directory / p for p in ['state.json', 'plan_result.json', 'candidates/candidate_pool.json',
                                        'candidates/unrefined_candidate_pool.json', 'plan/arx.yml', 'plan/scene.json']]
        paths += [Path(g['file_path']) for g in read(directory / 'state.json')['scene_meshes'].values()]
    reg = {'time': datetime.now().astimezone().isoformat(), 'cases': selected,
           'source_sha256': {str(p): sha(p) for p in set(paths)},
           'full_pose_search_seeds': 128, 'retained_solutions_per_goal': 32,
           'required_components': ['original ARX planner/checker', 'pure full-pose IK', 'position-only IK', 'independent URDF FK'],
           'physical_attempts': 0, 'simulator_access': False,
           'scope': 'Read saved inputs, replay unchanged endpoint gates, inspect wider kinematic witnesses and original-world collision checks. No original result edits, no commands or feedback updates.'}
    save(DEST / '核查登记.json', reg)
    static = [static_check(case) for case in selected]
    save(DEST / '坐标与实例检查.json', {'passed': True, 'cases': static})
    print('STATIC_CHECK_PASSED', len(static), flush=True)
    return reg


def cpu_pose(fk, context, world, qs, limits):
    expected = context.invbase @ world
    positions, rotations = [], []
    for q in qs:
        actual = fk.fk(q)
        positions.append(float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])))
        rotations.append(float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude()))
    bounds = ((qs >= limits[:, 0] - 1e-6) & (qs <= limits[:, 1] + 1e-6)).all(-1)
    return np.asarray(positions), np.asarray(rotations), bounds


def run():
    reg = register()
    import torch
    import resident_planner
    import planner as original
    import planning_rejection_audit as audit
    import ik_workspace_audit as helper
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner
    from physical_table_pair_20261118_01 import install_collision_policy
    from scene_tools import ArmFK

    install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner
    warmup = MS / 'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
    began = time.monotonic()
    pool = FastPlannerPool(DEST / '规划器预热', warmup)
    full = audit.make_pure(pool.contexts['open'].model)
    position_only = audit.make_pure(pool.contexts['open'].model, position_only=True)
    fk = ArmFK()
    probe_dir = DEST / '正向探针'
    probe_dir.mkdir()
    save(probe_dir / 'registration.json', {'warmup_trial': str(warmup)})
    helper.probe(pool, full, position_only, probe_dir)
    save(DEST / '组件就绪.json', {'physical_attempts': 0, 'model_load_counts': pool.loads,
         'full_pose_ik_loads': 1, 'position_only_ik_loads': 1, 'startup_seconds': time.monotonic() - began})

    all_results = []
    for case in reg['cases']:
        name = case['case']; directory = Path(case['directory']); target = DEST / name
        state = read(directory / 'state.json')
        context = pool.bind(target / '原场景模型', state)
        candidates = read(directory / 'candidates/candidate_pool.json')['candidates']
        if case['control']:
            chosen = read(directory / 'plan_result.json')['selected']['index']
            candidates = [candidates[chosen]]
        limits = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in original.NAMES])
        actual_q = context.measured(state)
        cpu_initial = context.base @ fk.fk(actual_q.cpu().numpy())
        assert np.abs(cpu_initial - np.asarray(state['links']['right_eef_link'])).max() < 2e-5
        trace = []

        def record(self, world, start, result):
            qs = result.solution.detach().clone().reshape(-1, 6)
            success = result.success.detach().clone().reshape(-1)
            p, r, bound = cpu_pose(fk, self, world, qs.cpu().numpy(), limits)
            kin = (p <= .002001) & (r <= .030001) & bound
            scene = self.checker.validate(qs[:, None]).reshape(-1).cpu().numpy()
            checker_bounds = (self.checker.get_bound(qs[:, None]).sum(-1).reshape(-1) == 0).cpu().numpy()
            trace.append({'candidate': self.diagnostic_candidate, 'stage': self.diagnostic_stage,
                          'world_goal': world.tolist(), 'q': qs.cpu().tolist(),
                          'solver_success': success.cpu().tolist(), 'position_error_m': p.tolist(),
                          'rotation_error_rad': r.tolist(), 'actual_bounds_valid': bound.tolist(),
                          'cpu_full_pose_valid': kin.tolist(), 'checker_bounds_valid': checker_bounds.tolist(),
                          'original_scene_valid': scene.tolist()})

        source = textwrap.dedent(inspect.getsource(original.Planner.ik))
        needle = '    if not result.success.any():'
        assert source.count(needle) == 1
        source = source.replace(needle, '    record(self, world, start, result)\n' + needle)
        namespace = {**original.__dict__, 'record': record}
        exec(compile(source, str(Path(__file__)) + ':traced_original_ik', 'exec'), namespace)
        original_ik = context.ik
        context.ik = types.MethodType(namespace['ik'], context)
        worlds = []; native_grasps = []; native_counts = Counter()
        for index, candidate in enumerate(candidates):
            world = np.asarray(candidate['pose']).copy(); world[:3, 3] -= .043 * world[:3, 2]
            worlds.append(world)
            context.diagnostic_candidate = index; context.diagnostic_stage = 'grasp'
            qg = context.ik(world, actual_q); native_grasps.append(qg is not None)
            if qg is None:
                native_counts['grasp_ik'] += 1
                continue
            pre = world.copy(); pre[:3, 3] -= .08 * world[:3, 2]
            lift = world.copy(); lift[2, 3] += .03
            context.diagnostic_stage = 'pregrasp'; qp = context.ik(pre, qg)
            context.diagnostic_stage = 'lift'; ql = context.ik(lift, qg)
            if qp is None or ql is None:
                native_counts['pre_or_lift_ik'] += 1
        context.ik = original_ik
        save(target / '原求解器分步记录.json', {'events': trace, 'counts': dict(native_counts)})
        summaries = {}; stage_arrays = {}
        for stage in ['grasp', 'pregrasp', 'lift']:
            goals = np.asarray(worlds).copy()
            if stage == 'pregrasp':goals[:, :3, 3] -= .08 * goals[:, :3, 2]
            if stage == 'lift':goals[:, 2, 3] += .03
            full.reset_seed()
            summary, arrays = audit.assess(context, full, fk, state, goals)
            np.savez_compressed(target / (stage + '_运动学与碰撞.npz'), goals=goals, **arrays)
            summaries[stage] = summary; stage_arrays[stage] = arrays
        missing = np.flatnonzero(~stage_arrays['grasp']['cpu_pose_valid'].any(-1))
        position_summary = None
        if len(missing):
            position_only.reset_seed()
            position_summary, arrays = audit.assess(context, position_only, fk, state, np.asarray(worlds)[missing], position_only=True)
            np.savez_compressed(target / '仅位置解.npz', candidate_indices=missing, **arrays)
        result = {'case': name, 'source': str(directory), 'control': case['control'], 'candidate_count': len(candidates),
                  'native_grasp_pass': native_grasps, 'native_rejections': dict(native_counts),
                  'historical_rejections': read(directory / 'plan_result.json')['rejected'],
                  'wide_search': summaries, 'position_only_for_missing_grasp': position_summary,
                  'physical_attempts': 0}
        save(target / '核查结果.json', result); all_results.append(result)
        save(DEST / '当前进度.json', {'completed': len(all_results), 'total': len(reg['cases']), 'physical_attempts': 0})
        print('CASE', name, 'candidates', len(candidates), 'full_pose', summaries['grasp']['cpu_pose_valid_candidates'],
              'scene_clear', summaries['grasp']['open_scene_clear_candidates'], flush=True)
    assert all(sha(p) == value for p, value in reg['source_sha256'].items())
    save(DEST / '运动学核查汇总.json', {'cases': all_results, 'physical_attempts': 0, 'source_hashes_unchanged': True,
                                    'wall_seconds': time.monotonic() - began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    if args.run:run()
