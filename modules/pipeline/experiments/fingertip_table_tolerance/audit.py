"""Register and audit archived IK solutions without simulation or new IK."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

from contact_policy import TableContactPolicy
from geometry import Geometry, JOINTS, collision_meshes, tip_plane

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parent
SOURCE = PROJECT / 'experiments/full_failure_distance_probe/runs/all53_20261123_01'
ROOT = HERE / 'runs/clearance67_20261123_02'
TOLERANCES = [0., .0005, .001, .002]
ASSET = WORKSPACE / 'maniskill/assets/arx'


def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False) + '\n')
    tmp.replace(path)


def guard(reg):
    assert not (ROOT / 'STOP').exists(), 'Registered STOP exists'
    for p, h in reg['input_sha256'].items():
        assert sha(p) == h, 'Frozen input changed: ' + p
    for p, h in reg['code_sha256'].items():
        assert sha(p) == h, 'Registered code changed: ' + p


def prepare():
    assert not ROOT.exists(), 'Never overwrite an existing registration'
    assert read(SOURCE / 'completion_verification.json')['passed']
    original = read(SOURCE / 'registration.json')
    cases = []
    for source in original['sources']:
        for level in [0, 5]:
            case = SOURCE / 'cases' / (source['id'] + f'_d{level}')
            result = read(case / 'result.json')
            if level == 5 and result['planning_success']: continue
            cases.append({'id': case.name, 'source_id': source['id'],
                          'cohort': 'original53' if level == 0 else 'unresolved_near14',
                          'source_case': str(case), 'legal_saved_q': len(np.load(case / 'endpoint_ik.npz')['solved_q'])})
    assert len(cases) == 67
    _, mesh_paths = collision_meshes(ASSET)
    inputs = {str(SOURCE / 'registration.json'): sha(SOURCE / 'registration.json')}
    for row in cases:
        case = Path(row['source_case'])
        for n in ['state.json', 'entry.json', 'result.json', 'endpoint_ik.npz', 'plan/arx.yml']:
            inputs[str(case / n)] = sha(case / n)
    for path in mesh_paths:
        inputs[str(path)] = sha(path)
    g = Geometry(Path(cases[0]['source_case']) / 'plan/arx.yml', read(Path(cases[0]['source_case']) / 'state.json'))
    inputs[str(g.urdf)] = sha(g.urdf)
    reg = {'created_at': now(), 'version': 'V2', 'source_run': str(SOURCE),
           'scope': '53 unchanged baselines plus 14 unresolved nearest positions; archived legal IK solutions only',
           'registered_positions': 67, 'archived_q': sum(r['legal_saved_q'] for r in cases),
           'schedule': cases, 'tolerance_m': TOLERANCES, 'distal_region_length_m': .010,
           'allowed_stages': ['final_approach', 'closure'],
           'physics_contact_offset_m': .001, 'physical_attempt_budget': 0,
           'new_ik_calls': 0, 'neural_inference_calls': 0, 'automatic_retries': 0,
           'other_world_checks': 'Original archived native cuRobo obstacle and ground receipts, indexed by exact saved candidate/branch; joint limits rechecked',
           'code_sha256': {str(p): sha(p) for p in HERE.glob('*.py')}, 'input_sha256': inputs}
    save(ROOT / 'registration.json', reg)
    save(ROOT / 'progress.json', {'phase': 'registered', 'registered': 67, 'completed': 0})
    save(HERE / 'active_run.json', {'root': str(ROOT), 'version': 'V2', 'physical_grasp_budget': 0})
    docs = PROJECT / 'docs/mainline_versions/debug/fingertip_table_tolerance'
    docs.mkdir(parents=True, exist_ok=True)
    (docs / '假设.md').write_text('# 待验证假设\n\nH1：存在不超过2毫米、只发生在指尖的桌面接触，局部余量可以恢复这些终点。反例：所有其他检查通过的解均需要更大余量或碰到非尖端区域。\n\nH2：部分碰桌来自碰撞球的保守外包络。检验：同一关节状态的真实凸碰撞网格不碰桌，而原球模型碰桌。\n\nH3：大部分未恢复解是夹指主体、手腕或手臂明显侵入，微小指尖余量不会起效。逐场景统计最小深度与部位，不能由第一条回执外推。\n')
    (docs / '已排除.md').write_text('# 已排除的解释\n\n待本轮核验。\n')
    (docs / '定位记录.md').write_text('# 初始状态\n\nV1的53例规划不出来；距离诊断后14例最近位置仍失败。本轮遍历67位置保存的全部合法IK解，核对桌面几何和局部指尖余量。新IK、模型推理与物理抓取均为0。\n')
    print('REGISTERED', len(cases), reg['archived_q'], flush=True)


def finite_value(x):
    return float(x) if np.isfinite(x) else None


def mesh_depths(geometry, meshes, q, table):
    """Exact minimum Z of each convex collider clipped to the table XY prism.

    A convex combination is used for each original collider separately. This
    includes crossing triangle edges and never fills the gap between fingers.
    """
    frames = geometry.frames(np.asarray(q)[None])
    center, half = np.asarray(table['center']), np.asarray(table['dims']) / 2
    records = {}
    for name, parts in meshes.items():
        if name not in frames: continue
        t = frames[name][0]
        best = None
        for vertices in parts:
            xyz = vertices @ t[:3, :3].T + t[:3, 3]
            # Fast rejection is exact for disjoint axis-aligned bounds.
            if (xyz[:, :2].max(0) < center[:2]-half[:2]).any() or (xyz[:, :2].min(0) > center[:2]+half[:2]).any(): continue
            A = np.stack([xyz[:, 0], -xyz[:, 0], xyz[:, 1], -xyz[:, 1]])
            b = [center[0]+half[0], -center[0]+half[0], center[1]+half[1], -center[1]+half[1]]
            lp = linprog(xyz[:, 2], A_ub=A, b_ub=b, A_eq=np.ones((1, len(xyz))), b_eq=[1.],
                         bounds=(0., None), method='highs')
            if lp.status == 2: continue
            assert lp.success, lp.message
            point = lp.x @ xyz
            if best is None or point[2] < best[2]: best = point
        if best is not None:
            eef = frames['right_eef_link'][0]
            records[name] = {'table_penetration_m': float(center[2]+half[2]-best[2]),
                             'lowest_point_world': best,
                             'lowest_point_eef': (best-eef[:3, 3]) @ eef[:3, :3]}
    return records


def audit_case(row, meshes):
    source = Path(row['source_case']); state = read(source / 'state.json')
    original = read(source / 'result.json')
    geometry = Geometry(source / 'plan/arx.yml', state)
    plane, per_tip = tip_plane(geometry, meshes)
    policy = TableContactPolicy(.002, plane)
    with np.load(source / 'endpoint_ik.npz') as saved:
        indices, qs = saved['solved_indices'], saved['solved_q']
        at = tuple(indices.T)
        native_clear = saved['scene_clear'][at]
        native_table = saved['table_collision'][at]
        native_ground = saved['ground_collision'][at]
        obstacle = saved['obstacle_collision'][at]
        count = len(saved['solved'])
    arrays = {n: [] for n in ['table', 'strict', 'tip', 'finger', 'nonfinger', 'ground', 'bounds', 'worst_sphere']}
    near_zero_mismatches = []
    for offset in range(0, len(qs), 128):
        q = qs[offset:offset+128]
        frame = geometry.frames(q); spheres = geometry.spheres(frame)
        result = policy.evaluate(spheres, frame['right_eef_link'], geometry.names, state['scene_cuboids']['table'],
                                 stage='final_approach', other_world_clear=np.ones(len(q), dtype=bool))
        from geometry import box_depth
        ground = box_depth(spheres, state['scene_cuboids']['ground']).max(axis=1)
        arrays['table'].append(result['table_depth_m']); arrays['strict'].append(result['strict_depth_m'])
        arrays['tip'].append(result['tip_depth_m']); arrays['finger'].append(result['all_finger_depth_m'])
        arrays['nonfinger'].append(result['nonfinger_depth_m']); arrays['ground'].append(ground)
        arrays['bounds'].append(((q >= geometry.lower-1e-6) & (q <= geometry.upper+1e-6)).all(axis=1))
        arrays['worst_sphere'].append(result['penetration_m'].argmax(axis=1))
    values = {n: np.concatenate(v) if v else np.zeros(0, dtype=bool if n == 'bounds' else float) for n, v in arrays.items()}
    other = ~obstacle & ~native_ground & values['bounds']
    mismatch = (values['table'] > 0) != native_table
    ground_mismatch = (values['ground'] > 0) != native_ground
    assert not np.any(mismatch & (np.abs(values['table']) > 2e-6)), 'Independent/native table classification differs by >2um'
    assert not np.any(ground_mismatch & (np.abs(values['ground']) > 2e-6)), 'Independent/native ground classification differs by >2um'
    baseline = other & (values['table'] <= 0)
    assert np.array_equal(baseline, native_clear), 'Independent zero-tolerance result does not reproduce V1 receipts'
    local_base = other & (values['strict'] <= 0)
    finger_base = other & (values['nonfinger'] <= 0)
    def total(mask): return {'solutions': int(mask.sum()), 'candidates': int(len(np.unique(indices[mask, 0])))}
    levels = []
    for tolerance in TOLERANCES:
        mask = local_base & (values['tip'] <= tolerance)
        loose = finger_base & (values['finger'] <= tolerance)
        levels.append({'tolerance_m': tolerance, 'distal_only': total(mask), 'whole_finger_upper_bound': total(loose),
                       'first_distal_solution': int(np.flatnonzero(mask)[0]) if mask.any() else None})
    def witness(mask, metric):
        if not mask.any(): return None
        chosen = int(np.flatnonzero(mask)[np.argmin(metric[mask])])
        item = {'solution_index': chosen, 'candidate_index': int(indices[chosen, 0]),
                'branch_index': int(indices[chosen, 1]), 'q': qs[chosen],
                'table_depth_m': finite_value(values['table'][chosen]),
                'strict_depth_m': finite_value(values['strict'][chosen]),
                'tip_depth_m': finite_value(values['tip'][chosen]),
                'worst_sphere_link': geometry.names[int(values['worst_sphere'][chosen])],
                'real_collision_mesh': mesh_depths(geometry, meshes, qs[chosen], state['scene_cuboids']['table'])}
        return item
    best = witness(other, values['table'])
    local_best = witness(local_base, values['tip'])
    from geometry import box_depth
    witness_errors = []
    for item in original['endpoint']['collision_witnesses']:
        sf = geometry.spheres(geometry.frames(np.asarray(item['q'])[None]))
        witness_errors.append(abs(float(box_depth(sf, state['scene_cuboids']['table']).max())-item['table_penetration_m']))
    # Old witness depths came from a large FP32 GPU matrix multiply. The native
    # FK spheres were independently checked against this CPU FK below 0.1 um;
    # retain old depth rounding separately, never use it as a policy allowance.
    assert not witness_errors or max(witness_errors) < 2e-4
    result = {**row, 'candidate_count': count, 'saved_q': len(qs), 'other_world_clear_q': int(other.sum()),
              'original_planning_success': original['planning_success'],
              'tip_plane_eef_z_m': plane, 'per_finger_tip_eef_z_m': per_tip,
              'local_tip_eligible_q': int(local_base.sum()), 'whole_finger_eligible_q': int(finger_base.sum()),
              'native_crosscheck': {'table_mismatches': int(mismatch.sum()), 'ground_mismatches': int(ground_mismatch.sum()),
                                    'zero_tolerance_matches_all_archived_scene_receipts': True,
                                    'maximum_witness_depth_error_m': max(witness_errors, default=0.),
                                    'depth_rounding_is_not_contact_tolerance': True},
              'thresholds': levels, 'minimum_table_depth_other_clear_m': finite_value(values['table'][other].min()) if other.any() else None,
              'minimum_distal_tolerance_m': finite_value(values['tip'][local_base].min()) if local_base.any() else None,
              'best_table_witness': best, 'best_local_witness': local_best, 'physical_attempts': 0}
    out = ROOT / 'cases' / row['id']; out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'clearance.npz', indices=indices, **values, other_world_clear=other)
    save(out / 'result.json', result)
    return result


def audit():
    reg = read(ROOT / 'registration.json'); guard(reg)
    assert not (ROOT / 'started.json').exists(), 'Do not retry a started audit'
    save(ROOT / 'started.json', {'at': now()})
    meshes, _ = collision_meshes(ASSET)
    rows = []; started = time.perf_counter()
    for row in reg['schedule']:
        assert not (ROOT / 'STOP').exists()
        result = audit_case(row, meshes); rows.append(result)
        save(ROOT / 'progress.json', {'phase': 'running', 'completed': len(rows), 'registered': 67,
                                      'last': row['id'], 'elapsed_s': time.perf_counter()-started, 'at': now()})
        print('AUDITED', len(rows), row['id'], 'q', result['saved_q'],
              'min_table_mm', None if result['minimum_table_depth_other_clear_m'] is None else round(result['minimum_table_depth_other_clear_m']*1000, 3),
              'distal2mm', result['thresholds'][-1]['distal_only']['candidates'], flush=True)
    cohorts = {}
    for group in ['original53', 'unresolved_near14']:
        part = [r for r in rows if r['cohort'] == group]
        cohorts[group] = {'positions': len(part), 'archived_solutions': sum(r['saved_q'] for r in part), 'thresholds': []}
        for i, t in enumerate(TOLERANCES):
            cohorts[group]['thresholds'].append({'tolerance_m': t,
                'distal_recovered_positions': sum(r['thresholds'][i]['distal_only']['candidates'] > 0 for r in part),
                'distal_valid_candidates': sum(r['thresholds'][i]['distal_only']['candidates'] for r in part),
                'whole_finger_upper_bound_positions': sum(r['thresholds'][i]['whole_finger_upper_bound']['candidates'] > 0 for r in part)})
    summary = {'completed_at': now(), 'registered': 67, 'completed': len(rows), 'archived_solutions': sum(r['saved_q'] for r in rows),
               'cohorts': cohorts, 'new_ik_calls': 0, 'physical_attempts': 0,
               'endpoint_recovery_is_not_complete_planning': True,
               'native_table_mismatches': sum(r['native_crosscheck']['table_mismatches'] for r in rows),
               'maximum_native_witness_error_m': max(r['native_crosscheck']['maximum_witness_depth_error_m'] for r in rows),
               'wall_s': time.perf_counter()-started}
    guard(reg)
    save(ROOT / 'summary.json', summary)
    save(ROOT / 'completion_verification.json', {'passed': True, 'completed': 67,
         'source_inputs_unchanged': True, 'zero_tolerance_reproduced_original_receipts': True, 'physical_attempts': 0})
    save(ROOT / 'progress.json', {'phase': 'complete', 'completed': 67, 'registered': 67, 'at': now()})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'audit'])
    args = p.parse_args()
    try:
        prepare() if args.stage == 'prepare' else audit()
    except BaseException:
        import traceback
        if ROOT.exists():
            save(ROOT / 'error.json', {'at': now(), 'error': traceback.format_exc()})
            (ROOT / 'STOP').write_text(traceback.format_exc())
        raise
