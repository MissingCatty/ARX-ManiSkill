"""Offline counterfactuals for fixed approach waypoints and gripper orientation."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

import 核查挡物规划 as first

ROOT, MS = first.ROOT, first.MS
OUT = first.DEST.parent / '20261119_02'
read, save, sha = first.read, first.save, first.sha


def pose_goals(candidates, flip=False, distance=0.):
    goals = np.asarray([c['pose'] for c in candidates], dtype=float).copy()
    if flip:goals[:, :3, :3] = goals[:, :3, :3] @ np.diag([-1., -1., 1.])
    goals[:, :3, 3] -= (.043 + distance) * goals[:, :3, 2]
    return goals


def attempt_plan(pool, state, directory, grasp, pregrasp, qgs, qps, torch):
    """Apply original full-world entry validation and original approach gates."""
    ctx = pool.bind(directory / '原全场景进场模型', state, include_target=True)
    q0 = ctx.measured(state)
    qp_valid = ctx.checker.validate(torch.as_tensor(qps, dtype=q0.dtype, device=q0.device)[:, None]).reshape(-1).cpu().numpy()
    allowed = np.flatnonzero(qp_valid)
    if not len(allowed):return {'passed': False, 'stage': 'pregrasp_pose_hits_full_scene_including_grasped_object'}
    pair_order = sorted(((float(np.linalg.norm(qp - qg) + .2 * np.linalg.norm(qp - q0.cpu().numpy())), i, j)
                         for i, qg in enumerate(qgs) for j, qp in enumerate(qps) if qp_valid[j]))
    attempts = []
    for _, i, j in pair_order[:3]:
        # Restore original collision world when returning from an approach attempt.
        ctx = pool.bind(directory / ('全场景进场_' + str(len(attempts))), state, include_target=True)
        qg = torch.as_tensor(qgs[i], dtype=q0.dtype, device=q0.device)
        qp = torch.as_tensor(qps[j], dtype=q0.dtype, device=q0.device)
        reach = ctx.path(q0, qp)
        if reach is None:
            attempts.append({'stage': 'full_scene_pregrasp_path'});continue
        try:
            route = ctx.optimize('pregrasp', reach, 6.)
        except RuntimeError as error:
            if 'cuda' in str(error).lower():raise
            attempts.append({'stage': 'pregrasp_optimizer', 'error': str(error)});continue
        dense = torch.as_tensor(route['q'], dtype=q0.dtype, device=q0.device)
        if not bool(ctx.checker.validate(dense[:, None]).all()):
            attempts.append({'stage': 'optimized_pregrasp_full_scene_check'});continue
        ctx = pool.bind(directory / ('原抓取场景_' + str(len(attempts))), state)
        lift = grasp.copy();lift[2, 3] += .03
        ql = ctx.ik(lift, qg)
        if ql is None:
            attempts.append({'stage': 'original_nominal_lift_ik'});continue
        approach = ctx.path(qp, qg)
        if approach is None:
            attempts.append({'stage': 'original_approach_path'});continue
        try:
            approach_route = ctx.optimize('approach', approach, 5.)
        except RuntimeError as error:
            if 'cuda' in str(error).lower():raise
            attempts.append({'stage': 'approach_optimizer', 'error': str(error)});continue
        check = torch.as_tensor(approach_route['q'], dtype=q0.dtype, device=q0.device)
        if not bool(ctx.checker.validate(check[:, None]).all()):
            attempts.append({'stage': 'optimized_approach_scene_check'});continue
        return {'passed': True, 'stage': 'all_original_initial_planning_gates',
                'physical_attempts': 0, 'q_grasp': qg.cpu().tolist(), 'q_pregrasp': qp.cpu().tolist(),
                'q_nominal_lift': ql.cpu().tolist(), 'grasp_world': grasp.tolist(), 'pregrasp_world': pregrasp.tolist(),
                'full_scene_pregrasp_checked_points': len(route['q']),
                'original_approach_checked_points': len(approach_route['q']),
                'paths': {'pregrasp': route, 'approach': approach_route}, 'earlier_attempts': attempts}
    return {'passed': False, 'stage': 'bounded_original_path_or_optimizer_checks_failed', 'attempts': attempts}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    previous = read(first.DEST / '核查登记.json')
    cases = previous['cases']
    cases = [c for c in cases if c['control'] or c['reason'].startswith('抓取规划:')]
    hashes = dict(previous['source_sha256']);hashes[str(Path(__file__).resolve())] = sha(__file__)
    for c in cases:
        for p in (first.DEST / c['case']).glob('*.npz'):hashes[str(p)] = sha(p)
    save(OUT / '核查登记.json', {'time': datetime.now().astimezone().isoformat(), 'cases': cases,
         'source_sha256': hashes, 'pregrasp_distances_m': [.08, .06, .04, .02],
         'orientation_counterfactual': '180 degrees about the original approach axis; center and approach direction preserved',
         'physical_attempts': 0, 'robot_scene_collision_success_threshold_changes': False,
         'scope': 'Counterfactual goal and waypoint generation only. Do not modify historical results or production entry.'})
    import torch
    import resident_planner
    import planning_rejection_audit as audit
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner
    from physical_table_pair_20261118_01 import install_collision_policy
    from scene_tools import ArmFK

    install_collision_policy();resident_planner.GraspCLPlanner = ShortLiftPlanner
    warmup = MS / 'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
    started = time.monotonic();pool = FastPlannerPool(OUT / '规划器预热', warmup)
    full = audit.make_pure(pool.contexts['open'].model);fk = ArmFK()
    results = []
    for case in cases:
        name = case['case'];directory = Path(case['directory']);destination = OUT / name
        state = read(directory / 'state.json');ctx = pool.bind(destination / '原场景模型', state)
        candidates = read(directory / 'candidates/candidate_pool.json')['candidates']
        if case['control']:
            candidates = [candidates[read(directory / 'plan_result.json')['selected']['index']]]
        variants = {}
        for flip in [False, True]:
            for stage, distance in [('grasp', 0.), ('pre_08', .08), ('pre_06', .06), ('pre_04', .04), ('pre_02', .02), ('lift', 0.)]:
                if flip and stage in ['pre_06', 'pre_04', 'pre_02']:continue
                goals = pose_goals(candidates, flip, distance)
                if stage == 'lift':goals[:, 2, 3] += .03
                full.reset_seed();summary, arrays = audit.assess(ctx, full, fk, state, goals)
                key = ('flip_' if flip else 'native_') + stage
                np.savez_compressed(destination / (key + '.npz'), goals=goals, **arrays)
                variants[key] = {'summary': summary, 'arrays': arrays, 'goals': goals}
        masks = {};rows = []
        for variant, pre_key in [('native_08', 'native_pre_08'), ('native_06', 'native_pre_06'),
                                 ('native_04', 'native_pre_04'), ('native_02', 'native_pre_02'), ('flip_08', 'flip_pre_08')]:
            prefix = 'flip_' if variant.startswith('flip') else 'native_'
            feasible = (variants[prefix + 'grasp']['arrays']['open_scene_clear'].any(-1)
                        & variants[pre_key]['arrays']['open_scene_clear'].any(-1)
                        & variants[prefix + 'lift']['arrays']['open_scene_clear'].any(-1))
            masks[variant] = feasible
            rows.append({'variant': variant, 'all_three_original_scene_endpoints': int(feasible.sum()),
                         'candidate_indices': np.flatnonzero(feasible).tolist()})
        proof = None
        # Preserve the longer original stand-off first; diagnostic alternatives follow.
        for variant in ['native_08', 'flip_08', 'native_06', 'native_04', 'native_02']:
            if proof and proof['passed']:break
            prefix = 'flip_' if variant.startswith('flip') else 'native_'
            pre = prefix + 'pre_' + variant.rsplit('_', 1)[1]
            for index in np.flatnonzero(masks[variant])[:3]:
                grasp = variants[prefix + 'grasp'];pregrasp = variants[pre]
                qgs = grasp['arrays']['q'][index][grasp['arrays']['open_scene_clear'][index]]
                qps = pregrasp['arrays']['q'][index][pregrasp['arrays']['open_scene_clear'][index]]
                result = attempt_plan(pool, state, destination / (variant + '_候选' + str(index)),
                                      grasp['goals'][index], pregrasp['goals'][index], qgs, qps, torch)
                result.update(variant=variant, candidate_index=int(index))
                save(destination / (variant + '_候选' + str(index) + '_规划证明.json'), result)
                if result['passed']:
                    proof = result;break
        output = {'case': name, 'control': case['control'], 'retained_candidates': len(candidates),
                  'endpoint_combinations': rows, 'planning_witness': proof,
                  'variant_summaries': {k:v['summary'] for k,v in variants.items()}, 'physical_attempts': 0}
        save(destination / '核查结果.json', output);results.append(output)
        save(OUT / '当前进度.json', {'completed': len(results), 'total': len(cases), 'physical_attempts': 0})
        print('COUNTERFACTUAL', name, [(r['variant'], r['all_three_original_scene_endpoints']) for r in rows],
              'PLAN', None if proof is None else proof['variant'], flush=True)
    assert all(sha(p) == h for p,h in hashes.items())
    save(OUT / '预抓取约束核查汇总.json', {'cases': results, 'source_hashes_unchanged': True,
         'physical_attempts': 0, 'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':main()
