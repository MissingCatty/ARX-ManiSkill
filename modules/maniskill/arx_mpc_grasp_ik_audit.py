"""Separate kinematic search failures from scene collisions for saved MPC benchmark grasps."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

from planning_rejection_audit import assess, fk_names, make_pure, read, save, sha

CODE = Path(__file__).resolve().parent
SOURCE = CODE / 'outputs/arx_graspcl_mpc_200_20261119_03'
ARM = SOURCE / 'free_graspcl_mpc'
CASE_PREFIXES = ('I0003', 'I0015', 'I0044', 'I0048', 'I0055', 'I0077')
CASE_SUFFIX = '_MPC200_01'
TCP_M = .043


def register(out):
    slots = {r['canonical_case_id'].split('_')[0]: r['id'] for r in read(SOURCE / 'registration.json')['schedule']}
    cases, sources = [], {}
    for prefix in CASE_PREFIXES:
        slot = slots[prefix]
        assert slot.endswith(CASE_SUFFIX)
        claim_path = ARM / 'claims' / (slot + '.json')
        claim = read(claim_path)
        assert claim['state'] == 'complete'
        trial = ARM / 'trials' / slot
        plan = read(trial / 'plan_result.json')
        pool = read(trial / 'graspcl/candidate_pool.json')['candidates']
        result = read(trial / 'result.json')
        assert claim['result_sha256'] == sha(trial / 'result.json')
        assert read(trial / 'independent_evaluation.json')['passed']
        assert not result['planning_success'] and not plan['success']
        assert plan['count'] == len(pool) == plan['rejected']['grasp_ik'] > 0
        cases.append({'id': result['canonical_case_id'], 'trial': str(trial), 'candidates': len(pool)})
        paths = (claim_path, trial / 'result.json', trial / 'state.json',
                 trial / 'graspcl/candidate_pool.json', trial / 'plan_result.json')
        sources.update({str(path): sha(path) for path in paths})
    assert sum(case['candidates'] for case in cases) == 322
    warmup = read(SOURCE / 'registration.json')['warmup_trial']
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    reg = {'source': str(SOURCE), 'warmup_trial': warmup, 'cases': cases, 'source_sha256': sources,
           'full_pose_seeds': 128, 'retained_solutions': 32, 'physical_attempts': 0,
           'scope': 'Use saved TCP43 candidates. Preserve target position and orientation for full-pose IK. '
                    'Check found solutions with the original measured open-gripper scene model. '
                    'Finite search results are not a proof of infeasibility.'}
    save(out / 'registration.json', reg)
    return reg


def confirm_path(context, start, world, q):
    """Send the recovered endpoint through the remaining original planning gates."""
    pre, lift = world.copy(), world.copy()
    pre[:3, 3] -= .08 * world[:3, 2]
    lift[2, 3] += .03
    q_pre, q_lift = context.ik(pre, q), context.ik(lift, q)
    if q_pre is None or q_lift is None:
        return {'stage': 'pregrasp_or_lift_ik', 'pregrasp': q_pre is not None, 'lift': q_lift is not None}
    reach, approach = context.path(start, q_pre), context.path(q_pre, q)
    if reach is None or approach is None:
        return {'stage': 'path', 'pregrasp': reach is not None, 'approach': approach is not None}
    try:
        routes = {'pregrasp': context.optimize('audit_pregrasp', reach, 6),
                  'approach': context.optimize('audit_approach', approach, 5)}
    except RuntimeError as error:
        return {'stage': 'optimizer', 'error': str(error)}
    return {'stage': 'complete_plan', 'routes': routes, 'q_grasp': q.tolist(),
            'q_pregrasp': q_pre.tolist(), 'q_lift': q_lift.tolist()}


def audit_case(pool, full_pose, position_only, fk, case, out):
    import numpy as np
    import torch

    trial = Path(case['trial'])
    state = read(trial / 'state.json')
    destination = out / case['id']
    context = pool.bind(destination / 'model', state)
    start = context.measured(state)
    expected = context.invbase @ np.asarray(state['links']['right_eef_link'])
    joints = [state['qpos'][state['joint_names'].index(name)] for name in fk_names()]
    assert np.linalg.norm(fk.fk(joints)[:3, 3] - expected[:3, 3]) < 2e-5
    candidates = read(trial / 'graspcl/candidate_pool.json')['candidates']
    worlds = np.asarray([g['pose'] for g in candidates], dtype=float)
    worlds[:, :3, 3] -= TCP_M * worlds[:, :3, 2]

    # Replay the exact endpoint call once, independently from the wider diagnostic search.
    original = [context.ik(world, start) for world in worlds]
    full_pose.reset_seed()
    summary, solutions = assess(context, full_pose, fk, state, worlds)
    np.savez_compressed(destination / 'full_pose_solutions.npz', goals=worlds, **solutions)
    kinematic = solutions['cpu_pose_valid'].any(-1)
    safe = solutions['open_scene_clear'].any(-1)
    bounds_and_floor = solutions['table_clear'] & solutions['ground_clear']
    blocked_by_table_in_all_found = kinematic & ~solutions['table_clear'].any(-1)
    collision_blocked = kinematic & ~safe
    only_position = np.zeros(len(worlds), dtype=bool)
    missing = np.flatnonzero(~kinematic)
    if len(missing):
        position_only.reset_seed()
        _, positions = assess(context, position_only, fk, state, worlds[missing], position_only=True)
        only_position[missing] = positions['cpu_pose_valid'].any(-1)
        np.savez_compressed(destination / 'position_solutions.npz',
                            candidate_indices=missing, goals=worlds[missing], **positions)

    recovered = []
    for index in range(len(worlds)):
        q_values = solutions['q'][index][solutions['open_scene_clear'][index]]
        if original[index] is not None:
            q_values = np.vstack([original[index].cpu().numpy(), q_values])
        if not len(q_values):
            continue
        nearest = np.argsort(np.linalg.norm(q_values - start.cpu().numpy(), axis=1))
        seen = set()
        for j in nearest:
            key = tuple(np.round(q_values[j], 3))
            if key in seen:
                continue
            seen.add(key)
            q = torch.as_tensor(q_values[j], dtype=start.dtype, device=start.device)
            receipt = confirm_path(context, start, worlds[index], q)
            recovered.append({'candidate': index, **receipt})
            if receipt['stage'] == 'complete_plan':
                break
        if recovered[-1]['stage'] == 'complete_plan':
            break

    rows = []
    for i in range(len(worlds)):
        outcome = ('scene_clear' if safe[i] or original[i] is not None
                   else 'found_ik_but_collision' if kinematic[i]
                   else 'full_pose_ik_not_found')
        rows.append({'candidate': i, 'outcome': outcome, 'original_replay_clear': original[i] is not None,
                     'full_pose_solutions': int(solutions['cpu_pose_valid'][i].sum()),
                     'open_scene_valid_solutions': int(solutions['open_scene_clear'][i].sum()),
                     'all_found_solutions_hit_table': bool(blocked_by_table_in_all_found[i]),
                     'collision_with_table_and_ground_clear': bool(collision_blocked[i] and bounds_and_floor[i].any()),
                     'position_only_solution_found': bool(only_position[i])})
    result = {**case, 'counts': dict(Counter(row['outcome'] for row in rows)),
              'all_found_solutions_hit_table': int(blocked_by_table_in_all_found.sum()),
              'position_only_solution_found': int(only_position.sum()),
              'original_replay_clear': sum(q is not None for q in original),
              'full_plan_recovered': any(r['stage'] == 'complete_plan' for r in recovered),
              'recovered_endpoint_checks': recovered, 'candidates_detail': rows}
    save(destination / 'audit.json', result)
    return result


def run(out):
    import numpy as np
    import resident_planner
    import ik_workspace_audit as helper
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner
    from scene_tools import ArmFK

    sys.path.insert(0, str(CODE / 'tmp'))
    from physical_table_pair_20261118_01 import install_collision_policy

    registration = register(out)
    started = time.perf_counter()
    install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner
    pool = FastPlannerPool(out / 'planner_warmup', Path(registration['warmup_trial']))
    context = pool.contexts['open']
    assert context.closed_checker is None
    full_pose = make_pure(context.model)
    position_only = make_pure(context.model, position_only=True)
    fk = ArmFK()
    helper.probe(pool, full_pose, position_only, out)
    save(out / 'ready.json', {'model_load_counts': pool.loads, 'full_pose_ik_loads': 1,
                             'position_only_ik_loads': 1, 'startup_seconds': time.perf_counter() - started})
    results, counts = [], Counter()
    for case in registration['cases']:
        result = audit_case(pool, full_pose, position_only, fk, case, out)
        counts.update(result['counts'])
        results.append(result)
        save(out / 'progress.json', {'completed': len(results), 'total': len(CASE_PREFIXES)})
        print('IK AUDIT', case['id'], result['counts'], 'full plan', result['full_plan_recovered'], flush=True)
    assert sum(counts.values()) == sum(c['candidates'] for c in registration['cases'])
    assert all(sha(path) == digest for path, digest in registration['source_sha256'].items())
    save(out / 'summary.json', {'completed': len(results), 'candidates': sum(counts.values()),
         'physical_attempts': 0, 'counts': dict(counts),
         'all_found_solutions_hit_table': sum(r['all_found_solutions_hit_table'] for r in results),
         'position_only_solution_found': sum(r['position_only_solution_found'] for r in results),
         'full_plan_recovered': sum(r['full_plan_recovered'] for r in results),
         'source_evidence_verified': True, 'wall_seconds': time.perf_counter() - started,
         'cases': [{k: v for k, v in r.items() if k not in ('candidates_detail', 'recovered_endpoint_checks')}
                   for r in results]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output.resolve())
