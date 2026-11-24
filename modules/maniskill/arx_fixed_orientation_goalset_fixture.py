"""Calibrate goalset MPC around a constructed boundary in the saved I0117 scene."""
import argparse
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from arx_fixed_orientation_goalset_probe import (
    BoundedApproach, CODE, ModelPredictiveControlCfg, OFFSETS, RADIUS, read, roll, save, tcp, world_fk,
)

REFERENCE = CODE / 'outputs/arx_graspcl_route_mpc_wrist_20261119_01/free_graspcl_mpc/trials'


def more_height(pose, metres):
    goal = pose.copy()
    goal[2, 3] += metres
    return goal


def boundary(context, target, q):
    # Lower the saved open-grasp pose in the actual table geometry. The first
    # solver rejection defines a local search interval; it is not a proof of IK infeasibility.
    last = q
    for millimetre in range(1, 81):
        lower = more_height(target, -millimetre * .001)
        result = context.ik(lower, last)
        if result is None:
            break
        last = result
    actual = world_fk(context, last)[0]
    safe = target.copy()
    safe[:3, 3] = actual[:3, 3]
    # A goal 2.9 mm below the validated endpoint lets the same scene
    # distinguish the original 2.5 mm arrival criterion from the 3 mm region.
    nominal = more_height(safe, -.0029)
    return nominal, safe, last, millimetre


def run(out):
    import resident_planner
    from short_lift_planner import ShortLiftPlanner

    sys.path.insert(0, str(CODE / 'tmp'))
    from physical_table_pair_20261118_01 import install_collision_policy

    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    trial, = REFERENCE.glob('I0117_*')
    state = read(trial / 'pregrasp_state.json')
    selected = read(trial / 'plan_result.json')['selected']
    started = time.perf_counter()
    install_collision_policy()
    with resident_planner.fixed_cache_capacity():
        context = ShortLiftPlanner(out / 'open', state)
    create = ModelPredictiveControlCfg.create
    with patch.object(ModelPredictiveControlCfg, 'create',
                      lambda **kwargs: create(max_goalset=len(OFFSETS), **kwargs)):
        bot = BoundedApproach(context, state)
    save(out / 'ready.json', {'model_loads': {'open': 1, 'mpc': 1},
                             'startup_seconds': time.perf_counter() - started})
    q_goal = torch.tensor(selected['q_grasp'], dtype=torch.float32, device='cuda')
    target = np.asarray(selected['grasp_world'])
    nominal, safe, q_safe, first_search_rejection_mm = boundary(context, target, q_goal)
    alternatives = [safe]
    for x in (-.0005, .0005):
        variant = safe.copy()
        variant[0, 3] += x
        assert np.linalg.norm(tcp(variant) - tcp(nominal)) < RADIUS
        if context.ik(variant, q_safe) is not None:
            alternatives.append(variant)
    q_start = context.ik(more_height(safe, .004), q_safe)
    if q_start is None:
        raise RuntimeError('Could not find a distinct collision-checked local start')
    initial_tcp_error = float(np.linalg.norm(tcp(world_fk(context, q_start)[0]) - tcp(nominal)))
    assert initial_tcp_error > RADIUS
    fixed = roll(bot, context, q_start, [nominal], nominal, False)
    region = roll(bot, context, q_start, alternatives, nominal, True)
    assert len(alternatives) > 1
    assert all(np.array_equal(goal[:3, :3], nominal[:3, :3]) for goal in alternatives)
    summary = {'scope': 'Constructed boundary target in the saved I0117 scene. Kinematic command rollout, with the original scene and joint checks. No physical grasp attempt.',
               'source_trial': str(trial), 'nominal_fixed_ik_found': context.ik(nominal, q_safe) is not None,
               'first_start_seed_search_rejection_mm': first_search_rejection_mm,
               'initial_tcp_error_m': initial_tcp_error, 'goal_count': len(alternatives),
               'nominal_goal': nominal.tolist(), 'alternative_goals': [g.tolist() for g in alternatives],
               'fixed': fixed, 'region': region}
    save(out / 'summary.json', summary)
    print('FIXTURE', {k: summary[k] for k in ('nominal_fixed_ik_found', 'goal_count', 'initial_tcp_error_m')},
          'fixed', fixed['settled'], fixed['endpoint'], 'region', region['settled'], region['endpoint'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
