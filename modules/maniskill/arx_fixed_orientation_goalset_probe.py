"""Probe same-orientation grasp endpoints in a 3 mm region on saved ARX scenes."""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from arx_mpc_approach import FRAME, TCP_M, CONTROL_HZ, POSITION_TOLERANCE_M
from arx_mpc_guarded_bench import BoundedApproach
from resident_backend import save
from curobo.model_predictive_control import ModelPredictiveControlCfg
from curobo.types import GoalToolPose, JointState

CODE = Path(__file__).resolve().parent
SAVED = CODE / 'outputs/arx_graspcl_relaxed_pose_20261118_01'
CASES = ('I0003', 'I0015', 'I0038', 'I0044', 'I0031', 'I0125')
RADIUS = .003
GRID_STEP = .0015
ROTATION_TOLERANCE = .03
MAX_CANDIDATES = 6
OFFSETS = [np.asarray(offset) * GRID_STEP for offset in
           sorted(product((-1, 0, 1), repeat=3), key=lambda point: sum(x*x for x in point))]


def read(path):
    return json.loads(path.read_text())


def world_fk(context, joints):
    q = joints.reshape(-1, 6)
    poses = context.planner.compute_kinematics(JointState.from_position(
        q.clone(), joint_names=context.planner.joint_names)).tool_poses
    xyz = poses.position.detach().cpu().numpy().reshape(-1, 3)
    quat = poses.quaternion.detach().cpu().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]
    world = np.repeat(np.eye(4)[None], len(q), axis=0)
    world[:, :3, :3] = context.base[:3, :3] @ Rotation.from_quat(quat).as_matrix()
    world[:, :3, 3] = xyz @ context.base[:3, :3].T + context.base[:3, 3]
    return world


def tcp(world):
    return world[..., :3, 3] + TCP_M * world[..., :3, 2]


def find_endpoints(context, start, target, nominal):
    q_goal = context.ik(target, start)
    if q_goal is None:
        return 'grasp_ik_or_collision', None
    actual = world_fk(context, q_goal)[0]
    error = np.linalg.norm(tcp(actual) - tcp(nominal))
    angle = Rotation.from_matrix(actual[:3, :3] @ nominal[:3, :3].T).magnitude()
    if error > RADIUS or angle > ROTATION_TOLERANCE:
        return 'outside_position_or_rotation_bound', None
    pre = target.copy()
    pre[:3, 3] -= .08 * target[:3, 2]
    lift = target.copy()
    lift[2, 3] += .03
    q_pre = context.ik(pre, q_goal)
    q_lift = context.ik(lift, q_goal)
    if q_pre is None or q_lift is None:
        return 'pregrasp_or_lift_ik_or_collision', None
    return 'endpoints_pass', (q_pre, q_goal)


def mpc_goals(bot, context, measured, poses):
    # Keep the native goalset shape fixed so it can reuse its CUDA graphs.
    repeated = [*poses, *([poses[0]] * (len(OFFSETS) - len(poses)))]
    local = context.invbase @ np.asarray(repeated)
    rotations = Rotation.from_matrix(local[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    tensor = lambda values: torch.as_tensor(values.copy(), device='cuda', dtype=torch.float32)
    goal = GoalToolPose(tool_frames=[FRAME],
                        position=tensor(local[:, :3, 3]).reshape(1, 1, 1, len(OFFSETS), 3),
                        quaternion=tensor(rotations).reshape(1, 1, 1, len(OFFSETS), 4))
    bot.context = context
    bot.solver.reset_robot(measured.clone())
    assert bot.solver.update_goal_tool_poses(goal, run_ik=False, use_ik_goal=False)


def roll(bot, context, q_start, targets, nominal, region):
    q = q_start.detach().cpu().numpy().reshape(6)
    previous = q.copy()
    measured = bot.joint_state(q, np.zeros(6))
    mpc_goals(bot, context, measured, targets)
    trace = []
    within = 0
    for step in range(80):
        velocity = (q - previous) * CONTROL_HZ
        measured = bot.joint_state(q, velocity)
        actual = world_fk(context, measured.position)[0]
        original_error = float(np.linalg.norm(tcp(actual) - tcp(nominal)))
        target_error = min(float(np.linalg.norm(tcp(actual) - tcp(target))) for target in targets)
        angle = float(Rotation.from_matrix(actual[:3, :3] @ nominal[:3, :3].T).magnitude())
        close = (original_error <= (RADIUS if region else POSITION_TOLERANCE_M)
                 and target_error <= POSITION_TOLERANCE_M and angle <= ROTATION_TOLERANCE
                 and max(map(abs, velocity)) <= .035)
        within = within + 1 if close else 0
        row = {'step': step, 'tcp_displacement_m': original_error, 'nearest_goal_m': target_error,
               'orientation_error_rad': angle, 'q': q.tolist()}
        trace.append(row)
        if within >= 2:
            return {'settled': True, 'trajectory': trace, 'endpoint': row}
        response = bot.solve(measured)
        if not response['success']:
            return {'settled': False, 'reason': response['reason'], 'trajectory': trace, 'endpoint': row}
        row.update(solver_seconds=response['solver_seconds'],
                   collision_samples=response['collision_checked_samples'])
        for position in response['positions']:
            previous = q
            q = np.asarray(position)
    return {'settled': False, 'reason': 'solve_limit', 'trajectory': trace, 'endpoint': trace[-1]}


def probe_case(pool, bot, case, destination):
    trial = Path(case['original_trial'])
    state = read(trial / 'state.json')
    context = pool.bind(destination, state)
    start = context.measured(state)
    candidates = read(trial / 'graspcl/candidate_pool.json')['candidates'][:MAX_CANDIDATES]
    rows, rollouts = [], []
    for index, candidate in enumerate(candidates):
        contact = np.asarray(candidate['pose'])
        nominal = contact.copy()
        nominal[:3, 3] -= TCP_M * nominal[:3, 2]
        reason, baseline = find_endpoints(context, start, nominal, nominal)
        row = {'index': index, 'fixed': reason, 'attempted_offsets': 0, 'region_endpoints_pass': False}
        rows.append(row)
        if baseline is not None:
            continue
        targets = []
        local_start = None
        for offset in OFFSETS[1:]:
            shifted = nominal.copy()
            shifted[:3, 3] += offset
            row['attempted_offsets'] += 1
            _, endpoints = find_endpoints(context, start, shifted, nominal)
            if endpoints is None:
                continue
            q_pre, q_goal = endpoints
            row['region_endpoints_pass'] = True
            if local_start is None:
                reach = context.path(start, q_pre)
                approach = context.path(q_pre, q_goal) if reach is not None else None
                if approach is None:
                    continue
                distances = np.linalg.norm(tcp(world_fk(context, approach)) - tcp(shifted), axis=1)
                start_index = int(np.argmin(np.abs(distances - .008)))
                if not .004 <= distances[start_index] <= .015:
                    continue
                local_start = approach[start_index].clone()
            targets.append(shifted)
            if len(targets) == 4:
                break
        if local_start is not None:
            fixed = roll(bot, context, local_start, [nominal], nominal, False)
            region = roll(bot, context, local_start, targets, nominal, True)
            rollouts.append({'candidate_index': index, 'fixed': fixed, 'region': region,
                             'nominal_goal': nominal.tolist(), 'goals': [pose.tolist() for pose in targets],
                             'goal_offset_m': [float(np.linalg.norm(tcp(pose) - tcp(nominal))) for pose in targets]})
            if region['settled']:
                break
    return {'id': case['canonical_case_id'], 'group': case['original_group'],
            'candidates': rows, 'rollouts': rollouts}


def run(out):
    import resident_planner
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner

    sys.path.insert(0, str(CODE / 'tmp'))
    from physical_table_pair_20261118_01 import install_collision_policy

    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    selected = read(SAVED / 'selection.json')
    cases = [row for row in selected['cases'] if row['canonical_case_id'].split('_')[0] in CASES]
    assert len(cases) == len(CASES)
    save(out / 'registration.json', {'cases': cases, 'candidate_limit': MAX_CANDIDATES,
          'radius_m': RADIUS, 'offsets_m': [x.tolist() for x in OFFSETS],
          'orientation_tolerance_rad': ROTATION_TOLERANCE,
          'scope': 'Same target orientation and unchanged open ARX scene checker. Saved states and candidates, kinematic MPC rollout only. No closure or lift executed.'})
    install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner
    start = time.perf_counter()
    warmup = Path(read(SAVED / 'registration.json')['warmup_trial'])
    pool = FastPlannerPool(out / 'planner_warmup', warmup)
    context = pool.contexts['open']
    assert context.closed_checker is None
    create = ModelPredictiveControlCfg.create
    with patch.object(ModelPredictiveControlCfg, 'create',
                      lambda **kwargs: create(max_goalset=len(OFFSETS), **kwargs)):
        bot = BoundedApproach(context, read(warmup / 'state.json'))
    pool.parameters['open'].extend(resident_planner.all_kinematics_params(bot.solver))
    save(out / 'ready.json', {'model_loads': pool.loads, 'goalset_size': len(OFFSETS),
                             'startup_seconds': time.perf_counter() - start})
    results = []
    for case in cases:
        started = time.perf_counter()
        result = probe_case(pool, bot, case, out / case['canonical_case_id'])
        result['seconds'] = time.perf_counter() - started
        save(out / (case['canonical_case_id'] + '.json'), result)
        results.append(result)
        print(case['canonical_case_id'], 'nominal endpoint passes',
              sum(row['fixed'] == 'endpoints_pass' for row in result['candidates']),
              'bounded endpoint passes', sum(row['region_endpoints_pass'] for row in result['candidates']),
              'MPC settles', sum(row['region']['settled'] for row in result['rollouts']), flush=True)
    save(out / 'summary.json', {'completed': len(results),
          'cases_with_new_valid_endpoints': [r['id'] for r in results if any(c['region_endpoints_pass'] for c in r['candidates'])],
          'cases_with_new_mpc_settled_endpoint': [r['id'] for r in results if any(
              c['region']['settled'] and not c['fixed']['settled'] for c in r['rollouts'])]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
