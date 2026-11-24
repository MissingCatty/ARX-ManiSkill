"""Final ManiSkill ARX approach with cuRobo MPC and wrist depth feedback."""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import textwrap
import time
import types
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from resident_backend import rpc, save

FRAME = 'right_eef_link'
CONTROL_HZ = 40
STEPS_PER_SOLVE = 4
SOLVE_DT = STEPS_PER_SOLVE / CONTROL_HZ
HANDOFF_M = 0.030
TCP_M = 0.043
POSITION_TOLERANCE_M = 0.0025
ORIENTATION_TOLERANCE_RAD = np.deg2rad(5)
MAX_CORRECTION_M = 0.006
MAX_SOLVES = 55


def read(path):
    return json.loads(Path(path).read_text())


def target_cloud(depth, mask, intrinsics, world_optical):
    depth = np.asarray(depth, dtype=float) * .001
    v, u = np.where(mask & (depth > .08) & (depth < 1.5))
    # Bound the registration workload without changing the camera or segmentation.
    stride = max(1, int(np.ceil(len(u) / 2400)))
    v, u = v[::stride], u[::stride]
    z = depth[v, u]
    k = np.asarray(intrinsics)
    points = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
    transform = np.asarray(world_optical)
    return points @ transform[:3, :3].T + transform[:3, 3]


class TargetTranslation:
    """Register the visible surface to the initial wrist observation in world coordinates."""

    def __init__(self, reference):
        if len(reference) < 50:
            raise RuntimeError('mpc_reference_target_not_visible')
        self.reference = reference
        self.tree = cKDTree(reference)
        self.offset = np.zeros(3)

    def update(self, points):
        if len(points) < 50:
            return {'status': 'too_few_target_points', 'target_points': len(points)}
        offset = self.offset.copy()
        for _ in range(5):
            distance, index = self.tree.query(points - offset)
            matching = distance < .008
            if np.count_nonzero(matching) < max(40, .55 * len(points)):
                return {'status': 'insufficient_surface_overlap', 'target_points': len(points),
                        'match_fraction': float(matching.mean())}
            residual = points[matching] - self.reference[index[matching]] - offset
            offset += np.median(residual, axis=0)
        magnitude = float(np.linalg.norm(offset))
        if magnitude > MAX_CORRECTION_M:
            return {'status': 'correction_outside_local_region', 'target_points': len(points),
                    'proposed_translation_m': offset.tolist()}
        self.offset = offset
        return {'status': 'tracked', 'target_points': len(points),
                'match_fraction': float(matching.mean()), 'translation_m': offset.tolist()}


def wrist_cloud(env):
    obs = env.get_obs()
    camera = obs['sensor_data'][env.perception_camera]
    params = obs['sensor_param'][env.perception_camera]
    segmentation = camera['segmentation'][0, :, :, 0].cpu().numpy()
    target = env.obj._objs[0].per_scene_id
    return target_cloud(camera['depth'][0, :, :, 0].cpu().numpy(), segmentation == target,
                        params['intrinsic_cv'][0].cpu().numpy(),
                        params['cam2world_gl'][0].cpu().numpy() @ np.diag([1, -1, -1, 1]))


def tool_transform(execution):
    return execution.env.agent.tcp.pose.to_transformation_matrix().cpu().numpy()[0].astype(float)


def tool_errors(actual, goal):
    actual_tcp = actual[:3, 3] + TCP_M * actual[:3, 2]
    goal_tcp = goal[:3, 3] + TCP_M * goal[:3, 2]
    return float(np.linalg.norm(actual_tcp - goal_tcp)), float(Rotation.from_matrix(
        actual[:3, :3] @ goal[:3, :3].T).magnitude())


def execute_approach(execution, path, selected):
    """Track the checked global prefix, then apply the first interval of each MPC solve."""
    from PIL import Image

    out = execution.out
    reg_path = Path(read(out / 'source_guard.json')['registration'])
    endpoint = Path(read(reg_path.parent / 'resident/backend_config.json')['sockets']['curobo'])
    # The registered arm directory links to the single suite-owned resident service.
    robot = execution.env.agent.robot
    indices = execution.arm_indices
    nominal = np.asarray(selected['grasp_world'])
    q, times = np.asarray(path['q']), np.asarray(path['times'])
    execution.label = 'approach'

    def measured():
        return {'q': robot.qpos[0, indices].cpu().numpy().tolist(),
                'qd': robot.qvel[0, indices].cpu().numpy().tolist()}

    def command(position):
        execution.command[indices] = position
        execution.command[execution.finger_indices] = .044
        execution.step()

    def request(kind, goal):
        try:
            reply = rpc(endpoint, {'directory': str(out), 'mpc': kind, **measured(),
                                   'goal_world': goal.tolist()}, timeout=60)
            if reply['status'] != 'ok':
                raise RuntimeError(reply.get('error', 'MPC service failed'))
        except Exception as error:
            save(out / 'resident_request_error.json', {'kind': 'curobo_mpc', 'error': repr(error)})
            raise RuntimeError('resident_backend_failed: curobo_mpc') from error
        return reply['result'], reply['wall_seconds']

    if np.max(np.abs(np.asarray(measured()['q']) - q[0])) >= .005:
        raise RuntimeError('approach_trajectory_start_mismatch')
    for k in range(int(np.ceil(times[-1] * CONTROL_HZ)) + 1):
        handoff_distance, _ = tool_errors(tool_transform(execution), nominal)
        if handoff_distance <= HANDOFF_M:
            break
        command([np.interp(k / CONTROL_HZ, times, q[:, j]) for j in range(6)])
    else:
        raise RuntimeError('mpc_handoff_not_reached')

    state = read(out / 'state.json')
    with Image.open(out / 'camera/depth.png') as depth, Image.open(out / 'camera/mask.png') as mask:
        tracker = TargetTranslation(target_cloud(np.asarray(depth), np.asarray(mask) > 0,
                                                state['intrinsics'], state['camera_world_optical']))
    began, _ = request('begin', nominal)
    if not began.get('success'):
        raise RuntimeError(began['reason'])
    stats = {'handoff_distance_m': handoff_distance, 'global_prefix_ticks': k, 'mpc_solves': 0,
             'commands': 0, 'fresh_wrist_frames': 0, 'tracked_frames': 0,
             'solve_period_sim_s': SOLVE_DT, 'settled': False, 'maximum_observed_correction_m': 0.0}
    goal = nominal.copy()
    try:
        with (out / 'mpc_steps.jsonl').open('x') as trace:
            settled_intervals = 0
            for step in range(MAX_SOLVES):
                # Feedback at 5 Hz in sim time, joint-state replanning at 10 Hz.
                vision = None
                if step % 2 == 0:
                    stats['fresh_wrist_frames'] += 1
                    vision = tracker.update(wrist_cloud(execution.env))
                    if vision['status'] == 'tracked':
                        stats['tracked_frames'] += 1
                        stats['maximum_observed_correction_m'] = max(
                            stats['maximum_observed_correction_m'], float(np.linalg.norm(tracker.offset)))
                    elif vision['status'] == 'correction_outside_local_region':
                        trace.write(json.dumps({'step': step, 'vision': vision}) + '\n')
                        raise RuntimeError('mpc_target_outside_local_region')
                    # A partial view near closure can lose registration. Retain the most recent
                    # trusted offset, which is bounded relative to the initial observed target.
                    goal[:3, 3] = nominal[:3, 3] + tracker.offset
                actual = tool_transform(execution)
                pos_err, rot_err = tool_errors(actual, goal)
                stopped = np.max(np.abs(measured()['qd'])) < .09
                settled_intervals = settled_intervals + 1 if (
                    pos_err <= POSITION_TOLERANCE_M and rot_err <= ORIENTATION_TOLERANCE_RAD and stopped
                ) else 0
                # Require at least two actual receding-horizon updates before closure.
                if settled_intervals >= 2 and stats['mpc_solves'] >= 2 and stats['tracked_frames']:
                    stats['settled'] = True
                    break
                response, elapsed = request('step', goal)
                stats['mpc_solves'] += 1
                trace.write(json.dumps({'step': step, 'vision': vision, 'position_error_m': pos_err,
                                        'rotation_error_rad': rot_err, 'rpc_wall_seconds': elapsed,
                                        **response}) + '\n')
                if not response['success']:
                    raise RuntimeError(response['reason'])
                for target in response['positions']:
                    command(target)
                    stats['commands'] += 1
        if not stats['settled']:
            raise RuntimeError('mpc_approach_not_converged')
        final = tool_transform(execution)
        pos_err, rot_err = tool_errors(final, goal)
        stats.update(tool_position_error_m=pos_err, tool_rotation_error_rad=rot_err)
        # Use the measured endpoint for the hold at closure and the subsequent +3 cm lift.
        q_end = measured()['q']
        lift = final.copy()
        lift[2, 3] += .03
        action_tcp = final @ np.asarray(selected['T_eef_action'])
        contact = dict(q_grasp=q_end, grasp_world=final.tolist(), lift_world=lift.tolist(),
                       action_tcp_world=action_tcp.tolist(), target_contact_world=action_tcp[:3, 3].tolist())
        save(out / 'mpc_contact.json', {'selected': contact})
        execution.stages['approach'] = {'settled': True, 'joint_error_rad': 0.0,
            'planned_seconds': k / CONTROL_HZ + stats['commands'] / CONTROL_HZ, 'mpc': True,
            'tool_position_error_m': pos_err, 'tool_rotation_error_rad': rot_err}
        save(out / 'stages.json', execution.stages)
        return contact
    finally:
        save(out / 'mpc_approach.json', stats)


@contextmanager
def execution_hook(run):
    """Install the MPC approach in this new benchmark process only."""
    original = run.trial
    source = original.registered_short_lift_source
    call = 'execution.segment("approach", path["q"], path["times"], goal_world=selected["grasp_world"])'
    assert source.count(call) == 1
    source = source.replace(call, 'selected = execute_mpc_approach(execution, path, selected)')
    namespace = dict(original.__globals__)
    exec(compile(source, __file__ + ':trial', 'exec'), namespace)
    updated = types.FunctionType(namespace['trial'].__code__, original.__globals__,
                                 original.__name__, original.__defaults__)
    updated.short_lift_protocol = original.short_lift_protocol
    with patch.object(run, 'trial', updated), patch.object(run, 'execute_mpc_approach', execute_approach, create=True):
        yield


class ResidentApproach:
    """One cuRobo MPC solver, sharing the resident open-arm collision world."""

    def __init__(self, context, state):
        import torch
        from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        self.torch = torch
        cfg = ModelPredictiveControlCfg.create(robot=str(context.model), scene_model=None,
            self_collision_check=False, optimization_dt=SOLVE_DT, interpolation_steps=STEPS_PER_SOLVE,
            optimizer_collision_activation_distance=.003, use_cuda_graph=True, num_seeds=1,
            num_control_points=8, warm_start_optimization_num_iters=25, cold_start_optimization_num_iters=75)
        # Match the frozen planner's metrics. This SDK applies the factory flag
        # to the optimizer rollouts, but the metrics rollout still defaults to
        # the SDK self-collision model. Its scene and joint checks stay enabled.
        for cost in cfg.core_cfg.metrics_rollout_config.get_cost_manager_configs():
            cost.disable_self_collision()
        self.solver = ModelPredictiveControl(cfg, context.planner.scene_collision_checker)
        assert self.solver.tool_frames == [FRAME]
        self.solver.update_tool_pose_criteria({FRAME: ToolPoseCriteria.track_position_and_orientation(
            rpy=[.4, .4, .4], non_terminal_scale=.5)})
        q = context.measured(state).cpu().numpy().tolist()
        measured = self.joint_state(q, [0.] * len(q))
        self.solver.setup(measured)
        self.context = context
        self.last_goal = None
        self.set_goal(np.asarray(state['links'][FRAME]))
        response = self.solve(measured)
        if not response['success']:
            raise RuntimeError('MPC startup validation failed: ' + response['reason'])
        self.directory = None

    def joint_state(self, position, velocity):
        from curobo.types import JointState

        torch = self.torch
        values = np.asarray([position, velocity], dtype=np.float32)
        if values.shape != (2, 6) or not np.isfinite(values).all():
            raise ValueError('Invalid measured ARX joint state')
        q, qd = torch.as_tensor(values, device='cuda').unbind(0)
        state = JointState.from_position(q[None].clone(), joint_names=self.solver.joint_names)
        state.velocity = qd[None].clone()
        state.acceleration = torch.zeros_like(state.position)
        return state

    def set_goal(self, world):
        from curobo.types import GoalToolPose

        if self.last_goal is not None and np.array_equal(self.last_goal, world):
            return
        local = self.context.invbase @ world
        quat = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
        torch = self.torch
        position = torch.tensor(local[:3, 3], dtype=torch.float32, device='cuda').reshape(1, 1, 1, 1, 3)
        orientation = torch.tensor(quat, dtype=torch.float32, device='cuda').reshape(1, 1, 1, 1, 4)
        goal = GoalToolPose(tool_frames=[FRAME], position=position.contiguous(), quaternion=orientation.contiguous())
        assert self.solver.update_goal_tool_poses(goal, run_ik=False, use_ik_goal=False)
        self.last_goal = world.copy()

    def solve(self, measured):
        torch = self.torch
        result = self.solver.optimize_action_sequence(measured.clone())
        commands = result.action_sequence.position.reshape(-1, 6)[:STEPS_PER_SOLVE]
        if commands.shape[0] != STEPS_PER_SOLVE or not torch.isfinite(commands).all():
            return {'success': False, 'reason': 'mpc_invalid_action_sequence'}
        path = torch.cat([measured.position, commands], dim=0)
        alpha = torch.arange(5, device=path.device) / 5
        dense = (path[:-1, None] * (1 - alpha[None, :, None]) + path[1:, None] * alpha[None, :, None]).reshape(-1, 6)
        dense = torch.cat([dense, commands[-1:]], dim=0)
        if not self.context.checker.validate(dense[:, None]).all():
            return {'success': False, 'reason': 'mpc_command_collision_or_joint_limit'}
        if not result.success.all():
            return {'success': False, 'reason': 'mpc_horizon_infeasible'}
        return {'success': True, 'positions': commands.detach().cpu().tolist(),
                'solver_seconds': float(result.solve_time), 'collision_checked_samples': len(dense)}

    def begin(self, directory, context, q, qd):
        self.context = context
        self.directory = directory
        self.last_goal = None
        self.nominal = np.asarray(read(directory / 'plan_result.json')['selected']['grasp_world'])
        measured = self.joint_state(q, qd)
        if not self.context.checker.validate(measured.position[:, None]).all():
            return {'success': False, 'reason': 'mpc_handoff_state_in_collision'}
        self.solver.reset_robot(measured)
        return {'success': True}

    def step(self, directory, q, qd, goal_world):
        if directory != self.directory:
            raise RuntimeError('MPC session does not own the requested trial')
        goal = np.asarray(goal_world)
        if (goal.shape != (4, 4) or not np.isfinite(goal).all()
                or np.linalg.norm(goal[:3, 3] - self.nominal[:3, 3]) > MAX_CORRECTION_M + 1e-9
                or not np.array_equal(goal[:3, :3], self.nominal[:3, :3])):
            raise ValueError('MPC goal outside the registered local correction region')
        self.set_goal(goal)
        return self.solve(self.joint_state(q, qd))


def install_resident():
    import resident_planner as resident
    import table_truncation_model as truncation

    request = resident.PlannerPool.request
    source = textwrap.dedent(inspect.getsource(request))
    old = "selected = json.loads((directory / 'plan_result.json').read_text())['selected']"
    new = "selected = json.loads((directory / ('mpc_contact.json' if lift else 'plan_result.json')).read_text())['selected']"
    assert source.count(old) == 1
    namespace = dict(request.__globals__)
    exec(compile(source.replace(old, new), __file__ + ':resident_lift', 'exec'), namespace)
    resident.PlannerPool.request = namespace['request']

    make_pool = truncation.make_pool

    def with_mpc(base):
        class MpcPool(make_pool(base)):
            def __init__(self, root, warmup_trial):
                super().__init__(root, warmup_trial)
                begin = time.perf_counter()
                self.local_mpc = ResidentApproach(self.contexts['open'], read(warmup_trial / 'state.json'))
                self.parameters['open'].extend(resident.all_kinematics_params(self.local_mpc.solver))
                save(root / 'mpc_ready.json', {'passed': True, 'model_load_count': 1,
                    'robot_trajectories_executed': 0, 'solver': 'cuRobo ModelPredictiveControl',
                    'warmup_seconds': time.perf_counter() - begin, 'solve_interval_s': SOLVE_DT})

            def request(self, directory, mpc=None, **kwargs):
                if mpc is None:
                    return super().request(directory, **kwargs)
                directory = Path(directory)
                if mpc == 'begin':
                    state = read(directory / 'pregrasp_state.json')
                    context = self.bind(directory / 'mpc_scene', state)
                    return self.local_mpc.begin(directory, context, kwargs['q'], kwargs['qd'])
                if mpc == 'step':
                    return self.local_mpc.step(directory, **kwargs)
                raise ValueError('Unknown MPC request')

        return MpcPool

    truncation.make_pool = with_mpc
