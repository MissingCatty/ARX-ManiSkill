"""Track a checked ARX route with receding-horizon control from the first motion."""
from contextlib import contextmanager
from pathlib import Path
import json
import time
import types
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import arx_mpc_approach as mpc
import arx_mpc_guarded_bench as guarded

ROUTE_SPEED_RAD_S = .65
WAYPOINT_TOLERANCE_RAD = .035
SETTLED_JOINT_ERROR_RAD = .005
SETTLED_SPEED_RAD_S = .04
ROUTE_CORRIDOR_RAD = .09
LOOKAHEAD_S = .2
HANDOFF_TCP_M = .026
ROUTE_NAMES = {'observe', 'pregrasp', 'approach'}


class ReferenceRoute:
    def __init__(self, positions, times):
        self.positions = np.asarray(positions, dtype=float)
        self.times = np.asarray(times, dtype=float)
        if (self.positions.shape != (len(self.times), 6) or len(self.times) < 2
                or not np.isfinite(self.positions).all() or not np.isfinite(self.times).all()
                or abs(self.times[0]) > 1e-8 or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Invalid registered route')
        self.duration = float(self.times[-1])
        self.progress = 0.

    def position(self, when):
        return np.array([np.interp(when, self.times, axis) for axis in self.positions.T])

    def next_target(self, q):
        # Advance the reference clock only while the measured arm is on track.
        if np.max(np.abs(q - self.position(self.progress))) <= WAYPOINT_TOLERANCE_RAD:
            self.progress = min(self.duration, self.progress + mpc.SOLVE_DT)
        return self.position(min(self.duration, self.progress + LOOKAHEAD_S))

    def corridor_error(self, positions):
        lo = max(0., self.progress - .4)
        hi = min(self.duration, self.progress + LOOKAHEAD_S + .1)
        saved = self.positions[(self.times >= lo) & (self.times <= hi)]
        reference = np.vstack([self.position(lo), saved, self.position(hi)])
        d = np.abs(np.asarray(positions)[:, None, :] - reference[None, :, :]).max(-1)
        return float(d.min(axis=1).max())


class RouteMPC(guarded.BoundedApproach):
    def __init__(self, context, state):
        super().__init__(context, state)
        current = context.measured(state).cpu().numpy()
        measured = self.joint_state(current, np.zeros(6))
        # Capture and warm joint mode before any scene is claimed.
        self.set_route_mode(measured)
        self.solver.update_goal_state(measured.clone())
        proof = self.solve(measured)
        assert proof['success'], proof
        self.set_local_mode()
        self.route_stage = None

    def set_route_mode(self, measured):
        self.solver.disable_tool_pose_tracking()
        self.solver.enable_joint_position_tracking()
        self.solver.reset_robot(measured)

    def begin(self, directory, context, q, qd):
        self.set_local_mode()
        self.route_stage = None
        return super().begin(directory, context, q, qd)

    def set_local_mode(self):
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        self.solver.disable_joint_position_tracking()
        self.solver.update_tool_pose_criteria({mpc.FRAME: ToolPoseCriteria.track_position_and_orientation(
            rpy=[1., 1., 1.], non_terminal_scale=.5)})

    def begin_route(self, directory, context, phase, route, q, qd):
        self.context = context
        points = np.asarray(route['q'])
        stamps = np.asarray(route['times'])
        if phase == 'approach':
            from curobo.types import JointState

            plan = mpc.read(directory / 'plan_result.json')
            goal = np.asarray(plan['selected']['grasp_world'])
            joints = self.torch.tensor(points, dtype=self.torch.float32, device='cuda')
            fk = self.solver.compute_kinematics(JointState.from_position(
                joints, joint_names=self.solver.joint_names)).tool_poses
            xyz = fk.position.cpu().numpy().reshape(-1, 3)
            quat = fk.quaternion.cpu().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]
            rot = self.context.base[:3, :3] @ Rotation.from_quat(quat).as_matrix()
            tcp = xyz @ self.context.base[:3, :3].T + self.context.base[:3, 3] + mpc.TCP_M * rot[:, :, 2]
            target_tcp = goal[:3, 3] + mpc.TCP_M * goal[:3, 2]
            distance = np.linalg.norm(tcp - target_tcp, axis=1)
            outside = np.flatnonzero(distance > HANDOFF_TCP_M)
            end = max(1, int(outside[-1]) + 1) if len(outside) else 1
            if end >= len(points):
                return {'success': False, 'reason': 'route_mpc_no_local_handoff'}
            points, stamps = points[:end + 1], stamps[:end + 1]
        self.route = ReferenceRoute(points, stamps)
        if np.max(np.abs(np.asarray(q) - self.route.positions[0])) >= SETTLED_JOINT_ERROR_RAD:
            return {'success': False, 'reason': 'route_mpc_start_mismatch'}
        measured = self.joint_state(q, qd)
        if not context.checker.validate(measured.position[:, None]).all():
            return {'success': False, 'reason': 'route_mpc_start_in_collision'}
        self.set_route_mode(measured)
        self.directory, self.route_stage = directory, phase
        return {'success': True, 'reference_seconds': self.route.duration}

    def route_step(self, directory, phase, q, qd):
        if self.directory != directory or self.route_stage != phase:
            raise RuntimeError('MPC session does not own this route')
        measured = self.joint_state(q, qd)
        error = float(np.max(np.abs(self.route.positions[-1] - q)))
        if (self.route.progress >= self.route.duration and error < SETTLED_JOINT_ERROR_RAD
                and max(map(abs, qd)) < SETTLED_SPEED_RAD_S):
            return {'success': True, 'settled': True, 'joint_error_rad': error,
                    'reference_progress_s': self.route.progress}
        target = self.route.next_target(np.asarray(q))
        self.solver.update_goal_state(self.joint_state(target, np.zeros(6)))
        response = self.checked_route_commands(measured, target)
        return {**response, 'settled': False, 'joint_error_rad': error,
                'reference_progress_s': self.route.progress, 'reference_q': target.tolist()}

    def checked_route_commands(self, measured, target):
        torch = self.torch
        result = self.solver.optimize_action_sequence(measured.clone())
        if result.action_sequence is None:
            return {'success': False, 'reason': 'route_mpc_invalid_action_sequence'}
        proposals = result.action_sequence.position.reshape(-1, 6)[:mpc.STEPS_PER_SOLVE]
        if proposals.shape[0] != mpc.STEPS_PER_SOLVE or not torch.isfinite(proposals).all():
            return {'success': False, 'reason': 'route_mpc_invalid_action_sequence'}
        start = measured.position
        waypoint = torch.as_tensor(target, device=start.device, dtype=start.dtype)
        # In joint tracking, retain MPC's direction of travel but prevent it
        # from crossing the current lookahead waypoint in any joint.
        projected = torch.maximum(torch.minimum(proposals, torch.maximum(start, waypoint)),
                                  torch.minimum(start, waypoint))
        change = torch.diff(torch.cat([start, projected]), dim=0).abs().max()
        proposed_speed = float(change) * mpc.CONTROL_HZ
        scale = min(1., ROUTE_SPEED_RAD_S / max(proposed_speed, 1e-9))
        alpha = torch.arange(5, device=start.device) / 5
        for _ in range(5):
            commands = start + scale * (projected - start)
            path = torch.cat([start, commands])
            corridor_error = self.route.corridor_error(path.cpu().numpy())
            dense = (path[:-1, None] * (1 - alpha[None, :, None])
                     + path[1:, None] * alpha[None, :, None]).reshape(-1, 6)
            dense = torch.cat([dense, commands[-1:]])
            if corridor_error <= ROUTE_CORRIDOR_RAD and self.context.checker.validate(dense[:, None]).all():
                return {'success': True, 'positions': commands.detach().cpu().tolist(),
                        'solver_seconds': float(result.solve_time), 'collision_checked_samples': len(dense),
                        'corridor_error_rad': corridor_error, 'command_scale': scale,
                        'projected_max_joint_speed_rad_s': proposed_speed * scale}
            scale *= .5
        return {'success': False, 'reason': 'route_mpc_no_safe_interval', 'corridor_error_rad': corridor_error}


def follow_route(execution, phase, goal_world=None):
    out = execution.out
    robot = execution.env.agent.robot
    indices = execution.arm_indices
    reg = Path(mpc.read(out / 'source_guard.json')['registration'])
    address = Path(mpc.read(reg.parent / 'resident/backend_config.json')['sockets']['curobo'])
    execution.label = phase

    def call(request):
        payload = {'directory': str(out), 'mpc': request, 'phase': phase,
                   'q': robot.qpos[0, indices].cpu().numpy().tolist(),
                   'qd': robot.qvel[0, indices].cpu().numpy().tolist()}
        started = time.perf_counter()
        try:
            reply = mpc.rpc(address, payload, timeout=60)
            if reply['status'] != 'ok':
                raise RuntimeError(reply.get('error', 'MPC resident request failed'))
        except Exception as exc:
            mpc.save(out / 'resident_request_error.json', {'kind': 'route_mpc', 'error': repr(exc)})
            raise RuntimeError('resident_backend_failed: route_mpc') from exc
        return reply['result'], reply['wall_seconds'], time.perf_counter() - started

    start = time.perf_counter()
    stats = {'phase': phase, 'settled': False, 'mpc_solves': 0, 'commands': 0,
             'gpu_solve_seconds': 0., 'backend_seconds': 0., 'round_trip_seconds': 0.}
    try:
        begin, backend, round_trip = call('route_begin')
        stats.update(backend_seconds=backend, round_trip_seconds=round_trip)
        if not begin['success']:
            raise RuntimeError(begin['reason'])
        reference_seconds = stats['reference_seconds'] = begin['reference_seconds']
        limit = 3 * int(np.ceil(reference_seconds / mpc.SOLVE_DT)) + 24
        with (out / f'route_mpc_{phase}_steps.jsonl').open('x') as trace:
            for i in range(limit):
                response, backend, round_trip = call('route_step')
                stats['backend_seconds'] += backend
                stats['round_trip_seconds'] += round_trip
                stats['joint_error_rad'] = response.get('joint_error_rad')
                trace.write(json.dumps({'step': i, 'backend_seconds': backend,
                                        'round_trip_seconds': round_trip, **response}) + '\n')
                if not response['success']:
                    raise RuntimeError(response['reason'])
                if response['settled']:
                    stats['settled'] = True
                    break
                stats['mpc_solves'] += 1
                stats['gpu_solve_seconds'] += response['solver_seconds']
                for q in response['positions']:
                    execution.command[indices] = q
                    execution.command[execution.finger_indices] = .044
                    execution.step()
                    stats['commands'] += 1
        if not stats['settled']:
            raise RuntimeError('route_mpc_' + phase + '_not_converged')
        if phase != 'approach':
            stage = {'settled': True, 'joint_error_rad': stats['joint_error_rad'],
                     'planned_seconds': stats['commands'] / mpc.CONTROL_HZ, 'mpc': True}
            if goal_world is not None:
                actual, goal = mpc.tool_transform(execution), np.asarray(goal_world)
                stage['tool_position_error_m'] = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
                stage['tool_rotation_error_rad'] = float(Rotation.from_matrix(
                    actual[:3, :3] @ goal[:3, :3].T).magnitude())
            execution.stages[phase] = stage
            mpc.save(out / 'stages.json', execution.stages)
        return stats
    finally:
        stats['wall_seconds'] = time.perf_counter() - start
        mpc.save(out / f'route_mpc_{phase}.json', stats)


def finish_approach(execution, selected):
    tracked = follow_route(execution, 'approach')
    q = execution.env.agent.robot.qpos[0, execution.arm_indices].cpu().numpy().tolist()
    # The existing wrist-feedback controller is handed a measured state already
    # inside the local region, so it issues no open-loop approach commands.
    position_error, _ = mpc.tool_errors(mpc.tool_transform(execution), np.asarray(selected['grasp_world']))
    if position_error > mpc.HANDOFF_M:
        raise RuntimeError('route_mpc_handoff_outside_local_region')
    contact = mpc.execute_approach(execution, {'q': [q, q], 'times': [0., .025]}, selected)
    execution.stages['approach']['planned_seconds'] += tracked['commands'] / mpc.CONTROL_HZ
    mpc.save(execution.out / 'stages.json', execution.stages)
    return contact


@contextmanager
def execution_hook(run):
    original = run.trial
    source = original.registered_short_lift_source
    replacements = {
        "execution.segment('observe',path['q'],path['times'],goal_world=entry['observation']['goal_world'])":
            "follow_registered_mpc_route(execution, 'observe', entry['observation']['goal_world'])",
        'execution.segment("pregrasp", path["q"], path["times"], goal_world=selected["pregrasp_world"])':
            'follow_registered_mpc_route(execution, "pregrasp", selected["pregrasp_world"])',
    }
    start = source.index('        worker("planner.py", out, planner_gpu, out / "approach_planner.log", approach=True)')
    end = source.index('        q = selected["q_grasp"]', start)
    # Preserve the native observation, grasp candidate planning and closed lift.
    source = source[:start] + '        selected = finish_registered_mpc_approach(execution, selected)\n' + source[end:]
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, str(Path(__file__)) + ':trial', 'exec'), namespace)
    trial = types.FunctionType(namespace['trial'].__code__, original.__globals__,
                               original.__name__, original.__defaults__)
    trial.short_lift_protocol = original.short_lift_protocol
    with patch.object(run, 'trial', trial), patch.object(run, 'follow_registered_mpc_route', follow_route, create=True), \
            patch.object(run, 'finish_registered_mpc_approach', finish_approach, create=True):
        yield


def install_resident():
    import table_truncation_model as truncation

    mpc.ResidentApproach = RouteMPC
    mpc.install_resident()
    make_pool = truncation.make_pool

    def with_route(base):
        class RoutedPool(make_pool(base)):
            def request(self, directory, mpc=None, **kwargs):
                if mpc == 'route_begin':
                    directory = Path(directory)
                    phase = kwargs['phase']
                    if phase not in ROUTE_NAMES:
                        raise ValueError('Unknown route stage')
                    observe = phase == 'observe'
                    plan = 'observation_plan_result.json' if observe else 'plan_result.json'
                    state = mpc_read(directory / ('pregrasp_state.json' if phase == 'approach' else 'state.json'))
                    context = self.bind(directory / f'{phase}_route_scene', state, include_target=observe)
                    path = mpc_read(directory / plan)['paths'][phase]
                    return self.local_mpc.begin_route(directory, context, phase, path, kwargs['q'], kwargs['qd'])
                if mpc == 'route_step':
                    return self.local_mpc.route_step(Path(directory), **kwargs)
                return super().request(directory, mpc=mpc, **kwargs)

        return RoutedPool

    mpc_read = mpc.read
    truncation.make_pool = with_route
