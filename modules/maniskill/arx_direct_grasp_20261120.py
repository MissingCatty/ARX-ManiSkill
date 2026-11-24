"""One continuous grasp route, with a checked terminal approach and no pregrasp goal."""
from contextlib import contextmanager
from copy import deepcopy
import inspect
from pathlib import Path
import time
import types
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import graspcl_pipeline as common

TERMINAL_RADIUS_M = .020
TERMINAL_LATERAL_M = .004
TERMINAL_ORIENTATION_RAD = .10
TERMINAL_SPEED_M_S = .05
TARGET_CONTACT_REGION_M = .026
PROTOCOL = 'arx_direct_grasp_continuous_mpc_20261120_v1'


def world_poses(context, positions):
    import torch
    from curobo.types import JointState
    q = torch.as_tensor(np.asarray(positions), device='cuda', dtype=torch.float32).reshape(-1, 6)
    kin = context.planner.compute_kinematics(JointState.from_position(q, joint_names=context.planner.joint_names)).tool_poses
    xyz = kin.position.detach().cpu().numpy().reshape(-1, 3)
    quat = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)
    rot = context.base[:3, :3] @ Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    xyz = xyz @ context.base[:3, :3].T + context.base[:3, 3]
    return xyz, rot


def terminal_geometry(xyz, rot, goal):
    tcp = xyz + .043*rot[:, :, 2]
    goal_tcp = goal[:3, 3] + .043*goal[:3, 2]
    # Measure the approach relative to the actual terminal FK, retaining a
    # separate strict check that this endpoint reaches the requested pose.
    local = (tcp-tcp[-1]) @ goal[:3, :3]
    outside = np.flatnonzero(np.linalg.norm(local, axis=1) > TERMINAL_RADIUS_M)
    start = int(outside[-1])+1 if len(outside) else 0
    tail = local[start:]
    angles = Rotation.from_matrix(rot[start:] @ goal[:3, :3].T).magnitude()
    lateral = float(np.linalg.norm(tail[:, :2], axis=1).max())
    axial_span = float(-tail[:, 2].min())
    backwards = float(np.maximum(-np.diff(tail[:, 2]), 0.).max(initial=0.))
    endpoint_position = float(np.linalg.norm(tcp[-1]-goal_tcp))
    endpoint_rotation = float(Rotation.from_matrix(rot[-1] @ goal[:3, :3].T).magnitude())
    passed = (len(tail) >= 2 and axial_span >= .002 and lateral <= TERMINAL_LATERAL_M
              and float(angles.max()) <= TERMINAL_ORIENTATION_RAD
              and float(tail[:, 2].max()) <= .001 and backwards <= .0015
              and endpoint_position <= .0025 and endpoint_rotation <= .04)
    return {'passed': bool(passed), 'terminal_start_sample': start, 'terminal_samples': len(tail),
            'terminal_axial_span_m': axial_span, 'terminal_max_lateral_m': lateral,
            'terminal_max_orientation_rad': float(angles.max()), 'terminal_max_backtracking_m': backwards,
            'endpoint_tcp_error_m': endpoint_position, 'endpoint_rotation_error_rad': endpoint_rotation,
            'fixed_pregrasp_goal': False}


def inspect_route(context, route, goal):
    q = np.asarray(route['q']);times = np.asarray(route['times'])
    alpha = (np.arange(4)/4)[None, :, None]
    dense = np.vstack([((1-alpha)*q[:-1, None]+alpha*q[1:, None]).reshape(-1, 6), q[-1]])
    xyz, rot = world_poses(context, dense)
    receipt = terminal_geometry(xyz, rot, goal)
    tcp = xyz+.043*rot[:, :, 2]
    stamps = np.r_[((1-np.arange(4)/4)[None, :]*times[:-1, None]
                   +(np.arange(4)/4)[None, :]*times[1:, None]).reshape(-1), times[-1]]
    start = receipt['terminal_start_sample']
    speeds = np.linalg.norm(np.diff(tcp, axis=0), axis=1)/np.diff(stamps)
    peak = float(speeds[max(0, start-1):].max(initial=0.))
    joint_speed = np.abs(np.diff(dense, axis=0)/np.diff(stamps)[:, None])
    near_joint_speed = float(joint_speed[max(0, start-1):].max(initial=0.))
    scale = max(1., peak/TERMINAL_SPEED_M_S, near_joint_speed/.065)
    route = deepcopy(route);route['times'] = (times*scale).tolist()
    receipt.update(terminal_speed_before_retime_m_s=peak, terminal_speed_after_retime_m_s=peak/scale,
                   continuous_time_scale=scale, collision_checked_dense_samples=len(dense))
    return route, receipt, dense, tcp


def make_continuous_mpc(base, smooth):
    class ContinuousMPC(base):
        def begin_route(self, directory, context, phase, path, q, qd):
            reg = common.read(Path(common.read(directory/'source_guard.json')['registration']))
            self.direct_mode = bool(reg['direct_grasp'] and phase == 'approach')
            if not self.direct_mode:
                return super().begin_route(directory, context, phase, path, q, qd)
            # This internal phase bypasses the old 26 mm truncation. The solver
            # tracks one reference all the way to the actual grasp endpoint.
            result = smooth.SmoothRouteMPC.begin_route(self, directory, context, 'continuous_full', path, q, qd)
            if result['success']:
                self.route_stage = 'approach'
                self.nominal = np.asarray(common.read(directory/'plan_result.json')['selected']['grasp_world'])
                self.visual_delta = np.zeros(6)
                self.visual_goal = self.nominal.copy()
            return result

        def route_step(self, directory, phase, q, qd, goal_world=None):
            if not getattr(self, 'direct_mode', False):
                return super().route_step(directory, phase, q, qd)
            import torch
            if directory != self.directory or phase != self.route_stage:
                raise RuntimeError('MPC session does not own this continuous route')
            goal = self.nominal if goal_world is None else np.asarray(goal_world)
            if (not np.array_equal(goal[:3, :3], self.nominal[:3, :3])
                    or np.linalg.norm(goal[:3, 3]-self.nominal[:3, 3]) > .006+1e-9):
                raise ValueError('Continuous visual correction exceeds the fixed-orientation 6 mm region')
            if np.linalg.norm(goal[:3, 3]-self.visual_goal[:3, 3]) >= .00025:
                self.visual_goal = goal.copy()
                seed = torch.as_tensor(self.route.positions[-1], device='cuda', dtype=torch.float32)
                solution = self.context.ik(goal, seed)
                if solution is not None:
                    delta = solution.reshape(6).cpu().numpy()-self.route.positions[-1]
                    if np.abs(delta).max() <= .035:
                        self.visual_delta = delta
            xyz, rotation = world_poses(self.context, np.asarray(q)[None])
            tcp = xyz[0]+.043*rotation[0, :, 2]
            target_tcp = goal[:3, 3]+.043*goal[:3, 2]
            distance = float(np.linalg.norm(tcp-target_tcp))
            rotation_error = float(Rotation.from_matrix(rotation[0] @ goal[:3, :3].T).magnitude())
            state = self.stream;duration = self.route.duration
            if (state.time >= duration-1e-8 and distance <= .0025
                    and rotation_error <= np.deg2rad(5) and max(map(abs, qd)) < .035):
                return {'success': True, 'settled': True, 'joint_error_rad': float(np.abs(np.asarray(q)-self.route.positions[-1]-self.visual_delta).max()),
                        'reference_progress_s': duration, 'tool_position_error_m': distance}
            measured = self.joint_state(q, qd)
            target = self.curve(min(duration, state.time+.2))+self.visual_delta
            self.solver.update_goal_state(self.joint_state(target, np.zeros(6)))
            prediction = self.solver.optimize_action_sequence(measured.clone())
            if prediction.action_sequence is None:
                return {'success': False, 'reason': 'continuous_mpc_invalid_action_sequence'}
            predicted = prediction.action_sequence.position.detach().cpu().numpy().reshape(-1, 6)
            tracking_error = float(np.abs(np.asarray(q)-state.last_command).max())
            nominal_end = self.curve(min(duration, state.time+.1))+self.visual_delta
            correction = np.clip(predicted[3]-nominal_end, -.012, .012)
            correction *= .65*np.clip(tracking_error/.015, 0., 1.)
            taper = np.clip((duration-state.time)/.4, 0., 1.)
            x = float(np.clip((distance-.015)/.04, 0., 1.));x = x*x*(3-2*x)
            speed_limit = .075+(.65-.075)*x
            derivative = float(np.abs(self.curve(min(duration, state.time+.1), 1)).max())
            pace = min(1., speed_limit/max(derivative, 1e-8))
            pace *= float(np.clip(1.-max(0., tracking_error-.02)/.06, .2, 1.))
            for feedback_scale, pace_scale in [(1., 1.), (.5, .7), (0., .4)]:
                after = deepcopy(state);commands = []
                wanted = self.visual_delta+feedback_scale*correction*taper
                for _ in range(4):
                    after.pace += float(np.clip(pace*pace_scale-after.pace, -.6*.025, .6*.025))
                    after.time = min(duration, after.time+.025*after.pace)
                    acceleration = np.clip(25*(wanted-after.offset)-10*after.velocity, -.3, .3)
                    after.acceleration += np.clip(acceleration-after.acceleration, -4*.025, 4*.025)
                    after.velocity = np.clip(after.velocity+.025*after.acceleration, -.035, .035)
                    after.offset += .025*after.velocity
                    commands.append(self.curve(after.time)+after.offset)
                commands = np.asarray(commands)
                speed = float(np.abs(np.diff(np.vstack([state.last_command, commands]), axis=0)).max())/.025
                if speed > max(speed_limit, float(np.abs(qd).max())+.02):
                    continue
                points = np.vstack([q, commands]);alpha = (np.arange(5)/5)[None, :, None]
                dense = np.vstack([((1-alpha)*points[:-1, None]+alpha*points[1:, None]).reshape(-1, 6), commands[-1]])
                tensor = torch.as_tensor(dense, device='cuda', dtype=torch.float32)
                if not bool(self.context.checker.validate(tensor[:, None]).all()):
                    continue
                after.last_command = commands[-1];self.stream = after
                return {'success': True, 'settled': False, 'positions': commands.tolist(),
                        'solver_seconds': float(prediction.solve_time), 'collision_checked_samples': len(dense),
                        'reference_progress_s': after.time, 'reference_pace': after.pace,
                        'joint_error_rad': float(np.abs(np.asarray(q)-self.route.positions[-1]-self.visual_delta).max()),
                        'projected_max_joint_speed_rad_s': speed, 'visual_joint_offset_rad': self.visual_delta.tolist(),
                        'continuous_reference': True}
            return {'success': False, 'reason': 'continuous_mpc_no_safe_interval'}
    return ContinuousMPC


def reverse_approach_seed(context, q_goal, goal, length):
    """Differential-kinematics seed only: no extra Cartesian IK target or stop."""
    import torch
    q = q_goal.detach().cpu().numpy().copy();reverse = [q.copy()]
    covered = 0.;axis = goal[:3, 2]
    for _ in range(int(np.ceil(length/.001))):
        probes = np.vstack([q, q+np.eye(6)*.001])
        xyz, rot = world_poses(context, probes)
        dp = (xyz[1:]-xyz[0]).T/.001
        dr = Rotation.from_matrix(rot[1:] @ rot[0].T).as_rotvec().T/.001
        jac = np.vstack([dp, .2*dr])
        desired = np.r_[-.001*axis, np.zeros(3)]
        delta = np.linalg.solve(jac.T@jac+np.eye(6)*1e-6, jac.T@desired)
        if not np.isfinite(delta).all() or np.abs(delta).max() > .06:
            break
        candidate = q+delta
        tensor = torch.as_tensor(candidate, device='cuda', dtype=torch.float32)
        if not bool(context.checker.validate(tensor[None, None]).all()):
            break
        new_xyz, new_rot = world_poses(context, candidate[None])
        progress = float((xyz[0]-new_xyz[0]) @ axis)
        angle = float(Rotation.from_matrix(new_rot[0] @ goal[:3, :3].T).magnitude())
        if progress < .0002 or angle > .08:
            break
        covered += progress;q = candidate;reverse.append(q.copy())
    if covered < .003:
        return None
    return torch.as_tensor(np.asarray(reverse[::-1]), device='cuda', dtype=torch.float32)


def make_planner(base):
    class ContinuousPlanner(base):
        def select_tcp43(self, state, rows, camera):
            import torch
            directory = self.out.parent
            reg = common.read(Path(common.read(directory/'source_guard.json')['registration']))
            direct = reg['direct_grasp']
            pool = common.read(directory/'graspcl/candidate_pool.json')['candidates']
            start = self.measured(state)
            rejected = {k: 0 for k in ['grasp_ik', 'pregrasp_ik', 'lift_ik', 'path', 'optimizer',
                                       'terminal_approach', 'early_target_collision']}
            details = []
            for index, candidate in enumerate(pool):
                contact = np.asarray(candidate['pose']);grasp = contact.copy()
                grasp[:3, 3] -= .043*contact[:3, 2]
                lift = grasp.copy();lift[2, 3] += .03
                qg = self.ik(grasp, start)
                if qg is None:
                    rejected['grasp_ik'] += 1;continue
                if direct:
                    qp = None
                else:
                    pre = grasp.copy();pre[:3, 3] -= .08*contact[:3, 2]
                    qp = self.ik(pre, qg)
                    if qp is None:
                        rejected['pregrasp_ik'] += 1;continue
                ql = self.ik(lift, qg)
                if ql is None:
                    rejected['lift_ik'] += 1;continue
                selected = {'index': index, 'score': candidate['score'], 'candidate_metadata': candidate['metadata'],
                            'candidate_protocol': PROTOCOL, 'contact_pose_world': contact.tolist(),
                            'grasp_world': grasp.tolist(), 'lift_world': lift.tolist(),
                            'target_contact_world': contact[:3, 3].tolist(), 'table_clearance_raise_m': 0.,
                            'q_grasp': qg.tolist(), 'q_lift': ql.tolist(), 'fixed_pregrasp_goal': not direct}
                if not direct:
                    reach, approach = self.path(start, qp), self.path(qp, qg)
                    if reach is None or approach is None:
                        rejected['path'] += 1;continue
                    selected.update(q_pre=qp.tolist(), pregrasp_world=pre.tolist())
                    paths = {'pregrasp': self.optimize('pregrasp', reach, 6.),
                             'approach': self.optimize('approach', approach, 5.)}
                    proof = {'passed': True, 'fixed_pregrasp_goal': True, 'pregrasp_offset_m': .08}
                else:
                    route = self.path(start, qg)
                    if route is None:
                        rejected['path'] += 1;continue
                    paths = None
                    for seed_index, length in enumerate([0., .020, .010, .005]):
                        if length:
                            suffix = reverse_approach_seed(self, qg, grasp, length)
                            if suffix is None:continue
                            prefix = self.path(start, suffix[0])
                            if prefix is None:continue
                            route = torch.cat([prefix[:-1], suffix])
                        try:
                            optimized = self.optimize('continuous_'+str(index)+'_'+str(seed_index), route, 11.)
                        except RuntimeError as error:
                            rejected['optimizer'] += 1
                            details.append({'index': index, 'seed': seed_index, 'error': str(error)})
                            continue
                        optimized, proof, _, _ = inspect_route(self, optimized, grasp)
                        if proof['passed']:
                            proof.update(seed_kind='direct_global' if not length else 'differential_terminal_seed',
                                         terminal_seed_requested_m=length,
                                         terminal_seed_is_mandatory_stop=False)
                            paths = {'approach': optimized};break
                        details.append({'index': index, 'seed': seed_index, 'terminal_geometry': proof})
                    if paths is None:
                        rejected['terminal_approach'] += 1;continue
                # Use the already resident robot model to check target avoidance
                # outside the common final-contact region in both pilot arms.
                path = paths['approach']
                points = np.asarray(path['q'])
                if not direct:points = np.vstack([paths['pregrasp']['q'], points])
                alpha = (np.arange(4)/4)[None, :, None]
                dense = np.vstack([((1-alpha)*points[:-1, None]+alpha*points[1:, None]).reshape(-1, 6), points[-1]])
                xyz, rotation = world_poses(self, dense)
                tcp = xyz+.043*rotation[:, :, 2];target_tcp = grasp[:3, 3]+.043*grasp[:3, 2]
                prefix = dense[np.linalg.norm(tcp-target_tcp, axis=1)>TARGET_CONTACT_REGION_M]
                saved_out = self.out
                try:
                    checked = self._direct_pool.bind(directory/'early_target_collision_check', state, include_target=True)
                    tensor = torch.as_tensor(prefix, device='cuda', dtype=torch.float32)
                    clear = not len(prefix) or all(bool(checked.checker.validate(p[:, None]).all()) for p in tensor.split(64))
                finally:
                    self._direct_pool.bind(saved_out, state, include_target=False)
                if not clear:
                    rejected['early_target_collision'] += 1;continue
                proof.update(early_target_collision_check=True, early_target_checked_samples=len(prefix),
                             target_contact_region_m=TARGET_CONTACT_REGION_M,
                             direct_grasp=direct, extra_pregrasp_ik_calls=0 if direct else 1)
                common.save(directory/'continuous_grasp_plan.json', proof)
                common.save(directory/'continuous_candidate_diagnostics.json', details)
                return {'success': True, 'selected': selected, 'paths': paths, 'rejected': rejected,
                        'continuous_grasp_protocol': PROTOCOL}
            common.save(directory/'continuous_candidate_diagnostics.json', details)
            return {'success': False, 'reason': 'no_feasible_native_post_or_recalled_candidate',
                    'rejected': rejected, 'count': len(pool), 'continuous_grasp_protocol': PROTOCOL}
    return ContinuousPlanner


def install_resident():
    import short_lift_planner
    import table_truncation_model as truncation
    import arx_route_mpc_wrist_bench as wrist
    route = wrist.smooth.route
    route.guarded.install_variant()
    route.RouteMPC = make_continuous_mpc(wrist.WristRouteMPC, wrist.smooth)
    route.install_resident()
    short_lift_planner.ShortLiftPlanner = make_planner(short_lift_planner.ShortLiftPlanner)
    factory = truncation.make_pool

    def pool(base):
        class DirectPool(factory(base)):
            def request(self, directory, mpc=None, **kwargs):
                directory = Path(directory)
                reg = common.read(Path(common.read(directory/'source_guard.json')['registration'])) if (directory/'source_guard.json').exists() else None
                if mpc is None and not any(kwargs.get(k) for k in ['observe', 'approach', 'lift', 'refine']):
                    self.contexts['open']._direct_pool = self
                if reg and reg['direct_grasp'] and mpc in ['route_begin', 'begin']:
                    # Legacy route bindings use this filename. It now records
                    # the actual current state, not a reached pregrasp waypoint.
                    source = directory/'direct_handoff_state.json' if mpc == 'begin' else directory/'state.json'
                    common.save(directory/'pregrasp_state.json', common.read(source))
                return super().request(directory, mpc=mpc, **kwargs)
        return DirectPool
    truncation.make_pool = pool


@contextmanager
def execution_hook(run, direct):
    import arx_route_mpc as route
    original = run.trial;source = original.registered_short_lift_source
    observe = "execution.segment('observe',path['q'],path['times'],goal_world=entry['observation']['goal_world'])"
    assert source.count(observe) == 1
    source = source.replace(observe, "follow_registered_mpc_route(execution, 'observe', entry['observation']['goal_world'])")
    start = source.index('        path = plan["paths"]["pregrasp"]')
    end = source.index('        q = selected["q_grasp"]', start)
    if direct:
        replacement = '        selected = finish_continuous_mpc_grasp(execution, selected)\n'
    else:
        replacement = ('        follow_registered_mpc_route(execution, "pregrasp", selected["pregrasp_world"])\n'
                       '        save(out / "pregrasp_state.json", snapshot(env))\n'
                       '        selected = finish_registered_mpc_approach(execution, selected)\n')
    source = source[:start]+replacement+source[end:]
    namespace = dict(original.__globals__)
    exec(compile(source, __file__+':continuous_execution', 'exec'), namespace)
    updated = types.FunctionType(namespace['trial'].__code__, original.__globals__, original.__name__, original.__defaults__)
    updated.short_lift_protocol = original.short_lift_protocol

    def finish(execution, selected):
        from PIL import Image
        import json
        out = execution.out;robot = execution.env.agent.robot;indices = execution.arm_indices
        registration = Path(common.read(out/'source_guard.json')['registration'])
        endpoint = Path(common.read(registration.parent/'resident/backend_config.json')['sockets']['curobo'])
        nominal = np.asarray(selected['grasp_world']);goal = nominal.copy()
        state = common.read(out/'state.json')
        with Image.open(out/'camera/depth.png') as depth, Image.open(out/'camera/mask.png') as mask:
            tracker = route.mpc.TargetTranslation(route.mpc.target_cloud(np.asarray(depth), np.asarray(mask)>0,
                                                   state['intrinsics'], state['camera_world_optical']))
        stats = {'settled': False, 'mpc_solves': 0, 'commands': 0, 'fresh_wrist_frames': 0,
                 'tracked_frames': 0, 'global_prefix_ticks': 0, 'fixed_pregrasp_goal': False,
                 'continuous_from_initial': True, 'intermediate_handoff_stops': 0,
                 'maximum_observed_correction_m': 0., 'round_trip_seconds': 0., 'phase': 'approach'}
        execution.label = 'approach';started = time.perf_counter()
        def request(kind):
            began = time.perf_counter()
            reply = route.mpc.rpc(endpoint, {'directory': str(out), 'mpc': kind, 'phase': 'approach',
                'q': robot.qpos[0, indices].cpu().numpy().tolist(), 'qd': robot.qvel[0, indices].cpu().numpy().tolist(),
                **({'goal_world': goal.tolist()} if kind == 'route_step' else {})}, timeout=60)
            stats['round_trip_seconds'] += time.perf_counter()-began
            if reply['status'] != 'ok':
                common.save(out/'resident_request_error.json', reply)
                raise RuntimeError('resident_backend_failed: continuous_mpc')
            return reply['result']
        try:
            began = request('route_begin')
            if not began['success']:raise RuntimeError(began['reason'])
            stats['reference_seconds'] = began['reference_seconds']
            limit = 4*int(np.ceil(began['reference_seconds']/.1))+80
            with (out/'route_mpc_approach_steps.jsonl').open('x') as trace:
                for step in range(limit):
                    distance, _ = route.mpc.tool_errors(route.mpc.tool_transform(execution), nominal)
                    vision = None
                    if distance <= .030 and step%2 == 0:
                        stats['fresh_wrist_frames'] += 1
                        vision = tracker.update(route.mpc.wrist_cloud(execution.env))
                        stats['tracked_frames'] += vision['status'] == 'tracked'
                        goal[:3, 3] = nominal[:3, 3]+tracker.offset
                        stats['maximum_observed_correction_m'] = max(stats['maximum_observed_correction_m'], float(np.linalg.norm(tracker.offset)))
                    response = request('route_step')
                    trace.write(json.dumps({'step': step, 'vision': vision, **response})+'\n')
                    if not response['success']:raise RuntimeError(response['reason'])
                    if response['settled']:
                        if stats['mpc_solves'] >= 2 and stats['tracked_frames']:
                            stats['settled'] = True;break
                        continue
                    stats['mpc_solves'] += 1
                    for q in response['positions']:
                        execution.command[indices] = q;execution.command[execution.finger_indices] = .044
                        execution.step();stats['commands'] += 1
            if not stats['settled']:raise RuntimeError('continuous_mpc_not_converged')
            actual = route.mpc.tool_transform(execution)
            position_error, rotation_error = route.mpc.tool_errors(actual, goal)
            q = robot.qpos[0, indices].cpu().numpy().tolist();lift = actual.copy();lift[2, 3] += .03
            tcp = actual @ np.asarray(selected['T_eef_action'])
            contact = dict(q_grasp=q, grasp_world=actual.tolist(), lift_world=lift.tolist(),
                           action_tcp_world=tcp.tolist(), target_contact_world=tcp[:3, 3].tolist())
            common.save(out/'mpc_contact.json', {'selected': contact})
            execution.stages['approach'] = {'settled': True, 'joint_error_rad': 0.,
                'planned_seconds': stats['commands']/40., 'mpc': True, 'continuous_from_initial': True,
                'tool_position_error_m': position_error, 'tool_rotation_error_rad': rotation_error}
            common.save(out/'stages.json', execution.stages)
            stats.update(tool_position_error_m=position_error, tool_rotation_error_rad=rotation_error)
            return contact
        finally:
            stats['wall_seconds'] = time.perf_counter()-started
            common.save(out/'route_mpc_approach.json', stats);common.save(out/'mpc_approach.json', stats)

    with patch.object(run, 'trial', updated), \
         patch.object(run, 'follow_registered_mpc_route', route.follow_route, create=True), \
         patch.object(run, 'finish_registered_mpc_approach', route.finish_approach, create=True), \
         patch.object(run, 'finish_continuous_mpc_grasp', finish, create=True):
        yield
