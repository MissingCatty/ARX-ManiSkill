"""Native batched PRM and TrajOpt, sharing the current six-joint world."""
from types import SimpleNamespace
import time

import torch
import yaml

from arx_project.layout import ROOT
from .groups import padded_indices


class ParallelPathSession:
    def __init__(self, context, capacity):
        from curobo.batch_motion_planner import BatchMotionPlanner
        from curobo.motion_planner import MotionPlannerCfg
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_trajopt import TrajOptSolver
        from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
        from resident_planner import all_kinematics_params

        if capacity not in (4, 8):
            raise ValueError('Unsupported trajectory batch capacity')
        self.capacity = capacity
        self.scene = context.planner.scene_collision_checker
        shared_scene = self.scene

        class SharedWorldPlanner(BatchMotionPlanner):
            def _initialize_components(self):
                self.scene_collision_checker = shared_scene
                self.ik_solver = IKSolver(self.config.ik_solver_config, shared_scene)
                self.trajopt_solver = TrajOptSolver(self.config.trajopt_solver_config, shared_scene)
                self.graph_planner = PRMGraphPlanner(self.config.graph_planner_config, shared_scene)

        task = yaml.load((ROOT / 'modules/curobo/source/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml').read_text(), Loader=yaml.CSafeLoader)
        task['rollout']['cost_cfg']['cspace_cfg']['weight'][0] *= 100
        cfg = MotionPlannerCfg.create(robot=str(context.model), self_collision_check=False,
            num_ik_seeds=32, num_trajopt_seeds=4, max_batch_size=capacity, multi_env=False,
            position_tolerance=.002, orientation_tolerance=.03, interpolation_dt=.025,
            optimizer_collision_activation_distance=.003, trajopt_optimizer_configs=[task])
        for rollout in [cfg.graph_planner_config.rollout_config,
                        cfg.ik_solver_config.core_cfg.metrics_rollout_config,
                        cfg.trajopt_solver_config.core_cfg.metrics_rollout_config]:
            for cost in rollout.get_cost_manager_configs():
                cost.disable_self_collision()
        self.planner = SharedWorldPlanner(cfg)
        self.parameters = all_kinematics_params(self.planner.ik_solver,
            self.planner.trajopt_solver, self.planner.graph_planner)
        self.names = list(context.js(context.limits[:, 0]).joint_names)
        if self.planner.action_dim != 6 or self.names != self.planner.joint_names:
            raise RuntimeError('Parallel planning changed the six-joint chain')

    def synchronize_model(self, context):
        if context.planner.scene_collision_checker is not self.scene:
            raise RuntimeError('Parallel planner lost its shared collision world')
        source = context.checker.kinematics.kinematics_config
        if source.num_dof != 6 or source.link_spheres.shape[-2] != 871:
            raise RuntimeError('Parallel planning requires the original 871-sphere model')
        for target in self.parameters:
            target.copy_(source)

    def plan(self, context, jobs, *, minimum_duration=6.):
        """Solve all real rows together; padded rows never enter selection."""
        from curobo.types import JointState
        from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
        from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
        from optimize_path import _validate_trajectory

        if not jobs:
            return {}, {'real_rows': 0, 'GPU_calls': 0}
        widths = [float(j['width']) for j in jobs]
        if len(set(widths)) != 1:
            raise ValueError('One six-joint batch must have exactly equal finger locks')
        locks = context.checker.kinematics.kinematics_config.lock_jointstate
        actual = dict(zip(locks.joint_names, locks.position.detach().cpu().reshape(-1).tolist()))
        if any(abs(actual[n] - widths[0] / 2) > 2e-7 for n in ['right_joint27', 'right_joint28']):
            raise RuntimeError('Batch collision model has the wrong opening')
        indices = padded_indices(len(jobs), self.capacity)
        self.synchronize_model(context)
        start = torch.stack([jobs[i]['start'] for i in indices]).clone()
        goal = torch.stack([jobs[i]['q_pre'] for i in indices]).clone()
        self.planner.reset_seed()
        began = time.perf_counter()
        graph = self.planner.graph_planner.find_path(start.clone(), goal.clone(),
            interpolate_waypoints=True, interpolation_steps=201,
            interpolation_type=TrajInterpolationType.LINEAR, validate_interpolated_trajectory=True)
        graph_seconds = time.perf_counter() - began
        seeds = graph.interpolated_waypoints.reshape(self.capacity, 201, 6).clone()
        graph_ok = graph.success.reshape(self.capacity, -1).all(-1)
        horizon = self.planner.trajopt_solver.action_horizon
        seed = torch.nn.functional.interpolate(seeds.transpose(1, 2), size=horizon,
            mode='linear', align_corners=True).transpose(1, 2)[:, None].contiguous()
        # Match the accepted interactive first attempt exactly. Failed rows
        # retain the complete original per-candidate planning fallback.
        began = time.perf_counter()
        result = self.planner.trajopt_solver.solve_cspace(
            JointState.from_position(goal, joint_names=self.names),
            JointState.from_position(start, joint_names=self.names), seed_traj=seed,
            num_seeds=1, initial_iters=25, finetune_attempts=0, finetune_dt_scale=.75)
        torch.cuda.synchronize()
        solve_seconds = time.perf_counter() - began
        success = result.success.reshape(self.capacity, -1).all(-1)
        accepted, rows = {}, []
        began = time.perf_counter()
        for i, job in enumerate(jobs):
            row = {'index': job['index'], 'graph_success': bool(graph_ok[i]),
                   'optimizer_success': bool(success[i]), 'accepted': False}
            rows.append(row)
            if not graph_ok[i] or not success[i]:
                continue
            trajectory = result.interpolated_trajectory[i:i + 1].clone()
            last = int(result.interpolated_last_tstep.reshape(-1)[i])
            trajectory = trim_joint_state_trajectory(trajectory, 0, last)
            single = SimpleNamespace(get_interpolated_plan=lambda t=trajectory: t)
            try:
                q, times, checks = _validate_trajectory(single, context.checker, seeds[i],
                    self.names, context.limits, minimum_duration)
            except AssertionError as error:
                row['validation_failure'] = str(error)
                continue
            accepted[job['index']] = {'q': q.cpu().numpy().tolist(), 'times': times.tolist(),
                'planner': 'native_batched_PRM_and_TrajOpt', 'batch_capacity': self.capacity}
            row.update(accepted=True, **checks)
        return accepted, {'capacity': self.capacity, 'real_rows': len(jobs),
            'padded_rows': self.capacity - len(jobs), 'width_m': widths[0], 'rows': rows,
            'graph_seconds': graph_seconds, 'solve_seconds': solve_seconds,
            'validation_seconds': time.perf_counter() - began,
            'GPU_calls': {'PRM': 1, 'TrajOpt': 1}, 'initial_iters': 25,
            'trajectory_seeds_per_target': 1, 'finetune_attempts': 0,
            'native_finger_locks_m': {n: actual[n] for n in ['right_joint27', 'right_joint28']}}

    def destroy(self):
        self.planner.destroy()
