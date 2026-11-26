"""Lazily solve ordered endpoint batches without changing candidate selection."""
from copy import deepcopy
from dataclasses import fields
import inspect
import time

import numpy as np
from scipy.spatial.transform import Rotation


def copy_optimizer_settings(current, original):
    """Keep numerical options while letting the new solver size its buffers."""
    for field in fields(original):
        if field.name in ['num_problems','debug_info']:
            continue
        value = getattr(original,field.name)
        # Device settings are immutable; numerical option containers must not
        # be shared with a live solver that may update its configuration.
        setattr(current,field.name,deepcopy(value) if isinstance(value,(list,dict)) else value)


def padded_worlds(worlds, capacity):
    """Return fixed-capacity targets; the caller keeps only the original rows."""
    values = np.asarray(worlds,dtype=float)
    if values.ndim != 3 or values.shape[1:] != (4,4) or not 0 < len(values) <= capacity:
        raise ValueError('IK target count is outside the fixed batch capacity')
    return np.concatenate([values,np.repeat(values[:1],capacity-len(values),axis=0)])


def row_safe_solve(solver, goal, current, returns, real_rows, subset_goal):
    """Retain each row's successful LM result; refine only unfinished rows."""
    import torch
    capacity = current.position.shape[0]
    if not solver.config.use_lm_seed:
        raise ValueError('This adapter requires the frozen LM-seeded IK configuration')
    seed_result = solver.seed_ik_solver.solve_batch(goal_tool_poses=goal,
        current_state=current.clone(),seed_config=current.position[:,None].clone(),
        return_seeds=solver.config.num_seeds)
    seeds = seed_result.js_solution.position.reshape(capacity,solver.config.num_seeds,-1).detach().clone()
    old_lm,old_exit = solver.config.use_lm_seed,solver.config.exit_early
    solver.config.use_lm_seed = False
    try:
        # The individual baseline returns as soon as this row has a valid
        # initial solution. A different failed target must not force that row
        # back through the optimizer and discard its valid initial result.
        coarse = solver.solve_pose(goal,current_state=current.clone(),seed_config=seeds,
            return_seeds=returns,run_optimizer=False)
        q = coarse.solution.reshape(capacity,returns,-1).detach().clone()
        valid = coarse.success.reshape(capacity,returns).detach().clone()
        valid &= torch.isfinite(q).all(-1)
        needs = ~valid[:real_rows].any(-1) if old_exit else torch.ones(real_rows,dtype=torch.bool,device=q.device)
        rows = torch.nonzero(needs,as_tuple=False).reshape(-1)
        indices = rows.detach().cpu().tolist()
        if len(rows):
            padded = torch.cat([rows,rows[:1].expand(capacity-len(rows))])
            refine_goal = subset_goal(padded)
            refine_seeds = seeds.index_select(0,padded).contiguous()
            solver.config.exit_early = False
            refined = solver.solve_pose(refine_goal,current_state=current.clone(),seed_config=refine_seeds,
                return_seeds=returns,run_optimizer=True)
            q[rows] = refined.solution.reshape(capacity,returns,-1)[:len(rows)].detach().clone()
            valid[rows] = refined.success.reshape(capacity,returns)[:len(rows)].detach().clone()
            valid &= torch.isfinite(q).all(-1)
        return q,valid,{'refined_source_rows':indices,
            'LM_success_rows_retained':real_rows-len(indices) if old_exit else 0,
            'LM_calls':1,'optimizer_calls':int(bool(indices)),
            'seeds_per_original_target':solver.config.num_seeds}
    finally:
        solver.config.use_lm_seed,solver.config.exit_early = old_lm,old_exit


class EndpointSequence:
    """Preserve candidate rows while compacting only failed rows for fallback."""

    def __init__(self, candidates, start, engine):
        self.engine, self.start = engine, start
        self.goals = []
        for candidate in candidates:
            goal = np.asarray(candidate['pose'], dtype=float).copy()
            goal[:3, 3] -= .043*goal[:3, 2]
            self.goals.append(goal)
        self.rows, self.receipts = {}, []
        self.cursor = 0

    def next(self, goal):
        index = self.cursor
        if index >= len(self.goals) or not np.allclose(goal, self.goals[index], atol=1e-12, rtol=0):
            raise ValueError('Endpoint request no longer matches the ordered original grasp')
        self.cursor += 1
        if index not in self.rows:
            stop = min(index+self.engine.capacity, len(self.goals))
            begin = time.perf_counter()
            native = self.engine.native(self.goals[index:stop], self.start)
            if len(native) != stop-index:
                raise RuntimeError('Batched native IK lost its target rows')
            missing = [i for i, q in enumerate(native) if q is None]
            pure = self.engine.pure([self.goals[index+i] for i in missing], self.start) if missing else []
            if len(pure) != len(missing):
                raise RuntimeError('Batched fallback IK lost its compacted target rows')
            fallback = dict(zip(missing, pure))
            for local, q in enumerate(native):
                self.rows[index+local] = {'native': q, 'pure': fallback.get(local)}
            self.receipts.append({'start_index': index, 'stop_index_exclusive': stop,
                'native_targets': stop-index, 'fallback_target_indices': [index+i for i in missing],
                'wall_seconds': time.perf_counter()-begin})
        return self.rows[index]

    def summary(self):
        return {'capacity': self.engine.capacity, 'consumed_candidates': self.cursor,
            'computed_candidates': len(self.rows), 'prefetched_unconsumed': len(self.rows)-self.cursor,
            'batches': self.receipts, 'solver_calls': list(self.engine.calls),
            'padding_rows_are_never_candidates': True, 'candidate_order_and_poses_unchanged': True}


class NativeBatchEngine:
    """Clone original IK configurations; share the current full collision world."""

    def __init__(self, pool, capacity):
        import torch
        from curobo.inverse_kinematics import InverseKinematics
        from resident_planner import all_kinematics_params
        if type(capacity) is not int or capacity not in (1, 16, 32):
            raise ValueError('Supported native IK capacities are 1, 16 and 32')
        self.capacity, self.calls, self.pool = capacity, [], pool
        self.context = pool.contexts['open']
        self.profile = pool.contact_profile
        source = self.context.planner.ik_solver
        native_cfg = self.fresh_config(source.config, capacity, pure=False)
        pure_cfg = self.fresh_config(pool.pure_solver.config, capacity, pure=True)
        for cfg in [native_cfg, pure_cfg]:
            if (cfg.position_tolerance != self.profile.position_tolerance_m
                    or cfg.orientation_tolerance != self.profile.rotation_tolerance_rad
                    or cfg.max_goalset != 1 or not cfg.use_cuda_graph):
                raise ValueError('Batch adaptation changed the original IK contract')
        self.full_solver = InverseKinematics(native_cfg,
            scene_collision_checker=self.context.planner.scene_collision_checker)
        self.pure_solver = InverseKinematics(pure_cfg)
        self.parameters = all_kinematics_params(self.full_solver)
        self._torch = torch

    def fresh_config(self, original, capacity, *, pure):
        """Rebuild configuration values without copying live CUDA callbacks."""
        from curobo.inverse_kinematics import InverseKinematicsCfg
        # Instantiating the upstream solver attaches live rollout callbacks to
        # its config. Reconstruct from the same task YAML, then retain every
        # original scalar field and optimizer iteration setting.
        arguments = inspect.signature(InverseKinematicsCfg.create).parameters
        options = {name: getattr(original, name) for name in arguments
            if hasattr(original, name) and name not in
            ['robot','optimizer_configs','metrics_rollout','transition_model','scene_model',
             'collision_cache','max_batch_size','multi_env','max_goalset']}
        config = InverseKinematicsCfg.create(robot=str(self.context.model),
            optimizer_configs=['ik/lbfgs_ik.yml'], scene_model=None,
            max_batch_size=capacity, multi_env=False, max_goalset=1,
            load_collision_spheres=not pure, **options)
        for field in fields(original):
            if field.name not in ['core_cfg','robot_config','max_batch_size']:
                setattr(config, field.name, getattr(original, field.name))
        for rollout in [config.core_cfg.metrics_rollout_config, *config.core_cfg.optimizer_rollout_configs]:
            for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
        if len(config.optimizer_configs) != len(original.optimizer_configs):
            raise ValueError('Original IK optimizer stages changed')
        for current, previous in zip(config.optimizer_configs, original.optimizer_configs):
            copy_optimizer_settings(current,previous)
        return config

    def bind(self, context):
        if context is not self.context:
            raise ValueError('Endpoint batches are only bound to the original open context')
        # Bind has already refreshed the exact original scene. Both native
        # solvers use that same scene object, including the parked arm.
        if self.full_solver.scene_collision_checker is not context.planner.scene_collision_checker:
            raise RuntimeError('Batched IK no longer shares the active collision world')
        source = context.checker.kinematics.kinematics_config
        for target in self.parameters:
            target.copy_(source)
        # Original bind resets full-world IK at each scheduling window, while
        # its pure fallback keeps its sequence until the next whole request.
        self.full_solver.reset_seed()
        self.calls = []

    def reset_seed(self):
        self.full_solver.reset_seed()
        self.pure_solver.reset_seed()

    def _solve(self, solver, worlds, start, returns, stage):
        from curobo.types import GoalToolPose, JointState
        from resident_planner import frozen
        torch = self._torch
        count = len(worlds)
        if not 0 < count <= self.capacity:
            raise ValueError('IK target count is outside the fixed batch capacity')
        begin = time.perf_counter()
        # Supply every JointState field at the fixed capacity. In particular,
        # upstream input padding does not expand its jerk tensor.
        poses = np.matmul(self.context.invbase, padded_worlds(worlds,self.capacity))
        quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        goal = GoalToolPose(tool_frames=['right_eef_link'],
            position=torch.as_tensor(poses[:, :3, 3], device=start.device,
                dtype=torch.float32).reshape(self.capacity, 1, 1, 1, 3).contiguous(),
            quaternion=torch.as_tensor(quaternions, device=start.device,
                dtype=torch.float32).reshape(self.capacity, 1, 1, 1, 4).contiguous())
        current = JointState.from_position(start.expand(self.capacity, -1).clone(), joint_names=frozen.NAMES)
        event_start, event_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        event_start.record()
        def subset(indices):
            return GoalToolPose(tool_frames=goal.tool_frames,
                position=goal.position.index_select(0,indices).contiguous(),
                quaternion=goal.quaternion.index_select(0,indices).contiguous())
        q,valid,stages = row_safe_solve(solver,goal,current,returns,count,subset)
        dispatched = time.perf_counter()
        event_end.record(); event_end.synchronize()
        q = q[:count].detach().clone()
        valid = valid[:count].detach().clone()
        valid &= torch.isfinite(q).all(-1)
        self.calls.append({'stage': stage, 'targets': count, 'padded_capacity': self.capacity,
            'optimization_seeds_per_target': solver.config.num_seeds,
            'LM_seeds_per_target': solver.config.seed_solver_num_seeds,
            'returned_branches_per_target': returns, 'gpu_seconds': event_start.elapsed_time(event_end)/1000.,
            'cpu_dispatch_seconds': dispatched-begin, 'wall_seconds': time.perf_counter()-begin,
            'successes_per_target':valid.sum(-1).detach().cpu().tolist(), **stages})
        return q, valid

    def native(self, worlds, start):
        torch = self._torch
        q, valid = self._solve(self.full_solver, worlds, start, 4, 'full_world_ik')
        admitted = q[valid]
        if len(admitted):
            clear = self.context.checker.validate(admitted[:, None]).reshape(-1)
            valid[valid.clone()] = clear
        rows = []
        for values, mask in zip(q, valid):
            values = values[mask]
            rows.append(values[(values-start).square().sum(-1).argmin()].clone() if len(values) else None)
        indices = [i for i,value in enumerate(rows) if value is not None]
        if indices:
            values = torch.stack([rows[i] for i in indices]).detach().cpu().numpy()
            geometry = self.pool.contact_views['open']['validator'].geometry
            actual = geometry.frames(values)['right_eef_link']
            expected = np.asarray(worlds)[indices]
            pe = np.linalg.norm(actual[:,:3,3]-expected[:,:3,3],axis=-1)
            re = Rotation.from_matrix(actual[:,:3,:3]@np.swapaxes(expected[:,:3,:3],1,2)).magnitude()
            bounds = self.context.limits.detach().cpu().numpy()
            if (np.any(pe>self.profile.position_tolerance_m) or np.any(re>self.profile.rotation_tolerance_rad)
                    or not ((values>=bounds[:,0]) & (values<=bounds[:,1])).all()):
                raise RuntimeError('A batched native endpoint failed independent FK or joint limits')
            self.calls[-1].update(independent_FK_checked=len(indices),
                maximum_independent_position_error_m=float(np.max(pe)),
                maximum_independent_rotation_error_rad=float(np.max(re)))
        return rows

    def pure(self, worlds, start):
        torch = self._torch
        q, valid = self._solve(self.pure_solver, worlds, start,
            self.profile.ik_return_seeds, 'pure_kinematic_fallback')
        rows = []
        for values, mask in zip(q, valid):
            values = values[mask]
            rows.append(values[torch.argsort((values-start).square().sum(-1), stable=True)].clone())
        return rows
