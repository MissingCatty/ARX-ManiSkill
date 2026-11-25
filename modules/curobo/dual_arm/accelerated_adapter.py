"""Bound the second arm's search and batch the unchanged opening collision gate.

This optional factory leaves the registered V3/V4 adapter intact. It uses the
same native candidate order, equal-rank joint-motion tie-break and sphere model.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from arx_project.layout import read_json, write_json
from .adapter import DualArmPlanner


class RankBoundPool:
    """Let the second arm test every candidate capable of winning selection."""
    def __init__(self, pool):
        self.pool = pool
        self.best_index = None
        self.receipts = []

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def request(self, directory, **flags):
        if any(flags.get(name, False) for name in ['observe', 'approach', 'lift']):
            return self.pool.request(directory, **flags)
        directory = Path(directory)
        path = directory/'candidates/candidate_pool.json'
        candidates = read_json(path)
        total = len(candidates['candidates'])
        limit = total if self.best_index is None else min(total, self.best_index+1)
        receipt = {'full_candidate_count': total, 'retained_prefix_count': limit,
                   'winning_candidate_before_request': self.best_index,
                   'equal_rank_included': True, 'world_poses_and_scores_unchanged': True}
        if limit < total:
            bounded = {**candidates, 'candidates': candidates['candidates'][:limit]}
            bounded['search_bound'] = receipt
            write_json(path, bounded)
        write_json(directory/'candidate_search_bound.json', receipt)
        result = self.pool.request(directory, **flags)
        result = deepcopy(result)
        if result['success']:
            index = int(result['selected']['index'])
            if not 0 <= index < limit:
                raise ValueError('Planner returned a candidate outside its admitted prefix')
            self.best_index = index if self.best_index is None else min(self.best_index, index)
        elif limit < total:
            # A bounded loser is not evidence that the arm is globally unable
            # to grasp; every omitted candidate has a lower model rank.
            result['backend_reason'] = result.get('reason')
            result['reason'] = 'no_candidate_in_prefix_can_improve_selected_arm'
            result['full_pool_exhausted'] = False
        result['search_bound'] = receipt
        write_json(directory/'plan_result.json', result)
        self.receipts.append({'arm': directory.name, **receipt,
                              'success': result['success'], 'best_index_after_request': self.best_index})
        return result


def opening_samples(canonical):
    target = canonical['planned_opening']['target_per_finger_m']
    observed = canonical['planned_opening']['observed_solver_finger_positions_m']
    if not np.isfinite([target, *observed.values()]).all():
        raise ValueError('Opening measurements must be finite')
    count = max(2, int(np.ceil(max(abs(target-v) for v in observed.values())/.0005))+1)
    if count > 512:
        raise ValueError('Opening measurements exceed the registered finger stroke')
    return np.linspace(0., 1., count), target, observed


def validate_sphere_batch(checker, spheres, q):
    """Apply exactly the bound/self/world sum used by native checker.validate."""
    import torch
    if spheres.ndim != 4 or spheres.shape[-1] != 4 or not torch.isfinite(spheres).all():
        raise ValueError('Invalid opening sphere batch')
    # Native Warp costs flatten their inputs; expanded joint tensors otherwise
    # have zero batch strides even though every sample has the same arm pose.
    spheres, q = spheres.contiguous(), q.contiguous()
    checker.setup_batch_tensors(spheres.shape[0], spheres.shape[1])
    terms = [checker.get_bound(q).sum(dim=-1, keepdim=True)]
    if checker.self_collision_cost is not None:
        terms.append(checker.get_self_collision(spheres).sum(dim=-1, keepdim=True))
    if checker.collision_constraint is not None:
        terms.append(checker.get_collision_constraint(spheres).sum(dim=-1, keepdim=True))
    return torch.sum(torch.cat(terms, dim=-1), dim=-1) == 0.


class AcceleratedDualArmPlanner(DualArmPlanner):
    def opening_sweep(self, work, canonical):
        """Reuse the collision world; retain native FK at every original width."""
        import torch
        import yaml
        work = Path(work)
        started = time.perf_counter()
        fractions, target, observed = opening_samples(canonical)
        initial = deepcopy(canonical)
        for name, value in observed.items():
            initial['qpos'][initial['joint_names'].index(name)] = float(value)
        try:
            context = self.pool.bind(work/'opening_batch_start', initial, include_target=True)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model':
                raise
            result = {'passed': False, 'reason': 'gripper_opening_sweep_in_collision',
                'fraction': 0., 'samples_checked': 1, 'first_invalid_sample': 0,
                'method': 'native_start_gate', 'wall_seconds': time.perf_counter()-started}
            write_json(work/'opening_sweep.json', result)
            return result
        # The exact affine lock-state refresh is derived and checked by the
        # unchanged FastPlannerPool against the original URDF factory.
        config = yaml.safe_load(context.model.read_text())['robot_cfg']
        q = context.measured(initial)[None, None]
        checker = context.checker
        parameters = checker.kinematics.kinematics_config
        spheres = []
        try:
            for fraction in fractions:
                for name, value in observed.items():
                    config['kinematics']['lock_joints'][name] = float(value+fraction*(target-value))
                robot = self.pool._measured_robot_view(config, context.planner.device_cfg)
                parameters.copy_(robot.kinematics.kinematics_config)
                spheres.append(checker.get_kinematics(q).robot_spheres.detach().clone())
            batch = torch.cat(spheres, dim=0)
            if batch.shape[2] != 871:
                raise ValueError('Opening check must retain the approved 871-sphere model')
            valid = validate_sphere_batch(checker, batch, q.expand(len(fractions), -1, -1)).reshape(-1)
            torch.cuda.synchronize()
            mask = valid.detach().cpu().numpy()
            failures = np.flatnonzero(~mask)
            first = int(failures[0]) if len(failures) else None
            result = {'passed': first is None, 'samples_checked': len(fractions),
                'first_invalid_sample': first,
                'fraction': None if first is None else float(fractions[first]),
                'reason': 'native_clear' if first is None else 'gripper_opening_sweep_in_collision',
                'maximum_finger_step_m': .0005, 'target_per_finger_m': target,
                'target_and_parked_arm_collision_checked': True,
                'method': 'native_FK_all_widths_batched_bound_self_world_collision',
                'sphere_count': 871, 'world_bind_count': 1,
                'sampling_fractions': fractions.tolist(), 'valid_per_sample': mask.tolist(),
                'wall_seconds': time.perf_counter()-started}
            write_json(work/'opening_sphere_batch.json', {'shape': list(batch.shape),
                'finite': bool(torch.isfinite(batch).all()), 'native_affine_lock_refresh': True})
        finally:
            # Restore every resident tensor and the admitted open state before
            # native grasp planning. This also restores the solver seed state.
            try:
                self.pool.bind(work/'opening_batch_restore', canonical, include_target=True)
            except AssertionError as error:
                if str(error) != 'Measured state collides in unchanged ARX model':
                    raise
                if 'result' not in locals() or result['passed']:
                    raise RuntimeError('Opening batch disagrees with native endpoint gate') from error
        result['world_bind_count'] = 2
        result['wall_seconds'] = time.perf_counter()-started
        write_json(work/'opening_sweep.json', result)
        return result

    def request(self, directory, **flags):
        if flags.get('approach') or flags.get('lift') or flags.get('observe'):
            return super().request(directory, **flags)
        original = self.pool
        bounded = RankBoundPool(original)
        try:
            self.pool = bounded
            result = super().request(directory, **flags)
        finally:
            self.pool = original
        result['acceleration'] = {'factory': 'curobo.dual_arm.accelerated_adapter',
                                  'rank_bounds': bounded.receipts, 'opening': 'batched_native_collision'}
        write_json(Path(directory)/'plan_result.json', result)
        return result


def create(args):
    from .adapter import create as create_original
    original = create_original(args)
    return AcceleratedDualArmPlanner(original.pool, original.model, original.root)
