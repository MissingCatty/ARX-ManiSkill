"""Ordinary joint-goal connector and revalidation of unchanged source commands."""
import copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, write, verify_files, sha
from scripts.curobo_arx.data_factory.teacher import Teacher as OriginalTeacher, Rejection
from .contract import replay_command, joint_values, POLICY


class Teacher(OriginalTeacher):
    def ik(self, world, start, planner, checker, inverse, seed=None):
        forced = getattr(self, 'forced_anchor', None)
        if forced is None or not np.allclose(world, forced['anchor_world'], atol=1e-10, rtol=0):
            return super().ik(world, start, planner, checker, inverse, seed)
        q = self.torch.tensor(forced['anchor_q'], device=start.device, dtype=start.dtype)
        if not checker.validate(q[None, None]).all():
            raise Rejection('seed_anchor_joint_goal_collision')
        state = planner.compute_kinematics(self.js(q)).tool_poses
        pos = state.position.detach().cpu().numpy().reshape(3)
        quat = state.quaternion.detach().cpu().numpy().reshape(4)
        matrix = np.eye(4); matrix[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(); matrix[:3, 3] = pos
        difference = inverse@np.asarray(world)
        if np.linalg.norm(pos-difference[:3, 3]) >= .002 or Rotation.from_matrix(matrix[:3, :3]@difference[:3, :3].T).magnitude() >= .01:
            raise Rejection('seed_anchor_fk_disagrees_with_recorded_pose')
        return q.clone()

    def handle(self, request):
        action = request['action']
        if action not in ('seed_connect', 'seed_replay'):
            return super().handle(request)
        seed = self.registration['seeds'][request['seed_id']]
        verify_files(seed['files'])
        if action == 'seed_connect':
            changed = copy.deepcopy(request)
            label = request.get('label', 'approach_connect')
            if label not in ('pregrasp', 'approach_connect'):
                raise ValueError('Unregistered anchor connector label')
            changed.update(action='plan', phase='reach', label=label,
                           goal_world=seed['anchor_world'], ik_seed=seed['anchor_q'])
            self.forced_anchor = seed
            try:
                result = super().handle(changed)
                if not result.get('rejected'):
                    write(Path(request['output'])/'connector_goal_validation.json', {
                        'seed_id': seed['seed_id'], 'joint_goal': seed['anchor_q'],
                        'source_entry_world': seed['anchor_world'], 'joint_goal_not_only_tool_ik': True,
                        'original_target_avoiding_path_and_all_checks_retained': True})
                return result
            finally:
                self.forced_anchor = None
        return self.replay(request, seed)

    def replay(self, request, seed):
        phase = request['phase']; out = Path(request['output'])
        if phase not in ('approach', 'lift'):
            raise ValueError('Only moving source phases use the replay planner')
        snapshot = request['snapshot']; self.initialize(read(request['scene']), snapshot)
        names = self.context['names']; mode = 'closed' if phase == 'lift' else 'open'
        planner, checker, updater, models = self.context[mode]
        measured = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
        updater.apply(models, {name: float(measured[name]) for name in updater.names})
        command = replay_command(seed, phase)
        if command['joint_names'] != names or snapshot['right_arm'] != names:
            raise RuntimeError('ARX replay joint identity changed')
        q = self.torch.tensor(command['position'], device='cuda', dtype=self.torch.float32)
        limits = self.torch.tensor([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names], device='cuda')
        try:
            actual_q = self.torch.tensor(joint_values(snapshot, names), device='cuda', dtype=q.dtype)
            fk = planner.compute_kinematics(self.js(actual_q)).tool_poses
            xyz = fk.position.detach().cpu().numpy().reshape(3)
            quat = fk.quaternion.detach().cpu().numpy().reshape(4)
            model_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            actual = np.linalg.inv(np.asarray(snapshot['link_world']['body4']))@np.asarray(snapshot['link_world']['right_eef_link'])
            if np.linalg.norm(xyz-actual[:3, 3]) >= .002 or Rotation.from_matrix(model_rotation@actual[:3, :3].T).magnitude() >= .01:
                raise RuntimeError('Live ARX/planner FK mismatch during source replay')
            start_error = float(np.max(np.abs(joint_values(snapshot, names)-np.asarray(command['position'][0]))))
            if start_error > POLICY['tail_start_match_rad']:
                raise Rejection('source_tail_start_state_mismatch', {'maximum_joint_error_rad': start_error})
            if not checker.validate(self.torch.tensor(joint_values(snapshot, names), device='cuda', dtype=q.dtype)[None, None]).all():
                raise Rejection('measured_start_collision')
            violation = float(self.torch.maximum(limits[:, 0]-q, q-limits[:, 1]).clamp_min(0).max())
            if violation > 1e-6:
                raise Rejection('source_replay_joint_limit')
            a = self.torch.arange(10, device=q.device)/10
            dense = self.torch.cat([(q[:-1, None]*(1-a[None, :, None])+q[1:, None]*a[None, :, None]).reshape(-1, len(names)), q[-1:]])
            invalid = sum(int((~checker.validate(chunk[:, None])).sum()) for chunk in dense.split(64))
            if invalid:
                raise Rejection('source_replay_current_scene_collision', {'invalid_samples': invalid})
            times = np.asarray(command['time_from_start']); dt = np.diff(times)
            if len(times) != len(q) or (dt <= 0).any() or not np.allclose(dt, dt[0], atol=1e-7, rtol=0):
                raise RuntimeError('Successful source command has invalid time grid')
            derivative = np.asarray(command['position'], dtype=float); maxima = {}
            for order, bound in ((1, 1.5), (2, 5.), (3, 100.)):
                derivative = np.diff(derivative, axis=0)/dt[0]
                value = float(np.max(np.abs(derivative))) if derivative.size else 0.
                maxima[str(order)] = value
                if not np.isfinite(value) or value > bound+1e-5:
                    raise Rejection('source_replay_derivative_bound')
            if phase == 'approach':
                inverse = np.linalg.inv(np.asarray(snapshot['link_world']['body4']))
                certificate = self.nominal_closed(np.asarray(request['grasp_world']), q[0], inverse, q_grasp_override=q[-1])
                write(out/'approach_nominal_closed_validation.json', certificate)
            self.visual_path({**request, 'action': 'plan', 'label': phase}, planner, q, out)
            original = read(Path(seed['episode'])/(phase+'_optimization.json'))
            cert = {'method': 'Unchanged optimized source command, revalidated in the current frozen ARX scene',
                    'source_optimization_certificate': original, 'stage': phase,
                    'success': True, 'max_limit_exceed_rad': violation, 'invalid_samples': invalid,
                    'dense_samples': len(dense), 'seed_replay': {
                        'source_command_sha256': sha(Path(seed['episode'])/(phase+'_command.json')),
                        'source_optimization_sha256': sha(Path(seed['episode'])/(phase+'_optimization.json')),
                        'joint_positions_times_and_gripper_unchanged': True,
                        'current_start_error_rad': start_error, 'current_scene_dense_revalidated': True,
                        'actual_command_finite_difference_maxima': maxima}}
            write(out/(phase+'_optimization.json'), cert)
            return {'command': command, 'desired_world': request['goal_world'], 'phase': phase,
                    'source_replay': cert['seed_replay']}
        except Rejection as exc:
            return {'rejected': str(exc), 'details': exc.details, 'phase': phase}
