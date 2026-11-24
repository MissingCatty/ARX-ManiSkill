"""Integrate scoped tabletop refinement into the resident ARX planner."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.assets import resource_path
from arx_project.layout import read_json, write_json
from .geometry import JOINTS
from .validator import LocalContactValidator


def contact_pool(base, profile):
    """Keep the original planner resident and extend only local contact stages."""
    class ContactPool(base):
        def __init__(self, root, warmup_trial):
            self.contact_profile = profile
            self.contact_views = {}
            self.contact_stage = 'reach'
            self.pure_solver = None
            super().__init__(root, warmup_trial)
            for role, filename in [('open', 'state.json'), ('closed', 'closed_state.json')]:
                self.contact_view(self.contexts[role], read_json(Path(warmup_trial)/filename))
            state = read_json(Path(warmup_trial)/'state.json')
            warm_context = self.contexts['open']
            measured = warm_context.measured(state)
            if not len(self.unconstrained_ik(warm_context, np.asarray(state['links']['right_eef_link']), measured)):
                raise RuntimeError('Local-contact IK warmup did not recover the measured tool pose')
            write_json(Path(root)/'fingertip_policy_ready.json', {
                'profile_id': profile.profile_id, 'collision_profile_id': profile.collision_profile_id,
                'maximum_penetration_m': profile.maximum_penetration_m,
                'distal_length_m': profile.distal_length_m, 'full_world_models_unchanged': True,
                'model_load_counts': self.loads, 'auxiliary_checkers_loaded': len(self.contact_views),
                'pure_ik_models_loaded': 1, 'new_components_warmed_before_first_request': True})

        def contact_view(self, context, state):
            import torch
            from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
            from resident_planner import all_kinematics_params, collision_scene_config
            role = 'closed' if context is self.contexts['closed'] else 'open'
            scene = read_json(context.out/'scene.json')
            other_scene = deepcopy(scene)
            if 'table' not in other_scene.get('cuboid', {}):
                raise ValueError('Measured table geometry is required')
            del other_scene['cuboid']['table']
            key = (str(context.out), json.dumps(state['qpos']), json.dumps(scene, sort_keys=True))
            view = self.contact_views.get(role)
            if view is None:
                checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                    robot_config=str(context.model), scene_model=other_scene, n_meshes=8, n_cuboids=8,
                    collision_activation_distance=0.))
                checker.self_collision_cost = None
                validator = LocalContactValidator(context, state, checker,
                    resource_path(profile.actual_geometry_resource), profile=profile)
                view = {'validator': validator, 'checker': checker,
                        'parameters': all_kinematics_params(checker), 'key': None}
                self.contact_views[role] = view
            if view['key'] != key:
                torch.cuda.synchronize()
                params = context.checker.kinematics.kinematics_config
                for target in view['parameters']:
                    target.copy_(params)
                view['checker'].clear_scene_cache()
                view['checker'].update_world(collision_scene_config(other_scene))
                view['validator'].update(context, state)
                view['key'] = key
            return view['validator']

        def validate_start_state(self, context, state, current, closed=False, include_target=False):
            if self.contact_stage not in ['final_approach', 'lift_release'] or include_target:
                return super().validate_start_state(context, state, current, closed, include_target)
            validator = self.contact_view(context, state)
            if self.contact_stage == 'lift_release':
                self.release_anchor(state)
            valid = bool(validator.validate(current[None], stage=self.contact_stage).all())
            write_json(context.out/'local_start_policy.json', {'valid': valid,
                'profile_id': profile.profile_id, **validator.receipts[-1]})
            return valid

        def release_anchor(self, state):
            anchor = state.get('fingertip_table_contact')
            if not anchor or not anchor.get('valid') or anchor.get('profile_id') != profile.profile_id or anchor.get('stage') != 'close':
                raise ValueError('Missing or mismatched measured closure contact receipt')
            if not np.allclose(anchor['qpos'], state['qpos'], atol=1e-6, rtol=0):
                raise ValueError('Closure contact receipt does not match measured joints')
            value = float(anchor['depth_m'])
            if not np.isfinite(value) or not 0 <= value <= profile.maximum_penetration_m:
                raise ValueError('Invalid measured closure penetration')
            return value

        def unconstrained_ik(self, context, world, start):
            import torch
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
            from curobo.types import GoalToolPose
            if self.pure_solver is None:
                # The EEF chain is independent of the prismatic finger locks;
                # changing those measured locks cannot change this IK model.
                geometry = self.contact_views['open']['validator'].geometry
                link = 'right_eef_link'
                while link != 'body4':
                    joint = geometry.parents[link]
                    if joint.get('name') in ['right_joint27', 'right_joint28']:
                        raise ValueError('Tool IK depends on a measured finger lock')
                    link = joint.find('parent').get('link')
                cfg = InverseKinematicsCfg.create(robot=str(context.model), scene_model=None,
                    optimizer_configs=['ik/lbfgs_ik.yml'], self_collision_check=False,
                    num_seeds=profile.ik_fallback_seeds, seed_solver_num_seeds=profile.ik_fallback_seeds,
                    position_tolerance=profile.position_tolerance_m,
                    orientation_tolerance=profile.rotation_tolerance_rad,
                    use_cuda_graph=True, max_batch_size=1, load_collision_spheres=False)
                for rollout in [cfg.core_cfg.metrics_rollout_config, *cfg.core_cfg.optimizer_rollout_configs]:
                    for cost in rollout.get_cost_manager_configs():
                        cost.disable_self_collision()
                self.pure_solver = InverseKinematics(cfg)
            pose = context.invbase @ world
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            goal = GoalToolPose(tool_frames=['right_eef_link'],
                position=torch.as_tensor(pose[:3, 3], device=start.device, dtype=torch.float32).reshape(1, 1, 1, 1, 3),
                quaternion=torch.as_tensor(quaternion, device=start.device, dtype=torch.float32).reshape(1, 1, 1, 1, 4))
            result = self.pure_solver.solve_pose(goal, current_state=context.js(start), return_seeds=profile.ik_return_seeds)
            q = result.solution.reshape(-1, 6).detach().clone()
            valid = result.success.reshape(-1).detach().clone() & torch.isfinite(q).all(-1)
            q = q[valid]
            return q[torch.argsort((q-start).square().sum(-1), stable=True)]

        def grasp_endpoint(self, context, state, grasp, start, validator):
            native = context.ik(grasp, start)
            if native is not None:
                return native, {'kind': 'native_full_world_ik'}
            solutions = self.unconstrained_ik(context, grasp, start)
            reasons = []
            for q in solutions:
                actual = validator.geometry.frames(q.detach().cpu().numpy()[None])['right_eef_link'][0]
                pe = float(np.linalg.norm(actual[:3, 3]-grasp[:3, 3]))
                re = float(Rotation.from_matrix(actual[:3, :3] @ grasp[:3, :3].T).magnitude())
                if pe > profile.position_tolerance_m or re > profile.rotation_tolerance_rad:
                    reasons.append('independent_fk_error'); continue
                if validator.validate(q[None], stage='final_approach').all():
                    return q.clone(), {'kind': 'bounded_ik_with_local_table_check',
                        'independent_position_error_m': pe, 'independent_rotation_error_rad': re,
                        'native_full_world_clear': bool(context.checker.validate(q[None, None]).all())}
                last = validator.receipts[-1]
                if not last['joint_bounds_clear']:
                    reasons.append('joint_limits')
                elif not last['other_world_clear']:
                    reasons.append('ground_or_other_object_collision')
                else:
                    reasons.append(last['narrow_phase'][-1]['reason'] if last['narrow_phase'] else 'native_collision')
            return None, {'kind': 'no_local_grasp_endpoint', 'kinematic_solutions': len(solutions), 'reasons': reasons}

        def select_contact(self, context, state, directory, validator):
            candidates = read_json(directory/'candidates/candidate_pool.json')['candidates']
            start = context.measured(state)
            rejected = {k: 0 for k in ['grasp_ik', 'pre_or_lift_ik', 'path', 'local_approach']}
            events = []
            for index, candidate in enumerate(candidates):
                contact = np.asarray(candidate['pose'], dtype=float)
                grasp = contact.copy(); grasp[:3, 3] -= .043*contact[:3, 2]
                pre = grasp.copy(); pre[:3, 3] -= .08*contact[:3, 2]
                lift = grasp.copy(); lift[2, 3] += .03
                qg, evidence = self.grasp_endpoint(context, state, grasp, start, validator)
                events.append({'candidate_index': index, **evidence})
                if qg is None:
                    rejected['grasp_ik'] += 1; continue
                qp, ql = context.ik(pre, qg), context.ik(lift, qg)
                if qp is None or ql is None:
                    rejected['pre_or_lift_ik'] += 1; continue
                reach = context.path(start, qp)
                if reach is None:
                    rejected['path'] += 1; continue
                approach = None
                if bool(context.checker.validate(qg[None, None]).all()):
                    native_path = context.path(qp, qg)
                    if native_path is not None:
                        approach = context.optimize('approach', native_path, 5.)
                if approach is None:
                    approach = validator.straight_approach(qp.cpu().numpy(), qg.cpu().numpy())
                if approach is None:
                    rejected['local_approach'] += 1; continue
                selected = {'index': index, 'score': float(candidate['score']),
                    'candidate_metadata': candidate.get('metadata', {}),
                    'contact_pose_world': contact.tolist(), 'grasp_world': grasp.tolist(),
                    'pregrasp_world': pre.tolist(), 'lift_world': lift.tolist(),
                    'target_contact_world': contact[:3, 3].tolist(), 'table_clearance_raise_m': 0.,
                    'q_grasp': qg.tolist(), 'q_pre': qp.tolist(), 'q_lift': ql.tolist(),
                    'contact_policy_id': profile.profile_id, 'pose_adjustment_m': 0.,
                    'grasp_ik_evidence': evidence}
                return {'success': True, 'selected': selected, 'rejected': rejected,
                    'paths': {'pregrasp': context.optimize('pregrasp', reach, 6.), 'approach': approach},
                    'candidate_events': events}
            return {'success': False, 'reason': 'no_feasible_native_post_or_recalled_candidate',
                    'count': len(candidates), 'rejected': rejected, 'candidate_events': events}

        def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl'):
            import torch
            if sum([observe, approach, lift]) > 1:
                raise ValueError('Conflicting planning stages')
            if observe:
                return super().request(directory, observe=True, selector=selector)
            directory = Path(directory)
            name = 'lift_plan' if lift else 'approach_plan' if approach else 'plan'
            state = read_json(directory/('closed_state.json' if lift else 'pregrasp_state.json' if approach else 'state.json'))
            self.contact_stage = 'lift_release' if lift else 'final_approach' if approach else 'reach'
            started = time.perf_counter()
            try:
                context = self.bind(directory/name, state, closed=lift, include_target=False)
                validator = self.contact_view(context, state)
                if not approach and not lift:
                    result = self.select_contact(context, state, directory, validator)
                else:
                    selected = read_json(directory/'plan_result.json')['selected']
                    current = context.measured(state)
                    if lift:
                        goal = context.ik(np.asarray(selected['lift_world']), current)
                        anchor = self.release_anchor(state)
                        path = None
                        if goal is not None:
                            if bool(context.checker.validate(current[None, None]).all()):
                                native = context.path(current, goal)
                                if native is not None:
                                    candidate = context.optimize('lift', native, 6.)
                                    if validator.validate_trajectory(candidate, stage='lift_release', release_anchor_m=anchor):
                                        path = candidate
                            if path is None:
                                path = validator.straight_approach(current.cpu().numpy(), goal.cpu().numpy(),
                                    duration_s=6., stage='lift_release', release_anchor_m=anchor)
                        label, reason = 'lift', 'measured_closure_lift_unreachable'
                    else:
                        goal = torch.as_tensor(selected['q_grasp'], dtype=torch.float32, device=current.device)
                        path = None
                        if bool(context.checker.validate(current[None, None]).all()) and bool(context.checker.validate(goal[None, None]).all()):
                            native = context.path(current, goal)
                            if native is not None:
                                path = context.optimize('approach', native, 5.)
                        if path is None:
                            path = validator.straight_approach(current.cpu().numpy(), goal.cpu().numpy())
                        label, reason = 'approach', 'measured_pregrasp_approach_unreachable'
                    result = {'success': path is not None, 'reason': 'success' if path is not None else reason}
                    if path is not None:
                        result['paths'] = {label: path}
                from optimize_path import flush_planning_artifacts
                flush_planning_artifacts(); torch.cuda.synchronize()
                result.update(contact_policy_id=profile.profile_id, maximum_tip_penetration_m=profile.maximum_penetration_m,
                              wall_seconds=time.perf_counter()-started)
                write_json(directory/(name+'_contact_receipts.json'), validator.receipts)
                write_json(directory/(name+'_result.json'), result)
                return result
            finally:
                self.contact_stage = 'reach'
    return ContactPool
