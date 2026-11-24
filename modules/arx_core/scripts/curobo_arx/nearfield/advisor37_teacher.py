"""Authorized seed-only changes; original ARX models/physics remain frozen."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.teacher import Rejection, bounded_planning_start
from .advisor33_teacher import Teacher as Original
from .advisor37_motion import FrozenURDF, cartesian_knots, command_times
from .advisor37_surfaces import SurfaceChecker
from training.nearfield_arx.contract import pose_error


class Teacher(Original):
    def authorize(self):
        gate = self.registration.get('advisor37_seed_protocol', {})
        if (self.registration['stage'] != 'advisor37_canonical_seed_preflight'
            or not gate.get('remove_nominal_closed_preflight_gate')
            or gate.get('training_authorized') is not False):
            raise RuntimeError('Advisor37 changes require explicit seed-only registration')

    def nominal_closed(self, grasp, start, inverse, q_grasp_override=None):
        self.authorize()
        planner, checker, _, _ = self.context['open']
        q = self.ik(grasp, start, planner, checker, inverse) if q_grasp_override is None else q_grasp_override
        if not checker.validate(q[None, None]).all():
            raise Rejection('preflight_open_goal_collision')
        return {'pass': True, 'q_grasp': q.tolist(), 'nominal_closed_check_executed': False,
                'scope': 'User-authorized seed preflight: actual open goal valid; physical closure/lift remain unverified',
                'geometry_changed': False, 'physical_success_claim': False}

    def visual_goal(self, request, world):
        result = super().visual_goal(request, world)
        asset = request.get('visual_context', {}).get('asset', {})
        if result is not None and 'masher' in str(asset.get('object_id', asset.get('usd_path', ''))):
            from .advisor37_geometry import clearance_bound
            result = copy.deepcopy(result)
            result['advisor37_masher_minimum_surface_distance_m'] = .010
            if clearance_bound(result) < .010:
                result['pass'] = False
                result['reasons'] = [*result.get('reasons', []), 'masher_wrist_clearance_below_10mm']
        return result

    def closure(self, request):
        self.authorize()
        binding, snapshot = request['binding'], request['snapshot']
        verify_files(binding['files'])
        if binding['snapshot_sha256'] != digest(snapshot):
            raise RuntimeError('Seed closure snapshot binding mismatch')
        self.initialize(read(request['scene']), snapshot)
        names = self.context['names']
        if names != snapshot['right_arm'] or names != binding['hold_joint_names']:
            raise RuntimeError('Seed closure joint identity mismatch')
        planner, checker, updater, models = self.context['open']
        measured = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
        updater.apply(models, {n: float(measured[n]) for n in updater.names})
        limits = [snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names]
        try:
            q = self.torch.tensor(bounded_planning_start([measured[n] for n in names], limits),
                                  device='cuda', dtype=self.torch.float32)
            hold = self.torch.tensor(binding['hold_q'], device='cuda', dtype=q.dtype)
            fit = pose_error(self.fk(planner, q, snapshot), snapshot['link_world']['right_eef_link'])
            if fit['translation_m'] >= .002 or fit['rotation_rad'] >= .01:
                raise RuntimeError('Live ARX/planner FK mismatch at seed closure')
            if not checker.validate(self.torch.stack([q, hold])[:, None]).all():
                raise Rejection('closure_open_state_collision')
            result = {'pass': True, 'binding_sha256': digest(binding), 'hold_q': binding['hold_q'],
                      'hold_world': self.fk(planner, hold, snapshot).tolist(), 'live_fk': fit,
                      'nominal_closed_check_executed': False, 'physical_execution_certified': False,
                      'motion_commands': 0, 'scope': 'Seed-only authorized removal of nominal closed prediction; actual open-state checks retained'}
        except Rejection as error:
            result = {'pass': False, 'rejected': str(error), 'binding_sha256': digest(binding), 'motion_commands': 0}
        write(Path(request['output']) / 'closure_model_validation.json', result)
        return result

    def fk(self, planner, q, snapshot):
        pose = planner.compute_kinematics(self.js(q)).tool_poses
        result = np.eye(4)
        result[:3, 3] = pose.position.detach().cpu().numpy().reshape(3)
        quat = pose.quaternion.detach().cpu().numpy().reshape(4)
        result[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return np.asarray(snapshot['link_world']['body4']) @ result

    def cartesian(self, request, *, insertion):
        self.authorize()
        out, snapshot = Path(request['output']), request['snapshot']
        start_time = time.monotonic()
        mode = 'open' if insertion else 'closed'
        self.initialize(read(request['scene']), snapshot, include_target=False)
        planner, checker, updater, models = self.context[mode]
        measured = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
        updater.apply(models, {n: float(measured[n]) for n in updater.names})
        names = self.context['names']
        limits = np.asarray([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names])
        original_urdf = Path(self.registration['model_reference']) / 'model/arx.urdf'
        urdf = FrozenURDF(original_urdf)
        binding = urdf.verify(snapshot)
        write(out / (request['label'] + '_URDF_binding.json'), binding)
        if not binding['pass']:
            raise RuntimeError('Read-only original ARX URDF disagrees with live link frames')
        geometry = read(request['surface_geometry'])
        # A missing/corrupted geometry record is an infrastructure stop, not a
        # candidate rejection that would allow another physical dispatch.
        target_check = SurfaceChecker(geometry) if insertion else None
        try:
            q = self.torch.tensor(bounded_planning_start([measured[n] for n in names], limits),
                                  device='cuda', dtype=self.torch.float32)
            if not checker.validate(q[None, None]).all():
                raise Rejection('microstep_measured_start_collision_in_unchanged_model',
                    {'mode': mode, 'target_excluded': True, 'closed_table_margin_m': .008 if not insertion else 0.,
                     'collision_is_not_proven_to_be_grasped_target_contact': True})
            actual = np.asarray(snapshot['link_world']['right_eef_link'])
            fit = pose_error(self.fk(planner, q, snapshot), actual)
            if fit['translation_m'] >= .002 or fit['rotation_rad'] >= .01:
                raise RuntimeError('Live ARX/planner FK mismatch at microstep start')
            goals, alignment_end = cartesian_knots(actual, request['goal_world'], insertion=insertion)
            inverse = np.linalg.inv(snapshot['link_world']['body4'])
            joints = [q]
            for goal in goals[1:]:
                if time.monotonic() - start_time > 300:
                    raise Rejection('finite_cartesian_solver_time_exhausted')
                q = self.ik(goal, joints[-1], planner, checker, inverse, seed=joints[-1].tolist())
                error = pose_error(self.fk(planner, q, snapshot), goal)
                if error['translation_m'] > .0005 or error['rotation_rad'] > np.deg2rad(.2):
                    raise Rejection('microstep_IK_fit_rejected', error)
                joints.append(q)
            path = self.torch.stack(joints)
            limit_tensor = self.torch.tensor(limits, device='cuda', dtype=path.dtype)
            violation = float(self.torch.maximum(limit_tensor[:, 0] - path, path - limit_tensor[:, 1]).clamp_min(0).max())
            if violation > 1e-6:
                raise Rejection('microstep_joint_limit')
            a = self.torch.arange(10, device='cuda', dtype=path.dtype) / 10
            dense = self.torch.cat([(path[:-1, None] * (1 - a[None, :, None]) +
                                     path[1:, None] * a[None, :, None]).reshape(-1, len(names)), path[-1:]])
            invalid = sum(int((~checker.validate(chunk[:, None])).sum()) for chunk in dense.split(64))
            if invalid:
                raise Rejection('microstep_dense_original_collision', {'invalid_samples': invalid})
            target_evidence = []
            alignment_dense = alignment_end * 10
            end = np.asarray(request['goal_world'])
            axis = end[:3, 2] if insertion else np.array([0., 0., 1.])
            origin = goals[alignment_end][:3, 3]
            maximum_lateral, maximum_angle = 0., 0.
            for index, values in enumerate(dense.detach().cpu().numpy()):
                state = copy.deepcopy(snapshot)
                local = {**measured, **dict(zip(names, values.tolist()))}
                poses = urdf.poses(local, snapshot['link_world']['body4'])
                state['link_world'].update({k: v.tolist() for k, v in poses.items() if k in state['link_world']})
                if index >= alignment_dense:
                    tool = poses['right_eef_link']
                    delta = tool[:3, 3] - origin
                    lateral = float(np.linalg.norm(delta - axis * (delta @ axis)))
                    angle = pose_error(tool, end)['rotation_rad']
                    maximum_lateral, maximum_angle = max(maximum_lateral, lateral), max(maximum_angle, angle)
                    if lateral > .00075 or angle > np.deg2rad(.35):
                        raise Rejection('microstep_not_axial_after_dense_FK', {'lateral_m': lateral, 'rotation_rad': angle})
                if target_check is not None:
                    result = target_check.check(state)
                    if not result['pass']:
                        raise Rejection('microstep_original_target_piece_intersection', {'index': index, 'check': result})
                    target_evidence.append(result['minimum_distance_m'])
                if time.monotonic() - start_time > 600:
                    raise Rejection('finite_cartesian_dense_check_time_exhausted')
            qnumpy = path.detach().cpu().numpy()
            times, derivatives = command_times(qnumpy, 8. if insertion else 12.)
            self.visual_path(request, planner, path, out)
            certificate = {'success': True, 'stage': request['label'], 'invalid_samples': 0,
                'dense_samples': len(dense), 'max_limit_exceed_rad': violation,
                'method': 'Frozen ARX sequential IK at 1mm Cartesian knots, dense collision/FK validation, bounded timing',
                'maximum_dense_axis_deviation_m': maximum_lateral, 'maximum_dense_rotation_rad': maximum_angle,
                'actual_command_finite_difference_maxima': derivatives, 'execution_duration_s': float(times[-1]),
                'native_target_check_samples': len(target_evidence),
                'minimum_native_target_distance_m': min(target_evidence) if target_evidence else None,
                'surface_geometry_sha256': geometry['geometry_sha256'], 'original_URDF_sha256': sha(original_urdf),
                'prepose_residual_alignment_knots': alignment_end if insertion else 0,
                'closed_table_margin_m': .008 if not insertion else 0., 'physics_or_model_modified': False,
                'planning_wall_seconds': time.monotonic() - start_time, 'physical_success_claim': False}
            write(out / (request['label'] + '_optimization.json'), certificate)
            return {'command': {'id': request['command_id'], 'joint_names': names, 'position': qnumpy.tolist(),
                    'time_from_start': times.tolist(), 'gripper_target': -3.2 if insertion else .1,
                    'position_tolerance_rad': .005, 'label': request['label'], 'planner': certificate['method']},
                    'desired_world': request['goal_world'], 'phase': request['phase'], 'cartesian_certificate': certificate}
        except (Rejection, ValueError) as error:
            result = {'rejected': str(error), 'details': getattr(error, 'details', {}), 'phase': request['phase'],
                      'physical_command_generated': False, 'frozen_collision_margins_retained': True}
            write(out / (request['label'] + '_microstep_rejection.json'), result)
            return result

    def handle(self, request):
        self.authorize()
        self.current_output = Path(request['output']) if request.get('output') else Path(request['scene']).parent
        self.current_label = request.get('label', request['action'])
        if request['action'] == 'closure_validate':
            return self.closure(request)
        if request['action'] == 'axial_seed_insert':
            return self.cartesian(request, insertion=True)
        if request['action'] == 'plan' and request.get('phase') == 'lift':
            result = super().handle(request)
            if not result.get('rejected'):
                return result
            from .advisor31_rules import fingers_touching
            closure = Path(request['output']) / 'close_execution.json'
            if not closure.exists() or len(fingers_touching(read(closure))) < 2:
                return result
            changed = copy.deepcopy(request)
            measured = np.array(request['snapshot']['link_world']['right_eef_link'])
            measured[2, 3] += .12
            changed['goal_world'] = measured.tolist()
            fallback = self.cartesian(changed, insertion=False)
            write(self.current_output / 'lift_fallback_receipt.json', {'original_planner': result,
                  'microstep_result': fallback, 'bilateral_closure_receipt_sha256': sha(closure)})
            return fallback
        return super().handle(request)


if __name__ == '__main__':
    from scripts.curobo_arx.seed_augmentation import worker
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--registration', required=True)
    a = p.parse_args()
    worker.Teacher = Teacher
    worker.serve(a.directory, read(a.registration))
