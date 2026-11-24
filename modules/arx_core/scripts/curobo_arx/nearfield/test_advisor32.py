"""Offline failure cases for the proposed advisor32 local-only rules."""
import ast
import inspect
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from . import advisor32_rules as rules
from . import advisor32_spec as spec


class LocalRules(unittest.TestCase):
    def setUp(self):
        self.ee = np.eye(4)
        self.visual = {'target_pixels': 100, 'between_fingers': True, 'depth_gate_pass': True,
                       'target_direction_EE_xy': [1., 0.], 'predicted_near_saturation': False}

    def test_no_prior_or_registration_runtime_argument(self):
        tree = ast.parse(inspect.getsource(rules))
        banned = {'G0_world', 'source_Gstar_world', 'active_trial', 'active_seed', 'registration'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn(node.value, banned)
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, banned)
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, banned)

    def test_local_axes_transform(self):
        self.ee[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        target = rules.goal_from_delta(self.ee, [.01, 0, 0], [0, 0, .1])
        np.testing.assert_allclose(target[:3, 3], [0, .01, 0], atol=1e-12)

    def test_large_step_rejected_not_clipped(self):
        out = rules.compose_first_goal(self.ee, 'REFINE 30 0 3 0 0 0', .06)
        self.assertFalse(out['admission']['pass'])
        self.assertEqual(out['used_translation_m'], [.03, 0., .06])

    def test_rotation_limit(self):
        target = rules.goal_from_delta(self.ee, [0, 0, 0], [0, 0, math.radians(26)])
        self.assertFalse(rules.local_admission(self.ee, target)['pass'])

    def test_arbitrary_world_origin_has_same_admission(self):
        a = rules.goal_from_delta(self.ee, [.06, 0, 0], [0, 0, .2])
        first = rules.local_admission(self.ee, a)
        self.ee[:3, 3] += [100., -25., 3.]; a[:3, 3] += [100., -25., 3.]
        last = rules.local_admission(self.ee, a)
        self.assertEqual(first['pass'], last['pass'])
        self.assertAlmostEqual(first['translation_m'], last['translation_m'])

    def test_nan_prediction_rejected(self):
        with self.assertRaises(ValueError): rules.parse_prediction('D', 'REFINE nan 0 0 0 0 0')

    def test_invalid_rotation_rejected(self):
        self.ee[0, 0] = -1.
        with self.assertRaises(ValueError): rules.local_admission(self.ee, np.eye(4))

    def test_grasp_requires_visual_gate(self):
        self.visual['target_pixels'] = 0
        self.assertFalse(rules.second_decision('GRASP', self.visual)['request_closure'])

    def test_small_translation_with_large_rotation_does_not_close(self):
        out = rules.second_decision('REFINE 1 0 0 0 0 10', self.visual)
        self.assertFalse(out['request_closure'])

    def test_small_residual_requests_independent_validation(self):
        out = rules.second_decision('REFINE 1 0 0 0 0 1', self.visual)
        self.assertTrue(out['request_closure']); self.assertFalse(out['model_explicit_GRASP'])
        self.assertTrue(out['independent_closure_check_required'])

    def test_retreat_intercept(self):
        out = rules.second_decision('REFINE 0 0 -10 0 0 0', self.visual)
        self.assertTrue(out['request_closure'])
        self.assertIn('backward_motion', out['reasons'])

    def test_target_opposite_lateral(self):
        self.assertTrue(rules.second_decision('REFINE -10 0 0 0 0 0', self.visual)['request_closure'])
        self.assertFalse(rules.second_decision('REFINE 10 0 0 0 0 0', self.visual)['request_closure'])

    def test_early_stop_cannot_bypass_in_hand_depth(self):
        self.visual['depth_gate_pass'] = False
        self.assertFalse(rules.second_decision('REFINE -10 0 0 0 0 0', self.visual)['request_closure'])

    def test_hidden_pose_in_visual_schema_rejected(self):
        self.visual['G0_world'] = np.eye(4).tolist()
        with self.assertRaises(ValueError): rules.second_decision('GRASP', self.visual)

    def test_empty_depth_fails(self):
        with self.assertRaises(ValueError): rules.backproject(np.zeros((2, 2)), np.ones((2, 2), dtype=bool), np.eye(3))

    def test_optical_and_EE_axes_are_not_assumed_equal(self):
        camera = np.eye(4); camera[:3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix()
        pts = np.array([[0., 0., .080], [.01, 0, .080]])
        out = rules.optical_first_advance(self.ee, camera, pts, 'REFINE 0 0 0 0 0 0')
        self.assertAlmostEqual(out['axial_advance_m'], (.080-.0235)/math.cos(math.radians(30)))
        self.assertFalse(out['admission']['pass'])

    def test_lift_retains_closed_pose_and_world_up(self):
        self.ee[:3, :3] = Rotation.from_euler('xyz', [30, 40, 20], degrees=True).as_matrix()
        rows = rules.lift_goals(self.ee, np.eye(3))
        self.assertEqual([r['rotation_degrees'] for r in rows[:2]], [0., 0.])
        self.assertEqual([r['height_m'] for r in rows[:2]], [.12, .10])
        for row in rows:
            self.assertLessEqual(row['rotation_degrees'], 10.+1e-9)
            p = np.asarray(row['goal_world'])[:3, 3]
            np.testing.assert_allclose(p-self.ee[:3, 3], [0, 0, row['height_m']], atol=1e-12)


class MeasuredStop(unittest.TestCase):
    @staticmethod
    def samples(n=11, v=1., drift=0.):
        return [rules.StopSample(i*.2, (0., 0., 0.), (i*.2*drift, 0.), (v, 0.)) for i in range(n)]

    def test_stationary_positions_with_high_reported_velocity(self):
        out = rules.settle(self.samples(), active=False, fault=None, elapsed_wall_s=2.)
        self.assertTrue(out['pass']); self.assertEqual(out['origin'], 'settled_by_position_window')
        self.assertEqual(out['reported_maximum_speed_rad_s'], 1.)

    def test_ten_5Hz_frames_are_only_1_8_seconds(self):
        self.assertFalse(rules.settle(self.samples(10), active=False, fault=None, elapsed_wall_s=2.)['pass'])

    def test_stationary_TCP_with_joint_motion_rejected(self):
        self.assertFalse(rules.settle(self.samples(drift=.02), active=False, fault=None, elapsed_wall_s=2.)['pass'])

    def test_active_trajectory_cannot_be_called_idle(self):
        self.assertFalse(rules.settle(self.samples(v=0.), active=True, fault=None, elapsed_wall_s=2.)['pass'])

    def test_fault_cannot_use_window(self):
        with self.assertRaises(RuntimeError): rules.settle(self.samples(), active=False, fault='fault', elapsed_wall_s=2.)

    def test_discontinuous_window_rejected(self):
        a = self.samples(); a = a[:5]+a[6:]
        self.assertFalse(rules.settle(a, active=False, fault=None, elapsed_wall_s=2.)['pass'])

    def test_duplicate_timestamp_rejected(self):
        a = self.samples(); a.append(a[-1])
        with self.assertRaises(ValueError): rules.settle(a, active=False, fault=None, elapsed_wall_s=2.)

    def test_bad_velocity_not_disguised_as_stationary(self):
        a = self.samples(); a[-1] = rules.StopSample(2., (0, 0, 0), (0, 0), (float('nan'), 0))
        with self.assertRaises(ValueError): rules.settle(a, active=False, fault=None, elapsed_wall_s=2.)

    def test_wait_is_bounded(self):
        out = rules.settle(self.samples(drift=.02), active=False, fault=None, elapsed_wall_s=10.)
        self.assertTrue(out['timeout']); self.assertFalse(out['pass'])

    def test_late_stationarity_does_not_override_hard_deadline(self):
        out = rules.settle(self.samples(v=0.), active=False, fault=None, elapsed_wall_s=10.)
        self.assertTrue(out['timeout']); self.assertFalse(out['pass'])

    def test_new_slot_cannot_start_without_isolation_evidence(self):
        fields = dict(record_intact=True, configuration_intact=True, services_healthy=True,
                      old_world_absent=True, new_world_reset_verified=True)
        self.assertTrue(rules.continuation_after_slot(**fields)['continue_next_slot'])
        for key in fields:
            self.assertFalse(rules.continuation_after_slot(**{**fields, key: False})['continue_next_slot'])


class FullStale(unittest.TestCase):
    def test_second_decision_does_not_revert_to_current_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/str(i) for i in range(6)]
            for p in paths: p.write_bytes(b'fixture')
            old = {'0': paths[:2], '1': paths[2:4]}
            for index in (0, 1):
                reply = spec.model_images('full_stale', index, paths[4:], old)
                self.assertEqual(reply['images'], list(map(str, old[str(index)])))
                self.assertTrue(reply['replaced'])

    def test_missing_second_donor_rejected(self):
        with self.assertRaises(ValueError): spec.model_images('full_stale', 1, ['a', 'b'], {'0': ['c', 'd']})

    def test_third_neural_decision_disallowed(self):
        with self.assertRaises(ValueError): spec.model_images('fresh', 2, ['a', 'b'], {})

    def test_geometry_baseline_never_queries_model(self):
        self.assertFalse(spec.model_images('g0', 0, [], {})['model_inference'])


if __name__ == '__main__':
    unittest.main()
