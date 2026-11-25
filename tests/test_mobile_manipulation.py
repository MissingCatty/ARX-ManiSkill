"""Placement frames, occlusion safety, limits and conservative transit checks."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from arx_project.layout import ROOT, read_json
from modules.curobo.dual_arm.model import ArmModel
from modules.mobile_manipulation.placement import (
    backproject, hypothetical_state, interpolate, pose_matrix, station_proposals, SweepChecker)


class MobileGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ArmModel()
        cls.settings = read_json(ROOT/'configs/experiments/sam3_graspgenx_v4.json')['placement']
        initial = read_json(ROOT/'modules/arx_core/configs/runtime/arx_initial_state.json')
        values = dict(zip(initial['joint_names'], initial['joint_positions']))
        names = [n for n, node in cls.model.joints.items() if node.get('type') != 'fixed']
        cls.state = {'joint_names': names, 'qpos': [values[n] for n in names],
            'joint_limits': [[float(cls.model.joints[n].find('limit').get(k)) for k in ['lower', 'upper']] for n in names],
            'links': {'base_link': pose_matrix(-1.05, .25, 0., -.5).tolist()},
            'object_world': np.eye(4).tolist(), 'object_center': [-.45, .17, .03],
            'scene_cuboids': {'table': {'center': [-.12, 0., -.25], 'dims': [1.209, 2.418, .5]}}}
        frames = cls.model.frames(cls.state)
        cls.state['links'].update({n: frames[n].tolist() for n in ['body4', 'right_eef_link', 'left_eef_link']})

    def test_actual_urdf_base_yaw_and_lift_transform_both_arms(self):
        before = self.model.frames(self.state)
        goal = pose_matrix(-1.08, .22, .12, -.5)
        after = hypothetical_state(self.model, self.state, goal, .35)
        delta = goal@np.linalg.inv(np.asarray(self.state['links']['base_link']))
        for name in ['right_eef_link', 'left_eef_link', 'body4']:
            expected = delta@before[name]
            expected[2, 3] += .35-self.state['qpos'][0]
            np.testing.assert_allclose(after['links'][name], expected, atol=1e-6)
        self.assertEqual(after['object_world'], self.state['object_world'])

    def test_proposals_obey_lift_limits_and_do_not_read_object_truth(self):
        bad_truth = deepcopy(self.state)
        bad_truth['object_center'] = [900., 900., 900.]
        target = np.array([-.48, .20, .07])
        a = station_proposals(self.state, target, self.settings)
        b = station_proposals(bad_truth, target, self.settings)
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)
        for row in a:
            self.assertGreaterEqual(row['lift_m'], .002)
            self.assertLessEqual(row['lift_m'], .458001)
            self.assertLessEqual(np.linalg.norm(np.array(row['base_world'])[:2, 3]-np.array(self.state['links']['base_link'])[:2, 3]), .65)

    def test_depth_invalid_is_rejected_instead_of_zero_distance(self):
        with self.assertRaisesRegex(ValueError, 'insufficient_valid_target_depth'):
            backproject(np.ones((10, 10)), np.zeros((10, 10)), np.eye(3), np.eye(4))
        depth = np.ones((10, 10))*.5
        p = backproject(np.ones((10, 10)), depth, np.diag([10., 10., 1.]), np.eye(4))
        self.assertEqual(p.shape, (100, 3))
        np.testing.assert_allclose(p[:, 2], .5)

    def test_transit_speed_and_exact_endpoints(self):
        goal = {'base_world': pose_matrix(-1.15, .35, .17, -.5).tolist(), 'lift_m': .32}
        path = list(interpolate(self.state, goal, self.settings))
        np.testing.assert_allclose(path[0]['base_world'], self.state['links']['base_link'], atol=2e-6)
        np.testing.assert_allclose(path[-1]['base_world'], goal['base_world'], atol=1e-9)
        self.assertAlmostEqual(path[-1]['lift_m'], .32)
        times = np.array([r['time_s'] for r in path])
        xy = np.array([r['base_world'] for r in path])[:, :2, 3]
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)/np.diff(times)
        self.assertLessEqual(speed.max(), self.settings['base_speed_mps']+1e-6)

    def test_original_robot_collision_rejects_base_inside_table(self):
        checker = SweepChecker(self.model, self.state)
        self.assertTrue(checker.check(self.state['links']['base_link'], self.state['qpos'][0])['passed'])
        bad = pose_matrix(-.65, .25, 0., -.5)
        result = checker.check(bad, self.state['qpos'][0])
        self.assertFalse(result['passed'])
        self.assertEqual(result['obstacle'], 'table')

    def test_clear_endpoints_do_not_skip_middle_of_sweep(self):
        checker = object.__new__(SweepChecker)
        checker.state = self.state
        checker.clearance = .003
        calls = []
        def check(base, lift):
            calls.append(base)
            # A blocked interior segment must reject the entire transit.
            return {'passed': len(calls) != 4, 'obstacle': 'test_obstacle'}
        checker.check = check
        goal = {'base_world': pose_matrix(-1.1, .3, 0., -.5).tolist(), 'lift_m': .4}
        result, samples = checker.path(goal, self.settings)
        self.assertFalse(result['passed'])
        self.assertEqual(result['sample'], 3)
        self.assertGreater(len(samples), len(calls))

    def test_robot_geometry_cache_is_exact_and_scene_obstacles_are_fresh(self):
        model = ArmModel()
        first = SweepChecker(model, self.state)
        expected = [(n, v.copy(), normals.copy(), edges.copy()) for n, v, normals, edges in first.parts]
        with patch.object(model, 'parts', wraps=model.parts) as parts:
            second = SweepChecker(model, self.state)
            parts.assert_not_called()
        self.assertIs(first.parts, second.parts)
        for a, b in zip(expected, second.parts):
            self.assertEqual(a[0], b[0])
            for va, vb in zip(a[1:], b[1:]):
                np.testing.assert_array_equal(va, vb)
        changed = deepcopy(self.state)
        changed['scene_cuboids']['table']['center'][0] = -1.05
        checker = SweepChecker(model, changed)
        self.assertIs(checker.parts, first.parts)
        self.assertFalse(checker.check(changed['links']['base_link'], changed['qpos'][0])['passed'])


if __name__ == '__main__':
    unittest.main()
