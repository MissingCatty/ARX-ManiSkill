"""Full-pose indexing, physical frame bindings and fail-closed station search."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, read_json, sha256, write_json
from modules.curobo.dual_arm.model import ArmModel
from modules.mobile_manipulation.inverse_reachability import InverseReachability
from modules.mobile_manipulation.placement import pose_matrix
from modules.mobile_manipulation.station_search import StationSearch
from modules.mobile_manipulation.hand_table_precheck import HandTablePrecheck
from modules.reachability.orientation_index import OrientationIndex, gravity_vectors


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT/'tmp/tests'; parent.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=parent, prefix='inverse_')
        cls.root = Path(cls.temp.name)
        cls.model = ArmModel()
        cls.action = np.array(read_json(ROOT/'simulation/robots/arx/runtime/action_tcp.json')['T_eef_action'])
        cls.settings = read_json(ROOT/'configs/experiments/sam3_graspgenx_v4.json')['placement']
        cls.options = read_json(ROOT/'configs/experiments/inverse_reachability_placement.json')
        initial = read_json(ROOT/'modules/arx_core/configs/runtime/arx_initial_state.json')
        values = dict(zip(initial['joint_names'], initial['joint_positions']))
        names = [n for n, j in cls.model.joints.items() if j.get('type') != 'fixed']
        cls.state = {'joint_names': names, 'qpos': [values[n] for n in names],
            'joint_limits': [[float(cls.model.joints[n].find('limit').get(k)) for k in ['lower', 'upper']]
                             for n in names],
            'links': {'base_link': pose_matrix(-1.05, .25, 0., -.5).tolist()}}
        cls.state['qpos'][names.index('joint4')] = .3
        # A small real-URDF witness bank also exercises independent FK.
        rng = np.random.default_rng(125)
        mapdir = cls.root/'map'; mapdir.mkdir()
        meta = deepcopy(read_json(ROOT/'tmp/reachability/products/前方精细_20261121_01/地图元数据.json'))
        bounds = np.asarray(meta['joint_limits'])
        data = np.zeros((160, len(meta['fields'])), dtype=np.float32)
        for i in range(len(data)):
            q = rng.uniform(bounds[:, 0]+.01, bounds[:, 1]-.01)
            data[i, :6] = q
            state = deepcopy(cls.state)
            state['links'] = {'base_link': np.eye(4).tolist()}
            for n, value in zip(meta['joint_names'], data[i, :6]):
                state['qpos'][names.index(n)] = float(value)
            frames = cls.model.frames(state)
            pose = np.linalg.inv(frames['body4'])@frames['right_eef_link']@cls.action
            data[i, 6:9] = pose[:3, 3]
            data[i, 9:13] = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        meta['witnesses'] = len(data)
        data.tofile(mapdir/meta['witness_file'])
        write_json(mapdir/'地图元数据.json', meta)
        write_json(mapdir/'完成.json', {'passed': True, 'file_sha256': {
            n: sha256(mapdir/n) for n in [meta['witness_file'], '地图元数据.json']}})
        cls.mapdir = mapdir
        OrientationIndex.build(mapdir, cls.root/'index')
        cls.index = OrientationIndex(mapdir, cls.root/'index')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def generator(self):
        return InverseReachability(self.index, self.model, self.action, self.options)

    def candidate(self, side='right', yaw=.0):
        state = deepcopy(self.state)
        state['links'] = {'base_link': pose_matrix(-1.05, .25, yaw, -.5).tolist()}
        arm_names = [f'{side}_joint{i}' for i in range(21 if side == 'right' else 11, 27 if side == 'right' else 17)]
        for n, value in zip(arm_names, self.index.bank[0, :6]):
            state['qpos'][state['joint_names'].index(n)] = float(value)
        eef = self.model.frames(state)[side+'_eef_link']
        contact = eef.copy(); contact[:3, 3] += .043*eef[:3, 2]
        return {'pose': contact.tolist(), 'score': .9}, state

    def test_index_is_an_exact_superset_of_gravity_cone(self):
        rng = np.random.default_rng(44)
        rotations = Rotation.random(30, random_state=rng).as_matrix()
        rotations = np.r_[rotations, Rotation.from_quat(self.index.bank[:12, [10, 11, 12, 9]]).as_matrix()]
        gravity = gravity_vectors(self.index.bank[:, 9:13])
        for goal in rotations:
            expected = np.flatnonzero(gravity@goal[2] >= np.cos(.0300001))
            actual = self.index.query(goal, .03)
            np.testing.assert_array_equal(np.sort(actual), expected)

    def test_gravity_alone_does_not_accept_wrong_yaw(self):
        candidate, state = self.candidate()
        candidate['pose'] = (np.asarray(candidate['pose']) @
            np.block([[Rotation.from_euler('z', np.pi).as_matrix(), np.zeros((3, 1))],
                      [np.zeros((1, 3)), np.ones((1, 1))]])).tolist()
        proposals, _ = self.generator().generate(state, [candidate], self.settings)
        self.assertFalse(any(p['source_map_row'] == 0 for p in proposals))

    def test_right_frame_tcp_offset_and_joint_witness(self):
        candidate, state = self.candidate()
        original = deepcopy(candidate)
        proposals, _ = self.generator().generate(state, [candidate], self.settings)
        match = next(p for p in proposals if p['arm'] == 'right' and p['source_map_row'] == 0)
        self.assertLess(match['base_travel_m'], 1e-6)
        self.assertLess(match['witness']['position_error_m'], 1e-6)
        self.assertLess(match['witness']['orientation_error_rad'], 1e-6)
        self.assertEqual(candidate, original)

    def test_left_map_uses_mount_translation_without_reflection(self):
        candidate, state = self.candidate('left', .12)
        proposals, _ = self.generator().generate(state, [candidate], self.settings)
        match = next(p for p in proposals if p['arm'] == 'left' and p['source_map_row'] == 0)
        self.assertLess(match['base_travel_m'], 1e-6)
        self.assertAlmostEqual(match['lift_m'], .3, places=6)

    def test_yaw_wrap_preserves_small_allowed_delta(self):
        candidate, state = self.candidate(yaw=np.pi-.01)
        proposals, _ = self.generator().generate(state, [candidate], self.settings)
        match = next(p for p in proposals if p['arm'] == 'right' and p['source_map_row'] == 0)
        self.assertLess(match['base_travel_m'], 1e-6)

    def test_robot_or_object_truth_does_not_replace_perceived_grasp(self):
        candidate, state = self.candidate()
        a, _ = self.generator().generate(state, [candidate], self.settings)
        state['object_world'] = (np.eye(4)*900).tolist()
        state['object_center'] = [999, 999, 999]
        b, _ = self.generator().generate(state, [candidate], self.settings)
        self.assertEqual(a, b)

    def test_lift_and_travel_bounds_are_preserved(self):
        candidate, state = self.candidate()
        moved = deepcopy(candidate)
        moved['pose'][0][3] += 10.
        moved['pose'][2][3] += 10.
        proposals, report = self.generator().generate(state, [moved], self.settings)
        self.assertEqual(proposals, [])
        self.assertTrue(report['unqueried_or_uncovered_candidates_retained_for_fallback'])

    def test_wrong_tcp_and_relaxed_tolerances_are_rejected(self):
        tcp = self.action.copy(); tcp[2, 3] += .001
        with self.assertRaises(ValueError):
            InverseReachability(self.index, self.model, tcp, self.options)
        with self.assertRaises(ValueError):
            InverseReachability(self.index, self.model, self.action,
                                {**self.options, 'orientation_tolerance_rad': .05})

    def test_invalid_rotation_rejected(self):
        for rotation in [np.diag([1, 1, -1]), np.eye(3)*2, np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                self.index.query(rotation, .03)

    def test_index_corruption_does_not_become_map_miss(self):
        with patch('modules.reachability.orientation_index.sha256', return_value='corrupt'):
            with self.assertRaisesRegex(ValueError, 'metadata'):
                OrientationIndex(self.mapdir, self.root/'index')

    def test_bad_witness_cannot_pass_independent_fk(self):
        candidate, state = self.candidate()
        wrong_goal = np.asarray(candidate['pose']).copy(); wrong_goal[0, 3] += .02
        with self.assertRaisesRegex(ValueError, 'independent'):
            self.generator().verify(state, state['links']['base_link'], .3,
                self.index.bank[0, :6], 'right', wrong_goal)

    def test_later_batch_keeps_original_candidate_indices(self):
        candidate, state = self.candidate()
        proposals, report = self.generator().generate(state, [candidate, candidate], self.settings, start=1)
        self.assertTrue(proposals)
        self.assertEqual({p['candidate_index'] for p in proposals}, {1})
        self.assertEqual(report['first_candidate_index'], 1)

    def test_hand_precheck_uses_each_physical_arm_original_meshes(self):
        state = deepcopy(self.state)
        state['scene_cuboids'] = {'table': {'center': [0, 0, -.25], 'dims': [2, 2, .5]}}
        original = deepcopy(state)
        checker = HandTablePrecheck(self.model, state, .002)
        self.assertEqual(state, original)
        for side in ['right', 'left']:
            self.assertTrue(all(link.startswith(side) for link, _, _ in checker.local[side]))


class HandPrecheckTests(unittest.TestCase):
    def checker(self, point, finger=True):
        checker = object.__new__(HandTablePrecheck)
        checker.local = {'right': [('finger' if finger else 'wrist', np.array([point]), finger)]}
        checker.center, checker.half = np.array([0, 0, -.25]), np.array([1, 1, .25])
        checker.maximum_tip_depth_m = .002
        return checker

    def check(self, checker):
        return checker.check({'arm': 'right', 'witness': {'achieved_eef_world': np.eye(4).tolist()}})

    def test_same_two_mm_finger_allowance_is_preserved(self):
        self.assertTrue(self.check(self.checker([0, 0, -.0015]))['passed'])
        self.assertFalse(self.check(self.checker([0, 0, -.003]))['passed'])

    def test_wrist_gets_no_table_allowance(self):
        self.assertFalse(self.check(self.checker([0, 0, -.0002], False))['passed'])

    def test_table_side_boundary_is_not_an_infinite_plane(self):
        self.assertTrue(self.check(self.checker([1.1, 0, -.02]))['passed'])

    def test_fk_achieved_pose_is_used_instead_of_nominal_pose(self):
        p = {'arm': 'right', 'witness': {'achieved_eef_world': np.eye(4).tolist()}}
        p['witness']['achieved_eef_world'][2][3] = .02
        self.assertTrue(self.checker([0, 0, -.01]).check(p)['passed'])

    def test_fast_geometry_matches_individual_vertex_oracle(self):
        rng = np.random.default_rng(52)
        checker = self.checker([0, 0, 0])
        checker.local['right'] = [('palm', rng.uniform(-.06, .06, (29, 3)), False),
                                  ('finger', rng.uniform(-.04, .04, (37, 3)), True)]
        for _ in range(100):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
            pose[:3, 3] = rng.uniform([-.1, -.1, -.03], [.1, .1, .08])
            expected = True
            for _, vertices, finger in checker.local['right']:
                for vertex in vertices:
                    p = pose[:3, :3]@vertex+pose[:3, 3]
                    allowance = .002 if finger else 0.
                    if (np.all(p > checker.center-checker.half+1e-7) and
                            np.all(p < checker.center+checker.half-1e-7) and
                            p[2] < checker.center[2]+checker.half[2]-allowance-1e-7):
                        expected = False
            actual = checker.check({'arm': 'right', 'witness': {'achieved_eef_world': pose.tolist()}})
            self.assertEqual(actual['passed'], expected)


class SearchTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'tmp/tests'; parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent, prefix='station_search_')
        self.directory = Path(self.temp.name)
        self.state = {'links': {'base_link': np.eye(4).tolist()}}
        self.settings = {'maximum_full_planning_stations': 3, 'transit_clearance_m': .003}
        self.options = {'maximum_destination_checks': 10, 'maximum_endpoint_checks': 10}
        self.pool = {'candidates': [{'pose': np.eye(4).tolist(), 'score': 1.},
                                    {'pose': np.eye(4).tolist(), 'score': .8}]}
        self.proposal = {'base_world': np.eye(4).tolist(), 'lift_m': .3,
                         'arm': 'right', 'candidate_index': 1, 'source': 'inverse_reachability'}
        self.patchers = [patch('modules.mobile_manipulation.station_search.SweepChecker'),
                         patch('modules.mobile_manipulation.station_search.hypothetical_state', return_value=self.state),
                         patch('modules.mobile_manipulation.station_search.station_proposals', return_value=[self.proposal]*4)]
        checker = self.patchers[0].start().return_value
        checker.check.return_value = {'passed': True}
        checker.path.return_value = ({'passed': True}, [{'time_s': 0}])
        self.checker = checker
        self.patchers[1].start(); self.patchers[2].start()
        from unittest.mock import Mock
        self.backend = Mock()
        self.backend.endpoint.return_value = {'passed': True}
        self.backend.plan.return_value = {'success': True, 'selected': {
            'index': 0, 'arm': 'right', 'contact_pose_world': np.eye(4).tolist()}}
        self.generator = Mock()
        self.generator.generate.return_value = ([self.proposal], {'query_seconds': .01})

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def search(self):
        return StationSearch(None, self.settings, self.backend, generator=self.generator, options=self.options)

    def test_map_success_preserves_source_index_and_pool(self):
        original = deepcopy(self.pool)
        chosen, report = self.search().select(self.state, self.pool, [0, 0, 0], self.directory)
        self.assertEqual(chosen['plan_result']['selected']['index'], 1)
        self.assertEqual(self.pool, original)
        self.assertEqual(report['full_planning_calls'], 1)
        self.assertFalse(report['fallback_used'])

    def test_map_missing_uses_complete_original_pool(self):
        self.generator.generate.return_value = ([], {'query_seconds': .01})
        chosen, report = self.search().select(self.state, self.pool, [0, 0, 0], self.directory)
        work = Path(chosen['work'])
        self.assertEqual(read_json(work/'candidates/candidate_pool.json'), self.pool)
        self.assertTrue(report['fallback_used'])
        self.backend.plan.assert_called_once_with(work, arm=None)

    def test_collision_falls_back_without_consuming_full_plan_budget(self):
        self.backend.endpoint.return_value = {'passed': False}
        chosen, report = self.search().select(self.state, self.pool, [0, 0, 0], self.directory)
        self.assertTrue(chosen['success'])
        self.assertTrue(report['fallback_used'])
        self.assertEqual(report['endpoint_checks'], 1)
        self.assertEqual(report['full_planning_calls'], 1)

    def test_full_planning_budget_shared_with_fallback(self):
        self.generator.generate.return_value = ([self.proposal]*5, {'query_seconds': .01})
        self.backend.plan.return_value = {'success': False}
        chosen, report = self.search().select(self.state, self.pool, [0, 0, 0], self.directory)
        self.assertIsNone(chosen)
        self.assertEqual(report['full_planning_calls'], 3)
        self.assertFalse(report['fallback_used'])

    def test_resource_fault_stops_without_fallback(self):
        self.backend.endpoint.side_effect = RuntimeError('CUDA device fault')
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.search().select(self.state, self.pool, [0, 0, 0], self.directory)
        self.backend.plan.assert_not_called()

    def test_transit_collision_is_rejected_before_full_planning(self):
        self.checker.path.return_value = ({'passed': False, 'sample': 4}, [])
        chosen, report = self.search().select(self.state, self.pool, [0, 0, 0], self.directory)
        self.assertIsNone(chosen)
        self.assertEqual(report['full_planning_calls'], 0)
        self.backend.plan.assert_not_called()

    def test_fast_mesh_rejection_avoids_native_binding_and_preserves_fallback(self):
        from unittest.mock import Mock
        fast = Mock(); fast.check.return_value = {'passed': False}
        search = self.search(); search.precheck_factory = lambda state: fast
        chosen, report = search.select(self.state, self.pool, [0, 0, 0], self.directory)
        self.backend.endpoint.assert_not_called()
        self.assertTrue(report['fallback_used'])
        self.assertEqual(read_json(Path(chosen['work'])/'candidates/candidate_pool.json'), self.pool)


if __name__ == '__main__':
    unittest.main()
