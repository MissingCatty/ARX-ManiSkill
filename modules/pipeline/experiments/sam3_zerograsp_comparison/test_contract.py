"""Offline fault and pairing checks; no model, CUDA or simulator calls."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from arx_project.layout import ROOT, sha256, write_json
from modules.zerograsp.arx_grasp_adapter import common_candidates
from .cohort import INPUTS, check_pair, eligible_rows
from .metrics import distribution
from .replay import state_difference, validate_difference


class ContractTests(unittest.TestCase):
    def test_tcp43_contact_frame_and_width_are_preserved(self):
        row = np.r_[.8, .054, .02, .015, np.eye(3).ravel(), [.2, -.1, .5], -1.]
        camera = np.eye(4)
        camera[:3, 3] = [.3, .1, .2]
        candidate = common_candidates(row[None], camera)[0]
        contact = np.asarray(candidate['pose'])
        eef = contact.copy(); eef[:3, 3] -= .043*contact[:3, 2]
        np.testing.assert_allclose(contact[:3, 3], [.515, 0., .7])
        np.testing.assert_allclose(eef[:3, 3], [.472, 0., .7])
        self.assertEqual(candidate['width'], .054)
        self.assertFalse(candidate['metadata']['finger_flip'])

    def test_native_score_order_without_candidate_filter_or_flip(self):
        rows = np.tile(np.r_[.2, .07, .02, .01, np.eye(3).ravel(), [.1, .2, .3], -1.], (4, 1))
        rows[:, 0] = [.2, .9, .9, .1]
        result = common_candidates(rows, np.eye(4))
        self.assertEqual([r['metadata']['postprocessed_row_index'] for r in result], [1, 2, 0, 3])
        self.assertEqual(len(result), 4)
        self.assertEqual(common_candidates(np.empty((0, 17)), np.eye(4)), [])

    def test_nonfinite_candidate_is_record_fault(self):
        rows = np.zeros((1, 17)); rows[0, 0] = np.nan
        with self.assertRaises(ValueError):
            common_candidates(rows, np.eye(4))

    def test_restore_difference_does_not_allow_a_different_object_position(self):
        state = {'joint_names': ['a'], 'qpos': [0.], 'links': {'base_link': np.eye(4).tolist()},
            'camera_world_optical': np.eye(4).tolist(),
            'objects': {'target': {'pose': [0., 0., 0., 1., 0., 0., 0.]}}}
        validate_difference(state_difference(state, deepcopy(state)))
        bad = deepcopy(state); bad['objects']['target']['pose'][0] = .002
        with self.assertRaises(RuntimeError):
            validate_difference(state_difference(state, bad))

    def test_restore_rotation_and_robot_joint_errors_stop(self):
        difference = {'joint_max_difference': 0., 'base_matrix_max_difference': 0.,
            'camera_matrix_max_difference': 0., 'object_translation_difference_m': {'target': 0.},
            'object_rotation_difference_rad': {'target': .1}}
        with self.assertRaises(RuntimeError):
            validate_difference(difference)
        difference['object_rotation_difference_rad']['target'] = 0.
        difference['joint_max_difference'] = .003
        with self.assertRaises(RuntimeError):
            validate_difference(difference)

    def test_latency_statistics_include_sample_count_and_no_invented_zeros(self):
        self.assertEqual(distribution([None])['samples'], 0)
        value = distribution([1., 2., 3., None])
        self.assertEqual(value['median_seconds'], 2.)
        self.assertAlmostEqual(value['p90_seconds'], 2.8)

    def test_cohort_keeps_grasp_failures_and_common_station_failure(self):
        parent = ROOT/'tmp/sam3_zerograsp_comparison'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='offline_', dir=parent) as name:
            root = Path(name)
            schedule = [{'id': key} for key in ['good', 'empty_grasp', 'station', 'sam', 'interrupted', 'unstarted']]
            write_json(root/'registration.json', {'schedule': schedule})
            for entry in schedule[:-1]:
                key = entry['id']; trial = root/'trials'/key
                result = {'id': key, 'success': key == 'good', 'planning_success': key in ['good', 'empty_grasp'],
                    'reason': {'sam': 'coarse_sam3_no_target_mask', 'station': 'unable_to_plan'}.get(key, 'no_bilateral_target_contact'),
                    'stationing': {'failed_stage': 'high_workspace_positioning'}}
                write_json(trial/'result.json', result)
                write_json(root/'claims'/(key+'.json'), {'state': 'claimed' if key == 'interrupted' else 'complete',
                    'result_sha256': sha256(trial/'result.json')})
                observation = trial/'stationing/coarse' if key == 'station' else trial
                for f in INPUTS:
                    path = observation/f; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{}')
                write_json(observation/'sam3.json', {'target_found': True})
                write_json(trial/'stationing/prepared_plan_handoff.json', {
                    'preferred_arm': None, 'workspace_scope': 'shared', 'prepared_target_world': [0, 0, 0],
                    'verified_high_cell': [0, 0, 0], 'workspace_resolution_m': .02})
            pairs, excluded = eligible_rows(root)
            self.assertEqual([p['id'] for p in pairs], ['good', 'empty_grasp', 'station'])
            self.assertTrue(pairs[-1]['common_precondition_failure'])
            self.assertEqual([p['kind'] for p in excluded], ['sam3_failure', 'interrupted', 'not_started'])

    def test_altered_paired_image_is_a_record_fault(self):
        parent = ROOT/'tmp/sam3_zerograsp_comparison'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='offline_', dir=parent) as name:
            root = Path(name); image = root/'mask.png'; image.write_bytes(b'original')
            pair = {'id': 'case', 'frozen_observation': str(root.relative_to(ROOT)),
                'input_hashes': {'mask.png': sha256(image)}}
            image.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                check_pair(pair)


if __name__ == '__main__':
    unittest.main()
