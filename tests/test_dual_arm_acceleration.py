"""Ranking ties, bounded losses, error propagation and all collision gates."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from arx_project.layout import ROOT, read_json, write_json
from modules.curobo.dual_arm.accelerated_adapter import (
    RankBoundPool, opening_samples, validate_sphere_batch)


class RankBoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='rank-bound-')
        self.root = Path(self.temp.name)
        for arm in ['right', 'left']:
            write_json(self.root/arm/'candidates/candidate_pool.json', {
                'candidates': [{'pose': np.eye(4).tolist(), 'score': 1.-i/6000, 'original_index': i}
                               for i in range(5048)]})

    def tearDown(self):
        self.temp.cleanup()

    def pool(self, results):
        class Pool:
            def request(inner, directory, **flags):
                result = deepcopy(results[Path(directory).name])
                candidates = read_json(Path(directory)/'candidates/candidate_pool.json')['candidates']
                if result.get('success') and result['selected']['index'] >= len(candidates):
                    return {'success': False, 'reason': 'prefix_exhausted'}
                return result
        return RankBoundPool(Pool())

    def test_second_arm_only_checks_winning_prefix_including_tie(self):
        pool = self.pool({'right': {'success': True, 'selected': {'index': 159}},
                          'left': {'success': False, 'reason': 'all_exhausted'}})
        pool.request(self.root/'right')
        result = pool.request(self.root/'left')
        data = read_json(self.root/'left/candidates/candidate_pool.json')
        self.assertEqual(len(data['candidates']), 160)
        self.assertEqual([r['original_index'] for r in data['candidates']], list(range(160)))
        self.assertFalse(result['full_pool_exhausted'])
        self.assertEqual(len(read_json(self.root/'right/candidates/candidate_pool.json')['candidates']), 5048)

    def test_equal_rank_remains_available_for_joint_travel_tie_break(self):
        pool = self.pool({side: {'success': True, 'selected': {'index': 0}} for side in ['right', 'left']})
        pool.request(self.root/'right')
        result = pool.request(self.root/'left')
        self.assertTrue(result['success'])
        self.assertEqual(result['search_bound']['retained_prefix_count'], 1)

    def test_better_second_arm_candidate_is_preserved(self):
        pool = self.pool({'right': {'success': True, 'selected': {'index': 159}},
                          'left': {'success': True, 'selected': {'index': 10}}})
        pool.request(self.root/'right')
        self.assertEqual(pool.request(self.root/'left')['selected']['index'], 10)
        self.assertEqual(pool.best_index, 10)

    def test_failed_first_arm_does_not_truncate_second_arm(self):
        pool = self.pool({'right': {'success': False, 'reason': 'all_exhausted'},
                          'left': {'success': True, 'selected': {'index': 5000}}})
        pool.request(self.root/'right')
        result = pool.request(self.root/'left')
        self.assertTrue(result['success'])
        self.assertEqual(result['search_bound']['retained_prefix_count'], 5048)

    def test_backend_fault_is_not_converted_to_candidate_failure(self):
        class Fault:
            def request(self, *args, **kwargs):
                raise RuntimeError('CUDA allocation failure')
        with self.assertRaisesRegex(RuntimeError, 'CUDA allocation'):
            RankBoundPool(Fault()).request(self.root/'right')


class OpeningBatchTests(unittest.TestCase):
    def test_file_registry_can_load_the_accelerated_factory(self):
        from arx_project.registry import load_adapter
        records = {'curobo_dual_arm': {'adapter': 'modules/curobo/dual_arm/accelerated_service.py'}}
        with patch('arx_project.registry.module_records', return_value=records):
            self.assertTrue(callable(load_adapter('curobo_dual_arm').create))

    def test_original_all_widths_and_asymmetric_observations_retained(self):
        state = {'planned_opening': {'target_per_finger_m': .044,
            'observed_solver_finger_positions_m': {'right_joint27': 0., 'right_joint28': .001}}}
        fractions, target, observed = opening_samples(state)
        self.assertEqual(len(fractions), 89)
        self.assertEqual(fractions[0], 0.)
        self.assertEqual(fractions[-1], 1.)
        for value in observed.values():
            widths = value+fractions*(target-value)
            self.assertLessEqual(np.diff(widths).max(), .0005+1e-12)

    def test_nonfinite_opening_is_not_admitted(self):
        with self.assertRaises(ValueError):
            opening_samples({'planned_opening': {'target_per_finger_m': .044,
                'observed_solver_finger_positions_m': {'right_joint27': float('nan')}}})

    def test_bound_self_and_world_failures_each_reject_their_sample(self):
        class Checker:
            self_collision_cost = object()
            collision_constraint = object()
            def setup_batch_tensors(self, batch, horizon):
                self.dimensions = batch, horizon
            def get_bound(self, q):
                assert q.is_contiguous(), 'Native Warp costs require packed joint tensors'
                out = torch.zeros_like(q); out[1, 0, 0] = 1.
                return out
            def get_self_collision(self, spheres):
                out = torch.zeros(spheres.shape[:2]+(1,)); out[2, 0, 0] = 1.
                return out
            def get_collision_constraint(self, spheres):
                out = torch.zeros(spheres.shape[:3]); out[3, 0, 0] = 1.
                return out
        checker = Checker()
        valid = validate_sphere_batch(checker, torch.zeros(4, 1, 871, 4), torch.zeros(1, 1, 6).expand(4, 1, 6))
        self.assertEqual(valid.flatten().tolist(), [True, False, False, False])
        self.assertEqual(checker.dimensions, (4, 1))

    def test_nan_sphere_is_record_fault(self):
        spheres = torch.zeros(2, 1, 871, 4); spheres[1, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            validate_sphere_batch(None, spheres, torch.zeros(2, 1, 6))


class ActivationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='activation-gates-')
        self.root = Path(self.temp.name)
        self.comparison, self.main = self.root/'comparison', self.root/'main'
        self.comparison.mkdir(); self.main.mkdir()
        write_json(self.comparison/'acceptance.json', {'passed': True, 'physical_grasps': 0,
            'paired_cases': [{'passed': True}]*3, 'opening_checks': [{'passed': True}]*3})
        write_json(self.main/'acceptance.json', {'passed': True, 'task_complete': True, 'completed': 100})
        write_json(self.main/'publication_complete.json', {'task_complete': True, 'audit_passed': True})
        write_json(self.main/'services_stopped.json', {'all_owned_exited': True})
        write_json(self.main/'started.json', {'pid': 999999999})

    def tearDown(self):
        self.temp.cleanup()

    def state(self):
        from modules.pipeline.experiments.dual_arm_grasp.activate_acceleration import readiness
        return readiness(self.comparison, self.main)

    def test_activation_waits_for_running_original_controller(self):
        write_json(self.main/'started.json', {'pid': os.getpid()})
        self.assertEqual(self.state(), 'waiting_for_main_shutdown')

    def test_partial_main_batch_cannot_activate(self):
        write_json(self.main/'acceptance.json', {'passed': True, 'task_complete': False, 'completed': 99})
        self.assertEqual(self.state(), 'main_acceptance_failed')

    def test_stop_is_preserved_and_blocks_activation(self):
        write_json(self.main/'STOP', {'reason': 'record fault'})
        self.assertEqual(self.state(), 'main_batch_stopped')
        self.assertTrue((self.main/'STOP').is_file())

    def test_comparison_requires_all_pairs_and_collision_checks(self):
        write_json(self.comparison/'acceptance.json', {'passed': True, 'physical_grasps': 0,
            'paired_cases': [], 'opening_checks': []})
        self.assertEqual(self.state(), 'comparison_failed')

    def test_only_completed_accepted_and_exited_main_is_ready(self):
        self.assertEqual(self.state(), 'ready')

    def transaction(self, validation):
        from arx_project.layout import sha256
        from modules.pipeline.experiments.dual_arm_grasp import activate_acceleration as activation
        paths = [self.root/'configs/modules.json', self.root/'configs/experiments/active_experiment.json',
                 self.root/'configs/experiments/scene_dataset.json', self.root/'configs/experiments/dual_arm_acceleration.json']
        write_json(paths[0], {'modules': {'curobo_dual_arm': {'adapter': 'old.py', 'policies': []}}})
        write_json(paths[1], {'profile_id': 'v4_original', 'configuration_hashes': {}})
        write_json(paths[2], {'grasp_tests_authorized': True, 'grasp_test_budget': 100})
        write_json(paths[3], {'activation': 'pending'})
        for name in [activation.ADAPTER, 'modules/curobo/dual_arm/accelerated_adapter.py']:
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
        hashes = {str(p): sha256(p) for p in paths}
        write_json(self.main/'registration.json', {'source_hashes': hashes})
        write_json(self.comparison/'registration.json', {'source_hashes': hashes})
        before = {p: p.read_bytes() for p in paths}
        records = {'curobo_dual_arm': {'adapter': 'old.py', 'policies': []}}
        with patch.object(activation, 'ROOT', self.root), \
             patch('arx_project.registry.module_records', return_value=records), \
             patch('arx_project.registry.load_adapter', return_value=SimpleNamespace(create=lambda args: None)), \
             patch.dict('sys.modules', {'modules.pipeline.experiments.current_configuration.configuration': SimpleNamespace(load_current=validation)}):
            try:
                result = activation.activate(self.comparison, self.main)
            except RuntimeError:
                for path, content in before.items():
                    self.assertEqual(path.read_bytes(), content)
                self.assertFalse((self.comparison/'activation/activation.json').exists())
                raise
        return result, paths

    def test_failed_configuration_validation_restores_all_four_files(self):
        def invalid(**flags):
            raise RuntimeError('validation failed')
        with self.assertRaisesRegex(RuntimeError, 'validation failed'):
            self.transaction(invalid)

    def test_successful_activation_updates_future_factory_without_new_budget(self):
        receipt, paths = self.transaction(lambda **flags: None)
        self.assertEqual(read_json(paths[0])['modules']['curobo_dual_arm']['adapter'],
                         'modules/curobo/dual_arm/accelerated_service.py')
        self.assertEqual(read_json(paths[1])['physical_grasp_budget'], 0)
        self.assertFalse(read_json(paths[2])['grasp_tests_authorized'])
        self.assertEqual(receipt['additional_physical_grasps'], 0)


if __name__ == '__main__':
    unittest.main()
