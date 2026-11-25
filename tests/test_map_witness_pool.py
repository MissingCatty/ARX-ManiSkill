"""Known endpoints keep FK/bounds/contact gates and all native fallbacks."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from modules.curobo.dual_arm.witness_pool import witness_pool
from modules.curobo.dual_arm.placement_service import connection_gate


class WitnessTests(unittest.TestCase):
    def setUp(self):
        import torch
        class Base:
            def grasp_endpoint(inner, *args):
                inner.fallback_calls += 1
                return 'native', {'kind': 'original'}
        self.pool = witness_pool(Base)()
        self.pool.fallback_calls = 0
        self.pool.contact_profile = SimpleNamespace(position_tolerance_m=.002, rotation_tolerance_rad=.03)
        self.context = SimpleNamespace(limits=torch.tensor([[-2., 2.]]*6), checker=Mock())
        self.context.checker.validate.return_value = np.array([True])
        self.validator = Mock()
        self.validator.geometry.frames.return_value = {'right_eef_link': np.eye(4)[None]}
        self.validator.validate.return_value = np.array([True])
        self.state = {'kinematic_endpoint_witness': {'q': [.1]*6,
            'eef_target_world': np.eye(4).tolist(), 'source_map_row': 123}}
        self.start = torch.zeros(6)

    def call(self, state=None, goal=None):
        return self.pool.grasp_endpoint(self.context, self.state if state is None else state,
            np.eye(4) if goal is None else goal, self.start, self.validator)

    def test_valid_endpoint_avoids_redundant_ik(self):
        q, evidence = self.call()
        np.testing.assert_allclose(q.numpy(), [.1]*6)
        self.assertEqual(self.pool.fallback_calls, 0)
        self.assertEqual(evidence['new_grasp_IK_calls'], 0)
        self.validator.validate.assert_called_once()

    def test_original_entry_without_witness_is_unchanged(self):
        self.assertEqual(self.call(state={})[0], 'native')
        self.assertEqual(self.pool.fallback_calls, 1)

    def test_other_goal_does_not_reuse_unrelated_witness(self):
        goal = np.eye(4); goal[0, 3] = .01
        self.assertEqual(self.call(goal=goal)[0], 'native')

    def test_corrupt_fk_stops_instead_of_dropping_record(self):
        actual = np.eye(4); actual[0, 3] = .003
        self.validator.geometry.frames.return_value = {'right_eef_link': actual[None]}
        with self.assertRaisesRegex(ValueError, 'independent FK'):
            self.call()
        self.assertEqual(self.pool.fallback_calls, 0)

    def test_bounds_and_nonfinite_data_stop(self):
        for q in [[3.]*6, [float('nan')]*6, [0.]*5]:
            state = deepcopy(self.state); state['kinematic_endpoint_witness']['q'] = q
            with self.assertRaises(ValueError):
                self.call(state=state)

    def test_collision_preserves_alternative_native_joint_search(self):
        self.validator.validate.return_value = np.array([False])
        self.assertEqual(self.call()[0], 'native')
        self.assertEqual(self.pool.fallback_calls, 1)

    def test_native_resource_fault_propagates(self):
        self.validator.validate.side_effect = RuntimeError('CUDA fault')
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.call()
        self.assertEqual(self.pool.fallback_calls, 0)


class ConnectionTests(unittest.TestCase):
    def test_original_eight_cm_pregrasp_and_three_cm_lift(self):
        context = Mock(); context.ik.return_value = object()
        target = np.eye(4); target[:3, 3] = [.2, .3, .4]
        original = target.copy(); q = object()
        result = connection_gate(context, target, [0, 1, 0], q)
        self.assertTrue(result['passed'])
        pre, lift = [call.args[0] for call in context.ik.call_args_list]
        np.testing.assert_allclose(pre[:3, 3], [.2, .22, .4])
        np.testing.assert_allclose(lift[:3, 3], [.2, .3, .43])
        np.testing.assert_array_equal(original, target)

    def test_missing_pregrasp_stops_before_transit(self):
        context = Mock(); context.ik.side_effect = [None, object()]
        self.assertFalse(connection_gate(context, np.eye(4), [0, 0, 1], object())['passed'])

    def test_missing_lift_stops_before_transit(self):
        context = Mock(); context.ik.side_effect = [object(), None]
        self.assertFalse(connection_gate(context, np.eye(4), [0, 0, 1], object())['passed'])

    def test_resource_fault_propagates(self):
        context = Mock(); context.ik.side_effect = RuntimeError('CUDA fault')
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            connection_gate(context, np.eye(4), [0, 0, 1], object())


if __name__ == '__main__':
    unittest.main()
