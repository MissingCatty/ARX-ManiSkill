"""Compare batching with the existing scalar policy, including failure paths."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from modules.curobo.fingertip_contact.planning import contact_pool
from modules.curobo.fingertip_contact.validator import LocalContactValidator
from .branch_validation import fallback_endpoint, validate_rows
from .stage_timing import StageTiming, wrap_method


PROFILE = SimpleNamespace(position_tolerance_m=.002, rotation_tolerance_rad=.03)


def fixture(codes):
    values = torch.zeros((len(codes), 6), dtype=torch.float32)
    values[:, 0] = torch.arange(len(codes))
    for i, code in enumerate(codes):
        if code == 'limits': values[i, 5] = 11.

    class Checker:
        def __init__(self, native): self.native = native

        def validate(self, q):
            allowed = ['native'] if self.native else [c for c in set(codes) if c != 'other']
            return torch.tensor([codes[int(v)] in allowed for v in q.reshape(-1, 6)[:, 0]])

    class Geometry:
        def frames(self, q):
            poses = np.repeat(np.eye(4)[None], len(q), axis=0)
            for j, row in enumerate(q):
                code = codes[int(row[0])]
                if code == 'FK_position': poses[j, 0, 3] = .0021
                if code == 'FK_rotation': poses[j, :3, :3] = Rotation.from_rotvec([.0301, 0, 0]).as_matrix()
            return {'right_eef_link': poses}

    class Policy:
        def evaluate_one(self, row, **kwargs):
            code = codes[int(row[0])]
            return {'valid': code == 'allowed_tip',
                    'reason': 'native_collision' if code == 'native_collision' else 'real_finger_depth_exceeds_tolerance'}

    context = SimpleNamespace(limits=torch.tensor([[-10., 10.]]*6), checker=Checker(True))
    validator = LocalContactValidator.__new__(LocalContactValidator)
    validator.context, validator.geometry = context, Geometry()
    validator.other_checker, validator.policy, validator.receipts = Checker(False), Policy(), []
    pool = contact_pool(object, PROFILE).__new__(contact_pool(object, PROFILE))
    pool.contact_profile = PROFILE
    pool.native_grasp_ik = lambda *args: None
    pool.unconstrained_ik = lambda *args: values.clone()
    return pool, context, validator


class BranchValidationTests(unittest.TestCase):
    def compare(self, codes):
        pool, context, validator = fixture(codes)
        original = pool.grasp_endpoint(context, {}, np.eye(4), torch.zeros(6), validator)
        pool, context, validator = fixture(codes)
        batched = fallback_endpoint(pool, context, np.eye(4), torch.zeros(6), validator)
        if original[0] is None:
            self.assertIsNone(batched[0])
        else:
            torch.testing.assert_close(original[0], batched[0], rtol=0, atol=0)
        self.assertEqual(original[1], batched[1])

    def test_no_returned_solution(self): self.compare([])

    def test_first_passing_original_branch_is_preserved(self):
        self.compare(['other', 'table', 'allowed_tip', 'native'])

    def test_native_branch_bypasses_fallback(self):
        pool, context, validator = fixture(['table'])
        value = torch.zeros(6)
        pool.native_grasp_ik = lambda *args: value
        pool.unconstrained_ik = lambda *args: self.fail('Native pass must not invoke fallback')
        result, evidence = fallback_endpoint(pool, context, np.eye(4), value, validator)
        self.assertIs(result, value)
        self.assertEqual(evidence, {'kind': 'native_full_world_ik'})
        self.assertEqual(validator.receipts, [])

    def test_mixed_failure_reasons_keep_their_row_order(self):
        self.compare(['FK_position', 'table', 'other', 'limits', 'FK_rotation', 'native_collision'])

    def test_all_FK_failures_skip_world_checks(self):
        pool, context, validator = fixture(['FK_position', 'FK_rotation'])
        validator.validate = lambda *a, **k: self.fail('Invalid FK branches must not be validated')
        result, evidence = fallback_endpoint(pool, context, np.eye(4), torch.zeros(6), validator)
        self.assertIsNone(result)
        self.assertEqual(evidence['reasons'], ['independent_fk_error']*2)

    def test_varied_partial_batches_match_original_policy(self):
        rng = np.random.default_rng(20261126)
        choices = ['native', 'allowed_tip', 'table', 'other', 'limits', 'FK_position', 'FK_rotation']
        for count in range(1, 9):
            for _ in range(24):
                codes = list(rng.choice(choices, count))
                with self.subTest(codes=codes): self.compare(codes)

    def test_nonfinite_branch_is_a_fault(self):
        pool, context, validator = fixture(['table'])
        pool.unconstrained_ik = lambda *a: torch.full((1, 6), float('nan'))
        with self.assertRaises(ValueError):
            fallback_endpoint(pool, context, np.eye(4), torch.zeros(6), validator)

    def test_corrupted_row_count_stops(self):
        pool, context, validator = fixture(['table'])
        original = validator.validate
        def damaged(*args, **kwargs):
            result = original(*args, **kwargs)
            validator.receipts[-1]['states'] += 1
            return result
        validator.validate = damaged
        with self.assertRaises(RuntimeError): validate_rows(validator, torch.zeros((1, 6)))

    def test_corrupted_bounds_receipt_stops(self):
        pool, context, validator = fixture(['table'])
        original = validator.validate
        def damaged(*args, **kwargs):
            result = original(*args, **kwargs)
            validator.receipts[-1]['joint_bounds_clear'] = 0
            return result
        validator.validate = damaged
        with self.assertRaises(RuntimeError): validate_rows(validator, torch.zeros((1, 6)))

    def test_duplicate_narrow_phase_row_stops(self):
        pool, context, validator = fixture(['table'])
        original = validator.validate
        def damaged(*args, **kwargs):
            result = original(*args, **kwargs)
            validator.receipts[-1]['narrow_phase'] *= 2
            return result
        validator.validate = damaged
        with self.assertRaises(RuntimeError): validate_rows(validator, torch.zeros((1, 6)))


class StageTimingTests(unittest.TestCase):
    def test_nested_inclusive_and_exclusive_times(self):
        ticks = iter([0., 1., 3., 5.])
        timer = StageTiming(lambda: next(ticks));timer.active = True
        with timer.stage('outer'):
            with timer.stage('inner'): pass
        result = timer.report()['stages']
        self.assertEqual(result['outer'], {'calls': 1, 'wall_seconds': 5., 'exclusive_seconds': 3.})
        self.assertEqual(result['inner']['wall_seconds'], 2.)

    def test_exception_unwinds_and_disabled_timing_is_empty(self):
        timer = StageTiming()
        with timer.stage('disabled'): pass
        self.assertEqual(timer.report()['events'], [])
        timer.active = True
        with self.assertRaises(ValueError):
            with timer.stage('failure'): raise ValueError('expected')
        self.assertEqual(timer.report()['stages']['failure']['calls'], 1)
        timer.reset()
        self.assertEqual(timer.report()['events'], [])

    def test_wrapper_only_changes_one_instance_and_preserves_arguments(self):
        class Worker:
            def call(self, a, *, b): return a+b
        first, second, timer = Worker(), Worker(), StageTiming()
        timer.active = True;wrap_method(first, 'call', timer, 'request')
        self.assertEqual(first.call(2, b=3), second.call(2, b=3))
        self.assertEqual(timer.report()['stages']['request']['calls'], 1)
        self.assertNotIn('call', second.__dict__)


if __name__ == '__main__': unittest.main()
