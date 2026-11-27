"""Narrower candidate widths must match the exact already-validated geometry."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from .opening import validate_opening, install


def selection(opening):
    return {'open_finger_m': opening, 'width_binding': {'candidate_width_m': 2 * opening,
        'open_finger_m': opening, 'actual_collision_finger_locks_m': {'a': opening, 'b': opening},
        'native_GPU_finger_locks_m': {'a': float(np.float32(opening)), 'b': float(np.float32(opening))},
        'narrowing_samples': np.linspace(.044, opening, max(2, int(np.ceil((.044-opening)/.0005))+1)).tolist(),
        'narrowing_target_collision_checked': True}}


class Opening(unittest.TestCase):
    def test_accepted_actual_widths_including_failed_banana(self):
        for opening in [.044, .040, .030, .027480319176659697, .020, .005]:
            self.assertEqual(validate_opening(selection(opening), opening)['open_finger_m'], opening)

    def test_width_and_lock_mismatch_never_clamped(self):
        original = selection(.027480319176659697)
        for mutation in ['selected', 'proof', 'CPU', 'GPU']:
            value = deepcopy(original)
            if mutation == 'selected': value['open_finger_m'] = .044
            if mutation == 'proof': value['width_binding']['candidate_width_m'] = .088
            if mutation == 'CPU': value['width_binding']['actual_collision_finger_locks_m']['a'] = .044
            if mutation == 'GPU': value['width_binding']['native_GPU_finger_locks_m']['b'] = .044
            with self.assertRaises(ValueError): validate_opening(value, .027480319176659697)

    def test_missing_narrowing_collision_proof_stops(self):
        for key in ['narrowing_samples', 'narrowing_target_collision_checked']:
            value = selection(.027480319176659697); del value['width_binding'][key]
            with self.assertRaises(ValueError): validate_opening(value, .027480319176659697)

    def test_coarse_narrowing_sweep_rejected(self):
        value = selection(.020); value['width_binding']['narrowing_samples'] = [.044, .020]
        with self.assertRaises(ValueError): validate_opening(value, .020)

    def test_invalid_physical_stroke_rejected(self):
        for opening in [0., -.001, .045, float('nan')]:
            with self.assertRaises(ValueError): validate_opening(selection(.030), opening)

    def test_motion_keeps_selected_width_and_preserves_close_lift_hold(self):
        class Parent:
            def segment(self, label, q, times, **flags): return flags['finger']
        executor = SimpleNamespace(Execution=Parent); install(executor)
        with tempfile.TemporaryDirectory() as tmp:
            instance = executor.Execution(); instance.out = Path(tmp); instance._planned_opening = .027480319176659697
            for label in ['pregrasp', 'approach']:
                self.assertEqual(instance.segment(label, [], []), instance._planned_opening)
            for label in ['close', 'lift', 'hold']:
                self.assertEqual(instance.segment(label, [], [], finger=0.), 0.)

    def test_motion_cannot_silently_reopen_fingers(self):
        class Parent:
            def segment(self, *args, **flags): raise AssertionError('Must be rejected before dispatch')
        executor = SimpleNamespace(Execution=Parent); install(executor)
        instance = executor.Execution(); instance._planned_opening = .027480319176659697
        with self.assertRaises(ValueError): instance.segment('pregrasp', [], [], finger=.044)


if __name__ == '__main__': unittest.main()
