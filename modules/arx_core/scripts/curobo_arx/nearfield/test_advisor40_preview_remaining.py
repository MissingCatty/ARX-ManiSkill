"""CPU checks for carrying only unstarted ARX preview slots forward."""
import copy
import unittest
from unittest.mock import patch

from scripts.curobo_arx.data_factory.common import read
from .advisor40_preview_remaining import PRIOR, validate_remaining, prior_exit


def fixture():
    previous = read(PRIOR / 'physical_spec.json')
    trials = [t for t in previous['trials'] if t['object_id'] != 'household_hope_milk_carton']
    return {**previous, 'trials': trials, 'maximum_physical_slots': 2,
            'seeds': {t['seed_id']: previous['seeds'][t['seed_id']] for t in trials},
            'expected_resets': {t['trial_id']: previous['expected_resets'][t['trial_id']] for t in trials}}


class RemainingTests(unittest.TestCase):
    def test_original_two_unstarted_slots_pass(self):
        validate_remaining(fixture())

    def test_started_milk_slot_cannot_be_replayed(self):
        spec = fixture(); spec['trials'].append(read(PRIOR / 'physical_spec.json')['trials'][0])
        with self.assertRaisesRegex(RuntimeError, 'milk must never replay'): validate_remaining(spec)

    def test_geometry_and_placement_cannot_be_changed(self):
        spec = copy.deepcopy(fixture()); spec['trials'][0]['entry_world'][0][3] += .001
        with self.assertRaises(RuntimeError): validate_remaining(spec)

    def test_thresholds_and_budget_cannot_be_relaxed(self):
        for key, value in [('maximum_physical_slots', 3), ('automatic_retry', True),
                           ('automatic_closure', True), ('maximum_refines_per_slot', 3)]:
            spec = copy.deepcopy(fixture()); spec[key] = value
            with self.assertRaises(RuntimeError): validate_remaining(spec)

    def test_unconfirmed_previous_world_blocks_continuation(self):
        with patch('scripts.curobo_arx.nearfield.advisor40_preview_remaining.read',
                   return_value={'status': 'fault', 'unconfirmed_world_or_process': True}):
            with self.assertRaisesRegex(RuntimeError, 'unconfirmed'): prior_exit()


if __name__ == '__main__':
    unittest.main()
