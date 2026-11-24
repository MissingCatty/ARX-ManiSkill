"""Authorization and frozen-case tests for the three ARX preview slots."""
import copy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from scripts.curobo_arx.data_factory.common import read, write
from .advisor40_preview import TRAINING, USER_REQUEST, check_authority, validate_preview, run
from .advisor40_online import summarize
from .visual_grounding_spec import expected_reset


def fixture():
    original = read(TRAINING / 'physical_spec.json')
    trials = [copy.deepcopy(t) for t in original['trials'] if t['condition'] == 'normal']
    seeds = {t['seed_id']: original['seeds'][t['seed_id']] for t in trials}
    return {**original, 'trials': trials, 'seeds': seeds, 'maximum_physical_slots': 3,
            'expected_resets': {t['trial_id']: expected_reset(seeds[t['seed_id']], t) for t in trials}}


class PreviewTests(unittest.TestCase):
    def test_actual_normal_cases_only(self):
        spec = fixture(); validate_preview(spec)
        changed = copy.deepcopy(spec); changed['trials'].append(copy.deepcopy(spec['trials'][0]))
        with self.assertRaises(ValueError): validate_preview(changed)
        for key, value in [('automatic_closure', True), ('automatic_retry', True), ('maximum_physical_slots', 9)]:
            changed = copy.deepcopy(spec); changed[key] = value
            with self.assertRaises(ValueError): validate_preview(changed)

    def test_entry_or_reset_change_is_rejected(self):
        spec = fixture()
        changed = copy.deepcopy(spec); changed['trials'][0]['entry_world'][0][3] += .001
        with self.assertRaisesRegex(ValueError, 'Frozen normal case'): validate_preview(changed)
        changed = copy.deepcopy(spec); changed['expected_resets'][changed['trials'][0]['trial_id']] = {}
        with self.assertRaisesRegex(ValueError, 'reset binding'): validate_preview(changed)

    def test_offline_failure_does_not_authorize_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'user_authority.json', {'user_request': USER_REQUEST,
                'kind': 'explicit_observational_preview', 'maximum_slots': 3,
                'allow_execution_despite_offline_performance_failure': False})
            with self.assertRaisesRegex(RuntimeError, 'No authorization'): check_authority(root)

    def test_missing_or_expanded_authority_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'user_authority.json', {})
            with self.assertRaisesRegex(RuntimeError, 'Missing explicit'): check_authority(root)
            write(root / 'user_authority.json', {'user_request': USER_REQUEST,
                'kind': 'explicit_observational_preview', 'maximum_slots': 9,
                'allow_execution_despite_offline_performance_failure': True})
            with self.assertRaisesRegex(RuntimeError, 'No authorization'): check_authority(root)

    def test_failed_preflight_records_start_and_stops_before_services(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write(root / 'physical_spec.json', fixture())
            write(root / 'user_authority.json', {'test': 'CPU fault injection'})
            old_handler = signal.getsignal(signal.SIGTERM)
            prefix = 'scripts.curobo_arx.nearfield.advisor40_preview.'
            roles = {'sim_uuid': 'sim', 'policy_uuid': 'policy', 'compute_uuid': 'compute'}
            try:
                with patch(prefix + 'check_authority', return_value={'explicit_user_authorization': True}), \
                     patch(prefix + 'devices', return_value=[]), \
                     patch(prefix + 'choose_roles', return_value=roles), \
                     patch(prefix + 'time.sleep'), \
                     patch(prefix + 'prepare_runtime', side_effect=RuntimeError('CPU preflight sentinel')), \
                     patch(prefix + 'owned_process') as launch:
                    with self.assertRaisesRegex(RuntimeError, 'CPU preflight sentinel'): run(root)
                    launch.assert_not_called()
                self.assertTrue((root / 'ONLINE_STARTED.json').exists())
                self.assertTrue((root / 'STOP').exists())
                self.assertEqual(read(root / 'state.json')['status'], 'fault')
            finally:
                signal.signal(signal.SIGTERM, old_handler)


if __name__ == '__main__':
    unittest.main()
