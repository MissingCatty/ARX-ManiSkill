"""Real ARX original48 fixtures: no replay, omission, or concealed prior result."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from scripts.curobo_arx.data_factory.common import ROOT, read
from .main48_resume_deferred import resumed_registration, check_unused_ledgers
from .main48_spec import validate_schedule

FIXTURES = Path(os.environ.get('ARX_TEST_FIXTURE_ROOT', ROOT))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = FIXTURES / 'runs/nearfield_main48_20261110_05'
        self.old = read(self.root / 'registration.json')
        self.report = read(self.root / 'report_recovery_01/results/main48_verification.json')

    def test_original_seven_and_all_prior_outcomes_preserved(self):
        r = resumed_registration(self.old, self.report)
        self.assertEqual(r['seed_trials'], self.old['deferred_trials'])
        self.assertEqual(r['carried_details'], self.report['trials'])
        self.assertEqual(r['deadline_unix'], self.old['deadline_unix'])
        self.assertEqual(r['learned_policy'], self.old['learned_policy'])
        self.assertEqual(r['deferred_trials'], [])
        self.assertEqual([int(t['trial_id'].split('_')[0]) for t in r['seed_trials']], [22,25,28,35,38,41,44])
        self.assertEqual(len(check_unused_ledgers(self.root / 'deferred_trial_ledgers', r['seed_trials'])), 7)

    def test_modified_or_unverified_prior_result_rejected(self):
        for alter in (lambda r: r.update(record_integrity_pass=False),
                      lambda r: r['trials'][0].update(physical_success=False),
                      lambda r: r['unexecuted'].pop(),
                      lambda r: r['unknown_or_incomplete'].append({'trial_id': 'unexpected'})):
            v = copy.deepcopy(self.report); alter(v)
            with self.assertRaises(ValueError): resumed_registration(self.old, v)

    def test_no_replay_duplicate_omission_or_reordering(self):
        for alter in (lambda r: r['seed_trials'].append(r['registered_trials'][21]),
                      lambda r: r['seed_trials'].append(r['seed_trials'][0]),
                      lambda r: r['seed_trials'].pop(),
                      lambda r: r['seed_trials'].reverse()):
            r = resumed_registration(self.old, self.report); alter(r)
            with self.assertRaises(ValueError): validate_schedule(r)

    def test_goal_cannot_be_changed(self):
        r = resumed_registration(self.old, self.report)
        r['seed_trials'][0]['G0_world'][0][3] += .001
        with self.assertRaises(ValueError): validate_schedule(r)

    def test_consumed_ledger_blocks_scheduling(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'ledgers'; shutil.copytree(self.root / 'deferred_trial_ledgers', p)
            target = p / (self.old['deferred_trials'][0]['trial_id'] + '.sqlite3')
            shutil.copy2(next((self.root / 'trial_ledgers').glob('*.sqlite3')), target)
            with self.assertRaisesRegex(ValueError, 'already claimed'):
                check_unused_ledgers(p, self.old['deferred_trials'])

    def test_missing_foreign_or_journal_ledger_blocks(self):
        for kind in ('missing', 'foreign', 'journal'):
            with tempfile.TemporaryDirectory() as temp:
                p = Path(temp) / 'ledgers'; shutil.copytree(self.root / 'deferred_trial_ledgers', p)
                first = next(p.glob('*.sqlite3'))
                if kind == 'missing': first.unlink()
                elif kind == 'foreign': shutil.copy2(first, p / 'foreign.sqlite3')
                else: first.with_name(first.name + '-wal').write_bytes(b'journal')
                with self.assertRaises(ValueError): check_unused_ledgers(p, self.old['deferred_trials'])


if __name__ == '__main__': unittest.main()
