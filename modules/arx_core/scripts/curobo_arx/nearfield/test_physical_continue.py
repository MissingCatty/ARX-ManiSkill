"""No replay or budget reset after the actual two ARX entry rejections."""
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.curobo_arx.data_factory.common import ROOT,read
from .physical_continue import remaining,remaining_after_serialization

PRIOR=ROOT/'runs/nearfield_physical_20261108_02'


class ContinuationTests(unittest.TestCase):
    def test_two_original_attempts_preserved_and_only34unused_slots(self):
        slots, done=remaining(PRIOR)
        self.assertEqual(len(slots),34);self.assertEqual(len(done),2)
        self.assertFalse({t['trial_id'] for t in slots}&{t['trial_id'] for t in done})
        self.assertEqual(slots,read(PRIOR/'registration.json')['seed_trials'][2:])
        self.assertTrue(all(not t['admitted'] for t in done))

    def changed(self,name,change):
        def replacement(path):
            value=read(path)
            if Path(path)==PRIOR/name:change(value)
            return value
        return patch('scripts.curobo_arx.nearfield.physical_continue.read',side_effect=replacement)

    def test_manual_stop_is_not_bypassed(self):
        with self.changed('STOP',lambda d:d.update(reason='manual_user_stop')):
            with self.assertRaises(ValueError):remaining(PRIOR)

    def test_unknown_attempt_or_repeated_continuation_blocks(self):
        with self.changed('summary.json',lambda d:d.update(incomplete_accounting=[{'trial_id':'unknown'}])):
            with self.assertRaises(ValueError):remaining(PRIOR)
        with self.changed('registration.json',lambda d:d.update(entry_wait_continuation_of=str(PRIOR))):
            with self.assertRaises(ValueError):remaining(PRIOR)

    def test_missing_terminal_receipt_cannot_requeue_started_attempt(self):
        with self.changed('summary.json',lambda d:d.update(trials=d['trials'][1:])):
            with self.assertRaisesRegex(ValueError,'Claimed or uncertain'):remaining(PRIOR)


class SerializationContinuationTests(unittest.TestCase):
    root=ROOT/'runs/nearfield_physical_20261108_03'

    def check(self):
        return remaining_after_serialization(self.root,self.root/'serialization_fault_reconciliation.json')

    def test_interrupted_slot_never_replayed_original36budget(self):
        slots,done=self.check()
        self.assertEqual(len(slots),33);self.assertEqual(len(done),3)
        self.assertFalse({t['trial_id'] for t in slots}&{t['trial_id'] for t in done})
        self.assertEqual(done[-1]['original_terminal_accounting'],'incomplete_preserved')
        self.assertFalse(done[-1]['success']);self.assertFalse(done[-1]['replayed'])

    def changed(self,name,change):
        def replacement(path):
            value=read(path)
            if Path(path)==self.root/name:change(value)
            return value
        return patch('scripts.curobo_arx.nearfield.physical_continue.read',side_effect=replacement)

    def test_possible_teacher_dispatch_blocks_this_specific_recovery(self):
        with self.changed('serialization_fault_reconciliation.json',lambda d:d.update(first_teacher_request_or_execution_present=True)):
            with self.assertRaises(ValueError):self.check()

    def test_another_unknown_claim_blocks(self):
        with self.changed('summary.json',lambda d:d['incomplete_accounting'].append(d['incomplete_accounting'][0])):
            with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
