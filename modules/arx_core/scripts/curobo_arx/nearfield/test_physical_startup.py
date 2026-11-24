from pathlib import Path
import shutil
import tempfile
import unittest
from scripts.curobo_arx.data_factory.common import ROOT,read,write
from .physical_startup import unissued,startup_expired


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.previous=ROOT/'runs/nearfield_physical_20261107_01'

    def test_actual_cancelled_cold_start_has_no_claimed_slots(self):
        self.assertEqual(sum(unissued(self.previous)['slot_claim_counts'].values()),0)

    def test_original_ARX_attempt_in_any_slot_blocks_continuation(self):
        source=next((ROOT/'runs/nearfield_stage1_20261105_04/recovery_24/trial_ledgers').glob('*.sqlite3'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name in ('registration.json','STOP','state.json'):shutil.copy2(self.previous/name,p/name)
            (p/'trial_ledgers').mkdir()
            for index,f in enumerate(sorted((self.previous/'trial_ledgers').glob('*.sqlite3'))):
                (p/'trial_ledgers'/f.name).symlink_to(source if index==0 else f)
            with self.assertRaises(ValueError):unissued(p)

    def test_manual_stop_and_repeat_continuation_are_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name in ('registration.json','STOP','state.json'):shutil.copy2(self.previous/name,p/name)
            write(p/'STOP',{'reason':'user stop'})
            with self.assertRaises(ValueError):unissued(p)
            reg=read(p/'registration.json');reg['startup_continuation_of']='already_used';write(p/'registration.json',reg)
            with self.assertRaises(ValueError):unissued(p)

    def test_wait_cap_and_original_ready_phase_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.assertFalse(startup_expired(p,1800,1800));self.assertTrue(startup_expired(p,1800.1,1800))
            write(p/'initialization.json',{'phase':'ready_for_uniform_warmup'})
            self.assertFalse(startup_expired(p,2000,1800))
            with self.assertRaises(ValueError):startup_expired(p,1,999999)


if __name__=='__main__':unittest.main()
