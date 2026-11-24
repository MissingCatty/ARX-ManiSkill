"""Regression for the real ARX failed image shown by the user; no simulation."""
from pathlib import Path
import unittest
from scripts.curobo_arx.data_factory.common import ROOT,read,sha
from .success_view import checked_episode


class SuccessViewTests(unittest.TestCase):
    def test_real_A6_failure_rejected_despite_legacy_flow_success_flag(self):
        ident='0e1955d65d7746bca5acb1e4ef8fb7de'
        root=ROOT/'runs/sdg_stage1_20261031_06/groups/group_6397bc8545f1/episodes'/ident
        before={name:sha(root/name) for name in ('grasp_report.json','physical_receipt.json','quality_receipt.json')}
        self.assertTrue(read(root/'grasp_report.json')['success'])
        self.assertFalse(read(root/'physical_receipt.json')['pass'])
        with self.assertRaisesRegex(ValueError,'not an admitted successful'):
            checked_episode(root,ident)
        self.assertEqual(before,{name:sha(root/name) for name in before})

    def test_same_actual_ARX_object_success_passes_measured_hold_recheck(self):
        ident='e89739f286284cc080d6bf7f3d5035c7'
        root=ROOT/'runs/sdg_stage1_20261031_06/groups/group_6397bc8545f1/episodes'/ident
        result=checked_episode(root,ident)
        self.assertTrue(result['grasp_success'])
        self.assertGreater(result['hold_min_rise_m'],.06)
        self.assertGreaterEqual(result['hold_seconds'],2.99)


if __name__=='__main__':unittest.main()
