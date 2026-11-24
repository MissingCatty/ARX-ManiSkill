"""Check the new log adapter against existing actual ARX execution archives."""
import json
from pathlib import Path
import unittest

from egad_evaluate import evaluate

ROOT=Path(__file__).resolve().parents[3]


class ActualArxProtocol(unittest.TestCase):
    def test_accepts_real_original_five_stage_arx_success(self):
        folder=ROOT/'runs/lilo_arx_teacher_20261026_05'
        if not folder.exists():self.skipTest('Recorded ARX validation archive not installed')
        score=evaluate(folder)
        self.assertTrue(score['pass'],score)
        self.assertTrue(score['bilateral_closure'])
        self.assertGreater(score['hold_min_rise_m'],.06)
        self.assertEqual(score['forbidden_contact_events'],0)

    def test_rejects_real_policy_trial_with_57mm_hold(self):
        folder=ROOT/'runs/lilo_execution_20261026_tracking01'
        if not folder.exists():self.skipTest('Recorded ARX validation archive not installed')
        score=evaluate(folder)
        self.assertFalse(score['pass'])
        self.assertFalse(score['all_recorded_hold_samples_above_6cm'])
        self.assertAlmostEqual(score['hold_min_rise_m'],.05702090263366699)

    def test_registered_dataset_is_49_matched_cases(self):
        folder=ROOT/'runs/egad49_arx_20261020'
        if not folder.exists():self.skipTest('Original EGAD49 ARX protocol not installed')
        rows=json.loads((folder/'scenarios.json').read_text())
        grouped={}
        for row in rows:grouped.setdefault(row['egad_id'],[]).append(row)
        self.assertEqual(len(rows),98);self.assertEqual(len(grouped),49)
        for name,pair in grouped.items():
            self.assertEqual(len(pair),2,name)
            self.assertEqual(pair[0]['placement'],pair[1]['placement'],name)
            self.assertEqual(pair[0]['asset'],pair[1]['asset'],name)


if __name__=='__main__':unittest.main()
