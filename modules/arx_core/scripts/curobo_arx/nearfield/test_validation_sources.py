"""Offline scheduling checks against the six retained real ARX sources."""
import copy
import unittest

from .prepare_validation import selected_records,source_checks


class ValidationSources(unittest.TestCase):
    def test_original_split_and_all_stationarity_outcomes_retained(self):
        records=selected_records();checks,seeds,_=source_checks(records)
        self.assertEqual(len(checks),6)
        self.assertEqual({c['source_episode_id'] for c in checks},{r['id'] for r in records})
        self.assertEqual(len({r['family'] for r in records}),4)
        self.assertEqual(len(seeds),3)
        for item in checks:
            self.assertEqual(item['eligible'],item['maximum_source_arm_velocity_rad_s']<=.01)
            if not item['eligible']:
                self.assertEqual(item['outcome'],'source_entry_not_stationary')
                self.assertEqual(item['physical_dispatches'],0)
        self.assertTrue(all(s['source_family']==next(r['family'] for r in records if r['id']==ident)
                            for ident,s in seeds.items()))

    def test_changed_physical_receipt_is_fault_not_ordinary_source_rejection(self):
        record=copy.deepcopy(selected_records()[0])
        record['files']['physical_receipt.json']='0'*64
        with self.assertRaisesRegex(RuntimeError,'Registered input/source changed'):
            source_checks([record])


if __name__=='__main__':unittest.main()
