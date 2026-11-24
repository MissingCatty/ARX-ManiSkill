import copy
from pathlib import Path
import unittest
from scripts.curobo_arx.data_factory.common import ROOT,read
from training.nearfield_arx.publish import lines
from .verify_physical import entry_matches_raw


class EntryVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=ROOT/'runs/nearfield_stage1_20261105_04/mustard_validation_01'
        trial=read(root/'summary.json')['trials'][0]
        ep=Path(trial['attempts'][0]['episode']);cls.entry=read(ep/'actual_entry.json')
        cls.row=next(r for r in lines(ep/'samples.jsonl') if r['frame_time']==cls.entry['frame_time'])

    def test_original_ARX_live_observation_matches_stored_row(self):
        self.assertTrue(entry_matches_raw(self.row,self.entry))

    def test_changed_robot_state_fails_even_with_same_image_time(self):
        row=copy.deepcopy(self.row);row['snapshot']['joint_positions'][0]+=.001
        with self.assertRaises(ValueError):entry_matches_raw(row,self.entry)

    def test_missing_or_extra_observation_fields_fail(self):
        row=copy.deepcopy(self.row);row.pop('camera_world_optical')
        with self.assertRaises(ValueError):entry_matches_raw(row,self.entry)
        row=copy.deepcopy(self.row);row['unregistered_measurement']=True
        with self.assertRaises(ValueError):entry_matches_raw(row,self.entry)


if __name__=='__main__':unittest.main()
