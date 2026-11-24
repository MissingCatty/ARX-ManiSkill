"""Data-leakage/input guards only; these are not physical demonstrations."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_dataset import build,transition_label


class DatasetContracts(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)

    def tearDown(self):self.temporary.cleanup()

    def source_metadata(self,label,object_name=None):
        folder=self.root/label;folder.mkdir()
        if object_name:(folder/'scenario.json').write_text(json.dumps({'asset':object_name}))
        return folder

    def test_same_episode_cannot_enter_both_splits(self):
        source=self.source_metadata('recorded_episode','rubiks_cube')
        with self.assertRaisesRegex(ValueError,'disjoint'):
            build([source],self.root/'dataset',[source],object_split=True)

    def test_different_episodes_of_same_object_cannot_leak(self):
        a=self.source_metadata('first_episode','rubiks_cube')
        b=self.source_metadata('second_episode','rubiks_cube')
        with self.assertRaisesRegex(ValueError,'Object leakage'):
            build([a],self.root/'dataset',[b],object_split=True)

    def test_object_split_requires_known_object_identity(self):
        source=self.source_metadata('unknown_object')
        with self.assertRaisesRegex(ValueError,'identity missing'):
            build([source],self.root/'dataset',object_split=True)

    def test_goal_codec_cannot_guess_goal_from_future_trajectory(self):
        source=self.source_metadata('missing_planned_goal','rubiks_cube')
        # Metadata only exercises the missing-goal guard. There are no images,
        # states or actions here and this fixture cannot pass dataset creation.
        (source/'verification.json').write_text(json.dumps({'pass':True}))
        (source/'teacher_recording.json').write_text(json.dumps({'error':None}))
        (source/'teacher_raw').mkdir();(source/'teacher_raw/samples.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError,'Explicit selected grasp'):
            build([source],self.root/'dataset',state_convention='grasp_goal_world_error_v1')

    def measured_rows(self):
        root=Path(__file__).resolve().parents[3]
        poses=json.loads((root/'servers/lilo_vla/tests/fixtures/arx_fixture.json').read_text())['measured_pose_pair']
        rows=[{'snapshot':{'link_world':{'right_eef_link':pose}},'stage':None,'teacher_command':None} for pose in poses]
        rows[1].update(stage='close',teacher_command={'id':'closing_command','gripper_target':.1})
        return rows

    def test_close_label_includes_real_observation_before_close_starts(self):
        first,second=self.measured_rows()
        stage,action,command,starts=transition_label(first,second,True)
        self.assertEqual(stage,'close');self.assertEqual(action[6],0.)
        self.assertTrue(starts);self.assertEqual(command['id'],'closing_command')
        self.assertIsNone(first['teacher_command'])

    def test_legacy_builder_semantics_remain_explicit(self):
        first,second=self.measured_rows()
        self.assertIsNone(transition_label(first,second))

    def test_ongoing_command_keeps_same_measured_action(self):
        first,second=self.measured_rows()
        first.update(stage='close',teacher_command=second['teacher_command'])
        legacy=transition_label(first,second);updated=transition_label(first,second,True)
        np.testing.assert_array_equal(legacy[1],updated[1]);self.assertFalse(updated[3])


if __name__=='__main__':unittest.main()
