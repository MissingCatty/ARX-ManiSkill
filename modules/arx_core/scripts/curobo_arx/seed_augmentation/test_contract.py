"""Offline fault injection using actual accepted ARX teacher records."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write
from .contract import source_seed, join_check, replay_command, command_payload, perturbed_entry, audit_episode, anchor_readiness


class SeedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        campaign = ROOT/'runs/graspness_household_20261104_resume_01'
        book = sqlite3.connect('file:'+str(campaign/'ledger.sqlite3')+'?mode=ro', uri=True); book.row_factory = sqlite3.Row
        rows = [dict(row) for row in book.execute(
            "SELECT * FROM attempts WHERE state='committed' AND object_id='household_ycb_brick' ORDER BY started")]
        book.close(); cls.seed = source_seed(rows[0])

    def copy_episode(self, destination):
        source = Path(self.seed['episode'])
        for path in source.glob('*.json'):
            shutil.copy2(path, destination/path.name)
        write(destination/'join_state_check.json', anchor_readiness(self.seed, self.seed['anchor_snapshot'], 1.0))

    def test_source_replay_changes_only_nonce(self):
        for phase in ('approach', 'close', 'lift', 'hold'):
            original = read(Path(self.seed['episode'])/(phase+'_command.json'))
            replayed = replay_command(self.seed, phase)
            self.assertNotEqual(original['id'], replayed['id'])
            self.assertEqual(command_payload(original), command_payload(replayed))

    def test_entry_changes_preserve_seed_and_are_bounded(self):
        before = copy.deepcopy(self.seed)
        entry = np.array(perturbed_entry(self.seed, [.01, 0, 0], [0, 0, 5]))
        anchor = np.array(self.seed['anchor_world'])
        self.assertAlmostEqual(np.linalg.norm(entry[:3, 3]-anchor[:3, 3]), .01)
        self.assertEqual(before, self.seed)
        with self.assertRaisesRegex(ValueError, 'bounds'):
            perturbed_entry(self.seed, [.03, 0, 0], [0, 0, 0])

    def test_join_accepts_the_recorded_actual_arx_source_state(self):
        self.assertTrue(join_check(self.seed, self.seed['anchor_snapshot'])['pass'])

    def test_same_tool_pose_with_other_joint_configuration_is_rejected(self):
        snapshot = copy.deepcopy(self.seed['anchor_snapshot'])
        index = snapshot['joint_names'].index(snapshot['right_arm'][0])
        snapshot['joint_positions'][index] += .03
        self.assertFalse(join_check(self.seed, snapshot)['pass'])

    def test_moved_object_or_finger_mismatch_is_rejected(self):
        for object_moved in (True, False):
            snapshot = copy.deepcopy(self.seed['anchor_snapshot'])
            if object_moved:snapshot['objects']['test_object']['world'][0][3] += .002
            else:snapshot['joint_positions'][snapshot['joint_names'].index('right_joint27')] += .01
            self.assertFalse(join_check(self.seed, snapshot)['pass'])

    def test_actual_complete_source_command_chain_passes_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.copy_episode(root)
            self.assertTrue(audit_episode(root, self.seed, 'control')['pass'])

    def test_execution_present_command_missing_is_record_fault(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.copy_episode(root); (root/'close_command.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing evidence'):
                audit_episode(root, self.seed, 'control')

    def test_altered_tail_or_missing_join_stops_admission(self):
        for alter in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.copy_episode(root)
                if alter:
                    cmd = read(root/'approach_command.json'); cmd['position'][-1][0] += .001; write(root/'approach_command.json', cmd)
                else:(root/'join_state_check.json').unlink()
                with self.assertRaises(RuntimeError):
                    audit_episode(root, self.seed, 'control')

    def test_rejected_join_cannot_dispatch_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.copy_episode(root); write(root/'join_state_check.json', {'pass': False})
            with self.assertRaisesRegex(RuntimeError, 'rejected join'):
                audit_episode(root, self.seed, 'stitch')

    def test_matching_point_state_cannot_skip_anchor_settling(self):
        result = anchor_readiness(self.seed, self.seed['anchor_snapshot'], 0.)
        self.assertFalse(result['pass']); self.assertEqual(result['anchor_wait']['status'], 'waiting')
        self.assertTrue(anchor_readiness(self.seed, self.seed['anchor_snapshot'], 1.)['pass'])

    def test_velocity_mismatch_waits_then_rejects_without_relaxing_gate(self):
        snapshot = copy.deepcopy(self.seed['anchor_snapshot'])
        snapshot['joint_velocities'][snapshot['joint_names'].index(snapshot['right_arm'][0])] += .04
        self.assertEqual(anchor_readiness(self.seed, snapshot, 1.)['anchor_wait']['status'], 'waiting')
        last = anchor_readiness(self.seed, snapshot, 2.)
        self.assertFalse(last['pass']); self.assertEqual(last['anchor_wait']['status'], 'rejected')

    def test_missing_anchor_wait_receipt_stops_tail_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.copy_episode(root)
            write(root/'join_state_check.json', join_check(self.seed, self.seed['anchor_snapshot']))
            with self.assertRaisesRegex(RuntimeError, 'settling interval'):
                audit_episode(root, self.seed, 'stitch')

    def test_connector_execution_without_request_is_record_fault(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.copy_episode(root)
            for suffix in ('_command.json', '_execution.json', '_settling.json', '_optimization.json'):
                shutil.copy2(root/('pregrasp'+suffix), root/('approach_connect'+suffix))
            with self.assertRaisesRegex(RuntimeError, 'outside the command request ledger'):
                audit_episode(root, self.seed, 'stitch')


if __name__ == '__main__':
    unittest.main()
