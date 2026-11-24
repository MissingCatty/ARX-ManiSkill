"""Ensure extra-action auditing cannot hide unsafe archived ARX executions."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from audit_feedback_actions import audit
from check_replayed_observation import check


class ExtraActionAuditTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.source = root / 'runs/autoresearch_arx_20261019_round7/trials/26_whitepackerbottle_a01_robust'
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        for name in ('capture_snapshot.json', 'scenario.json'):
            shutil.copy2(self.source / name, self.folder / name)
        self.label = 'feedback_history0_approach'
        for suffix in ('command', 'execution', 'optimization'):
            shutil.copy2(self.source / f'approach_{suffix}.json', self.folder / f'{self.label}_{suffix}.json')
        goal = json.loads((self.source / 'selected_grasp.json').read_text())['grasp_world']
        self.write(self.label + '_goal.json', goal)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data):
        (self.folder / name).write_text(json.dumps(data))

    def test_archived_real_arx_approach_passes(self):
        self.assertTrue(audit(self.folder)['pass'])

    def test_added_unwanted_contact_is_rejected(self):
        data = json.loads((self.folder / f'{self.label}_execution.json').read_text())
        data['contacts'].append({'a': '/World/Robot/right_link27', 'b': '/World/table', 'num_contacts': 1})
        self.write(f'{self.label}_execution.json', data)
        self.assertFalse(audit(self.folder)['pass'])

    def test_loaded_asset_finger_contact_uses_captured_actual_target_path(self):
        snapshot = json.loads((self.folder / 'capture_snapshot.json').read_text())
        target = snapshot['objects']['test_object']['path']
        self.assertNotEqual(target, '/World/Task/test_object')
        data = json.loads((self.folder / f'{self.label}_execution.json').read_text())
        data['contacts'].append({'a': '/World/Robot/right_link27', 'b': target, 'num_contacts': 1})
        self.write(f'{self.label}_execution.json', data)
        self.assertTrue(audit(self.folder)['pass'])

    def test_command_limit_violation_is_rejected_without_trusting_certificate(self):
        data = json.loads((self.folder / f'{self.label}_command.json').read_text())
        data['position'][0][0] = 99.
        self.write(f'{self.label}_command.json', data)
        self.assertFalse(audit(self.folder)['pass'])

    def test_closed_extra_alignment_is_rejected(self):
        data = json.loads((self.folder / f'{self.label}_command.json').read_text())
        data['gripper_target'] = .1
        self.write(f'{self.label}_command.json', data)
        self.assertFalse(audit(self.folder)['pass'])

    def test_missing_extra_optimizer_evidence_is_rejected(self):
        (self.folder / f'{self.label}_optimization.json').unlink()
        self.assertFalse(audit(self.folder)['pass'])


class ReplayObservationGateTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.reference = root / 'runs/autoresearch_arx_20261019_round7/trials/26_whitepackerbottle_a01_robust'
        self.temp = tempfile.TemporaryDirectory(); self.folder = Path(self.temp.name)
        (self.folder / 'camera').mkdir()
        for name in ('capture_snapshot.json', 'camera_world_optical.json', 'camera/params.json'):
            shutil.copy2(self.reference / name, self.folder / name)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, name, change):
        file = self.folder / name; data = json.loads(file.read_text()); change(data); file.write_text(json.dumps(data))

    def test_same_actual_arx_observation_passes(self):
        self.assertTrue(check(self.folder, self.reference, 'test_object')['pass'])

    def test_target_moved_without_reinference_is_rejected(self):
        def moved(data): data['objects']['test_object']['world'][0][3] += .01
        self.mutate('capture_snapshot.json', moved)
        with self.assertRaisesRegex(RuntimeError, 'consistency rejected'):
            check(self.folder, self.reference, 'test_object')

    def test_camera_world_pose_change_is_rejected(self):
        def moved(data): data[0][3] += .01
        self.mutate('camera_world_optical.json', moved)
        with self.assertRaisesRegex(RuntimeError, 'consistency rejected'):
            check(self.folder, self.reference, 'test_object')

    def test_intrinsics_change_is_rejected(self):
        def changed(data): data['intrinsics']['fx'] *= 1.01
        self.mutate('camera/params.json', changed)
        with self.assertRaisesRegex(RuntimeError, 'consistency rejected'):
            check(self.folder, self.reference, 'test_object')


if __name__ == '__main__':
    unittest.main()
