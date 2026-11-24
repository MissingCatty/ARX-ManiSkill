"""Protect physical geometry and backup exclusions during route retirement."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from arx_project.layout import ROOT, sha256
from modules.arx_core.scripts.curobo_arx.action_frame import EEF_TCP, target_pose
from modules.curobo_adapter.live.action_tcp import ActionTCP
from tools.drive_backup import groups


class RobotGeometryTests(unittest.TestCase):
    def test_pad_configuration_bytes_preserved(self):
        self.assertEqual(sha256(ROOT/'simulation/robots/arx/finger_pad_geometry.json'),
            '2d0c4e9f144287d431157cd2f0eace3341b69b4298baea19fd73c20d286461c0')

    def test_rotating_at_tcp_preserves_43mm_pivot(self):
        tcp = ActionTCP()
        action = SimpleNamespace(translation=SimpleNamespace(distance_cm=0., direction=np.zeros(3)),
            rotation=SimpleNamespace(angle_deg=35., axis=np.array([1., 0., 0.])))
        before = np.eye(4)
        after = tcp.target_pose(before, action)
        np.testing.assert_allclose((after@tcp.frame)[:3, 3], (before@tcp.frame)[:3, 3], atol=1e-12)
        self.assertGreater(np.linalg.norm(after[:3, 3]), .001)
        self.assertEqual(tcp.frame[2, 3], .043)

    def test_local_x_translation_follows_authored_eef_y(self):
        action = SimpleNamespace(translation=SimpleNamespace(distance_cm=1., direction=np.array([1.,0.,0.])),
            rotation=SimpleNamespace(angle_deg=0., axis=np.zeros(3)))
        after = target_pose(np.eye(4), action)
        np.testing.assert_allclose(after[:3,3], [0.,.01,0.], atol=1e-12)
        np.testing.assert_allclose(after[:3,:3], np.eye(3), atol=1e-12)

    def test_corrupted_tcp_profile_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            ActionTCP({'action_tcp_profile':str(ROOT/'simulation/robots/arx/runtime/action_tcp.json'),
                       'action_tcp_profile_sha256':'0'*64})

    def test_retired_assets_cannot_reenter_backup_by_directory_scan(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for member in ['weights/easg','weights/spar3d','weights/retained','simulation/assets/models/source_easg',
                           'simulation/assets/models/object','configs']:
                (root/member).mkdir(parents=True)
            (root/'configs/cloud_backup.json').write_text(json.dumps({'exclude':[
                'weights/easg','weights/spar3d','simulation/assets/models/source_easg']}))
            planned = groups(root)
            members = [m for group in planned for m in group['members']]
            self.assertNotIn('weights/easg', members)
            self.assertNotIn('weights/spar3d', members)
            self.assertNotIn('simulation/assets/models/source_easg', members)
            self.assertIn('weights/retained', members)
            self.assertIn('simulation/assets/models/object', members)


    def test_internal_legacy_aliases_are_rejected_by_layout_validation(self):
        from tools.validate_ownership import check_ownership
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for member in ['modules','tmp','simulation/assets','weights']:
                (root/member).mkdir(parents=True, exist_ok=True)
            (root/'modules/tmp').symlink_to('../tmp', target_is_directory=True)
            (root/'modules/asset').symlink_to('../simulation/assets', target_is_directory=True)
            result = check_ownership(root)
            self.assertFalse(result['passed'])
            self.assertEqual(result['outside'], [])
            self.assertEqual(result['broken'], [])
            self.assertCountEqual(result['forbidden_module_entries'], ['modules/tmp','modules/asset'])


if __name__ == '__main__':
    unittest.main()
