"""Adversarial audit checks using copied real ARX records; no simulation calls."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from audit_contact_drag import audit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'runs/autoresearch_arx_20261020_round13/trials/00_marker_boundary_fixed'
CANONICAL = json.loads((SOURCE.parent.parent/'configuration_guard.json').read_text())['invariant_sha256']


class PreparationAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        files = ['capture_snapshot.json','initial_fingerprint.json','final_fingerprint.json',
                 'selected_grasp.json','scenario.json']
        for stage in ('pregrasp','approach','close'):
            files += [stage+'_command.json',stage+'_execution.json']
        files += ['pregrasp_optimization.json','approach_optimization.json']
        self.original_hashes = {}
        for name in files:
            self.original_hashes[name] = hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
            shutil.copy2(SOURCE/name,self.folder/name)
        shutil.copy2(ROOT/'runs/research_contract_20261019/contact_drag_geometry_preflight/drag_geometry.json', self.folder/'drag_geometry.json')

    def tearDown(self):
        for name,digest in self.original_hashes.items():
            self.assertEqual(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(),digest)
        self.temp.cleanup()

    def test_safe_partial_archive_never_counts_as_preparation_success(self):
        result = audit(self.folder,CANONICAL)
        self.assertTrue(result['guard_pass'],result['guard_violations'])
        self.assertFalse(result['preparation_pass'])
        self.assertFalse(result['grasp_success_evaluated'])

    def test_out_of_bounds_command_is_guard_failure(self):
        p = self.folder/'pregrasp_command.json'
        command = json.loads(p.read_text());snap = json.loads((self.folder/'capture_snapshot.json').read_text())
        upper = snap['joint_limits'][snap['joint_names'].index(command['joint_names'][0])][1]
        command['position'][0][0] = upper+.02;p.write_text(json.dumps(command))
        result = audit(self.folder,CANONICAL)
        self.assertFalse(result['guard_pass'])
        self.assertTrue(any('bounds' in s for s in result['guard_violations']))

    def test_robot_table_contact_is_guard_failure(self):
        p = self.folder/'approach_execution.json';data = json.loads(p.read_text())
        data['contacts'].append({'a':'/World/Robot/right_link27','b':'/World/Environment/breakfast_table','num_contacts':1})
        p.write_text(json.dumps(data));result = audit(self.folder,CANONICAL)
        self.assertFalse(result['guard_pass'])
        self.assertTrue(any('Unwanted physical contact' in s for s in result['guard_violations']))


if __name__ == '__main__':
    unittest.main()
