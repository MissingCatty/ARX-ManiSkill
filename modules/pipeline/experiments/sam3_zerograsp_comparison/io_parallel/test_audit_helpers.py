"""Offline faults for inherited provenance and native GGX candidate readback."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from arx_project.artifacts import atomic_json
from arx_project.layout import sha256
from .audit_helpers import native_trial_origin,rebuild_generator_candidates


class AuditHelpers(unittest.TestCase):
    def test_inherited_receipt_resolves_to_original_and_fresh_case_stays_local(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root/'old', root/'new'
            old.mkdir();new.mkdir()
            atomic_json(old/'registration.json', {})
            atomic_json(new/'registration.json', {'continuation': {
                'predecessor': 'old',
                'predecessor_evidence': {'registration.json':sha256(old/'registration.json')},
                'method_slots': {'zerograsp': {'complete': ['a'], 'interrupted': [], 'not_started':['b']}}}})
            self.assertEqual(native_trial_origin(new,'zerograsp','a',project_root=root),old/'methods/zerograsp/trials/a')
            self.assertEqual(native_trial_origin(new,'zerograsp','b',project_root=root),new/'methods/zerograsp/trials/b')
            atomic_json(old/'registration.json', {'changed':True})
            with self.assertRaises(RuntimeError):native_trial_origin(new,'zerograsp','a',project_root=root)

    def test_native_converter_works_when_evaluator_common_is_already_loaded(self):
        from modules.maniskill.planning_common import common
        previous = sys.modules.get('common')
        sys.modules['common'] = common
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder);raw=root/'graspgenx_raw';raw.mkdir()
                from modules.graspgenx.arx_frames import VERSION,T_EEF_GENERATOR
                atomic_json(raw/'inference.json', {'repair_contract':VERSION,
                    'T_eef_generator':T_EEF_GENERATOR.tolist(),'branch_tags':['test']})
                np.save(raw/'grasps_generator_world.npy',np.eye(4)[None],allow_pickle=False)
                np.save(raw/'scores.npy',np.asarray([.7]),allow_pickle=False)
                atomic_json(root/'state.json',{'camera_world_optical':np.eye(4).tolist()})
                rebuild_generator_candidates(root)
                pool=json.loads((root/'candidates/candidate_pool.json').read_text())
                self.assertEqual(len(pool['candidates']),1)
                self.assertEqual(pool['candidates'][0]['score'],.7)
                self.assertIs(sys.modules['common'],common)
        finally:
            if previous is None:sys.modules.pop('common',None)
            else:sys.modules['common']=previous


if __name__ == '__main__':unittest.main()
