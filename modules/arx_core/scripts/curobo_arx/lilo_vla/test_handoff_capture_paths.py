"""Filesystem/RPC capture-path regression; all planning callbacks are mocked."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import numpy as np

import grasp_handoff

ROOT=Path(__file__).resolve().parents[3]
FIXTURES=ROOT/'servers/lilo_vla/tests/fixtures'
OBSERVATION=json.loads((FIXTURES/'arx_fixture.json').read_text())['observation']


class CaptureDirectoryContract:
    """Mimic native mkdir(exist_ok=False) using a recorded ARX wrist frame."""
    def request(self,action,directory):
        if action!='grasp_lilo_observation':raise AssertionError('No physical command is supported by this shim')
        directory=Path(directory);directory.mkdir(parents=True)
        image=directory/'wrist.png';shutil.copy2(FIXTURES/'arx_wrist_pregrasp.png',image)
        observation=copy.deepcopy(OBSERVATION);observation['rgb_path']=str(image)
        (directory/'observation.json').write_text(json.dumps(observation))
        return observation


class HandoffCapturePaths(unittest.TestCase):
    def selection(self):
        snapshot=OBSERVATION['snapshot']
        return {'grasp_world':snapshot['link_world']['right_eef_link'],
            'target_contact_world':np.asarray(snapshot['objects']['test_object']['world'])[:3,3].tolist()}

    def rejected_check(self,out,reference=None):
        with patch.object(grasp_handoff,'local_callbacks',return_value=(MagicMock(),MagicMock())), \
             patch.object(grasp_handoff,'frustum',return_value={'pass':True}), \
             patch.object(grasp_handoff,'corridor',return_value=(None,{'pass':False,'rejection':'Mocked planning rejection'})):
            with self.assertRaisesRegex(RuntimeError,'No grasp-relative handoff'):
                grasp_handoff.select(CaptureDirectoryContract(),out,self.selection(),None,None,None,
                    MagicMock(),None,[],None,MagicMock(),[],grasp_handoff.RULE,camera_reference_dir=reference)

    def test_candidate_checks_can_use_distinct_native_capture_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            for number in range(2):
                reference=out/f'candidate_{number}'/'camera_reference'
                self.rejected_check(out,reference)
                report=json.loads((out/'handoff_candidates.json').read_text())
                self.assertEqual(report['camera_reference'],str(reference/'observation.json'))
                self.assertTrue((reference/'wrist.png').exists())

    def test_single_check_default_path_is_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);self.rejected_check(out)
            self.assertTrue((out/'handoff_camera_reference/observation.json').exists())

    def test_near_rules_preserve_arx_goal_axes_and_all_existing_gates(self):
        goal=np.asarray(self.selection()['grasp_world'])
        for rule in grasp_handoff.NEAR_RULES.values():
            option=grasp_handoff.candidates(goal,rule)[0];pose=np.asarray(option['world'])
            np.testing.assert_allclose(pose[:3,3],goal[:3,3]-rule['ordered_backoff_m'][0]*goal[:3,2],atol=1e-12)
            np.testing.assert_array_equal(pose[:3,:3],goal[:3,:3])
            for key in ('cartesian_step_m','rotation_step_rad','image_margin_pixels','reject_pregrasp_target_contact'):
                self.assertEqual(rule[key],grasp_handoff.RULE[key])
        altered=copy.deepcopy(grasp_handoff.NEAR_RULES['five_mm']);altered['image_margin_pixels']=0
        with self.assertRaises(ValueError):grasp_handoff.candidates(goal,altered)

    def via_entry(self, out, via_contacts, write_via=True):
        """Mock only planning, retaining the real recorded ARX wrist contract."""
        import torch
        selection=self.selection();selection['reach_via']={'world':selection['grasp_world']}
        (out/'pregrasp_execution.json').write_text(json.dumps({'contacts':[]}))
        if write_via:(out/'pregrasp_via_execution.json').write_text(json.dumps({'contacts':via_contacts}))
        observation=CaptureDirectoryContract().request('grasp_lilo_observation',str(out/'actual'))
        with patch.object(grasp_handoff,'local_callbacks',return_value=(MagicMock(),MagicMock())), \
             patch.object(grasp_handoff,'frustum',return_value={'pass':True}), \
             patch.object(grasp_handoff,'corridor',return_value=(torch.zeros(6),{'pass':True})):
            _,validate=grasp_handoff.select(CaptureDirectoryContract(),out,selection,torch.zeros(6),
                torch.zeros(6),lambda first,last:torch.zeros(2,6),MagicMock(),MagicMock(),
                [],MagicMock(),MagicMock(),[],grasp_handoff.NEAR_RULES['ten_mm'])
            return validate(observation)

    def test_via_contact_cannot_be_hidden_by_a_contact_free_final_leg(self):
        target=OBSERVATION['snapshot']['objects']['test_object']['path']
        event={'a':'/World/Robot/right_link27','b':target,'num_contacts':1}
        with tempfile.TemporaryDirectory() as temporary:
            result=self.via_entry(Path(temporary),[event])
        self.assertFalse(result['pass'])
        self.assertEqual(result['pregrasp_target_contact_events'],1)
        self.assertEqual(len(result['reach_execution_files']),2)
        self.assertEqual(result['rejection'],'Target was touched before VLA entry')

    def test_missing_via_execution_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):self.via_entry(Path(temporary),[],write_via=False)


if __name__=='__main__':unittest.main()
