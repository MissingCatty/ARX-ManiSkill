import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read
from .load_lift import goals,retime
from .paired_runtime import PlannerArm


class LoadLiftTests(unittest.TestCase):
    def episode(self):
        return next((ROOT/'runs/graspness_success_20261104_03/physical_M06').glob(
            'groups/*/trials/01*/episodes/*/episode.json')).parent

    def test_goal_search_preserves_actual_arx_orientation_height_and_input(self):
        snapshot=read(self.episode()/'close_execution.json')['final']
        original=np.array(snapshot['link_world']['right_eef_link']);original[2,3]+=.12
        before=original.copy();body=np.array(snapshot['link_world']['body4'])
        candidates=goals(original,body,[.075,.15,.225,.30],.30)
        self.assertGreater(len(candidates),1)
        for distance,goal in candidates:
            np.testing.assert_equal(goal[:3,:3],original[:3,:3]);self.assertEqual(goal[2,3],original[2,3])
            self.assertAlmostEqual(np.linalg.norm(goal[:2,3]-original[:2,3]),distance)
            self.assertGreaterEqual(np.linalg.norm(goal[:2,3]-body[:2,3]),.30-1e-9)
        np.testing.assert_equal(original,before)

    def test_retiming_only_slows_real_arx_recorded_configuration_sequence(self):
        rows=[json.loads(line) for line in (self.episode()/'samples.jsonl').read_text().splitlines()]
        poses=np.array([rows[0]['snapshot']['link_world']['right_eef_link'],rows[-1]['snapshot']['link_world']['right_eef_link']])
        distance=np.linalg.norm(poses[1,:3,3]-poses[0,:3,3])
        joint=[]
        for row in [rows[0],rows[-1]]:
            snapshot=row['snapshot'];joint.append([snapshot['joint_positions'][snapshot['joint_names'].index(n)] for n in snapshot['right_arm']])
        command={'time_from_start':[0.,distance/.4],'position':joint}
        original=copy.deepcopy(command);result,evidence=retime(command,poses)
        self.assertGreaterEqual(evidence['scale'],1.999)
        self.assertLessEqual(distance/result['time_from_start'][-1],.200001)
        self.assertEqual(result['position'],command['position']);self.assertEqual(command,original)

    def test_planner_comparison_arm_is_explicit_and_original_payload_unchanged(self):
        class Client:
            def submit(self,action,**kwargs):self.action,self.payload=action,kwargs;return 'job'
            def poll(self,job):return {'job':job}
        client=Client();payload={'phase':'lift','goal_world':[[1,0],[0,1]]};before=copy.deepcopy(payload)
        arm=PlannerArm(client,'baseline');self.assertEqual(arm.submit('plan',**payload),'job')
        self.assertEqual(client.payload['comparison_arm'],'baseline');self.assertEqual(payload,before)
        self.assertEqual(arm.poll('job'),{'job':'job'})


if __name__=='__main__':unittest.main()
