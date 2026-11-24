"""Protocol/control checks against recorded ARX data; physical replay is separate."""
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from observations import action_target,teacher_action,relative_state,encode_state,smooth_segment,mask_distractors,target_projection

ROOT=Path(__file__).resolve().parents[3]
FIXTURES=ROOT/'servers/lilo_vla/tests/fixtures'
DATA=json.loads((FIXTURES/'arx_fixture.json').read_text())


class ArxInteraction(unittest.TestCase):
    def test_recorded_arx_relative_state_convention(self):
        observation=DATA['observation']
        np.testing.assert_allclose(relative_state(observation['snapshot']),observation['state'],rtol=0,atol=1e-7)

    def test_versioned_default_preserves_recorded_arx_state(self):
        observation=DATA['observation']
        np.testing.assert_array_equal(encode_state(observation['snapshot']),relative_state(observation['snapshot']))

    def test_grasp_error_uses_same_world_axes_as_executed_action(self):
        snapshot=DATA['observation']['snapshot']
        ee=np.asarray(snapshot['link_world']['right_eef_link'])
        action=np.array([.2,-.1,.3,.1,-.2,.15,1.])
        goal,_=action_target(ee,action)
        error=encode_state(snapshot,'grasp_goal_world_error_v1',goal)
        np.testing.assert_allclose(error[:3],-.05*action[:3],rtol=0,atol=1e-12)
        np.testing.assert_allclose(error[3:6],-.5*action[3:6],rtol=0,atol=1e-6)
        np.testing.assert_array_equal(error[6:],relative_state(snapshot)[6:])

    def test_measured_eef_at_goal_has_zero_goal_error(self):
        snapshot=DATA['observation']['snapshot']
        ee=snapshot['link_world']['right_eef_link']
        np.testing.assert_allclose(encode_state(snapshot,'grasp_goal_world_error_v1',ee)[:6],0,atol=1e-6)

    def test_goal_state_requires_explicit_rigid_planned_goal(self):
        snapshot=DATA['observation']['snapshot']
        with self.assertRaises(ValueError):encode_state(snapshot,'grasp_goal_world_error_v1')
        nonrigid=np.eye(4);nonrigid[0,0]=2
        with self.assertRaises(ValueError):encode_state(snapshot,'grasp_goal_world_error_v1',nonrigid)
        with self.assertRaises(ValueError):encode_state(snapshot,'unknown_version',np.eye(4))

    def test_measured_arx_motion_round_trip(self):
        a,b=DATA['measured_pose_pair']
        row=lambda pose:{'snapshot':{'link_world':{'right_eef_link':pose}},'teacher_command':{'gripper_target':-3.2}}
        action=teacher_action(row(a),row(b));reconstructed,gripper=action_target(a,action)
        np.testing.assert_allclose(reconstructed,b,rtol=0,atol=1e-6)
        self.assertEqual(gripper,-3.2)

    def test_cartesian_units_and_existing_gripper_api(self):
        ee=np.asarray(DATA['measured_pose_pair'][0]);action=np.zeros(7);action[0]=.1
        target,gripper=action_target(ee,action)
        self.assertAlmostEqual(target[0,3]-ee[0,3],.005,places=9)
        self.assertEqual(gripper,.1)
        action[6]=1.;self.assertEqual(action_target(ee,action)[1],-3.2)

    def test_invalid_policy_actions_are_rejected(self):
        for action in (np.full(7,np.nan),np.r_[1.01,np.zeros(6)]):
            with self.assertRaises(ValueError):action_target(DATA['measured_pose_pair'][0],action)

    def test_arx_segment_preserves_playback_bounds(self):
        start,goal=DATA['joint_pair'];q,times=smooth_segment(start,goal)
        np.testing.assert_allclose(q[0],start,atol=1e-12)
        np.testing.assert_allclose(q[-1],goal,atol=1e-12)
        delta=np.diff(times);self.assertTrue((delta>0).all())
        derivative=q
        for bound in (1.5,5.,100.):
            derivative=np.diff(derivative,axis=0)/delta[0]
            self.assertLessEqual(float(np.abs(derivative).max()),bound+1e-6)

    def test_mask_keeps_actual_arx_target_visible(self):
        image=np.asarray(Image.open(FIXTURES/'arx_wrist_pregrasp.png'))
        masked,boxes=mask_distractors(image,DATA['observation'])
        xy=target_projection(DATA['observation'],None);x,y=np.round(xy).astype(int)
        self.assertTrue(0<=x<image.shape[1] and 0<=y<image.shape[0])
        np.testing.assert_array_equal(masked[y,x],image[y,x])
        self.assertTrue(boxes)
        self.assertTrue(np.any(masked!=image))


if __name__=='__main__':unittest.main()
