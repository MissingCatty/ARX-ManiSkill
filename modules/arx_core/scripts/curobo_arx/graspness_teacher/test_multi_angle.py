"""Proposal-interface regressions; physical ARX evidence is recorded separately."""
import unittest
import numpy as np
import torch
from .multi_angle import decode,merge_pools


def rotation(approach,angle):
    # Only test field selection here, not substitute robot kinematics.
    return torch.eye(3).repeat(len(angle),1,1)


class MultiAngleTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        return {'point_clouds':[None],'grasp_score_pred':torch.rand(1,7,12,4),
            'grasp_width_pred':torch.rand(1,7,12,4),'xyz_graspable':torch.rand(1,7,3),
            'grasp_top_view_xyz':torch.rand(1,7,3)}

    def test_primary_scores_depths_and_widths_match_source_argmax(self):
        e=self.fixture();rows=decode(e,rotation)[0];score=e['grasp_score_pred'][0].reshape(7,-1)
        best,index=score.max(-1)
        self.assertTrue(torch.equal(rows[:7,0],best))
        self.assertTrue(torch.equal(rows[:7,3],(index%4+1)*.01))
        expected=torch.clamp((1.2*e['grasp_width_pred'][0]/10.).reshape(7,-1).gather(1,index[:,None]),0.,.1)
        self.assertTrue(torch.equal(rows[:7,1:2],expected))
        self.assertEqual(rows.shape,(28,17))
        for rank in range(4):self.assertTrue(torch.equal(rows[rank*7:(rank+1)*7,13:16],e['xyz_graspable'][0]))

    def test_each_roll_uses_its_own_depth_and_predicted_aperture(self):
        e=self.fixture();e['grasp_score_pred'].zero_()
        for roll in range(12):
            e['grasp_score_pred'][0,:,roll,roll%4]=12-roll
            e['grasp_width_pred'][0,:,roll,:]=roll*.01
        rows=decode(e,rotation)[0]
        for rank in range(4):
            np.testing.assert_allclose(rows[rank*7:(rank+1)*7,3],(rank%4+1)*.01)
            np.testing.assert_allclose(rows[rank*7:(rank+1)*7,1],1.2*rank*.01/10.)

    def test_prefix_is_preserved_and_new_ids_are_deduplicated(self):
        old=np.eye(4);new=np.eye(4);new[0,3]=.02
        original={'candidates':[{'goal_id':'old','score':.9,'grasp_world':old.tolist()}],'point_cloud':{'hash':'same'}}
        extension={'candidates':[{'goal_id':'old','score':.8,'grasp_world':old.tolist()},{'goal_id':'new','score':.7,'grasp_world':new.tolist()}]}
        merged=merge_pools(original,extension,32)
        self.assertEqual(merged['candidates'][0],original['candidates'][0])
        self.assertEqual(len(merged['candidates']),2)
        self.assertEqual(merged['point_cloud'],original['point_cloud'])
        self.assertEqual(len(original['candidates']),1)
        with self.assertRaises(ValueError):merge_pools(original,extension,0)

    def test_float_rounding_cannot_create_a_new_grasp_from_a_different_id(self):
        old=np.eye(4);shifted=old.copy();shifted[:3,3]+=.000001
        original={'candidates':[{'goal_id':'gpu_id','grasp_world':old.tolist()}]}
        extension={'candidates':[{'goal_id':'cpu_id','grasp_world':shifted.tolist()}]}
        merged=merge_pools(original,extension,32)
        self.assertEqual(len(merged['candidates']),1)

    def test_unknown_head_shapes_fail(self):
        for k in [0,13]:
            with self.assertRaises(ValueError):decode(self.fixture(),rotation,angles=k)


if __name__=='__main__':unittest.main()
