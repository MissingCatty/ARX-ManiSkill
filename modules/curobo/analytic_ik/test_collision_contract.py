"""Validate measured-width packing without allocating a CUDA context."""
from copy import deepcopy
import unittest

import torch
import yaml

from arx_project.layout import ROOT
from .collision import BranchCollisionChecker,unlocked_collision_model


class CollisionContractTests(unittest.TestCase):
    def test_unlocking_changes_only_finger_locks(self):
        path=ROOT/'modules/arx_core/scripts/curobo_arx/static_pick/reference/model/arx.yml'
        original=yaml.load(path.read_text(),Loader=yaml.CSafeLoader);expected=deepcopy(original)
        expected['robot_cfg']['kinematics']['lock_joints']={}
        actual=unlocked_collision_model(original)
        self.assertEqual(actual,expected)
        self.assertEqual(len(original['robot_cfg']['kinematics']['lock_joints']),2)

    def test_each_candidate_branch_has_its_own_measured_width(self):
        checker=BranchCollisionChecker.__new__(BranchCollisionChecker)
        q=torch.arange(3*8*6,dtype=torch.float32).reshape(3,8,6)
        widths=torch.tensor([.088,.06,.04])
        result=checker.joint_rows(q,widths).reshape(3,8,8)
        torch.testing.assert_close(result[:,:,:6],q)
        torch.testing.assert_close(result[:,:,6],widths[:,None].expand(3,8)/2)
        torch.testing.assert_close(result[:,:,7],widths[:,None].expand(3,8)/2)

    def test_wrong_branch_capacity_is_rejected(self):
        checker=BranchCollisionChecker.__new__(BranchCollisionChecker)
        with self.assertRaises(ValueError):checker.joint_rows(torch.zeros(3,4,6),torch.ones(3))


if __name__=='__main__':unittest.main()
