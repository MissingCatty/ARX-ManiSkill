"""Check precision refinement and raw-target acceptance without a GPU."""
import unittest

import numpy as np
import torch

from modules.curobo.indexed_planning.geometric.kinematics import ArmChain
from .executable import ExecutableAnalyticIK


class ExecutableIKTests(unittest.TestCase):
    def test_known_near_singular_legal_state_has_float32_command(self):
        chain=ArmChain();q=np.array([[-.29189541,3.07382645,.99317040,-.39388423,-1.57045996,.62576842]])
        result=ExecutableAnalyticIK().solve(torch.tensor(chain.fk(q)))
        self.assertTrue(result['valid'].any())
        self.assertEqual(result['q'].dtype,torch.float32)

    def test_raw_nonorthogonal_rotations_still_require_original_pose_tolerance(self):
        chain=ArmChain();q=np.array([[.3,1.,1.2,.4,.2,.1]])
        original=chain.fk(q).astype(np.float32).astype(float)
        result=ExecutableAnalyticIK().solve(torch.tensor(original))
        self.assertTrue(result['valid'].any())
        self.assertLess(result['position_error_m'][result['valid']].max(),.002)
        self.assertLess(result['rotation_error_rad'][result['valid']].max(),.03)

    def test_bad_input_remains_rejected_after_rotation_normalization(self):
        target=torch.eye(4,dtype=torch.float64)[None].clone();target[0,0,0]=2
        result=ExecutableAnalyticIK().solve(target)
        self.assertFalse(result['valid_input'].any());self.assertFalse(result['valid'].any())


if __name__=='__main__':unittest.main()
