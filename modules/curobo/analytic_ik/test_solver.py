"""Verify analytical branches against independently composed physical URDF FK."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from modules.curobo.indexed_planning.geometric.kinematics import ArmChain
from .model import AnalyticModel
from .solver import BatchAnalyticIK


class AnalyticIKTests(unittest.TestCase):
    def test_geometry_has_intersecting_axes_and_unchanged_axis_lines(self):
        for arm in ['right','left']:
            model = AnalyticModel(arm)
            self.assertLess(model.intersection_residual_m,1e-12)
            self.assertLess(np.linalg.norm(model.P[:,5]),1e-12)
            np.testing.assert_allclose(model.H[:,1],model.H[:,2],atol=1e-14)
            np.testing.assert_allclose(model.H[:,1],model.H[:,3],atol=1e-14)

    def test_recover_legal_random_physical_states_both_arms(self):
        for arm in ['right','left']:
            chain = ArmChain(arm);rng=np.random.default_rng(20261127)
            q=rng.uniform(chain.limits[:,0]+.01,chain.limits[:,1]-.01,(256,6))
            answer=BatchAnalyticIK(arm).solve(torch.tensor(chain.fk(q)))
            valid=answer['valid'].numpy();values=answer['q'].numpy()
            self.assertTrue(valid.any(-1).all())
            distance=np.linalg.norm(values-q[:,None],axis=-1)
            self.assertTrue((np.where(valid,distance,np.inf).min(-1)<1e-7).all())

    def test_every_admitted_branch_passes_independent_URDF_FK(self):
        chain=ArmChain();rng=np.random.default_rng(641)
        q=rng.uniform(chain.limits[:,0],chain.limits[:,1],(256,6));targets=chain.fk(q)
        result=BatchAnalyticIK().solve(torch.tensor(targets))
        indices,branches=np.nonzero(result['valid'].numpy())
        values=result['q'].numpy()[indices,branches];actual=chain.fk(values)
        self.assertLess(np.linalg.norm(actual[:,:3,3]-targets[indices,:3,3],axis=-1).max(),1e-10)
        angular=Rotation.from_matrix(actual[:,:3,:3]@targets[indices,:3,:3].transpose(0,2,1)).magnitude()
        self.assertLess(angular.max(),1e-10)

    def test_angles_above_pi_are_retained_inside_authored_limits(self):
        chain=ArmChain();q=np.array([[.3,3.5,3.2,.4,.2,-.6]])
        result=BatchAnalyticIK().solve(torch.tensor(chain.fk(q)))
        values=result['q'][0][result['valid'][0]].numpy()
        self.assertLess(np.linalg.norm(values-q[0],axis=-1).min(),1e-8)

    def test_singular_wrist_is_explicitly_flagged(self):
        chain=ArmChain();q=np.array([[.3,1.,1.2,.4,np.pi/2,.2],[.3,1.,1.2,.4,-np.pi/2,.2]])
        result=BatchAnalyticIK().solve(torch.tensor(chain.fk(q)))
        self.assertTrue(result['needs_fallback'].all())
        self.assertTrue(result['singular'].all())

    def test_unreachable_target_has_no_admitted_branch(self):
        target=torch.eye(4,dtype=torch.float64)[None].clone();target[0,:3,3]=torch.tensor([5.,5.,5.])
        result=BatchAnalyticIK().solve(target)
        self.assertFalse(result['valid'].any())
        self.assertTrue(result['needs_fallback'].all())

    def test_invalid_pose_is_masked_and_never_produces_nan_output(self):
        targets=torch.eye(4,dtype=torch.float64)[None].repeat(3,1,1)
        targets[0,0,0]=float('nan');targets[1,0,0]=2;targets[2,3,0]=1
        result=BatchAnalyticIK().solve(targets)
        self.assertFalse(result['valid_input'].any());self.assertFalse(result['valid'].any())
        self.assertTrue(torch.isfinite(result['q']).all())

    def test_float32_returns_executable_joints_inside_double_precision_limits(self):
        chain=ArmChain();rng=np.random.default_rng(751)
        q=rng.uniform(chain.limits[:,0],chain.limits[:,1],(1024,6))
        result=BatchAnalyticIK(dtype=torch.float32).solve(torch.tensor(chain.fk(q),dtype=torch.float32))
        valid=result['valid'].numpy();values=result['q'].numpy()[valid].astype(float)
        self.assertTrue((values>=chain.limits[:,0]).all());self.assertTrue((values<=chain.limits[:,1]).all())
        self.assertTrue(result['valid'].any(-1).all())


if __name__=='__main__':unittest.main()
