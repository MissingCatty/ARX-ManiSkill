"""Endpoint staging, score order, table exceptions, and partial batches."""
import unittest

import numpy as np

from .bank import EndpointBank,endpoint_poses


class FakeSession:
    def __init__(self):self.sizes=[]
    def host_query(self,poses,widths):
        n=len(poses);self.sizes.append(n)
        q=np.repeat(poses[:,:3,3],2,axis=1)[:,None].repeat(8,axis=1)
        valid=np.zeros((n,8),bool);valid[:,0]=True
        world=valid.copy();world[::3]=False
        return {'q':q,'valid_ik':valid,'valid_world':world,'valid_collision':world,
            'valid_input':np.ones(n,bool),'needs_fallback':np.zeros(n,bool)}


class BankTests(unittest.TestCase):
    def candidates(self,n):
        return [{'pose':np.eye(4),'width':.04+i*.00001,'score':n-i} for i in range(n)]

    def test_original_grasp_pre_and_lift_offsets(self):
        poses,widths=endpoint_poses(self.candidates(1))
        np.testing.assert_allclose(poses[:,2,3],[-.043,-.123,-.013])
        np.testing.assert_allclose(widths,[.04]*3)

    def test_batches_cover_all_targets_and_tail_without_score_changes(self):
        session=FakeSession();bank=EndpointBank(session,self.candidates(100),np.eye(4))
        self.assertEqual(session.sizes,[256,44]);self.assertEqual(len(bank.poses),300)

    def test_strict_table_collision_does_not_delete_grasp_branch(self):
        bank=EndpointBank(FakeSession(),self.candidates(1),np.eye(4))
        self.assertEqual(len(bank.branches('grasp',np.zeros(6))),1)
        bank.results['valid_world'][1]=False
        self.assertEqual(len(bank.branches('pregrasp',np.zeros(6))),0)

    def test_wrong_candidate_target_does_not_match(self):
        bank=EndpointBank(FakeSession(),self.candidates(1),np.eye(4));pose=bank.poses[0].copy();pose[0,3]+=.001
        self.assertFalse(bank.matches('grasp',pose))


if __name__=='__main__':unittest.main()
