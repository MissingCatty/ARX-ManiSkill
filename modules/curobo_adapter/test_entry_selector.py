"""Ensure future grasp goals cannot block the executed pregrasp-only stage."""
import json
from pathlib import Path
import unittest

import numpy as np

from entry_selector import select_pregrasp, decode_candidate

TRIAL = Path('/home/zhaoyechong/projects/ARX-ManiSkill/modules/maniskill/outputs/arx_native_postprocessed_retry11_20261117_01/trials/B0001_S1_open_003_cracker_box')


class StubPlanner:
    def __init__(self, expected, invalid=False, path_failure=False, optimizer_failure=False):
        self.expected=expected;self.invalid=invalid;self.path_failure=path_failure
        self.optimizer_failure=optimizer_failure;self.calls=[]

    @property
    def closed_checker(self):
        raise AssertionError('Future nominal closure must not gate open transit')

    def measured(self, state):return np.zeros(6)

    def ik(self, target, start):
        assert np.array_equal(target,self.expected), 'Only unchanged H may be solved'
        self.calls.append('H')
        return None if self.invalid else np.ones(6)*.1

    def path(self, start, goal):
        self.calls.append('path')
        return None if self.path_failure else np.stack([start,goal])

    def optimize(self, label, path, duration):
        self.calls.append('optimize')
        if self.optimizer_failure:raise RuntimeError('optimizer rejected')
        return {'q':path.tolist(),'times':[0.,duration]}


class EntrySelectorTests(unittest.TestCase):
    def setUp(self):
        self.state=json.loads((TRIAL/'state.json').read_text())
        rows=np.load(TRIAL/'postprocessed_grasps.npy');rows=rows[np.argsort(-rows[:,0],kind='stable')]
        self.rows=rows[:1];self.camera=np.asarray(self.state['camera_world_optical'])
        self.h=decode_candidate(self.rows[0],self.camera)[1]

    def test_future_G_closure_lift_do_not_gate_H(self):
        p=StubPlanner(self.h);r=select_pregrasp(p,self.state,self.rows,self.camera)
        self.assertTrue(r['success']);self.assertFalse(r['full_grasp_validated'])
        self.assertEqual(p.calls,['H','path','optimize'])
        self.assertEqual(set(r['paths']),{'pregrasp'})

    def test_invalid_H_cannot_reach_path(self):
        p=StubPlanner(self.h,invalid=True);r=select_pregrasp(p,self.state,self.rows,self.camera)
        self.assertFalse(r['success']);self.assertEqual(p.calls,['H'])

    def test_invalid_path_cannot_reach_optimizer(self):
        p=StubPlanner(self.h,path_failure=True);r=select_pregrasp(p,self.state,self.rows,self.camera)
        self.assertFalse(r['success']);self.assertEqual(p.calls,['H','path'])

    def test_optimizer_failure_cannot_fall_back_to_PRM(self):
        p=StubPlanner(self.h,optimizer_failure=True)
        with self.assertRaisesRegex(RuntimeError,'optimizer rejected'):
            select_pregrasp(p,self.state,self.rows,self.camera)

    def test_unqueried_cached_goal_is_not_an_IK_failure(self):
        p=StubPlanner(self.h);r=select_pregrasp(p,self.state,self.rows,self.camera,solutions={})
        self.assertFalse(r['success']);self.assertEqual(r['rejected']['not_queried'],1)
        self.assertEqual(r['rejected']['pregrasp_ik'],0);self.assertEqual(p.calls,[])


if __name__=='__main__':unittest.main()
