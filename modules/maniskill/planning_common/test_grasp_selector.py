"""Offline routing and pose checks with explicit mathematical fixtures."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np

from common import ROOT,read,ordinary_planner_rejection
from grasp_selector import ARX_AXES,decode_candidate,select_postprocessed


def fixture(count=48):
    rows=np.zeros((count,17),dtype=float)
    rows[:,0]=np.linspace(.95,.1,count);rows[:,1:4]=[.07,.02,.03]
    rows[:,4:13]=np.eye(3).reshape(1,9)
    rows[:,13:16]=np.c_[np.linspace(-.05,.05,count),np.full(count,.02),np.full(count,.4)]
    rows[:,16]=-1
    camera=np.array([[0.,-1.,0.,.2],[1.,0.,0.,-.1],[0.,0.,1.,.3],[0.,0.,0.,1.]])
    state={'camera_world_optical':camera.tolist(),'object_center':[.2,.1,.7]}
    return rows,state,camera


def planner():
    return SimpleNamespace(measured=MagicMock(return_value=np.zeros(6)),
        ik=MagicMock(return_value=np.zeros(6)),
        closed_checker=SimpleNamespace(validate=MagicMock(return_value=np.array([True]))),
        path=MagicMock(return_value=np.zeros((2,6))),
        optimize=MagicMock(side_effect=lambda name,path,seconds:dict(q=path.tolist(),times=[0.,seconds])))


class NativeSelectorTests(unittest.TestCase):
    def setUp(self):
        self.rows,self.state,self.camera=fixture(48)

    def test_first_native_box_pose_reaches_feasibility_checks_without_geometric_filter(self):
        p=planner();state=dict(self.state);state.pop('object_center')
        result=select_postprocessed(p,state,self.rows,self.camera)
        self.assertTrue(result['success']);self.assertEqual(result['selected']['index'],0)
        np.testing.assert_array_equal(result['selected']['network_row'],self.rows[0])
        self.assertFalse(result['geometric_prefilters'])
        expected=decode_candidate(self.rows[0],self.camera)[0]
        np.testing.assert_allclose(p.ik.call_args_list[0].args[0],expected,rtol=0,atol=1e-12)

    def test_all_48_candidates_are_submitted_once_when_ik_rejects_them(self):
        p=planner();p.ik.return_value=None
        result=select_postprocessed(p,self.state,self.rows,self.camera)
        self.assertEqual(p.ik.call_count,48);self.assertEqual(result['rejected']['grasp_ik'],48)
        self.assertEqual(ordinary_planner_rejection(result),'no_executable_postprocessed_candidate')
        p.path.assert_not_called()

    def test_all_42_candidates_are_submitted_without_width_filter(self):
        rows,state,camera=fixture(42);p=planner();p.ik.return_value=None
        result=select_postprocessed(p,state,rows,camera)
        self.assertEqual(p.ik.call_count,42);self.assertEqual(result['rejected']['grasp_ik'],42)

    def test_object_center_cannot_change_decoded_pose_or_candidate_choice(self):
        p=planner();one=select_postprocessed(p,self.state,self.rows[:1],self.camera)
        state={**self.state,'object_center':[999.,-999.,999.]};p=planner()
        two=select_postprocessed(p,state,self.rows[:1],self.camera)
        np.testing.assert_array_equal(one['selected']['grasp_world'],two['selected']['grasp_world'])
        self.assertEqual(two['selected']['height_adjustment_m'],0.)

    def test_decode_only_applies_existing_camera_and_tool_frame_mapping(self):
        row=self.rows[0];g,pre,lift,contact=decode_candidate(row,self.camera)
        r=self.camera[:3,:3]@row[4:13].reshape(3,3)
        native=self.camera[:3,:3]@row[13:16]+self.camera[:3,3]+row[3]*r[:,0]
        np.testing.assert_allclose(contact,native,rtol=0,atol=1e-12)
        np.testing.assert_allclose(g[:3,:3],r@ARX_AXES,rtol=0,atol=1e-12)
        np.testing.assert_allclose(g[:3,3]+.040*g[:3,2],native,rtol=0,atol=1e-12)
        np.testing.assert_allclose(pre[:3,3],g[:3,3]-.08*g[:3,2],rtol=0,atol=1e-12)
        np.testing.assert_allclose(lift[:3,3],g[:3,3]+[0,0,.12],rtol=0,atol=1e-12)

    def test_closed_robot_collision_still_rejects_candidate(self):
        p=planner();p.closed_checker.validate.return_value=np.array([False])
        result=select_postprocessed(p,self.state,self.rows[:1],self.camera)
        self.assertFalse(result['success']);self.assertEqual(result['rejected']['closed_collision'],1)
        p.path.assert_not_called()

    def test_pregrasp_and_lift_rejections_are_recorded_separately(self):
        for answers,label in [([np.zeros(6),None],'pregrasp_ik'),
                              ([np.zeros(6),np.zeros(6),None],'lift_ik')]:
            with self.subTest(label=label):
                p=planner();p.ik.side_effect=answers
                result=select_postprocessed(p,self.state,self.rows[:1],self.camera)
                self.assertEqual(result['rejected'][label],1);p.path.assert_not_called()

    def test_path_validation_still_rejects_candidate(self):
        p=planner();p.path.return_value=None
        result=select_postprocessed(p,self.state,self.rows[:1],self.camera)
        self.assertFalse(result['success']);self.assertEqual(result['rejected']['pregrasp_path'],1)
        p.optimize.assert_not_called()

    def test_optimizer_error_propagates_without_changing_candidate(self):
        p=planner();p.optimize.side_effect=RuntimeError('optimizer rejection')
        with self.assertRaisesRegex(RuntimeError,'optimizer rejection'):
            select_postprocessed(p,self.state,self.rows,self.camera)
        self.assertEqual(p.ik.call_count,3)


if __name__=='__main__':unittest.main(verbosity=2)
