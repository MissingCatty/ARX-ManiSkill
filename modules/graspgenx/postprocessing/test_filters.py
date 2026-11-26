"""Physical-frame, table-contact and native NMS regression tests."""
import itertools
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from modules.graspgenx.arx_frames import T_EEF_GENERATOR
from .geometry import ConvexPart,GripperGeometry,poses_from_candidates,observed_cloud
from .nms import native_nms


def cube(lower,upper):return np.array(list(itertools.product(*zip(lower,upper))),dtype=float)


class GeometryTests(unittest.TestCase):
    def test_strict_interior_avoids_surface_contact_rejection(self):
        part=ConvexPart.create('cube',cube([-1,-1,-1],[1,1,1]))
        np.testing.assert_array_equal(part.contains([[0,0,0],[1,0,0],[.9995,0,0]],.001),[True,False,False])

    def test_planar_piece_has_no_invented_solid_volume(self):
        part=ConvexPart.create('plane',[[0,0,0],[1,0,0],[0,1,0]])
        self.assertIsNone(part.equations);self.assertFalse(part.contains([[.1,.1,0]]).any())

    def test_nonfinite_mesh_is_rejected(self):
        with self.assertRaises(ValueError):ConvexPart.create('bad',[[np.nan,0,0]])

    def test_generator_to_TCP_and_back_preserves_EEF(self):
        eef=np.eye(4);eef[:3,:3]=Rotation.from_euler('xyz',[.3,-.7,1.1]).as_matrix();eef[:3,3]=[.4,.1,.2]
        generator=eef@T_EEF_GENERATOR;tcp=eef.copy();tcp[:3,3]+=.043*eef[:3,2]
        original=tcp.copy()
        _,actual=poses_from_candidates([{'pose':tcp,'metadata':{'source_generator_pose_world':generator}}])
        np.testing.assert_allclose(actual[0],eef,atol=1e-12);np.testing.assert_array_equal(tcp,original)

    def test_wrong_generator_offset_is_rejected(self):
        eef=np.eye(4);tcp=eef.copy();tcp[2,3]=.043
        with self.assertRaises(ValueError):poses_from_candidates([{'pose':tcp,'metadata':{'source_generator_pose_world':eef}}])

    def test_nonrigid_candidate_is_rejected(self):
        t=np.eye(4);t[0,0]=2
        with self.assertRaises(ValueError):poses_from_candidates([{'pose':t}])

    def table_fixture(self,finger=True,lower=.04,upper=.05):
        model=GripperGeometry.__new__(GripperGeometry);model.tip=.05
        model.parts=[ConvexPart.create('finger' if finger else 'palm',cube([-.004,-.004,lower],[.004,.004,upper]),finger)]
        table={'center':[0,0,-.2],'dims':[2,2,.4]}
        pose=np.eye(4);pose[:3,:3]=np.diag([1,-1,-1]);pose[2,3]=.049
        return model,table,pose

    def test_one_mm_distal_contact_retains_V2_allowance(self):
        model,table,pose=self.table_fixture();bad,_=model.table_rejections(pose[None],table)
        self.assertFalse(bad[0])

    def test_more_than_two_mm_distal_penetration_rejected(self):
        model,table,pose=self.table_fixture();pose[2,3]=.0479
        self.assertTrue(model.table_rejections(pose[None],table)[0][0])

    def test_proximal_finger_contact_not_given_tip_allowance(self):
        model,table,pose=self.table_fixture(lower=.02,upper=.03);pose[2,3]=.029
        self.assertTrue(model.table_rejections(pose[None],table)[0][0])

    def test_palm_contact_not_given_tip_allowance(self):
        model,table,pose=self.table_fixture(finger=False)
        self.assertTrue(model.table_rejections(pose[None],table)[0][0])

    def test_table_height_not_treated_as_infinite_plane(self):
        model,table,pose=self.table_fixture(finger=False);pose[0,3]=5
        self.assertFalse(model.table_rejections(pose[None],table)[0][0])

    def test_authored_left_and_right_gripper_frames_have_valid_cavities(self):
        for side in ['right','left']:
            model=GripperGeometry(side)
            lower,upper=model.cavity
            self.assertTrue((upper>lower).all());self.assertTrue(.02<model.tip<.08)
            self.assertTrue(model.parts);self.assertTrue(all(p.link.startswith(side) for p in model.parts))

    def test_camera_backprojection_uses_saved_optical_transform(self):
        t=np.eye(4);t[:3,3]=[1,2,3]
        s={'intrinsics':np.eye(3),'camera_world_optical':t}
        points,mask=observed_cloud(s,np.array([[.5,0.]]),np.array([[1,0]]))
        np.testing.assert_allclose(points,[[1,2,3.5]]);np.testing.assert_array_equal(mask,[True])

    def test_batch_convex_kernel_matches_reference_for_both_physical_hands(self):
        rng=np.random.default_rng(526)
        points=rng.uniform([-.2,-.2,-.2],[.2,.2,.2],size=(5000,3))
        poses=np.repeat(np.eye(4)[None],24,axis=0)
        poses[:,:3,:3]=Rotation.random(24,random_state=rng).as_matrix()
        poses[:,:3,3]=rng.uniform(-.05,.05,size=(24,3))
        skip=np.arange(24)%7==0
        for side in ['right','left']:
            model=GripperGeometry(side)
            for actual,expected in zip(model.point_rejections(poses,points,skip=skip),
                    model.point_rejections_reference(poses,points,skip=skip)):
                np.testing.assert_array_equal(actual,expected)

    def test_batch_cavity_query_matches_unpruned_reference(self):
        rng=np.random.default_rng(527);model=GripperGeometry()
        points=rng.uniform([-.06,-.1,-.02],[.06,.1,.09],size=(5000,3))
        poses=np.repeat(np.eye(4)[None],24,axis=0)
        poses[:,:3,:3]=Rotation.random(24,random_state=rng).as_matrix()
        poses[:,:3,3]=rng.uniform(-.03,.03,size=(24,3))
        np.testing.assert_array_equal(model.cavity_counts(poses,points),model.cavity_counts_reference(poses,points))

    def test_batch_convex_kernel_handles_empty_cloud_and_skipped_rows(self):
        model=GripperGeometry();poses=np.repeat(np.eye(4)[None],3,axis=0)
        bad,counts=model.point_rejections(poses,np.empty((0,3)),skip=[False,True,False])
        self.assertFalse(bad.any());self.assertFalse(counts.any())

    def test_batch_convex_kernel_rejects_invalid_memory_shapes(self):
        model=GripperGeometry();poses=np.eye(4)[None]
        with self.assertRaises(ValueError):model.point_rejections(poses,np.zeros((1,2)))
        with self.assertRaises(ValueError):model.point_rejections(poses,np.array([[np.nan,0,0]]))
        with self.assertRaises(ValueError):model.point_rejections(poses,np.zeros((1,3)),skip=[False,False])


class NativeNMSTests(unittest.TestCase):
    def setup_poses(self):
        poses=np.repeat(np.eye(4)[None],3,axis=0);poses[1,0,3]=.001;poses[2,0,3]=.1
        return poses,np.array([.9,.8,.7]),np.full(3,.088)

    def test_native_NMS_keeps_original_ID_and_deferred_fallback(self):
        p,s,w=self.setup_poses();a=native_nms(p,s,w,.005,np.deg2rad(5),[7,21,99])
        self.assertEqual(a['representatives'],[7,99]);self.assertEqual(a['deferred'],[21])
        self.assertEqual(a['representative_for_input'],[7,7,99])

    def test_180_degree_flip_is_not_silently_equated(self):
        p,s,w=self.setup_poses();p[1,:3,:3]=Rotation.from_euler('z',np.pi).as_matrix()
        a=native_nms(p,s,w,.005,np.deg2rad(5))
        self.assertEqual(len(a['representatives']),3)

    def test_candidates_and_scores_are_never_modified(self):
        p,s,w=self.setup_poses();before=[x.copy() for x in [p,s,w]]
        native_nms(p,s,w,.005,np.deg2rad(5))
        for a,b in zip([p,s,w],before):np.testing.assert_array_equal(a,b)

    def test_empty_candidates_remain_empty(self):
        a=native_nms(np.empty((0,4,4)),np.array([]),np.array([]),.005,np.deg2rad(5))
        self.assertEqual(a['representatives'],[]);self.assertEqual(a['deferred'],[])

    def test_duplicate_source_ID_is_rejected(self):
        p,s,w=self.setup_poses()
        with self.assertRaises(ValueError):native_nms(p,s,w,.005,np.deg2rad(5),[7,7,9])


if __name__=='__main__':unittest.main()
