"""Failure and geometry tests for hard filtering and explicit refinements."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from modules.graspgenx.arx_frames import T_EEF_GENERATOR,generator_to_contact
from modules.graspgenx.postprocessing.geometry import GripperGeometry
from .kernels import native_detect,variable_collision
from .processor import hard_nms,load_raw,make_pool,pad_refinement,table_bad


def reference(poses,points):
    """Independent direct NumPy transcription of the original detector."""
    bad=[];rechecked=[];widths=[];depths=[];counts=[]
    for pose in poses:
        p=(points-pose[:3,3])@pose[:3,:3]
        q=np.c_[p[:,2]+.04,p[:,1],-p[:,0]]
        h=np.abs(q[:,2])<.01;x=(q[:,0]>-.02)&(q[:,0]<.04)
        inner=h&x&(q[:,1]>=-.044)&(q[:,1]<=.044)
        contacts=q.copy();contacts[:,1]*=inner
        il=np.argmin(.044+contacts[:,1]);ir=np.argmin(.044-contacts[:,1])
        d=min(.044+contacts[il,1],.044-contacts[ir,1]);w=.088+2*(np.clip(d,.0025,.01)-d)
        depth=np.clip(max(contacts[il,0],contacts[ir,0]),0,.04)
        def collision(width):
            half=width/2
            fingers=(q[:,0]>-.02)&(q[:,0]<depth)&(((q[:,1]>-half-.01)&(q[:,1]<-half))|((q[:,1]>half)&(q[:,1]<half+.01)))
            base=(q[:,0]<-.02)&(q[:,0]>-.03)&(q[:,1]>-half-.01)&(q[:,1]<half+.01)
            return (h&(fingers|base)).any() or inner.sum()<10
        bad.append(collision(.088));rechecked.append(collision(w) or w>.088+1e-12 or w<=0)
        widths.append(w);depths.append(depth);counts.append(inner.sum())
    return dict(bad=bad,final_bad=rechecked,widths=widths,depths=depths,counts=counts)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.gripper=GripperGeometry()

    def test_native_algorithm_matches_independent_reference(self):
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(42);poses=np.tile(np.eye(4),(15,1,1))
        poses[:,:3,:3]=Rotation.random(15,random_state=rng).as_matrix()
        poses[:,:3,3]=rng.uniform(-.015,.015,(15,3));points=rng.uniform(-.08,.08,(2000,3))
        actual=native_detect(poses,points);expected=reference(poses,points)
        for name in expected:np.testing.assert_allclose(actual[name],expected[name],atol=1e-14,rtol=0,err_msg=name)

    def test_empty_cloud_is_rejected_without_fallback(self):
        value=native_detect(np.eye(4)[None],np.empty((0,3)))
        self.assertTrue(value['bad'][0]);self.assertTrue(value['final_bad'][0])

    def test_expansion_cannot_exceed_real_stroke(self):
        p=np.tile([0,.043,-.01],(20,1))
        value=native_detect(np.eye(4)[None],p)
        self.assertGreater(value['widths'][0],.088);self.assertTrue(value['final_bad'][0])

    def test_all_rejected_stays_empty_after_nms_and_packaging(self):
        poses=np.tile(np.eye(4),(3,1,1));ids,_=hard_nms(poses,np.ones(3),np.full(3,.088),np.zeros(3,bool),.03,30)
        self.assertEqual(ids,[])
        value=make_pool({'candidates':[{}]*3},{'selected':ids},np.arange(3),'native')
        self.assertEqual(value['candidates'],[]);self.assertFalse(value['raw_fallback'])

    def test_nms_really_deletes_duplicates(self):
        poses=np.tile(np.eye(4),(4,1,1));poses[2,0,3]=.001;poses[3,0,3]=.04
        ids,result=hard_nms(poses,np.array([4.,3.,2.,1.]),np.full(4,.088),np.ones(4,bool),.03,30)
        self.assertEqual(ids,[0,3]);self.assertEqual(len(result['deferred']),2)

    def test_raw_pose_mapping_is_sorted_and_frame_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'graspgenx_raw').mkdir()
            eef=np.eye(4);eef[:3,3]=[.7,-.2,.3]
            from scipy.spatial.transform import Rotation
            eef[:3,:3]=Rotation.from_euler('xyz',[.1,.3,.8]).as_matrix()
            generator=np.array([eef@T_EEF_GENERATOR,eef@T_EEF_GENERATOR]);scores=np.array([.1,.9])
            np.save(root/'graspgenx_raw/grasps_generator_world.npy',generator);np.save(root/'graspgenx_raw/scores.npy',scores)
            payload={'candidates':[{'pose':generator_to_contact(generator[i]),'score':scores[i],
                'metadata':{'source_generator_index':i}} for i in [1,0]]}
            _,_,order=load_raw(root,payload);self.assertEqual(order.tolist(),[1,0])
            payload['candidates'][0]['pose']=generator[1]
            with self.assertRaises(ValueError):load_raw(root,payload)

    def test_pad_refinement_keeps_good_depth_and_limits_width(self):
        rng=np.random.default_rng(3);points=rng.uniform([-.012,-.015,.034],[.001,.02,.04],(100,3))
        eef,width,advance,counts,bad,_=pad_refinement(np.eye(4)[None],points,self.gripper.cavity)
        self.assertFalse(bad[0]);self.assertLess(width[0],.088);self.assertLessEqual(abs(advance[0]),.005)
        np.testing.assert_allclose(eef[0,:3,:3],np.eye(3))

    def test_variable_width_collisions_match_authored_geometry(self):
        rng=np.random.default_rng(2);points=rng.uniform(-.15,.15,(4000,3))
        for width in [.088,.060,.020]:
            exact=GripperGeometry(opening=width/2)
            actual=variable_collision(self.gripper,np.eye(4)[None],np.array([width]),points)
            expected=exact.point_rejections_reference(np.eye(4)[None],points)[0]
            np.testing.assert_array_equal(actual,expected)

    def test_table_check_matches_original_policy_at_each_width(self):
        rng=np.random.default_rng(6);poses=np.tile(np.eye(4),(8,1,1));poses[:,:3,3]=rng.uniform([-.1,-.1,0],[.1,.1,.1],(8,3))
        table={'center':[0,0,-.1],'dims':[1,1,.2]}
        for width in [.088,.06]:
            actual=table_bad(self.gripper,poses,np.full(8,width),table)
            expected=GripperGeometry(opening=width/2).table_rejections(poses,table)[0]
            np.testing.assert_array_equal(actual,expected)

    def test_packaging_preserves_source_but_exposes_modification(self):
        pose=np.eye(4);old={'pose':pose.tolist(),'width':.088,'score':1.,'metadata':{'source_generator_index':9}}
        original=deepcopy(old);new=pose.copy();new[2,3]=.002
        result={'selected':[0],'poses':new[None],'widths':np.array([.04]),'axial_shift_m':np.array([.002])}
        out=make_pool({'candidates':[old]},result,np.array([9]),'arx_observed')['candidates'][0]
        self.assertEqual(old,original);self.assertEqual(out['metadata']['source_generator_index'],9)
        self.assertTrue(out['metadata']['pose_refinement_applied']);self.assertTrue(out['metadata']['width_refinement_applied'])

    def test_single_variant_avoids_unused_gripper_checks_and_matches_full_detector(self):
        from .processor import Processor
        rng=np.random.default_rng(52);points=rng.uniform(-.08,.08,(8000,3));poses=np.eye(4)[None]
        # No gripper assets are needed when only the native variant is requested.
        values=[Processor({},use_spatial_crop=enabled).process(poses,np.ones(1),points,points[:100],None,
            variants=('native_rechecked',)) for enabled in [False,True]]
        self.assertEqual(list(values[0]),['native_rechecked'])
        for key in ['poses','widths','bad','empty','cavity_counts','axial_shift_m']:
            np.testing.assert_array_equal(values[0]['native_rechecked'][key],values[1]['native_rechecked'][key])
        self.assertEqual(values[0]['native_rechecked']['selected'],values[1]['native_rechecked']['selected'])

    def test_zero_raw_candidates_are_a_valid_empty_pool(self):
        from .processor import Processor
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'graspgenx_raw').mkdir()
            np.save(root/'graspgenx_raw/grasps_generator_world.npy',np.empty((0,4,4)))
            np.save(root/'graspgenx_raw/scores.npy',np.empty(0))
            poses,scores,order=load_raw(root,{'candidates':[]})
            self.assertEqual(poses.shape,(0,4,4))
            result=Processor({}).process(poses,scores,np.empty((0,3)),np.empty((0,3)),None,
                variants=('native_rechecked',))['native_rechecked']
            self.assertEqual(result['selected'],[])


if __name__=='__main__':unittest.main()
