"""Offline ARX interface checks; neural inference and physical execution separate."""
import copy
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT,read
from scripts.curobo_arx.data_factory.placements import placement_bank
from scripts.curobo_arx.anygrasp_teacher.backend import cloud_input
from .backend import sample_cloud,candidates


class GraspnessInterfaceTests(unittest.TestCase):
    def fixture(self):
        row={'intrinsics':[[100,0,5],[0,100,5],[0,0,1]],'camera_world_optical':np.eye(4).tolist(),
            'snapshot':{'objects':{'test_object':{'world':np.eye(4).tolist()}}}}
        row['snapshot']['objects']['test_object']['world'][2][3]=.5
        asset={'object_id':'mustard_bottle','size_m':[.1,.1,.1]}
        cfg=read(ROOT/'configs/lilo_vla/graspness_teacher.json');cfg['checkpoint_sha256']='offline_fixture_not_a_checkpoint'
        return row,asset,cfg

    def test_native_linear_depth_is_used_in_metres_without_mm_rescaling(self):
        row,asset,_=self.fixture();depth=np.ones((10,10))*.5
        points,mask=cloud_input(depth,row,asset)
        self.assertEqual(points.shape,(100,3));np.testing.assert_allclose(points[:,2],.5)
        self.assertEqual(mask.sum(),100)

    def test_invalid_or_absent_target_depth_is_rejected(self):
        row,asset,_=self.fixture()
        for depth in (np.zeros((10,10)),np.full((10,10),np.nan),np.ones((10,10))*1000):
            with self.assertRaisesRegex(ValueError,'Insufficient'):cloud_input(depth,row,asset)

    def test_point_sampling_is_deterministic_and_does_not_invent_coordinates(self):
        points=np.arange(90,dtype=np.float32).reshape(30,3)
        for count in (10,50):
            sampled=sample_cloud(points,count,42)
            np.testing.assert_array_equal(sampled,sample_cloud(points,count,42))
            self.assertEqual(sampled.shape,(count,3))
            self.assertTrue(all(any(np.array_equal(p,q) for q in points) for p in sampled))
        with self.assertRaises(ValueError):sample_cloud(np.empty((0,3)),10,1)

    def test_pose_mapping_pregrasp_retreat_and_source_identity(self):
        row,asset,cfg=self.fixture();prediction=np.zeros((1,17))
        prediction[0,:4]=[.9,.04,.02,.02]
        prediction[0,4:13]=np.eye(3).reshape(-1);prediction[0,13:16]=[0,0,.5]
        pool=candidates(prediction,row,asset,cfg)
        self.assertEqual(len(pool['candidates']),1)
        goal=pool['candidates'][0];g=np.asarray(goal['grasp_world']);h=np.asarray(goal['pregrasp_world'])
        np.testing.assert_allclose(g[:3,2],[1,0,0])
        np.testing.assert_allclose(g[:3,1],[0,1,0])
        np.testing.assert_allclose(h[:3,3],g[:3,3]-.1*g[:3,2])
        np.testing.assert_allclose(h[:3,:3],g[:3,:3])
        self.assertEqual(goal['candidate_source']['name'],'Graspness / GSNet')
        self.assertNotIn('AnyGrasp',str(goal['candidate_source']))

    def test_bounded_candidates_keep_high_scores_and_frozen_aperture(self):
        row,asset,cfg=self.fixture();cfg['candidate_limit']=2
        rows=np.zeros((12,17));rows[:,:4]=[.8,.04,.02,.02]
        rows[:,4:13]=np.eye(3).reshape(-1);rows[:,15]=.5
        rows[:,0]=np.linspace(.1,.9,len(rows));rows[:,14]=np.linspace(-.04,.04,len(rows))
        rows[-1,1]=.2
        pool=candidates(rows,row,asset,cfg)
        self.assertLessEqual(len(pool['candidates']),2)
        self.assertTrue(all(g['width_m']<=.088 for g in pool['candidates']))
        self.assertEqual([g['score'] for g in pool['candidates']],sorted([g['score'] for g in pool['candidates']],reverse=True))
        with self.assertRaises(ValueError):candidates(np.zeros((3,16)),row,asset,cfg)

    def test_camera_prefilter_retains_a_parallel_jaw_flip_that_avoids_clipping(self):
        row,asset,cfg=self.fixture();prediction=np.zeros((1,17));prediction[0,:4]=[.9,.04,.02,.02]
        prediction[0,4:13]=np.eye(3).reshape(-1);prediction[0,15]=.5
        class Geometry:
            calls=0
            def inspect(self,*args):
                self.calls+=1
                return {'pass':self.calls==2,'reasons':[] if self.calls==2 else ['near_clip']}
        context={'eef_from_camera':np.eye(4),'intrinsics':row['intrinsics'],'shape':[10,10],'clipping_range_m':[.05,10]}
        geometry=Geometry();pool=candidates(prediction,row,asset,cfg,context,geometry)
        self.assertEqual(len(pool['candidates']),1)
        self.assertEqual(pool['candidates'][0]['camera_safe_equivalent_flips'],[True])
        self.assertEqual(geometry.calls,2)

    def test_reference_pose_first_does_not_discard_remaining_coverage(self):
        profile=read(ROOT/'configs/lilo_vla/graspness_placement_diversity.json')
        asset={'object_id':'mustard_bottle','placement':{'position':[4.,2.,.75],'orientation':[1,0,0,0]}}
        support={'polygon_xy':[[3.5,1.5],[4.5,1.5],[4.5,2.5],[3.5,2.5]]}
        vertices=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.05,.05)])
        original=copy.deepcopy(asset);bank=placement_bank(asset,vertices,profile,support,42)
        self.assertEqual(len(bank['placements']),72)
        self.assertEqual(bank['placements'][0]['placement'],asset['placement'])
        self.assertEqual(asset,original)


if __name__=='__main__':unittest.main()
