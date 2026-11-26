"""Check frame conventions, antipodal rotations and false map witnesses."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import write_json,sha256,read_json
from modules.curobo.indexed_planning.geometric.kinematics import ArmChain
from .lookup import FullPoseIndex,MAP


class LookupTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.chain=ArmChain();self.meta=deepcopy(read_json(MAP/'地图元数据.json'))
        self.q=self.chain.limits.mean(axis=1).astype(np.float32).astype(float)
        self.eef=self.chain.fk(self.q)[0];tcp=np.asarray(self.meta['T_eef_action'])
        action=self.eef@tcp
        rows=np.zeros((2,16),dtype=np.float32);rows[:,:6]=self.q
        rows[:,6:9]=action[:3,3];rows[:,9:13]=Rotation.from_matrix(action[:3,:3]).as_quat()[[3,0,1,2]]
        rows[1,9:13]*=-1;self.rows=rows;self.meta['witnesses']=len(rows)
        self.save();self.index=FullPoseIndex(self.root)

    def save(self):
        self.rows.tofile(self.root/self.meta['witness_file'])
        write_json(self.root/'地图元数据.json',self.meta)
        write_json(self.root/'完成.json',{'passed':True,'file_sha256':{
            n:sha256(self.root/n) for n in ['地图元数据.json',self.meta['witness_file']]}})

    def test_exact_self_pose_is_verified_without_modifying_target(self):
        before=self.eef.copy();result=self.index.direct(self.eef,self.q)
        self.assertEqual(len(result['witnesses']),2)
        self.assertLess(result['witnesses'][0]['independent_position_error_m'],1e-10)
        np.testing.assert_array_equal(self.eef,before)
        self.assertFalse(result['scene_checked'])

    def test_antipodal_quaternions_match_the_same_full_rotation(self):
        self.assertEqual(len(self.index.direct(self.eef,self.q)['witnesses']),2)

    def test_same_position_with_different_full_rotation_does_not_match(self):
        action=self.eef@self.index.tcp
        action[:3,:3]=action[:3,:3]@Rotation.from_euler('z',.2).as_matrix()
        other=action@np.linalg.inv(self.index.tcp)
        result=self.index.direct(other,self.q)
        self.assertEqual(result['position_neighbors'],2)
        self.assertFalse(result['witnesses'])

    def test_TCP_lever_does_not_drop_an_original_EEF_tolerance_match(self):
        other=self.eef.copy()
        other[:3,:3]=other[:3,:3]@Rotation.from_euler('x',.029).as_matrix()
        shift=(other[:3,:3]-self.eef[:3,:3])@self.index.tcp[:3,3]
        other[:3,3]+=shift/np.linalg.norm(shift)*.0019
        self.assertGreater(np.linalg.norm((other@self.index.tcp)[:3,3]-(self.eef@self.index.tcp)[:3,3]),.003)
        result=self.index.direct(other,self.q)
        self.assertTrue(result['witnesses'])
        self.assertAlmostEqual(result['witnesses'][0]['independent_position_error_m'],.0019)

    def test_stored_near_pose_cannot_bypass_independent_FK(self):
        self.rows[:,:6]=0.;self.save();self.index=FullPoseIndex(self.root)
        result=self.index.direct(self.eef,self.q)
        self.assertEqual(result['orientation_neighbors'],2)
        self.assertFalse(result['witnesses'])

    def test_map_miss_remains_unknown(self):
        other=self.eef.copy();other[:3,3]=[-10.,0.,0.]
        result=self.index.direct(other,self.q)
        self.assertFalse(result['witnesses']);self.assertFalse(result['map_miss_is_unreachable'])

    def test_changed_bank_checksum_is_a_fault(self):
        with (self.root/self.meta['witness_file']).open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'Map input changed'):FullPoseIndex(self.root)

    def test_changed_limits_are_a_fault(self):
        self.meta['joint_limits'][0][0]-=.1;self.save()
        with self.assertRaisesRegex(ValueError,'bounds differ'):FullPoseIndex(self.root)

    def test_wrong_TCP_is_a_fault(self):
        self.meta['T_eef_action'][2][3]=.063;self.save()
        with self.assertRaisesRegex(ValueError,'TCP differs'):FullPoseIndex(self.root)

    def test_zero_quaternion_is_a_fault(self):
        self.rows[:,9:13]=0.;self.save();self.index=FullPoseIndex(self.root)
        with self.assertRaisesRegex(ValueError,'Zero map quaternion'):self.index.direct(self.eef,self.q)

    def test_seed_count_current_state_and_row_provenance(self):
        for count in [1,8,32]:
            result=self.index.seeds(self.eef,self.q,count=count)
            self.assertEqual(len(result['seeds']),count);self.assertEqual(len(result['bank_rows']),count)
            np.testing.assert_allclose(result['seeds'][0],self.q,atol=1e-7)
            self.assertIsNone(result['bank_rows'][0]);self.assertFalse(result['target_pose_changed'])

    def test_invalid_pose_and_budget_are_rejected(self):
        for pose in [np.eye(3),np.full((4,4),np.nan),np.diag([-1.,1.,1.,1.])]:
            with self.assertRaises(ValueError):self.index.direct(pose,self.q)
        with self.assertRaises(ValueError):self.index.direct(self.eef,self.q,limit=0)
        with self.assertRaises(ValueError):self.index.seeds(self.eef,self.q,count=64)

    def test_wider_orientation_returns_seeds_without_relaxing_acceptance(self):
        action=self.eef@self.index.tcp
        action[:3,:3]=action[:3,:3]@Rotation.from_euler('z',4,degrees=True).as_matrix()
        other=action@np.linalg.inv(self.index.tcp);before=other.copy()
        self.assertFalse(self.index.direct(other,np.zeros(6))['witnesses'])
        result=self.index.neighborhood(other,np.zeros(6),angle=np.deg2rad(10))
        self.assertGreater(len(result['seeds']),1)
        self.assertEqual(result['final_rotation_tolerance_rad'],.03)
        np.testing.assert_array_equal(other,before)

    def test_angular_coverage_uses_full_rotation(self):
        action=self.eef@self.index.tcp
        action[:3,:3]=action[:3,:3]@Rotation.from_euler('z',4,degrees=True).as_matrix()
        result=self.index.neighborhood(action@np.linalg.inv(self.index.tcp),np.zeros(6))
        self.assertEqual(list(result['angular_counts'].values()),[0,0,2,2])

    def test_empty_neighborhood_preserves_only_original_current_seed(self):
        other=self.eef.copy();other[0,3]+=10
        result=self.index.neighborhood(other,self.q)
        self.assertEqual(result['bank_rows'],[None]);self.assertIsNone(result['best'])
        np.testing.assert_allclose(result['seeds'][0],self.q,atol=1e-7)

    def test_invalid_seed_neighborhood_is_rejected(self):
        for args in [{'radius':0},{'angle':1.},{'count':33}]:
            with self.assertRaises(ValueError):self.index.neighborhood(self.eef,self.q,**args)


if __name__=='__main__':unittest.main()
