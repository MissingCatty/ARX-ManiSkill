"""Offline ARX readback equivalence and failure gates; no simulator starts."""
from pathlib import Path
from types import SimpleNamespace
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .common import read
from .fast_fingerprint import object_records,capture_fast


def pose(position,quaternion):
    from scipy.spatial.transform import Rotation
    matrix=np.eye(4)
    matrix[:3,:3]=Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix()
    matrix[:3,3]=position
    return matrix


class ObjectReadbackTests(unittest.TestCase):
    def setUp(self):
        self.paths=['/World/Envs/env_0/Task/assets/asset_milk','/World/Envs/env_0/Task/assets/asset_mustard']
        self.objects={'asset:milk':SimpleNamespace(paths=[self.paths[0]]),
            'asset:mustard':SimpleNamespace(paths=[self.paths[1]]),
            'test_object':SimpleNamespace(paths=[self.paths[1]])}
        self.mass=np.asarray([[.061],[.312]],dtype=np.float32)
        self.inertia=np.asarray([np.diag([.001,.002,.003]).reshape(-1),np.diag([.004,.005,.006]).reshape(-1)],dtype=np.float32)
        self.position=np.asarray([[4.1,1.8,.73],[4.2,1.9,.74]],dtype=np.float32)
        self.orientation=np.asarray([[1.,0,0,0],[.70710677,0,0,.70710677]],dtype=np.float32)

    def call(self,**changes):
        args=dict(objects=self.objects,paths=self.paths,masses=self.mass,inertias=self.inertia,
            positions=self.position,orientations=self.orientation,pose_matrix=pose)
        args.update(changes);return object_records(**args)

    def test_all_scalar_values_shapes_order_and_test_object_alias_are_preserved(self):
        poses,runtime=self.call()
        self.assertEqual(len(poses),3);self.assertEqual(len(runtime),2)
        for name,obj in self.objects.items():
            i=self.paths.index(obj.paths[0])
            self.assertEqual(poses[name],{'path':obj.paths[0],'world':pose(self.position[i],self.orientation[i]).tolist()})
            self.assertEqual(runtime[obj.paths[0]],{'get_masses':self.mass[i:i+1].tolist(),'get_inertias':self.inertia[i:i+1].tolist()})

    def test_batch_order_is_mapped_by_path_rather_than_object_insertion_order(self):
        actual=self.call(paths=self.paths[::-1],masses=self.mass[::-1],inertias=self.inertia[::-1],positions=self.position[::-1],orientations=self.orientation[::-1])
        self.assertEqual(actual,self.call())

    def test_every_capture_reads_new_values_not_cached_runtime_masses_or_poses(self):
        before=self.call();self.mass[1,0]+=.1;self.position[1,2]+=.2
        after=self.call();self.assertNotEqual(before,after)
        self.assertNotEqual(before[1][self.paths[1]],after[1][self.paths[1]])

    def test_missing_or_duplicate_body_paths_stop_the_gate(self):
        for paths in [self.paths[:1],self.paths*2]:
            with self.assertRaises(ValueError):self.call(paths=paths)

    def test_bad_tensor_shapes_are_not_squeezed_into_false_equivalence(self):
        for changes in [{'masses':self.mass[:,0]},{'inertias':self.inertia.reshape(2,3,3)},
                        {'positions':self.position[:1]},{'orientations':self.orientation[:1]}]:
            with self.assertRaises(ValueError):self.call(**changes)


class CaptureEquivalenceTests(unittest.TestCase):
    def run_capture(self,changed=False,changed_pose=False):
        with tempfile.TemporaryDirectory() as tmp:
            bridge=SimpleNamespace(root=Path(tmp))
            original={'invariant':{'runtime_objects':{'arx_object':{'mass':.1}}},
                'invariant_sha256':'arx_original','scene_state':{'objects':{'arx_object':[4.,1.,.7]}}}
            fast=copy.deepcopy(original)
            if changed:fast['invariant']['runtime_objects']['arx_object']['mass']=.2
            if changed_pose:fast['scene_state']['objects']['arx_object'][2]=.8
            calls=[]
            def capture(*args,**kwargs):
                calls.append(kwargs)
                return fast if 'object_reader' in kwargs else original
            with patch('scripts.curobo_arx.data_factory.fast_fingerprint.BatchedObjectReader',return_value=object()):
                if changed or changed_pose:
                    with self.assertRaisesRegex(RuntimeError,'Frozen physical configuration mismatch'):
                        capture_fast(bridge,capture,{'verify_scalar_equivalence':True})
                    receipt=read(bridge.root/'batched_fingerprint_equivalence.json')
                    self.assertFalse(receipt['invariants_exactly_equal'] and receipt['scene_state_exactly_equal'])
                else:
                    a,first=capture_fast(bridge,capture,{'verify_scalar_equivalence':True})
                    b,second=capture_fast(bridge,capture,{'verify_scalar_equivalence':True})
                    self.assertEqual(a,original);self.assertEqual(b,original)
                    self.assertTrue(first['initial_equivalence_checked']);self.assertFalse(second['initial_equivalence_checked'])
                    self.assertEqual(len(calls),3)

    def test_scalar_oracle_once_then_fresh_batched_readbacks_each_block(self):self.run_capture()
    def test_mass_difference_stops_before_motion(self):self.run_capture(changed=True)
    def test_pose_difference_stops_before_motion(self):self.run_capture(changed_pose=True)


if __name__=='__main__':unittest.main()
