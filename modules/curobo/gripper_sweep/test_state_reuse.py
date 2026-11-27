from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .width_refresh import state_signature,copy_lock_dependent_fields
from .sphere_cache import geometry_key,SphereMemo


class StateSignatureTests(unittest.TestCase):
    def state(self):
        return {'joint_names':['right_joint21','right_joint27','right_joint28'],
            'qpos':[.1,.044,.044],'joint_limits':[[-3,3],[0,.044],[0,.044]],
            'links':{'body4':np.eye(4).tolist()},
            'scene_cuboids':{'table':{'center':[0,0,.7],'dims':[1,1,.1]}},'scene_meshes':{}}

    def test_only_finger_width_and_sweep_declaration_can_change(self):
        a=self.state();b=deepcopy(a);b['qpos'][1:]=[.03,.02]
        b['planned_opening']={'target_per_finger_m':.03}
        self.assertEqual(state_signature(a,False),state_signature(b,False))

    def test_arm_base_table_limits_and_target_policy_invalidate(self):
        a=self.state();key=state_signature(a,False)
        for modify in [lambda b:b['qpos'].__setitem__(0,.2),
                       lambda b:b['links']['body4'][0].__setitem__(3,.1),
                       lambda b:b['scene_cuboids']['table']['center'].__setitem__(2,.71),
                       lambda b:b['joint_limits'][0].__setitem__(0,-2)]:
            b=deepcopy(a);modify(b);self.assertNotEqual(key,state_signature(b,False))
        self.assertNotEqual(key,state_signature(a,True))

    def test_stale_mesh_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh=Path(directory)/'mesh.obj';mesh.write_bytes(b'old mesh')
            state=self.state();state['scene_meshes']['target']={'file_path':str(mesh),
                'file_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'target':True}
            state_signature(state,False)
            mesh.write_bytes(b'changed mesh')
            with self.assertRaisesRegex(RuntimeError,'registered content'):state_signature(state,False)


class SphereMemoTests(unittest.TestCase):
    class Value:
        def __init__(self,n):self.n=n
        def numel(self):return self.n
        def element_size(self):return 4

    def test_exact_geometry_inputs_and_device_participate_in_key(self):
        cfg={'kinematics':{'lock_joints':{'right_joint27':.044},'sphere_radius':.01}}
        args=[cfg,np.zeros((1,1,6),dtype=np.float32),np.linspace(0,1,3),.03,{'right_joint27':.044},'cuda:0']
        key=geometry_key(*args)
        for index,replacement in [(1,np.ones((1,1,6),dtype=np.float32)),(2,np.linspace(0,1,4)),
                                  (3,.02),(4,{'right_joint27':.02}),(5,'cuda:1')]:
            changed=list(args);changed[index]=replacement
            self.assertNotEqual(key,geometry_key(*changed))
        changed=deepcopy(args);changed[0]['kinematics']['sphere_radius']=.02
        self.assertNotEqual(key,geometry_key(*changed))

    def test_hit_reuses_original_native_tensor(self):
        cache=SphereMemo();value=self.Value(10)
        self.assertEqual(cache.get('a',lambda:value),(value,False))
        def forbidden():raise AssertionError('FK repeated on a hit')
        self.assertEqual(cache.get('a',forbidden),(value,True))
        self.assertEqual(cache.report()['hits'],1)

    def test_memory_is_bounded_and_recently_used_entry_survives(self):
        cache=SphereMemo(max_bytes=80)
        for name in ['a','b']:cache.get(name,lambda:self.Value(10))
        cache.get('a',lambda:None);cache.get('c',lambda:self.Value(10))
        self.assertEqual(set(cache.entries),{'a','c'});self.assertLessEqual(cache.bytes,80)
        self.assertEqual(cache.evictions,1)

    def test_failed_build_cannot_create_a_cache_hit(self):
        cache=SphereMemo()
        def fail():raise RuntimeError('native FK failed')
        with self.assertRaises(RuntimeError):cache.get('a',fail)
        self.assertNotIn('a',cache.entries)


class DynamicCopyTests(unittest.TestCase):
    def test_static_fields_stay_untouched_while_all_dynamic_fields_update(self):
        from types import SimpleNamespace
        class Tensor:
            def __init__(self,value):self.value=value
            def copy_(self,other):self.value=other.value
        fields=['fixed_transforms','link_spheres','reference_link_spheres','lock_jointstate']
        target=SimpleNamespace(**{n:Tensor(0) for n in fields},static_map=Tensor(42))
        source=SimpleNamespace(**{n:Tensor(i+1) for i,n in enumerate(fields)},static_map=Tensor(99))
        copy_lock_dependent_fields(target,source)
        self.assertEqual([getattr(target,n).value for n in fields],[1,2,3,4])
        self.assertEqual(target.static_map.value,42)
        source.link_spheres=None
        with self.assertRaisesRegex(RuntimeError,'layout changed'):copy_lock_dependent_fields(target,source)


if __name__=='__main__':unittest.main()
