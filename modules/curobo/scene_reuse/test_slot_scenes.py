"""Native CPU scene storage exercises the actual fixed-slot update operations."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import trimesh
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

from arx_project.layout import ROOT, sha256
from .slot_scenes import SlotSceneUpdates


class SlotTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'tmp/curobo/tests'; parent.mkdir(parents=True,exist_ok=True)
        self.work=tempfile.TemporaryDirectory(dir=parent,prefix='slot-test-'); self.root=Path(self.work.name)
        self.files=[]
        for index in range(9):
            path=self.root/(str(index)+'.ply'); trimesh.creation.box(extents=[.01*(index+1),.02,.03]).export(path); self.files.append(path)
        self.device=DeviceCfg(device='cpu')
        self.checker=SceneCollision.from_config(SceneCollisionCfg(device_cfg=self.device,cache={'mesh':8,'cuboid':8}))
        self.loads=0
        def load(scene): self.loads+=1; self.checker.load_collision_model(scene)
        self.owner=SimpleNamespace(scene_model=self.checker,update_world=load,clear_scene_cache=self.checker.clear_cache)
        self.policy=SlotSceneUpdates(SceneCfg.create,lambda p:Pose.from_list(p,self.device),sha256)
        self.scene={'mesh':{('parked_robot_arm' if i==7 else 'obj'+str(i)):{'file_path':str(self.files[i]),'pose':[0,0,0,1,0,0,0]} for i in range(8)},
                    'cuboid':{'table':{'dims':[1.,1.,.1],'pose':[0,0,-1,1,0,0,0]}}}

    def tearDown(self): self.work.cleanup()

    def test_switch_parked_arm_at_capacity_changes_one_slot(self):
        self.policy.update(self.owner,self.scene); changed=deepcopy(self.scene)
        changed['mesh']['parked_robot_arm']['file_path']=str(self.files[8])
        before=self.checker.data.meshes.mesh_ids.clone()
        self.policy.update(self.owner,changed); self.policy.update(self.owner,self.scene)
        self.assertEqual(self.loads,1); self.assertEqual(int(self.checker.data.meshes.count[0]),8)
        self.assertTrue(torch.equal(before,self.checker.data.meshes.mesh_ids))
        self.assertEqual(self.policy.report()['slot_replacements'],2)

    def test_pose_and_enable_changes_preserve_geometry(self):
        self.policy.update(self.owner,self.scene); handles=self.checker.data.meshes.mesh_ids.clone()
        changed=deepcopy(self.scene); del changed['mesh']['obj0']; changed['mesh']['obj1']['pose'][0]=.2
        self.policy.update(self.owner,changed); self.assertFalse(bool(self.checker.data.meshes.enable[0,0]))
        self.policy.update(self.owner,self.scene)
        self.assertTrue(torch.equal(handles,self.checker.data.meshes.mesh_ids)); self.assertEqual(self.loads,1)

    def test_new_geometry_bounds_are_updated(self):
        self.policy.update(self.owner,self.scene); changed=deepcopy(self.scene)
        changed['mesh']['parked_robot_arm']['file_path']=str(self.files[8]); self.policy.update(self.owner,changed)
        data=self.checker.data.meshes; index=next(i for i,n in enumerate(data.names[0]) if n.startswith('parked_robot_arm_'))
        self.assertTrue(torch.isfinite(data.bounds_lower[0,index]).all())
        self.assertAlmostEqual(float(data.dims[0,index,0]),.09,places=6)

    def test_cube_dimensions_update_in_place(self):
        self.policy.update(self.owner,self.scene); changed=deepcopy(self.scene); changed['cuboid']['table']['dims'][0]=2.
        self.policy.update(self.owner,changed); self.assertEqual(float(self.checker.data.cuboids.dims[0,0,0]),2.)
        self.assertEqual(self.loads,1)

    def test_boundary_reset_reuses_unmodified_scene(self):
        self.policy.update(self.owner,self.scene); self.policy.report(); self.policy.reset()
        self.policy.update(self.owner,self.scene); self.assertEqual(self.loads,1)

    def test_external_clear_invalidates_boundary_cache(self):
        self.policy.update(self.owner,self.scene); self.policy.report(); self.checker.clear_cache(); self.policy.reset()
        self.policy.update(self.owner,self.scene); self.assertEqual(self.loads,2)
        self.assertEqual(self.policy.report()['request_boundary_invalidations'],1)

    def test_external_pose_change_invalidates_boundary_cache(self):
        self.policy.update(self.owner,self.scene); self.policy.report()
        name=next(n for n in self.checker.data.meshes.names[0] if n and n.startswith('obj0_'))
        self.checker.update_obstacle_pose(name,Pose.from_list([1,0,0,1,0,0,0],self.device)); self.policy.reset()
        self.policy.update(self.owner,self.scene); self.assertEqual(self.loads,2)

    def test_original_capacity_still_rejects_nine_active_meshes(self):
        self.policy.update(self.owner,self.scene); changed=deepcopy(self.scene)
        changed['mesh']['extra']={'file_path':str(self.files[8]),'pose':[0,0,0,1,0,0,0]}
        with self.assertRaisesRegex(ValueError,'capacity'): self.policy.update(self.owner,changed)
        self.assertEqual(self.policy.records,{})

    def test_duplicate_normalized_role_is_rejected(self):
        changed=deepcopy(self.scene)
        changed['mesh']['obj0_'+sha256(self.files[0])[:20]]=deepcopy(changed['mesh']['obj0'])
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.policy.update(self.owner,changed)

    def test_wrong_qualified_content_is_rejected(self):
        changed=deepcopy(self.scene); value=changed['mesh'].pop('obj0')
        changed['mesh']['obj0_'+'0'*20]=value
        with self.assertRaisesRegex(ValueError,'digest'): self.policy.update(self.owner,changed)


if __name__=='__main__':unittest.main()
