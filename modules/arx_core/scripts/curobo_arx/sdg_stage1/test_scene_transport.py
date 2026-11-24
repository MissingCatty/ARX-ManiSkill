"""Offline ARX topology/pose equivalence, not physical execution evidence."""
import hashlib
from pathlib import Path
import unittest
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT,read,sha
from .scene_transport import (triangle_faces,ExportActorPoses,actor_relative_frame,
                             geometry_pose,original_vertices)


def original_faces(counts,indices):
    faces=[];offset=0
    for n in counts:
        ids=indices[offset:offset+n];offset+=n
        faces.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,n-1)])
    return np.asarray(faces,dtype=np.int64).reshape(-1,3)


class SceneTransportTests(unittest.TestCase):
    def test_actual_ARX_wheel_and_target_triangle_order_is_byte_identical(self):
        reg=read(ROOT/'runs/graspness_teacher_validation_20261102_04/registration.json')
        sim=reg['render_configs']['RaytracedLighting']['simulation']
        files=[(sim['robot'],'/wheel1/collisions/wheel1')]
        files += [(a['usd_path'],None) for a in reg['catalog']['objects'] if a.get('collection_enabled',True)]
        for path,suffix in files:
            stage=Usd.Stage.Open(path)
            prim=next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and
                (suffix is None or str(p.GetPath()).endswith(suffix)))
            mesh=UsdGeom.Mesh(prim);counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get()
            before=original_faces(counts,indices);after=triangle_faces(counts,indices)
            self.assertEqual(before.shape,after.shape)
            self.assertEqual(before.tobytes(),after.tobytes())

    def test_polygon_fan_and_empty_topology_keep_original_order(self):
        for counts,indices in (([4,3,5],list(range(12))),([],[]),([2,3],[0,1,2,3,4])):
            np.testing.assert_array_equal(triangle_faces(counts,indices),original_faces(counts,indices))
        for counts,indices in (([3],[0,1]),([-1],[]),([[3]],[0,1,2])):
            with self.assertRaises(ValueError):triangle_faces(counts,indices)

    def test_ARX_pose_readback_is_shared_only_within_one_export(self):
        ep=ROOT/'runs/graspness_teacher_validation_20261102_04/groups/group_5455b3b5f6ad/episodes/1573f0d69c3e4dfcb1941e30141913e6'
        fixture=ROOT/'runs/geometry_cache_fix_20261104_01/fixtures'
        before=fixture/'arx_pose_pregrasp_execution.json'
        after=fixture/'arx_pose_lift_execution.json'
        if not before.exists():before,after=ep/'pregrasp_execution.json',ep/'lift_execution.json'
        start=np.asarray(read(before)['final']['objects']['test_object']['world'])
        lifted=np.asarray(read(after)['final']['objects']['test_object']['world'])
        self.assertGreater(lifted[2,3]-start[2,3],.06)
        current=[start];calls=[]
        def read_pose(path):calls.append(path);return current[0]
        first=ExportActorPoses(read_pose)
        np.testing.assert_array_equal(first.get('/World/Task/assets/asset_mustard_bottle'),start)
        np.testing.assert_array_equal(first.get('/World/Task/assets/asset_mustard_bottle'),start)
        self.assertEqual((first.readbacks,first.reuses,len(calls)),(1,1,1))
        current[0]=lifted;second=ExportActorPoses(read_pose)
        np.testing.assert_array_equal(second.get('/World/Task/assets/asset_mustard_bottle'),lifted)
        self.assertEqual((second.readbacks,second.reuses,len(calls)),(1,0,2))

    def test_readback_fault_is_propagated_without_cached_substitute(self):
        def broken(path):raise RuntimeError('ARX pose unavailable')
        cache=ExportActorPoses(broken)
        with self.assertRaisesRegex(RuntimeError,'pose unavailable'):cache.get('/World/Robot')
        self.assertEqual(cache.values,{})


FIXTURES=ROOT/'runs/geometry_cache_fix_20261104_01/fixtures'


@unittest.skipUnless((FIXTURES/'index.json').exists(),'Recorded ARX mesh-cache regression fixtures unavailable')
class StableAuthoredGeometryTests(unittest.TestCase):
    """Actual ARX USD and recorded mesh exports; no simulator or USD file edits."""
    @classmethod
    def setUpClass(cls):
        cls.records=read(FIXTURES/'index.json')
        reg=read(FIXTURES/'source_registration.json')
        cls.robot=reg['render_configs']['RaytracedLighting']['simulation']['robot']
        cls.source_sha=sha(cls.robot)
        cls.stage=Usd.Stage.Open(cls.robot)
        cls.stage.SetEditTarget(cls.stage.GetSessionLayer())
        cls.prim=next(p for p in cls.stage.Traverse() if p.IsA(UsdGeom.Mesh) and
                     str(p.GetPath()).endswith('/wheel1/collisions/wheel1'))
        cls.actor=cls.prim
        while cls.actor and not cls.actor.HasAPI(UsdPhysics.RigidBodyAPI):cls.actor=cls.actor.GetParent()
        mesh=UsdGeom.Mesh(cls.prim);cls.points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        cls.faces=triangle_faces(mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get())

    @classmethod
    def tearDownClass(cls):
        cls.stage.GetSessionLayer().Clear()
        if sha(cls.robot)!=cls.source_sha:raise AssertionError('The actual robot USD changed')

    def setUp(self):self.stage.GetSessionLayer().Clear()

    def test_same_ARX_wheel_has_one_cache_key_across_recorded_motion_poses(self):
        keys=[];maximum=0.
        for record in self.records:
            base=np.asarray(read(record['scene_file'])['base_world'])
            # Recorded motion is authored ONLY to an anonymous in-memory session;
            # no robot configuration asset or live simulator is changed.
            op=UsdGeom.Xformable(self.actor).MakeMatrixXform()
            op.Set(Gf.Matrix4d(*base.T.reshape(-1).tolist()))
            cache=UsdGeom.XformCache()
            matrix=np.asarray(cache.GetLocalToWorldTransform(self.prim)).T
            authored=np.asarray(cache.GetLocalToWorldTransform(self.actor)).T
            relative=actor_relative_frame(cache,self.prim,self.actor)
            self.assertIsNotNone(relative)
            local,pose,_,error=geometry_pose(self.points,matrix,authored,authored,np.linalg.inv(base),
                                            relative_frame=relative)
            maximum=max(maximum,error)
            keys.append(hashlib.sha256(local.tobytes()+self.faces.tobytes()).hexdigest())
        self.assertEqual(len(set(keys)),1)
        self.assertLessEqual(maximum,1e-6)

    def test_all_recorded_wheel_vertices_and_triangle_order_are_preserved(self):
        import trimesh
        cache=UsdGeom.XformCache()
        matrix=np.asarray(cache.GetLocalToWorldTransform(self.prim)).T
        authored=np.asarray(cache.GetLocalToWorldTransform(self.actor)).T
        relative=actor_relative_frame(cache,self.prim,self.actor)
        local=(np.c_[self.points,np.ones(len(self.points))]@relative.T)[:,:3]
        self.assertEqual(len(local),933510)
        for record in self.records:
            mesh=trimesh.load_mesh(record['fixture_geometry'],process=False,maintain_order=True)
            observed=np.asarray(mesh.vertices)
            self.assertEqual(observed.shape,local.shape)
            self.assertEqual(np.asarray(mesh.faces,dtype=np.int64).tobytes(),self.faces.tobytes())
            self.assertLessEqual(float(np.max(np.abs(local-observed))),1e-8)
            scene=read(record['scene_file']);base=np.asarray(scene['base_world']);pose=record['mesh']['pose']
            transform=np.eye(4);transform[:3,3]=pose[:3]
            transform[:3,:3]=Rotation.from_quat([pose[4],pose[5],pose[6],pose[3]]).as_matrix()
            live=base@transform
            direct=original_vertices(self.points,matrix,authored,live,np.linalg.inv(base))
            _,encoded,_,error=geometry_pose(self.points,matrix,authored,live,np.linalg.inv(base),
                                           relative_frame=relative)
            output=local@Rotation.from_quat([encoded[4],encoded[5],encoded[6],encoded[3]]).as_matrix().T+encoded[:3]
            recorded=observed@transform[:3,:3].T+transform[:3,3]
            self.assertLessEqual(error,1e-6)
            self.assertLessEqual(float(np.max(np.abs(output-direct))),1e-6)
            self.assertLessEqual(float(np.max(np.abs(output-recorded))),1e-6)

    def test_authored_child_transform_changes_cannot_reuse_old_geometry(self):
        before=actor_relative_frame(UsdGeom.XformCache(),self.prim,self.actor)
        op=UsdGeom.Xformable(self.prim).AddTranslateOp(opSuffix='offline_cache_fault')
        op.Set(Gf.Vec3d(.001,0,0))
        after=actor_relative_frame(UsdGeom.XformCache(),self.prim,self.actor)
        self.assertFalse(np.array_equal(before,after))
        self.assertGreater(float(np.max(np.abs(after-before))),.0009)

    def test_reset_stack_retains_original_world_matrix_fallback(self):
        UsdGeom.Xformable(self.prim).SetResetXformStack(True)
        self.assertIsNone(actor_relative_frame(UsdGeom.XformCache(),self.prim,self.actor))

    def test_small_real_geometry_change_is_not_rounded_out_of_cache_identity(self):
        relative=actor_relative_frame(UsdGeom.XformCache(),self.prim,self.actor)
        local=(np.c_[self.points,np.ones(len(self.points))]@relative.T)[:,:3]
        altered=local.copy();altered[0,0]+=1e-10
        self.assertNotEqual(hashlib.sha256(local.tobytes()+self.faces.tobytes()).hexdigest(),
                            hashlib.sha256(altered.tobytes()+self.faces.tobytes()).hexdigest())


if __name__=='__main__':unittest.main()
