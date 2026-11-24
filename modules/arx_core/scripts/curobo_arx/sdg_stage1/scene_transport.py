"""Share immutable collider meshes; refresh every actual live pose per request.

Exactly the original export inclusion rule, fan triangulation, primitive boxes,
actor readback and 1.4-m bound. Verify base-frame vertex equivalence on each
export. This only changes serialization, not collision geometry or constraints.
"""
import hashlib
from pathlib import Path
import time
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import sha
from scripts.curobo_arx.data_factory.vector_env import ScopedStage


def original_vertices(points,matrix,authored_actor=None,live_actor=None,world_to_base=None):
    """The original ValidationBridge formula, including multiplication order."""
    matrix=np.asarray(matrix,dtype=float)
    if authored_actor is not None:
        matrix=np.asarray(live_actor)@np.linalg.inv(authored_actor)@matrix
    direct=(np.c_[points,np.ones(len(points))]@(np.asarray(world_to_base)@matrix).T)[:,:3]
    if not np.isfinite(direct).all():raise RuntimeError('Nonfinite original collider vertices')
    return direct


def within_original_bound(vertices):
    nearest=np.maximum(vertices.min(0),0)+np.minimum(vertices.max(0),0)
    return bool(np.linalg.norm(nearest)<=1.4)


def triangle_faces(counts,indices):
    """Original polygon-fan order, avoiding Python/VtArray work for triangles."""
    counts=np.asarray(counts,dtype=np.int64);indices=np.asarray(indices,dtype=np.int64)
    if counts.ndim!=1 or indices.ndim!=1 or (counts<0).any() or counts.sum()!=len(indices):
        raise ValueError('Invalid composed collider face topology')
    if (counts==3).all():return indices.reshape(-1,3)
    faces=[];offset=0
    for n in counts:
        ids=indices[offset:offset+n];offset+=n
        faces.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,n-1)])
    return np.asarray(faces,dtype=np.int64).reshape(-1,3)


class ExportActorPoses:
    """One synchronous export only; never retain a pose across physics steps."""
    def __init__(self,read_pose):
        self.read_pose=read_pose;self.values={};self.readbacks=0;self.reuses=0

    def get(self,path):
        if path not in self.values:
            self.values[path]=self.read_pose(path);self.readbacks+=1
        else:self.reuses+=1
        return self.values[path]


def actor_relative_frame(cache,prim,actor):
    """Read authored child transforms without cancelling two world matrices.

    inv(actor_world) @ collider_world adds pose-dependent floating-point noise
    to an otherwise immutable mesh. USD's relative transform composes only the
    child transform chain. A reset stack retains the original representation.
    """
    relative,resets=cache.ComputeRelativeTransform(prim,actor)
    if resets:return None
    frame=np.asarray(relative,dtype=float).T
    if not np.isfinite(frame).all():raise RuntimeError('Nonfinite authored actor-relative transform')
    return frame


def geometry_pose(points,matrix,authored_actor=None,live_actor=None,world_to_base=None,origin=None,
                  direct=None,evidence=None,relative_frame=None):
    """Return stable-frame geometry, rigid base pose, and original base vertices."""
    matrix=np.asarray(matrix,dtype=float);points=np.asarray(points,dtype=float)
    inverse=np.asarray(world_to_base,dtype=float)
    if authored_actor is None:
        shift=np.eye(4);shift[:3,3]=np.zeros(3) if origin is None else origin
        frame=np.linalg.inv(shift)@matrix;pose=inverse@shift
    else:
        frame=(np.linalg.inv(authored_actor)@matrix if relative_frame is None
               else np.asarray(relative_frame,dtype=float))
        pose=inverse@live_actor
    local=(np.c_[points,np.ones(len(points))]@frame.T)[:,:3]
    vertices=(np.c_[local,np.ones(len(local))]@pose.T)[:,:3]
    if not np.isfinite(local).all() or not np.isfinite(pose).all():
        raise RuntimeError('Nonfinite collider serialization transform')
    if direct is None:direct=original_vertices(points,matrix,authored_actor,live_actor,inverse)
    error=float(np.max(np.abs(vertices-direct)))
    if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6):
        raise RuntimeError('Live scene pose is not rigid')
    quat=Rotation.from_matrix(pose[:3,:3]).as_quat()
    encoded=local@Rotation.from_quat(quat).as_matrix().T+pose[:3,3]
    encoded_error=float(np.max(np.abs(encoded-direct)))
    if error>1e-8 or encoded_error>1e-6:
        # Lossless representation fallback, not a larger error tolerance: the
        # original base-frame vertices with identity pose need no rotation roundtrip.
        if evidence is not None:evidence.update(identity_pose_fallback=True,
            initial_decomposition_error_m=error,initial_quaternion_error_m=encoded_error)
        local=direct.copy();pose=np.eye(4);quat=np.array([0.,0.,0.,1.]);vertices=direct.copy()
        encoded=local@Rotation.from_quat(quat).as_matrix().T+pose[:3,3]
        error=encoded_error=float(np.max(np.abs(encoded-direct)))
    if encoded_error>1e-6:raise RuntimeError('Serialized rigid pose changes collider vertices by more than one micrometre')
    error=max(error,encoded_error)
    pose7=np.r_[pose[:3,3],quat[3],quat[:3]].tolist()
    return local,pose7,vertices,error


def export_scene(bridge):
    from isaacsim.core.experimental.prims import XformPrim
    from pxr import UsdGeom,UsdPhysics
    began=time.monotonic();context=bridge.context;stage=context.stage
    prior=bridge.raw_snapshot
    try:
        bridge.raw_snapshot=True
        base=np.asarray(bridge.snapshot()['link_world']['body4'])
    finally:bridge.raw_snapshot=prior
    inverse=np.linalg.inv(base);cache=UsdGeom.XformCache()
    actors=ExportActorPoses(lambda path:context.controller._world_pose(XformPrim(path)))
    folder=bridge.root.parents[1]/'scene_geometry';folder.mkdir(exist_ok=True)
    if not hasattr(bridge,'_scene_geometry_hashes'):bridge._scene_geometry_hashes={}
    meshes=[];maximum=0.;new_files=0;excluded=0;fallbacks=[];stable_frames=0;reset_fallbacks=[]
    for prim in ScopedStage(stage,bridge.environment_root).Traverse():
        path=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
        if '/right_link' in path or '/right_eef_link' in path:continue
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).T
        ancestor=prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):ancestor=ancestor.GetParent()
        authored=live=None
        if ancestor and ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            authored=np.asarray(cache.GetLocalToWorldTransform(ancestor)).T
            live=actors.get(str(ancestor.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        else:
            bounds=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_]).ComputeLocalBound(prim).ComputeAlignedRange()
            lo,hi=np.array(bounds.GetMin()),np.array(bounds.GetMax())
            points=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        if not len(points):continue
        # Exclusion is decided from the unchanged original formula, before a
        # serialization check that is relevant only to meshes actually exported.
        direct=original_vertices(points,matrix,authored,live,inverse)
        if not within_original_bound(direct):
            excluded+=1
            continue
        # The original bound uses only vertices. Do not expand topology for
        # objects it already excludes; no collider, vertex or pose is removed
        # beyond the unchanged original export rule.
        if prim.IsA(UsdGeom.Mesh):
            faces=triangle_faces(mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get())
        else:
            faces=np.asarray([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],dtype=np.int64)
        details={}
        relative_frame=None
        if authored is not None:
            relative_frame=actor_relative_frame(cache,prim,ancestor)
            if relative_frame is None:reset_fallbacks.append(bridge.roles.path(path))
            else:stable_frames+=1
        try:
            local,pose,vertices,error=geometry_pose(points,matrix,authored,live,inverse,bridge.roles.origin,
                direct=direct,evidence=details,relative_frame=relative_frame)
        except Exception:
            diagnostics=folder.parent/'scene_precision_diagnostics';diagnostics.mkdir(exist_ok=True)
            np.savez_compressed(diagnostics/('failed_'+str(time.time_ns())+'.npz'),path=path,
                points=points,matrix=matrix,has_actor=authored is not None,
                authored_actor=np.eye(4) if authored is None else authored,
                live_actor=np.eye(4) if live is None else live,world_to_base=inverse,
                origin=bridge.roles.origin,direct=direct)
            raise
        if details:
            details['path']=bridge.roles.path(path);fallbacks.append(details)
        maximum=max(maximum,error)
        faces=np.asarray(faces,dtype=np.int64)
        key=hashlib.sha256(local.tobytes()+faces.tobytes()).hexdigest()
        file=folder/(key+'.obj')
        if key not in bridge._scene_geometry_hashes:
            if not file.exists():
                trimesh.Trimesh(vertices=local,faces=faces,process=False).export(file,digits=17)
                new_files+=1
            bridge._scene_geometry_hashes[key]=sha(file)
        meshes.append({'path':bridge.roles.path(path),'raw_path':path,'geometry_file':str(file),
            'geometry_sha256':bridge._scene_geometry_hashes[key],'pose':pose,
            'vertex_count':len(local),'triangle_count':len(faces)})
    base[:3,3]-=bridge.roles.origin
    return {'format':'arx_geometry_files_live_pose_v1','base_world':base.tolist(),'meshes':meshes,
        'serialization_evidence':{'original_export_vertex_max_abs_difference_m':maximum,
            'live_poses_refreshed':True,'original_1_4m_bound_preserved':True,'new_mesh_files':new_files,
            'wall_seconds':time.monotonic()-began,'geometry_or_collision_parameters_modified':False,
            'verification_includes_serialized_quaternion':True,'original_bound_excluded_meshes':excluded,
            'serialization_optimization':'Defer excluded topology; vectorize original triangle order; actor poses shared within one synchronous export',
            'fresh_actor_readbacks':actors.readbacks,'same_export_actor_pose_reuses':actors.reuses,
            'actor_pose_cache_scope':'Recreated every export; physics does not step inside export_scene',
            'geometry_cache_identity':'authored_actor_relative_chain_v2',
            'stable_actor_relative_frames':stable_frames,
            'reset_stack_original_representation_fallbacks':reset_fallbacks,
            'lossless_identity_pose_fallbacks':fallbacks}}
