"""Read-only wrist/target geometry checks, independent of task success.

The camera, optics, robot collision model and target mesh are never modified.
Dark RGB pixels alone are not evidence of clipping (some objects are black).
"""
import ast
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from .geometry import rigid


POLICY = {'version':'wrist_target_geometry_v1', 'minimum_surface_distance_m':.010,
          'near_plane_guard_m':.002, 'path_step_m':.002, 'path_step_rad':float(np.deg2rad(2))}


def _clip(polygon, plane):
    result=[]
    for first,second in zip(polygon,np.roll(polygon,-1,axis=0)):
        a,b=np.dot(first,plane[:3])+plane[3],np.dot(second,plane[:3])+plane[3]
        if a>=0:result.append(first)
        if (a>=0)!=(b>=0):result.append(first+(second-first)*a/(a-b))
    return np.asarray(result).reshape(-1,3)


def near_intersection(vertices, faces, camera_from_object, intrinsics, shape, near_m):
    """Exact triangle/slab/frustum intersection, including crossing edges."""
    height,width=shape; k=np.asarray(intrinsics,dtype=float)
    if k.shape!=(3,3) or k[0,0]<=0 or k[1,1]<=0 or not np.isfinite(k).all():
        raise ValueError('Invalid recorded camera intrinsics')
    vertices=(np.c_[vertices,np.ones(len(vertices))]@camera_from_object.T)[:,:3]
    triangles=vertices[np.asarray(faces)]
    # Optical +Z; x/y planes follow the unchanged calibrated K and image bounds.
    planes=np.array([[0,0,1,-1e-8],[0,0,-1,near_m],
        [k[0,0],0,k[0,2],0],[-k[0,0],0,width-1-k[0,2],0],
        [0,k[1,1],k[1,2],0],[0,-k[1,1],height-1-k[1,2],0]],dtype=float)
    possible=np.ones(len(triangles),dtype=bool)
    for plane in planes:
        possible &= (triangles@plane[:3]+plane[3]).max(axis=1)>=0
    for index in np.flatnonzero(possible):
        polygon=triangles[index]
        for plane in planes:
            polygon=_clip(polygon,plane)
            if not len(polygon):break
        if len(polygon):return int(index)
    return None


class TargetGeometry:
    def __init__(self, mesh):
        # Merge duplicate UV-seam vertices for topology queries only. No point
        # positions, source USD, collider or render geometry are changed.
        self.mesh=mesh.copy()
        self.mesh.merge_vertices(digits_vertex=12)
        self.parts=self.mesh.split(only_watertight=False)

    @classmethod
    def from_asset(cls, asset):
        from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
        mesh,_=composed_mesh(asset['usd_path'])
        return cls(mesh)

    def inspect(self, object_world, camera_world, intrinsics, shape, clipping_range_m,
                policy=POLICY):
        obj,cam=rigid(object_world),rigid(camera_world)
        clip=np.asarray(clipping_range_m,dtype=float)
        if clip.shape!=(2,) or not np.isfinite(clip).all() or not 0<clip[0]<clip[1]:
            raise ValueError('Missing/invalid measured camera clipping range')
        origin=(np.linalg.inv(obj)@cam[:,3])[:3]
        lower=float(np.linalg.norm(np.maximum(np.maximum(self.mesh.bounds[0]-origin,origin-self.mesh.bounds[1]),0)))
        object_to_camera=np.linalg.inv(cam)@obj
        minimum_z=float((np.c_[self.mesh.vertices,np.ones(len(self.mesh.vertices))]@object_to_camera[2]).min())
        if lower>=policy['minimum_surface_distance_m'] and minimum_z>float(clip[0])+policy['near_plane_guard_m']:
            return {'pass':True,'reasons':[],'camera_surface_distance_m':None,
                'camera_surface_distance_lower_bound_m':lower,'minimum_target_optical_z_m':minimum_z,
                'camera_inside_target':False,'containment_uncertain':False,'near_slab_triangle':None,
                'clipping_range_m':clip.tolist(),'policy':policy,'scope':'Conservative whole-mesh near-plane and AABB clearance bounds'}
        distance=float(trimesh.proximity.closest_point(self.mesh,origin[None])[1][0])
        inside=False;uncertain=False
        for part in self.parts:
            if not ((origin>=part.bounds[0]).all() and (origin<=part.bounds[1]).all()):continue
            if part.is_watertight:
                inside |= bool(part.contains(origin[None])[0])
            else:uncertain=True
        triangle=near_intersection(self.mesh.vertices,self.mesh.faces,object_to_camera,
            intrinsics,shape,float(clip[0])+policy['near_plane_guard_m'])
        reasons=[]
        if inside:reasons.append('camera_inside_target')
        if uncertain:reasons.append('camera_inside_open_mesh_bounds_unverified')
        if distance<policy['minimum_surface_distance_m']:reasons.append('camera_surface_clearance')
        if triangle is not None:reasons.append('target_intersects_camera_near_slab')
        return {'pass':not reasons,'reasons':reasons,'camera_surface_distance_m':distance,
            'camera_inside_target':inside,'containment_uncertain':uncertain,
            'near_slab_triangle':triangle,'clipping_range_m':clip.tolist(), 'policy':policy,
            'scope':'Target visual mesh and calibrated optical camera only; original collision/contact gates remain required'}


def camera_path(eef_world, eef_from_camera, policy=POLICY):
    """Densify actual FK optical poses; do not infer a straight TCP trajectory."""
    mount=rigid(eef_from_camera)
    cameras=[rigid(p)@mount for p in eef_world]
    if not cameras:raise ValueError('Empty actual FK path')
    yield cameras[0]
    for start,end in zip(cameras,cameras[1:]):
        rotation=Rotation.from_matrix(np.stack([start[:3,:3],end[:3,:3]]))
        angle=(rotation[1]*rotation[0].inv()).magnitude()
        steps=max(1,int(np.ceil(np.linalg.norm(end[:3,3]-start[:3,3])/policy['path_step_m'])),
                  int(np.ceil(angle/policy['path_step_rad'])))
        slerp=Slerp([0,1],rotation)
        for fraction in np.linspace(0,1,steps+1)[1:]:
            result=np.eye(4);result[:3,3]=(1-fraction)*start[:3,3]+fraction*end[:3,3]
            result[:3,:3]=slerp(fraction).as_matrix();yield result


def inspect_row(geometry,row,shape,clipping_range_m=None):
    clipping=row.get('camera_clipping_range_m',clipping_range_m)
    return geometry.inspect(row['snapshot']['objects']['test_object']['world'],
        row['camera_world_optical'],row['intrinsics'],shape,clipping)


def registered_clipping(campaign):
    """Recover only the literal, hash-bound old camera constructor setting.

    Old raw rows did not save the live clipping range. This is source evidence,
    explicitly distinct from a measured runtime-property receipt.
    """
    from .common import read,sha
    reg=read(Path(campaign)/'registration.json')
    path=Path(reg['source_snapshot'])/'arx-sim/arx_sim/inprocess_app.py'
    expected=reg['execution_source_files'].get(str(path))
    if expected is None or sha(path)!=expected:raise ValueError('Unverified historical camera source')
    tree=ast.parse(path.read_text());values=[]
    function=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='_apply_camera_definition')
    for node in ast.walk(function):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='Set':continue
        receiver=node.func.value
        if not isinstance(receiver,ast.Call) or not isinstance(receiver.func,ast.Attribute) or receiver.func.attr!='CreateClippingRangeAttr':continue
        if len(node.args)!=1 or not isinstance(node.args[0],ast.Call):raise ValueError('Dynamic historical clipping setting')
        value=[float(ast.literal_eval(arg)) for arg in node.args[0].args]
        if len(value)!=2 or not 0<value[0]<value[1]:raise ValueError('Invalid historical clipping setting')
        values.append(value)
    if len(values)!=1:raise ValueError('Ambiguous historical clipping setting')
    return values[0],{'source':str(path),'source_sha256':expected,'clipping_range_m':values[0],
        'evidence':'Hash-bound immutable camera-constructor source; old rows have no measured live clipping receipt',
        'historical_live_clipping_verified':False}
