"""Read-only target penetration checks for independent static ARX data.

This supplements the original target-excluding grasp planner. It does not edit
the robot model or PhysX. Convex-decomposed target meshes use their enclosing
convex hull here, so an overlap may reject a usable observation. This is not a
PhysX cooked-shape or physical grasp certificate.
"""
import numpy as np
from scripts.curobo_arx.data_factory.common import digest


def export_body(stage, path):
    from pxr import Usd, UsdGeom, UsdPhysics
    prim=stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():raise RuntimeError('Missing body for static surface check: '+str(path))
    cache=UsdGeom.XformCache();frame=np.asarray(cache.GetLocalToWorldTransform(prim)).T.copy()
    scale=np.linalg.norm(frame[:3,:3],axis=0)
    if np.any(scale<=0):raise RuntimeError('Degenerate collider-body scale')
    frame[:3,:3]/=scale
    if not np.allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(frame[:3,:3])<=0:
        raise RuntimeError('Unsupported collider-body shear/reflection')
    # Snapshot body poses are rigid. Retain the authored body scale in the
    # vertices instead of cancelling it with a full scaled-matrix inverse.
    inverse=np.linalg.inv(frame)
    shapes=[]
    for child in Usd.PrimRange(prim):
        if not child.HasAPI(UsdPhysics.CollisionAPI):continue
        if not UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get():continue
        if not child.IsA(UsdGeom.Mesh):raise RuntimeError('Unaccounted ARX collider type: '+str(child.GetPath()))
        kind=UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()
        if kind not in ('convexHull','convexDecomposition','boundingSphere'):
            raise RuntimeError('Unaccounted ARX collider approximation: '+str(kind))
        v=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(),dtype=float)
        matrix=inverse@np.asarray(cache.GetLocalToWorldTransform(child)).T
        points=v@matrix[:3,:3].T+matrix[:3,3]
        if len(points)<4 or not np.isfinite(points).all():raise RuntimeError('Invalid collision surface vertices')
        shapes.append({'path':str(child.GetPath()),'approximation':kind,'points_body':points.tolist()})
    return shapes


def export_geometry(bridge):
    stage=bridge.context.stage
    robot={name:export_body(stage,body.paths[0]) for name,body in bridge.links.items()}
    target=export_body(stage,bridge.objects['test_object'].paths[0])
    if not target or not all(robot.get(k) for k in ('right_link26','right_link27','right_link28')):
        raise RuntimeError('Incomplete actual target/gripper collision geometry')
    data={'schema':'arx_static_target_surfaces_v1','robot':robot,'target':target,
        'physics_or_collision_configuration_modified':False,
        'scope':'Original enabled collider vertices in rigid-body coordinates; enclosing convex hull for decomposition; conservative bounding sphere from mesh vertices'}
    return {**data,'geometry_sha256':digest(data)}


class SurfaceChecker:
    def __init__(self, geometry):
        import fcl
        import trimesh
        data={k:v for k,v in geometry.items() if k!='geometry_sha256'}
        if digest(data)!=geometry['geometry_sha256']:raise ValueError('Static collision geometry digest mismatch')
        self.fcl=fcl;self.geometry=geometry;self.version=fcl.__version__
        def shape(row):
            vertices=np.asarray(row['points_body'],dtype=float)
            if row['approximation']=='boundingSphere':
                center=(vertices.min(0)+vertices.max(0))/2
                radius=float(np.linalg.norm(vertices-center,axis=1).max())
                return row['path'],fcl.Sphere(radius),center
            hull=trimesh.Trimesh(vertices=vertices,process=False).convex_hull
            faces=np.column_stack([np.full(len(hull.faces),3),hull.faces]).reshape(-1)
            # Convex volume queries detect containment as well as surface
            # intersection; triangle-only BVHs would miss full containment.
            return row['path'],fcl.Convex(hull.vertices,len(hull.faces),faces),np.zeros(3)
        self.robot={name:[shape(r) for r in rows] for name,rows in geometry['robot'].items()}
        self.target=[shape(r) for r in geometry['target']]

    def check(self,snapshot):
        fcl=self.fcl
        def obj(shape,matrix):
            _,geom,center=shape;matrix=np.asarray(matrix)
            return fcl.CollisionObject(geom,fcl.Transform(matrix[:3,:3],matrix[:3,3]+matrix[:3,:3]@center))
        targets=[obj(s,snapshot['objects']['test_object']['world']) for s in self.target]
        checks=[]
        for name,shapes in self.robot.items():
            if name not in snapshot['link_world']:raise RuntimeError('Actual collision-body pose missing: '+name)
            distances=[];hits=[]
            for s in shapes:
                body=obj(s,snapshot['link_world'][name])
                for target,ts in zip(targets,self.target):
                    hit=fcl.collide(body,target,fcl.CollisionRequest(),fcl.CollisionResult())
                    distance=fcl.distance(body,target,fcl.DistanceRequest(),fcl.DistanceResult())
                    if not np.isfinite(distance):raise RuntimeError('Nonfinite static surface distance')
                    distances.append(float(distance))
                    if hit or distance<=0:hits.append({'robot_collider':s[0],'target_collider':ts[0]})
            if distances:checks.append({'link':name,'minimum_distance_m':min(distances),'intersections':hits,'pieces':len(shapes)})
        if not checks:raise RuntimeError('No static surface checks')
        return {'pass':not any(r['intersections'] for r in checks),'checks':checks,
            'sample_snapshot_sha256':digest(snapshot),'geometry_sha256':self.geometry['geometry_sha256'],
            'fcl_version':self.version,'physical_trajectory_executed':False,
            'original_curobo_model_and_collision_settings_unchanged':True,
            'scope':self.geometry['scope'],'minimum_distance_m':min(r['minimum_distance_m'] for r in checks)}
