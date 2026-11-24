"""Bounded static convex-collider check for four confirmed table proxy false positives."""
from __future__ import annotations
import json,xml.etree.ElementTree as ET
import numpy as np
import trimesh,yaml
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from urdf_reference import HERE,NAMES


def main() -> None:
    root=HERE/'research/20261117_01';scope=json.loads((root/'scope.json').read_text());reference=HERE.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model'
    stage=Usd.Stage.Open(json.loads((reference/'model_manifest.json').read_text())['source']);cache=UsdGeom.XformCache();geometry={}
    for link in stage.Traverse():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI) or not link.GetName().startswith('right_link'):continue
        inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(link)))
        for p in Usd.PrimRange(link):
            if not p.IsA(UsdGeom.Mesh) or not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():continue
            assert p.GetAttribute('physics:approximation').Get()=='convexHull'
            points=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float);matrix=np.array(cache.GetLocalToWorldTransform(p))@inv
            geometry[link.GetName()]=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
    joints=[]
    for j in ET.parse(reference/'arx.urdf').getroot().findall('joint'):
        o=j.find('origin');t=np.eye(4)
        if o is not None:t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
        axis=j.find('axis');axis=np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0.,0.])
        joints.append((j.get('name'),j.find('parent').get('link'),j.find('child').get('link'),j.get('type'),t,axis))
    def frames(q):
        result={'body4':np.eye(4)};pending=list(joints)
        while pending:
            ready=[x for x in pending if x[1] in result];assert ready
            for name,parent,child,kind,origin,axis in ready:
                m=np.eye(4)
                if kind=='revolute':m[:3,:3]=Rotation.from_rotvec(axis*q.get(name,0.)).as_matrix()
                elif kind=='prismatic':m[:3,3]=axis*q.get(name,0.)
                result[child]=result[parent]@origin@m
            pending=[x for x in pending if x not in ready]
        return result
    def intersection(a,b):
        gap=np.maximum(a.min(0)-b.max(0),b.min(0)-a.max(0))
        if gap.max()>1e-7:return {'overlap':False,'aabb_separating_gap_mm':float(gap.max()*1000)}
        ha=ConvexHull(a).equations;hb=ConvexHull(b).equations;planes=np.vstack([ha,hb]);A=np.c_[planes[:,:3],np.ones(len(planes))]
        lp=linprog([0,0,0,-1],A_ub=A,b_ub=-planes[:,3],bounds=[(None,None)]*4,method='highs',options={'time_limit':2.})
        assert lp.success,lp.message
        radius=float(lp.x[3]);return {'overlap':radius>1e-7,'near_contact':abs(radius)<=1e-7,'common_interior_ball_radius_mm':radius*1000}
    usd=json.loads((root/'usd_table_geometry_check_v2.json').read_text())['details'];records=json.loads((root/'reference_validation_new/records.json').read_text());assert len(usd)==len(records)
    controls={c['id'] for c in scope['cases'] if c['control']};selected={}
    for d,r in zip(usd,records):
        assert (d['case'],d['index'])==(r['case'],r['index'])
        if r['case'] in controls or (d['all_usd_mesh_aabbs_disjoint_table'] and not r['constraints_valid']):selected.setdefault((r['case'],r['index']),r)
    results=[]
    for (identifier,index),r in selected.items():
        trial=__import__('pathlib').Path(scope['source_campaign'])/'trials'/identifier;cfg=yaml.safe_load((trial/'plan/arx.yml').read_text())['robot_cfg']['kinematics'];state=json.loads((trial/'state.json').read_text());scene=json.loads((trial/'plan/scene.json').read_text());q=dict(cfg['lock_joints']);q.update(dict(zip(NAMES,np.asarray(r['q'],dtype=np.float32).astype(float))));transforms=frames(q);obstacles=[]
        for name,box in scene['cuboid'].items():
            p=box['pose'];R=Rotation.from_quat([*p[4:],p[3]]).as_matrix();v=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*np.asarray(box['dims']);obstacles.append((name,v@R.T+np.asarray(p[:3]),True))
        for name,m in scene.get('mesh',{}).items():
            mesh=trimesh.load(m['file_path'],force='mesh',process=False);parts=list(mesh.split(only_watertight=False));exact=len(parts)==state['scene_meshes'][name]['native_convex_parts'];parts=parts if exact else [mesh];p=m['pose'];R=Rotation.from_quat([*p[4:],p[3]]).as_matrix()
            for i,part in enumerate(parts):obstacles.append((f'{name}:part{i}',np.asarray(part.vertices)@R.T+np.asarray(p[:3]),exact))
        potential=[]
        for link,v in geometry.items():
            base=v@transforms[link][:3,:3].T+transforms[link][:3,3]
            for name,other,exact in obstacles:
                check=intersection(base,other)
                if check.get('overlap') or check.get('near_contact'):potential.append({'link':link,'obstacle':name,'obstacle_native_parts_recovered':exact,**check})
        result={'case':identifier,'index':index,'planning_positive_control':identifier in controls,'original_sphere_model_valid':r['constraints_valid'],'all_static_environment_convex_hulls_disjoint':not potential,'overlap_or_near_contact_pairs':potential,'physical_trials':0};results.append(result);print(identifier,index,'USD hull potential pairs',len(potential),flush=True)
    (root/'proxy_candidate_convex_check.json').write_text(json.dumps({'cases':results,'scope':'Static authored robot convexHull versus recorded native convex obstacle parts, or conservative enclosing hull if parts cannot be recovered; no dynamics or execution','native_ik_calls':0,'physical_trials':0},indent=2))


if __name__=='__main__':main()
