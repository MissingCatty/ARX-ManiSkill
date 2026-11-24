"""Read-only static USD collider/table checks at saved mathematical IK witnesses."""
from __future__ import annotations

import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics

HERE=Path(__file__).resolve().parent


def main() -> None:
    root=HERE/'research/20261117_01';scope=json.loads((root/'scope.json').read_text())
    reference=HERE.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model'
    manifest=json.loads((reference/'model_manifest.json').read_text());source=Path(manifest['source'])
    stage=Usd.Stage.Open(str(source));cache=UsdGeom.XformCache()
    selected={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName().startswith('right_link')}
    geometry={}
    for name,link in selected.items():
        inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(link)));geometry[name]=[]
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            transform=np.array(cache.GetLocalToWorldTransform(prim))@inv
            vertices=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
            geometry[name].append((str(prim.GetPath()),vertices))
    xml=ET.parse(reference/'arx.urdf').getroot();joints=[]
    for joint in xml.findall('joint'):
        origin=joint.find('origin');t=np.eye(4)
        if origin is not None:
            t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        axis=joint.find('axis');axis=np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0.,0.])
        joints.append((joint.get('name'),joint.find('parent').get('link'),joint.find('child').get('link'),joint.get('type'),t,axis))
    def forward(qs):
        frames={'body4':np.eye(4)};pending=list(joints)
        while pending:
            ready=[j for j in pending if j[1] in frames];assert ready
            for name,parent,child,kind,origin,axis in ready:
                motion=np.eye(4);q=qs.get(name,0.)
                if kind=='revolute':motion[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix()
                elif kind=='prismatic':motion[:3,3]=axis*q
                frames[child]=frames[parent]@origin@motion
            pending=[j for j in pending if j not in ready]
        return frames
    records=json.loads((root/'reference_validation_new/records.json').read_text());details=[];max_error=0.
    for case in scope['cases']:
        items=[r for r in records if r['case']==case['id']]
        if not items:continue
        trial=Path(scope['source_campaign'])/'trials'/case['id'];cfg=json.loads('{}')
        import yaml
        cfg=yaml.safe_load((trial/'plan/arx.yml').read_text())['robot_cfg']['kinematics'];scene=json.loads((trial/'plan/scene.json').read_text());table=scene['cuboid']['table'];box=table['pose'];box_rot=Rotation.from_quat([*box[4:],box[3]]).as_matrix();half=np.asarray(table['dims'])/2
        arrays=np.load(root/'reference_validation_new'/f"{case['id']}.npz");native_spheres=arrays['spheres'].reshape(len(items),-1,4)
        for i,item in enumerate(items):
            qs=dict(cfg['lock_joints']);qs.update(dict(zip([f'right_joint{k}' for k in range(21,27)],arrays['q'][i].astype(float))));frames=forward(qs)
            own=[]
            for link in cfg['collision_link_names']:
                for sphere in cfg['collision_spheres'][link]:
                    point=frames[link][:3,:3]@np.asarray(sphere['center'])+frames[link][:3,3];own.append([*point,sphere['radius']])
            error=float(np.max(np.abs(np.asarray(own)-native_spheres[i])));max_error=max(max_error,error);assert error<2e-6,'Independent link transforms disagree with actual cuRobo spheres'
            hits=[];separations=[]
            for link,meshes in geometry.items():
                for path,vertices in meshes:
                    base_vertices=vertices@frames[link][:3,:3].T+frames[link][:3,3];local=(base_vertices-np.asarray(box[:3]))@box_rot
                    inside=np.all(np.abs(local)<half-1e-7,axis=1)
                    gap=np.maximum(local.min(axis=0)-half,-half-local.max(axis=0))
                    if case['id']=='B0209_S5_occluded_048_hammer' and item['index']==14 and link=='right_link26':
                        lowest=local[np.argmin(local[:,2])]
                        (root/'hammer14_table_gap_point.json').write_text(json.dumps({'lowest_collider_point_in_table_frame_m':lowest.tolist(),'point_xy_inside_table':bool(np.all(np.abs(lowest[:2])<half[:2])),'vertical_gap_mm':float((lowest[2]-half[2])*1000),'table_half_extents_m':half.tolist(),'source_collider':path},indent=2))
                    separations.append({'link':link,'collider':path,'aabb_disjoint':bool(np.any(gap>1e-7)),
                                        'maximum_separating_axis_gap_mm':float(gap.max()*1000)})
                    all_xy=np.all(np.abs(local[:,:2])<half[:2]-1e-7)
                    crosses=all_xy and local[:,2].min()<half[2]-1e-7 and local[:,2].max()>-half[2]+1e-7
                    if inside.any() or crosses:
                        hits.append({'link':link,'collider':path,'proof':'vertex_inside_box' if inside.any() else 'xy_contained_vertical_span','minimum_box_local_z_m':float(local[:,2].min()),'height_below_table_top_mm':float((half[2]-local[:,2].min())*1000)})
            details.append({'case':case['id'],'index':item['index'],'sphere_model_valid':item['constraints_valid'],'usd_collider_table_intersects':bool(hits),'all_usd_mesh_aabbs_disjoint_table':all(s['aabb_disjoint'] for s in separations),'hits':hits,'mesh_separations':separations})
    controls={c['id'] for c in scope['cases'] if c['control']};failed={(d['case'],d['index']) for d in details if d['case'] not in controls};intersecting={(d['case'],d['index']) for d in details if d['case'] not in controls and d['usd_collider_table_intersects']}
    proved_clear={(d['case'],d['index']) for d in details if d['case'] not in controls and d['all_usd_mesh_aabbs_disjoint_table']}
    result={'asset':str(source),'asset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'read_only_stage':True,'sphere_coordinate_crosscheck_max_m':max_error,'witness_rows':len(details),'unique_failed_goals':len(failed),'unique_failed_goals_with_usd_table_intersection':len(intersecting),'unique_failed_goals_with_proved_table_clear_witness':len(proved_clear),'proved_table_clear_goal_indices':sorted(proved_clear),'positive_control_usd_table_intersections':sum(d['usd_collider_table_intersects'] for d in details if d['case'] in controls),'details':details,'physical_trials':0,'native_ik_calls':0}
    (root/'usd_table_geometry_check_v2.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='details'},flush=True)


if __name__=='__main__':main()
