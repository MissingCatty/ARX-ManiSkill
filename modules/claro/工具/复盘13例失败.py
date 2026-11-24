"""Read archived scene evidence and locate release-state sphere/mesh overlap on CPU."""
from pathlib import Path
from arx_project.layout import canonical_path, read_json
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/06_做配对验证/resources'
OUT=BASE/'失败复盘_20261120'


def read(p):return read_json(p)


def sha(p):return hashlib.sha256(canonical_path(p).read_bytes()).hexdigest()


def spheres(state,config):
    qs=dict(zip(state['joint_names'],state['qpos']))
    transforms={config['base_link']:np.asarray(state['links'][config['base_link']])}
    joints=list(ET.parse(config['urdf_path']).getroot().findall('joint'))
    for _ in range(len(joints)):
        changed=False
        for j in joints:
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if parent not in transforms or child in transforms:continue
            t=np.eye(4);o=j.find('origin');t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            motion=np.eye(4);kind=j.get('type')
            if kind!='fixed':
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');q=qs[j.get('name')]
                if kind=='prismatic':motion[:3,3]=q*axis
                else:motion[:3,:3]=Rotation.from_rotvec(q*axis).as_matrix()
            transforms[child]=transforms[parent]@t@motion;changed=True
        if not changed:break
    error=float(np.abs(transforms['right_eef_link']-np.asarray(state['links']['right_eef_link'])).max())
    assert error<2e-5,error
    points=[];radii=[];identities=[]
    for link,items in config['collision_spheres'].items():
        t=transforms[link]
        for i,sphere in enumerate(items):
            points.append(t[:3,:3]@sphere['center']+t[:3,3]);radii.append(sphere['radius']+config['collision_sphere_buffer'])
            identities.append((link,i))
    return np.asarray(points),np.asarray(radii),identities,error


def main():
    OUT.mkdir(exist_ok=False)
    aggregate=BASE/'S5_固定挡物接续01/配对合并验收.json';original=read(aggregate)
    result={'scope':'Read-only archived evidence; CPU FK and nearest triangle geometry, no simulator or GPU access.',
            'physical_attempts':0,'original_summary_sha256':sha(aggregate),'release_collision_details':[],
            'entry_world_comparisons':[],'source_sha256':{str(aggregate):sha(aggregate)}}
    for case in ['I0048','I0100']:
        directory=BASE/'S5_固定挡物配对01/试次'/case/'02_放置'
        state_path=directory/'released_state.json';model_path=directory/'撤离规划/arx.yml'
        state=read(state_path);config=yaml.safe_load(model_path.read_text())['robot_cfg']['kinematics']
        points,radii,ids,error=spheres(state,config)
        table=state['scene_cuboids']['table'];delta=np.abs(points-np.asarray(table['center']))-np.asarray(table['dims'])/2
        table_gap=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)-radii
        pairs=[]
        for name,geometry in state['scene_meshes'].items():
            assert sha(geometry['file_path'])==geometry['file_sha256']
            transform=np.asarray(geometry['world']);local=(points-transform[:3,3])@transform[:3,:3]
            mesh=trimesh.load(geometry['file_path'],force='mesh',process=False)
            _,distances,_=trimesh.proximity.closest_point(mesh,local)
            gap=distances-radii
            for index in np.flatnonzero(gap<0):
                link,sphere=ids[index]
                pairs.append({'object':name,'model':geometry.get('model_id'),'link':link,'sphere':sphere,
                     'radius_m':float(radii[index]),'nearest_surface_distance_m':float(distances[index]),
                     'surface_overlap_m':float(-gap[index])})
            result['source_sha256'][geometry['file_path']]=geometry['file_sha256']
        result['release_collision_details'].append({'case':case,'independent_fk_maximum_element_error':error,
            'minimum_sphere_table_clearance_m':float(table_gap.min()),'sphere_table_overlap_count':int((table_gap<0).sum()),
            'sphere_mesh_surface_overlap_count':len(pairs),'largest_overlaps':sorted(pairs,key=lambda p:-p['surface_overlap_m'])[:12],
            'interpretation':'Sphere/triangle surface overlap localizes geometry. It is not a PhysX contact measurement or an exact reproduction of every cuRobo internal distance cost.'})
        for p in [state_path,model_path]:result['source_sha256'][str(p)]=sha(p)
    new=read(ROOT.parent/'reachability/实验/自动研究_20261120_01/轮次/00_基线/汇总.json')
    for case in ['I0167','I0190']:
        d=BASE/'S5_固定挡物接续01/试次'/case/'01_抓挡'
        old=read(d/'plan/scene.json');full=read(d/'全场景轨迹复核/scene.json')
        a=yaml.safe_load((d/'plan/arx.yml').read_text());b=yaml.safe_load((d/'全场景轨迹复核/arx.yml').read_text())
        selected=read(d/'plan_result.json')['selected'];replayed=next(r for r in new['rows'] if r['case']==case)
        result['entry_world_comparisons'].append({'case':case,'robot_configuration_equal':a==b,
            'static_cuboids_equal':old.get('cuboid')==full.get('cuboid'),
            'common_meshes_equal':all(g==full['mesh'].get(n) for n,g in old.get('mesh',{}).items()),
            'meshes_added_only_in_final_validation':sorted(set(full.get('mesh',{}))-set(old.get('mesh',{}))),
            'added_object_role':'distractor_00 is the wood block selected for removal, not the original task target',
            'old_selected_candidate':selected['index'],'replayed_selected_candidate':replayed['selected']['candidate_index'],
            'replayed_full_scene_initial_plan_passed':replayed['success'],
            'maximum_G_difference':float(np.abs(np.asarray(selected['grasp_world'])-replayed['selected']['grasp_world']).max()),
            'maximum_H_difference':float(np.abs(np.asarray(selected['pregrasp_world'])-replayed['selected']['pregrasp_world']).max())})
        for p in [d/'plan/scene.json',d/'全场景轨迹复核/scene.json',d/'plan/arx.yml',d/'全场景轨迹复核/arx.yml',d/'plan_result.json']:
            result['source_sha256'][str(p)]=sha(p)
    p=BASE/'S5_固定挡物配对01/试次/I0055/01_抓挡/execution.jsonl'
    rows=[json.loads(line) for line in p.read_text().splitlines()];z0=rows[0]['object_pose'][2]
    result['hold_failure']={'case':'I0055','peak_object_rise_m':max(r['object_pose'][2]-z0 for r in rows),
                            'last_object_rise_m':rows[-1]['object_pose'][2]-z0,
                            'classification':'Actual grasp did not maintain lift; planning and arm tracking had passed.'}
    result['source_sha256'][str(p)]=sha(p)
    assert all(sha(p)==h for p,h in result['source_sha256'].items())
    result['source_hashes_unchanged']=True
    (OUT/'复盘依据.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for r in result['release_collision_details']:print('RELEASE',r['case'],r['sphere_table_overlap_count'],r['largest_overlaps'][:3],flush=True)
    print('ENTRY',result['entry_world_comparisons'],flush=True)


if __name__=='__main__':main()
