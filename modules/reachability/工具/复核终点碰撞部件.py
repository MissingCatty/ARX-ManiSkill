"""Independent CPU reconstruction of collision pairs for saved IK witnesses."""
from 公共 import EXPERIMENTS
from pathlib import Path
import sys

import fcl
import numpy as np
import trimesh
import yaml

from 诊断七轴抓取终点 import ROOT,MS,read,save,sha
sys.path.insert(0,str(MS))
from 七轴升降规划 import BodySweep,NAMES

def main():
    run=EXPERIMENTS / '抓取位姿局部修正_20261121_01'
    state_path=next((MS/'outputs/arx_joint7_pair_20261121_03/fixed_zerograsp/trials').glob('I0056_*/state.json'))
    state=read(state_path);config=EXPERIMENTS / '升降联合规划_20261120_01/七关节.yml'
    k=yaml.safe_load(config.read_text())['robot_cfg']['kinematics'];assert k['collision_sphere_buffer']==0
    sweep=BodySweep();obstacles=[];sources=[state_path,config,Path(__file__)]
    for name,g in state['scene_cuboids'].items():
        obstacles.append((name,fcl.CollisionObject(fcl.Box(*g['dims']),fcl.Transform(np.asarray(g['center'])))))
    for name,g in state['scene_meshes'].items():
        if g['target']:continue
        sources.append(Path(g['file_path']));mesh=trimesh.load_mesh(g['file_path'],process=False)
        parts=mesh.split(only_watertight=False);assert len(parts)==g['native_convex_parts'];t=np.asarray(g['world'])
        for part in parts:
            vertices=np.asarray(part.vertices)@t[:3,:3].T+t[:3,3]
            obstacles.append((name,fcl.CollisionObject(sweep.convex(vertices))))
    records=[]
    for p in sorted((run/'诊断/I0056').glob('候选*.json')):
        row=read(p);sources.append(p)
        for index,w in enumerate(row['witnesses']):
            frames=sweep.fk(state,dict(zip(NAMES,w['q'])));hits=set();table_pen=[]
            for link,spheres in k['collision_spheres'].items():
                t=frames[link]
                for sphere in spheres:
                    center=np.asarray(sphere['center'])@t[:3,:3].T+t[:3,3];radius=sphere['radius']
                    box=state['scene_cuboids']['table'];delta=np.abs(center-np.asarray(box['center']))-np.asarray(box['dims'])/2
                    sdf=np.linalg.norm(np.maximum(delta,0))+min(float(delta.max()),0.);table_pen.append(radius-sdf)
                    obj=fcl.CollisionObject(fcl.Sphere(radius),fcl.Transform(center))
                    for name,other in obstacles:
                        if fcl.collide(obj,other,fcl.CollisionRequest(),fcl.CollisionResult()):hits.add((link,name))
            error=abs(max(table_pen)-w['table_penetration_m'])
            assert error<.0002,('CPU/native reconstruction differs by >0.2mm',error)
            records.append({'candidate':row['index'],'witness':index,'colliding_link_obstacle_pairs':[list(x) for x in sorted(hits)],
                'table_crosscheck_error_m':error,'original_native_checker_clear':w['scene_clear']})
    summary={}
    for c in sorted(set(r['candidate'] for r in records)):
        part=[r for r in records if r['candidate']==c]
        summary[str(c)]={'witnesses':len(part),'collision_pairs':sorted(set(tuple(pair) for r in part for pair in r['colliding_link_obstacle_pairs'])),
            'witnesses_with_collision':sum(bool(r['colliding_link_obstacle_pairs']) for r in part)}
    save(ROOT/'docs/resource/抓取位姿局部修正首轮/I0056独立碰撞定位.json',{
        'scope':'CPU URDF FK and original spheres vs native convex scene parts; target excluded as in endpoint checker',
        'original_collision_gates_overridden':False,'CPU_crosscheck_tolerance_m':.0002,'physical_attempts':0,
        'summary':summary,'records':records,'source_sha256':{str(p):sha(p) for p in sources}})
    print(summary,flush=True)

if __name__=='__main__':main()
