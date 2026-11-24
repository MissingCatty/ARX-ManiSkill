"""Diagnose archived grasp endpoints; no simulator or robot commands."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
MS=ROOT.parent/'maniskill'
SOURCE=MS/'outputs/arx_joint7_pair_20261121_03/fixed_zerograsp/trials'
TARGETS=['I0052','I0056','I0199']
CONTROLS=['I0002','I0004']
GPU='GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now().astimezone().isoformat()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def prepare(out):
    assert not out.exists(),'Use a new registration; do not overwrite an earlier run'
    files=[Path(__file__),MS/'七轴升降规划.py',ROOT/'工具/验证升降联合规划.py',
        ROOT/'工具/机器人几何.py',MS/'resident_planner.py',MS/'planning_rejection_audit.py',
        MS/'assets/arx/robot.urdf',MS/'assets/arx/manifest.json']
    cases=[]
    for id in TARGETS+CONTROLS:
        paths=list(SOURCE.glob(id+'_*'));assert len(paths)==1,(id,paths)
        p=paths[0];s=read(p/'state.json');result=read(p/'result.json')
        role='diagnostic_failure' if id in TARGETS else 'successful_control'
        assert result['reason']=='no_feasible_native_post_or_recalled_candidate' if id in TARGETS else result['success']
        pool=read(p/'graspcl/candidate_pool.json')['candidates'];assert pool
        cases.append({'case':id,'case_id':p.name,'source':str(p),'role':role,'candidates':len(pool)})
        files += [p/'state.json',p/'result.json',p/'plan_result.json',p/'graspcl/candidate_pool.json',
                  p/'camera/color.png',p/'camera/depth.png',p/'camera/mask.png',p/'camera/params.json']
        files += [Path(g['file_path']) for g in s.get('scene_meshes',{}).values()]
    out.mkdir(parents=True);(out/'源码').mkdir()
    for f in files:
        if f.suffix=='.py':shutil.copy2(f,out/'源码'/f.name)
    reg={'created_at':now(),'cases':cases,'gpu_uuid':GPU,'required_components':['seven_dof_planner_and_checker','collision_free_diagnostic_ik','body_sweep'],
         'source_hashes':{str(f.resolve()):sha(f) for f in set(files)},'pure_ik_seeds':128,'pure_return_seeds':32,
         'position_tolerance_m':.002,'orientation_tolerance_rad':.03,'case_budget_s':180,
         'physical_attempts':0,'simulator_connection':False,'candidate_pose_changes':0,'automatic_retries':False,
         'scope':'Original seven-axis endpoint solver and separate collision-free diagnostic IK on identical archived goals. Each solution rechecked against original full scene and carried-body checker. No mathematical unreachability claims.'}
    save(out/'登记.json',reg);save(out/'进度.json',{'phase':'prepared','completed':0,'total':len(cases),'physical_attempts':0})
    print('PREPARED',len(cases),'cases',sum(c['candidates'] for c in cases),'goals',flush=True)

def guard(reg):
    for f,d in reg['source_hashes'].items():assert sha(f)==d,f

def run(out):
    reg=read(out/'登记.json');guard(reg)
    with (out/'已开始.json').open('x') as stream:json.dump({'time':now(),'physical_attempts':0},stream)
    os.environ['CUDA_VISIBLE_DEVICES']=reg['gpu_uuid'];os.environ['GRASP_TRAJOPT_POLICY']='interactive'
    os.environ['OMP_NUM_THREADS']='4'
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(ROOT/'工具'),str(MS.parent/'arx_core/scripts/curobo_arx')]
    import torch
    from curobo.types import GoalToolPose,JointState
    from planning_rejection_audit import make_pure
    from ik_workspace_audit import box_penetration
    from 七轴升降规划 import seven_class,BodySweep
    started=time.monotonic();warm=read(Path(reg['cases'][0]['source'])/'state.json')
    model=seven_class(out/'七轴模型',warm);pure=make_pure(model.path)
    body=BodySweep()
    def goal(world):
        t=model.inv@world;q=Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
        return GoalToolPose(tool_frames=['right_eef_link'],
            position=torch.as_tensor(t[:3,3],device='cuda',dtype=torch.float32).reshape(1,1,1,1,3),
            quaternion=torch.as_tensor(q,device='cuda',dtype=torch.float32).reshape(1,1,1,1,4))
    def solve(world,start):
        r=pure.solve_pose(goal(world),current_state=JointState.from_position(start[None].clone(),joint_names=model.names),return_seeds=32)
        q=r.solution.reshape(-1,7).detach().clone();ok=r.success.reshape(-1).detach().clone()
        actual=model.independent_fk(q);pe=np.linalg.norm(actual[:,:3,3]-world[:3,3],axis=1)
        re=Rotation.from_matrix(actual[:,:3,:3]@world[:3,:3].T).magnitude()
        valid=ok & torch.as_tensor((pe<=.002001)&(re<=.030001),device='cuda')
        valid &= ((q>=model.limits[:,0]-1e-6)&(q<=model.limits[:,1]+1e-6)).all(-1)
        return q[valid],{'solver_claimed':int(ok.sum()),'independently_valid':int(valid.sum()),
             'minimum_position_error_m':float(pe.min()),'minimum_orientation_error_rad':float(re.min())}
    initial=model.measured();world=np.asarray(warm['links']['right_eef_link'])
    probe,detail=solve(world,initial);assert len(probe),'Diagnostic solver cannot recover recorded pose'
    assert model.checker.validate(probe[:,None]).any(),'No scene-valid measured-pose diagnostic witness'
    save(out/'组件就绪.json',{'time':now(),'passed':True,'model_load_counts':{'seven_dof':1,'pure_ik':1,'body_sweep':1},
         'startup_seconds':time.monotonic()-started,'measured_pose_probe':detail,'physical_attempts':0})
    print('READY',round(time.monotonic()-started,2),flush=True)
    all_rows=[]
    for number,case in enumerate(reg['cases'],1):
        source=Path(case['source']);state=read(source/'state.json');pool=read(source/'graspcl/candidate_pool.json')['candidates']
        model.bind(state);start=model.measured();old_height=float(start[0]);rows=[];began=time.monotonic()
        params=model.checker.kinematics.kinematics_config
        names={int(v):k for k,v in params.link_name_to_idx_map.items()}
        sphere_names=[names[int(v)] for v in params.link_sphere_idx_map.cpu().tolist()]
        folder=out/'诊断'/case['case'];folder.mkdir(parents=True,exist_ok=False)
        for index,c in enumerate(pool):
            if time.monotonic()-began>reg['case_budget_s']:
                rows.append({'index':index,'reason':'diagnostic_time_budget_exceeded'});break
            contact=np.asarray(c['pose']);g=contact.copy();g[:3,3]-=.043*g[:3,2]
            original=model.ik(g,start);qs,detail=solve(g,start)
            if len(qs):
                full=model.checker.validate(qs[:,None]).reshape(-1).cpu().numpy()
                spheres=model.checker.get_kinematics(qs[:,None]).robot_spheres.reshape(len(qs),-1,4).detach()
                table=box_penetration(spheres,model.base,state['scene_cuboids']['table']).cpu().numpy()
                ground=box_penetration(spheres,model.base,state['scene_cuboids']['ground']).cpu().numpy()
                witnesses=[]
                for i,q in enumerate(qs):
                    height=float(q[0]);sweep=body.check(state,min(old_height,height),max(old_height,height))
                    worst=int(np.argmax(table[i]))
                    witnesses.append({'q':q.cpu().tolist(),'scene_clear':bool(full[i]),'table_penetration_m':float(table[i].max()),
                        'ground_penetration_m':float(ground[i].max()),'worst_table_link':sphere_names[worst],
                        'body_sweep':sweep,'table_worst_sphere_world':(spheres[i,worst,:3]@torch.as_tensor(model.base[:3,:3].T,device='cuda',dtype=spheres.dtype)+torch.as_tensor(model.base[:3,3],device='cuda',dtype=spheres.dtype)).cpu().tolist()})
            else:witnesses=[]
            valid=sum(w['scene_clear'] and w['body_sweep']['passed'] for w in witnesses)
            classification='original_solver_found_endpoint' if len(original) else ('valid_unchanged_goal_found_by_diagnostic_solver' if valid else ('kinematic_solutions_rejected_by_current_scene_or_body_sweep' if len(qs) else 'no_kinematic_witness_in_bounded_search'))
            row={'case':case['case'],'index':index,'source_score':c['score'],'contact_pose_world':contact.tolist(),
                 'eef_goal_world':g.tolist(),'classification':classification,'original_valid_solutions':len(original),
                 'pure_ik':detail,'scene_and_body_valid_witnesses':valid,'witnesses':witnesses,
                 'candidate_goal_modified':False,'physical_attempts':0}
            save(folder/f'候选{index:03d}.json',row);rows.append(row)
            print('GOAL',case['case'],index,classification,'kinematic',len(qs),'scene+body',valid,flush=True)
        case_row={'case':case,'rows':rows,'wall_seconds':time.monotonic()-began}
        save(folder/'汇总.json',case_row);all_rows.append(case_row)
        save(out/'进度.json',{'phase':'diagnosing_archived_goals','completed':number,'total':len(reg['cases']),
             'latest_case':case['case'],'physical_attempts':0,'updated_at':now()})
    guard(reg)
    counts=Counter(r['classification'] for case in all_rows for r in case['rows'] if 'classification' in r)
    result={'passed':True,'time':now(),'cases':all_rows,'classification_counts':dict(counts),
            'source_hashes_unchanged':True,'physical_attempts':0,'candidate_pose_changes':0,'wall_seconds':time.monotonic()-started}
    save(out/'终点诊断结果.json',result);save(out/'进度.json',{'phase':'complete','completed':len(all_rows),'total':len(reg['cases']),
         'classification_counts':dict(counts),'physical_attempts':0,'updated_at':now()})
    print('COMPLETE',dict(counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.stage=='prepare':prepare(out)
    else:
        try:run(out)
        except BaseException:
            save(out/'故障.json',{'time':now(),'error':traceback.format_exc(),'physical_attempts':0});raise
