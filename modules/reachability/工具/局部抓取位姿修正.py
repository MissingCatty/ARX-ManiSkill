"""Bounded contact-axis grasp refinement on archived observations, no robot I/O."""
from 公共 import EXPERIMENTS
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from 诊断七轴抓取终点 import ROOT,MS,read,save,sha,now,guard

DIAG=EXPERIMENTS / '抓取位姿局部修正_20261121_01'
ANGLES=[0.,-5.,5.,-10.,10.,-15.,15.,-20.,20.,-30.,30.]

def prepare(out):
    assert not out.exists(),'New output required; do not overwrite a registered run'
    d=read(DIAG/'登记.json');assert read(DIAG/'终点诊断结果.json')['passed'];guard(d)
    files={Path(p):h for p,h in d['source_hashes'].items()}
    for p in [Path(__file__),Path(__file__).with_name('诊断七轴抓取终点.py'),ROOT/'工具/能力索引.py',
              MS/'table_truncation_model.py',ROOT/'工具/比较升降高度.py',DIAG/'终点诊断结果.json',
              MS.parents[1]/'simulation/robots/arx/finger_pad_geometry.json']:
        files[p]=sha(p)
    for c in d['cases']:
        p=Path(c['source'])/'graspcl/candidate_pool.json';files[p]=sha(p)
    out.mkdir(parents=True);(out/'源码').mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copy2(p,out/'源码'/p.name)
    reg={**d,'created_at':now(),'source_hashes':{str(p):h for p,h in files.items()},
         'source_diagnosis':str(DIAG),'required_components':['seven_dof_planner','fixed_lift_six_dof_planner','body_sweep','capability_map'],
         'angles_deg':ANGLES,'modes':['original','local_search','map_local_search'],'goal_ik_budget':24,
         'path_branch_budget':6,'per_case_mode_wall_budget_s':120,'axis':'Candidate contact-frame +Y, through preserved midpoint',
         'pose_changes':'Rotation about original candidate contact line; nominal point1/point2 and width preserved',
         'endpoint_policy':'TCP43, fixed 8cm pregrasp, nominal +3cm world-Z lift, fixed lift during approach/lift',
         'geometry_policy':'Original open-gripper sphere table clearance and observed-target closing sweep; actual pad corridor also recorded',
         'entry_world':'Full scene including the target for all methods; identical strengthened offline baseline',
         'scope':'Development-only archived inputs, no new perception/robot commands. Candidate contact points are nominal, not physical contact evidence. Grasp quality and physical success unverified.'}
    save(out/'登记.json',reg);save(out/'进度.json',{'phase':'prepared','completed':0,'total':15,'physical_attempts':0})
    print('PREPARED refinement',len(d['cases']),'cases',flush=True)

def run(out):
    reg=read(out/'登记.json');guard(reg)
    with (out/'已开始.json').open('x') as f:f.write(now())
    os.environ['CUDA_VISIBLE_DEVICES']=reg['gpu_uuid'];os.environ['GRASP_TRAJOPT_POLICY']='interactive'
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS.parent/'arx_core/scripts/curobo_arx')]
    import torch,yaml
    from 七轴升降规划 import seven_class,BodySweep,NAMES
    from table_truncation_model import target_points,box_clearance,closing_sweep_points
    from 比较升降高度 import shifted
    from 能力索引 import CapabilityMap
    started=time.monotonic();warm=read(Path(reg['cases'][0]['source'])/'state.json')
    seven=seven_class(out/'模型',warm)
    (out/'模型/六关节.yml').write_text((EXPERIMENTS / '升降联合规划_20261120_01/六关节.yml').read_text())
    six=seven.__class__(6,warm);q=six.measured();six.check_fk(q[None]);assert len(six.ik(np.asarray(warm['links']['right_eef_link']),q))
    body=BodySweep();atlas=CapabilityMap(ROOT/'产物/首版_01')
    pads=read(MS.parents[1]/'simulation/robots/arx/finger_pad_geometry.json')['parts']
    low=np.max([v['pad_neutral_EEF_bounds_m'][0] for v in pads.values()],axis=0)
    high=np.min([v['pad_neutral_EEF_bounds_m'][1] for v in pads.values()],axis=0)
    save(out/'组件就绪.json',{'passed':True,'time':now(),'startup_seconds':time.monotonic()-started,
        'model_load_counts':{'seven':1,'six':1,'body_sweep':1,'map':1},'map_load_seconds':atlas.load_seconds,'physical_attempts':0})
    print('READY_REFINEMENT',round(time.monotonic()-started,2),flush=True)

    def template(state):
        seven.bind(state);q=seven.measured();s=seven.checker.get_kinematics(q[None,None]).robot_spheres.reshape(-1,4).detach().cpu().numpy().copy()
        p=seven.checker.kinematics.kinematics_config;names={int(v):k for k,v in p.link_name_to_idx_map.items()}
        links=np.array([names[int(x)] for x in p.link_sphere_idx_map.cpu().tolist()]);keep=np.isin(links,['right_link26','right_link27','right_link28'])&(s[:,3]>0)
        s=s[keep];links=links[keep];xyz=s[:,:3]@seven.base[:3,:3].T+seven.base[:3,3]
        inv=np.linalg.inv(np.asarray(state['links']['right_eef_link']));s[:,:3]=xyz@inv[:3,:3].T+inv[:3,3]
        return (s,links,0.)

    def variants(case,state):
        source=Path(case['source']);pool=read(source/'graspcl/candidate_pool.json')['candidates'];points=target_points(source)
        shape=template(state);invbody=np.linalg.inv(np.asarray(state['links']['body4']));result=[]
        current=[state['qpos'][state['joint_names'].index(n)] for n in NAMES[1:]]
        for i,c in enumerate(pool):
            original=np.asarray(c['pose']);width=float(c['width'])
            for angle in reg['angles_deg']:
                contact=original.copy();contact[:3,:3]=original[:3,:3]@Rotation.from_euler('y',angle,degrees=True).as_matrix()
                p1=contact[:3,3]-width/2*contact[:3,1];p2=contact[:3,3]+width/2*contact[:3,1]
                assert np.linalg.norm(p1-np.asarray(c['point1']))<1e-6 and np.linalg.norm(p2-np.asarray(c['point2']))<1e-6
                g=contact.copy();g[:3,3]-=.043*g[:3,2]
                clearance=box_clearance(shape[0],g,state['scene_cuboids']['table'],0.)
                swept=closing_sweep_points(points,g,shape);support=int(np.count_nonzero(np.abs(swept[:,1])<=width/2))
                local=(points-g[:3,3])@g[:3,:3]
                actual_pad=int(np.count_nonzero((local[:,0]>=low[0])&(local[:,0]<=high[0])&(local[:,2]>=low[2])&(local[:,2]<=high[2])&(np.abs(local[:,1])<=width/2)))
                t=time.monotonic();query=atlas.query_pose(invbody@g@atlas.tcp,current_q=current,num_seeds=1,neighbors=2048)
                best=query['best_pose_sample'];map_score=best['translation_error_m']/.02+best['orientation_error_rad']/.35
                result.append({'id':f'{i:03d}_{angle:+.0f}','source_index':i,'angle_deg':angle,'contact_pose_world':contact.tolist(),
                    'eef_goal_world':g.tolist(),'point1':p1.tolist(),'point2':p2.tolist(),'width':width,'source_score':c['score'],
                    'source_score_is_rescored':False,'table_clearance_m':clearance,'observed_closing_sweep_points':support,
                    'observed_actual_pad_corridor_points':actual_pad,'geometry_admitted':bool(clearance>=0 and support>0 and actual_pad>0),
                    'map_score':map_score,'map_query_seconds':time.monotonic()-t,'map_nearest_witness':best,
                    'map_static_validity_reused':False,'physical_grasp_quality_verified':False})
        return result

    def full_plan(state,v,folder,remaining,deadline):
        seven.bind(state);start=seven.measured();g=np.asarray(v['eef_goal_world']);h=g.copy();h[:3,3]-=.08*g[:3,2];l=g.copy();l[2,3]+=.03
        solutions=seven.ik(g,start)
        if not len(solutions):return None,'grasp_ik',0
        old_height=float(start[0]);branches=0
        for qg7 in solutions:
            if branches>=remaining or time.monotonic()>deadline:break
            branches+=1;height=float(qg7[0]);sweep=body.check(state,min(old_height,height),max(old_height,height))
            if not sweep['passed']:continue
            virtual=shifted(state,height,old_height);six.bind(virtual);qg=qg7[1:].clone()
            qhs,qls=six.ik(h,qg),six.ik(l,qg)
            if not len(qhs) or not len(qls):continue
            qh,ql=qhs[0],qls[0];qh7=torch.cat([qg7[:1],qh])
            entry=deepcopy(state)
            for mesh in entry.get('scene_meshes',{}).values():mesh['target']=False
            seven.bind(entry)
            if not bool(seven.checker.validate(start[None,None]).all()) or not bool(seven.checker.validate(qh7[None,None]).all()):continue
            reach=seven.path_to(start,qh7);approach=six.path_to(qh,qg);lift=six.path_to(qg,ql)
            if reach is None or approach is None or lift is None:continue
            folder.mkdir(parents=True,exist_ok=True)
            try:
                seven.optimize(reach,folder,'进场');six.optimize(approach,folder,'接近');six.optimize(lift,folder,'名义抬升')
            except RuntimeError as e:
                if str(e)=='Trajectory optimizer rejected all attempts; no PRM-only fallback':continue
                raise
            command=read(folder/'进场_指令.json');q=np.asarray(command['q']);sweep=body.check(state,float(q[:,0].min()),float(q[:,0].max()))
            if not sweep['passed']:continue
            if time.monotonic()>deadline:return None,'wall_budget_exceeded',branches
            return {'variant':v,'selected_lift_m':height,'q_grasp_7':qg7.cpu().tolist(),'q_pre_7':qh7.cpu().tolist(),
                    'q_lift_6':ql.cpu().tolist(),'body_sweep':sweep,'full_scene_target_in_entry':True,
                    'pregrasp_world':h.tolist(),'lift_world':l.tolist(),'physical_execution_approved':False,
                    'note':'Open-finger nominal lift only; actual measured closure and holding remain untested'},'full_offline_plan_passed',branches
        return None,'pre_lift_body_or_path_failed',branches

    rows=[]
    for case in reg['cases']:
        state=read(Path(case['source'])/'state.json');t=time.monotonic();options=variants(case,state)
        save(out/'候选'/f'{case["case"]}.json',{'case':case,'variants':options,'proposal_seconds':time.monotonic()-t})
        baseline=None
        for mode in reg['modes']:
            directory=out/'比较'/case['case']/mode;directory.mkdir(parents=True,exist_ok=False)
            if mode!='original' and baseline and baseline['success']:
                result={**baseline,'mode':mode,'baseline_plan_reused_without_changes':True,'physical_attempts':0}
                save(directory/'结果.json',result);rows.append(result)
            else:
                eligible=[v for v in options if v['geometry_admitted'] and (mode!='original' or v['angle_deg']==0)]
                if mode=='map_local_search':eligible.sort(key=lambda v:(v['map_score'],abs(v['angle_deg']),v['source_index']))
                elif mode=='local_search':eligible.sort(key=lambda v:(abs(v['angle_deg']),v['source_index'],v['angle_deg']>0))
                began=time.monotonic();deadline=began+reg['per_case_mode_wall_budget_s'];attempts=[];selected=None;paths=0
                for v in eligible[:reg['goal_ik_budget']]:
                    if time.monotonic()>deadline or paths>=reg['path_branch_budget']:break
                    selected,reason,used=full_plan(state,v,directory/v['id'],reg['path_branch_budget']-paths,deadline);paths+=used
                    attempts.append({'variant':v['id'],'reason':reason,'path_branches':used})
                    print('TRY',case['case'],mode,v['id'],reason,flush=True)
                    if selected is not None:break
                result={'case':case['case'],'role':case['role'],'mode':mode,'success':selected is not None,'selected':selected,
                    'attempts':attempts,'ik_calls':len(attempts),'path_branches':paths,'geometry_admitted':len(eligible),
                    'wall_seconds':time.monotonic()-began,'wall_budget_exceeded':time.monotonic()>deadline,'physical_attempts':0,
                    'proposal_and_map_cost_excluded_from_mode_search_seconds':True,'original_candidates_retained':True}
                save(directory/'结果.json',result);rows.append(result)
            if mode=='original':baseline=result
            save(out/'进度.json',{'phase':'offline_local_refinement','completed':len(rows),'total':len(reg['cases'])*3,
                'latest':{'case':case['case'],'mode':mode,'success':result['success']},'physical_attempts':0,'updated_at':now()})
    guard(reg);save(out/'结果汇总.json',{'passed':True,'rows':rows,'source_hashes_unchanged':True,'physical_attempts':0,'wall_seconds':time.monotonic()-started,
        'scope':reg['scope'],'mode_counts':{m:{'completed':sum(r['mode']==m for r in rows),'success':sum(r['mode']==m and r['success'] for r in rows)} for m in reg['modes']}})
    save(out/'进度.json',{'phase':'complete','completed':len(rows),'total':len(rows),'physical_attempts':0,'updated_at':now()})
    print('COMPLETE_REFINEMENT',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.stage=='prepare':prepare(out)
    else:
        try:run(out)
        except BaseException:
            save(out/'故障.json',{'time':now(),'error':traceback.format_exc(),'physical_attempts':0});raise
