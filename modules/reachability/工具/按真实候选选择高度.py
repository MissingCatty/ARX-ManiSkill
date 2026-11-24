"""Select one lift using matched archived grasp/pregrasp/lift endpoint feasibility."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from 公共 import ROOT, MS, JOINTS, MODEL, URDF, TCP, read, save, sha, now
from 比较升降高度 import shifted

SELF = Path(__file__).resolve()
PRIOR = MS/'outputs/arx_physical_table_parallel_20261118_01'
ARMS = [('zerograsp','fixed_zerograsp'),('graspcl','fixed_graspcl')]
BATCH = 64


def prepare(out, scan, gpu):
    result=read(scan/'高度扫描.json');assert result['passed']
    ranked=sorted(result['rows'],key=lambda r:(-r['vertical_pose_fraction'],-r['vertical_position_fraction'],abs(r['lift_m']-.22)))
    heights=sorted({.22,.28,.32,.36,.40,.44,.46,ranked[0]['lift_m']})
    files=[SELF,Path(__file__).with_name('公共.py'),Path(__file__).with_name('比较升降高度.py'),
           scan/'登记.json',scan/'高度扫描.json',MODEL,URDF,TCP,PRIOR/'registration.json']
    records=[]
    for group,arm in ARMS:
        root=PRIOR/group/arm
        for entry in read(root/'registration.json')['schedule']:
            directory=root/'trials'/entry['id']
            pool=directory/'graspcl/candidate_pool.json'
            row={'group':arm,'id':entry['id'],'layout':entry['layout'],'directory':str(directory),
                 'candidate_count':len(read(pool)['candidates']) if pool.exists() else 0,
                 'archived_input_available':pool.exists()}
            if pool.exists():
                files += [pool,directory/'state.json']
                files += [Path(g['file_path']) for g in read(directory/'state.json')['scene_meshes'].values()]
            records.append(row)
    assert len(records)==400
    out.mkdir(parents=True,exist_ok=False)
    reg={'created_at':now(),'source_sha256':{str(p.resolve()):sha(p) for p in set(files)},
         'gpu_uuid':gpu,'scan':str(scan),'coarse_lifts_m':heights,'records':records,'record_count':400,
         'fine_step_m':.002,'fine_half_width_m':.006,'maximum_height_count':15,
         'selection':'Maximise the number of scene/algorithm pairs with at least one same candidate passing all G,H,L endpoints; then maximise the worse algorithm count; then choose the smaller change from 22cm.',
         'fine_selection':'After coarse endpoints, check +/-6mm at 2mm spacing around the coarse winner within the original joint4 limits.',
         'limits_m':[0.,.46],'num_seeds':32,'return_seeds':4,'position_tolerance_m':.002,
         'orientation_tolerance_rad':.03,'collision_activation_distance_m':0.,
         'world':'Original per-record full clutter world and original target-contact exemption. Static geometry is unchanged; only robot body4/world relation changes with joint4.',
         'finger_state':'One unchanged open-gripper model, fixed at the archived reference opening for this selection diagnostic. Formal execution uses actual measured locks.',
         'endpoint_protocol':{'tcp_offset_m':.043,'pregrasp_m':.08,'lift_world_z_m':.03},
         'scope':'Offline endpoint selection, not path validity or actual grasp success. Both arms retain their own archived candidate pools. No test outcomes used as training feedback.',
         'physical_attempts':0,'new_feedback':0,'downstream_tests':400}
    save(out/'登记.json',reg)
    print('PREPARED',heights,'400 records',sum(r['candidate_count'] for r in records),'candidates',flush=True)


def run(out):
    reg=read(out/'登记.json')
    assert all(sha(p)==digest for p,digest in reg['source_sha256'].items())
    os.environ['CUDA_VISIBLE_DEVICES']=reg['gpu_uuid']
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS/'tmp')]
    import torch
    import resident_planner as resident
    from physical_table_pair_20261118_01 import install_collision_policy
    from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    from curobo.types import GoalToolPose,JointState
    from 机器人几何 import RobotGeometry
    install_collision_policy()
    cfg=yaml.safe_load(MODEL.read_text());cfg['robot_cfg']['kinematics']['urdf_path']=str(URDF)
    path=out/'机器人.yml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    seed_record=next(r for r in reg['records'] if r['candidate_count'])
    reference=read(Path(seed_record['directory'])/'state.json')
    temporary=SimpleNamespace()
    _,scene,_=resident.bind_inputs(temporary,out/'初始碰撞世界',reference,False,False)
    scene=resident.collision_scene_config(scene)
    config=InverseKinematicsCfg.create(robot=str(path),scene_model=scene,
        self_collision_check=False,num_seeds=32,seed_solver_num_seeds=32,position_tolerance=.002,
        orientation_tolerance=.03,optimizer_collision_activation_distance=.003,max_batch_size=BATCH,
        collision_cache={'mesh':8,'primitive':8},use_cuda_graph=True)
    for rollout in [config.core_cfg.metrics_rollout_config,*config.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
    solver=InverseKinematics(config)
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(path),
        scene_model=scene,n_meshes=8,n_cuboids=8,collision_activation_distance=0.))
    checker.self_collision_cost=None
    profile=np.asarray(read(TCP)['T_eef_action']);geometry=RobotGeometry(profile,device='cuda')
    max_eef=0.;max_rotation=0.;query_count=0

    def solve(goals, starts, base):
        nonlocal max_eef,max_rotation,query_count
        n=len(goals);flags=np.zeros(n,bool);solutions=np.zeros((n,6))
        if not n:return flags,solutions
        inverse=np.linalg.inv(base)
        for offset in range(0,n,BATCH):
            count=min(BATCH,n-offset);indices=np.minimum(np.arange(offset,offset+BATCH),n-1)
            local=inverse@goals[indices]
            quat=Rotation.from_matrix(local[:,:3,:3]).as_quat()[:,[3,0,1,2]]
            target=GoalToolPose(tool_frames=['right_eef_link'],
                position=torch.as_tensor(np.ascontiguousarray(local[:,:3,3]),device='cuda',dtype=torch.float32).reshape(BATCH,1,1,1,3),
                quaternion=torch.as_tensor(np.ascontiguousarray(quat),device='cuda',dtype=torch.float32).reshape(BATCH,1,1,1,4))
            current=torch.as_tensor(np.ascontiguousarray(starts[indices]),device='cuda',dtype=torch.float32)
            result=solver.solve_pose(target,current_state=JointState.from_position(current,joint_names=JOINTS),return_seeds=4)
            qs=result.solution.reshape(BATCH,4,6)
            valid=result.success.reshape(BATCH,4)&checker.validate(qs.reshape(-1,6)[:,None]).reshape(BATCH,4)
            cpu=qs.detach().cpu().numpy();ok=valid.detach().cpu().numpy()
            ok &= ((cpu>=geometry.lower-1e-6)&(cpu<=geometry.upper+1e-6)).all(-1)
            distance=((cpu-starts[indices,None])**2).sum(-1);distance[~ok]=np.inf
            best=distance.argmin(-1);chosen=cpu[np.arange(BATCH),best];found=ok.any(-1)
            if found.any():
                xyz,rot=geometry.fk(torch.as_tensor(chosen[found],device='cuda'),jacobian=False)
                world_r=base[:3,:3]@rot.cpu().numpy();eef_r=world_r@profile[:3,:3].T
                world_tcp=xyz.cpu().numpy()@base[:3,:3].T+base[:3,3]
                world_eef=world_tcp-.043*eef_r[:,:,2]
                ep=np.linalg.norm(world_eef-goals[indices[found],:3,3],axis=1)
                er=Rotation.from_matrix(eef_r@goals[indices[found],:3,:3].transpose(0,2,1)).magnitude()
                assert ep.max()<.00205 and er.max()<.0301,(ep.max(),er.max())
                max_eef=max(max_eef,float(ep.max()));max_rotation=max(max_rotation,float(er.max()))
            flags[offset:offset+count]=found[:count];solutions[offset:offset+count]=chosen[:count]
            query_count+=count
        return flags,solutions

    begun=time.perf_counter();all_rows=[];height_rows=[]
    save(out/'组件就绪.json',{'pid':os.getpid(),'models':{'ik':1,'checker':1,'fk':1},'physical_attempts':0})
    def assess_height(height):
        rows=[];started=time.perf_counter()
        for number,record in enumerate(reg['records']):
            directory=Path(record['directory']);n=record['candidate_count']
            row={k:record[k] for k in ['id','group','layout','candidate_count','archived_input_available']}
            row.update(lift_m=height,grasp_candidates=0,triplet_candidates=0,endpoint_pass=False)
            if n:
                state=read(directory/'state.json');old=state['qpos'][state['joint_names'].index('joint4')]
                state=shifted(state,height,old);base=np.array(state['links']['body4'])
                _,world,_=resident.bind_inputs(temporary,out/'碰撞世界'/record['group']/record['id'],state,False,False)
                world=resident.collision_scene_config(world);solver.update_world(world);checker.update_world(world);solver.reset_seed()
                poses=np.array([c['pose'] for c in read(directory/'graspcl/candidate_pool.json')['candidates']])
                g=poses.copy();g[:,:3,3]-=.043*poses[:,:3,2]
                h=g.copy();h[:,:3,3]-=.08*g[:,:3,2]
                lift=g.copy();lift[:,2,3]+=.03
                current=np.array([state['qpos'][state['joint_names'].index(j)] for j in JOINTS])
                current=np.clip(current,geometry.lower+1e-4,geometry.upper-1e-4)
                passed_g,qg=solve(g,np.tile(current,(n,1)),base)
                indices=np.flatnonzero(passed_g)
                passed_h,qh=solve(h[indices],qg[indices],base)
                passed_l,ql=solve(lift[indices],qg[indices],base)
                complete=passed_h&passed_l
                row.update(grasp_candidates=int(passed_g.sum()),triplet_candidates=int(complete.sum()),endpoint_pass=bool(complete.any()))
                np.savez_compressed(out/'见证'/record['group']/f'{record["id"]}_{round(height*1000):03d}.npz',
                    g=g,h=h,lift=lift,passed_g=passed_g,qg=qg,after_g_indices=indices,passed_h=passed_h,qh=qh,passed_l=passed_l,ql=ql)
            rows.append(row)
            if (number+1)%20==0:
                save(out/'进度.json',{'phase':'真实候选终点核验','lift_m':height,'records':number+1,'records_per_height':400,
                    'heights_completed':len(height_rows),'query_count':query_count,'physical_attempts':0})
        groups={arm:{'passed':sum(r['endpoint_pass'] for r in rows if r['group']==arm),'total':200} for _,arm in ARMS}
        summary={'lift_m':height,'groups':groups,'passed':sum(r['endpoint_pass'] for r in rows),
                 'total':400,'worst_group_passes':min(g['passed'] for g in groups.values()),'wall_seconds':time.perf_counter()-started}
        all_rows.extend(rows);height_rows.append(summary)
        save(out/'高度终点汇总.json',{'heights':height_rows,'rows':all_rows,'query_count':query_count,'physical_attempts':0})
        print('CANDIDATE_HEIGHT',height,summary['passed'],groups,'seconds',round(summary['wall_seconds'],1),flush=True)

    for _,arm in ARMS:(out/'见证'/arm).mkdir(parents=True)
    for height in reg['coarse_lifts_m']:assess_height(height)
    rank=lambda r:(-r['passed'],-r['worst_group_passes'],abs(r['lift_m']-.22))
    coarse=sorted(height_rows,key=rank)[0]['lift_m']
    fine=sorted({round(coarse+d/1000,3) for d in range(-6,7,2) if 0<=coarse+d/1000<=.46}
                -{r['lift_m'] for r in height_rows})
    assert len(fine)+len(height_rows)<=reg['maximum_height_count']
    save(out/'细化登记.json',{'created_at':now(),'coarse_winner_m':coarse,'fine_lifts_m':fine,'no_physical_tests_claimed':True})
    for height in fine:assess_height(height)
    chosen=sorted(height_rows,key=rank)[0]
    assert all(sha(p)==digest for p,digest in reg['source_sha256'].items())
    result={'passed':True,'selected_lift_m':chosen['lift_m'],'selected':chosen,'heights':height_rows,
            'scope':reg['scope'],'physical_attempts':0,'new_feedback':0,'query_count':query_count,
            'maximum_independent_eef_error_m':max_eef,'maximum_independent_orientation_error_rad':max_rotation,
            'source_files_unchanged':True,'wall_seconds':time.perf_counter()-begun,'model_loads':{'ik':1,'checker':1,'fk':1}}
    save(out/'高度选择.json',result);save(out/'进度.json',{'phase':'完成','selected_lift_m':chosen['lift_m'],'heights':len(height_rows),'physical_attempts':0})
    print('SELECTED',chosen,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scan',type=Path);parser.add_argument('--gpu')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out,args.scan.resolve(),args.gpu)
    else:
        try:run(out)
        except BaseException:
            save(out/'故障.json',{'error':traceback.format_exc(),'physical_attempts':0});raise
