"""Isolated seven-DOF offline pilot; never imports a simulator or commands a robot."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from 公共 import ROOT, MS, MODEL, URDF, TCP, JOINTS, read, save, sha, now
from 比较升降高度 import shifted

SELF=Path(__file__).resolve()
FORMAL=MS/'outputs/arx_selected_height_pair_20261120_01'
AUDIT=ROOT/'产物/高度细扫_20261120_01/400次正式结果核验.json'
OLD=MS/'outputs/arx_claro_continuous_mpc_20261120_01/trials'
CONTROL=ROOT.parent/'claro/docs/06_做配对验证/resources/S5_固定挡物配对01/试次/I0069/01_抓挡'
CASES=['I0015','I0018','I0038','I0044','I0063','I0119','I0130','I0069']
LIFT=.452
TASK=MS.parent/'curobo/source/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml'


def prerequisite():
    p=read(FORMAL/'progress.json');a=read(AUDIT);end=read(FORMAL/'运行结束.json')
    assert p['phase']=='complete' and p['completed']==400 and a['passed']
    assert a['source_registration_sha256']==sha(FORMAL/'registration.json')
    assert end['all_owned_services_exited'] and not end['failed']
    fields=dict(s.split('=',1) for s in subprocess.check_output(['systemctl','--user','show',
        'arx-height-grasp-pair-20261120-02.service','-p','MainPID','-p','ExecMainStatus'],text=True).splitlines())
    assert fields['MainPID']=='0' and fields['ExecMainStatus']=='0'
    assert not any((FORMAL/n).exists() for n in ['STOP.json','控制器故障.json','调度故障.json'])


def prepare(out,gpu):
    prerequisite()
    paths=[SELF,Path(__file__).with_name('公共.py'),Path(__file__).with_name('比较升降高度.py'),
           Path(__file__).with_name('机器人几何.py'),MODEL,URDF,TCP,TASK,MS/'assets/arx/robot.urdf',
           MS/'assets/arx/manifest.json',MS/'resident_planner.py',
           MS.parent/'arx_core/scripts/curobo_arx/optimize_path.py',AUDIT,
           FORMAL/'registration.json',FORMAL/'progress.json',FORMAL/'运行结束.json']
    records=[]
    for case in CASES:
        d=CONTROL if case=='I0069' else OLD/case/'01_抓挡'
        s=read(d/'state.json');pool=read(d/'graspcl/candidate_pool.json')
        paths += [d/'state.json',d/'graspcl/candidate_pool.json']
        paths += [Path(g['file_path']) for g in s.get('scene_meshes',{}).values()]
        records.append({'case':case,'source':str(d),'candidates':len(pool['candidates']),
            'role':'historical_success_control' if case=='I0069' else 'historical_initial_planning_failure'})
    out.mkdir(parents=True,exist_ok=False);(out/'代码快照').mkdir()
    for p in paths:
        if p.suffix=='.py':shutil.copy2(p,out/'代码快照'/p.name)
    original=ET.parse(URDF);root=original.getroot();sim=ET.parse(MS/'assets/arx/robot.urdf').getroot()
    original_links={x.get('name') for x in root.findall('link')}
    assert 'body4' in original_links and 'base_link' not in original_links
    ET.SubElement(root,'link',name='base_link');ET.SubElement(root,'link',name='joint4_frame')
    for name in ['joint4','joint4_offset']:
        element=next(j for j in sim.findall('joint') if j.get('name')==name)
        root.append(deepcopy(element))
    arm_elements={x.get('name'):ET.tostring(x) for x in ET.parse(URDF).getroot()}
    extended={x.get('name'):ET.tostring(x) for x in root}
    assert all(extended[k]==v for k,v in arm_elements.items())
    derived=out/'含原升降关节.urdf';original.write(derived,encoding='unicode')
    cfg=yaml.safe_load(MODEL.read_text());cfg['robot_cfg']['kinematics']['urdf_path']=str(URDF)
    (out/'六关节.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    seven=deepcopy(cfg);kin=seven['robot_cfg']['kinematics'];kin['urdf_path']=str(derived);kin['base_link']='base_link'
    c=kin['cspace'];c['joint_names']=['joint4']+c['joint_names']
    c['default_joint_position']=[LIFT]+c['default_joint_position']
    for key in ['cspace_distance_weight','null_space_weight']:c[key]=[1.]+c[key]
    # These are conservative trajectory planning caps, not edits to the physical robot.
    count=len(c['joint_names']);c['max_acceleration']=[.3]+[float(c['max_acceleration'])]*(count-1)
    c['max_jerk']=[1.5]+[float(c['max_jerk'])]*(count-1)
    assert kin['collision_spheres']==cfg['robot_cfg']['kinematics']['collision_spheres']
    (out/'七关节.yml').write_text(yaml.safe_dump(seven,sort_keys=False))
    paths += [derived,out/'六关节.yml',out/'七关节.yml']
    reg={'created_at':now(),'gpu_uuid':gpu,'records':records,'cases':8,'comparisons':16,
        'source_sha256':{str(p.resolve()):sha(p) for p in set(paths)},'fixed_start_lift_m':LIFT,
        'joint_names_7':['joint4']+JOINTS,'lift_limits_m':[0.,.46000000834465027],
        'lift_velocity_limit_m_s':.15000000596046448,'lift_planning_acceleration_cap_m_s2':.3,
        'lift_planning_jerk_cap_m_s3':1.5,'candidate_budget':'Original candidate count and original order; no new candidates.',
        'per_case_branch_wall_budget_s':90,'max_ik_seeds':32,'return_ik_seeds':4,
        'eef_position_tolerance_m':.002,'orientation_tolerance_rad':.03,
        'endpoint_protocol':{'tcp_m':.043,'pregrasp_m':.08,'nominal_lift_world_z_m':.03},
        'motion_policy':'Joint lift+arm to H; hold chosen lift for H-to-G and G-to-L. Right-arm FK and dense collision checks retained.',
        'scope':'Offline integration pilot using archived observations; no simulator, physical grasps or new perception. Same original right-arm collision sphere coverage; torso and left-arm swept geometry are NOT represented, so successful plans are NOT execution approval.',
        'prerequisite':str(AUDIT),'physical_attempts':0,'robot_motion_commands':0,'automatic_retries':False,
        'user_authorization':'Try including the lift joint, only after the original 400-case batch has completed and exited.',
        'frozen':'Robot base/table/objects, all original right-arm joints and spheres, TCP43, finger measurements, native candidate poses/order, scene geometry and existing collision policy.',
        'new_configuration':'A separate planner-only chain adds the unchanged existing joint4; originals and physical model are not edited.'}
    save(out/'登记.json',reg)
    save(out/'模型扩展核验.json',{'passed':True,'original_arm_elements_identical':True,
        'lift_chain_copied_from_simulator_urdf':True,'original_collision_spheres_identical':True,
        'physical_robot_configuration_edits':0,'physical_attempts':0})
    print('PREPARED',8,'cases',sum(r['candidates'] for r in records),'frozen candidates',flush=True)


def run(out):
    prerequisite();reg=read(out/'登记.json')
    for p,h in reg['source_sha256'].items():assert sha(p)==h,p
    os.environ['CUDA_VISIBLE_DEVICES']=reg['gpu_uuid']
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS.parent/'arx_core/scripts/curobo_arx')]
    import torch
    from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    from curobo.types import GoalToolPose,JointState
    from curobo._src.types.robot import RobotCfg
    from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
    from resident_planner import all_kinematics_params,collision_scene_config
    from optimize_path import optimize_path,flush_planning_artifacts
    from 机器人几何 import RobotGeometry
    geometry=RobotGeometry(np.array(read(TCP)['T_eef_action']),device='cuda')
    profile=np.array(read(TCP)['T_eef_action']);sim=ET.parse(MS/'assets/arx/robot.urdf').getroot()
    joint=next(j for j in sim.findall('joint') if j.get('name')=='joint4')
    origin=np.eye(4);origin[:3,3]=np.fromstring(joint.find('origin').get('xyz'),sep=' ')
    assert joint.find('axis').get('xyz')=='0.0 0.0 1.0'
    model_loads={};fk_checks=[]

    def scene(state,base):
        inv=np.linalg.inv(base);result={'cuboid':{},'mesh':{}}
        def pose(t):
            q=Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
            return [*t[:3,3].tolist(),*q.tolist()]
        for key,g in state['scene_cuboids'].items():
            t=np.eye(4);t[:3,3]=g['center'];result['cuboid'][key]={'pose':pose(inv@t),'dims':g['dims']}
        for key,g in state.get('scene_meshes',{}).items():
            if g['target']:continue
            result['mesh'][key]={'pose':pose(inv@np.array(g['world'])),'file_path':g['file_path']}
        return collision_scene_config(result)

    class Planner:
        def __init__(self,dof,state):
            self.dof=dof;self.names=(['joint4'] if dof==7 else [])+JOINTS
            self.cfg=yaml.safe_load((out/('七关节.yml' if dof==7 else '六关节.yml')).read_text())
            self.path=out/('七关节.yml' if dof==7 else '六关节.yml')
            self.base=np.array(state['links']['base_link' if dof==7 else 'body4'])
            task=yaml.safe_load(TASK.read_text());task['rollout']['cost_cfg']['cspace_cfg']['weight'][0]*=100
            config=MotionPlannerCfg.create(robot=str(self.path),scene_model=scene(state,self.base),
                self_collision_check=False,collision_cache={'mesh':8,'primitive':8},interpolation_dt=.025,
                optimizer_collision_activation_distance=.003,position_tolerance=.002,orientation_tolerance=.03,
                trajopt_optimizer_configs=[task])
            for rollout in [config.graph_planner_config.rollout_config,
                config.ik_solver_config.core_cfg.metrics_rollout_config,config.trajopt_solver_config.core_cfg.metrics_rollout_config]:
                for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
            self.planner=MotionPlanner(config)
            self.checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(self.path),
                scene_model=scene(state,self.base),n_meshes=8,n_cuboids=8,collision_activation_distance=0.))
            self.checker.self_collision_cost=None
            assert self.planner.joint_names==self.names,(self.planner.joint_names,self.names)
            self.parameters=all_kinematics_params(self.planner,self.checker)
            self.limits=torch.tensor(np.stack([np.r_[0.,geometry.lower],np.r_[reg['lift_limits_m'][1],geometry.upper]],-1)
                if dof==7 else np.stack([geometry.lower,geometry.upper],-1),device='cuda',dtype=torch.float32)
            model_loads[str(dof)]={'planner':1,'checker':1}
            self.bind(state)

        def bind(self,state):
            self.state=state;self.base=np.array(state['links']['base_link' if self.dof==7 else 'body4'])
            self.inv=np.linalg.inv(self.base)
            cfg=deepcopy(self.cfg['robot_cfg']);measured=dict(zip(state['joint_names'],state['qpos']))
            for n in ['right_joint27','right_joint28']:cfg['kinematics']['lock_joints'][n]=measured[n]
            params=RobotCfg.create(cfg,self.planner.device_cfg).kinematics.kinematics_config
            for old in self.parameters:
                assert old.joint_names==params.joint_names and old.num_dof==params.num_dof
                old.copy_(params)
            world=scene(state,self.base)
            for obj in [self.planner,self.checker]:obj.clear_scene_cache();obj.update_world(world)
            self.planner.reset_seed()

        def js(self,q):return JointState.from_position(q[None].clone(),joint_names=self.names)

        def measured(self):
            q=torch.tensor([self.state['qpos'][self.state['joint_names'].index(n)] for n in self.names],device='cuda',dtype=torch.float32)
            return torch.maximum(torch.minimum(q,self.limits[:,1]-1e-4),self.limits[:,0]+1e-4)

        def fk(self,q):
            kin=self.planner.compute_kinematics(JointState.from_position(q,joint_names=self.names)).tool_poses
            xyz=kin.position.detach().cpu().numpy().reshape(-1,3)
            quat=kin.quaternion.detach().cpu().numpy().reshape(-1,4)
            r=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
            t=np.repeat(np.eye(4)[None],len(q),axis=0);t[:,:3,:3]=self.base[:3,:3]@r
            t[:,:3,3]=xyz@self.base[:3,:3].T+self.base[:3,3]
            return t

        def independent_fk(self,q):
            a=q[:,1:] if self.dof==7 else q
            tcp,r=geometry.fk(a,jacobian=False);tcp=tcp.cpu().numpy();r=r.cpu().numpy()@profile[:3,:3].T
            eef=tcp-.043*r[:,:,2]
            if self.dof==7:
                body=np.repeat(origin[None],len(q),axis=0);body[:,2,3]+=q[:,0].cpu().numpy()
                world=np.array(self.state['links']['base_link'])@body
            else:world=np.repeat(self.base[None],len(q),axis=0)
            t=np.repeat(np.eye(4)[None],len(q),axis=0);t[:,:3,:3]=world[:,:3,:3]@r
            t[:,:3,3]=(world[:,:3,:3]@eef[:,:,None]).squeeze(-1)+world[:,:3,3]
            return t

        def check_fk(self,q,goals=None):
            native=self.fk(q);independent=self.independent_fk(q)
            pos=float(np.linalg.norm(native[:,:3,3]-independent[:,:3,3],axis=1).max())
            rot=float(Rotation.from_matrix(native[:,:3,:3]@independent[:,:3,:3].transpose(0,2,1)).magnitude().max())
            assert pos<2e-5 and rot<2e-4,(pos,rot)
            if goals is not None:
                assert np.linalg.norm(independent[:,:3,3]-goals[:,:3,3],axis=1).max()<.00205
                assert Rotation.from_matrix(independent[:,:3,:3]@goals[:,:3,:3].transpose(0,2,1)).magnitude().max()<.0301
            fk_checks.append({'dof':self.dof,'samples':len(q),'position_m':pos,'rotation_rad':rot})

        def ik(self,world,start):
            local=self.inv@world;quat=Rotation.from_matrix(local[:3,:3]).as_quat()[[3,0,1,2]]
            goal=GoalToolPose(tool_frames=['right_eef_link'],
                position=torch.tensor(local[:3,3],device='cuda',dtype=torch.float32).reshape(1,1,1,1,3),
                quaternion=torch.tensor(quat,device='cuda',dtype=torch.float32).reshape(1,1,1,1,4))
            result=self.planner.ik_solver.solve_pose(goal,current_state=self.js(start),return_seeds=4)
            q=result.solution.reshape(-1,self.dof);ok=result.success.reshape(-1)
            ok=ok&self.checker.validate(q[:,None]).reshape(-1)
            ok=ok&((q>=self.limits[:,0]-1e-6)&(q<=self.limits[:,1]+1e-6)).all(-1)
            q=q[ok].clone()
            if len(q):
                self.check_fk(q,np.repeat(world[None],len(q),axis=0))
                q=q[(q-start).square().sum(-1).argsort()]
            return q

        def path_to(self,start,goal):
            result=self.planner.graph_planner.find_path(start[None].clone(),goal[None].clone(),
                interpolate_waypoints=True,interpolation_steps=201,interpolation_type=TrajInterpolationType.LINEAR,
                validate_interpolated_trajectory=True)
            if not result.success.all():return None
            q=result.interpolated_waypoints.reshape(-1,self.dof).clone()
            assert (q[0]-start).abs().max()<1e-4 and (q[-1]-goal).abs().max()<1e-4
            if not self.validate_dense(q):return None
            return q

        def validate_dense(self,q):
            a=torch.arange(10,device='cuda')/10
            dense=torch.cat([(q[:-1,None]*(1-a[None,:,None])+q[1:,None]*a[None,:,None]).reshape(-1,self.dof),q[-1:]])
            return all(bool(self.checker.validate(c[:,None]).all()) for c in dense.split(64))

        def optimize(self,path,directory,label):
            q,times=optimize_path(self.planner,self.checker,path,self.names,self.limits,6.,directory,label)
            flush_planning_artifacts();times=np.asarray(times)
            caps=[np.r_[.15,[1.5]*6],np.r_[.3,[5.]*6],np.r_[1.5,[100.]*6]] if self.dof==7 else [np.full(6,1.5),np.full(6,5.),np.full(6,100.)]
            dt=times[1]-times[0];v=q.detach().cpu().numpy();scale=1.
            for order,cap in enumerate(caps,1):
                v=np.diff(v,axis=0)/dt
                if v.size:scale=max(scale,float(np.max(np.abs(v)/cap))**(1/order))
            times*=scale
            assert times[-1]<=120,'Time-scaled path exceeds original 120-second cap'
            assert self.validate_dense(q)
            self.check_fk(q)
            if self.dof==7:
                assert abs(float(q[0,0]-path[0,0]))<.0005 and abs(float(q[-1,0]-path[-1,0]))<.0005
            save(directory/(label+'_指令.json'),{'joint_names':self.names,'q':q.cpu().numpy().tolist(),
                'times_s':times.tolist(),'additional_per_joint_time_scale':scale,'executed':False})
            return q

    reference=read(CONTROL/'state.json');old=reference['qpos'][reference['joint_names'].index('joint4')]
    warm=shifted(reference,LIFT,old);started=time.perf_counter()
    six=Planner(6,warm);seven=Planner(7,warm)
    # Exercise the actual prismatic kinematics at multiple legal heights.
    for model in [six,seven]:
        current=model.measured();q=current.repeat(5,1)
        if model.dof==7:q[:,0]=torch.tensor([.01,.12,.22,.35,.452],device='cuda')
        model.check_fk(q)
        if model.dof==7:
            poses=model.fk(q);assert np.max(np.abs(poses[:,:3,:3]-poses[0,:3,:3]))<2e-6
            assert np.max(np.abs((poses[:,2,3]-poses[0,2,3])-(q[:,0]-q[0,0]).cpu().numpy()))<2e-6
        model.check_fk(current[None],np.array(warm['links']['right_eef_link'])[None])
        assert model.checker.validate(current[None,None]).all(),'Warmup start collision'
        goal=model.fk(current[None])[0];assert len(model.ik(goal,current))
    save(out/'组件就绪.json',{'passed':True,'model_loads':model_loads,'fk_checks':fk_checks,
        'startup_seconds':time.perf_counter()-started,'physical_attempts':0})
    print('READY 6DOF+7DOF exact FK, legal lift range, static model checks',flush=True)
    results=[]
    for record in reg['records']:
        source=Path(record['source']);original_state=read(source/'state.json')
        old_lift=original_state['qpos'][original_state['joint_names'].index('joint4')]
        state=shifted(original_state,LIFT,old_lift);pool=read(source/'graspcl/candidate_pool.json')['candidates']
        for mode in ['fixed6','joint7']:
            directory=out/'规划'/record['case']/mode;directory.mkdir(parents=True,exist_ok=False)
            save(directory/'state.json',state);begun=time.perf_counter();rejected=[];chosen=None;planned=False
            six.bind(state);seven.bind(state);model=seven if mode=='joint7' else six
            start=model.measured()
            if not bool(model.checker.validate(start[None,None]).all()):
                rejected.append({'reason':'measured_start_collision'})
            else:
                for index,candidate in enumerate(pool):
                    if time.perf_counter()-begun>reg['per_case_branch_wall_budget_s']:
                        rejected.append({'reason':'offline_planning_budget_exceeded'});break
                    g=np.array(candidate['pose']);g[:3,3]-=.043*g[:3,2]
                    h=g.copy();h[:3,3]-=.08*g[:3,2];l=g.copy();l[2,3]+=.03
                    qgs=model.ik(g,start)
                    if not len(qgs):rejected.append({'index':index,'reason':'grasp_ik'});continue
                    # Return all four original IK branches, preserving G exactly.
                    for branch,qg_model in enumerate(qgs):
                        height=float(qg_model[0]) if mode=='joint7' else LIFT
                        adjusted=shifted(original_state,height,old_lift);six.bind(adjusted)
                        qg=qg_model[1:].clone() if mode=='joint7' else qg_model.clone()
                        qs_h=six.ik(h,qg);qs_l=six.ik(l,qg)
                        if not len(qs_h) or not len(qs_l):
                            rejected.append({'index':index,'branch':branch,'reason':'pre_or_lift_ik'});continue
                        qh,ql=qs_h[0],qs_l[0]
                        reach_model=seven if mode=='joint7' else six
                        qh_reach=torch.cat([qg_model[:1],qh]) if mode=='joint7' else qh
                        reach=reach_model.path_to(start,qh_reach)
                        approach=six.path_to(qh,qg);lift=six.path_to(qg,ql)
                        if any(p is None for p in [reach,approach,lift]):
                            rejected.append({'index':index,'branch':branch,'reason':'path'});continue
                        chosen={'candidate_index':index,'ik_branch':branch,'lift_m':height,
                            'contact_pose_world':candidate['pose'],'grasp_world':g.tolist(),'pregrasp_world':h.tolist(),
                            'lift_world':l.tolist(),'q_grasp_6':qg.tolist(),'q_pre_6':qh.tolist(),'q_lift_6':ql.tolist()}
                        pathdir=directory/'轨迹';pathdir.mkdir(exist_ok=False)
                        try:
                            reach_model.optimize(reach,pathdir,'到预抓取')
                            six.optimize(approach,pathdir,'最后接近')
                            six.optimize(lift,pathdir,'名义抬升')
                            planned=True
                        except RuntimeError as error:
                            rejected.append({'index':index,'reason':'trajectory_optimization_failed','detail':str(error)})
                        break
                    if chosen is not None:break
            row={'case':record['case'],'mode':mode,'initial_lift_m':LIFT,'success':planned,
                'selected':chosen,'rejected':rejected,'wall_seconds':time.perf_counter()-begun,
                'scope':reg['scope'],'physical_attempts':0}
            save(directory/'结果.json',row);results.append(row)
            save(out/'进度.json',{'phase':'离线规划','completed':len(results),'registered':16,
                'groups':{m:{'completed':sum(r['mode']==m for r in results),'planned':sum(r['mode']==m and r['success'] for r in results)} for m in ['fixed6','joint7']},
                'latest':{'case':record['case'],'mode':mode,'success':planned},'physical_attempts':0})
            print('PLAN',record['case'],mode,planned,'height',chosen['lift_m'] if chosen else None,
                'seconds',round(row['wall_seconds'],2),flush=True)
    for p,h in reg['source_sha256'].items():assert sha(p)==h,p
    save(out/'最终结果.json',{'passed':True,'rows':results,'model_loads':model_loads,
        'fk_checks':fk_checks,'source_files_unchanged':True,'physical_attempts':0,'scope':reg['scope'],
        'wall_seconds':time.perf_counter()-started})
    save(out/'进度.json',{'phase':'完成','completed':16,'registered':16,
        'groups':{m:{'completed':8,'planned':sum(r['mode']==m and r['success'] for r in results)} for m in ['fixed6','joint7']},'physical_attempts':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--gpu')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out,args.gpu)
    else:
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Offline service terminated')))
        try:run(out)
        except BaseException:
            save(out/'故障.json',{'time':now(),'error':traceback.format_exc(),'physical_attempts':0});raise
