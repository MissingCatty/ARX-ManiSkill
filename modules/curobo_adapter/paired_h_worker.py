"""Resident old/new cuRobo worker for a matched, offline H-entry comparison."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
import curobo
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams

HERE=Path(__file__).resolve().parent
MS=HERE.parent/'maniskill';ARX=HERE.parent/'arx_core'
sys.path[:0]=[str(MS),str(ARX/'scripts/curobo_arx')]
from resident_backend import read_message,save
from gripper_update import WarmedGripperUpdate
from optimize_path import optimize_path,flush_planning_artifacts
from entry_selector import select_pregrasp

NAMES=[f'right_joint{i}' for i in range(21,27)]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def parameter_views(*roots):
    from types import SimpleNamespace
    todo=list(roots);seen=set();found=[]
    while todo:
        obj=todo.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        if isinstance(obj,KinematicsParams):found.append(SimpleNamespace(config=SimpleNamespace(kinematics_config=obj)))
        elif isinstance(obj,dict):todo.extend(obj.values())
        elif isinstance(obj,(list,tuple)):todo.extend(obj)
        elif type(obj).__module__.startswith('curobo.') and hasattr(obj,'__dict__'):todo.extend(vars(obj).values())
        assert len(seen)<100000
    assert found
    return found


class Planner:
    def __init__(self,root,reg,backend):
        self.root=root;self.reg=reg;self.backend=backend;self.calls=0;self.loads={'open':1}
        root.mkdir(parents=True,exist_ok=False);self.out=root/'warmup';self.out.mkdir()
        warm=Path(reg['warmup_input']);self.state=read(warm/'state.json');self.base=np.asarray(self.state['links']['body4']);self.invbase=np.linalg.inv(self.base)
        for name in ['arx.yml','scene.json']:
            shutil.copyfile(warm/'plan'/name,self.out/name)
        self.model=self.out/'arx.yml';self.cfg=yaml.safe_load(self.model.read_text());self.scene=read(self.out/'scene.json')
        package=Path(curobo.__file__).parent
        assert package.is_relative_to(Path(reg['version_sources'][backend])),str(package)
        task=yaml.safe_load((package/'content/configs/task/trajopt/lbfgs_bspline_trajopt.yml').read_text())
        task['rollout']['cost_cfg']['cspace_cfg']['weight'][0]*=100
        kwargs=dict(robot=str(self.model),scene_model=self.scene,self_collision_check=False,
                    collision_cache={'mesh':8,'primitive':8},interpolation_dt=.025,
                    optimizer_collision_activation_distance=.003,position_tolerance=.002,orientation_tolerance=.03,
                    random_seed=123,use_cuda_graph=True,num_ik_seeds=32,
                    ik_optimizer_configs=['ik/lbfgs_ik.yml'],trajopt_optimizer_configs=[task])
        if backend=='new':kwargs['enable_timeopt']=False
        cfg=MotionPlannerCfg.create(**kwargs)
        graph=cfg.prm_graph_planner_config if backend=='new' else cfg.graph_planner_config
        for rollout in [graph.rollout_config,cfg.ik_solver_config.core_cfg.metrics_rollout_config,cfg.trajopt_solver_config.core_cfg.metrics_rollout_config]:
            for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
        self.planner=MotionPlanner(cfg)
        self.checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(self.model),scene_model=self.scene,n_meshes=8,n_cuboids=8,collision_activation_distance=0.))
        self.checker.self_collision_cost=None
        self.limits=torch.tensor([self.state['joint_limits'][self.state['joint_names'].index(n)] for n in NAMES],device='cuda',dtype=torch.float32)
        self.views=parameter_views(self.planner,self.checker)
        self.updater=WarmedGripperUpdate(self.model,self.planner.kinematics)
        self.configuration={'backend':backend,'version':importlib.metadata.version('nvidia-curobo'),
            'module':curobo.__file__,'model_load_count':self.loads,'num_seeds':cfg.ik_solver_config.num_seeds,
            'seed_orientation_weight':getattr(cfg.ik_solver_config,'seed_orientation_weight',None),
            'use_lm_seed':getattr(cfg.ik_solver_config,'use_lm_seed',None),
            'position_tolerance':cfg.ik_solver_config.position_tolerance,'orientation_tolerance':cfg.ik_solver_config.orientation_tolerance,
            'random_seed':123,'cache_capacity':{'mesh':8,'primitive':8},'model_sha256':digest(self.model),
            'torch':torch.__version__,'warp':importlib.metadata.version('warp-lang'),
            'task_configuration_scope':'Version-native L-BFGS IK and B-spline YAMLs; shared public settings matched; private version defaults may differ',
            'parameter_views':len(self.views),'runtime_robot_rebuilds':0}
        save(root/'configuration.json',self.configuration)
        self.case='__warmup__';self.index=None
        start=self.measured(self.state);self.check_state(self.state)
        if self.ik(np.asarray(self.state['links']['right_eef_link']),start) is None:raise RuntimeError('Resident warmup IK failed')
        path=self.path(start,start)
        if path is None:raise RuntimeError('Resident warmup PRM failed')
        self.optimize('warmup',path,6.)
        self.planner.reset_seed()

    def js(self,q):return JointState.from_position(q[None].clone(),joint_names=NAMES)
    def measured(self,state):
        q=torch.tensor([state['qpos'][state['joint_names'].index(n)] for n in NAMES],device='cuda',dtype=torch.float32)
        return torch.maximum(torch.minimum(q,self.limits[:,1]-1e-4),self.limits[:,0]+1e-4)

    def check_state(self,state):
        q=torch.tensor([state['qpos'][state['joint_names'].index(n)] for n in NAMES],device='cuda',dtype=torch.float32)
        kin=self.planner.compute_kinematics(self.js(q)).tool_poses
        xyz=kin.position.cpu().numpy().reshape(-1,3)[0];quat=kin.quaternion.cpu().numpy().reshape(-1,4)[0]
        expected=self.invbase@np.asarray(state['links']['right_eef_link']);rot=Rotation.from_quat([*quat[1:],quat[0]]).as_matrix()
        err={'position_m':float(np.linalg.norm(xyz-expected[:3,3])),'orientation_rad':float(Rotation.from_matrix(rot@expected[:3,:3].T).magnitude())}
        assert err['position_m']<2e-5 and err['orientation_rad']<2e-4,err
        assert self.checker.validate(self.measured(state)[None,None]).all(),'Original measured state collision'
        save(self.out/'FK_validation.json',err)
        kin=self.checker.get_kinematics(q[None,None]);np.savez(self.out/'effective_model.npz',q=q.cpu().numpy(),spheres=kin.robot_spheres.cpu().numpy(),bounds=self.limits.cpu().numpy())

    def bind(self,trial,out):
        state=read(trial/'state.json');cfg=yaml.safe_load((trial/'plan/arx.yml').read_text());expected=deepcopy(self.cfg)
        expected['robot_cfg']['kinematics']['lock_joints']=deepcopy(cfg['robot_cfg']['kinematics']['lock_joints'])
        assert expected==cfg,'Frozen robot property other than measured finger state changed'
        assert state['joint_names']==self.state['joint_names'] and state['joint_limits']==self.state['joint_limits']
        assert state['scene_cuboids']==self.state['scene_cuboids']
        self.out=out;out.mkdir(parents=True,exist_ok=False)
        for name in ['arx.yml','scene.json','nominal_closed.yml']:shutil.copyfile(trial/'plan'/name,out/name)
        self.base=np.asarray(state['links']['body4']);self.invbase=np.linalg.inv(self.base)
        values=dict(zip(state['joint_names'],state['qpos']));self.updater.apply(self.views,values)
        scene=read(trial/'plan/scene.json');content=deepcopy(scene)
        for geometry in state.get('scene_meshes',{}).values():assert digest(geometry['file_path'])==geometry['file_sha256']
        assert len(content.get('mesh',{}))<=8 and len(content['cuboid'])<=8
        if content.get('mesh'):content['mesh']={n+'_'+digest(g['file_path'])[:20]:g for n,g in content['mesh'].items()}
        self.planner.clear_scene_cache();self.planner.update_world(SceneCfg.create(content))
        self.checker.clear_scene_cache();self.checker.update_world(SceneCfg.create(content))
        self.planner.reset_seed();self.check_state(state)
        save(out/'resident_binding.json',{'model_load_count':1,'runtime_robot_rebuilds':0,'finger_state':{n:values[n] for n in ['right_joint27','right_joint28']},'source_model_sha256':digest(trial/'plan/arx.yml'),'source_scene_sha256':digest(trial/'plan/scene.json')})
        return state

    def ik(self,world,start):
        if self.calls>=self.reg['maximum_queries_per_version']:raise RuntimeError('Registered IK budget exhausted')
        t=self.invbase@world;quat=Rotation.from_matrix(t[:3,:3]).as_quat()
        goal=GoalToolPose(tool_frames=['right_eef_link'],position=torch.tensor(t[:3,3],device='cuda',dtype=torch.float32).reshape(1,1,1,1,3),quaternion=torch.tensor([quat[3],*quat[:3]],device='cuda',dtype=torch.float32).reshape(1,1,1,1,4))
        request={'request_id':self.calls,'case':self.case,'index':self.index,'H_world':world.tolist(),'current_q':start.tolist()}
        with (self.root/'requests.jsonl').open('a') as f:f.write(json.dumps(request)+'\n')
        self.calls+=1;torch.cuda.synchronize();before=time.perf_counter()
        result=self.planner.ik_solver.solve_pose(goal,current_state=self.js(start),return_seeds=4)
        q=None;native=bool(result.success.any())
        if native:
            qs=result.solution[result.success].reshape(-1,6);qs=qs[self.checker.validate(qs[:,None]).flatten()]
            if len(qs):q=qs[(qs-start).square().sum(-1).argmin()].clone()
        torch.cuda.synchronize();elapsed=time.perf_counter()-before
        receipt={**request,'accepted':q is not None,'native_returned_success':native,'q':None if q is None else q.tolist(),'wall_seconds':elapsed}
        if q is not None:
            kin=self.planner.compute_kinematics(self.js(q)).tool_poses;p=kin.position.cpu().numpy().reshape(-1,3)[0];w=kin.quaternion.cpu().numpy().reshape(-1,4)[0];R=Rotation.from_quat([*w[1:],w[0]]).as_matrix()
            pe=float(np.linalg.norm(p-t[:3,3]));re=float(Rotation.from_matrix(R@t[:3,:3].T).magnitude());within=bool(((q>=self.limits[:,0])&(q<=self.limits[:,1])).all());valid=bool(self.checker.validate(q[None,None]).all())
            assert pe<.002 and re<.03 and within and valid
            receipt['independent']={'position_error_m':pe,'orientation_error_rad':re,'joint_limits':within,'collision_valid':valid}
        with (self.root/'calls.jsonl').open('a') as f:f.write(json.dumps(receipt,allow_nan=False)+'\n')
        return q

    def path(self,start,goal):
        if self.backend=='new':from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType
        else:from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
        result=self.planner.graph_planner.find_path(start[None].clone(),goal[None].clone(),interpolate_waypoints=True,interpolation_steps=201,interpolation_type=TrajInterpolationType.LINEAR,validate_interpolated_trajectory=True)
        if not result.success.all():return None
        q=result.interpolated_waypoints.reshape(-1,6).clone()
        assert (q[0]-start).abs().max()<1e-4 and (q[-1]-goal).abs().max()<1e-4
        a=torch.arange(5,device='cuda')/5
        dense=torch.cat([(q[:-1,None]*(1-a[None,:,None])+q[1:,None]*a[None,:,None]).reshape(-1,6),q[-1:]])
        if not all(self.checker.validate(chunk[:,None]).all() for chunk in dense.split(64)):return None
        return q

    def optimize(self,label,path,seconds):
        q,times=optimize_path(self.planner,self.checker,path.clone(),NAMES,self.limits,seconds,self.out,label)
        flush_planning_artifacts()
        return {'q':q.cpu().numpy().tolist(),'times':np.asarray(times).tolist()}

    def case_request(self,case):
        self.case=case['id'];trial=Path(self.reg['source_campaign'])/'trials'/self.case
        out=self.root.parent.parent/'results'/self.backend/self.case;out.mkdir(parents=True,exist_ok=False)
        state=self.bind(trial,out/'model');rows=np.load(trial/'postprocessed_grasps.npy',allow_pickle=False);rows=rows[np.argsort(-rows[:,0],kind='stable')]
        solutions={};before=self.calls;start=self.measured(state);began=time.perf_counter()
        from grasp_selector import decode_candidate
        for index in case['indices']:
            self.index=index;world=decode_candidate(rows[index],state['camera_world_optical'])[1];q=self.ik(world,start)
            solutions[index]=None if q is None else q.clone()
        try:
            plan=select_pregrasp(self,state,rows,np.asarray(state['camera_world_optical']),solutions=solutions,max_path_attempts=8)
        except RuntimeError as error:
            if str(error)!='Trajectory optimizer rejected all attempts; no PRM-only fallback':raise
            plan={'success':False,'reason':'trajectory_optimizer_rejected','other_paths_not_exhausted':True,'planning_scope':'pregrasp_entry_only','full_grasp_validated':False}
        save(out/'entry_plan.json',plan)
        endpoint=None
        if plan['success']:
            q=torch.tensor(plan['paths']['pregrasp']['q'][-1],device='cuda',dtype=torch.float32);target=np.asarray(plan['selected']['pregrasp_world']);kin=self.checker.get_kinematics(q[None,None]);p=kin.tool_poses.position.cpu().numpy().reshape(-1,3)[0];w=kin.tool_poses.quaternion.cpu().numpy().reshape(-1,4)[0];T=self.invbase@target
            endpoint={'position_error_m':float(np.linalg.norm(p-T[:3,3])),'orientation_error_rad':float(Rotation.from_matrix(Rotation.from_quat([*w[1:],w[0]]).as_matrix()@T[:3,:3].T).magnitude()),'collision_valid':bool(self.checker.validate(q[None,None]).all())}
            assert endpoint['position_error_m']<.002 and endpoint['orientation_error_rad']<.03 and endpoint['collision_valid']
        result={'id':self.case,'backend':self.backend,'control':case['control'],'registered_H':len(case['indices']),
            'accepted_H_indices':[i for i,q in solutions.items() if q is not None],
            'entry_path_passed':bool(plan['success']),'selected_index':plan.get('selected',{}).get('index'),
            'entry_failure_reason':plan.get('reason'),'endpoint':endpoint,'native_queries':self.calls-before,
            'wall_seconds':time.perf_counter()-began,'model_load_count':1,'runtime_robot_rebuilds':0,
            'physical_trials':0,'cloud_requests':0}
        save(out/'result.json',result);print(json.dumps(result),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--backend',choices=['old','new'],required=True);args=parser.parse_args();root=args.campaign.resolve();reg=read(root/'registration.json');backend=args.backend
    for path,h in reg['source_hashes'].items():assert digest(path)==h,path
    started=time.monotonic();worker=Planner(root/'resident'/backend,reg,backend)
    ready={'status':'ready','kind':backend,'pid':os.getpid(),'ready_at':datetime.now().astimezone().isoformat(),'startup_seconds':time.monotonic()-started,'details':{'model_load_count':worker.loads,'version':worker.configuration['version']},'requests_completed':0}
    sock=Path(reg['sockets'][backend]);sock.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    assert not sock.exists()
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(sock));os.chmod(sock,0o600);server.listen(4);save(root/'resident'/(backend+'_ready.json'),ready)
            while True:
                with server.accept()[0] as connection:
                    with connection.makefile('rwb') as stream:
                        request=read_message(stream);command=request.get('command')
                        if command=='shutdown':stream.write(b'{"status":"stopped"}\n');stream.flush();break
                        if command=='health':response=ready
                        elif command=='case':
                            try:
                                for path,h in reg['runtime_hashes'].items():assert digest(path)==h,path
                                response={'status':'ok','result':worker.case_request(request['case'])}
                            except Exception:
                                import traceback
                                response={'status':'error','error':traceback.format_exc()}
                            ready['requests_completed']+=1;save(root/'resident'/(backend+'_ready.json'),ready)
                        else:response={'status':'error','error':'Unknown paired H command'}
                        stream.write((json.dumps(response)+'\n').encode());stream.flush()
    finally:
        flush_planning_artifacts();worker.planner.destroy()
        if sock.exists():sock.unlink()
        save(root/'resident'/(backend+'_stopped.json'),{'pid':os.getpid(),'queries':worker.calls,'model_load_count':1})


if __name__=='__main__':main()
