"""Benchmark archived ARX paths without issuing simulator or robot commands."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType

from optimize_path import optimize_path,flush_planning_artifacts,get_optimization_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=('interactive','first_valid','historical'),default='first_valid')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--full-request',action='store_true',help='Include fresh IK, PRM and the original PRM dense checks')
    args=parser.parse_args();source=args.source_run.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.environ['GRASP_TRAJOPT_POLICY']=args.policy
    snap=json.loads((source/'capture_snapshot.json').read_text())
    sources=json.loads((source/'scene_sources.json').read_text())
    target=snap['objects']['test_object']['path']
    scene=yaml.safe_load((source/'scene.yml').read_text())
    scene['mesh']={k:v for k,v in scene['mesh'].items() if not(sources[k]==target or sources[k].startswith(target+'/'))}
    task=json.loads((source/'trajopt_effective_config.json').read_text())
    names=snap['right_arm'];limits=torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in names],device='cuda')
    protected=[source/'scene.yml',source/'scene_sources.json',source/'capture_snapshot.json',source/'trajopt_effective_config.json']
    protected += [source/d/n for d in ('model','model_closed') for n in ('arx.yml','arx.urdf')]
    digest=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    before=digest();report={'scope':'ARX planning-model regression only; no new Isaac Sim execution or contact validation','source_run':str(source),'policy':args.policy,'gpu':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cases':[],'protected_before':before}
    dependency=Path(__file__).resolve().parents[3]/'curobo/source'
    report['swept_mesh_bounds']=os.environ.get('CUROBO_SWEPT_MESH_BOUNDS','0')=='1'
    report['request_kind']='IK + PRM + trajectory optimization + original independent dense checks' if args.full_request else 'Archived PRM seed + trajectory optimization + original independent dense checks'
    report['implementation_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('optimize_path.py'),dependency/'curobo/_src/geom/data/data_mesh.py',dependency/'curobo/_src/geom/collision/wp_sweep_collision_kernel.py')}
    report['gpu_state_before']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],text=True)
    def save():(out/'benchmark.json').write_text(json.dumps(report,indent=2))
    save()
    for model,stages in (('model',('pregrasp','approach')),('model_closed',('lift',))):
        planning_scene=copy.deepcopy(scene)
        if model=='model_closed':
            invbase=np.linalg.inv(np.array(snap['link_world']['body4']))
            for key,mesh in planning_scene['mesh'].items():
                if 'breakfast_table' in sources[key]:mesh['pose'][:3]=(np.array(mesh['pose'][:3])+invbase[:3,2]*.008).tolist()
        robot=str(source/model/'arx.yml')
        cfg=MotionPlannerCfg.create(robot=robot,scene_model=planning_scene,trajopt_optimizer_configs=[copy.deepcopy(task)],self_collision_check=False,collision_cache={'mesh':len(scene['mesh'])+1,'primitive':8},interpolation_dt=.025,optimizer_collision_activation_distance=.003,position_tolerance=.002,orientation_tolerance=.03)
        for rollout in (cfg.graph_planner_config.rollout_config,cfg.ik_solver_config.core_cfg.metrics_rollout_config,cfg.trajopt_solver_config.core_cfg.metrics_rollout_config):
            for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
        planner=MotionPlanner(cfg)
        checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=robot,scene_model=planning_scene,collision_activation_distance=0.,n_meshes=len(scene['mesh'])+1));checker.self_collision_cost=None
        try:
            for repeat in range(args.repeats):
                for stage in stages:
                    label=f'{stage}_{repeat}';path=torch.tensor(json.loads((source/(stage+'_prm_seed.json')).read_text()),device='cuda',dtype=torch.float32)
                    torch.cuda.synchronize();start=time.monotonic();phases={}
                    if args.full_request:
                        phase=time.monotonic();js=lambda q:JointState.from_position(q[None],joint_names=names)
                        goal=planner.compute_kinematics(js(path[-1])).tool_poses.as_goal()
                        ik=planner.ik_solver.solve_pose(goal,current_state=js(path[0]),return_seeds=4)
                        assert ik.success.any(),'ARX IK rejected the archived target'
                        candidates=ik.solution[ik.success].reshape(-1,len(names));candidates=candidates[checker.validate(candidates[:,None]).flatten()]
                        assert len(candidates),'No IK solution passes original scene checks'
                        target=candidates[(candidates-path[0]).square().sum(-1).argmin()].clone()
                        torch.cuda.synchronize();phases['ik_seconds']=time.monotonic()-phase;phase=time.monotonic()
                        route=planner.graph_planner.find_path(path[0:1].clone(),target[None].clone(),interpolate_waypoints=True,interpolation_steps=201,interpolation_type=TrajInterpolationType.LINEAR,validate_interpolated_trajectory=True)
                        assert route.success.all(),'ARX PRM rejected the IK target'
                        path=route.interpolated_waypoints.reshape(-1,len(names)).clone()
                        alpha=torch.arange(5,device='cuda')/5
                        dense=torch.cat([(path[:-1,None]*(1-alpha[None,:,None])+path[1:,None]*alpha[None,:,None]).reshape(-1,len(names)),path[-1:]])
                        assert all(checker.validate(chunk[:,None]).all() for chunk in dense.split(64)),'ARX PRM failed original independent dense checks'
                        torch.cuda.synchronize();phases['prm_plus_validation_seconds']=time.monotonic()-phase
                    phase=time.monotonic()
                    q,times=optimize_path(planner,checker,path,names,limits,5 if stage=='approach' else 6,out,label)
                    torch.cuda.synchronize()
                    phases['trajectory_plus_validation_seconds']=time.monotonic()-phase
                    row={'stage':stage,'repeat':repeat,'wall_seconds':time.monotonic()-start,'phase_seconds':phases,'samples':len(q),'execution_seconds':float(times[-1]),'optimization':get_optimization_report(out,label)}
                    np.savez(out/(label+'_trajectory.npz'),position=q.cpu().numpy(),time_from_start=times)
                    report['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='optimization'}),flush=True)
        finally:
            planner.destroy()
            flush_started=time.monotonic();flush_planning_artifacts()
            report.setdefault('artifact_flush_seconds',[]).append(time.monotonic()-flush_started)
            for case in report['cases']:
                label=f"{case['stage']}_{case['repeat']}"
                assert json.loads((out/(label+'_optimization.json')).read_text())==case['optimization'],'Persisted planning evidence differs from completed report'
    report['protected_after']=digest();report['protected_unchanged']=before==report['protected_after'];report['pass']=report['protected_unchanged'] and all(c['optimization']['success'] for c in report['cases']);save()
    return 0 if report['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
