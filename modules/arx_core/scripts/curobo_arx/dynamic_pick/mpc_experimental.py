"""Live ARX/cuRobo MPC side task; archived ZeroGrasp goal, moving USD collider."""
import argparse
import ast
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import JointState, GoalToolPose, Pose
from curobo.config_io import load_yaml
from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType

ROOT=Path("/home/zhaoyechong/projects/COOLAS-ARX")
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from client import Client
from optimize_path import optimize_path
from settling import assess_settling
from gripper_update import WarmedGripperUpdate


def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='attempt_00')
    parser.add_argument('--speed',type=float,default=.04)
    parser.add_argument('--cross-time',type=float,default=5.)
    parser.add_argument('--warmup-only',action='store_true')
    parser.add_argument('--cached-controls',action='store_true',help='Reuse archived fully optimized reference paths; still check them against fresh ARX models.')
    parser.add_argument('--guard-pregrasp',action='store_true',help='Reactive avoidance at pregrasp; native precision grasp after crossing.')
    parser.add_argument('--mpc-approach-only',action='store_true',help='Full optimized global pregrasp, then reactive MPC for terminal approach.')
    args=parser.parse_args()
    torch.manual_seed(0);np.random.seed(0)
    task=ROOT/'runs/arx_dynamic_pick';out=task/args.name;out.mkdir(exist_ok=False)
    client=Client(task/'mailbox');snap=client.request('snapshot')
    names=snap['right_arm'];base=np.array(snap['link_world']['body4']);invbase=np.linalg.inv(base)
    selected=read(task/'selected_grasp.json');target=snap['objects']['test_object']['path']
    initial_z=np.array(snap['objects']['test_object']['world'])[2,3]
    limits=torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in names],device='cuda')
    scene=yaml.safe_load((task/'scene.yml').read_text());sources=read(task/'scene_sources.json')
    # Import the original functions through AST without running its top-level script.
    context=dict(copy=copy,yaml=yaml,json=json,np=np,torch=torch,MotionPlanner=MotionPlanner,
        MotionPlannerCfg=MotionPlannerCfg,RobotCollisionChecker=RobotCollisionChecker,
        RobotCollisionCheckerCfg=RobotCollisionCheckerCfg,JointState=JointState,
        TrajInterpolationType=TrajInterpolationType,scene=scene,sources=sources,root=ROOT,
        out=out,invbase=invbase,names=names,use_trajopt=True,weight_multiplier=1.)
    original=ast.parse((ROOT/'scripts/curobo_arx/grasp_execute.py').read_text())
    for name in ('initialize','js','path'):
        node=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'scripts/curobo_arx/grasp_execute.py'),'exec'),context)
    initialize=context['initialize'];js=context['js'];path=context['path']
    report={'success':False,'status':'warming_up','stages':{},'mpc_updates':0,
        'new_neural_inference':False,'grasp_source':read(task/'task_contract.json')['source'],
        'robot_configuration_changed':False,'requested_obstacle_speed_m_s':args.speed}
    write(out/'result.json',report)
    write(out/'initial_snapshot.json',snap)
    def measured(s=None):
        if s is None:s=read(client.root/'stream_state.json')
        q=torch.tensor([s['joint_positions'][s['joint_names'].index(n)] for n in names],device='cuda')
        # Same measured-state boundary projection as the original grasp executor.
        # Raw measured positions remain unchanged in every state/record on disk.
        return torch.maximum(torch.minimum(q,limits[:,1]-1e-4),limits[:,0]+1e-4)
    warm_start=time.monotonic()
    if args.cached_controls:
        print('WARMUP original ARX open/closed checkers; cached optimized references',flush=True)
        def make_checker(model,clearance):
            world=copy.deepcopy(scene)
            if clearance:
                for k,m in world['mesh'].items():
                    if 'breakfast_table' in sources[k]:m['pose'][:3]=(np.array(m['pose'][:3])+invbase[:3,2]*clearance).tolist()
            checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),scene_model=world,n_meshes=len(scene['mesh'])+1,collision_activation_distance=0.))
            checker.self_collision_cost=None
            return checker
        open_checker=make_checker(task/'model/arx.yml',0.)
        closed_checker=make_checker(task/'model_closed/arx.yml',.008)
        closed_models=[closed_checker.kinematics]
        open_planner=closed_planner=None
    else:
        print('WARMUP original open planner',flush=True)
        open_planner,open_checker=initialize(str(task/'model/arx.yml'))
        context.update(planner=open_planner,checker=open_checker)
        print('WARMUP original closed planner',flush=True)
        closed_planner,closed_checker=initialize(str(task/'model_closed/arx.yml'),table_clearance=.008)
        rollouts=closed_planner.ik_solver.get_all_rollout_instances()+closed_planner.trajopt_solver.get_all_rollout_instances()+closed_planner.graph_planner.get_all_rollout_instances()
        closed_models=[r.transition_model.robot_model for r in rollouts]+[closed_checker.kinematics]
    gripper_update=WarmedGripperUpdate(task/'model_closed/arx.yml',closed_models[0])
    start=measured(); qp,qg,ql=[torch.tensor(selected[k],device='cuda') for k in ('q_pre','q_grasp','q_lift')]
    assert open_checker.validate(torch.stack([start,qp,qg])[:,None]).all(),'Historical open goal invalid'
    assert closed_checker.validate(torch.stack([qg,ql])[:,None]).all(),'Historical closed goal invalid'
    # Obtain a globally feasible static guide; MPC will re-optimize its commands online.
    if args.cached_controls:
        reference=read(task/'attempt_00/static_pregrasp_guide.json')
        nominal=torch.tensor(reference['position'],device='cuda');times=np.asarray(reference['time_from_start'])
        source=Path(read(task/'task_contract.json')['source'])
        ac,lc=[read(source/(label+'_command.json')) for label in ('approach','lift')]
        approach=torch.tensor(ac['position'],device='cuda');approach_times=np.array(ac['time_from_start'])
        lift=torch.tensor(lc['position'],device='cuda');lift_times=np.array(lc['time_from_start'])
        assert float((nominal[0]-start).abs().max())<.01,'Cached reference initial state mismatch'
        for label,route,checker in [('pregrasp',nominal,open_checker),('approach',approach,open_checker),('lift',lift,closed_checker)]:
            dense=torch.nn.functional.interpolate(route.T[None],size=10*(len(route)-1)+1,mode='linear',align_corners=True).transpose(1,2).reshape(-1,len(names)).contiguous()
            assert all(checker.validate(chunk[:,None]).all() for chunk in dense.split(64)),label+' cached path invalid in current full model'
        write(out/'cached_reference_provenance.json',{'pregrasp':str(task/'attempt_00/static_pregrasp_guide.json'),'terminal_paths':str(source),'full_mesh_and_joint_bounds_revalidated':True,'new_trajectory_optimization':False})
    else:
        nominal=path(start,qp);assert nominal is not None,'Static pregrasp guide failed'
        nominal,times=optimize_path(open_planner,open_checker,nominal,names,limits,6.,out,'static_pregrasp_guide')
    write(out/'static_pregrasp_guide.json',{'position':nominal.tolist(),'time_from_start':times.tolist()})
    if not args.cached_controls:
        approach=path(qp,qg);assert approach is not None
        approach,approach_times=optimize_path(open_planner,open_checker,approach,names,limits,5.,out,'static_approach_guide')
        context.update(planner=closed_planner,checker=closed_checker)
        lift=path(qg,ql);assert lift is not None,'Closed lift route failed'
        lift,lift_times=optimize_path(closed_planner,closed_checker,lift,names,limits,6.,out,'lift')
    # The only new environment object: a 10 x 12 x 18 cm physical cuboid.
    dims=np.array([.10,.12,.18])
    crossing=np.array([3.96,2.17,.88]);velocity=np.array([args.speed,0.,0.])
    obstacle_start=crossing-velocity*args.cross_time
    R=Rotation.from_matrix(invbase[:3,:3]).as_quat();quat=np.r_[R[3],R[:3]]
    pose=lambda world:Pose.from_numpy((invbase@np.r_[world,1])[:3],quat)
    local=(invbase@np.r_[obstacle_start,1])[:3]
    mpc_scene=copy.deepcopy(scene)
    mpc_scene['cuboid']={'moving_obstacle':{'dims':dims.tolist(),'pose':[*local,*quat]}}
    prediction_horizons=(.4,.8,1.2)
    for i,horizon in enumerate(prediction_horizons):
        future_local=(invbase@np.r_[obstacle_start+velocity*horizon,1])[:3]
        mpc_scene['cuboid'][f'moving_obstacle_prediction_{i}']={'dims':dims.tolist(),'pose':[*future_local,*quat]}
    mpc_optimizer=load_yaml(str(ROOT.parent/'curobo/source/curobo/content/configs/task/mpc/lbfgs_mpc.yml'))
    mpc_optimizer['optimizer']['inner_iters']=4
    mpc_optimizer['rollout']['cost_cfg']['cspace_cfg']['squared_l2_regularization_weight'][1]=1000.
    mpc_optimizer['rollout']['constraint_cfg']['scene_collision_cfg']['weight']=100000.
    write(out/'mpc_effective_optimizer.json',{'optimizer':mpc_optimizer,'prediction_horizons_s':prediction_horizons,
        'physical_obstacles':1,'prediction_only_cuboids':3,'robot_and_physics_unchanged':True})
    cfg=ModelPredictiveControlCfg.create(robot=str(task/'model/arx.yml'),scene_model=mpc_scene,optimizer_configs=[mpc_optimizer],
        self_collision_check=False,collision_cache={'mesh':len(scene['mesh'])+1,'primitive':8},
        optimization_dt=.05,interpolation_steps=4,num_control_points=12,
        position_tolerance=.002,orientation_tolerance=.03,
        optimizer_collision_activation_distance=.03,
        warm_start_optimization_num_iters=8,cold_start_optimization_num_iters=40)
    for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
    print('WARMUP cuRobo MPC (original ARX model)',flush=True)
    mpc=ModelPredictiveControl(cfg)
    state=js(qp if args.mpc_approach_only else start);state.velocity=torch.zeros_like(state.position);state.acceleration=torch.zeros_like(state.position)
    mpc.setup(state)
    for _ in range(3):mpc.optimize_action_sequence(state)
    report['warmup_seconds']=time.monotonic()-warm_start;report['status']='ready'
    write(out/'result.json',report)
    print('READY ALL WARMUPS',report['warmup_seconds'],flush=True)
    if args.warmup_only:return
    static_checker=open_checker
    recording=False;streaming=False
    samples=[];updates=[]
    def atomic_command(s,q,ts):
        command={'id':uuid.uuid4().hex,'joint_names':names,'position':q.tolist(),
            'time_from_start':ts.tolist(),'sim_time':s['sim_time']}
        temp=client.root/'stream_command.tmp';temp.write_text(json.dumps(command));temp.replace(client.root/'stream_command.json')
        return command['id']
    def dynamic_clearance(q,s,ts):
        spheres=mpc.kinematics.compute_kinematics(JointState.from_position(q,joint_names=names)).robot_spheres
        spheres=spheres.reshape(q.shape[0],-1,4)
        p=spheres[..., :3];r=spheres[...,3]
        rot=torch.tensor(base[:3,:3],device='cuda',dtype=torch.float32)
        origin=torch.tensor(base[:3,3],device='cuda',dtype=torch.float32)
        world=p@rot.T+origin
        center=torch.tensor(s['moving_obstacle']['center'],device='cuda',dtype=torch.float32)
        vel=torch.tensor(velocity,device='cuda',dtype=torch.float32)
        delta=torch.abs(world-center-ts[:,None,None]*vel)-torch.tensor(dims/2,device='cuda',dtype=torch.float32)
        sd=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.amax(-1).clamp_max(0)-r
        return float(sd.min())
    def mpc_stage(label,guide,goal_world,goal_q,max_sim=45.):
        s=read(client.root/'stream_state.json');stage_start=s['sim_time'];last=-1.;goal_index=0;near=0;stage_records=[]
        wall_deadline=time.monotonic()+180.
        # Keep the already warmed optimizer/action buffer across stage changes.
        # reset_robot here would force a cold solve after starting the obstacle.
        guide_np=guide.cpu().numpy()
        previous_q=None;previous_t=None;filtered_velocity=torch.zeros((1,len(names)),device='cuda')
        while True:
            s=read(client.root/'stream_state.json')
            if time.monotonic()>wall_deadline:raise TimeoutError(label+' MPC wall-time budget exceeded')
            if time.time()-s['wall_time']>3.:raise RuntimeError('Live ARX state stopped updating')
            if s['sim_time']==last:
                time.sleep(.005);continue
            last=s['sim_time'];elapsed=last-stage_start
            if elapsed>max_sim:raise RuntimeError(label+' MPC did not converge within '+str(max_sim)+' simulation seconds')
            if time.time()-s['wall_time']>2.:raise RuntimeError('Stale live ARX state')
            q=measured(s)
            events_path=client.root/'stream_events.json'
            if events_path.exists():
                recent=read(events_path)['recent']
                def unexpected(e):
                    if e['num_contacts']<=0:return False
                    actors=e['a']+e['b']
                    if '/World/DynamicPickObstacle' in actors:return True
                    if not any(n in actors for n in ('/right_link','/right_eef_link')):return False
                    if '/World/Robot/' in e['a'] and '/World/Robot/' in e['b']:return False
                    return not (target in actors and any(n in actors for n in ('right_link27','right_link28')))
                forbidden=[e for e in recent if unexpected(e)]
                if forbidden:raise RuntimeError('Measured unexpected contact during '+label)
            error=float((q-goal_q).abs().max())
            stage_records.append({'time':elapsed,'position':q.tolist(),'target':goal_q.tolist(),'ee_world':s['link_world']['right_eef_link']})
            samples.append({'stage':label,**s})
            # Track a forward point along the global guide; measured feedback selects progress.
            distances=np.max(np.abs(guide_np-q.cpu().numpy()),axis=1)
            nearest=int(np.argmin(distances))
            goal_index=max(goal_index,nearest)
            far=np.flatnonzero(np.max(np.abs(guide_np[goal_index:]-guide_np[goal_index]),axis=1)>.25)
            look=goal_index+int(far[0]) if len(far) else len(guide)-1
            guide_q=guide[look]
            kin=mpc.compute_kinematics(js(guide_q))
            mpc.update_goal_tool_poses(kin.tool_poses.as_goal(mpc.tool_frames),run_ik=False)
            mpc.update_goal_state(js(guide_q));mpc.enable_joint_position_tracking()
            mpc.scene_collision_checker.update_obstacle_pose('moving_obstacle',pose(np.array(s['moving_obstacle']['center'])))
            for i,horizon in enumerate(prediction_horizons):
                mpc.scene_collision_checker.update_obstacle_pose(f'moving_obstacle_prediction_{i}',pose(np.array(s['moving_obstacle']['center'])+velocity*horizon))
            # Estimate control-rate velocity from measured positions. Instantaneous
            # PhysX drive velocities contain substep oscillations (retained raw).
            if previous_q is not None:
                finite_velocity=(q-previous_q)[None]/(last-previous_t)
                filtered_velocity=.5*filtered_velocity+.5*finite_velocity
            previous_q=q.clone();previous_t=last
            state=js(q);state.velocity=filtered_velocity.clone()
            state.acceleration=torch.zeros_like(state.position)
            tick=time.monotonic();result=mpc.optimize_action_sequence(state);solve=time.monotonic()-tick
            seq=result.action_sequence.position.reshape(-1,len(names)).detach().clone()
            # MPC's four subcommands cover the next .2 seconds. Keep measured start.
            command_q=torch.cat([q[None],seq],dim=0)
            command_times=torch.arange(len(command_q),device='cuda')*.05
            valid=torch.isfinite(command_q).all() and not ((command_q<limits[:,0]-1e-6)|(command_q>limits[:,1]+1e-6)).any()
            dense=torch.cat([command_q[:-1,None]*(1-torch.arange(5,device='cuda')[None,:,None]/5)+command_q[1:,None]*(torch.arange(5,device='cuda')[None,:,None]/5)],dim=0).reshape(-1,len(names))
            dense=torch.cat([dense,command_q[-1:]])
            dts=torch.linspace(0,float(command_times[-1]),len(dense),device='cuda')
            static_valid=bool(static_checker.validate(dense[:,None]).all())
            clearance=dynamic_clearance(dense,s,dts)
            velmax=float((torch.diff(command_q,dim=0)/.05).abs().max())
            costs=mpc.trajectory_execution_manager.get_current_metrics().costs_and_constraints
            feasible=costs.get_feasible(include_all_hybrid=False,sum_horizon=False)
            mpc_prefix_feasible=bool(feasible[:, :8].all())
            cspace_values=[v for n,v in zip(costs.constraints.names,costs.constraints.values) if n=='cspace']
            cspace_prefix_feasible=bool(cspace_values and all((v[:,:8]<=0).all() for v in cspace_values))
            derivative_maxima={field:float(getattr(result.action_sequence,field).abs().max())
                for field in ('velocity','acceleration','jerk') if getattr(result.action_sequence,field) is not None}
            derivative_valid=all(k in derivative_maxima and derivative_maxima[k]<=v+1e-5
                for k,v in [('velocity',1.5),('acceleration',5.),('jerk',100.)])
            # Future occupancy replicas are optimization guidance, not extra
            # physical bodies at the current instant. Keep their raw metric,
            # and admit only with ORIGINAL cspace/dynamic limits plus the full
            # original static mesh check and time-indexed real moving-box check.
            admitted=bool(valid) and static_valid and clearance>=.002 and velmax<=1.5 and cspace_prefix_feasible and derivative_valid
            if admitted:
                cmd_id=atomic_command(s,command_q.cpu().numpy(),command_times.cpu().numpy())
            else:
                cmd_id=atomic_command(s,np.repeat(q.cpu().numpy()[None],2,axis=0),np.array([0.,.2]))
            update={'stage':label,'sim_time':last,'elapsed':elapsed,'solve_wall_s':solve,'admitted':admitted,
                'prefix_static_valid':static_valid,'prefix_dynamic_clearance_m':clearance,'velocity_max_rad_s':velmax,
                'mpc_prefix_feasible':mpc_prefix_feasible,'mpc_derivative_maxima':derivative_maxima,
                'cspace_prefix_feasible':cspace_prefix_feasible,'predictive_proxy_scene_is_guidance':True,
                'guide_index':goal_index,'lookahead_index':look,'endpoint_joint_error_rad':error,
                'command_id':cmd_id,'command_positions':command_q.tolist(),'obstacle_center':s['moving_obstacle']['center']}
            updates.append(update)
            with (out/'mpc_updates.jsonl').open('a') as f:f.write(json.dumps(update)+'\n')
            if len(updates)%10==0:print('MPC',label,round(elapsed,2),'error',round(error,4),'guide',goal_index,'admit',admitted,'clearance',round(clearance,4),'solve',round(solve,3),flush=True)
            future_clearance=None
            if label in ('approach','avoidance') and error<.01:
                # Do not commit to ordinary closure/lift while the moving box
                # would intersect those next stages. Continue reactive control.
                if label=='avoidance':
                    future_q=torch.cat([approach,qg[None].repeat(15,1),lift[1:],ql[None].repeat(15,1)])
                    at=float(approach_times[-1])
                    future_t=np.r_[approach_times,at+np.linspace(.2,3,15),at+3+lift_times[1:],at+3+lift_times[-1]+np.linspace(.2,3,15)]
                else:
                    future_q=torch.cat([qg[None].repeat(16,1),lift[1:],ql[None].repeat(15,1)])
                    future_t=np.r_[np.linspace(0,3,16),3+lift_times[1:],3+lift_times[-1]+np.linspace(.2,3,15)]
                future_clearance=dynamic_clearance(future_q,s,torch.tensor(future_t,device='cuda',dtype=torch.float32))
                with (out/'terminal_commit_checks.jsonl').open('a') as f:
                    f.write(json.dumps({'sim_time':last,'joint_error_rad':error,'future_close_lift_hold_clearance_m':future_clearance})+'\n')
            commit_safe=label not in ('approach','avoidance') or (future_clearance is not None and future_clearance>.015)
            near=near+1 if error<.003 and commit_safe else 0
            if near>=4:
                break
        r={'settled':True,'records':stage_records,'final':s}
        assessment=assess_settling(r,goal_world)
        write(out/(label+'_execution.json'),r);write(out/(label+'_settling.json'),assessment)
        assert assessment['accepted'],label+' original convergence gate failed'
        report['stages'][label]=assessment
        print('MPC COMPLETE',label,elapsed,flush=True)
    def execute(label,q,ts,gripper=None):
        command={'joint_names':names,'position':q.tolist(),'time_from_start':np.asarray(ts).tolist(),
            'position_tolerance_rad':.003}
        if gripper is not None:command['gripper_target']=gripper
        write(out/(label+'_command.json'),command)
        r=client.request('trajectory',timeout=120,**command)
        write(out/(label+'_execution.json'),r)
        key={'pregrasp':'pregrasp_world','approach':'grasp_world','close':'grasp_world','lift':'lift_world','hold':'lift_world'}[label]
        assessment=assess_settling(r,selected[key])
        write(out/(label+'_settling.json'),assessment);report['stages'][label]=assessment
        assert assessment['accepted'],label+' failed unchanged convergence gate'
        return r
    try:
        info=client.request('dynamic_obstacle',center=obstacle_start.tolist(),dims=dims.tolist(),velocity=velocity.tolist())
        write(out/'obstacle_spec.json',info)
        client.request('grasp_video_start',directory='/workspace/COOLAS-ARX/'+str((out/'frames').relative_to(ROOT)));recording=True
        report['status']='executing';write(out/'result.json',report)
        if args.mpc_approach_only:
            obstacle_state=read(client.root/'stream_state.json')
            # The crossing is timed for terminal approach. Verify the complete
            # preceding optimized trajectory against the moving box before use.
            clearance=dynamic_clearance(nominal,obstacle_state,torch.tensor(times,device='cuda',dtype=torch.float32)+.2)
            write(out/'global_pregrasp_dynamic_check.json',{'minimum_clearance_m':clearance,'full_reference_checked':True})
            assert clearance>.01,'Moving obstacle already blocks the global pregrasp'
            execute('pregrasp',nominal.cpu().numpy(),times)
        client.request('dynamic_stream_start',trace_path='/workspace/COOLAS-ARX/'+str((out/'live_state.jsonl').relative_to(ROOT)));streaming=True
        if not args.mpc_approach_only:mpc_stage('pregrasp',nominal,selected['pregrasp_world'],qp)
        if args.guard_pregrasp:
            mpc_stage('avoidance',torch.stack([qp,qp]),selected['pregrasp_world'],qp,max_sim=35.)
        else:
            mpc_stage('approach',approach,selected['grasp_world'],qg,max_sim=30.)
        stop=client.request('dynamic_stream_stop');streaming=False
        write(out/'stream_stop.json',stop)
        if args.guard_pregrasp:execute('approach',approach.cpu().numpy(),approach_times)
        q=measured(client.request('snapshot'))
        close=execute('close',torch.stack([q,qg]).cpu().numpy(),[0.,3.],gripper=.1)
        contact_fingers={n for e in close['contacts'] if e['num_contacts']>0 and target in e['a']+e['b'] for n in ('right_link27','right_link28') if n in e['a']+e['b']}
        assert len(contact_fingers)==2,'No bilateral closure; do not attempt lift'
        closed=client.request('snapshot');joints=dict(zip(closed['joint_names'],closed['joint_positions']))
        write(out/'closed_snapshot.json',closed);gripper_update.apply(closed_models,joints)
        dense_lift=torch.nn.functional.interpolate(lift.T[None],size=10*(len(lift)-1)+1,mode='linear',align_corners=True).transpose(1,2).reshape(-1,len(names)).contiguous()
        assert all(closed_checker.validate(chunk[:,None]).all() for chunk in dense_lift.split(64)),'Measured-closure lift path invalid'
        execute('lift',lift.cpu().numpy(),lift_times,gripper=.1)
        q=measured(client.request('snapshot'))
        hold=execute('hold',torch.stack([q,ql]).cpu().numpy(),[0.,3.],gripper=.1)
        heights=[np.asarray(r['objects']['test_object'])[2,3]-initial_z for r in hold['records']]
        all_contacts=client.request('dynamic_contacts')['contacts']
        write(out/'all_contacts.json',all_contacts)
        obstacle_contacts=[e for e in all_contacts if e['num_contacts']>0 and '/World/DynamicPickObstacle' in e['a']+e['b']]
        forbidden=[e for e in all_contacts if e['num_contacts']>0 and any(n in e['a']+e['b'] for n in ('/right_link','/right_eef_link')) and not ('/World/Robot/' in e['a'] and '/World/Robot/' in e['b']) and not (target in e['a']+e['b'] and any(n in e['a']+e['b'] for n in ('right_link27','right_link28')))]
        report.update(success=bool(min(heights)>.06 and not forbidden and not obstacle_contacts),
            status='completed',hold_min_lift_m=float(min(heights)),bilateral_closure=True,
            obstacle_contact_events=len(obstacle_contacts),forbidden_contact_events=len(forbidden))
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        if streaming:
            try:write(out/'stream_stop.json',client.request('dynamic_stream_stop'))
            except Exception as exc:report['stream_stop_error']=str(exc)
        if recording:
            try:report['recording']=client.request('grasp_video_stop')
            except Exception as exc:report['record_stop_error']=str(exc)
        try:client.request('dynamic_obstacle_stop')
        except Exception as exc:report['obstacle_stop_error']=str(exc)
        report['mpc_updates']=len(updates)
        if updates:report['mpc_solve_median_ms']=float(np.median([u['solve_wall_s'] for u in updates])*1000)
        write(out/'result.json',report);print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        name=sys.argv[sys.argv.index('--name')+1] if '--name' in sys.argv else 'attempt_00'
        result_path=ROOT/'runs/arx_dynamic_pick'/name/'result.json'
        if result_path.exists():
            value=read(result_path)
            if value.get('status')!='failed':
                value.update(status='failed',error=str(exc))
                write(result_path,value)
        raise
