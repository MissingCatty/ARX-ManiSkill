"""Isolated H001 copy of the baseline: lower a fixed proposal, retain all gates."""
import argparse,copy,json,time,os,sys,random

random.seed(0)
from pathlib import Path
import numpy as np
import torch,yaml
from scipy.spatial.transform import Rotation
from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
from curobo.types import JointState,GoalToolPose
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
from client import Client
from build_model import build
from gripper_update import WarmedGripperUpdate
from settling import assess_settling
from optimize_path import optimize_path

np.random.seed(0)
torch.manual_seed(0)

parser=argparse.ArgumentParser();parser.add_argument('--resume-lift',action='store_true');args=parser.parse_args()
target=os.environ.get('GRASP_TARGET','red_cube')
assert target in ('red_cube','blue_cylinder','green_sphere','test_object'),target
target_path='/World/Task/'+target
warmup_started=time.monotonic()
stage_timing={}
stage_convergence={}
closed_update_seconds=None
use_trajopt=os.environ.get('GRASP_PLANNER','prm')=='prm_trajopt'
weight_multiplier=float(os.environ.get('GRASP_BOUND_WEIGHT_MULTIPLIER','100'))
assert weight_multiplier>0 and np.isfinite(weight_multiplier)
planner_label='cuRobo IK + PRM + trajectory optimization' if use_trajopt else 'cuRobo IK + PRM'
root=Path(__file__).resolve().parents[2];out=root/'runs/arx_grasp';client=Client(root/'runs/arx_curobo/mailbox')
(out/'grasp_report.json').write_text(json.dumps({'target':target,'success':False,'status':'running'},indent=2))
def record_failure(kind,error,traceback):
    report_path=out/'grasp_report.json'
    report=json.loads(report_path.read_text())
    report.update(success=False,status='failed',error=str(error),stage_timing=stage_timing,stage_convergence=stage_convergence,planner=planner_label,bound_weight_multiplier=weight_multiplier if use_trajopt else None)
    report_path.write_text(json.dumps(report,indent=2))
    sys.__excepthook__(kind,error,traceback)
sys.excepthook=record_failure
snap=client.request('snapshot');names=snap['right_arm'];base=np.array(snap['link_world']['body4']);invbase=np.linalg.inv(base)
target_path=snap['objects'][target]['path']
scene=yaml.safe_load((out/'scene.yml').read_text());sources=json.loads((out/'scene_sources.json').read_text())
assert json.loads((out/'input_report.json').read_text()).get('target','red_cube')==target,'Input target mismatch'
# Only the target object may be touched. The table and other objects remain obstacles.
scene['mesh']={k:v for k,v in scene['mesh'].items() if not (sources[k]==target_path or sources[k].startswith(target_path+'/'))}
robot=str(out/('model_closed/arx.yml' if args.resume_lift else 'model/arx.yml'))
def initialize(robot,table_clearance=0.0):
    planning_scene=copy.deepcopy(scene)
    if table_clearance:
        for key,mesh in planning_scene['mesh'].items():
            if 'breakfast_table' in sources[key]:
                mesh['pose'][:3]=(np.array(mesh['pose'][:3])+invbase[:3,2]*table_clearance).tolist()
    extra={}
    if use_trajopt:
        task=yaml.safe_load((root.parent/'curobo/source/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml').read_text())
        task['rollout']['cost_cfg']['cspace_cfg']['weight'][0]*=weight_multiplier
        extra['trajopt_optimizer_configs']=[task]
        (out/'trajopt_effective_config.json').write_text(json.dumps(task,indent=2))
    config=MotionPlannerCfg.create(**extra,robot=robot,scene_model=planning_scene,self_collision_check=False,
        collision_cache={'mesh':len(scene['mesh'])+1,'primitive':8},interpolation_dt=.025,
        optimizer_collision_activation_distance=.003,position_tolerance=.002,orientation_tolerance=.03)
    for rollout in [config.graph_planner_config.rollout_config,config.ik_solver_config.core_cfg.metrics_rollout_config,config.trajopt_solver_config.core_cfg.metrics_rollout_config]:
        for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
    planner=MotionPlanner(config)
    planner.warmup(enable_graph=True,num_warmup_iterations=1)
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=robot,scene_model=planning_scene,n_meshes=len(scene['mesh'])+1,collision_activation_distance=0.0));checker.self_collision_cost=None
    return planner,checker
planner,checker=initialize(robot)
open_planner,open_checker=planner,checker
if not args.resume_lift:
    # Compile both fixed-shape planners before sending any motion command.
    nominal_closed=copy.deepcopy(snap)
    for name in ['right_joint27','right_joint28']:
        nominal_closed['joint_positions'][nominal_closed['joint_names'].index(name)]=.022
    build(root.parent/'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd',nominal_closed,out/'model_closed',fine_gripper=True,gripper_pitch=.004)
    closed_planner,closed_checker=initialize(str(out/'model_closed/arx.yml'),table_clearance=.008)
    # Resolve the pinned cuRobo API before allowing any robot motion.
    closed_rollouts=closed_planner.ik_solver.get_all_rollout_instances()+closed_planner.trajopt_solver.get_all_rollout_instances()+closed_planner.graph_planner.get_all_rollout_instances()
    closed_models=[rollout.transition_model.robot_model for rollout in closed_rollouts]+[closed_checker.kinematics]
    for model in closed_models:
        assert model.config.kinematics_config.fixed_transforms.shape==closed_planner.kinematics.config.kinematics_config.fixed_transforms.shape
    gripper_update=WarmedGripperUpdate(out/'model_closed/arx.yml',closed_planner.kinematics)
    print('READY: open and closed cuRobo planners warmed',flush=True)
warmup_seconds=time.monotonic()-warmup_started
limits=torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in names],device='cuda')
def measured():
    s=client.request('snapshot');q=torch.tensor([s['joint_positions'][s['joint_names'].index(n)] for n in names],device='cuda')
    return torch.maximum(torch.minimum(q,limits[:,1]-1e-4),limits[:,0]+1e-4)
def js(q):return JointState.from_position(q[None],joint_names=names)
ik_diagnostics={"no_ik_solution":0,"ik_solution_in_world_collision":0}
def ik(world,start):
    t=invbase@world;quat=Rotation.from_matrix(t[:3,:3]).as_quat();quat=np.r_[quat[3],quat[:3]]
    goal=GoalToolPose(tool_frames=['right_eef_link'],position=torch.tensor(t[:3,3],dtype=torch.float32,device='cuda').reshape(1,1,1,1,3),quaternion=torch.tensor(quat,dtype=torch.float32,device='cuda').reshape(1,1,1,1,4))
    r=planner.ik_solver.solve_pose(goal,current_state=js(start),return_seeds=4)
    if not r.success.any():ik_diagnostics["no_ik_solution"]+=1;return None
    qs=r.solution[r.success].reshape(-1,len(names));qs=qs[checker.validate(qs[:,None]).flatten()]
    if not len(qs):ik_diagnostics["ik_solution_in_world_collision"]+=1;return None
    return qs[(qs-start).square().sum(-1).argmin()].clone()
def path(start,goal):
    r=planner.graph_planner.find_path(start[None].clone(),goal[None].clone(),interpolate_waypoints=True,
        interpolation_steps=201,interpolation_type=TrajInterpolationType.LINEAR,validate_interpolated_trajectory=True)
    if not r.success.all():return None
    q=r.interpolated_waypoints.reshape(-1,len(names)).clone()
    a=torch.arange(5,device='cuda')/5
    dense=torch.cat([(q[:-1,None]*(1-a[None,:,None])+q[1:,None]*a[None,:,None]).reshape(-1,len(names)),q[-1:]])
    if not all(checker.validate(chunk[:,None]).all() for chunk in dense.split(64)):return None
    return q

def execute(label,q,gripper=-3.2,duration=6):
    times=np.linspace(0,duration,len(q))
    if use_trajopt and label in ('pregrasp','approach','lift'):
        print('OPTIMIZING',label,'bound multiplier',weight_multiplier,flush=True)
        q,times=optimize_path(planner,checker,q,names,limits,duration,out,label)
        print('OPTIMIZED',label,'samples',len(q),'duration',float(times[-1]),flush=True)
    command={'joint_names':names,'position':q.tolist(),'time_from_start':times.tolist(),'gripper_target':gripper,'position_tolerance_rad':.005}
    (out/f'{label}_command.json').write_text(json.dumps(command))
    print('EXECUTING',label,'wall',time.time(),flush=True)
    r=client.request('trajectory',timeout=300,**command);(out/f'{label}_execution.json').write_text(json.dumps(r))
    stage_timing[label]={k:r[k] for k in ('wall_duration_s','sim_duration_s') if k in r}
    goal_key={'pregrasp':'pregrasp_world','approach':'grasp_world','close':'grasp_world','lift':'lift_world','hold':'lift_world'}[label]
    assessment=assess_settling(r,selected[goal_key])
    stage_convergence[label]={**assessment,'settled':assessment['accepted']}
    (out/f'{label}_settling.json').write_text(json.dumps(assessment,indent=2))
    external=[c for c in r['contacts'] if c['num_contacts']>0 and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])]
    unwanted=[c for c in external if not (target_path in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27','right_link28')))]
    print(label,'settled',r['settled'],'external events',len(external),'unwanted',len(unwanted),flush=True)
    assert not unwanted,f'Unexpected contact during {label}'
    # Keep raw execution.settled unchanged on disk; report the separate task-level gate.
    return {**r,'settled':assessment['accepted']}

if args.resume_lift:
    selected=json.loads((out/'selected_grasp.json').read_text())
    ql=torch.tensor(selected['q_lift'],device='cuda')
    initial=json.loads((out/'capture_snapshot.json').read_text())
    cube=np.array(initial['objects'][target]['world'])[:3,3]
else:
    start=measured();assert checker.validate(start[None,None]).all(),'ARX initial state collides'
    cam=np.array(json.loads((out/'camera_world_optical.json').read_text()));cube=np.array(snap['objects'][target]['world'])[:3,3]
    frozen=os.environ.get('GRASP_FIXED_SELECTION')
    if frozen:
        selected=json.loads(Path(frozen).read_text())
        # H001: fixed proposal and orientation, change only world-Z contact depth.
        # Baseline remains immutable; all original IK, collision, optimization,
        # execution and hold checks below are retained.
        shift = float(os.environ['GRASP_RESEARCH_DEPTH_SHIFT_M'])
        assert np.isfinite(shift) and -.012 <= shift <= 0.0
        original = copy.deepcopy(selected)
        for key in ('pregrasp_world', 'grasp_world', 'lift_world'):
            selected[key][2][3] += shift
        selected['target_contact_world'][2] += shift
        qg_new = ik(np.asarray(selected['grasp_world']), start)
        assert qg_new is not None, 'Depth variant grasp IK failed'
        qp_new = ik(np.asarray(selected['pregrasp_world']), qg_new)
        ql_new = ik(np.asarray(selected['lift_world']), qg_new)
        assert qp_new is not None and ql_new is not None, 'Depth variant pre/lift IK failed'
        for key, value in [('q_pre', qp_new), ('q_grasp', qg_new), ('q_lift', ql_new)]:
            selected[key] = value.tolist()
        selected['research_change'] = {'hypothesis': os.environ.get('GRASP_RESEARCH_HYPOTHESIS', 'depth_ablation'), 'world_z_shift_m': shift,
            'source_selected_grasp': frozen, 'original_contact_world': original['target_contact_world'],
            'method_scope': 'engineering ablation; not VFAS or GNP reproduction'}
        (out/'research_variant.json').write_text(json.dumps(selected['research_change'], indent=2))
        qp,qg,ql=[torch.tensor(selected[k],dtype=torch.float32,device='cuda') for k in ('q_pre','q_grasp','q_lift')]
        assert checker.validate(torch.stack([qp,qg])[:,None]).all(),'Frozen open-gripper goals invalid'
        assert closed_checker.validate(torch.stack([qg,ql])[:,None]).all(),'Frozen closed-gripper goals invalid'
        route=path(start,qp)
        assert route is not None,'Frozen pregrasp PRM failed'
        (out/'fixed_trial_state.json').write_text(json.dumps({'source':frozen,'seed':0,'initial_arm':start.tolist(),'target_world':snap['objects'][target]['world'],'base_world':snap['link_world']['body4']},indent=2))
    else:
        raise RuntimeError('Research depth policy requires a paired fixed proposal')
    (out/'selected_grasp.json').write_text(json.dumps(selected,indent=2));print('SELECTED',json.dumps(selected),flush=True)
    if os.environ.get('GRASP_VIDEO_DIR'):
        client.request('grasp_video_start',directory=os.environ['GRASP_VIDEO_DIR'])
    execution_started=time.monotonic()
    r=execute('pregrasp',route);assert r['settled'],'Pregrasp did not converge'
    q=measured();route=path(q,qg);assert route is not None,'Final approach planning failed'
    r=execute('approach',route,duration=5);assert r['settled'],'Grasp approach did not converge'
    q=measured();r=execute('close',torch.stack([q,qg]),gripper=.1,duration=3)
    assert r['settled'],'Closed grasp did not converge; lift was not attempted'
    finger_contacts={n for event in r['contacts'] if event['num_contacts']>0 and target_path in event['a']+event['b'] for n in ('right_link27','right_link28') if n in event['a']+event['b']}
    assert len(finger_contacts)==2,'No bilateral target contact after closure; lift was not attempted'
    # Apply the precomputed gripper update at the measured closing width.
    update_start=time.monotonic()
    closed=client.request('snapshot')
    (out/'closed_snapshot.json').write_text(json.dumps(closed,indent=2))
    cfg=yaml.safe_load((out/'model_closed/arx.yml').read_text())
    joints=dict(zip(closed['joint_names'],closed['joint_positions']))
    for name in cfg['robot_cfg']['kinematics']['lock_joints']:
        cfg['robot_cfg']['kinematics']['lock_joints'][name]=joints[name]
    (out/'model_closed/arx.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    planner,checker=closed_planner,closed_checker
    rollouts=planner.ik_solver.get_all_rollout_instances()+planner.trajopt_solver.get_all_rollout_instances()+planner.graph_planner.get_all_rollout_instances()
    models=[rollout.transition_model.robot_model for rollout in rollouts]+[checker.kinematics]
    gripper_update.apply(models,joints)
    planner.graph_planner.reset_buffer()
    check_state=js(measured())
    expected=checker.kinematics.compute_kinematics(check_state).robot_spheres
    for model in models:
        assert torch.allclose(model.compute_kinematics(check_state).robot_spheres,expected,atol=1e-6),'Closed gripper model update mismatch'
    closed_update_seconds=time.monotonic()-update_start
    print('Closed model updated without recompilation in',closed_update_seconds,'seconds',flush=True)
q=measured();route=path(q,ql);assert route is not None,'Lift planning failed'
r=execute('lift',route,gripper=.1,duration=6)
q=measured();hold=execute('hold',torch.stack([q,ql]),gripper=.1,duration=3)
heights=[np.array(rec['objects'][target])[2,3] for rec in hold['records']]
object_held=bool(min(heights)>cube[2]+.06)
result={'target':target,'success':object_held and all(s['settled'] for s in stage_convergence.values()),'status':'completed','object_held':object_held,'stage_convergence':stage_convergence,'initial_object_z_m':float(cube[2]),'held_object_min_z_m':float(min(heights)),
        'held_object_final_z_m':float(heights[-1]),'lift_m':float(heights[-1]-cube[2]),
        'planner':planner_label,'bound_weight_multiplier':weight_multiplier if use_trajopt else None,'prediction':'ZeroGrasp synthetic checkpoint','self_collision_enabled':False,
        'selected_grasp_index':selected['index'],'closed_model_update_seconds':closed_update_seconds,'warmup_seconds':warmup_seconds,'stage_timing':stage_timing,
        'execution_wall_seconds':time.monotonic()-execution_started if not args.resume_lift else None}
(out/'grasp_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
assert result['success'],'Object hold or trajectory convergence check failed'
planner.destroy()
if not args.resume_lift:
    open_planner.destroy()
