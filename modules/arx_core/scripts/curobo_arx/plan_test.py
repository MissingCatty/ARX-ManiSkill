"""Choose a blocked ARX motion, plan around the obstacle, and save trajectories."""
import itertools,json,time
from pathlib import Path
import numpy as np,torch,yaml
from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
from curobo.types import JointState
from client import Client

root=Path(__file__).resolve().parents[2];out=root/'runs/arx_curobo';torch.manual_seed(0)
robot=str(out/'model/arx.yml'); scene=yaml.safe_load((out/'scene.yml').read_text())
client=Client(out/'mailbox');snap=client.request('snapshot');names=snap['right_arm'];d=dict(zip(snap['joint_names'],snap['joint_positions']))
limits=torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in names],device='cuda')
q0=torch.tensor([d[n] for n in names],device='cuda');q0=torch.maximum(torch.minimum(q0,limits[:,1]-1e-4),limits[:,0]+1e-4)
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=robot,scene_model=scene,n_meshes=len(scene['mesh'])+1,collision_activation_distance=0.0));checker.self_collision_cost=None
assert checker.validate(q0[None,None]).all(),'ARX start collides with the room or violates limits'
goals=torch.tensor([[a,b,c,e,0,0] for a,b,c,e in itertools.product([-.8,-.4,0,.4,.8],[.3,.7,1.1],[.3,.8,1.3],[-.5,0,.5])],device='cuda',dtype=torch.float32)
valid=checker.validate(goals[:,None]).flatten()
print('Valid goals',int(valid.sum()),'/',len(goals),flush=True)
t=torch.linspace(0,1,101,device='cuda'); selected=None;counts=[0,0,0];best=-100.0
for goal in goals[valid]:
    line=q0+t[:,None]*(goal-q0)
    if not checker.validate(line[:,None]).all():continue
    counts[0]+=1
    st=checker.get_kinematics(line[:,None]);pos=st.tool_poses.position.reshape(-1,3)
    if torch.linalg.vector_norm(pos[-1]-pos[0])<.25:continue
    counts[1]+=1
    center=pos[50];sph=st.robot_spheres[[0,-1],0];delta=(sph[...,:3]-center).abs()-.05
    clearance=delta.clamp_min(0).norm(dim=-1)+delta.max(dim=-1).values.clamp_max(0)-sph[...,3]
    best=max(best,float(clearance.min()))
    if clearance.min()<.01:continue
    selected=(goal,line,center);break
print('candidate counts',counts,'best endpoint box clearance',best,flush=True)
assert selected is not None,'Could not find a safe ARX start/goal with a blocked direct path'
goal,line,center=selected
scene['cuboid']={'validation_box':{'dims':[.1,.1,.1],'pose':center.tolist()+[1,0,0,0]}}
(out/'scene_obstacle.yml').write_text(yaml.safe_dump(scene))
print('Selected ARX goal',goal.tolist(),'box (body4 frame)',center.tolist(),flush=True)
config=MotionPlannerCfg.create(robot=robot,scene_model=scene,self_collision_check=False,
    collision_cache={'mesh':len(scene['mesh'])+1,'primitive':8},interpolation_dt=.025)
# This cuRobo revision omits self_collision_check in PRM and metrics rollouts.
rollouts=[config.graph_planner_config.rollout_config,
          config.ik_solver_config.core_cfg.metrics_rollout_config,
          config.trajopt_solver_config.core_cfg.metrics_rollout_config]
for rollout in rollouts:
    for cost in rollout.get_cost_manager_configs():
        cost.disable_self_collision()
planner=MotionPlanner(config)
planner.warmup(enable_graph=True,num_warmup_iterations=3)
js=lambda q:JointState.from_position(q[None],joint_names=names)
start=time.monotonic();result=planner.plan_cspace(js(goal),js(q0),max_attempts=10)
if result is not None:
    print('Planner success',result.success,'feasible',result.feasible,'cspace_error',result.cspace_error,flush=True)
method='trajectory_optimization'
if result is not None and result.success.all():
    traj=result.get_interpolated_plan().reorder(names).position.reshape(-1,len(names)).detach()
else:
    # cuRobo PRM is also a collision-aware planner; check its path independently
    # before slow simulation playback when time-optimal optimization fails.
    from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
    graph=planner.graph_planner.find_path(q0[None].clone(),goal[None].clone(),
        interpolate_waypoints=True,interpolation_steps=301,
        interpolation_type=TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory=True)
    print('PRM',graph.success,flush=True)
    assert graph.success.all(),'ARX cuRobo PRM also failed'
    traj=graph.interpolated_waypoints.reshape(-1,len(names)).detach()
    method='curobo_prm'

# Independent checker for the scene including the inserted obstacle.
check=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=robot,scene_model=scene,n_meshes=len(scene['mesh'])+1,collision_activation_distance=0.0));check.self_collision_cost=None
def dense(q):
    a=torch.arange(10,device='cuda')/10
    return torch.cat([(q[:-1,None]*(1-a[None,:,None])+q[1:,None]*a[None,:,None]).reshape(-1,len(names)),q[-1:]])
qd=dense(traj);blocked=~check.validate(line[:,None]);free=check.validate(qd[:,None])
assert free.all(),'Planned ARX trajectory collides on dense validation'
assert blocked.any(),'The baseline is not blocked'
base=np.asarray(snap['link_world']['body4']);center_world=(base@np.r_[center.cpu().numpy(),1])[:3]
# Rotate the body-frame cuboid into a world-axis-aligned box. ARX base yaw is -90deg
# in this scene, and a cubic obstacle is invariant under this rotation.
assert np.max(np.abs(np.abs(base[:3,:3])-np.round(np.abs(base[:3,:3]))))<1e-3,'Box export needs oriented-box support for this base pose'
report={'planner_method':method,'joint_names':names,'start':q0.tolist(),'goal':goal.tolist(),'obstacle_center_base':center.tolist(),
        'obstacle_center_world':center_world.tolist(),'obstacle_dims':[.1,.1,.1],
        'baseline_collision_samples':int(blocked.sum()),'baseline_samples':len(line),
        'planned_samples':len(traj),'dense_samples':len(qd),'planned_invalid_samples':int((~free).sum()),
        'planning_elapsed_s':time.monotonic()-start,'self_collision_enabled':False}
(out/'plan_report.json').write_text(json.dumps(report,indent=2))
# Slow playback gives the real simulated drives time to follow the path.
for label,q in [('planned',traj),('baseline',line)]:
    duration=max(8.0,float((q[-1]-q[0]).abs().max())*6)
    data={'joint_names':names,'position':q.tolist(),'time_from_start':np.linspace(0,duration,len(q)).tolist()}
    (out/f'{label}.json').write_text(json.dumps(data))
print(json.dumps(report,indent=2),flush=True)
planner.destroy()
