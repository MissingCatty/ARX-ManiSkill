"""ARX-model API validation only: no mailbox calls and no physical commands."""
import copy
import argparse
import json
import sys
from pathlib import Path

import torch
from curobo.model_predictive_control import ModelPredictiveControl,ModelPredictiveControlCfg
from curobo.types import JointState

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from build_model import build

p=argparse.ArgumentParser();p.add_argument('--name',default='mpc_api_preflight');args=p.parse_args()
out=ROOT/'runs/arx_dynamic_pick'/args.name;out.mkdir(exist_ok=False)
snap=json.loads((ROOT/'runs/weight_comparison_20261015/01_rubiks_cube_w1/01_rubiks_cube/capture_snapshot.json').read_text())
build(ROOT.parent/'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd',snap,out/'model',fine_gripper=True)
cfg=ModelPredictiveControlCfg.create(robot=str(out/'model/arx.yml'),scene_model={'cuboid':{'far_box':{'dims':[.1,.12,.18],'pose':[3.,3.,3.,1.,0.,0.,0.]}}},
    self_collision_check=False,optimization_dt=.2,interpolation_steps=4,num_control_points=12,
    position_tolerance=.002,orientation_tolerance=.03,optimizer_collision_activation_distance=.03,
    warm_start_optimization_num_iters=100,cold_start_optimization_num_iters=300)
for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
    for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
mpc=ModelPredictiveControl(cfg)
q=torch.tensor([[snap['joint_positions'][snap['joint_names'].index(n)] for n in mpc.joint_names]],device='cuda')
limits=torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in mpc.joint_names],device='cuda')
q=torch.maximum(torch.minimum(q,limits[:,1]-1e-4),limits[:,0]+1e-4)
js=JointState.from_position(q,joint_names=mpc.joint_names);js.velocity=torch.zeros_like(q);js.acceleration=torch.zeros_like(q)
mpc.setup(js)
mpc.update_goal_tool_poses(mpc.compute_kinematics(js).tool_poses.as_goal(mpc.tool_frames),run_ik=False)
mpc.update_goal_state(js);mpc.enable_joint_position_tracking()
r=mpc.optimize_action_sequence(js)
feasible=mpc.trajectory_execution_manager.get_current_metrics().costs_and_constraints.get_feasible(include_all_hybrid=False,sum_horizon=False)
value={'scope':'ARX-model API only; no full scene, no physical grasp, no collision-avoidance success claim',
    'joint_names':mpc.joint_names,'sequence_shape':list(r.action_sequence.position.shape),
    'feasible_shape':list(feasible.shape),'prefix_feasible':bool(feasible[:,:8].all()),
    'derivative_fields':{k:None if getattr(r.action_sequence,k) is None else list(getattr(r.action_sequence,k).shape) for k in ('velocity','acceleration','jerk')},
    'action_dt':r.action_dt,'cold_solve_s':r.solve_time}
collection=mpc.trajectory_execution_manager.get_current_metrics().costs_and_constraints
value['prefix_constraint_max']={k:float(v[:,:8].max()) for k,v in zip(collection.constraints.names,collection.constraints.values)}
assert r.action_sequence.position.shape[1]==4
for k in ('velocity','acceleration','jerk'):assert getattr(r.action_sequence,k) is not None
(out/'report.json').write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))
assert value['prefix_feasible'],'Unchanged original MPC constraints reject stationary ARX state'
