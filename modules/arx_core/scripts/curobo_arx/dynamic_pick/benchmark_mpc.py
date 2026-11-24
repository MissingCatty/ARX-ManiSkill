"""Full living-room model timing/trajectory-grid check; zero physical commands."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from curobo.model_predictive_control import ModelPredictiveControl,ModelPredictiveControlCfg
from curobo.types import JointState
from curobo.config_io import load_yaml

root=Path(__file__).resolve().parents[3];task=root/'runs/arx_dynamic_pick'
scene=yaml.safe_load((task/'scene.yml').read_text())
scene['cuboid']={'moving_obstacle':{'dims':[.1,.12,.18],'pose':[3.,3.,3.,1.,0.,0.,0.]}}
opt=load_yaml(str(root.parent/'curobo/source/curobo/content/configs/task/mpc/lbfgs_mpc.yml'))
opt['optimizer']['inner_iters']=4
cfg=ModelPredictiveControlCfg.create(robot=str(task/'model/arx.yml'),scene_model=scene,optimizer_configs=[opt],
    self_collision_check=False,collision_cache={'mesh':len(scene['mesh'])+1,'primitive':8},
    optimization_dt=.05,interpolation_steps=4,num_control_points=12,
    position_tolerance=.002,orientation_tolerance=.03,optimizer_collision_activation_distance=.03,
    warm_start_optimization_num_iters=8,cold_start_optimization_num_iters=40)
for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
    for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
mpc=ModelPredictiveControl(cfg)
guide=torch.tensor(json.loads((task/'attempt_00/static_pregrasp_guide.json').read_text())['position'],device='cuda')
state=JointState.from_position(guide[:1].clone(),joint_names=mpc.joint_names)
state.velocity=torch.zeros_like(state.position);state.acceleration=torch.zeros_like(state.position)
mpc.setup(state)
goal=JointState.from_position(guide[15:16].clone(),joint_names=mpc.joint_names)
mpc.update_goal_tool_poses(mpc.compute_kinematics(goal).tool_poses.as_goal(mpc.tool_frames),run_ik=False)
mpc.update_goal_state(goal);mpc.enable_joint_position_tracking()
rows=[]
for i in range(12):
    start=time.monotonic();r=mpc.optimize_action_sequence(state);elapsed=time.monotonic()-start
    full=r.robot_state_sequence.joint_state
    rows.append({'iteration':i,'wall_s':elapsed,'solver_s':r.solve_time,
        'command_dt':torch.as_tensor(r.action_sequence.dt).tolist(),
        'full_dt':torch.as_tensor(full.dt).tolist(), 'full_shape':list(full.position.shape),
        'first_9_positions':full.position[0,:9].tolist(),'first_9_velocities':full.velocity[0,:9].tolist(),
        'command_positions':r.action_sequence.position.tolist()})
    print(i,elapsed,flush=True)
value={'scope':'Full ARX living-room planning model only, no robot commands',
    'optimizer_inner_iters':4,'warm_iters':8,'cold_iters':40,'optimization_dt':.05,
    'warm_median_s':float(np.median([v['wall_s'] for v in rows[2:]])),'rows':rows}
(task/'mpc_full_scene_timing_v3.json').write_text(json.dumps(value,indent=2))
print('MEDIAN',value['warm_median_s'],flush=True)
