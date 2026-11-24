"""Resident ARX analysis components; no physical or simulator execution."""
from pathlib import Path
import json
import os
import sys
import time
import torch

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
import graspcl_pipeline as common
from graspcl_execute import guard
import resident_planner
from postprocess_planner import PostprocessPlanner
from fast_resident_planner import FastPlannerPool
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

root=Path(sys.argv[1]).resolve()
reg=common.read(root / 'registration.json')
guard(reg)
resident_planner.GraspCLPlanner=PostprocessPlanner
started=time.perf_counter()
pool=FastPlannerPool(root / 'planner',Path(reg['warmup_trial']))
context=pool.contexts['open']
model=context.model


def make_pure(position_only=False):
    cfg=InverseKinematicsCfg.create(robot=str(model),scene_model=None,
        optimizer_configs=['ik/lbfgs_ik.yml'],self_collision_check=False,
        num_seeds=32,seed_solver_num_seeds=32,position_tolerance=.002,
        orientation_tolerance=1e6 if position_only else .03,
        optimizer_collision_activation_distance=.003,max_batch_size=64,
        use_cuda_graph=True,load_collision_spheres=False)
    for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
    solver=InverseKinematics(cfg)
    if position_only:
        solver.update_tool_pose_criteria({'right_eef_link':ToolPoseCriteria.track_position()})
    return solver


pure=make_pure()
position=make_pure(True)
common.save(root / 'components_loaded.json',{'pid':os.getpid(),'model_only':True,'physical_attempts':0,
    'components':['resident_open_closed_ARX','original_scene_checkers','pure_full_pose_IK','position_only_IK'],
    'load_seconds':time.perf_counter()-started,'num_optimization_seeds':32,'pure_batch_size':64,
    'diagnostic_only':'Pure IK omits environment collision only for attribution; position-only branch also ignores orientation. Frozen baseline checks, configuration and results are not changed.'})
print('AUDIT COMPONENTS LOADED',flush=True)
