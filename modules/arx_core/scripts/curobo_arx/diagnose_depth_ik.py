"""Compare IK seeds on an archived ARX model; sends no simulator commands.

An obstacle-free solve is diagnostic only. Its solution must still pass the
original independent world checker before it can be used by any executor.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from curobo.types import GoalToolPose, JointState
from curobo._src.geom.types import SceneCfg

ROOT = Path(__file__).resolve().parents[2]


def run(out):
    out = Path(out).resolve()
    np.random.seed(0);random.seed(0);torch.manual_seed(0)
    snap = json.loads((out/'snapshot.json').read_text())
    selected = json.loads((out/'selected_grasp.json').read_text())
    sources = json.loads((out/'scene_sources.json').read_text())
    scene = yaml.safe_load((out/'scene.yml').read_text())
    target_path = snap['objects']['green_sphere']['path']
    scene['mesh'] = {k:v for k,v in scene['mesh'].items() if not (sources[k]==target_path or sources[k].startswith(target_path+'/'))}
    names = snap['right_arm']
    invbase = np.linalg.inv(snap['link_world']['body4'])
    limits = torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in names],device='cuda')
    home = torch.tensor([snap['joint_positions'][snap['joint_names'].index(n)] for n in names],device='cuda')
    home = torch.maximum(torch.minimum(home,limits[:,1]-1e-4),limits[:,0]+1e-4)
    original = torch.tensor(selected['q_grasp'],device='cuda')
    # Reuse the actual baseline initializer without running its top-level robot
    # program. Archive the source and its hash to make the model check auditable.
    source = (ROOT/'scripts/curobo_arx/grasp_execute.py').read_text()
    tree = ast.parse(source)
    subset = ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name=='initialize'],type_ignores=[])
    ns={'root':ROOT,'out':out,'scene':scene,'sources':sources,'invbase':invbase,'use_trajopt':False}
    exec(compile(subset,str(ROOT/'scripts/curobo_arx/grasp_execute.py'),'exec'),ns)
    (out/'baseline_initializer_source.py').write_text(source)
    started=time.monotonic()
    planner,checker=ns['initialize'](str(out/'model/arx.yml'))
    warmup=time.monotonic()-started
    shifted = copy.deepcopy(scene)
    for key,mesh in shifted['mesh'].items():
        if 'breakfast_table' in sources[key]:
            mesh['pose'][:3]=(np.asarray(mesh['pose'][:3])+invbase[:3,2]*.008).tolist()
    closed=ns['RobotCollisionChecker'](ns['RobotCollisionCheckerCfg'].load_from_config(
        robot_config=str(out/'model_closed/arx.yml'),scene_model=shifted,
        n_meshes=len(shifted['mesh'])+1,collision_activation_distance=0.))
    closed.self_collision_cost=None
    report={'scope':'Archived ARX planning-model diagnosis only; no PhysX execution',
            'baseline_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'warmup_seconds':warmup,'tests':[]}
    for world_mode in ['original_world','no_world_diagnostic']:
        if world_mode=='no_world_diagnostic':
            planner.update_world(SceneCfg())
        for shift in [0.,-.003,-.006,-.010]:
            world=np.asarray(selected['grasp_world']).copy();world[2,3]+=shift
            t=invbase@world;q=Rotation.from_matrix(t[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
            goal=GoalToolPose(tool_frames=['right_eef_link'],position=torch.tensor(t[:3,3],dtype=torch.float32,device='cuda').reshape(1,1,1,1,3),quaternion=torch.tensor(q,dtype=torch.float32,device='cuda').reshape(1,1,1,1,4))
            for mode in ['home_current','nearby_current','explicit_nearby_seed']:
                planner.ik_solver.reset_seed()
                seed=home if mode!='nearby_current' else original
                args={'current_state':JointState.from_position(seed[None],joint_names=names),'return_seeds':4}
                if mode=='explicit_nearby_seed':args['seed_config']=original[None,None].clone()
                start=time.monotonic();result=planner.ik_solver.solve_pose(goal,**args)
                qs=result.solution.reshape(-1,len(names))
                valid=checker.validate(qs[:,None]).flatten()
                closed_valid=closed.validate(qs[:,None]).flatten()
                row={'world_mode':world_mode,'shift_m':shift,'seed_mode':mode,
                     'solve_seconds':time.monotonic()-start,'success':result.success.flatten().tolist(),
                     'feasible':result.feasible.flatten().tolist(),
                     'position_error_m':result.position_error.flatten().tolist(),
                     'rotation_error_rad':result.rotation_error.flatten().tolist(),
                     'independent_original_world_valid':valid.tolist(),
                     'nominal_closed_8mm_table_valid':closed_valid.tolist(),'solutions':qs.tolist()}
                report['tests'].append(row)
                (out/'ik_report.json').write_text(json.dumps(report,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k!='solutions'}),flush=True)
    planner.destroy()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
