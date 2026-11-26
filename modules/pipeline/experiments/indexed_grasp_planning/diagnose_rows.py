"""Inspect mixed-goal IK rows without loading trajectory planners or simulation."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from modules.curobo.indexed_planning.batches import padded_worlds, row_safe_solve


def run(root, source, gpu, row_safe=False):
    root.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, JointState
    from modules.curobo.dual_arm.model import ArmModel
    torch.set_num_threads(4)
    state = read_json(source/'resident/dual_arm/warmup/state.json')
    model = source/'solver_prewarm/binding/arx.yml'
    names = [f'right_joint{i}' for i in range(21,27)]
    q = np.array([state['qpos'][state['joint_names'].index(n)] for n in names])
    limits = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in names])
    q = np.maximum(np.minimum(q,limits[:,1]-1e-4),limits[:,0]+1e-4)
    base = np.asarray(state['links']['body4']); invbase = np.linalg.inv(base)
    measured = np.asarray(state['links']['right_eef_link'])
    far = measured.copy(); far[:3,3] += base[:3,:3]@np.array([10.,0.,0.])
    physical = ArmModel()
    shifted = dict(state);shifted['qpos'] = list(state['qpos']);shifted['links']={'base_link':state['links']['base_link']}
    i = state['joint_names'].index(names[-1])
    shifted['qpos'][i] = float(q[-1]+(.12 if q[-1]+.12 < limits[-1,1] else -.12))
    nearby = physical.frames(shifted)['right_eef_link']
    reports = []
    for capacity in [1,16,32]:
        cfg = InverseKinematicsCfg.create(robot=str(model),scene_model=None,
            optimizer_configs=['ik/lbfgs_ik.yml'],self_collision_check=False,
            num_seeds=32,seed_solver_num_seeds=32,position_tolerance=.002,orientation_tolerance=.03,
            use_cuda_graph=True,max_batch_size=capacity,load_collision_spheres=False)
        for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
            for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
        solver = InverseKinematics(cfg)
        queries = [('positive',[measured]),('negative',[far])]
        if capacity>1:queries += [('negative_positive',[far,measured]),('positive_negative',[measured,far]),
            ('two_positive',[measured,nearby]),('coarse_negative_positive',[far,measured]),
            ('forced_positive_negative',[measured,far]),('forced_two_positive',[measured,nearby])]
        if row_safe:queries=[row for row in queries if not row[0].startswith(('forced_','coarse_'))]
        for label,targets in queries:
            solver.config.exit_early = not label.startswith('forced_')
            solver.config.exit_early_batch_success_threshold = 0. if label.startswith('coarse_') else 1.
            count = len(targets)
            targets = padded_worlds(targets,capacity)
            poses = invbase@targets
            rotations = Rotation.from_matrix(poses[:,:3,:3]).as_quat()[:,[3,0,1,2]]
            goal = GoalToolPose(tool_frames=['right_eef_link'],
                position=torch.tensor(poses[:,:3,3],dtype=torch.float32,device='cuda').reshape(capacity,1,1,1,3).contiguous(),
                quaternion=torch.tensor(rotations,dtype=torch.float32,device='cuda').reshape(capacity,1,1,1,4).contiguous())
            current = JointState.from_position(torch.tensor(q,dtype=torch.float32,device='cuda').expand(capacity,-1).clone(),joint_names=names)
            began = time.perf_counter()
            stages = None
            if row_safe:
                def subset(indices):
                    return GoalToolPose(tool_frames=goal.tool_frames,
                        position=goal.position.index_select(0,indices).contiguous(),
                        quaternion=goal.quaternion.index_select(0,indices).contiguous())
                values,valid,stages = row_safe_solve(solver,goal,current,8,count,subset)
                solutions = values.detach().cpu().numpy().reshape(capacity,8,6)
                flags = valid.detach().cpu().numpy().reshape(capacity,8)
            else:
                result = solver.solve_pose(goal,current_state=current,return_seeds=8)
                solutions = result.solution.detach().cpu().numpy().reshape(capacity,8,6)
                flags = result.success.detach().cpu().numpy().reshape(capacity,8)
            row = {'capacity':capacity,'query':label,'seconds':time.perf_counter()-began,
                'success_counts':flags.sum(-1).tolist(),
                'targets_world_first_two':targets[:2].tolist(),
                'solutions_first_two_rows':solutions[:2].tolist(),
                'position_errors':None if row_safe else result.position_error.detach().cpu().numpy().reshape(capacity,-1).tolist(),
                'rotation_errors':None if row_safe else result.rotation_error.detach().cpu().numpy().reshape(capacity,-1).tolist(),
                'per_row_stages':stages}
            checks = []
            for index in range(min(capacity,2)):
                errors = []
                for value in solutions[index][flags[index]]:
                    sample = dict(state);sample['qpos'] = list(state['qpos'])
                    sample['links'] = {'base_link':state['links']['base_link']}
                    for name,angle in zip(names,value):sample['qpos'][state['joint_names'].index(name)] = float(angle)
                    actual = physical.frames(sample)['right_eef_link']
                    expected = targets[index]
                    errors.append([float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
                        float(Rotation.from_matrix(actual[:3,:3]@expected[:3,:3].T).magnitude())])
                checks.append(errors)
            row['independent_errors_first_two'] = checks
            reports.append(row)
            write_json(root/'results.json',{'at':now(),'reports':reports,'physical_grasps':0,'trajectory_planning_requests':0})
            print(capacity,label,row['success_counts'][:4],flush=True)
    correct_fk = all(pe<=.002 and re<=.03 for report in reports
        for row in report['independent_errors_first_two'] for pe,re in row)
    retained = all(report['success_counts'][1]>0 and report['success_counts'][0]==0
        for report in reports if report['query']=='negative_positive')
    write_json(root/'complete.json',{'at':now(),'direct_IK_calls':len(reports),'physical_grasps':0,
        'row_safe':row_safe,'independent_FK_passed':correct_fk,'known_reachable_mixed_rows_retained':retained})
    if row_safe and (not correct_fk or not retained):raise RuntimeError('Per-row IK contract failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--row-safe',action='store_true')
    args=parser.parse_args();run(ensure_run_path(args.root),ensure_run_path(args.source),args.gpu,args.row_safe)
