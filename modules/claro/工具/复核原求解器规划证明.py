"""Check counterfactual witnesses with the original IK and collision routines."""
from datetime import datetime
from pathlib import Path
import time

import numpy as np

import 核查挡物规划 as first
from 核查预抓取约束 import attempt_plan

read, save, sha = first.read, first.save, first.sha
OUT = first.DEST.parent / '20261119_04'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    earlier = read(first.DEST.parent / '20261119_02/预抓取约束核查汇总.json')
    cpu = read(first.DEST.parent / '20261119_03_02/独立运动学汇总.json')
    sources = {str(Path(__file__).resolve()): sha(__file__),
               str(first.DEST.parent / '20261119_02/预抓取约束核查汇总.json'): sha(first.DEST.parent / '20261119_02/预抓取约束核查汇总.json'),
               str(first.DEST.parent / '20261119_03_02/独立运动学汇总.json'): sha(first.DEST.parent / '20261119_03_02/独立运动学汇总.json')}
    save(OUT / '核查登记.json', {'time': datetime.now().astimezone().isoformat(), 'physical_attempts': 0,
         'source_sha256': sources, 'scope': 'Original IK routine for the same G and different pregrasp distances; original full-world and approach checks. Verify independent CPU witnesses using original checker and IK metrics.'})
    import torch
    import planner as original
    import resident_planner
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner
    from physical_table_pair_20261118_01 import install_collision_policy
    from curobo.types import GoalToolPose

    install_collision_policy();resident_planner.GraspCLPlanner = ShortLiftPlanner
    warmup = first.MS / 'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
    began = time.monotonic();pool = FastPlannerPool(OUT / '规划器预热', warmup)
    registered = {c['case']: c for c in first.cases()};rows = []
    for case in earlier['cases']:
        proof = case['planning_witness']
        if proof is None:continue
        name = case['case'];state = read(Path(registered[name]['directory']) / 'state.json')
        destination = OUT / name;ctx = pool.bind(destination / '原模型', state)
        g = np.asarray(proof['grasp_world']);start = ctx.measured(state)
        qg = ctx.ik(g, start);events=[];new_proof=None
        for distance in sorted(set([.08, np.linalg.norm(g[:3, 3]-np.asarray(proof['pregrasp_world'])[:3, 3])]), reverse=True):
            if qg is None:break
            q_before = qg.clone();ctx.planner.reset_seed()
            pre = g.copy();pre[:3, 3] -= distance*g[:3, 2]
            lift = g.copy();lift[2, 3] += .03
            qp, ql = ctx.ik(pre, qg), ctx.ik(lift, qg)
            assert torch.equal(q_before, qg)
            event = {'pregrasp_distance_m': float(distance), 'same_grasp_q': qg.cpu().tolist(),
                     'original_pregrasp_ik_passed': qp is not None, 'original_lift_ik_passed': ql is not None}
            if qp is not None and ql is not None:
                checked = attempt_plan(pool, state, destination / ('原求解器_' + str(round(distance*100))),
                                       g, pre, qg.cpu().numpy()[None], qp.cpu().numpy()[None], torch)
                event['full_initial_planning'] = checked
                if checked['passed']:new_proof = event
            events.append(event)
        row = {'case': name, 'original_grasp_ik_passed': qg is not None, 'events': events,
               'planning_proof': new_proof, 'physical_attempts': 0}
        rows.append(row);save(destination / '原求解器复核.json', row)
        print('ORIGINAL_IK', name, [(round(e['pregrasp_distance_m']*100),e['original_pregrasp_ik_passed']) for e in events],
              'FULL_PLAN', new_proof is not None, flush=True)
    cpu_rows=[]
    for record in cpu['queries']:
        if not record['ik_found']:continue
        name=record['case'];state=read(Path(registered[name]['directory'])/'state.json')
        ctx=pool.bind(OUT/(name+'_CPU证明')/'原模型',state)
        q=torch.as_tensor(record['witness']['q'],dtype=torch.float32,device='cuda')
        kin=ctx.checker.get_kinematics(q[None,None])
        bound=float(ctx.checker.get_bound(q[None,None]).sum())
        collision=float(ctx.checker.get_collision_constraint(kin).sum())
        valid=bool(ctx.checker.validate(q[None,None]).all())
        goal=ctx.invbase@np.asarray(record['world_goal']);quat=original.pose7(goal)[3:]
        target=GoalToolPose(tool_frames=['right_eef_link'],
            position=torch.as_tensor(goal[:3,3],dtype=torch.float32,device='cuda').reshape(1,1,1,1,3),
            quaternion=torch.as_tensor(quat,dtype=torch.float32,device='cuda').reshape(1,1,1,1,4))
        n=ctx.planner.ik_solver.config.num_seeds
        seed=q[None,None].repeat(1,n,1)
        metrics=ctx.planner.ik_solver.solve_pose(target,current_state=ctx.js(ctx.measured(state)),
                     seed_config=seed,return_seeds=4,run_optimizer=False)
        row={'case':name,'stage':record['stage'],'candidate':record['candidate'],
             'original_bounds_cost':bound,'original_scene_collision_cost':collision,
             'original_checker_passed':valid,'original_ik_metrics_passed_without_optimization':bool(metrics.success.any()),
             'q':q.cpu().tolist(),'physical_attempts':0}
        cpu_rows.append(row);print('CPU_WITNESS',name,record['stage'],'checker',valid,'bounds',bound,'scene',collision,'ik_metrics',row['original_ik_metrics_passed_without_optimization'],flush=True)
    assert all(sha(p)==v for p,v in sources.items())
    save(OUT/'原求解器复核汇总.json',{'cases':rows,'cpu_witnesses':cpu_rows,'physical_attempts':0,
        'source_hashes_unchanged':True,'wall_seconds':time.monotonic()-began})


if __name__=='__main__':main()
