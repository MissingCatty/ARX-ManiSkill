"""Separate saved grasp-endpoint IK failures from scene collision, without motion."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
MS=PROJECT.parent/'maniskill'
SOURCE=PROJECT/'experiments/repaired_grasp_benchmark/runs/geometry200_repaired_20261123_01'
ROOT=HERE/'runs/planning_failure_split_20261123_01'
IDS=['D0004_S1_open_carrot','D0007_S1_open_banana','D0008_S1_open_teapot',
     'D0014_S1_open_scissors','D0018_S1_open_boxed_food','D0023_S1_open_baguette',
     'D0024_S1_open_pear','D0025_S1_open_book','D0026_S1_open_sponge',
     'D0035_S1_open_donut','D0036_S1_open_cell_phone']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def prepare():
    assert not ROOT.exists();ROOT.mkdir(parents=True)
    inputs={};rows=[]
    for slot in IDS:
        d=SOURCE/'trials'/slot
        claim=read(SOURCE/'claims'/(slot+'.json'))
        assert claim['state']=='complete' and sha(d/'result.json')==claim['result_sha256']
        plan=read(d/'plan_result.json')
        assert not plan['success'] and plan['count']==plan['rejected']['grasp_ik']
        rows.append({'id':slot,'candidates':plan['count']})
        for name in ['state.json','result.json','plan_result.json','candidates/candidate_pool.json','plan/arx.yml','plan/scene.json']:
            inputs[str(d/name)]=sha(d/name)
    save(ROOT/'registration.json',{'source_run':str(SOURCE),'cases':rows,'input_sha256':inputs,
        'source_sha256':{str(Path(__file__)):sha(__file__)},'physical_attempts':0,
        'snapshot':'11 planning failures at the previous 45/200 progress report',
        'seeds':128,'retained_solutions':32,'position_tolerance_m':.002,'orientation_tolerance_rad':.03,
        'purpose':'Full-pose collision-free IK followed by original open-gripper scene checker; position-only IK as a diagnostic',
        'modify_running_service':False,'finite_search_is_not_global_unreachability_proof':True})


def run():
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline')]
    import torch
    import yaml
    from curobo.types import JointState
    from curobo._src.types.robot import RobotCfg
    from curobo._src.geom.types import SceneCfg
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    from planning_rejection_audit import make_pure
    from ik_workspace_audit import solve,collision_details
    from resident_planner import all_kinematics_params
    from scene_tools import ArmFK
    from scipy.spatial.transform import Rotation
    reg=read(ROOT/'registration.json')
    for p,h in {**reg['input_sha256'],**reg['source_sha256']}.items():assert sha(p)==h,p
    assert not (ROOT/'started.json').exists()
    save(ROOT/'started.json',{'pid':os.getpid(),'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'physical_attempts':0})
    first=SOURCE/'trials'/IDS[0]
    model=first/'plan/arx.yml'
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),
        scene_model=read(first/'plan/scene.json'),n_meshes=8,n_cuboids=8,collision_activation_distance=0.))
    checker.self_collision_cost=None
    full=make_pure(model);position=make_pure(model,True)
    fk=ArmFK();names=[f'right_joint{i}' for i in range(21,27)]
    params=all_kinematics_params(checker)
    rows=[]
    for slot in IDS:
        begin=time.monotonic();d=SOURCE/'trials'/slot;out=ROOT/'cases'/slot
        out.mkdir(parents=True,exist_ok=False)
        state=read(d/'state.json');scene=read(d/'plan/scene.json')
        checker.clear_scene_cache();checker.update_world(SceneCfg.create(scene))
        cfg=yaml.safe_load((d/'plan/arx.yml').read_text())
        new_params=RobotCfg.create(cfg['robot_cfg'],checker.device_cfg).kinematics.kinematics_config
        for target in params:target.copy_(new_params)
        base=np.asarray(state['links']['body4']);invbase=np.linalg.inv(base)
        limits=torch.tensor([state['joint_limits'][state['joint_names'].index(n)] for n in names],device='cuda')
        measured=torch.tensor([state['qpos'][state['joint_names'].index(n)] for n in names],device='cuda')
        start=torch.maximum(torch.minimum(measured,limits[:,1]-1e-4),limits[:,0]+1e-4)
        context=SimpleNamespace(base=base,invbase=invbase,checker=checker,
            planner=SimpleNamespace(ik_solver=SimpleNamespace(joint_names=names)))
        pool=read(d/'candidates/candidate_pool.json')['candidates']
        worlds=[]
        for item in pool:
            t=np.array(item['pose']);t[:3,3]-=.043*t[:3,2];worlds.append(t)
        # Verify that the pure solver can recover this saved measured pose.
        full.reset_seed()
        tq,tok,_,_=solve(full,context,[np.asarray(state['links']['right_eef_link'])],start)
        assert bool(tok.any()),'Pure solver failed measured-pose control'

        def assess(solver,pos_only=False):
            solver.reset_seed();q,ok,pe,re=solve(solver,context,worlds,start)
            ok &= pe<=.002001
            if not pos_only:ok &= re<=.030001
            ok &= ((q>=limits[:,0]-1e-6)&(q<=limits[:,1]+1e-6)).all(-1)
            clear=torch.zeros_like(ok);table=torch.zeros_like(ok);ground=torch.zeros_like(ok)
            found=ok.nonzero();solutions=q[ok]
            table_links=Counter();witness=[]
            linkmap=checker.kinematics.kinematics_config
            index_to_name={int(v):k for k,v in linkmap.link_name_to_idx_map.items()}
            sphere_names=[index_to_name[int(v)] for v in linkmap.link_sphere_idx_map.cpu().tolist()]
            maximum_table=0.
            for offset in range(0,len(solutions),1024):
                segment=solutions[offset:offset+1024]
                detail=collision_details(context,state,segment)
                loc=found[offset:offset+len(segment)]
                clear[loc[:,0],loc[:,1]]=detail['full_scene_valid']
                table[loc[:,0],loc[:,1]]=detail['table_penetration']>0
                ground[loc[:,0],loc[:,1]]=detail['ground_penetration']>0
                mask=detail['table_penetration']>0
                maximum_table=max(maximum_table,float(detail['table_penetration'].max()))
                table_links.update(sphere_names[i] for i in detail['worst_table_sphere'][mask].cpu().tolist())
                for choice in [detail['full_scene_valid'],~detail['full_scene_valid']]:
                    indices=choice.nonzero().flatten()
                    if len(indices) and len(witness)<4:
                        k=int(indices[0]);candidate=int(loc[k,0]);joints=segment[k].cpu().numpy()
                        actual=fk.fk(joints);goal=invbase@worlds[candidate]
                        cpu_error=float(np.linalg.norm(actual[:3,3]-goal[:3,3]))
                        cpu_rot=float(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude())
                        assert cpu_error<.00201 and (pos_only or cpu_rot<.03001)
                        witness.append({'candidate':candidate,'q':joints.tolist(),'position_error_m':cpu_error,
                            'rotation_error_rad':cpu_rot,'scene_clear':bool(detail['full_scene_valid'][k]),
                            'table_penetration_m':float(detail['table_penetration'][k]),
                            'worst_table_link':sphere_names[int(detail['worst_table_sphere'][k])]})
            np.savez_compressed(out/('position_only.npz' if pos_only else 'full_pose.npz'),q=q.cpu().numpy(),
                solver_success=ok.cpu().numpy(),scene_clear=clear.cpu().numpy(),table_collision=table.cpu().numpy())
            return {'goals':len(worlds),'kinematic_solutions_candidates':int(ok.any(-1).sum()),
                'scene_clear_candidates':int(clear.any(-1).sum()),
                'table_colliding_candidates':int(table.any(-1).sum()),
                'ground_colliding_candidates':int(ground.any(-1).sum()),
                'max_table_penetration_m':maximum_table,'table_collision_links':dict(table_links),'witnesses':witness}

        result={'id':slot,'candidates':len(worlds),'full_pose':assess(full),
            'position_only':assess(position,True),'seconds':time.monotonic()-begin,'physical_attempts':0}
        a=result['full_pose'];b=result['position_only']
        result['diagnosis']=('original_search_missed_scene_clear_solution' if a['scene_clear_candidates'] else
            'found_kinematic_solutions_but_scene_collision' if a['kinematic_solutions_candidates'] else
            'fixed_orientation_not_solved_position_reachable' if b['scene_clear_candidates'] else
            'no_scene_clear_endpoint_found_in_finite_search')
        save(out/'result.json',result);rows.append(result)
        save(ROOT/'progress.json',{'completed':len(rows),'registered':len(IDS),'last':result})
        print(json.dumps(result,ensure_ascii=False),flush=True)
    for p,h in reg['input_sha256'].items():assert sha(p)==h,p
    save(ROOT/'summary.json',{'completed':len(rows),'registered':len(IDS),'physical_attempts':0,
        'diagnoses':dict(Counter(r['diagnosis'] for r in rows)),'cases':rows,'source_inputs_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:run()
