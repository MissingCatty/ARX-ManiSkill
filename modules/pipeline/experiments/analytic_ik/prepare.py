"""Freeze raw endpoint batches and independent CPU EAIK reference branches."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.curobo.indexed_planning.geometric.kinematics import ArmChain, GeometricSolver
from modules.curobo.analytic_ik.model import AnalyticModel


CONFIG = ROOT/'configs/experiments/gpu_analytic_ik.json'


def guard(root):
    reg = read_json(root/'registration.json')
    for group in ['sources','inputs']:
        for path, value in reg[group].items():
            if sha256(path) != value:
                raise ValueError('Analytic benchmark frozen file changed: '+path)
    return reg


def prepare(root, gpu):
    if root.exists():
        raise ValueError('Preserve old registrations and use a fresh directory')
    root.mkdir(parents=True);cfg=read_json(CONFIG);source=ROOT/cfg['source_run']
    sources=[*list((ROOT/'modules/curobo/analytic_ik').glob('*.py')),
        *list(Path(__file__).parent.glob('*.py')),
        ROOT/'modules/curobo/indexed_planning/geometric/kinematics.py', CONFIG]
    inputs={};cases=[];capacity=cfg['batch_capacity']
    for arm in ['right','left']:
        chain=ArmChain(arm);rng=np.random.default_rng(20261127+(arm=='left'))
        q=rng.uniform(chain.limits[:,0]+.001,chain.limits[:,1]-.001,(cfg['random_states_per_arm'],6))
        targets=chain.fk(q);out=root/'random'/arm;out.mkdir(parents=True)
        reference=GeometricSolver(arm);answers=[];began=time.perf_counter()
        for target in targets[:cfg['reference_EAIK_targets_per_arm']]:
            answers.append(reference.solve(target,np.zeros(6)))
        seconds=time.perf_counter()-began
        write_json(out/'EAIK_reference.json',{'answers':answers,'wall_seconds':seconds,
            'targets':len(answers),'native_solver_seconds':sum(a['native_solver_seconds'] for a in answers)})
        np.savez_compressed(out/'states.npz',q=q,targets=targets)
        write_json(out/'model.json',AnalyticModel(arm).receipt())
        for path in out.iterdir():inputs[str(path)]=sha256(path)
        inputs[str(chain.urdf)]=sha256(chain.urdf)
    for case in cfg['cases']:
        original=source/'plans'/case/'batched_1'
        pool=read_json(original/'candidates/candidate_pool.json')
        if len(pool['candidates'])*3>capacity:raise ValueError('Frozen case exceeds registered fixed endpoint capacity')
        for arm in ['right','left']:
            out=root/'cases'/case/arm;out.mkdir(parents=True)
            state=read_json(original/'arms'/arm/'state.json');base=np.asarray(state['links']['body4'])
            poses=[];widths=[];labels=[]
            for index,candidate in enumerate(pool['candidates']):
                grasp=np.asarray(candidate['pose'],dtype=float).copy();grasp[:3,3]-=.043*grasp[:3,2]
                pre=grasp.copy();pre[:3,3]-=.08*grasp[:3,2]
                lift=grasp.copy();lift[2,3]+=.03
                for label,target in [('grasp',grasp),('pregrasp',pre),('lift',lift)]:
                    poses.append(np.linalg.inv(base)@target);widths.append(candidate['width'])
                    labels.append({'candidate_index':index,'stage':label,
                        'source_generator_index':candidate['metadata']['source_generator_index']})
            size=len(poses);poses=np.asarray(poses);widths=np.asarray(widths)
            poses=np.concatenate([poses,np.repeat(poses[:1],capacity-size,axis=0)])
            widths=np.r_[widths,np.repeat(widths[0],capacity-size)]
            np.savez_compressed(out/'targets.npz',targets=poses,widths=widths,input_valid=np.arange(capacity)<size)
            write_json(out/'labels.json',labels)
            for name in ['arx.yml','scene.json']:
                src=original/'arms'/arm/'plan'/name
                shutil.copy2(src,out/name);inputs[str(src)]=sha256(src);inputs[str(out/name)]=sha256(out/name)
            for p in [original/'state.json',original/'candidates/candidate_pool.json',out/'targets.npz',out/'labels.json']:
                inputs[str(p)]=sha256(p)
            scene=read_json(out/'scene.json')
            for m in scene.get('mesh',{}).values():inputs[m['file_path']]=sha256(m['file_path'])
            cases.append({'case':case,'arm':arm,'directory':str(out.relative_to(ROOT)),
                'real_targets':size,'padded_targets':capacity-size,'input_candidates':len(pool['candidates'])})
    preserve={str(ROOT/name):sha256(ROOT/name) for name in ['configs/modules.json',
        'configs/experiments/active_experiment.json','simulation/robots/arx/collision/arx_refined_871.json',
        'configs/policies/fingertip_table.json']}
    inputs.update(preserve)
    for p in sources:
        out=root/'source_registered'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    write_json(root/'registration.json',{'at':now(),'config':cfg,'gpu':gpu,'cases':cases,
        'source_run':str(source.relative_to(ROOT)),'sources':{str(p):sha256(p) for p in sources},
        'inputs':inputs,'production_bytes':preserve,'planning_requests':0,'physical_grasps':0,
        'simulation_instances':0,'active_planner_modified':False})
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_collision_cases':0,'collision_cases':len(cases)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpu',required=True)
    args=parser.parse_args();prepare(ensure_run_path(args.root),args.gpu)
