"""Register full-path and nonempty-contact controls for the contact-offset test."""
from arx_project.assets import load_object_catalog
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import contact_replay as replay
from attribute_empty_grasps import FK

ROOT = Path(__file__).parent/'runs/contact_confirmation_20261123_01'


def prepare(root):
    assert not root.exists()
    root.mkdir(parents=True)
    original = replay.read(replay.DEFAULT/'registration.json')
    inputs = root/'controlled_inputs'
    source_hashes = {}
    fk = FK()
    for slot in replay.SLOTS:
        source = replay.SOURCE/'trials'/slot
        target = inputs/'trials'/slot; target.mkdir(parents=True)
        entry, state, plan = [replay.read(source/name) for name in ['entry.json','state.json','plan_result.json']]
        transform = np.array(plan['selected']['grasp_world']); transform[2,3] -= .015
        names = state['joint_names']; arm = [names.index(f'right_joint{i}') for i in range(21,27)]
        qseed = np.array(state['qpos']); qseed[arm] = plan['selected']['q_grasp']
        limits = np.array(state['joint_limits'])[arm]
        base = np.asarray(state['links']['base_link'])
        def residual(q):
            value = qseed.copy(); value[arm] = q
            pose = fk.at('right_eef_link', dict(zip(names,value)),base)
            return np.r_[pose[:3,3]-transform[:3,3],
                Rotation.from_matrix(pose[:3,:3]@transform[:3,:3].T).as_rotvec()*.1]
        solved = least_squares(residual,np.asarray(plan['selected']['q_grasp']),
            bounds=(limits[:,0]+1e-5,limits[:,1]-1e-5),gtol=1e-12,xtol=1e-12,ftol=1e-12)
        assert np.linalg.norm(residual(solved.x)) < 1e-6
        qseed[arm] = solved.x
        trace = [__import__('json').loads(s) for s in (source/'execution.jsonl').open()]
        ix = next(i for i,r in enumerate(trace) if r['stage']=='close')
        before = deepcopy(trace[ix-1]); before['qpos'] = qseed.tolist()
        before['stage'] = 'approach'; before['tick'] = 1
        close = deepcopy(before); close.update(stage='close',tick=2)
        import json
        (target/'execution.jsonl').write_text(json.dumps(before)+'\n'+json.dumps(close)+'\n')
        plan['selected']['q_grasp'] = solved.x.tolist()
        plan['selected']['grasp_world'] = transform.tolist()
        replay.save(target/'plan_result.json',plan)
        for name in ['entry.json','state.json','approach_plan_result.json']:
            shutil.copy2(source/name,target/name)
        replay.save(target/'controlled_pose_provenance.json',{
            'source_run':str(replay.SOURCE),'source_slot':slot,'eef_world_delta_m':[0,0,-.015],
            'intention':'diagnostic contact-positive control; not model prediction or benchmark result',
            'end_pose_solved_by':'independent URDF FK and bounded least squares',
            'residual_norm':float(np.linalg.norm(residual(solved.x))), 'target_model_unchanged':True,
            'source_plan_sha256':replay.sha(source/'plan_result.json')})
        for p in target.iterdir(): source_hashes[str(p)]=replay.sha(p)
    shutil.copy2(replay.SOURCE/'object_catalog.json',inputs/'object_catalog.json')
    schedule=[]
    for condition,settings in [('original',{}),('contact_offset_1mm',{'contact_offset':.001})]:
        for slot in replay.SLOTS:
            schedule.append({'id':f'deeper15_{condition}__{slot}','condition':condition,'settings':settings,
                'slot':slot,'replay':'closure_only','source_run':str(inputs)})
    for slot in replay.SLOTS:
        schedule.append({'id':'fullpath_contact_offset_1mm__'+slot,'condition':'contact_offset_1mm',
            'settings':{'contact_offset':.001},'slot':slot,'replay':'full_path','source_run':str(replay.SOURCE)})
    reg = deepcopy(original)
    reg.update(purpose='Confirm contact-offset causality with full-path and 15mm deeper contact-positive controls',
        schedule=schedule,maximum_closure_diagnostics=len(schedule),
        diagnostic_source_sha256={str(p):replay.sha(p) for p in [Path(__file__),Path(replay.__file__),
            Path(__file__).parent/'attribute_empty_grasps.py']},
        input_sha256={**reg['input_sha256'],**source_hashes})
    replay.save(root/'registration.json',reg)


def run(root):
    reg=replay.read(root/'registration.json')
    for p,h in {**reg['input_sha256'],**reg['diagnostic_source_sha256']}.items():assert replay.sha(p)==h,p
    assert replay.sha(replay.SOURCE/'STOP')==reg['source_benchmark_stop_sha256']
    import os
    os.environ.setdefault('MS_ASSET_DIR',str(replay.MS/'assets'))
    sys.path[:0]=[str(replay.PROJECT/'experiments/diverse_object_scenes'),str(replay.MS),
        str(replay.MS/'arx_pipeline'),str(replay.PROJECT/'experiments/pointcloud_benchmark')]
    import arx_runtime_imports
    arx_runtime_imports.install()
    import environment,run as original_run,sapien,torch,gymnasium as gym
    import asset_environment
    from head_camera_config import head_configuration
    environment.INITIAL_JOINTS={**environment.INITIAL_JOINTS,'joint4':.452}
    override,head=head_configuration(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS=deepcopy(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS['camera']['definitions']['head']=head['camera_definition']
    runtime=(environment,original_run,sapien,torch,gym,override)
    catalog=load_object_catalog(replay.SOURCE/'object_catalog.json')['models'];keeper=[];results=[]
    for item in reg['schedule']:
        if (root/'trials'/item['id']).exists():raise RuntimeError('already claimed '+item['id'])
        if (root/'STOP').exists():raise RuntimeError('Diagnostic STOP')
        replay.SOURCE=Path(item['source_run'])
        results.append(replay.run_one(root,item,runtime,catalog,keeper))
        replay.save(root/'progress.json',{'completed':len(results),'registered':len(reg['schedule']),'last':results[-1]})
    replay.save(root/'summary.json',{'completed':len(results),'registered':len(reg['schedule']),
        'results':results,'formal_benchmark_resumed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    if a.stage=='prepare':prepare(ROOT)
    else:run(ROOT)
