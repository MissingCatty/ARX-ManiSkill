"""Test a pure gripper-reference-frame change with shared frozen GraspGenX weights."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
WORKSPACE=PROJECT.parent
SOURCE=PROJECT/'experiments/diverse_grasp_benchmark/runs/geometry200_fixed452_20261122_01'
ROOT=HERE/'runs/gripper_frame_probe_20261123_01'
SLOTS=['D0002_S1_open_bottled_water','D0004_S1_open_carrot','D0015_S1_open_remote_control',
       'D0021_S1_open_apple','D0031_S1_open_bread','D0033_S1_open_bowl']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(v,indent=2,default=encode)+'\n')


def prepare():
    assert not ROOT.exists();ROOT.mkdir(parents=True)
    inputs={str(SOURCE/'gripper_descriptor.json'):sha(SOURCE/'gripper_descriptor.json')}
    schedule=[]
    for slot in SLOTS:
        d=SOURCE/'trials'/slot
        for p in [d/'graspgenx_raw/model_input_points_world.npy',d/'graspgenx_raw/inference.json',d/'state.json']:
            inputs[str(p)]=sha(p)
        for offset in [0.,.1]:schedule.append({'id':slot+('_eef_origin' if offset==0 else '_origin_back_100mm'),
            'slot':slot,'origin_behind_eef_m':offset,'seed':read(d/'graspgenx_raw/inference.json')['seed']})
    checkpoint=WORKSPACE/'research_dependencies/graspgenx_checkpoints/release'
    save(ROOT/'registration.json',{'purpose':'diagnose GraspGenX frame conditioning; identical physical finger geometry and input points',
        'required_components':['GraspGenX'], 'weights_loads':1,'new_physical_attempts':0,
        'schedule':schedule,'inputs_sha256':inputs,'source_sha256':{str(Path(__file__)):sha(__file__)},
        'weights_sha256':{str(p):sha(p) for p in [checkpoint/'gen/epoch_736.pth',checkpoint/'dis/epoch_1056.pth']},
        'gripper_reference_change':'translation only along approach axis; both conditioning box and output-to-EEF transform change together',
        'native_settings':{'planner':'graspmoe','diffusion_samples':512,'obb_density':'dense','threshold':-1.},
        'not_a_success_rate_experiment':True})


def run():
    reg=read(ROOT/'registration.json')
    for p,h in {**reg['inputs_sha256'],**reg['source_sha256']}.items():assert sha(p)==h,p
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.samplers.planner import run_planner_on_object
    import torch
    checkpoint=WORKSPACE/'research_dependencies/graspgenx_checkpoints/release'
    cfg=load_model_cfg(str(checkpoint/'gen'),str(checkpoint/'dis'))
    cfg.diffusion.ptv3vanilla.enable_flash=False;cfg.discriminator.ptv3vanilla.enable_flash=False
    original=read(SOURCE/'gripper_descriptor.json')
    sampler=GraspGenXSampler.from_sweep_volume(cfg,original['parameters'])
    model=sampler.model
    samplers={0.:sampler}
    for shift in [.1]:
        params=deepcopy(original['parameters'])
        for name in ['offset_open','offset_mid']:params[name][2]+=shift
        params['fingertip_depth']+=shift
        samplers[shift]=GraspGenXSampler.from_sweep_volume(cfg,params,model=model)
    summary=[]
    for item in reg['schedule']:
        out=ROOT/'trials'/item['id'];out.mkdir(parents=True,exist_ok=False)
        save(out/'claim.json',item)
        points=np.load(SOURCE/'trials'/item['slot']/'graspgenx_raw/model_input_points_world.npy')
        shift=item['origin_behind_eef_m'];active=samplers[shift]
        params=original['parameters'];off=np.array(params['offset_open']);off[2]+=shift
        ext=np.array(params['extents_open'])
        torch.manual_seed(item['seed']);np.random.seed(item['seed']%2**32)
        start=time.monotonic()
        grasps,scores,tags,obb=run_planner_on_object(points.astype(np.float32),active,
            planner='graspmoe',grasp_threshold=-1.,num_grasps=512,topk_num_grasps=-1,moe_obb_density='dense')
        np.save(out/'grasps_generator_world.npy',grasps);np.save(out/'scores.npy',scores)
        counts=[]
        for g in grasps:
            local=(points-g[:3,3])@g[:3,:3]
            counts.append(int(np.all(np.abs(local-off)<=ext/2,axis=1).sum()))
        counts=np.array(counts);order=np.argsort(-scores,kind='stable');diff=np.array(tags)=='diff'
        data={'id':item['id'],'slot':item['slot'],'origin_behind_eef_m':shift,
            'seconds':time.monotonic()-start,'diffusion_count':int(diff.sum()),
            'diffusion_candidates_with_target_points':int((diff&(counts>0)).sum()),
            'all_candidates':len(scores),'top1':{'score':float(scores[order[0]]),'tag':tags[order[0]],
                'input_points_in_cavity':int(counts[order[0]])},
            'top20_with_target_points':int((counts[order[:20]]>0).sum()),
            'top20_diffusion':int(diff[order[:20]].sum()),'weight_load_count':1}
        save(out/'result.json',data);save(out/'tags.json',tags)
        np.save(out/'cavity_counts.npy',counts)
        summary.append(data);print(json.dumps(data),flush=True)
        save(ROOT/'progress.json',{'completed':len(summary),'registered':len(reg['schedule']),'last':data})
    save(ROOT/'summary.json',{'results':summary,'completed':len(summary),'new_physical_attempts':0,'weight_load_count':1})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:run()
