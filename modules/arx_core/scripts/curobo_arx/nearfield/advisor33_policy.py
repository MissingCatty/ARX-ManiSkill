"""Pinned C/D inference, two full-stale inputs, and G0-free D interpretation."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files,digest
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from training.nearfield_arx import lateral_recovery,closure_recovery
from training.nearfield_arx.projection import Projector,model_views
from training.nearfield_arx import native_mask
from .advisor31_policy import Models
from .advisor32_spec import model_images
from . import advisor32_rules as local
from . import advisor33_geometry as geometry
from .attribution_spec import geometry_prediction

VERSION='arx_advisor33_zero_prior_v1'


def stationary_dispatch(observed,current,stop):
    from training.nearfield_arx.learned_policy import dispatch_drift
    original=dispatch_drift(observed,current)
    passed=(original['maximum_joint_drift_rad']<=.0005 and original['object_drift']['translation_m']<=.001
        and original['object_drift']['rotation_rad']<=.01 and stop['pass'])
    return {**original,'pass':bool(passed),'original_velocity_only_pass':original['pass'],'stop_proof':stop,
        'change':'Registered dual-route measured stop, original joint/object drift bounds retained'}


def decision(req, token, gate, points, row, scale):
    if req['mode'] not in ('decision','final_gate'):raise ValueError('Unknown decision mode')
    if req['method']=='D' and req['vision_arm']!='g0' and 'G0_world' in req['inputs']:
        raise ValueError('Coarse prior crossed D boundary')
    current=req['inputs']['eef_world']; method=req['method']; index=req['motions_completed']
    result={'kind':'reject','reason':None,'closure_origin':None,'goal_world':None,'interpretation':None}
    if req['mode']=='final_gate':
        return {**result,'kind':'close' if gate['pass'] else 'reject',
                'closure_origin':'visual_gate_after_motion_budget','reason':None if gate['pass'] else 'visual_gate_rejected'}
    if req.get('contact_closure') and gate['pass']:
        return {**result,'kind':'close','closure_origin':'C_measured_contact_and_finger_box'}
    parts=local.parse_prediction(method,token)
    if parts is None:
        return {**result,'kind':'close' if gate['pass'] else 'reject','closure_origin':'model_GRASP',
                'reason':None if gate['pass'] else 'model_GRASP_outside_finger_box'}
    dp,dr=parts
    if req['vision_arm']=='g0':
        # Deliberate geometry baseline; this branch is excluded from zero-prior D.
        norm=float(np.linalg.norm(dp)); factor=min(1.,.065/norm) if norm else 1.
        goal=local.goal_from_delta(current,dp*factor,dr*factor)
        if norm<=.004 and np.linalg.norm(dr)<=np.deg2rad(4) and gate['pass']:
            return {**result,'kind':'close','closure_origin':'fixed_control_small_residual_and_finger_box'}
    elif method=='D' and index==0:
        first=geometry.first_plane(current,token,points,gate['box'])
        return {**result,'kind':'move' if first['pass'] else 'reject','reason':first['reason'],
                'goal_world':first.get('goal_world'),'interpretation':first}
    else:
        if method=='C':
            transformed=closure_recovery.action_transform('C',token)
            dp,dr=transformed['translation_m'],transformed['rotation_rad']
        goal=local.goal_from_delta(current,dp,dr)
        visual={'target_pixels':gate['target_points'],'between_fingers':bool(gate['in_hand_points']),
                'depth_gate_pass':gate['pass'],'target_direction_EE_xy':gate['target_direction_EE_xy'],
                'predicted_near_saturation':geometry.predicted_saturation(row,points,goal,scale['near_m'])}
        early=local.second_decision(token,visual,method)
        if early['request_closure']:
            return {**result,'kind':'close','closure_origin':'visual_second_step_intercept','interpretation':early}
    admission=local.local_admission(current,goal)
    return {**result,'kind':'move' if admission['pass'] else 'reject',
            'reason':None if admission['pass'] else 'local_motion_bound',
            'goal_world':goal.tolist(),'interpretation':admission}


def serve(campaign):
    root=Path(campaign).resolve(); reg=read(root/'registration.json'); config=reg['learned_policy']; directory=root/'workers/policy'
    os.environ.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        models=Models(reg); projector=native_mask.projector_for(reg)
        warm={m:[models.generate(m,item) for item in config['warmups'][m]] for m in ('D','C')}
        write(directory/'warmup.json',{'calls':warm,'scored':False});write(directory/'ready.json',{'registration_sha256':sha(root/'registration.json'),'wall':time.time()})
        while not (directory/'STOP').exists() and not (root/'STOP').exists():
            for queued in sorted((directory/'queue').glob('*.json')):
                done=directory/'done'/queued.name
                if done.exists():continue
                path=Path(read(queued)['request']);req=read(path);verify_files(req['files'])
                if req['registration_sha256']!=sha(root/'registration.json'):raise ValueError('Wrong policy registration')
                if req['method']=='D' and req['vision_arm']!='g0' and 'G0_world' in req['inputs']:
                    raise ValueError('Prior crossed the zero-prior D request boundary')
                trial=next(t for t in reg['seed_trials'] if t['trial_id']==req['trial_id'])
                if trial['method']!=req['method'] or trial['vision_arm']!=req['vision_arm']:raise ValueError('Wrong policy branch')
                row=(native_mask.request_row(req) if native_mask.configured(reg) else read(req['observation']))
                depth=np.load(req['depth'],allow_pickle=False)
                gate,masks,points=geometry.observe(projector,req['asset'],row,depth)
                render=native_mask.model_views if native_mask.configured(reg) else model_views
                views=render(Image.open(req['rgb']),depth,masks,config['visual_contract']['depth_scale'])
                actual=[path.with_name(req['request_id']+'_'+x+'.png') for x in ('rgb','depth')]
                for im,p in zip(views,actual):im.save(p)
                used=req['vision_arm']!='g0' and req['mode']=='decision' and not (req.get('contact_closure') and gate['pass'])
                visual={'model_inference':False,'images':[],'replaced':False}
                if used:
                    donor=reg['full_stale_donors'].get(req['object_id'],{}).get('images_by_plane',{})
                    if req['vision_arm']=='full_stale':
                        for record in reg['full_stale_donors'][req['object_id']]['records']:verify_files(record['files'])
                    # C is a frozen-interface Spam control; it has up to12 microsteps.
                    visual=(model_images(req['vision_arm'],req['motions_completed'],actual,donor) if req['method']=='D'
                            else {'model_inference':True,'images':list(map(str,actual)),'replaced':False})
                item=lateral_recovery.inference_example(req['inputs'],req['method'],visual['images'] if used else list(map(str,actual)),config['visual_contract'])
                prompt=path.with_name(req['request_id']+'_input.json');write(prompt,{**item,'model_consumed_input':used,'visual_control':visual})
                generated={'prediction':'GRASP','token_ids':[],'policy_seconds':0.,'prompt_sha256':None,'input_token_sha256':None}
                if used:generated=models.generate(req['method'],item)
                elif req['vision_arm']=='g0' and req['mode']=='decision':generated['prediction']=geometry_prediction(req['inputs'])
                try: action=decision(req,generated['prediction'].strip(),gate,points,row,config['visual_contract']['depth_scale'])
                except ValueError as exc:action={'kind':'reject','reason':'invalid_prediction: '+str(exc),'goal_world':None,'closure_origin':None}
                files=set(actual+[prompt]+[Path(p) for p in visual['images']])
                reply={**generated,'ok':True,'request_id':req['request_id'],'method':req['method'],
                    'request_sha256':sha(path),'registration_sha256':req['registration_sha256'],'model_inference':used,
                    'checkpoint':config['checkpoints'][req['method']] if used else None,
                    'visual_control':visual,'finger_box':gate,'action':action,'files':{str(p):sha(p) for p in files}}
                verify_files(req['files']);dest=path.with_name(req['request_id']+'_reply.json')
                if dest.exists():raise RuntimeError('Never repeat or overwrite inference')
                write(dest,reply);write(done,{'reply':str(dest),'sha256':sha(dest)})
            time.sleep(.05)
        write(directory/'exit.json',{'pass':True,'wall':time.time()})
    except BaseException:
        error=traceback.format_exc();write(directory/'FAULT.json',{'error':error});exclusive_stop(root/'STOP',{'reason':'policy_fault','error':error});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');serve(p.parse_args().campaign)
