"""One frozen D model; an explicitly registered stale-image control."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from training.nearfield_arx.visual_causality import SingleD
from training.nearfield_arx import lateral_recovery,closure_recovery,entry_guard
from training.nearfield_arx.projection import Projector,model_views
from .attribution_spec import geometry_prediction
from .visual_grounding_spec import binding,first_decision,alignment,verify_reply


def serve(campaign):
    root=Path(campaign).resolve();reg=read(root/'registration.json');p=reg['learned_policy'];directory=root/'workers/policy'
    os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        verify_files({**reg['execution_source_files'],**reg['dependencies']});model=SingleD(reg)
        projector=Projector(p['robot_usd'])
        warmup=[model.generate(item) for item in p['warmups']['D']]
        write(directory/'warmup.json',{'calls':warmup,'scored':False})
        write(directory/'ready.json',{'gpu_uuid':p['gpu_uuid'],'checkpoint':p['checkpoints']['D'],'registration_sha256':sha(root/'registration.json'),'wall':time.time()})
        while not (root/'STOP').exists() and not (directory/'STOP').exists() and time.time()<reg['deadline_unix']:
            for queued in sorted((directory/'queue').glob('*.json')):
                done=directory/'done'/queued.name
                if done.exists():continue
                path=Path(read(queued)['request']);request=read(path);trial=binding(reg,request)
                if request['registration_sha256']!=sha(root/'registration.json'):raise ValueError('Wrong registration')
                verify_files(request['files']);row=read(request['observation']);depth=np.load(request['depth'],allow_pickle=False)
                masks,projection,_=projector.project(request['asset'],row,depth)
                images=model_views(Image.open(request['rgb']),depth,masks,p['visual_contract']['depth_scale'])
                actual=[path.with_name(request['request_id']+'_'+name+'.png') for name in ('rgb','depth')]
                for im,file in zip(images,actual):im.save(file)
                donor=reg['visual_donors'][trial['object_id']]
                match=alignment(request,donor) if first_decision(request) else None
                applicable=match is None or match['pass']
                swapped=applicable and trial['vision_arm']=='stale_first' and first_decision(request)
                supplied=donor['images'] if swapped else list(map(str,actual))
                if swapped:verify_files({x:reg['dependencies'][x] for x in supplied})
                item=lateral_recovery.inference_example(request['inputs'],'D',supplied,p['visual_contract'])
                proof={'applicable':bool(applicable),'replaced':bool(swapped),'alignment':match,
                       'model_images':supplied,'actual_observation':request['observation'],
                       'replacement_scope':'First decision only; later observations fresh; no physical state, mask truth or safety gate replacement'}
                prompt=path.with_name(request['request_id']+'_input.json')
                write(prompt,{'instruction':item['instruction'],'images':item['images'],'visual_control':proof,
                              'model_consumed_input':applicable and trial['vision_arm']!='g0'})
                if not applicable:
                    result={'prediction':'','token_ids':[],'policy_seconds':0.,'prompt_tokens':0,'prompt_sha256':None,'input_token_sha256':None}
                elif trial['vision_arm']=='g0':
                    result={'prediction':geometry_prediction(request['inputs']),'token_ids':[],'policy_seconds':0.,'prompt_tokens':0,'prompt_sha256':None,'input_token_sha256':None}
                else:result=model.generate(item)
                try:closure_recovery.parse_action('D',result['prediction'].strip())
                except ValueError as exc:guard={'version':entry_guard.VERSION,'pass':False,'reason':'invalid_model_action','error':str(exc)}
                else:guard=entry_guard.action_admission(request,result['prediction'],projector,p['visual_contract']['depth_scale'],lambda f:np.load(f,allow_pickle=False),read,None)
                guard.update(request_id=request['request_id'],request_sha256=sha(path),prediction=result['prediction'].strip())
                verify_files(request['files']);used=bool(applicable and trial['vision_arm']!='g0')
                reply={**result,'ok':True,'method':'D','request_id':request['request_id'],'request_sha256':sha(path),
                    'registration_sha256':request['registration_sha256'],'vision_arm':trial['vision_arm'],'visual_control':proof,
                    'model_inference':used,'checkpoint':p['checkpoints']['D'] if used else None,
                    'projection':projection,'action_admission':guard,
                    'files':{str(f):sha(f) for f in set(actual+[prompt]+[Path(x) for x in supplied])}}
                verify_reply(reg,request,reply);dest=path.with_name(request['request_id']+'_reply.json')
                if dest.exists():raise RuntimeError('No inference retry or overwritten reply')
                write(dest,reply);write(done,{'request':str(path),'reply_sha256':sha(dest)})
            time.sleep(.05)
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        write(directory/'exit.json',{'pass':True,'wall':time.time()})
    except BaseException:
        error=traceback.format_exc();write(directory/'FAULT.json',{'error':error})
        exclusive_stop(root/'STOP',{'reason':'visual_grounding_policy_failure','error':error});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');serve(p.parse_args().campaign)
