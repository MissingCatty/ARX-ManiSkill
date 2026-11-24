"""Auditable deterministic G0 and independent trained/foundation inference models."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from training.nearfield_arx import learned_policy, lateral_recovery, closure_recovery, entry_guard
from .closedloop30_policy import validate_checkpoints
from .attribution_spec import ARMS, bind_arm, identity, geometry_prediction, verify_generation


class Models(learned_policy.Models):
    def __init__(self, reg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        from safetensors.torch import load_file
        from training.nearfield_arx.visual_fit import projector_module
        validate_checkpoints(reg)
        self.reg=reg;self.models={};self.processors={};self.by_arm={};torch.set_num_threads(4)
        folder=Path(reg['learned_policy']['checkpoints']['D'])
        self.processors['D']=AutoProcessor.from_pretrained(folder,local_files_only=True,min_pixels=65536,max_pixels=65536)
        for arm in ('trained','base'):
            model=AutoModelForImageTextToText.from_pretrained(reg['learned_policy']['base_model'],
                local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
            if arm=='trained': model=PeftModel.from_pretrained(model,folder,is_trainable=False)
            name,merger=projector_module(model);merger.float()
            if arm=='trained':
                if name!=read(folder/'arx_policy_contract.json')['projector_module']:
                    raise ValueError('Trained projector identity changed')
                merger.load_state_dict(load_file(folder/'projector.safetensors'),strict=True)
            elif hasattr(model,'peft_config') and model.peft_config:
                raise ValueError('Foundation baseline unexpectedly contains a LoRA adapter')
            model.requires_grad_(False);model.eval();model.config.use_cache=True
            self.by_arm[arm]=model.to('cuda')

    def generate_arm(self, arm, item, inputs=None):
        if arm=='g0':
            if inputs is None:raise ValueError('G0 control requires the fresh measured pose')
            start=time.perf_counter();token=geometry_prediction(inputs)
            return {'prediction':token,'token_ids':[],'prompt_tokens':0,
                    'policy_seconds':time.perf_counter()-start,'prompt_sha256':None,'input_token_sha256':None,
                    **identity(self.reg,arm)}
        self.models['D']=self.by_arm[arm]
        return {**super().generate('D',item),**identity(self.reg,arm)}


def serve(campaign):
    root=Path(campaign).resolve();reg=read(root/'registration.json');policy=reg['learned_policy']
    directory=root/'workers/policy'
    os.environ.update(CUDA_VISIBLE_DEVICES=policy['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from PIL import Image
    from training.nearfield_arx.projection import Projector,model_views
    try:
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        models=Models(reg);projector=Projector(policy['robot_usd']);warmups=[]
        for arm in ('trained','base'):
            for item in policy['warmups']['D']:
                warmups.append({'arm':arm,'sample_id':item['sample_id'],**models.generate_arm(arm,item)})
        write(directory/'warmup.json',{'calls':warmups,'physical_trials':0})
        write(directory/'ready.json',{'registration_sha256':sha(root/'registration.json'),
            'gpu_uuid':policy['gpu_uuid'],'identities':{a:identity(reg,a) for a in ARMS},
            'warmup_calls':len(warmups),'wall':time.time()})
        while not (directory/'STOP').exists() and not (root/'STOP').exists() and time.time()<reg['deadline_unix']:
            for queued in sorted((directory/'queue').glob('*.json')):
                done=directory/'done'/queued.name
                if done.exists():continue
                request_path=Path(read(queued)['request']);request=read(request_path)
                if request['registration_sha256']!=sha(root/'registration.json'):
                    raise ValueError('Request belongs to another registration')
                arm=bind_arm(reg,request);verify_files(request['files'])
                row=read(request['observation']);depth=np.load(request['depth'],allow_pickle=False)
                masks,projection,_=projector.project(request['asset'],row,depth)
                images=model_views(Image.open(request['rgb']),depth,masks,policy['visual_contract']['depth_scale'])
                paths=[request_path.with_name(request['request_id']+'_'+n+'.png') for n in ('rgb','depth')]
                for path,im in zip(paths,images):im.save(path)
                item=lateral_recovery.inference_example(request['inputs'],'D',[str(p) for p in paths],policy['visual_contract'])
                prompt=request_path.with_name(request['request_id']+'_input.json')
                write(prompt,{'instruction':item['instruction'],'images':item['images'],'projection':projection,
                              'generator':arm,'model_consumed_input':arm!='g0'})
                result=models.generate_arm(arm,item,request['inputs'])
                try:closure_recovery.parse_action('D',result['prediction'].strip())
                except ValueError as exc:
                    guard={'version':entry_guard.VERSION,'pass':False,'reason':'invalid_model_action','error':str(exc)}
                else:
                    guard=entry_guard.action_admission(request,result['prediction'],projector,
                        policy['visual_contract']['depth_scale'],lambda p:np.load(p,allow_pickle=False),read,None)
                guard.update(request_id=request['request_id'],request_sha256=sha(request_path),prediction=result['prediction'].strip())
                verify_files(request['files'])
                reply={**result,'ok':True,'method':'D','request_id':request['request_id'],
                    'request_sha256':sha(request_path),'registration_sha256':request['registration_sha256'],
                    'files':{str(p):sha(p) for p in paths+[prompt]},'projection':projection,'action_admission':guard}
                verify_generation(reg,request,reply)
                reply_path=request_path.with_name(request['request_id']+'_reply.json')
                if reply_path.exists():raise RuntimeError('No reply overwrite or retries')
                write(reply_path,reply);write(done,{'request':str(request_path),'reply_sha256':sha(reply_path),'generator':arm})
            time.sleep(.05)
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        write(directory/'exit.json',{'pass':True,'wall':time.time()})
    except BaseException:
        error=traceback.format_exc();write(directory/'FAULT.json',{'error':error,'wall':time.time()})
        exclusive_stop(root/'STOP',{'reason':'attribution_policy_infrastructure_failure','error':error});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');serve(p.parse_args().campaign)
