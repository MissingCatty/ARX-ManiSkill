"""ARX pilot adaptation of LiLo-VLA's regression head on real wrist teacher data.

The released vision/language backbone and proprio projector stay frozen. This
is an interaction-head pilot, not a claim of full LoRA training or generalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'servers/lilo_vla'))
from backend import Policy
from interaction_contract import model_semantics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--initial-adapter',type=Path,help='Same frozen base/projector; initialize both comparison heads from this adapter')
    args=parser.parse_args();dataset=args.dataset.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'features').mkdir()
    torch.manual_seed(20261026);rng=np.random.default_rng(20261026)
    specification=json.loads((dataset/'dataset.json').read_text())
    statistics=json.loads((dataset/'dataset_statistics.json').read_text())
    key=next(iter(statistics));stats=statistics[key]['action']
    low,high=np.asarray(stats['q01']),np.asarray(stats['q99']);mask=np.asarray(stats['mask'])
    policy=Policy(args.checkpoint,args.initial_adapter)
    initial_arx_finetuned=policy.metadata['arx_finetuned']
    policy.stats=statistics;policy.key=key;policy.model.norm_stats=statistics
    convention=specification.get('state_convention','object_relative_v1')
    policy.metadata.update(state_convention=convention,dataset_key=key)
    train=[];validation=[];cache_manifest=[]
    for episode_index,episode in enumerate(specification['episodes']):
        folder=dataset/episode['folder'];data=np.load(folder/'trajectory.npz')
        for index in episode['chunk_starts']:
            item=episode['items'][index];path=dataset/item['image']
            prediction,features=policy.predict(Image.open(path),data['state'][index],episode['instruction'],return_features=True)
            actions=data['action'][index:index+8]
            target=np.clip(np.where(mask,2*(actions-low)/(high-low+1e-8)-1,actions),-1,1).astype(np.float32)
            label=f'{episode_index:03d}_{index:04d}'
            np.savez_compressed(out/'features'/f'{label}.npz',features=features,target=target)
            entry=(torch.tensor(features[0],dtype=torch.float32),torch.tensor(target,dtype=torch.float32))
            (validation if episode['split']=='validation' else train).append(entry)
            cache_manifest.append({'episode':episode['folder'],'frame':index,'split':episode['split'],
                'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'feature_file':f'features/{label}.npz','inference_seconds':prediction['inference_seconds']})
            if len(cache_manifest)%16==0:print('ENCODED',len(cache_manifest),flush=True)
    (out/'feature_manifest.json').write_text(json.dumps(cache_manifest,indent=2)+'\n')
    if not train:raise ValueError('No training chunks')
    head=policy.head.float()
    # Free frozen GPU weights before training the 150M-parameter action head.
    policy.model.to('cpu');policy.projector.to('cpu');torch.cuda.empty_cache()
    head.train();optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
    def evaluate(entries):
        if not entries:return None
        head.eval();loss_sum=0.;element_count=0
        with torch.no_grad():
            for start in range(0,len(entries),args.batch_size):
                batch=entries[start:start+args.batch_size]
                x=torch.stack([x for x,y in batch]).cuda();y=torch.stack([y for x,y in batch]).cuda()
                loss_sum+=float(torch.nn.functional.l1_loss(head.predict_action(x),y,reduction='sum'))
                element_count+=y.numel()
        return loss_sum/element_count
    baseline={'train_l1':evaluate(train),'validation_l1':evaluate(validation)}
    report={'adaptation':'ARX action regression head only; frozen LiLo vision/language backbone and proprio projector',
        'scope':specification.get('scope','Single-object bottle pilot; held-out trajectory prediction is separate from new physical success.'),
        'state_convention':convention,'initial_adapter':str(args.initial_adapter.resolve()) if args.initial_adapter else None,
        'action_alignment_version':specification.get('action_alignment_version','frame_command_v1'),
        'interaction_contracts':specification.get('interaction_contracts',{}),
        **({'interaction_semantics':model_semantics(specification['state_convention'])} if specification.get('interaction_contracts') else {}),
        'exclude_controller_closure_dwell_and_hold_chunk_starts':specification.get('exclude_controller_closure_dwell_and_hold_chunk_starts',False),
        'train_objects':specification.get('train_objects'),'validation_objects':specification.get('validation_objects'),
        'training_steps':args.steps,'batch_size':args.batch_size,'seed':20261026,
        'initial_arx_finetuned':initial_arx_finetuned,
        'initial_action_head_sha256':hashlib.sha256((args.initial_adapter/'action_head.pt').read_bytes()).hexdigest() if args.initial_adapter else None,
        'base_checkpoint_revision':'f54c56709004f6680d2ce8bc9e06d8cc96567ad8',
        'train_chunks':len(train),'validation_chunks':len(validation),'baseline':baseline,'history':[]}
    # Retain the released head if adaptation does not improve held-out error.
    best=baseline['validation_l1'] if validation else baseline['train_l1']
    best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    report['selected_step']=0
    for step in range(args.steps):
        indices=rng.integers(0,len(train),size=min(args.batch_size,len(train)))
        batch=[train[i] for i in indices]
        x=torch.stack([x for x,y in batch]).cuda();y=torch.stack([y for x,y in batch]).cuda()
        head.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.l1_loss(head.predict_action(x),y)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        optimizer.step()
        if (step+1)%25==0 or step==0 or step+1==args.steps:
            row={'step':step+1,'train_l1':evaluate(train),'validation_l1':evaluate(validation)}
            report['history'].append(row);print(json.dumps(row),flush=True)
            selection=row['validation_l1'] if validation else row['train_l1']
            if selection<best:
                best=selection;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                report['selected_step']=step+1
            (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(best_state,out/'action_head.pt')
    shutil.copy2(dataset/'dataset_statistics.json',out/'dataset_statistics.json')
    manifest={k:report[k] for k in ('base_checkpoint_revision','adaptation','scope','train_chunks','validation_chunks','selected_step','state_convention','initial_adapter',
        'action_alignment_version','exclude_controller_closure_dwell_and_hold_chunk_starts',
        'train_objects','validation_objects','training_steps','batch_size','seed')}
    if specification.get('interaction_contracts'):
        manifest['interaction_contracts'] = report['interaction_contracts']
        manifest['interaction_semantics'] = report['interaction_semantics']
    manifest['arx_finetuned']=bool(report['selected_step']>0 or initial_arx_finetuned)
    manifest['initial_action_head_sha256']=report['initial_action_head_sha256']
    manifest['dataset_manifest_sha256']=hashlib.sha256((dataset/'dataset.json').read_bytes()).hexdigest()
    manifest['physical_execution_validation']='pending; training loss does not establish an executed grasp'
    (out/'adapter_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('ARX interaction-head adapter saved:',out,flush=True)


if __name__=='__main__':main()
