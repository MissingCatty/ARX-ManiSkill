"""Rebuild the eight archived model inputs on CPU, without loading weights."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path


def main(root):
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoProcessor
    from training.nearfield_arx.processor_check import messages
    from scripts.curobo_arx.data_factory.common import digest
    from .advisor35_diagnose import OLD,TRAIN,Evidence,dump
    torch.set_num_threads(2)
    out=Path(root).resolve()
    if (out/'processor_checks.json').exists():raise FileExistsError('Preserve completed CPU input check')
    ev=Evidence();ev.pin(__file__)
    first=ev.read(out/'first_inputs.json')['requests']
    reg=ev.read(OLD/'registration.json');folder=Path(reg['learned_policy']['checkpoints']['D'])
    for name in ('processor_config.json','processor_contract.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        ev.pin(folder/name)
    source=Path(reg['source_snapshot'])
    for rel in ('training/nearfield_arx/processor_check.py','training/nearfield_arx/learned_policy.py',
                'scripts/curobo_arx/nearfield/advisor31_policy.py','scripts/curobo_arx/nearfield/advisor34_policy.py'):
        ev.pin(source/rel);ev.pin(OLD.parents[1]/rel,hashlib.sha256((source/rel).read_bytes()).hexdigest())
    processor=AutoProcessor.from_pretrained(folder,local_files_only=True,min_pixels=65536,max_pixels=65536)
    results=[]
    for row in first:
        req=Path(row['request_file']);item=ev.read(req.with_name(row['request_id']+'_input.json'))
        chat=messages(item,'D',answer=False)
        assert len(chat)==1 and chat[0]['role']=='user'
        batch=processor.apply_chat_template(chat,tokenize=True,return_dict=True,return_tensors='pt',
                                             add_generation_prompt=True,enable_thinking=False)
        token=digest(batch['input_ids'].tolist())
        assert token==row['input_token_sha256']
        pixel=batch['pixel_values'].contiguous()
        assert pixel.numel()>0 and torch.isfinite(pixel).all()
        results.append({'trial_id':row['trial_id'],'token_hash_matches_record':True,
            'pixel_values_shape':list(pixel.shape),'pixel_values_sha256':hashlib.sha256(pixel.numpy().tobytes()).hexdigest(),
            'image_grid_thw':batch['image_grid_thw'].tolist(),'neural_inferences':0,'model_weights_loaded':False})
    samples={s['id']:s for s in ev.lines(TRAIN/'dataset/paired_samples.jsonl')}
    groups=defaultdict(list)
    for r in ev.lines(TRAIN/'model_rows/D/train.jsonl'):
        groups[r['instruction'].split('Local state: ')[-1]].append(r)
    states=[]
    for state,rs in groups.items():
        states.append({'state':json.loads(state),'rows':len(rs),
            'objects':dict(Counter(samples[r['sample_id']]['object_id'] for r in rs)),
            'labels':dict(Counter(r['output'].split()[0] for r in rs))})
    entry=groups['{"stopped":true,"remaining_corrections":2}']
    dump(out/'processor_checks.json',{'scope':'CPU preprocessing and frozen-source comparison only',
        'checks':results,'unique_pixel_tensors':len({r['pixel_values_sha256'] for r in results}),
        'new_neural_inferences':0,'GPU_used':False,'model_weights_loaded':False,
        'text_token_hash_scope':'input_ids excludes image pixel_values; equal token hashes do not mean equal visual input.',
        'training_prompt_states':states,
        'entry_state_training_labels':[{'object_id':samples[r['sample_id']]['object_id'],
            'sample_id':r['sample_id'],'output':r['output'],'weight':r['sample_weight']} for r in entry],
        'evidence_files':ev.files})
    print(json.dumps({'CPU_inputs_checked':len(results),'unique_pixel_tensors':len({r['pixel_values_sha256'] for r in results}),
                      'training_entry_state_rows':len(entry),'model_weights_loaded':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
