"""Create shared SAM3 masks from RGB and text only, with one resident model."""
import json
import argparse
import shutil
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from prepare_scenes import RUN, PROJECT, sha, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-name',default='segmentation')
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--expand-prompts',action='store_true')
    parser.add_argument('--inherit-from')
    args=parser.parse_args()
    sys.path.insert(0,str(PROJECT.parent/'claro/依赖/SAM3源码'))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    manifest=json.loads((RUN/'scene_selection.json').read_text())
    assert (RUN/'capture_complete.json').exists()
    checkpoint=PROJECT.parent/'claro/权重/sam3.pt'
    start=time.perf_counter()
    model=build_sam3_image_model(device='cuda',checkpoint_path=str(checkpoint),load_from_HF=False,enable_inst_interactivity=False)
    processor=Sam3Processor(model,device='cuda',confidence_threshold=args.threshold)
    torch.cuda.synchronize();load_s=time.perf_counter()-start
    receipts=[]
    for case in [manifest['warmup']]+manifest['cases']:
        source=RUN/'observations'/case['id'];dest=RUN/args.output_name/case['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'receipt.json').exists():
            receipt=json.loads((dest/'receipt.json').read_text());assert receipt['rgb_sha256']==sha(source/'rgb.png');receipts.append(receipt);continue
        inherited=RUN/args.inherit_from/case['id'] if args.inherit_from else None
        if inherited and (inherited/'receipt.json').exists():
            previous=json.loads((inherited/'receipt.json').read_text())
            if previous['success'] and previous.get('confidence_threshold')==args.threshold:
                assert previous['rgb_sha256']==sha(source/'rgb.png')
                for f in ['mask.png','target_rgba.png','overlay.jpg']:
                    shutil.copyfile(inherited/f,dest/f)
                previous['inherited_from']=str(inherited)
                write(dest/'receipt.json',previous);receipts.append(previous)
                print('SAM3 inherited',case['id'],flush=True)
                continue
        frame=Image.open(source/'rgb.png').convert('RGB')
        torch.cuda.synchronize();start=time.perf_counter();selected=None;prompt_receipts=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(frame)
            expanded={'canned_food':['tin can','metal can','container'],'milk':['carton','box','bottle','container'],
                      'ketchup':['container'],'bowl':['dish','container']}
            prompts=case['prompts']+(expanded.get(case['category'],[])+['object'] if args.expand_prompts else [])
            for prompt in prompts:
                state=processor.set_text_prompt(prompt=prompt,state=state)
                scores=state['scores'].detach().float().cpu().numpy()
                masks=state['masks'].detach().cpu().numpy().astype(bool).reshape(-1,frame.height,frame.width)
                prompt_receipts.append({'prompt':prompt,'scores':scores.tolist(),'mask_pixel_counts':masks.sum((1,2)).tolist()})
                if len(scores):
                    index=int(np.argmax(scores));selected=(prompt,float(scores[index]),masks[index]);break
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        mask=selected[2] if selected is not None else np.zeros((frame.height,frame.width),dtype=bool)
        Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mask.png')
        rgba=np.concatenate([np.asarray(frame),mask[...,None].astype(np.uint8)*255],axis=-1)
        Image.fromarray(rgba).save(dest/'target_rgba.png')
        overlay=np.asarray(frame).copy();overlay[mask]=(overlay[mask]*.6+np.array([50,220,80])*.4).astype(np.uint8)
        Image.fromarray(overlay).save(dest/'overlay.jpg',quality=85)
        receipt={'id':case['id'],'rgb_sha256':sha(source/'rgb.png'),'mask_sha256':sha(dest/'mask.png'),
                 'prompt_attempts':prompt_receipts,'selected_prompt':selected[0] if selected else None,'selected_score':selected[1] if selected else None,
                 'mask_pixels':int(mask.sum()),'success':bool(mask.any()),'seconds':seconds,'ground_truth_used':False,
                 'confidence_threshold':args.threshold,'selection':'first registered text prompt with a candidate above threshold, then maximum SAM3 score; no GT matching'}
        write(dest/'receipt.json',receipt);receipts.append(receipt)
        write(RUN/(args.output_name+'_progress.json'),{'completed':len(receipts),'receipts':receipts,'model_reconstruction_trials':0})
        print('SAM3',case['id'],receipt['success'],receipt['mask_pixels'],round(seconds,3),flush=True)
    write(RUN/(args.output_name+'_complete.json'),{'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'load_s':load_s,
                                          'threshold':args.threshold,'expanded_prompts':args.expand_prompts,'inherit_from':args.inherit_from,
                                          'new_observation_inferences':sum('inherited_from' not in x for x in receipts),
                                          'receipts':receipts,'model_load_count':1,'warmup_count':1,'formal_count':50})


if __name__=='__main__':main()
