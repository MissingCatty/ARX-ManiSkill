"""Finish registration after a zero-request configuration precheck failure."""
import argparse
from pathlib import Path
import numpy as np

from arx_project.layout import ROOT,read_json,write_json,sha256,ensure_run_path,now
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard
from .prepare import CONFIG,register


def finish(root,gpu):
    if any((root/name).exists() for name in ['registration.json','started.json','STOP']):
        raise RuntimeError('This tool only finishes an unregistered preparation')
    if not read_json(root/'preparation_error.json')['registration_created'] is False:
        raise ValueError('Missing zero-request preparation failure receipt')
    cfg=read_json(CONFIG);offline=ROOT/cfg['offline_source'];old=guard(offline,completed=True)
    reviewed=read_json(root/'offline_review.json');inputs={};cases=[];checked=0
    if not reviewed['passed'] or reviewed['cases']!=len(old['cases']):raise ValueError('Offline processing was incomplete')
    for case in old['cases']:
        source=Path(case['source_directory']);prior=offline/'cases'/case['id']/'native_rechecked';dest=root/'offline'/case['id']
        coarse=read_json(dest/'coarse.json');fine=read_json(dest/'fine.json');metrics=read_json(dest/'metrics.json')
        if coarse!=read_json(prior/'candidate_pool.json'):raise ValueError('Original coarse pool changed')
        data=np.load(prior/'stages.npz',allow_pickle=False);scores=np.load(source/'graspgenx_raw/scores.npy',allow_pickle=False)
        order=np.argsort(-scores,kind='stable')
        if [c['metadata']['original_native_rank'] for c in fine['candidates']]!=metrics['fine_selected_ranks']:
            raise ValueError('Fine NMS output rank receipt differs')
        for c in fine['candidates']:
            rank=c['metadata']['original_native_rank']
            if (not np.array_equal(np.asarray(c['pose']),data['poses'][rank]) or data['bad'][rank]
                or c['width']!=data['widths'][rank] or c['score']!=scores[order[rank]]
                or c['metadata']['source_generator_index']!=order[rank]):
                raise ValueError('Prepared candidate differs from original accepted geometry or score')
            checked+=1
        for p in [source/'state.json',source/'candidates/candidate_pool.json',prior/'stages.npz',prior/'candidate_pool.json',
                  dest/'fine.json',dest/'coarse.json',source/'graspgenx_raw/scores.npy']:
            inputs[str(p)]=sha256(p)
        if case['id'] in cfg['cases']:cases.append(case)
    write_json(root/'preparation_finish_review.json',{'at':now(),'passed':True,'cases':len(old['cases']),
        'modified_outputs_checked':checked,'reprocessed_NMS':False,'claimed_requests':0,
        'previous_error_receipt_sha256':sha256(root/'preparation_error.json')})
    register(root,cfg,gpu,cases,inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',required=True)
    args=p.parse_args();finish(ensure_run_path(args.root),args.gpu)
