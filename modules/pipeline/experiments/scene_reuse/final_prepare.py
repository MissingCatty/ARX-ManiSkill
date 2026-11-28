"""Freeze all 68 originally matched scenes for one final paired comparison."""
import argparse
from pathlib import Path
import shutil

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from .prepare import prepare as base_prepare

CONFIG=ROOT/'configs/experiments/scene_reuse_final.json'


def prepare(root):
    cfg=read_json(CONFIG); previous=ROOT/'tmp/state_updates/scene_reuse_slots5_20261128_01'
    accepted=read_json(previous/'acceptance.json')
    if (not accepted['record_integrity'] or not accepted['exact_selection_and_paths']
            or not read_json(previous/'services_stopped.json')['all_owned_exited']):
        raise RuntimeError('Finish and independently review the third pilot before the final comparison')
    gpu=ROOT/'tmp/state_updates/scene_reuse_gpu5_20261128_01'
    if not read_json(gpu/'acceptance.json')['optional_implementation_accepted']:
        raise RuntimeError('The GPU-sharing baseline was not accepted')
    cfg['slot_updates_enabled']=accepted['optional_implementation_accepted']
    origin=ROOT/cfg['source_run']; old=read_json(origin/'registration.json')
    original=read_json(origin/'acceptance.json'); progress=read_json(origin/'progress.json')
    excluded=set(original['interrupted_timing_cases_excluded'])
    ids=sorted({r['id'] for r in progress['results'] if not r['common_precondition_unexecuted']
                and not r.get('interrupted') and r['id'] not in excluded})
    if len(ids)!=68: raise RuntimeError('The original paired cohort changed')
    base_prepare(root); reg=read_json(root/'registration.json'); inputs=reg['input_sha256']; cases=[]
    for cid in ids:
        item=next(c for c in old['cases'] if c['id']==cid); source=ROOT/item['input_directory']; target=root/'inputs'/cid
        for name in ['state.json','candidates/candidate_pool.json']:
            p=source/name; out=target/name
            if sha256(p)!=old['input_sha256'][str(p)]: raise ValueError('Original frozen input changed')
            if not out.exists(): out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,out)
            if sha256(out)!=sha256(p): raise ValueError('Final input copy mismatch')
            inputs[str(p)]=sha256(p); inputs[str(out)]=sha256(out)
        p=origin/'trials'/cid/'baseline/plan_result.json'
        claim=read_json(origin/'claims'/(cid+'_baseline.json'))
        if sha256(p)!=claim['files_sha256']['plan_result.json']: raise ValueError('Original arm-selection receipt changed')
        inputs[str(p)]=sha256(p); inputs[str(origin/'claims'/(cid+'_baseline.json'))]=sha256(origin/'claims'/(cid+'_baseline.json'))
        forced=read_json(p).get('forced_arm')
        out=root/'offline'/cid/'fine.json'
        if not out.exists(): out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(target/'candidates/candidate_pool.json',out)
        inputs[str(out)]=sha256(out)
        cases.append({'id':cid,'source_directory':str(target),'original_input':str(source.relative_to(ROOT)),
                      'forced_arm':forced})
    schedule=[]
    for index,case in enumerate(cases):
        for mode in (['control','cached'] if index%2==0 else ['cached','control']):
            schedule.append({'id':case['id']+'_'+mode+'_1','case_id':case['id'],
                             'mode':mode,'variant':mode+'_1','repeat':1})
    if len(schedule)!=136: raise RuntimeError('Final planning budget changed')
    for prior in [previous,gpu]:
        for name in ['registration.json','acceptance.json','analysis.json','independent_review.json','services_stopped.json']:
            p=prior/name; inputs[str(p)]=sha256(p)
    out=root/'source_registered'/CONFIG.relative_to(ROOT); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(CONFIG,out)
    reg['source_sha256'][str(CONFIG)]=sha256(CONFIG)
    reg.update(config=cfg,experiment='V8_final_scene_update_combination',cases=cases,schedule=schedule,
               planning_requests=136,excluded_interrupted_cases=sorted(excluded),input_sha256=inputs)
    write_json(root/'registration.json',reg)
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,'total_requests':136,'results':[]})
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'stage':cfg['stage'],'phase':'registered'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    prepare(ensure_run_path(parser.parse_args().root))
