"""Re-NMS the accepted hard-filter outputs, then freeze a matched pilot."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from arx_project.layout import ROOT,now,read_json,write_json,sha256,ensure_run_path
from modules.graspgenx.native_postprocessing.processor import hard_nms,make_pool
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard


CONFIG=ROOT/'configs/experiments/analytic_endpoint_planning.json'


def prepare(root,gpu):
    if root.exists():raise ValueError('Preserve old runs and use a fresh directory')
    cfg=read_json(CONFIG);offline=ROOT/cfg['offline_source'];old=guard(offline,completed=True)
    if not read_json(offline/'independent_review.json')['passed']:raise ValueError('Previous offline review failed')
    root.mkdir(parents=True);inputs={};rows=[];cases=[]
    for case in old['cases']:
        source=Path(case['source_directory']);folder=offline/'cases'/case['id']/'native_rechecked'
        payload=read_json(source/'candidates/candidate_pool.json');previous=read_json(folder/'candidate_pool.json')
        data=np.load(folder/'stages.npz',allow_pickle=False)
        scores=np.asarray([c['score'] for c in payload['candidates']]);order=np.asarray([c['metadata']['source_generator_index'] for c in payload['candidates']])
        coarse,_=hard_nms(data['poses'],scores,data['widths'],~data['bad'],.03,30)
        if coarse!=[c['metadata']['original_native_rank'] for c in previous['candidates']]:
            raise ValueError('Original coarse NMS cannot be reproduced')
        began=time.perf_counter();fine,nms=hard_nms(data['poses'],scores,data['widths'],~data['bad'],.015,15)
        seconds=time.perf_counter()-began
        result={k:data[k] for k in ['poses','widths','axial_shift_m']}
        result.update(selected=fine,nms_parameters=cfg['fine_nms'])
        pool=make_pool(payload,result,order,'native_rechecked')
        for rank,candidate in zip(fine,pool['candidates']):
            if (candidate['score']!=scores[rank] or not np.array_equal(np.asarray(candidate['pose']),data['poses'][rank])
                or data['bad'][rank] or not 0<candidate['width']<=.088+1e-12):
                raise ValueError('Fine NMS changed the accepted hard-filter result')
        dest=root/'offline'/case['id'];write_json(dest/'fine.json',pool);write_json(dest/'coarse.json',previous)
        row={'id':case['id'],'coarse':len(coarse),'fine':len(fine),'hard_filter_survivors':int((~data['bad']).sum()),
            'fine_NMS_seconds':seconds,'coarse_selected_ranks':coarse,'fine_selected_ranks':fine,
            'suppressed_candidates':nms['deferred'],'poses_widths_scores_unchanged':True}
        write_json(dest/'metrics.json',row);rows.append(row)
        for p in [source/'state.json',source/'candidates/candidate_pool.json',folder/'stages.npz',folder/'candidate_pool.json',dest/'fine.json',dest/'coarse.json']:
            inputs[str(p)]=sha256(p)
        if case['id'] in cfg['cases']:cases.append(case)
    if {c['id'] for c in cases}!=set(cfg['cases']):raise ValueError('Pilot fixture list changed')
    write_json(root/'offline_review.json',{'at':now(),'passed':True,'cases':len(rows),'rows':rows,
        'coarse_median':float(np.median([r['coarse'] for r in rows])),
        'fine_median':float(np.median([r['fine'] for r in rows])),
        'original_hard_filters_and_refinements_unchanged':True})
    register(root,cfg,gpu,cases,inputs)


def register(root,cfg,gpu,cases,inputs):
    if (root/'registration.json').exists() or (root/'started.json').exists() or (root/'STOP').exists():
        raise RuntimeError('Do not replace a registered or claimed run')
    for name in ['state.json','closed_state.json']:
        p=root/'warmup'/name;p.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(ROOT/cfg['warmup_source']/name,p);inputs[str(p)]=sha256(p)
    sources={CONFIG,ROOT/'configs/experiments/graspgenx_native_postprocessing.json',ROOT/'configs/modules.json',
        ROOT/'configs/experiments/active_experiment.json',ROOT/'configs/policies/fingertip_table.json'}
    for folder in ['modules/curobo/analytic_ik','modules/curobo/gripper_sweep','modules/curobo/dual_arm',
        'modules/curobo/fingertip_contact','modules/curobo/candidate_io','modules/graspgenx/native_postprocessing',
        'modules/graspgenx/postprocessing','modules/pipeline/experiments/analytic_planning','arx_project']:
        sources.update((ROOT/folder).glob('*.py'))
    for name in ['resident_planner.py','fast_resident_planner.py','checked_native_services.py','short_lift_planner.py']:
        sources.add(ROOT/'modules/maniskill'/name)
    sources.update(ROOT/'modules/curobo/indexed_planning'/name for name in ['binding_serialization.py','stage_timing.py'])
    sources.update(ROOT/'modules/curobo'/name for name in ['arx_adapter.py','scene_cache.py'])
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    sources.update(Path(p) for p in load_current()['configuration_hashes'])
    for p in sources:
        target=root/'source_registered'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    schedule=[]
    for i,case in enumerate(cases):
        modes=cfg['variants'];modes=modes[i%3:]+modes[:i%3]
        sequence=[(m,1) for m in modes]+[(m,2) for m in reversed(modes)]
        schedule.extend({'id':case['id']+'_'+m+'_'+str(repeat),'case_id':case['id'],
            'mode':m,'variant':m+'_'+str(repeat),'repeat':repeat} for m,repeat in sequence)
    (root/'claims').mkdir(exist_ok=True)
    write_json(root/'registration.json',{'at':now(),'config':cfg,'gpu':gpu,'cases':cases,'schedule':schedule,
        'planning_requests':len(schedule),'physical_grasps':0,'simulation_instances':0,
        'source_sha256':{str(p):sha256(p) for p in sources},'input_sha256':inputs,'automatic_retries':False})
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,'total_requests':len(schedule),'results':[]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--gpu',required=True)
    args=p.parse_args();prepare(ensure_run_path(args.root),args.gpu)
