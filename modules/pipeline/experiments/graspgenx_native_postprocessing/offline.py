"""Freeze raw arrays, evaluate hard postprocessing, and verify every output."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
from arx_project.layout import ROOT,now,read_json,sha256,write_json,ensure_run_path
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.postprocessing.geometry import GripperGeometry,RobotPointFilter
from modules.graspgenx.postprocessing.filters import CandidatePostprocessor
from modules.graspgenx.native_postprocessing.processor import Processor,VARIANTS,load_raw,make_pool
from modules.graspgenx.native_postprocessing.kernels import backend

CONFIG=ROOT/'configs/experiments/graspgenx_native_postprocessing.json'


def raw(path):return json.loads(Path(path).read_text())


def guard(root,*,completed=False):
    reg=raw(root/'registration.json')
    if completed and not raw(root/'acceptance.json')['task_complete']:
        raise RuntimeError('Archive validation is only for completed processing')
    for path,digest in reg['source_sha256'].items():
        actual=root/'source_registered'/Path(path).relative_to(ROOT) if completed else Path(path)
        if sha256(actual)!=digest:raise RuntimeError('Frozen source changed: '+str(actual))
    for path,digest in reg['input_sha256'].items():
        if sha256(path)!=digest:raise RuntimeError('Frozen input changed: '+path)
    return reg


def prepare(root):
    if root.exists():raise ValueError('Use a new registration; preserve old results and STOP')
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    current=load_current();cfg=raw(CONFIG);source=ROOT/cfg['source']
    cases=raw(source/'registration.json')['cases']
    if len(cases)!=cfg['offline_cases']:raise ValueError('Frozen cohort changed')
    sources={CONFIG,ROOT/'configs/modules.json',ROOT/'configs/experiments/active_experiment.json',
        ROOT/'configs/policies/fingertip_table.json',ROOT/'simulation/robots/arx/finger_pad_geometry.json'}
    for folder in ['modules/graspgenx/native_postprocessing','modules/graspgenx/postprocessing',
        'modules/pipeline/experiments/graspgenx_native_postprocessing','modules/curobo/dual_arm']:
        sources.update(p for p in (ROOT/folder).glob('*') if p.suffix in ['.py','.cpp'])
    sources.add(ROOT/'modules/graspgenx/arx_frames.py')
    sources.add(ROOT/'modules/zerograsp/server/vendor/zerograsp_mainline_minimal/zerograsp/utils/collision_detector.py')
    sources.update(Path(p) for p in current['configuration_hashes'])
    geometry=GripperGeometry();robot=RobotPointFilter();sources.update(geometry.source_paths+robot.source_paths)
    sources.add(backend()[1]);inputs={}
    for case in cases:
        directory=Path(case['source_directory'])
        for name in ['state.json','camera/depth_m.npy','camera/mask.png','completed_points_world.npy',
            'candidates/candidate_pool.json','graspgenx_raw/grasps_generator_world.npy','graspgenx_raw/scores.npy',
            'graspgenx_raw/inference.json','plan_result.json','result.json']:
            inputs[str(directory/name)]=sha256(directory/name)
    root.mkdir(parents=True)
    for path in sources:
        target=root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(root/'registration.json',{'at':now(),'experiment':'V7_full_native_postprocessing',
        'authorization':cfg['authorization'],'cases':cases,'config':cfg,
        'source_sha256':{str(p):sha256(p) for p in sorted(sources)},'input_sha256':inputs,
        'physics_unchanged':True,'default_activation':False,'physical_grasps':0,'automatic_retries':False,
        'new_perception_calls':0,'planning_request_budget':cfg['planning_requests'],
        'common_cloud_preparation':'2mm voxels; measured robot removed; analytic V2 table owns support plane',
        'native_initial_depth_m':.04,'native_initial_width_m':.088,'native_TCP_endpoint_m':.043,
        'ARX_max_axial_correction_m':.005,'target_ground_truth_used':False})
    write_json(root/'research.md','# GraspGenX完整后处理研究\n\n'
        '## Goal\n从原始候选减少空抓和碰撞候选，保留可规划抓法并降低完整搜索耗时。\n\n'
        '## Success Metric\n五场景规划通过不少于原始控制，后处理加规划累计耗时下降。\n\n'
        '## Constraints\n- max_iterations:2\n- pause_every:never\n- keep_policy:score_improvement\n'
        '- Evaluator: offline.review and planning review\n- guard: original inputs,physics,871,TCP43,V2,real opening limit; no raw fallback\n'
        '- budgets:69 offline observations,20 planning requests,0 physical grasps; no automatic retries\n'
        '- planning RPC1800s deadline supersedes setup shell timeout\n\n'
        '## Current Approach\n原版算法及修正后复检；第二次设计用实际ARX指腹、目标点空抓检测、观测点碰撞。\n\n'
        '## Search Space\n仅两次预定后处理设计，NMS原版30mm/30度与ARX15mm/15度。\n\n'
        '## History\n| Iteration | Change | Result | Status |\n| --- | --- | --- | --- |\n')
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_cases':0,'total_cases':len(cases)})


def summary(rows):
    result={}
    for variant in VARIANTS:
        values=[r['variants'][variant] for r in rows]
        success=[v for r,v in zip(rows,values) if r['historical_success']]
        result[variant]={'cases':len(values),'output_candidates':sum(v['after_nms'] for v in values),
            'before_nms_median':float(np.median([v['before_nms'] for v in values])),
            'after_nms_median':float(np.median([v['after_nms'] for v in values])),
            'nonempty_cases':sum(v['after_nms']>0 for v in values),
            'historical_success_cases':len(success),
            'historical_selected_seed_passes':sum(v['historical_selected']['passes_before_nms'] for v in success),
            'historical_selected_seed_after_nms':sum(v['historical_selected']['kept_after_nms'] for v in success),
            'postprocessing_median_ms':float(np.median([v['seconds'] for v in values])*1000),
            'postprocessing_p90_ms':float(np.percentile([v['seconds'] for v in values],90)*1000)}
    return result


def run(root):
    reg=guard(root)
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Do not repeat a started registration')
    write_json(root/'started.json',{'at':now()})
    model=ArmModel();robot=RobotPointFilter();grippers={s:GripperGeometry(s) for s in ['right','left']}
    clouds=CandidatePostprocessor(grippers,robot,[]);processor=Processor(grippers);rows=[]
    for case in reg['cases']:
        directory=Path(case['source_directory']);out=root/'cases'/case['id'];out.mkdir(parents=True)
        write_json(out/'claimed.json',{'at':now(),'automatic_retry':False})
        payload=raw(directory/'candidates/candidate_pool.json');poses,scores,order=load_raw(directory,payload)
        state=read_json(directory/'state.json');depth=np.load(directory/'camera/depth_m.npy',allow_pickle=False)
        mask=np.asarray(Image.open(directory/'camera/mask.png'))>0
        completed=np.load(directory/'completed_points_world.npy',allow_pickle=False)
        observed,target,observation=clouds.observations(state,depth,mask,completed,model.frames(state))
        result=processor.process(poses,scores,observed,target,state['scene_cuboids']['table'])
        old=raw(directory/'plan_result.json')['selected'];index=int(old['index']);history=raw(directory/'result.json')
        variants={}
        for variant,value in result.items():
            dest=out/variant;dest.mkdir()
            pool=make_pool(payload,value,order,variant);write_json(dest/'candidate_pool.json',pool)
            arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
            for side,checks in value.get('arms',{}).items():arrays.update({side+'_'+k:v for k,v in checks.items()})
            np.savez_compressed(dest/'stages.npz',**arrays)
            metrics={'before_nms':len(value['before_nms']),'after_nms':len(value['selected']),
                'original_ranks':value['selected'],'before_nms_original_ranks':value['before_nms'],
                'raw_generator_indices':order[value['selected']].tolist(),
                'seconds':value['seconds']+observation['seconds'],
                'empty_rejections':int(value['empty'].sum()),'geometry_rejections':int((value['bad']&~value['empty']).sum()),
                'NMS_rejections':len(value['before_nms'])-len(value['selected']),
                'output_width_median_m':float(np.median(value['widths'][value['selected']])) if value['selected'] else None,
                'max_output_axial_correction_m':float(np.max(np.abs(value['axial_shift_m'][value['selected']]))) if value['selected'] else None,
                'historical_selected':{'original_rank':index,'arm':old['arm'],
                    'passes_before_nms':index in value['before_nms'],'kept_after_nms':index in value['selected'],
                    'empty':bool(value['empty'][index]),'refined_width_m':float(value['widths'][index]),
                    'axial_shift_m':float(value['axial_shift_m'][index])}}
            write_json(dest/'metrics.json',metrics);variants[variant]=metrics
        row={'id':case['id'],'raw_count':len(order),'historical_success':bool(history['success']),
            'observation':observation,'variants':variants}
        write_json(out/'result.json',row);rows.append(row)
        write_json(out/'complete.json',{'at':now(),'result_sha256':sha256(out/'result.json')})
        write_json(root/'progress.json',{'at':now(),'phase':'offline','completed_cases':len(rows),'total_cases':len(reg['cases']),
            'current':case['id'],'summaries':summary(rows),'planning_requests':0,'physical_grasps':0})
        print('NATIVE_POSTPROCESS',len(rows),case['id'],len(order),{k:v['after_nms'] for k,v in variants.items()},flush=True)
    guard(root)
    write_json(root/'acceptance.json',{'at':now(),'task_complete':True,'cases':len(rows),
        'raw_candidates':sum(r['raw_count'] for r in rows),'summaries':summary(rows),
        'planning_requests':0,'physical_grasps':0,'default_activated':False})
    review(root)


def review(root):
    reg=guard(root,completed=True);checked=0
    for case in reg['cases']:
        source=Path(case['source_directory']);payload=raw(source/'candidates/candidate_pool.json')
        poses,scores,order=load_raw(source,payload)
        for variant in VARIANTS:
            dest=root/'cases'/case['id']/variant;output=raw(dest/'candidate_pool.json')
            metrics=raw(dest/'metrics.json');stages=np.load(dest/'stages.npz',allow_pickle=False)
            ranks=metrics['original_ranks']
            if len(output['candidates'])!=len(ranks) or ranks!=sorted(set(ranks)):
                raise RuntimeError('Output count/rank corruption')
            for rank,item in zip(ranks,output['candidates']):
                meta=item['metadata'];pose=np.asarray(item['pose']);old=poses[rank]
                delta=pose[:3,3]-old[:3,3];axis=old[:3,2]
                # Saved FP32 rotations have tiny norm errors. Projection onto
                # a nonunit vector requires division by its squared norm.
                axial=float(delta@axis/(axis@axis))
                bound=.005 if variant=='arx_observed' else .04
                if (not np.allclose(pose[:3,:3],old[:3,:3],atol=1e-12,rtol=0)
                    or np.linalg.norm(delta-axial*old[:3,2])>1e-10 or abs(axial)>bound+1e-10
                    or not 0<item['width']<=.088+1e-12 or meta['source_generator_index']!=order[rank]
                    or item['score']!=scores[rank] or meta['original_native_rank']!=rank
                    or not np.allclose(pose,stages['poses'][rank],atol=0,rtol=0)
                    or stages['bad'][rank]):raise RuntimeError('Refinement provenance/geometry corruption')
                if not np.allclose(item['point1'],pose[:3,3]-item['width']/2*pose[:3,1],atol=1e-12,rtol=0):raise RuntimeError('Contact marker width corruption')
                checked+=1
            if output['raw_fallback'] or not output['hard_deletion']:raise RuntimeError('Hidden raw fallback')
    write_json(root/'independent_review.json',{'at':now(),'passed':True,'record_integrity':True,
        'checked_cases':len(reg['cases']),'checked_modified_outputs':checked,
        'raw_array_source_and_score_order_verified':True,'hard_deletion_verified':True,
        'width_and_axial_limits_verified':True,'physics_modified':False,'physical_grasps':0,
        'processing_source_archive_verified':True,'review_source_sha256':sha256(__file__)})
    progress=raw(root/'progress.json');progress.update(phase='offline_complete',at=now());write_json(root/'progress.json',progress)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','review'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root)
    else:
        try:run(root) if args.stage=='run' else review(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
