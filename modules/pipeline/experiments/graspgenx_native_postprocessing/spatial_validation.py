"""Matched CPU full-cloud/certified-crop tests on all 69 frozen observations."""
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
from arx_project.layout import ROOT,now,write_json,read_json,sha256
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.native_postprocessing.processor import load_raw
from modules.graspgenx.native_postprocessing.kernels import native_detect,backend
from modules.graspgenx.native_postprocessing.spatial_crop import detect,relevant_points
from modules.graspgenx.postprocessing.filters import CandidatePostprocessor
from modules.graspgenx.postprocessing.geometry import GripperGeometry,RobotPointFilter


def run():
    root=ROOT/'tmp/graspgenx_native_postprocessing/spatial69_20261127_01'
    if root.exists():raise ValueError('Do not repeat registered CPU comparisons')
    source=ROOT/'tmp/graspgenx_native_postprocessing/offline69_20261127_01'
    reg=json.loads((source/'registration.json').read_text())
    sources={str(Path(__file__)):sha256(__file__),str(backend()[1]):sha256(backend()[1])}
    for name in ['spatial_crop.py','kernels.py','kernel.cpp','processor.py']:
        p=ROOT/'modules/graspgenx/native_postprocessing'/name;sources[str(p)]=sha256(p)
    root.mkdir(parents=True)
    for name in sources:
        out=root/'source_registered'/Path(name).relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,out)
    write_json(root/'registration.json',{'at':now(),'cases':reg['cases'],'source_sha256':sources,
        'input_sha256':reg['input_sha256'],'authorization':'优化当前后处理的计算实现；不改变候选输出',
        'comparisons':69,'planning_requests':0,'physical_grasps':0,'iterations':1,
        'strict_guard':'identical full-cloud collision masks,counts,widths,depths and archived output stages',
        'schedule':'AB then BA alternation; same CPU thread limits; no reordering of retained points'})
    model=ArmModel();clouds=CandidatePostprocessor({s:GripperGeometry(s) for s in ['right','left']},RobotPointFilter(),[])
    results=[]
    for ordinal,case in enumerate(reg['cases']):
        p=Path(case['source_directory']);state=read_json(p/'state.json');payload=json.loads((p/'candidates/candidate_pool.json').read_text())
        poses,_,_=load_raw(p,payload)
        observed,target,_=clouds.observations(state,np.load(p/'camera/depth_m.npy'),np.asarray(Image.open(p/'camera/mask.png'))>0,
            np.load(p/'completed_points_world.npy'),model.frames(state))
        points=np.concatenate([target,observed]);outputs={};timing={}
        for mode in (['full','crop'] if ordinal%2==0 else ['crop','full']):
            began=time.perf_counter();outputs[mode]=(native_detect if mode=='full' else detect)(poses,points);timing[mode]=time.perf_counter()-began
        for name in outputs['full']:np.testing.assert_array_equal(outputs['full'][name],outputs['crop'][name],err_msg=name)
        archived=np.load(source/'cases'/case['id']/'native/stages.npz')
        actual=outputs['crop'];stroke=(actual['widths']>.088+1e-12)|(actual['widths']<=0)
        np.testing.assert_array_equal(actual['bad']|stroke,archived['bad'])
        np.testing.assert_array_equal(actual['counts'],archived['cavity_counts'])
        np.testing.assert_array_equal(actual['widths'],archived['widths'])
        np.testing.assert_array_equal(actual['depths']-.04,archived['axial_shift_m'])
        kept,ids=relevant_points(poses,points)
        row={'case':case['id'],'candidate_rows':len(poses),'original_points':len(points),'cropped_points':len(kept),
            'full_seconds':timing['full'],'crop_seconds':timing['crop'],'all_outputs_exactly_equal':True}
        write_json(root/'cases'/(case['id']+'.json'),row);results.append(row)
        write_json(root/'progress.json',{'at':now(),'completed_cases':len(results),'total_cases':69,'phase':'running'})
        print('SPATIAL',len(results),case['id'],len(points),len(kept),timing,flush=True)
    for name,digest in sources.items():
        if sha256(name)!=digest:raise RuntimeError('Registered computation changed')
    for name,digest in reg['input_sha256'].items():
        if sha256(name)!=digest:raise RuntimeError('Frozen input changed')
    a=np.array([r['full_seconds'] for r in results]);b=np.array([r['crop_seconds'] for r in results])
    proof={'at':now(),'passed':True,'record_integrity':True,'cases':69,'candidate_rows':sum(r['candidate_rows'] for r in results),
        'all_detector_outputs_exactly_equal':True,'archived_stage_outputs_exactly_equal':True,
        'full_total_seconds':float(a.sum()),'crop_total_seconds':float(b.sum()),'saving_fraction':float(1-b.sum()/a.sum()),
        'full_median_seconds':float(np.median(a)),'crop_median_seconds':float(np.median(b)),
        'full_p90_seconds':float(np.percentile(a,90)),'crop_p90_seconds':float(np.percentile(b,90)),
        'point_reduction_median':float(np.median([1-r['cropped_points']/r['original_points'] for r in results])),
        'results':results,'new_IK_calls':0,'planning_requests':0,'physical_grasps':0}
    write_json(root/'acceptance.json',proof)
    write_json(root/'progress.json',{'at':now(),'completed_cases':69,'total_cases':69,'phase':'complete'})


if __name__=='__main__':
    try:run()
    except BaseException:
        root=ROOT/'tmp/graspgenx_native_postprocessing/spatial69_20261127_01'
        if root.exists():write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False})
        raise
