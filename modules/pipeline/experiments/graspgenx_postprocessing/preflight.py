"""Compare the batch CPU kernel against the independent NumPy reference."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.postprocessing.geometry import GripperGeometry,RobotPointFilter,poses_from_candidates
from modules.graspgenx.postprocessing.filters import CandidatePostprocessor
from modules.graspgenx.postprocessing.native_geometry import backend
from .offline import raw


def main(root):
    if root.exists():raise ValueError('Keep previous component preflight evidence')
    source=ROOT/'tmp/indexed_grasp_planning/native_batch_69_20261126_01/registration.json'
    case=read_json(source)['cases'][0];directory=Path(case['source_directory'])
    cfg=read_json(ROOT/'configs/experiments/graspgenx_postprocessing.json')
    state=read_json(directory/'state.json');pool=directory/'candidates/candidate_pool.json'
    candidates=raw(pool)['candidates']
    sources=list((ROOT/'modules/graspgenx/postprocessing').glob('*.py'))
    sources+=list((ROOT/'modules/graspgenx/postprocessing').glob('*.cpp'))+[Path(__file__)]
    hashes={str(p):sha256(p) for p in sources}
    root.mkdir(parents=True)
    write_json(root/'registration.json',{'at':now(),'case':case['id'],'source_sha256':hashes,
        'candidate_pool_sha256':sha256(pool),'formal_slots':0,'IK_calls':0,'physical_grasps':0})
    begin=time.perf_counter();model=ArmModel();robot=RobotPointFilter()
    grippers={side:GripperGeometry(side) for side in ['right','left']}
    engine=CandidatePostprocessor(grippers,robot,cfg['offline']['thresholds']);backend()
    points,target,obs=engine.observations(state,np.load(directory/'camera/depth_m.npy'),
        np.asarray(Image.open(directory/'camera/mask.png'))>0,
        np.load(directory/'completed_points_world.npy'),model.frames(state))
    _,eef=poses_from_candidates(candidates);equivalence={}
    for side,g in grippers.items():
        table,_=g.table_rejections(eef,state['scene_cuboids']['table'])
        started=time.perf_counter();actual=g.point_rejections(eef,points,skip=table)
        collision_batch=time.perf_counter()-started
        started=time.perf_counter();expected=g.point_rejections_reference(eef,points,skip=table)
        collision_reference=time.perf_counter()-started
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        started=time.perf_counter();actual_c=g.cavity_counts(eef,target)
        cavity_batch=time.perf_counter()-started
        started=time.perf_counter();expected_c=g.cavity_counts_reference(eef,target)
        cavity_reference=time.perf_counter()-started
        np.testing.assert_array_equal(actual_c,expected_c)
        equivalence[side]={'candidate_count':len(eef),'all_masks_and_counts_identical':True,
            'point_batch_seconds':collision_batch,'point_reference_seconds':collision_reference,
            'cavity_batch_seconds':cavity_batch,'cavity_reference_seconds':cavity_reference}
        print(side,equivalence[side],flush=True)
    result=engine.process(candidates,state,points,target)
    selected=raw(directory/'plan_result.json')['selected'];selected_index=int(selected['index'])
    if not all(selected_index in v['input_indices'] for v in result['variants'].values()):
        raise ValueError('First saved successful grasp was removed by postprocessing')
    if any(sha256(p)!=digest for p,digest in hashes.items()):raise RuntimeError('Component source changed during preflight')
    write_json(root/'acceptance.json',{'at':now(),'passed':True,'case':case['id'],
        'raw_candidates':len(candidates),'equivalence':equivalence,'geometry_seconds':result['geometry_seconds'],
        'source_sha256':hashes,'observations':obs,'all_variants_keep_original_selection':True,
        'variants':{name:{'representatives':len(v['representatives']),'deferred':len(v['deferred']),
            'seconds':v['postprocessing_seconds']} for name,v in result['variants'].items()},
        'seconds_including_reference_checks':time.perf_counter()-begin,'new_IK_calls':0,'physical_grasps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(ensure_run_path(parser.parse_args().root))
