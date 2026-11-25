"""Check the unchanged reconstruction worker on three saved ARX head views."""
import argparse
import os
from pathlib import Path
import shutil
import statistics
import time
from types import SimpleNamespace

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT,now,sha256

CASES=['M0001_S1_open_apple','M0030_S2_edge_peach','M0086_S5_occluded_bell_pepper']


def run(root,gpu):
    import numpy as np
    import torch
    from modules.pipeline.experiments.sam3_zerograsp_comparison.cohort import check_pair
    from modules.rayst3r.fast_inference.service import create
    from modules.rayst3r.fast_inference.adapter import FastAdapter
    from modules.rayst3r.fast_inference.benchmark import cloud_difference
    if root.exists():raise RuntimeError('Interface check cannot be repeated in the same directory')
    reg=read_json(ROOT/'tmp/sam3_zerograsp_comparison/v6_paired_20261125_01/registration.json')
    selected=[next(pair for pair in reg['cohort']['pairs'] if pair['id']==case) for case in CASES]
    root.mkdir(parents=True)
    atomic_json(root/'registration.json',{'at':now(),'cases':CASES,'gpu':gpu,
        'profile_sha256':sha256(ROOT/'configs/experiments/rayst3r_fast_inference_profile.json'),
        'source_sha256':sha256(__file__),'model_load_budget':1,'warmup_requests':2,'measured_requests':6,
        'physical_trials':0,'planning_calls':0})
    torch.set_num_threads(4)
    worker=create(SimpleNamespace(root=root))
    fast=worker.model
    shared=SimpleNamespace(model=fast.native_model,dino=fast.dino,scene_function=fast.scene_function,
        configuration=dict(fast.configuration))
    baseline=FastAdapter(baseline=shared,variant='baseline')
    models={'baseline':baseline,'accelerated':fast}
    def request(pair,variant,destination):
        source=check_pair(pair)
        for name in ['state.json','camera/color.png','camera/mask.png','camera/depth_m.npy']:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
            if sha256(target)!=pair['input_hashes'][name]:raise RuntimeError('Interface input changed')
        worker.model=models[variant]
        torch.cuda.synchronize();begin=time.perf_counter()
        result=worker.request(destination)
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        camera=np.load(destination/'completed_points_camera.npy',allow_pickle=False)
        world=np.load(destination/'completed_points_world.npy',allow_pickle=False)
        transform=np.asarray(read_json(destination/'state.json')['camera_world_optical'])
        expected=camera@transform[:3,:3].T+transform[:3,3]
        if (not result['success'] or result['actual_novel_views']!=21 or len(camera)<20
                or not np.isfinite(camera).all() or not np.isfinite(world).all()
                or not np.allclose(expected,world,atol=1e-6,rtol=0)):
            raise RuntimeError('Reconstruction worker output contract failed')
        return camera,{'request_seconds':seconds,'adapter_report_seconds':result['wall_seconds'],
            'point_count':len(camera),'model_load_count':result['model_load_count']}
    for variant in models:request(selected[0],variant,root/'warmup'/variant)
    results=[]
    for i,pair in enumerate(selected):
        points,measurements={},{}
        for variant in (['baseline','accelerated'] if i%2==0 else ['accelerated','baseline']):
            points[variant],measurements[variant]=request(pair,variant,root/'trials'/pair['id']/variant)
        comparison=cloud_difference(points['baseline'],points['accelerated'])
        row={'id':pair['id'],'measurements':measurements,'point_difference':comparison}
        atomic_json(root/'trials'/pair['id']/'result.json',row);results.append(row)
        print('INTERFACE_CASE',row,flush=True)
    report={'at':now(),'passed':True,'physical_trials':0,'planning_calls':0,'model_load_count':1,
        'cases':results,'median_request_seconds':{variant:statistics.median(row['measurements'][variant]['request_seconds'] for row in results)
            for variant in models},'world_transform_and_native_output_contract_passed':True}
    atomic_json(root/'acceptance.json',report)
    from modules.rayst3r.fast_inference.evaluation import report as write_report
    write_report(root.parent,read_json(root.parent/'summary.json'),read_json(root.parent/'acceptance.json'))
    print('INTERFACE_CHECK_PASSED',report['median_request_seconds'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--comparison',type=Path,required=True)
    args=parser.parse_args();comparison=args.comparison
    while True:
        if (comparison/'STOP').exists():raise RuntimeError('Acceleration comparison stopped')
        acceptance=comparison/'acceptance.json'
        if acceptance.exists() and read_json(acceptance).get('independent_surface_readback'):
            if not read_json(acceptance)['eligible_variants']:
                atomic_json(comparison/'interface_skipped.json',{'at':now(),'reason':'No accelerated variant passed speed and geometry gates'})
                return
            break
        time.sleep(10)
    from modules.rayst3r.optimized.benchmark import gpu_inventory
    gpu=read_json(comparison/'preflight/device.json')['uuid']
    while gpu_inventory()[gpu]<23000:time.sleep(10)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    try:run(comparison/'interface',gpu)
    except BaseException:
        import traceback
        atomic_json(comparison/'interface_STOP.json',{'at':now(),'error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
