"""One fixed method, one resident load, 50 immutable formal slots."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
from PIL import Image
import torch
import trimesh

from prepare_scenes import RUN,sha,write
from model_adapters import load_adapter, SUPPORTED_METHODS
from geometry_evaluation import unproject,subsample,calibrate_from_observation,surface_metrics,oracle_shape_metrics


def guard(registration):
    for name,digest in registration['frozen_files'].items():
        if sha(Path(name))!=digest:raise RuntimeError('Frozen file changed: '+name)
    for name,expected in registration['weights'].items():
        stat=Path(name).stat()
        if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(expected['size'],expected['mtime_ns'],expected['inode']):
            raise RuntimeError('Registered model payload changed: '+name)


def load_input(case_id,registration):
    source=RUN/'observations'/case_id;seg=RUN/registration['segmentation_directory']/case_id
    rgb=np.asarray(Image.open(source/'rgb.png').convert('RGB'));depth=np.load(source/'depth_m.npy')
    mask=np.asarray(Image.open(seg/'mask.png'))>0
    camera=json.loads((source/'camera.json').read_text());k=np.array(camera['intrinsics'])
    return rgb,depth,mask,k,camera


def reference_data(case_id,camera):
    truth=RUN/'ground_truth'/case_id
    mesh=trimesh.load(truth/'surface_world.ply',force='mesh',process=False)
    rng_state=np.random.get_state();np.random.seed(20261122)
    points,_=trimesh.sample.sample_surface(mesh,50000);np.random.set_state(rng_state)
    world_from_camera=np.array(camera['camera_world_optical'])
    reference=(points-world_from_camera[:3,3])@world_from_camera[:3,:3]
    return reference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True,choices=SUPPORTED_METHODS);args=parser.parse_args()
    registration_path=RUN/'registration.json';registration=json.loads(registration_path.read_text());guard(registration)
    destination=RUN/'methods'/args.method;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'started.json').exists():raise RuntimeError('Method already started; formal slots are not automatically retried')
    write(destination/'started.json',{'method':args.method,'registration_sha256':sha(registration_path),'pid':os.getpid(),'started_at':time.time()})
    torch.set_num_threads(6);random.seed(12345);np.random.seed(12345);torch.manual_seed(12345)
    began=time.perf_counter();adapter=load_adapter(args.method);torch.cuda.synchronize();load_s=time.perf_counter()-began
    warmup=load_input('warmup',registration);began=time.perf_counter()
    warmup_result=adapter.predict(*warmup[:4],destination/'warmup_native_input');torch.cuda.synchronize()
    warmup_s=time.perf_counter()-began
    warmup_empty=('mesh' in warmup_result and warmup_result['mesh'] is None)
    # A native no-surface result is a data outcome. Adapter correctness has a
    # separate successful upstream-demo preflight; no iso threshold is changed.
    write(destination/'ready.json',{'load_s':load_s,'warmup_s':warmup_s,'warmup_empty':warmup_empty,
                                  'model_load_count':1,'configuration':adapter.configuration,
                                  'device':torch.cuda.get_device_name(),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES')})
    del warmup_result
    results=[]
    for case in registration['cases']:
        cid=case['id'];output=destination/cid;output.mkdir(exist_ok=False)
        write(output/'attempt_started.json',{'id':cid,'method':args.method,'time':time.time()})
        began=time.perf_counter();result={'id':cid,'method':args.method,'category':case['category'],'style_id':case['style_id'],'success':False}
        input_start=time.perf_counter();rgb,depth,mask,k,camera=load_input(cid,registration)
        for name,digest in case['input_hashes'].items():
            if sha(name)!=digest:raise RuntimeError('Frozen observation changed: '+name)
        result['input_read_s']=time.perf_counter()-input_start
        try:
            if not mask.any():
                result.update(status='not_executed',reason='shared_segmentation_empty')
            else:
                random.seed(12345);np.random.seed(12345);torch.manual_seed(12345)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                prediction=adapter.predict(rgb,depth,mask,k,output/'native_input');torch.cuda.synchronize()
                result['generation_s']=time.perf_counter()-t
                result['peak_gpu_gb']=torch.cuda.max_memory_allocated()/1e9
                result['peak_reserved_gpu_gb']=torch.cuda.max_memory_reserved()/1e9
                if 'mesh' in prediction:
                    mesh=prediction['mesh']
                    if mesh is None or len(mesh.vertices)==0 or len(mesh.faces)==0:
                        raise ValueError('empty_native_mesh')
                    t=time.perf_counter();mesh.export(output/'native_mesh.ply')
                    points,_=trimesh.sample.sample_surface(mesh,registration['point_budget'])
                    result['mesh_to_points_and_save_s']=time.perf_counter()-t
                    result['native_vertex_count']=len(mesh.vertices);result['native_face_count']=len(mesh.faces)
                else:
                    points=np.asarray(prediction['points']);np.save(output/'native_points.npy',points)
                if 'conditioning_points' in prediction:np.save(output/'conditioning_points.npy',prediction['conditioning_points'])
                result['native_point_count']=len(points)
                finite=np.isfinite(points).all(1);result['nonfinite_point_count']=int((~finite).sum())
                points=subsample(points[finite],registration['point_budget'])
                if not len(points):raise ValueError('empty_native_point_cloud')
                np.save(output/'native_sampled_points.npy',points)
                result['evaluated_point_count']=len(points)
                observed=unproject(depth,mask,k)
                if prediction['metric']:
                    metric_points=points;result['metric_placement']='native_RGBD_scale_and_camera_coordinates'
                    result['calibration_s']=0.
                else:
                    metric_points,calibration=calibrate_from_observation(points,observed)
                    result['metric_placement']='RGB_only_shape_registered_using_observed_depth_without_GT'
                    result['calibration']=calibration;result['calibration_s']=calibration['seconds']
                np.save(output/'metric_points.npy',metric_points)
                result['reconstruction_and_placement_s']=result['generation_s']+result.get('mesh_to_points_and_save_s',0)+result['calibration_s']
                t=time.perf_counter();reference=reference_data(cid,camera)
                gt_mask=np.asarray(Image.open(RUN/'ground_truth'/cid/'mask.png'))>0
                visible_reference=unproject(depth,gt_mask,k)
                result['geometry']=surface_metrics(metric_points,reference,visible_reference)
                result['oracle_shape']=oracle_shape_metrics(points,reference)
                result['evaluation_s']=time.perf_counter()-t
                result.update(success=True,status='completed')
                del prediction
        except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
            # Runtime/CUDA errors are infrastructure/interface events. Preserve
            # the current attempt and stop this method rather than fabricate
            # the remaining slots or silently cold-start a replacement.
            result.update(status='stopped',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            result['total_case_wall_s']=time.perf_counter()-began;write(output/'result.json',result)
            results.append(result);write(destination/'STOP.json',result);write(destination/'progress.json',{'results':results,'completed':len(results),'stopped':True})
            raise
        except ValueError as exc:
            result.update(status='model_output_failure',reason=str(exc))
        result['total_case_wall_s']=time.perf_counter()-began;write(output/'result.json',result);results.append(result)
        write(destination/'progress.json',{'completed':len(results),'successes':sum(x['success'] for x in results),'results':results})
        print(args.method,cid,result['status'],round(result.get('generation_s',0),3),flush=True)
    write(destination/'complete.json',{'method':args.method,'results':results,'registered':50,
                                      'successes':sum(x['success'] for x in results),'load_s':load_s,'warmup_s':warmup_s,
                                      'model_load_count':1,'physical_trials':0})


if __name__=='__main__':main()
