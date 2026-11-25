"""Register and run same-device speed/accuracy comparisons on 50 frozen inputs."""
import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time
import traceback

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT, ensure_run_path, now, sha256

CONFIG = ROOT/'configs/experiments/rayst3r_inference_acceleration.json'


def prepare(root):
    if root.exists():raise RuntimeError('Use a fresh acceleration registration')
    config = read_json(CONFIG)
    previous = read_json(ROOT/config['reference_validation']/'registration.json')
    reference = Path(previous['reference'])
    inputs = dict(previous['input_hashes'])
    for case in previous['cases']:
        truth=reference/'ground_truth'/case['id']/'surface_world.ply'
        inputs[str(truth)] = sha256(truth)
    for filename,digest in inputs.items():
        if sha256(filename)!=digest:raise RuntimeError('Frozen input changed: '+filename)
    sources={str(CONFIG):sha256(CONFIG)}
    for folder in ['modules/rayst3r','modules/dinov2']:
        for path in (ROOT/folder).rglob('*.py'):sources[str(path)]=sha256(path)
    root.mkdir(parents=True)
    for name in ['source_registered','preflight','trials']:(root/name).mkdir()
    for filename in sources:
        target=root/'source_registered'/Path(filename).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(filename,target)
    atomic_json(root/'registration.json',{'at':now(),'settings':config,
        'cases':previous['cases'],'reference':str(reference),'segmentation':previous['segmentation'],
        'input_hashes':inputs,'source_hashes':sources,'weights':previous['weights'],
        'physical_trials':0,'planning_calls':0,'automatic_activation':False,
        'measurement_budget':len(previous['cases'])*len(config['variants'])*config['repetitions'],
        'accuracy_uses_truth_only_after_inference':True})
    atomic_json(root/'progress.json',{'at':now(),'phase':'registered','completed_cases':0,'registered_cases':50})
    print('REGISTERED_RAY_ACCELERATION',root,flush=True)


def guard(root, *, payload=False):
    if (root/'STOP').exists():raise RuntimeError('Stopped acceleration run cannot be restarted')
    reg=read_json(root/'registration.json')
    for filename,digest in {**reg['source_hashes'],**reg['input_hashes']}.items():
        if sha256(filename)!=digest:raise RuntimeError('Registered input/source changed: '+filename)
    for filename,record in reg['weights'].items():
        if Path(filename).stat().st_size!=record['size'] or payload and sha256(filename)!=record['sha256']:
            raise RuntimeError('Frozen weights changed: '+filename)
    return reg


def protected_devices(root):
    reg=read_json(root/'registration.json')
    protected=ROOT/reg['settings']['protected_campaign']
    stopped=protected/'services_stopped.json'
    if stopped.exists() and read_json(stopped)['all_owned_exited']:
        return set()
    resources=read_json(protected/'preflight/resources.json').get('per_method',{})
    reserved=set()
    for method,record in resources.items():
        stopped=protected/'methods'/method/'services_stopped.json'
        if not stopped.exists() or not read_json(stopped)['all_owned_exited']:
            reserved.update(record['gpu_roles'].values())
    return reserved


def launch(root):
    from modules.rayst3r.optimized.benchmark import gpu_inventory
    reg=guard(root)
    if (root/'started.json').exists():raise RuntimeError('Acceleration worker already started')
    while True:
        free=gpu_inventory();blocked=protected_devices(root)
        choices=sorted((memory,gpu) for gpu,memory in free.items()
            if memory>=reg['settings']['minimum_gpu_free_mib'] and gpu not in blocked)
        for memory,gpu in reversed(choices):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            python=str(ROOT/'environments/reconstruction/bin/python')
            probe=subprocess.run([python,'-c','import torch; x=torch.zeros(1,device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name())'],
                env=env,cwd=ROOT,capture_output=True,text=True,timeout=40)
            atomic_json(root/'preflight/device_probe.json',{'at':now(),'gpu':gpu,'free_mib':memory,
                'passed':probe.returncode==0,'stdout':probe.stdout,'stderr':probe.stderr[-2000:]})
            if probe.returncode:continue
            guard(root,payload=True)
            if gpu in protected_devices(root) or gpu_inventory()[gpu]<reg['settings']['minimum_gpu_free_mib']:continue
            atomic_json(root/'preflight/device.json',{'at':now(),'uuid':gpu,'free_mib':gpu_inventory()[gpu]})
            with (root/'worker.log').open('x') as log:
                result=subprocess.run([python,'-u','-m',__package__+'.benchmark','run','--root',str(root)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Offline acceleration comparison failed; see worker.log')
            return
        atomic_json(root/'progress.json',{'at':now(),'phase':'waiting_for_gpu','completed_cases':0,
            'registered_cases':50,'free_mib':free,'protected_devices':sorted(blocked)})
        time.sleep(5)


def cloud_difference(a,b):
    import numpy as np
    from scipy.spatial import cKDTree
    if not len(a) or not len(b):
        return {'same_count':len(a)==len(b),'maximum_coordinate_difference_m':0. if not len(a) and not len(b) else None,
                'symmetric_nearest_mean_mm':0. if not len(a) and not len(b) else None,'symmetric_nearest_p95_mm':None}
    forward=cKDTree(a).query(b,workers=1)[0];reverse=cKDTree(b).query(a,workers=1)[0]
    return {'same_count':a.shape==b.shape,
        'maximum_coordinate_difference_m':float(np.max(np.abs(a-b))) if a.shape==b.shape else None,
        'symmetric_nearest_mean_mm':float((forward.mean()+reverse.mean())*.5*1000),
        'symmetric_nearest_p95_mm':float(max(np.percentile(forward,95),np.percentile(reverse,95))*1000)}


def quality(points,truth):
    from modules.pipeline.experiments.pointcloud_benchmark.geometry_evaluation import distances,subsample
    if len(points)<20:return {'valid':False,'chamfer_l1_mm':None,'fscore5':0.}
    forward,reverse=distances(subsample(points,20000),truth)
    p=float((forward<=.005).mean());r=float((reverse<=.005).mean())
    return {'valid':True,'chamfer_l1_mm':float((forward.mean()+reverse.mean())*.5*1000),
            'fscore5':2*p*r/(p+r) if p+r else 0.}


def summarize(root):
    import numpy as np
    reg=read_json(root/'registration.json');settings=reg['settings']
    rows=[read_json(path) for path in sorted((root/'trials').glob('*/result.json'))]
    variants={}
    for variant in settings['variants']:
        times=[t for row in rows for t in row['measurements'][variant]['seconds']]
        if not times:continue
        paired=[statistics.median(row['measurements']['baseline']['seconds'])/statistics.median(row['measurements'][variant]['seconds']) for row in rows]
        fdelta=[row['quality'][variant]['fscore5']-row['quality']['baseline']['fscore5'] for row in rows]
        cd=[row['quality'][variant]['chamfer_l1_mm'] for row in rows if row['quality'][variant]['valid']]
        bcd=[row['quality']['baseline']['chamfer_l1_mm'] for row in rows if row['quality']['baseline']['valid']]
        cdelta=[row['quality'][variant]['chamfer_l1_mm']-row['quality']['baseline']['chamfer_l1_mm'] for row in rows
            if row['quality'][variant]['valid'] and row['quality']['baseline']['valid']]
        strict=sum(row['differences'][variant]['same_count']
            and row['differences'][variant]['maximum_coordinate_difference_m'] is not None
            and row['differences'][variant]['maximum_coordinate_difference_m']<=settings['strict_coordinate_atol_m'] for row in rows)
        valid=all(not row['quality']['baseline']['valid'] or row['quality'][variant]['valid'] for row in rows)
        stable=(valid and sum(fdelta)/len(fdelta)>=-settings['maximum_mean_fscore5_drop']
            and min(fdelta)>=-settings['maximum_case_fscore5_drop']
            and cd and bcd and statistics.median(cd)-statistics.median(bcd)<=settings['maximum_median_chamfer_increase_mm']
            and cdelta and max(cdelta)<=settings['maximum_case_chamfer_increase_mm'])
        variants[variant]={'measurements':len(times),'median_seconds':statistics.median(times),
            'p90_seconds':float(np.percentile(times,90)),'paired_speedup_median':statistics.median(paired),
            'peak_allocated_mib':max(max(row['measurements'][variant]['peak_allocated_mib']) for row in rows),
            'fscore5_mean':statistics.mean(row['quality'][variant]['fscore5'] for row in rows),
            'chamfer_l1_mm_median':statistics.median(cd) if cd else None,
            'strict_equivalent_cases':strict,'geometry_quality_stable':bool(stable),
            'maximum_case_fscore5_drop':-min(fdelta),'maximum_case_chamfer_increase_mm':max(cdelta) if cdelta else None}
    result={'at':now(),'completed_cases':len(rows),'registered_cases':50,'variants':variants,
        'physical_trials':0,'planning_calls':0,'automatic_activation':False}
    atomic_json(root/'summary.json',result)
    atomic_json(root/'progress.json',{'at':now(),'phase':'running','completed_cases':len(rows),'registered_cases':50})
    return result


def run(root):
    import numpy as np
    import trimesh
    import torch
    from modules.rayst3r.arx_reconstruction import RaySt3RAdapter
    from modules.rayst3r.optimized.worker import load_input
    from .adapter import FastAdapter
    from .profile import stages
    from .surface import uniform_surface
    reg=guard(root,payload=True);settings=reg['settings']
    if (root/'started.json').exists():raise RuntimeError('Worker cannot be retried')
    if torch.cuda.mem_get_info()[0]/2**20<settings['minimum_gpu_free_mib']:raise RuntimeError('GPU memory changed before loading')
    atomic_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
        'torch_version':torch.__version__,'physical_trials':0})
    torch.set_num_threads(4)
    native=RaySt3RAdapter()
    adapters={name:FastAdapter(baseline=native,variant=name) for name in settings['variants']}
    def predict(name,values,directory, *, profile=False):
        torch.manual_seed(settings['seed']);np.random.seed(settings['seed']);random.seed(settings['seed'])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begun=time.perf_counter()
        if profile:
            with stages(adapters[name]) as timing:output=adapters[name].predict(*values,directory)
        else:output=adapters[name].predict(*values,directory);timing=None
        torch.cuda.synchronize();elapsed=time.perf_counter()-begun
        points=np.asarray(output['points'],dtype=np.float32).copy()
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise RuntimeError('Invalid point output')
        result={'seconds':elapsed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'remaining_mib':torch.cuda.mem_get_info()[0]/2**20,'trace':adapters[name].last_trace,'stages':timing}
        return points,result
    values=load_input(reg,reg['cases'][0]['id'])
    for name in settings['variants']:
        _,record=predict(name,values,root/'preflight'/name,profile=True)
        atomic_json(root/'preflight'/(name+'.json'),record)
        print('ACCELERATION_WARMED',name,record,flush=True)
    if torch.cuda.mem_get_info()[0]/2**20<settings['minimum_after_warmup_free_mib']:
        raise RuntimeError('Insufficient GPU headroom after warmup')
    for index,case in enumerate(reg['cases']):
        if torch.cuda.mem_get_info()[0]/2**20<settings['minimum_after_warmup_free_mib']:raise RuntimeError('GPU headroom exhausted')
        folder=root/'trials'/case['id'];folder.mkdir()
        atomic_json(folder/'claim.json',{'at':now(),'attempt':1,'id':case['id']})
        values=load_input(reg,case['id']);measurements={name:{'seconds':[],'peak_allocated_mib':[]} for name in settings['variants']}
        clouds={};traces={}
        for repeat in range(settings['repetitions']):
            order=settings['variants'][index%4:]+settings['variants'][:index%4]
            if repeat%2:order=list(reversed(order))
            for name in order:
                points,record=predict(name,values,folder/name/str(repeat))
                measurements[name]['seconds'].append(record['seconds'])
                measurements[name]['peak_allocated_mib'].append(record['peak_allocated_mib'])
                if repeat==0:
                    clouds[name]=points;traces[name]=record['trace']
                    np.save(folder/name/'points.npy',points,allow_pickle=False)
        truth_path=Path(reg['reference'])/'ground_truth'/case['id']/'surface_world.ply'
        truth_mesh=trimesh.load(truth_path,force='mesh',process=False)
        truth=uniform_surface(truth_mesh.vertices,truth_mesh.faces)
        transform=read_json(Path(reg['reference'])/'observations'/case['id']/'camera.json')['camera_world_optical']
        transform=np.asarray(transform)
        geometry={name:quality(points@transform[:3,:3].T+transform[:3,3],truth) for name,points in clouds.items()}
        differences={name:cloud_difference(clouds['baseline'],points) for name,points in clouds.items()}
        atomic_json(folder/'result.json',{'id':case['id'],'category':case['category'],'measurements':measurements,
            'quality':geometry,'differences':differences,'traces':traces,
            'outputs':{name:sha256(folder/name/'points.npy') for name in settings['variants']}})
        summary=summarize(root)
        print('ACCELERATION_CASE',index+1,case['id'],{n:{'seconds':statistics.median(m['seconds']),
            'F5':geometry[n]['fscore5'],'CDmm':geometry[n]['chamfer_l1_mm']} for n,m in measurements.items()},flush=True)
    guard(root,payload=True)
    issues=[]
    for case in reg['cases']:
        folder=root/'trials'/case['id'];row=read_json(folder/'result.json')
        for name,digest in row['outputs'].items():
            if sha256(folder/name/'points.npy')!=digest:issues.append(case['id']+'/'+name)
            if len(row['measurements'][name]['seconds'])!=settings['repetitions']:issues.append('Missing timing '+case['id'])
    summary=summarize(root)
    atomic_json(root/'acceptance.json',{'at':now(),'record_integrity':not issues,'passed':not issues,
        'issues':issues,'completed_cases':50,'variants':summary['variants'],'automatic_activation':False,
        'eligible_variants':[name for name,v in summary['variants'].items() if name!='baseline'
            and v['geometry_quality_stable'] and v['paired_speedup_median']>=settings['minimum_paired_speedup']]})
    atomic_json(root/'progress.json',{'at':now(),'phase':'complete','completed_cases':50,'registered_cases':50})
    print('ACCELERATION_COMPLETE',summary,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','launch','run'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root);return
    with (root/('launcher.lock' if args.stage=='launch' else 'worker.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:(launch if args.stage=='launch' else run)(root)
        except BaseException:
            atomic_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retries':False})
            raise


if __name__=='__main__':main()
