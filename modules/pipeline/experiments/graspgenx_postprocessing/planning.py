"""Register and replay matched NMS-order comparisons without new perception."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
from arx_project.client import Client
from modules.graspgenx.postprocessing.ordering import make_queue
from modules.graspgenx.postprocessing.nms import native_backend
from .offline import raw

CONFIG=ROOT/'configs/experiments/graspgenx_postprocessing_planning.json'
VARIANTS=('control','nms15')


def verify(reg):
    for name,digest in {**reg['source_sha256'],**reg['input_sha256']}.items():
        if sha256(name)!=digest:raise RuntimeError('Registered planning source/input changed: '+name)


def prepare(root,phase):
    if root.exists():raise ValueError('Keep previous V7 planning registrations')
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    current=load_current();cfg=read_json(CONFIG);source=ROOT/cfg['offline_source']
    if not raw(source/'independent_review.json')['passed']:raise ValueError('Offline source records were not accepted')
    original=raw(source/'registration.json');cases=original['cases']
    if phase=='pilot':cases=[c for c in cases if c['id'] in cfg['pilot']['cases']]
    if len(cases)!=(5 if phase=='pilot' else 69):raise ValueError('V7 planning cohort differs')
    if phase=='full':
        pilot=ROOT/cfg['pilot']['run'];review=raw(pilot/'planning_review.json')
        if not review['passed'] or not review['control_successes_retained']:
            raise ValueError('Pilot identity/FK/retention gate did not pass')
    warm=ROOT/'tmp/sam3_zerograsp_comparison/v6_complete74_20261126_01/methods/zerograsp/preflight/head_warmup'
    inputs={str(source/'registration.json'):sha256(source/'registration.json'),
        str(source/'independent_review.json'):sha256(source/'independent_review.json')}
    for c in cases:
        directory=Path(c['source_directory'])
        for name in ['state.json','candidates/candidate_pool.json','plan_result.json','result.json']:
            inputs[str(directory/name)]=sha256(directory/name)
        saved=source/'cases'/c['id']/'nms_p015_r15.json';inputs[str(saved)]=sha256(saved)
    sources=[CONFIG,ROOT/'configs/modules.json',ROOT/'configs/experiments/active_experiment.json',
        ROOT/'configs/policies/fingertip_table.json',native_backend()[1],
        ROOT/'configs/experiments/dual_arm_interleaving.json',ROOT/'modules/curobo/arx_adapter.py',
        ROOT/'modules/curobo/scene_cache.py',ROOT/'modules/maniskill/resident_planner.py',
        ROOT/'modules/pipeline/monitoring/gpu_auto_start.py',
        ROOT/'modules/maniskill/fast_resident_planner.py',ROOT/'modules/maniskill/arx_pipeline/planner.py',
        ROOT/'modules/maniskill/checked_native_services.py']
    for folder in ['modules/curobo/dual_arm','modules/curobo/fingertip_contact','modules/curobo/candidate_io',
                   'modules/curobo/indexed_planning','modules/graspgenx/postprocessing',
                   'modules/pipeline/experiments/graspgenx_postprocessing','arx_project']:
        sources.extend((ROOT/folder).glob('*.py'))
    sources+=[Path(p) for p in current['configuration_hashes']]
    sources+=[ROOT/'modules/curobo/source/curobo/_src/solver'/p for p in ['solver_ik.py','solver_ik_cfg.py']]
    sources=sorted(set(p.resolve() for p in sources))
    root.mkdir(parents=True)
    (root/'claims').mkdir()
    from .planning_launch import monitor_policy
    monitor_config=root/'monitor_config.json';write_json(monitor_config,monitor_policy(root))
    inputs[str(monitor_config)]=sha256(monitor_config)
    for name in ['state.json','closed_state.json']:
        target=root/'warmup'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(warm/name,target);inputs[str(warm/name)]=sha256(warm/name)
    for path in sources:
        target=root/'source_registered'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    reg={'at':now(),'version':'V7','experiment':'NMS15_planning_comparison','phase':phase,
        'authorization':cfg['authorization'],'cases':cases,'gpu':'pending_resource_selection',
        'device_assignment_pending':True,'planning_requests':len(cases)*2,'variants':list(VARIANTS),
        'same_backend':'native_batch16','same_scheduler':cfg['scheduler'],'seed':cfg['random_seed'],
        'offline_source':str(source),'geometry_hard_filter_enabled':False,
        'NMS_phase_order':'representatives_then_all_other_original_candidates',
        'minimum_startup_free_mib':10000,'minimum_request_free_mib':1536,
        'physical_grasps':0,'perception_calls':0,'simulation_instances':0,'automatic_retries':False,
        'source_sha256':{str(p):sha256(p) for p in sources},'input_sha256':inputs}
    write_json(root/'registration.json',reg);write_json(root/'registration_pending.json',reg)
    write_json(root/'progress.json',{'at':now(),'phase':'waiting_for_gpu','completed_requests':0,
        'total_requests':reg['planning_requests'],'results':[]})


def summaries(rows):
    result={}
    for mode in VARIANTS:
        values=[r['variants'][mode] for r in rows];out={'requests':len(values),'successes':sum(v['success'] for v in values)}
        for key in ['postprocessing_seconds','planning_request_seconds','postprocessing_plus_planning_seconds','pipeline_wall_seconds','candidate_checks']:
            times=[v[key] for v in values]
            out[key]={'median':float(np.median(times)),'p90':float(np.percentile(times,90)),'total':float(sum(times))}
        out['selected_deferred']=sum(v.get('selected_priority_phase')=='deferred' for v in values)
        result[mode]=out
    return result


def run(root):
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Never repeat a started V7 planning run')
    reg=raw(root/'registration.json');verify(reg)
    if reg['device_assignment_pending']:raise ValueError('No CUDA-verified device assignment')
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu':reg['gpu']})
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=reg['gpu'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    socket_path=root/'rpc.sock'
    if len(str(socket_path).encode())>=108:raise ValueError('Planner socket path is too long')
    log=(root/'resident.log').open('a')
    command=[str(ROOT/'environments/curobo/bin/python'),'-m',
        'modules.pipeline.experiments.graspgenx_postprocessing.planning_worker','--root',str(root/'resident'),
        '--socket',str(socket_path),'--warmup-trial',str(root/'warmup'),'--descriptor',str(ROOT/'configs/modules.json')]
    worker=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    write_json(root/'worker.json',{'at':now(),'pid':worker.pid,'command':command,'gpu':reg['gpu']})
    client=Client(socket_path,timeout=1800,candidate_transport=True);rows=[];completed=0
    try:
        ready=root/'resident/curobo_indexed_planning_ready.json';deadline=time.monotonic()+1200
        write_json(root/'progress.json',{'at':now(),'phase':'warming','completed_requests':0,'total_requests':reg['planning_requests'],'results':[]})
        while not ready.exists():
            if worker.poll() is not None:raise RuntimeError('Planning worker exited during startup; inspect resident.log')
            if time.monotonic()>deadline:raise TimeoutError('V7 planning warmup exceeded its deadline')
            time.sleep(1)
        write_json(root/'startup.json',{'at':now(),'passed':True,**raw(root/'resident/startup.json'),
            'single_resident_worker':True,'variants_share_models_and_GPU':True})
        for ordinal,case in enumerate(reg['cases']):
            source=Path(case['source_directory']);begin=time.perf_counter()
            payload=raw(source/'candidates/candidate_pool.json');input_read=time.perf_counter()-begin
            expected=raw(Path(reg['offline_source'])/'cases'/case['id']/'nms_p015_r15.json')
            pair={};order=VARIANTS if ordinal%2==0 else tuple(reversed(VARIANTS))
            for mode in order:
                directory=root/'plans'/case['id']/mode;directory.mkdir(parents=True)
                claim=root/'claims'/(case['id']+'_'+mode+'.json')
                with claim.open('x') as stream:json.dump({'at':now(),'state':'claimed'},stream)
                shutil.copy2(source/'state.json',directory/'state.json')
                write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':completed,
                    'total_requests':reg['planning_requests'],'current':{'id':case['id'],'variant':mode},'results':rows})
                print('V7_PLAN_BEGIN',case['id'],mode,flush=True)
                began=time.perf_counter();queue,manifest=make_queue(payload,mode)
                post_seconds=time.perf_counter()-began
                if mode=='nms15' and any(manifest[k]!=expected[k] for k in ['representatives','deferred']):
                    raise ValueError('Live native NMS differs from the frozen offline result')
                write_json(directory/'candidate_order.json',manifest)
                pack=time.perf_counter();descriptor=publish_candidate_pool(directory,queue)
                producer_seconds=time.perf_counter()-pack
                response=client.request(directory,candidate_transport=descriptor)
                if response['status']!='ok':raise RuntimeError('Planner RPC fault: '+str(response))
                metrics=response['result'];finish_candidate_transport(directory)
                pipeline_seconds=time.perf_counter()-began
                io=raw(directory/'candidate_io.json')
                if io['source']!='shared_memory' or io['decode_count']!=1 or io['full_arm_pool_copies']!=0:
                    raise ValueError('V7 planning lost the shared-memory single-decode contract')
                metrics.update(postprocessing_seconds=post_seconds,producer_publish_seconds=producer_seconds,
                    candidate_decode_seconds=io['load_seconds'],input_read_seconds=input_read,
                    postprocessing_plus_planning_seconds=post_seconds+metrics['planning_request_seconds'],
                    pipeline_wall_seconds=pipeline_seconds,representatives=manifest['representative_count'],
                    original_candidate_count=manifest['source_count'])
                write_json(directory/'metrics.json',metrics);pair[mode]=metrics;completed+=1
                write_json(claim,{'at':now(),'state':'complete','metrics_sha256':sha256(directory/'metrics.json'),
                    'plan_sha256':sha256(directory/'plan_result.json'),'order_sha256':sha256(directory/'candidate_order.json')})
                print('V7_PLAN_END',case['id'],mode,'success',metrics['success'],'plan_seconds',round(metrics['planning_request_seconds'],3),'checks',metrics['candidate_checks'],flush=True)
            comparison={'control_success_retained':not pair['control']['success'] or pair['nms15']['success'],
                'same_original_selection':all(pair['control'][k]==pair['nms15'][k] for k in ['success','selected_arm','selected_original_source_index'])}
            rows.append({'id':case['id'],'variants':pair,'comparison':comparison})
            write_json(root/'progress.json',{'at':now(),'phase':'comparing','completed_requests':completed,
                'total_requests':reg['planning_requests'],'results':rows,'summaries':summaries(rows)})
        verify(reg)
        write_json(root/'acceptance.json',{'at':now(),'task_complete':completed==reg['planning_requests'],
            'record_integrity':True,'planning_requests':completed,'summaries':summaries(rows),'cases':rows,
            'control_successes_retained':all(r['comparison']['control_success_retained'] for r in rows),
            'physical_grasps':0,'automatic_default_activation':False})
        write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_requests':completed,
            'total_requests':reg['planning_requests'],'results':rows,'summaries':summaries(rows)})
    finally:
        close_owned_transports()
        if worker.poll() is None:
            try:client.call({'command':'shutdown'})
            except Exception:worker.terminate()
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=10)
        log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=['pilot','full'])
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,args.phase)
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
