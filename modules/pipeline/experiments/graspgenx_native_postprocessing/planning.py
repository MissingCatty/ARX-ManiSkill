"""A fixed five-scene, four-group planning comparison; no robot motion."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
from arx_project.layout import ROOT,now,read_json,write_json,sha256,ensure_run_path
from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
from arx_project.client import Client
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel
from .offline import raw,guard,CONFIG


def prepare(root,offline,gpu):
    if root.exists():raise ValueError('Keep old registrations; use a new directory')
    old=guard(offline,completed=True);cfg=raw(CONFIG)
    if not raw(offline/'independent_review.json')['passed']:raise ValueError('Offline output review failed')
    cases=[c for c in old['cases'] if c['id'] in cfg['pilot_cases']]
    if {c['id'] for c in cases}!=set(cfg['pilot_cases']):raise ValueError('Pilot cohort changed')
    inputs={};sources={CONFIG,ROOT/'configs/modules.json',ROOT/'configs/experiments/active_experiment.json',ROOT/'configs/policies/fingertip_table.json'}
    sources.update(Path(p) for p in old['source_sha256'])
    for folder in ['modules/curobo/dual_arm','modules/curobo/fingertip_contact','modules/curobo/candidate_io',
        'modules/maniskill/planning_common','modules/maniskill/arx_pipeline','arx_project',
        'modules/graspgenx/native_postprocessing',
        'modules/pipeline/experiments/graspgenx_native_postprocessing']:
        sources.update((ROOT/folder).glob('*.py'))
    sources.update(ROOT/'modules/curobo'/name for name in ['arx_adapter.py','scene_cache.py'])
    sources.update(ROOT/'modules/maniskill'/name for name in ['resident_planner.py','fast_resident_planner.py','checked_native_services.py'])
    for name in ['registration.json','acceptance.json','independent_review.json','official_parity.json']:
        inputs[str(offline/name)]=sha256(offline/name)
    stop_acknowledgement=None
    if (offline/'STOP').exists():
        stopped=raw(offline/'STOP')
        if 'Refinement provenance/geometry corruption' not in stopped['error']:
            raise RuntimeError('Unresolved predecessor fault prevents planning')
        inputs[str(offline/'STOP')]=sha256(offline/'STOP')
        stop_acknowledgement={'preserved_STOP_sha256':sha256(offline/'STOP'),
            'processing_completed_before_review_failure':True,
            'review_projection_corrected_for_nonunit_saved_FP32_axes':True,
            'all_original_outputs_independently_reviewed_without_reprocessing':True,
            'review_sha256':sha256(offline/'independent_review.json')}
    for case in cases:
        source=Path(case['source_directory'])
        for name in ['state.json','candidates/candidate_pool.json']:
            inputs[str(source/name)]=sha256(source/name)
        for variant in cfg['variants']:
            for name in ['candidate_pool.json','metrics.json']:
                path=offline/'cases'/case['id']/variant/name;inputs[str(path)]=sha256(path)
    warm=ROOT/'tmp/sam3_zerograsp_comparison/v6_complete74_20261126_01/methods/zerograsp/preflight/head_warmup'
    root.mkdir(parents=True);(root/'claims').mkdir();(root/'warmup').mkdir()
    for name in ['state.json','closed_state.json']:
        shutil.copy2(warm/name,root/'warmup'/name);inputs[str(root/'warmup'/name)]=sha256(root/'warmup'/name)
    for path in sources:
        target=root/'source_registered'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    modes=['control',*cfg['variants']]
    schedule=[]
    for ordinal,case in enumerate(cases):
        order=modes[ordinal%len(modes):]+modes[:ordinal%len(modes)]
        schedule.extend({'id':case['id']+'_'+mode,'case_id':case['id'],'variant':mode} for mode in order)
    write_json(root/'registration.json',{'at':now(),'experiment':'V7_full_native_postprocessing_planning',
        'authorization':cfg['authorization'],'offline_source':str(offline.relative_to(ROOT)),
        'cases':cases,'variants':modes,'schedule':schedule,'gpu':gpu,
        'planning_requests':len(schedule),'physical_grasps':0,'automatic_retries':False,
        'predecessor_review_stop_acknowledgement':stop_acknowledgement,
        'unchanged_shared_planner':'original_V3_scalar_16_64_interleaved',
        'source_sha256':{str(p):sha256(p) for p in sorted(sources)},'input_sha256':inputs})
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,'total_requests':len(schedule),'results':[]})


def summarize(rows,modes):
    result={}
    for mode in modes:
        values=[r for r in rows if r['variant']==mode]
        if not values:continue
        total=[r['postprocessing_seconds']+r['planning_seconds'] for r in values]
        result[mode]={'completed':len(values),'planning_successes':sum(r['success'] for r in values),
            'postprocessing_empty':sum(r['reason']=='postprocessing_empty' for r in values),
            'candidate_checks':sum(r['candidate_checks'] for r in values),
            'combined_total_seconds':float(sum(total)),'combined_median_seconds':float(np.median(total)),
            'combined_p90_seconds':float(np.percentile(total,90))}
    return result


def run(root):
    reg=guard(root)
    if (root/'STOP').exists() or (root/'started.json').exists():raise ValueError('No automatic retry')
    write_json(root/'started.json',{'at':now(),'gpu':reg['gpu'],'pid':os.getpid()})
    from modules.pipeline.experiments.indexed_grasp_planning.physical.controller import environment,wait_ready
    socket=ROOT/'tmp/rpc'/('native'+sha256(root/'registration.json')[:10]+'.sock')
    socket.parent.mkdir(exist_ok=True,parents=True)
    command=[str(ROOT/'environments/curobo/bin/python'),'-u','-m',
        'modules.pipeline.experiments.graspgenx_native_postprocessing.worker',
        '--root',str(root/'resident'),'--socket',str(socket),'--warmup-trial',str(root/'warmup'),
        '--descriptor',str(ROOT/'configs/modules.json')]
    offline=ROOT/reg['offline_source'];rows=list(reg.get('inherited_results',[]));client=Client(socket,timeout=1800,candidate_transport=True)
    inherited={r['id'] for r in rows};interrupted=set(reg.get('interrupted_request_ids',[]))
    cases={c['id']:c for c in reg['cases']}
    with (root/'planner.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment(reg['gpu']),stdout=log,stderr=subprocess.STDOUT)
        write_json(root/'planner_process.json',{'pid':child.pid,'command':command})
        try:
            wait_ready(child,root/'resident/curobo_dual_arm_ready.json')
            write_json(root/'preflight.json',{'at':now(),'passed':True,'both_arms_models_warmed':True,'physical_grasps':0})
            for item in reg['schedule']:
                if item['id'] in inherited|interrupted:continue
                guard(root);case=cases[item['case_id']];mode=item['variant'];source=Path(case['source_directory'])
                claim=root/'claims'/(item['id']+'.json');out=root/'plans'/case['id']/mode
                if claim.exists() or out.exists():raise RuntimeError('Request already claimed')
                out.mkdir(parents=True);write_json(claim,{'at':now(),'state':'claimed',**item})
                write_json(root/'progress.json',{'at':now(),'phase':'planning','current':item,
                    'completed_requests':len(rows),'total_requests':len(reg['schedule']),'results':rows})
                shutil.copy2(source/'state.json',out/'state.json')
                path=source/'candidates/candidate_pool.json' if mode=='control' else offline/'cases'/case['id']/mode/'candidate_pool.json'
                payload=raw(path);post=0. if mode=='control' else raw(path.with_name('metrics.json'))['seconds']
                transport_begin=time.perf_counter()
                descriptor=publish_candidate_pool(out,payload)
                # Start the planner against shared memory while persistence runs.
                transport_seconds=time.perf_counter()-transport_begin
                if payload['candidates']:
                    rpc_begin=time.perf_counter()
                    response=client.request(out,candidate_transport=descriptor)
                    rpc_seconds=time.perf_counter()-rpc_begin
                    if response['status']!='ok':raise RuntimeError('Planning infrastructure error: '+str(response))
                    metrics=response['result'];metrics.update(planning_attempted=True)
                else:
                    rpc_seconds=0.
                    metrics={'success':False,'reason':'postprocessing_empty','planning_seconds':0.,
                        'candidate_checks':0,'rejected':{},'planning_attempted':False}
                    write_json(out/'plan_result.json',metrics);write_json(out/'planning_metrics.json',metrics)
                finish_candidate_transport(out)
                metrics.update(planner_internal_seconds=metrics['planning_seconds'],
                    planning_seconds=rpc_seconds+transport_seconds,
                    shared_memory_publish_seconds=transport_seconds,complete_RPC_seconds=rpc_seconds)
                row={**item,**metrics,'postprocessing_seconds':post,'input_candidates':len(payload['candidates'])}
                write_json(out/'metrics.json',row)
                write_json(claim,{'at':now(),'state':'complete',**item,
                    'plan_sha256':sha256(out/'plan_result.json'),'metrics_sha256':sha256(out/'metrics.json'),
                    'candidate_pool_sha256':sha256(out/'candidates/candidate_pool.json')})
                rows.append(row)
                write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':len(rows),
                    'total_requests':len(reg['schedule']),'results':rows,'summaries':summarize(rows,reg['variants'])})
                print('NATIVE_PLAN',len(rows),case['id'],mode,row['success'],round(row['planning_seconds'],3),flush=True)
            write_json(root/'acceptance.json',{'at':now(),'task_complete':True,'planning_requests':len(rows),
                'interrupted_requests':len(interrupted),'new_requests':len(rows)-len(inherited),
                'summaries':summarize(rows,reg['variants']),'physical_grasps':0})
        finally:
            close_owned_transports()
            if child.poll() is None:
                try:client.call({'command':'shutdown'})
                except Exception:child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':child.poll() is not None,'returncode':child.returncode})
    review(root)


def review(root):
    reg=guard(root);model=ArmModel();endpoints=[];requests=[]
    for item in reg['schedule']:
        out=root/'plans'/item['case_id']/item['variant'];claim=raw(root/'claims'/(item['id']+'.json'))
        if item['id'] in reg.get('interrupted_request_ids',[]):
            if claim['state']!='interrupted':raise RuntimeError('Historical interruption was overwritten')
            continue
        if claim['state']!='complete':raise RuntimeError('Incomplete request')
        for field,name in [('plan_sha256','plan_result.json'),('metrics_sha256','metrics.json'),('candidate_pool_sha256','candidates/candidate_pool.json')]:
            if claim[field]!=sha256(out/name):raise RuntimeError('Missing/corrupt receipt: '+name)
        payload=raw(out/'candidates/candidate_pool.json');metrics=raw(out/'metrics.json');plan=raw(out/'plan_result.json')
        if metrics['success']!=plan['success']:raise RuntimeError('Plan/metrics disagree')
        requests.append(metrics)
        if not plan['success']:continue
        selected=plan['selected'];candidate=payload['candidates'][selected['index']]
        if not np.array_equal(np.asarray(candidate['pose']),np.asarray(selected['contact_pose_world'])):
            raise RuntimeError('Planner used a different grasp from the delivered candidate')
        if selected['source_generator_index']!=candidate['metadata']['source_generator_index']:
            raise RuntimeError('Selected source generator index corrupted')
        width=float(candidate['width']);binding=selected['width_binding']
        if abs(selected['open_finger_m']-width/2)>1e-12 or any(abs(v-width/2)>1e-12 for v in binding['actual_collision_finger_locks_m'].values()):
            raise RuntimeError('Candidate/selection/collision opening mismatch')
        if any(abs(v-width/2)>2e-7 for v in binding['native_GPU_finger_locks_m'].values()):
            raise RuntimeError('Native GPU opening mismatch')
        import yaml
        actual_locks=yaml.safe_load(Path(binding['robot_model']).read_text())['robot_cfg']['kinematics']['lock_joints']
        if any(abs(actual_locks[n]-width/2)>1e-12 for n in get_arm('right').finger_joints):
            raise RuntimeError('Actual saved collision robot did not bind the selected width')
        spec=get_arm(selected['arm']);state=read_json(out/'state.json')
        for stage,key in [('grasp','grasp_world'),('pre','pregrasp_world'),('lift','lift_world')]:
            q=selected['q_'+stage];measured=deepcopy(state);measured['links']={'base_link':state['links']['base_link']}
            for name,value in zip(spec.joints,q):
                idx=state['joint_names'].index(name);lo,hi=state['joint_limits'][idx]
                if not lo<=value<=hi:raise RuntimeError('Selected joint limit violation')
                measured['qpos'][idx]=value
            actual=model.frames(measured)[spec.eef];goal=np.asarray(selected[key])
            position=float(np.linalg.norm(actual[:3,3]-goal[:3,3]));rotation=float(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude())
            if position>.002 or rotation>.03:raise RuntimeError('Independent physical-arm FK failed')
            endpoints.append({'case':item['case_id'],'variant':item['variant'],'stage':stage,
                'position_error_m':position,'rotation_error_rad':rotation})
    accepted=raw(root/'acceptance.json')
    interrupted=len(reg.get('interrupted_request_ids',[]))
    if not accepted['task_complete'] or len(requests)+interrupted!=reg['planning_requests']:raise RuntimeError('Budget incomplete')
    write_json(root/'independent_review.json',{'at':now(),'passed':True,'record_integrity':True,
        'checked_requests':len(requests),'checked_endpoints':len(endpoints),'endpoints':endpoints,
        'interrupted_requests':interrupted,'all_registered_slots_accounted':True,
        'physical_grasps':0,'default_activated':False})
    progress=raw(root/'progress.json');progress.update(at=now(),phase='complete');write_json(root/'progress.json',progress)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','review'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--offline',type=Path);p.add_argument('--gpu')
    args=p.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,ensure_run_path(args.offline),args.gpu)
    else:
        try:run(root) if args.stage=='run' else review(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
