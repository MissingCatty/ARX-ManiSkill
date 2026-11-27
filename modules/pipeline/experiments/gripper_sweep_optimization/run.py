"""Register and run a bounded matched planning experiment on frozen V7 pools."""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import time
import traceback

import numpy as np
from arx_project.layout import ROOT,now,read_json,write_json,sha256,ensure_run_path
from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
from arx_project.client import Client
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard,raw

CONFIG=ROOT/'configs/experiments/gripper_sweep_optimization.json'


def prepare(root,gpu):
    if root.exists():raise ValueError('Keep old results and use a new registration')
    cfg=raw(CONFIG);source=ROOT/cfg['source_run'];old=raw(source/'registration.json')
    if not raw(source/'independent_review.json')['passed']:raise ValueError('Previous V7 review not passed')
    for name,digest in old['source_sha256'].items():
        if sha256(source/'source_registered'/Path(name).relative_to(ROOT))!=digest:raise RuntimeError('Source archive changed')
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    current=load_current();cases=[];inputs={};sources={CONFIG,ROOT/'configs/modules.json',ROOT/'configs/policies/fingertip_table.json',
        ROOT/'configs/experiments/active_experiment.json',ROOT/'modules/curobo/arx_adapter.py',ROOT/'modules/curobo/scene_cache.py'}
    for case_id in cfg['cases']:
        directory=source/'plans'/case_id/cfg['source_variant']
        if not raw(directory/'plan_result.json')['success']:raise RuntimeError('Missing successful source fixture')
        cases.append({'id':case_id,'source_directory':str(directory)})
        for name in ['state.json','candidates/candidate_pool.json','plan_result.json']:
            inputs[str(directory/name)]=sha256(directory/name)
    for folder in ['modules/curobo/gripper_sweep','modules/curobo/dual_arm','modules/curobo/fingertip_contact',
        'modules/curobo/candidate_io','modules/graspgenx/native_postprocessing','arx_project',
        'modules/pipeline/experiments/gripper_sweep_optimization','modules/maniskill/planning_common']:
        sources.update((ROOT/folder).glob('*.py'))
    for name in ['resident_planner.py','fast_resident_planner.py','checked_native_services.py','short_lift_planner.py']:
        sources.add(ROOT/'modules/maniskill'/name)
    sources.update(ROOT/'modules/curobo/indexed_planning'/n for n in ['stage_timing.py'])
    sources.add(ROOT/'modules/pipeline/experiments/dual_arm_grasp/benchmark_acceleration.py')
    sources.update(Path(p) for p in current['configuration_hashes'])
    root.mkdir(parents=True);(root/'claims').mkdir();(root/'warmup').mkdir()
    for name in ['state.json','closed_state.json']:
        shutil.copy2(source/'warmup'/name,root/'warmup'/name);inputs[str(root/'warmup'/name)]=sha256(root/'warmup'/name)
    for path in sources:
        target=root/'source_registered'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    schedule=[]
    for ordinal,case in enumerate(cases):
        order=[('legacy',1),('batched',1),('batched',2),('legacy',2)] if ordinal%2==0 else \
              [('batched',1),('legacy',1),('legacy',2),('batched',2)]
        schedule.extend({'id':case['id']+'_'+mode+'_'+str(repeat),'case_id':case['id'],
            'variant':mode+'_'+str(repeat),'mode':mode,'repeat':repeat} for mode,repeat in order)
    write_json(root/'registration.json',{'at':now(),'authorization':cfg['authorization'],'phase':cfg['phase'],
        'gpu':gpu,'cases':cases,'variants':sorted({i['variant'] for i in schedule}),'schedule':schedule,
        'planning_requests':len(schedule),'physical_grasps':0,'automatic_retries':False,
        'source_sha256':{str(p):sha256(p) for p in sorted(sources)},'input_sha256':inputs,
        'same_candidates_robot_physics_sampling_and_native_constraints':True,
        'performance_gate':'lower total and median; P90 no increase above5percent; original success and selection retained',
        'request_deadline_seconds':1800,'default_activation':False})
    write_json(root/'research.md','# 开合爪批量原生校验\n\n## Goal\n降低完整规划请求与候选搜索耗时。\n\n'
        '## Success Metric\n原规划/抓法/选臂/宽度保持，完整RPC累计与中位下降，P90不增加5%。\n\n'
        '## Constraints\n- max_iterations:1\n- pause_every:never\n- keep_policy:score_improvement\n'
        '- Evaluator: native per-sample sphere/mask preflight and paired independent review\n'
        '- guard: same input,samples,871,TCP43,V2,32seeds,numeric thresholds and independent FK\n'
        '- noise_runs:2\n- 20 requests ABBA/BAAB,0physical,no automatic retries\n'
        '- per-request1800s deadlines supersede whole-campaign shell setup timeout\n\n'
        '## Current Approach\nReuse existing native sphere batching for opening and narrowing.\n\n'
        '## Search Space\nOne batched sweep change; candidate search profiling for next phase.\n\n'
        '## History\n| Iteration | Work | Result | Status |\n| --- | --- | --- | --- |\n')
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,'total_requests':len(schedule),'results':[]})


def summarize(rows):
    result={}
    for mode in ['legacy','batched']:
        group=[r for r in rows if r['mode']==mode]
        if not group:continue
        seconds=[r['RPC_seconds'] for r in group];stages={}
        for row in group:
            for name,value in row['stages'].items():
                target=stages.setdefault(name,{'calls':0,'wall_seconds':0.,'exclusive_seconds':0.})
                for field in target:target[field]+=value[field]
        result[mode]={'requests':len(group),'successes':sum(r['success'] for r in group),
            'total_seconds':sum(seconds),'median_seconds':float(np.median(seconds)),'p90_seconds':float(np.percentile(seconds,90)),
            'candidate_checks':sum(r['candidate_checks'] for r in group),'stages':stages}
    return result


def run(root):
    reg=guard(root)
    if (root/'STOP').exists() or (root/'started.json').exists():raise ValueError('Do not repeat a registered execution')
    from modules.pipeline.experiments.indexed_grasp_planning.physical.controller import environment,wait_ready
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu':reg['gpu']})
    socket=ROOT/'tmp/rpc'/('sweep'+sha256(root/'registration.json')[:10]+'.sock');socket.parent.mkdir(parents=True,exist_ok=True)
    client=Client(socket,timeout=1800,candidate_transport=True);rows=[];cases={c['id']:c for c in reg['cases']}
    command=[str(ROOT/'environments/curobo/bin/python'),'-u','-m','modules.pipeline.experiments.gripper_sweep_optimization.worker',
        '--root',str(root/'resident'),'--socket',str(socket),'--warmup-trial',str(root/'warmup'),'--descriptor',str(ROOT/'configs/modules.json')]
    with (root/'planner.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment(reg['gpu']),stdout=log,stderr=subprocess.STDOUT)
        write_json(root/'planner_process.json',{'at':now(),'pid':process.pid,'command':command})
        try:
            wait_ready(process,root/'resident/curobo_dual_arm_ready.json')
            for item in reg['schedule']:
                guard(root);out=root/'plans'/item['case_id']/item['variant'];claim=root/'claims'/(item['id']+'.json')
                if out.exists() or claim.exists():raise RuntimeError('Request already claimed')
                out.mkdir(parents=True);source=Path(cases[item['case_id']]['source_directory'])
                shutil.copy2(source/'state.json',out/'state.json');payload=raw(source/'candidates/candidate_pool.json')
                write_json(claim,{'at':now(),'state':'claimed',**item})
                write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':len(rows),
                    'total_requests':len(reg['schedule']),'current':item,'results':rows,'summaries':summarize(rows)})
                began=time.perf_counter();descriptor=publish_candidate_pool(out,payload)
                response=client.request(out,variant=item['mode'],candidate_transport=descriptor)
                seconds=time.perf_counter()-began;finish_candidate_transport(out)
                if response['status']!='ok':raise RuntimeError('Planning service fault: '+str(response))
                row={**response['result'],**item,'RPC_seconds':seconds};write_json(out/'metrics.json',row)
                write_json(claim,{'at':now(),'state':'complete',**item,'plan_sha256':sha256(out/'plan_result.json'),
                    'metrics_sha256':sha256(out/'metrics.json'),'candidate_pool_sha256':sha256(out/'candidates/candidate_pool.json')})
                rows.append(row)
                write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':len(rows),
                    'total_requests':len(reg['schedule']),'results':rows,'summaries':summarize(rows)})
                print('SWEEP_PLAN',len(rows),item['case_id'],item['variant'],row['success'],round(seconds,4),flush=True)
            write_json(root/'acceptance.json',{'at':now(),'task_complete':True,'planning_requests':len(rows),
                'summaries':summarize(rows),'physical_grasps':0,'default_activated':False})
        finally:
            close_owned_transports()
            if process.poll() is None:
                try:client.call({'command':'shutdown'})
                except Exception:process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':process.poll() is not None,'returncode':process.returncode})
    from modules.pipeline.experiments.graspgenx_native_postprocessing.planning import review
    review(root);paired_review(root)


def paired_review(root):
    reg=guard(root);pairs=[]
    for case in reg['cases']:
        for repeat in [1,2]:
            a,b=[raw(root/'plans'/case['id']/(m+'_'+str(repeat))/'plan_result.json') for m in ['legacy','batched']]
            if not a['success'] or a['success']!=b['success']:raise RuntimeError('Control success lost')
            sa,sb=a['selected'],b['selected']
            if any(sa[k]!=sb[k] for k in ['arm','original_native_rank','contact_pose_world','open_finger_m']):raise RuntimeError('Selected grasp/arm/width changed')
            deltas={k:float(np.abs(np.asarray(sa[k])-np.asarray(sb[k])).max()) for k in ['q_grasp','q_pre','q_lift']}
            if max(deltas.values())>2e-5:raise RuntimeError('Joint endpoint changed beyond comparison precision')
            path_deltas={}
            for stage in ['pregrasp','approach']:
                for field in ['q','times']:
                    x,y=[np.asarray(p['paths'][stage][field]) for p in [a,b]]
                    if x.shape!=y.shape:raise RuntimeError('Path shape changed')
                    path_deltas[stage+'_'+field]=float(np.abs(x-y).max())
            pairs.append({'case':case['id'],'repeat':repeat,'joint_deltas':deltas,'path_deltas':path_deltas,
                'joint_endpoints_bitwise_equal':all(v==0 for v in deltas.values()),
                'paths_bitwise_equal':all(v==0 for v in path_deltas.values())})
    summaries=raw(root/'acceptance.json')['summaries'];a,b=[summaries[m] for m in ['legacy','batched']]
    timing=b['total_seconds']<a['total_seconds'] and b['median_seconds']<a['median_seconds'] and b['p90_seconds']<=a['p90_seconds']*1.05
    write_json(root/'paired_review.json',{'at':now(),'passed':True,'record_integrity':True,'pairs':pairs,
        'all_original_successes_grasps_arms_widths_retained':True,'timing_gate_passed':timing,
        'total_saving_fraction':1-b['total_seconds']/a['total_seconds'],
        'all_endpoints_bitwise_equal':all(p['joint_endpoints_bitwise_equal'] for p in pairs),
        'all_paths_bitwise_equal':all(p['paths_bitwise_equal'] for p in pairs),
        'maximum_joint_delta_rad':max(max(p['joint_deltas'].values()) for p in pairs),
        'maximum_path_delta':max(max(p['path_deltas'].values()) for p in pairs),'physical_grasps':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','review']);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu')
    args=p.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,args.gpu)
    else:
        try:run(root) if args.stage=='run' else paired_review(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
