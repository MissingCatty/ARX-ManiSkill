"""Register isolated sphere-count variants and run planning-only comparisons."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np
import yaml

from arx_project.layout import ROOT,now,read_json,write_json,sha256,ensure_run_path
from arx_project.client import Client
from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard
from modules.pipeline.experiments.gripper_sweep_optimization import run as base
from modules.curobo.sphere_study.profiles import create_profile,digest

CONFIG=ROOT/'configs/experiments/sphere_count_diagnosis.json'


def prepare(root,gpu):
    cfg=read_json(CONFIG)
    with patch.object(base,'CONFIG',CONFIG):base.prepare(root,gpu)
    reg=read_json(root/'registration.json');source=read_json(ROOT/'simulation/robots/arx/collision/arx_refined_871.json')
    current=ROOT/cfg['accepted_fast_source']
    if not read_json(current/'reuse_review.json')['kept'] or not read_json(current/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Accepted predecessor is not complete and stopped')
    reg['source_sha256'].update({str(p):sha256(p) for folder in [
        ROOT/'modules/curobo/sphere_study',ROOT/'modules/pipeline/experiments/sphere_count_diagnosis']
        for p in folder.glob('*.py')})
    reg['source_sha256'][str(ROOT/'modules/curobo/indexed_planning/ik_profiling.py')]=sha256(ROOT/'modules/curobo/indexed_planning/ik_profiling.py')
    for name in reg['source_sha256']:
        target=root/'source_registered'/Path(name).relative_to(ROOT)
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    refs=ROOT/'modules/arx_core/scripts/curobo_arx/static_pick/reference';profiles=[]
    for count in cfg['sphere_counts']:
        profile=create_profile(source,count);out=root/'profiles'/str(count)
        write_json(out/'profile.json',profile)
        reg['input_sha256'][str(out/'profile.json')]=sha256(out/'profile.json')
        for kind in ['model','model_closed']:
            target=out/'arx_source/scripts/curobo_arx/static_pick/reference'/kind
            target.mkdir(parents=True)
            model=yaml.load((refs/kind/'arx.yml').read_text(),Loader=yaml.CSafeLoader)
            if digest(model['robot_cfg']['kinematics']['collision_spheres'])!=source['sphere_parameter_sha256']:
                raise RuntimeError('Actual source references differ from approved spheres')
            model['robot_cfg']['kinematics']['collision_spheres']=deepcopy(profile['collision_spheres'])
            model['robot_cfg']['kinematics']['urdf_path']=str(target/'arx.urdf')
            (target/'arx.yml').write_text(yaml.dump(model,Dumper=yaml.CSafeDumper,sort_keys=False))
            (target/'arx.urdf').symlink_to(refs/kind/'arx.urdf')
            for file in [target/'arx.yml',target/'arx.urdf']:
                reg['input_sha256'][str(file)]=sha256(file)
        profiles.append({'sphere_count':count,'directory':str(out.relative_to(ROOT)),
            'parameter_sha256':profile['sphere_parameter_sha256'],
            'counts':{n:len(v) for n,v in profile['collision_spheres'].items()}})
    schedule=[]
    for count in cfg['sphere_counts']:
        for repeat in [1,2]:
            cases=reg['cases'] if repeat==1 else list(reversed(reg['cases']))
            schedule.extend({'id':c['id']+f'_s{count}_r{repeat}','case_id':c['id'],
                'variant':f's{count}_r{repeat}','sphere_count':count,'diagnostic_profile':False} for c in cases)
        if count==871:
            schedule.extend({'id':case+f'_s{count}_profile','case_id':case,
                'variant':f's{count}_profile','sphere_count':count,'diagnostic_profile':True} for case in cfg['profile_cases'])
    micro={}
    for case in reg['cases']:
        source_dir=current/'plans'/case['id']/'batched_1';micro[case['id']]=str(source_dir.relative_to(ROOT))
        for name in ['state.json','plan_result.json']:
            reg['input_sha256'][str(source_dir/name)]=sha256(source_dir/name)
    reg.update(profiles=profiles,schedule=schedule,variants=sorted({i['variant'] for i in schedule}),
        planning_requests=len(schedule),unprofiled_requests=30,profiled_requests=2,micro_sources=micro,
        fixed_work_RPC_requests=5,same_candidates_robot_physics_sampling=True,
        same_non_geometry_constraints=True,native_collision_geometry_varied=True,
        performance_gate='measurement completeness and source preservation; no automatic sphere-count adoption',
        sphere_count_changes_only_in_registered_child_references=True,production_sphere_count=871,
        physical_grasps=0,default_activation=False,automatic_retries=False,
        preserve_source_bytes={str(p):sha256(p) for p in [ROOT/'configs/modules.json',
            ROOT/'configs/experiments/active_experiment.json',ROOT/'simulation/robots/arx/collision/arx_refined_871.json',
            refs/'model/arx.yml',refs/'model_closed/arx.yml']})
    reg.pop('same_candidates_robot_physics_sampling_and_native_constraints',None)
    write_json(root/'registration.json',reg)
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,
        'total_requests':len(schedule),'results':[]})
    (root/'research.md').write_text('# 碰撞球数量对完整规划的影响\n\n## Goal\n量化871/512/256球对固定工作量和完整规划的影响。\n\n'
        '## Success Metric\n30普通规划＋2剖析请求均有完整回执；固定关节输入一致、计时与碰撞差异完整、默认配置未改。\n\n'
        '## Constraints\n- max_iterations:1\n- pause_every:never\n- keep_policy:pass_only\n'
        '- noise_runs:2\n- No production adoption; simulation/physical0\n'
        '- Resource10000/1536MiB and original CUDA probe; no retries\n'
        '- Profiler measurements excluded from ordinary timing\n'
        '- Same source scenes/poses/scores/widths,6DOF,32seeds,TCP43,V2,tolerances\n'
        '- Per-request1800s deadline; no whole-campaign five-minute limit\n\n'
        '## Search Space\nThree conservative same-link sphere-count profiles.\n\n'
        '## History\n| Iteration | Work | Result | Status |\n| --- | --- | --- | --- |\n')


def summarize(rows):
    summaries={}
    for count in sorted({r['sphere_count'] for r in rows},reverse=True):
        values=[r for r in rows if r['sphere_count']==count and not r['diagnostic_profile']]
        if not values:continue
        seconds=[r['RPC_seconds'] for r in values]
        stages={}
        for row in values:
            for name,value in row['stages'].items():
                item=stages.setdefault(name,{'wall_seconds':0.,'exclusive_seconds':0.,'calls':0})
                for key in item:item[key]+=value[key]
        summaries[str(count)]={'requests':len(values),'successes':sum(r['success'] for r in values),
            'mean_seconds':float(np.mean(seconds)),'median_seconds':float(np.median(seconds)),
            'p90_seconds':float(np.percentile(seconds,90)),'total_seconds':sum(seconds),
            'candidate_checks':sum(r['candidate_checks'] for r in values),'stages':stages}
    return summaries


def run(root):
    reg=guard(root)
    if (root/'started.json').exists() or (root/'STOP').exists():raise RuntimeError('Do not restart a registered execution')
    from modules.pipeline.experiments.indexed_grasp_planning.physical.controller import environment,wait_ready
    write_json(root/'started.json',{'at':now(),'gpu':reg['gpu']})
    cases={c['id']:c for c in reg['cases']};rows=[];stops=[]
    try:
        for profile in reg['profiles']:
            count=profile['sphere_count'];worker=root/'workers'/str(count);worker.mkdir(parents=True)
            socket=ROOT/'tmp/rpc'/('sph'+sha256(root/'registration.json')[:8]+str(count)+'.sock')
            socket.parent.mkdir(parents=True,exist_ok=True);client=Client(socket,timeout=1800,candidate_transport=True)
            command=[str(ROOT/'environments/curobo/bin/python'),'-u','-m','modules.pipeline.experiments.sphere_count_diagnosis.worker',
                '--root',str(worker),'--socket',str(socket),'--warmup-trial',str(root/'warmup'),
                '--descriptor',str(ROOT/'configs/modules.json'),'--profile',str(ROOT/profile['directory']/'profile.json'),
                '--reference-root',str(ROOT/profile['directory']/'arx_source')]
            write_json(root/'progress.json',{'at':now(),'phase':'warming','current_sphere_count':count,
                'completed_requests':len(rows),'total_requests':len(reg['schedule']),'results':rows,'summaries':summarize(rows)})
            with (worker/'planner.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=environment(reg['gpu']),stdout=log,stderr=subprocess.STDOUT)
                write_json(worker/'process.json',{'at':now(),'pid':child.pid,'command':command})
                write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':False,'active_sphere_count':count,'groups':stops})
                try:
                    wait_ready(child,worker/'curobo_dual_arm_ready.json')
                    for item in [s for s in reg['schedule'] if s['sphere_count']==count]:
                        guard(root);out=root/'plans'/item['case_id']/item['variant'];claim=root/'claims'/(item['id']+'.json')
                        if out.exists() or claim.exists():raise RuntimeError('Planning request already claimed')
                        out.mkdir(parents=True);source=Path(cases[item['case_id']]['source_directory'])
                        shutil.copy2(source/'state.json',out/'state.json');write_json(claim,{'at':now(),'state':'claimed',**item})
                        write_json(root/'progress.json',{'at':now(),'phase':'planning','current':item,
                            'completed_requests':len(rows),'total_requests':len(reg['schedule']),'results':rows,'summaries':summarize(rows)})
                        began=time.perf_counter();descriptor=publish_candidate_pool(out,read_json(source/'candidates/candidate_pool.json'))
                        response=client.request(out,diagnostic_profile=item['diagnostic_profile'],candidate_transport=descriptor)
                        seconds=time.perf_counter()-began;finish_candidate_transport(out)
                        if response['status']!='ok':raise RuntimeError('Sphere study service fault: '+str(response))
                        row={**response['result'],**item,'RPC_seconds':seconds};write_json(out/'metrics.json',row)
                        write_json(claim,{'at':now(),'state':'complete',**item,'plan_sha256':sha256(out/'plan_result.json'),
                            'metrics_sha256':sha256(out/'metrics.json'),'candidate_pool_sha256':sha256(out/'candidates/candidate_pool.json')})
                        rows.append(row)
                        write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':len(rows),
                            'total_requests':len(reg['schedule']),'results':rows,'summaries':summarize(rows)})
                        print('SPHERE_PLAN',len(rows),count,item['case_id'],item['variant'],row['success'],round(seconds,4),flush=True)
                    for ordinal,case in enumerate(reg['cases'] if count==871 else []):
                        guard(root);out=root/'micro'/str(count)/case['id']
                        if out.exists():raise RuntimeError('Fixed-work microbenchmark already claimed')
                        out.mkdir(parents=True)
                        write_json(root/'progress.json',{'at':now(),'phase':'fixed_work',
                            'current_fixed_work_case':case['id'],'fixed_work_cases_completed':ordinal,
                            'completed_requests':len(rows),'total_requests':len(reg['schedule']),
                            'results':rows,'summaries':summarize(rows)})
                        response=client.request(out,micro_source=reg['micro_sources'][case['id']],ordinal=ordinal)
                        if response['status']!='ok':raise RuntimeError('Fixed-work microbenchmark failed: '+str(response))
                finally:
                    close_owned_transports()
                    if child.poll() is None:
                        try:client.call({'command':'shutdown'})
                        except Exception:child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                    stop={'sphere_count':count,'all_owned_exited':child.poll() is not None,'returncode':child.returncode}
                    stops.append(stop);write_json(worker/'services_stopped.json',{'at':now(),**stop})
                    write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':all(s['all_owned_exited'] for s in stops),'groups':stops})
        guard(root)
        for path,digest_value in reg['preserve_source_bytes'].items():
            if sha256(path)!=digest_value:raise RuntimeError('Production configuration changed during sphere diagnosis')
        write_json(root/'acceptance.json',{'at':now(),'task_complete':True,'planning_requests':len(rows),
            'summaries':summarize(rows),'physical_grasps':0,'default_activated':False})
        from modules.pipeline.experiments.graspgenx_native_postprocessing.planning import review
        review(root)
    finally:close_owned_transports()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpu');args=parser.parse_args()
    root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,args.gpu)
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
