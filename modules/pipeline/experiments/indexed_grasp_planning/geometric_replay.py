"""Compare D's geometric-first endpoint checks with the unchanged scalar IK."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from .replay import prepare as prepare_inputs, verify_hashes


def launch(root, gpu, selected=None):
    cfg_path=ROOT/'configs/experiments/indexed_grasp_planning.json'
    config=read_json(cfg_path)
    active=ROOT/config['execution']['active_run']
    if (active/'controller.json').is_file():
        unit=read_json(active/'controller.json')['unit']
        if subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID'],text=True).strip()!='MainPID=0':
            raise RuntimeError('The preceding active V8 planner has not exited')
    preceding=ROOT/config['binding_reuse_implementation']['full_run']
    if not read_json(preceding/'independent_review.json')['passed']:
        raise RuntimeError('Complete and independently review C before D uses its GPU')
    state=subprocess.check_output(['systemctl','--user','show',read_json(preceding/'controller.json')['unit'],
        '-p','MainPID'],text=True).strip()
    if state!='MainPID=0':raise RuntimeError('C planner has not exited')
    preflight=ROOT/config['geometric_ik']['contract_run']
    if not read_json(preflight/'acceptance.json')['passed']:
        raise RuntimeError('Geometric physical-chain contract has not passed')
    verify_hashes(read_json(preflight/'registration.json'))
    if not selected:
        pilot=ROOT/config['geometric_ik']['pilot_run']
        accepted=read_json(pilot/'acceptance.json')
        if (not read_json(pilot/'independent_review.json')['passed']
                or not all(accepted['control_successes_retained'].values())):
            raise RuntimeError('D pilot must retain control successes and pass independent review')
    prepare_inputs(root,ROOT/config['baseline_run'],gpu,selected)
    reg=read_json(root/'registration.json')
    reg.update(experiment='geometric_ik',native_batching=False,context_reuse=False,
        user_authorization='2026-11-26：继续执行V8优化方案，C完整对照后比较D几何IK。',
        variants=({'control_1':False,'geometric_1':True,'geometric_2':True,'control_2':False}
            if selected else {'control':False,'geometric':True}),
        original_IK_fallback='no_geometric_branch_or_all_geometric_branches_rejected',
        geometric_prefetch_capacity=16,independent_CPU_environment='environments/geometric_ik',
        selection_changes='record_original_candidate_and_arm_changes_then_independent_FK_review')
    reg['planning_requests']=len(reg['cases'])*len(reg['variants'])
    sources=list((ROOT/'modules/curobo/indexed_planning/geometric').glob('*.py'))
    sources += [Path(__file__),Path(__file__).with_name('geometric_followup.py'),
        ROOT/'configs/experiments/indexed_grasp_geometric_ik.json',ROOT/'configs/environments/geometric_ik.txt']
    for source in sources:
        target=root/'source_registered'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        reg['source_hashes'][str(source)]=sha256(source)
    reg['source_hashes'].update({p:d for p,d in read_json(preflight/'registration.json')['source_hashes'].items()
        if 'site-packages/eaik/' in p})
    write_json(root/'registration.json',reg)
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,'total_requests':reg['planning_requests']})
    write_json(root/'configuration_before.json',config)
    config['execution'].update(active_run=str(root.relative_to(ROOT)),
        current_phase='geometric_ik_pilot' if selected else 'geometric_ik_full_replay',
        next_stage='review_geometric_IK_then_index_reuse_and_effective_combination')
    config['implementation_status']='B_C_complete_geometric_IK_comparison_running'
    config['geometric_ik']['pilot_run' if selected else 'full_run']=str(root.relative_to(ROOT))
    for variant in config['variants']:
        if variant['id']=='geometric_ik':variant['state']='pilot_running' if selected else 'full_replay_running'
    write_json(cfg_path,config)
    environment=read_json(preceding/'controller.json')['cuda_environment']
    unit='arx-v8-'+root.name.replace('_','-')+'.service'
    command=['systemd-run','--user','--unit='+unit,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(root/'controller.log'),
        '--property=StandardError=inherit']
    command += ['--setenv='+name+'='+value for name,value in environment.items()]
    command += [str(ROOT/'environments/curobo/bin/python'),'-m',
        'modules.pipeline.experiments.indexed_grasp_planning.geometric_replay','run','--root',str(root)]
    subprocess.run(command,check=True)
    write_json(root/'controller.json',{'at':now(),'unit':unit,'cuda_environment':environment,'physical_grasps':0})
    print({'unit':unit,'planning_requests':reg['planning_requests']},flush=True)


def run(root):
    reg=read_json(root/'registration.json')
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Do not repeat a registered D request')
    verify_hashes(reg)
    os.environ.update(CUDA_VISIBLE_DEVICES=reg['gpu'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        GRASP_TRAJOPT_POLICY='interactive',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import torch
    from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
    from modules.curobo.indexed_planning.geometric.service import create
    from modules.curobo.scene_cache import mesh_stores,prune_inactive_meshes
    torch.set_num_threads(4)
    free,_=torch.cuda.mem_get_info()
    if free/1024**2<reg['minimum_startup_free_mib']:raise RuntimeError('Insufficient CUDA memory before D loading')
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu':reg['gpu'],'free_mib':free/1024**2})
    planner=None
    try:
        began=time.perf_counter()
        planner=create(SimpleNamespace(root=root/'resident',warmup_trial=root/'warmup'));pool=planner.pool
        state=read_json(root/'warmup/state.json')
        for side in ['right','left']:
            canonical=planner.model.bind(state,side,root/'prewarm'/side/'state')
            context=pool.bind(root/'prewarm'/side/'open',canonical)
            pool.contact_view(context,canonical)
            start=context.measured(canonical);far=np.eye(4);far[:3,3]=[10.,10.,10.]
            if context.ik(far,start) is not None or len(pool.unconstrained_ik(context,far,start)):
                raise RuntimeError('Negative native IK warmup unexpectedly passed')
            kin=context.planner.compute_kinematics(context.js(start)).tool_poses
            quaternion=kin.quaternion.detach().cpu().numpy().reshape(-1,4)[0]
            positive=np.eye(4);positive[:3,3]=kin.position.detach().cpu().numpy().reshape(-1,3)[0]
            positive[:3,:3]=Rotation.from_quat([*quaternion[1:],quaternion[0]]).as_matrix()
            targets=[positive.tolist(),far.tolist()]
            result=pool.geometric_client.solve(targets,start.cpu().numpy().tolist())
            if not result['answers'][0]['admitted_branches'] or result['answers'][1]['admitted_branches']:
                raise RuntimeError('Positive/negative geometric warmup failed')
        torch.cuda.synchronize()
        write_json(root/'startup.json',{'at':now(),'seconds':time.perf_counter()-began,
            'model_loads':pool.loads,'worker':pool.geometric_client.ready,
            'allocated_mib':torch.cuda.memory_allocated()/1024**2,'negative_paths_warmed':True})
        rows=[];completed=0
        for ordinal,case in enumerate(reg['cases']):
            pair={};order=list(reg['variants'])
            if ordinal%2:order=([order[1],order[0],order[3],order[2]] if len(order)==4 else order[::-1])
            for mode in order:
                free,_=torch.cuda.mem_get_info()
                if free/1024**2<reg['minimum_request_free_mib']:raise RuntimeError('Insufficient CUDA reserve for D')
                directory=root/'plans'/case['id']/mode;directory.mkdir(parents=True)
                source=Path(case['source_directory']);shutil.copy2(source/'state.json',directory/'state.json')
                descriptor=publish_candidate_pool(directory,read_json(source/'candidates/candidate_pool.json'))
                pool.geometric_enabled=reg['variants'][mode]
                torch.manual_seed(reg['random_seed']);np.random.seed(reg['random_seed'])
                for context in pool.contexts.values():context.planner.reset_seed()
                pool.pure_solver.reset_seed()
                write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':completed,
                    'total_requests':reg['planning_requests'],'current':{'id':case['id'],'variant':mode},'results':rows})
                print('PLAN_BEGIN',case['id'],mode,flush=True)
                torch.cuda.synchronize();began=time.perf_counter()
                result=planner.request(directory,candidate_transport=descriptor)
                torch.cuda.synchronize();seconds=time.perf_counter()-began
                finish_candidate_transport(directory)
                io=read_json(directory/'candidate_io.json')
                if io['source']!='shared_memory' or io['decode_count']!=1 or io['full_arm_pool_copies']!=0:
                    raise RuntimeError('D changed the shared-memory candidate contract')
                checks=0;geometry=[]
                for side in ['right','left']:
                    work=directory/'arms'/side
                    if (work/'plan_result.json').is_file():checks+=len(read_json(work/'plan_result.json').get('candidate_events',[]))
                    for receipt in sorted((work/'search_windows').glob('*.json')):
                        data=read_json(receipt)['result'].get('geometric_ik')
                        if data:geometry.append({'arm':side,**data})
                pair[mode]={'seconds':seconds,'success':result['success'],
                    'selected_arm':result.get('selected',{}).get('arm'),
                    'selected_source_index':result.get('selected',{}).get('index'),
                    'selected_contact_pose_world':result.get('selected',{}).get('contact_pose_world'),
                    'candidate_checks':checks,'geometric':geometry,'request_free_mib':free/1024**2}
                write_json(directory/'replay_metrics.json',pair[mode]);completed+=1
                print('PLAN_END',case['id'],mode,round(seconds,4),'success',result['success'],'checks',checks,flush=True)
                prune_inactive_meshes(mesh_stores(planner),torch.cuda.synchronize)
            comparison={}
            for mode,value in pair.items():
                if not reg['variants'][mode]:continue
                control=pair[mode.replace('geometric','control')]
                comparison[mode]={'success_retained':not control['success'] or value['success'],
                    'same_selection':all(value[k]==control[k] for k in
                        ['success','selected_arm','selected_source_index','selected_contact_pose_world']),
                    'speedup':control['seconds']/value['seconds']}
            rows.append({'id':case['id'],'variants':pair,'comparison':comparison})
            write_json(root/'progress.json',{'at':now(),'phase':'comparison','completed_requests':completed,
                'total_requests':reg['planning_requests'],'results':rows})
        verify_hashes(reg);summaries={}
        for group,enabled in [('control',False),('geometric',True)]:
            entries=[v for r in rows for m,v in r['variants'].items() if reg['variants'][m]==enabled]
            times=[e['seconds'] for e in entries]
            summaries[group]={'requests':len(entries),'successes':sum(e['success'] for e in entries),
                'median_seconds':float(np.median(times)),'p90_seconds':float(np.percentile(times,90)),
                'total_seconds':sum(times),'candidate_checks':sum(e['candidate_checks'] for e in entries),
                'geometric_accepted_endpoints':sum(event['accepted_geometric_branch'] for e in entries for g in e['geometric'] for event in g['events']),
                'original_endpoint_fallbacks':sum(event['original_IK_fallback'] for e in entries for g in e['geometric'] for event in g['events'])}
        retained={m:all(r['comparison'][m]['success_retained'] for r in rows) for m,en in reg['variants'].items() if en}
        write_json(root/'acceptance.json',{'at':now(),'passed':True,'record_integrity':True,
            'task_complete':completed==reg['planning_requests'],'planning_requests':completed,
            'summaries':summaries,'control_successes_retained':retained,'paired_cases':rows,
            'physical_grasps':0,'simulation_instances':0,'perception_calls':0,'automatic_default_activation':False})
        write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_requests':completed,
            'total_requests':reg['planning_requests'],'results':rows})
    finally:
        close_owned_transports()
        if planner is not None:planner.pool.geometric_client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['launch','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpu');parser.add_argument('--cases',nargs='+')
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='launch':launch(root,args.gpu,args.cases)
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'physical_grasps':0});raise
