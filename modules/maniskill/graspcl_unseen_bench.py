"""50 fresh ZeroGrasp experience scenes, followed by three frozen 100-scene arms."""
from arx_project.layout import canonical_path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import argparse
import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
import time
import traceback

import numpy as np
import graspcl_pipeline as common
import graspcl_unseen_inputs as data
import native_pair_bench as core
import short_lift_bench as base
import short_lift_protocol as lift
from benchmark_lifecycle import mark_ready, require_ready
from graspcl_execute import guard
from resident_backend import rpc

CODE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
PHASES = ['collection','A_zerograsp','B_graspcl_no_experience','C_graspcl_experience50']
GPU0 = 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169'
GPUS = ['GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e','GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834']


def aggregate(root,phase=None):
    states = {}
    for group in PHASES:
        file = root/group/'progress.json'
        states[group] = common.read(file) if file.exists() else {'phase':'pending','completed':0,'successes':0,'total':50 if group=='collection' else 100}
    old = common.read(root/'progress.json') if (root/'progress.json').exists() else {}
    common.save(root/'progress.json',{'phase':phase or old.get('phase','running'),'registered':350,
        'completed':sum(s['completed'] for s in states.values()),'groups':states,'updated_at':core.now()})


def prepare(root,inputs):
    data.prepare(inputs)
    root.mkdir(parents=True,exist_ok=False)
    (root/'source').mkdir()
    mapping = common.read(inputs/'语义类别映射.json')
    sources = {}
    files = list(CODE.glob('*.py')) + list((CODE/'arx_pipeline').glob('*.py')) + [CODE/'tmp/physical_table_pair_20261118_01.py']
    # Exclude unrelated entry scripts from the runtime guard, while pinning every
    # reused implementation listed by the validated physical-table benchmark.
    historical = common.read(CODE/'outputs/arx_physical_table_parallel_20261118_01/zerograsp/registration.json')
    wanted = {Path(p).resolve() for p in historical['source_hashes'] if Path(p).suffix in {'.py','.yml','.yaml','.urdf','.json','.sh'}}
    wanted.update(p.resolve() for p in files if p.name.startswith('graspcl_unseen_'))
    wanted.update(p.resolve() for p in (CODE/'arx_pipeline').glob('*.py'))
    wanted.add((CODE/'tmp/physical_table_pair_20261118_01.py').resolve())
    for path in sorted(wanted):
        if path.is_file(): sources[str(path)] = common.sha(path)
    for path in [inputs/'split_frozen.json',inputs/'语义类别映射.json',inputs/'isolation_preflight.json']:
        sources[str(path)] = common.sha(path)
    for path in files:
        if str(path.resolve()) in sources:
            dest=root/'source'/('arx_pipeline' if path.parent.name=='arx_pipeline' else '')/path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path,dest)
    reg={'created_at':core.now(),'inputs':str(inputs),'source_hashes':sources,'total_slots':350,
        'experience_scenes':50,'test_scenes':100,'test_arms':PHASES[1:],
        'semantic_map':str(inputs/'语义类别映射.json'),
        'experience_categories':sorted({v['category'] for v in mapping.values() if v['partition']=='experience'}),
        'test_categories':sorted({v['category'] for v in mapping.values() if v['partition']=='test'}),
        'runtime_backend_gpu':GPU0,'runtime_scene_gpus':GPUS,
        'collision_geometry':'Existing validated static_pick/reference ARX model, identical across all arms',
        'collision_policy':'arx_tcp43_lift30_open_grasp_actual_table_no_nominal_v1',
        'orientation_mode':'fixed_orientation','mpc':False,'action_tcp_eef_z_m':.043,
        'planned_lift_m':.03,'minimum_continuous_object_lift_m':.01,'required_hold_seconds':3.,
        'candidate_policy':historical['candidate_policy'],
        'shared_inputs':'A creates a frozen per-scene wrist RGB-D and native postprocessed ZeroGrasp cache; B/C reuse exact arrays and their measured capture extrinsics. Current measured joint/object states remain planner inputs. Max paired reset discrepancy: 5mrad joint, 2mm camera, 0.002 pose component.',
        'common_frontend_failure':'If A has no inference input because shared observation/admission failed, retain the same common-front-end failure in B/C, without another physical grasp or a replacement scene.',
        'bank_policy':'B: clean pretrained encoder/offline scoring only, no ARX feedback/recall. C: same pretrained components plus only clean, executed outcomes of these 50 new source scenes. Both frozen during test.',
        'primary_comparisons':['C_graspcl_experience50 vs B_graspcl_no_experience','C_graspcl_experience50 vs A_zerograsp'],
        'secondary_comparison':'B_graspcl_no_experience vs A_zerograsp',
        'statistics':{'paired_all_100_scene_success_difference':True,'bootstrap_unit':'target model',
            'bootstrap_replicates':20000,'two_sided_cluster_sign_flip_replicates':50000,'seed':2026111950100,
            'no_optional_stopping':True,'mcnemar':'exact descriptive only; repeated target models are clustered for primary intervals'},
        'no_historical_arx_feedback':True,'test_feedback_updates':False,'retry_claimed_slots':False}
    common.save(root/'registration.json',reg)
    common.save(root/'progress.json',{'phase':'prepared','registered':350,'completed':0})


def prepare_group(root,name,inputs,bank=None):
    master=common.read(root/'registration.json'); guard(master)
    manifest=common.read(inputs/'manifest.json')
    group=root/name; group.mkdir(exist_ok=False)
    profile='graspcl_200' if name.startswith(('B_','C_')) else 'zerograsp_200'
    arm_name='fixed_graspcl' if profile=='graspcl_200' else 'fixed_zerograsp'
    required=['zerograsp','graspcl','curobo'] if bank else ['zerograsp','curobo']
    source_ids=common.read(bank/'bank_complete.json')['source_ids'] if bank else []
    sources={**master['source_hashes'],**manifest['input_sha256'],**manifest['admission_sha256'],**manifest['declared_reset_sha256']}
    if bank:
        sources.update(common.read(bank/'bank_complete.json')['model_sha256'])
        sources[str(bank/'bank_complete.json')]=common.sha(bank/'bank_complete.json')
    reg={k:deepcopy(manifest[k]) for k in ['schedule','layouts','warmup_trial','success','camera','table_height_m',
        'recording_camera','test_object_ids','excluded_old_object_ids','observation_protocol','asset_sha256']}
    reg.update(source_hashes=sources,inputs=str(inputs),suite=str(group),bank=str(bank or root/'unused_bank'),
        source_feedback_ids_in_memory_order=source_ids,required_components=required,
        arms=[[arm_name,profile,'fixed_orientation']],total_slots=len(reg['schedule']),
        common_reset_source=str(inputs/'common_initial_conditions.json'),backend_lock=str(group/'backend.lock'),
        gpu_backend_free_mib_floor=2000,scene_workers=2,backend_concurrency=1,
        action_tcp_profile=str(base.existing.TCP),action_tcp_eef_z_m=.043,
        candidate_protocol='arx_tcp43_native_post_observed_insertion_truncation_v1',
        candidate_policy=master['candidate_policy'],collision_policy=master['collision_policy'],
        semantic_map=master['semantic_map'],forbidden_categories=master['test_categories'] if name=='collection' else master['experience_categories'],
        shared_input_role=None if name=='collection' else 'producer' if name=='A_zerograsp' else 'consumer',
        shared_input_cache=str(root/'shared_test_inputs'),online_updates_during_evaluation=False,
        planned_lift_m=.03,minimum_continuous_object_lift_m=.01,required_hold_seconds=3.)
    common.save(group/'registration.json',reg)
    arm=group/arm_name; arm.mkdir()
    for folder in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:
        (arm/folder).mkdir()
    for src,dest in [('common_initial_conditions.json','reference/common_initial_conditions.json'),('storage_policy.json','storage_policy.json')]:
        shutil.copy2(inputs/src,arm/dest)
    (arm/'resident').symlink_to(group/'resident',target_is_directory=True)
    reg.update(experiment_arm=arm_name,pipeline_profile=profile,orientation_mode='fixed_orientation',attempt_budget=len(reg['schedule']))
    common.save(arm/'registration.json',reg)
    for target in [group,arm]:
        common.save(target/'progress.json',{'phase':'prepared','completed':0,'successes':0,'total':len(reg['schedule'])})
    return group,arm


class Backends(core.Backends):
    def start(self):
        root=self.campaign/'resident'; root.mkdir(exist_ok=False)
        key=hashlib.sha256(str(self.campaign).encode()).hexdigest()[:12]
        sockets=CODE/'tmp'/('unseen_'+key); sockets.mkdir(mode=0o700,exist_ok=False)
        required=self.reg['required_components']
        self.config={'gpu':core.batch.BACKEND,'owner_pid':os.getpid(),
            'required_components':required,'startup_wait_seconds_per_component':600,
            'sockets':{k:str(sockets/(k+'.sock')) for k in required}}
        common.save(root/'backend_config.json',self.config)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=core.batch.BACKEND,ARX_NEURAL_GPU=core.batch.BACKEND,
            ARX_PLANNER_GPU=core.batch.BACKEND,ARX_SIM_GPU=core.batch.BACKEND,LILO_GPU=core.batch.BACKEND,
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GRASP_TRAJOPT_POLICY='interactive')
        env.pop('PYTHONPATH',None)
        for kind in required:
            python=canonical_path(CODE.parent/{'zerograsp':'tmp/arx-fullchain/env/bin/python',
                'graspcl':'tmp/arx-graspcl-env/bin/python','curobo':'curobo/.venv/bin/python'}[kind])
            command=['bash','-c','source "$1"; shift; exec "$@"','unseen-resident-service',
                str(CODE.parent/'arx_core/scripts/curobo_arx/environment.sh'),str(python),
                str(CODE/'graspcl_unseen_services.py'),'--kind',kind,'--socket',self.config['sockets'][kind],
                '--root',str(root),'--warmup-trial',self.reg['warmup_trial']]
            child_env=dict(env)
            if kind=='graspcl':
                child_env['CUDA_VISIBLE_DEVICES']=''
                command.extend(['--bank',self.reg['bank'],'--policy-registration',str(self.campaign/'registration.json')])
            common.save(root/'startup_progress.json',{'phase':'loading','component':kind,'started_at':core.now(),'timeout_seconds':600})
            stream=(root/(kind+'.log')).open('x'); self.streams.append(stream)
            child=subprocess.Popen(command,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
            self.children[kind]=child
            deadline=time.monotonic()+600
            while not Path(self.config['sockets'][kind]).exists():
                if child.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Resident startup failed: '+kind)
                time.sleep(.5)
            health=rpc(Path(self.config['sockets'][kind]),{'command':'health'})
            assert health['status']=='ready'
            print('RESIDENT READY',kind,child.pid,health['startup_seconds'],flush=True)
        mark_ready(root,required,self.config,self.children)
        common.save(root/'startup_progress.json',{'phase':'ready','at':core.now()})
        if 'graspcl' not in self.reg['required_components']:
            self.config['sockets']['graspcl']=str(Path(self.config['sockets']['zerograsp']).with_name('graspcl.sock'))
            self.config['local_adapters']={'graspcl':'Native candidate adapter; GraspCL not loaded/invoked'}
            common.save(self.campaign/'resident/backend_config.json',self.config)


class Campaign(base.Campaign):
    def progress(self,worker,**state):
        with self.mutex:
            self.states[worker]={**self.states.get(worker,{}),**state,'updated_at':core.now()}
            common.save(self.out/'workers'/(worker+'.json'),self.states[worker])
            summary=base.existing.summarize(self.out,'halted' if self.failed.is_set() else 'running',self.states)
            common.save(self.suite/'progress.json',common.read(self.out/'progress.json'))
            aggregate(self.suite.parent)

    def worker(self,index,gpu):
        worker='scene_'+str(index)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        _,baseline=core.batch.original_modules(); lift.install_verifier(baseline)
        try:
            while not self.failed.is_set():
                if (self.suite.parent/'STOP').exists(): raise RuntimeError('Explicit STOP; no further claims')
                if not self.resources(worker,gpu): break
                guard(self.reg)
                entry,claim=self.next_entry(worker,gpu)
                if entry is None: break
                trial=self.out/'trials'/entry['id']
                self.progress(worker,phase='executing',current=entry['id'],gpu=gpu)
                reason='independent_scene_admission_failed' if entry['admission_failed'] else None
                if self.reg.get('shared_input_role')=='consumer' and not (Path(self.reg['shared_input_cache'])/entry['id']/'manifest.json').exists():
                    source=self.suite.parent/'A_zerograsp/fixed_zerograsp/trials'/entry['id']/'result.json'
                    prior=common.read(source)
                    assert not prior['planning_success'] and not prior['physical_attempt']
                    reason='common_observation_failed_no_grasp_input'
                if reason:
                    trial.mkdir(exist_ok=False)
                    common.save(trial/'result.json',{**entry,'success':False,'planning_success':False,'physical_attempt':False,'reason':reason})
                    base.placeholder(trial,reason)
                else:
                    command=['bash',str(CODE/'run.sh'),str(SELF),'execute','--output',str(trial),'--entry',str(self.out/'entries'/(entry['id']+'.json'))]
                    with (self.out/'entries'/(entry['id']+'.log')).open('x') as stream:
                        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                        self.progress(worker,child_pid=child.pid)
                        code=child.wait()
                    if code: raise RuntimeError('Execution failed; preserve claimed case without replay: '+entry['id'])
                baseline.verify_trial(trial)
                result=common.read(trial/'result.json')
                guard(self.reg)
                claim.update(state='complete',finished_at=core.now(),result_sha256=common.sha(trial/'result.json'))
                common.save(self.out/'claims'/(entry['id']+'.json'),claim)
                self.progress(worker,phase='between_scenes',current=None,child_pid=None)
                print('RESULT',self.suite.name,entry['id'],result['success'],result['reason'],flush=True)
        except BaseException:
            self.failed.set()
            error={'worker':worker,'error':traceback.format_exc(),'time':core.now()}
            self.errors.append(error)
            common.save(self.out/'workers'/(worker+'_halt.json'),error)
            self.progress(worker,phase='halted',error=error['error'])

    def start(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(self.worker,i,gpu) for i,gpu in enumerate(GPUS)]
            for future in futures: future.result()
        summary=base.existing.summarize(self.out,'halted' if self.failed.is_set() else 'complete',self.states)
        common.save(self.suite/'progress.json',common.read(self.out/'progress.json'))
        aggregate(self.suite.parent)
        if self.failed.is_set(): raise RuntimeError('Campaign halted: '+str(self.errors))
        assert summary['completed']==self.reg['attempt_budget']


def run_group(group,arm):
    reg=common.read(group/'registration.json'); guard(reg)
    core.batch.BACKEND=GPU0; core.batch.SCENES=GPUS
    backends=Backends(group)
    started=time.perf_counter()
    clock={'started_at':core.now(),'phase':'starting','registered':reg['total_slots']}
    try:
        backends.start()
        require_ready(group/'resident',reg['required_components'])
        assert common.read(group/'resident/planner_warmup/physical_table_policy_ready.json')['passed']
        clock['startup_seconds']=time.perf_counter()-started
        clock['phase']='running'; common.save(group/'suite_timing.json',clock)
        common.save(group/'test_started.json',{'at':core.now(),'all_components_ready_before_first_claim':True})
        Campaign(arm).start()
        for file in (arm/'claims').glob('*.json'):
            claim=common.read(file); trial=arm/'trials'/file.stem
            assert claim['state']=='complete' and claim['result_sha256']==common.sha(trial/'result.json')
            assert common.read(trial/'independent_evaluation.json')['passed']
            assert (trial/'zg_outputs/grasp_top10.json').exists()
        guard(reg)
        health={k:rpc(Path(backends.config['sockets'][k]),{'command':'health'}) for k in reg['required_components']}
        common.save(group/'resident_health_final.json',health)
        common.save(group/'final_verification.json',{'passed':True,'terminal_results':reg['total_slots'],
            'all_slots_once':True,'bank_unchanged':True,'independent_execution_verifier':True})
        clock['phase']='complete'
    finally:
        ending=time.perf_counter(); backends.stop()
        clock.update(ended_at=core.now(),shutdown_seconds=time.perf_counter()-ending,
            total_wall_seconds=time.perf_counter()-started,all_owned_services_exited=all(p.poll() is not None for p in backends.children.values()))
        common.save(group/'suite_timing.json',clock)


def execute(trial,entry):
    # Reuse the validated fixed-pose, short-lift execution. Physical table
    # policy is installed in the resident planner, not by editing old sources.
    base.execute(trial,entry)


def analyze(root):
    from scipy.stats import binomtest
    rng=np.random.default_rng(2026111950100)
    results={}
    for name in PHASES[1:]:
        reg=common.read(root/name/'registration.json')
        arm=root/name/('fixed_zerograsp' if name=='A_zerograsp' else 'fixed_graspcl')
        results[name]={e['id']:common.read(arm/'trials'/e['id']/'result.json') for e in reg['schedule']}
    ids=list(results[PHASES[1]])
    models=sorted({r['model_id'] for r in results[PHASES[1]].values()})
    comparisons={}
    for newer,older in [(PHASES[3],PHASES[2]),(PHASES[3],PHASES[1]),(PHASES[2],PHASES[1])]:
        diffs=np.array([int(results[newer][i]['success'])-int(results[older][i]['success']) for i in ids])
        sums=np.array([sum(diffs[k] for k,i in enumerate(ids) if results[newer][i]['model_id']==m) for m in models])
        counts=np.array([sum(results[newer][i]['model_id']==m for i in ids) for m in models])
        ix=rng.integers(0,len(models),(20000,len(models)))
        boot=sums[ix].sum(1)/counts[ix].sum(1)
        signs=rng.choice([-1,1],(50000,len(models)))
        null=(signs*sums).sum(1)
        gains=int((diffs==1).sum()); losses=int((diffs==-1).sum())
        comparisons[newer+' vs '+older]={'gain_scenes':gains,'loss_scenes':losses,'paired_difference':float(diffs.mean()),
            'model_cluster_bootstrap_95_ci':np.quantile(boot,[.025,.975]).tolist(),
            'two_sided_model_cluster_signflip_p':float((1+(np.abs(null)>=abs(diffs.sum())).sum())/50001),
            'mcnemar_exact_descriptive_p':float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.}
    # Shared image/candidate equality is checked independently of arm outcomes.
    shared=0
    for slot in ids:
        receipts=[]
        for name in PHASES[1:]:
            arm='fixed_zerograsp' if name=='A_zerograsp' else 'fixed_graspcl'
            file=root/name/arm/'trials'/slot/'shared_input_receipt.json'
            if file.exists(): receipts.append(common.read(file)['sha256'])
        if receipts:
            assert len(receipts)==3 and receipts[0]==receipts[1]==receipts[2]
            shared+=1
    primary=list(comparisons)[:2]
    order=sorted(primary,key=lambda k:comparisons[k]['two_sided_model_cluster_signflip_p'])
    previous=0.
    for i,key in enumerate(order):
        previous=max(previous,min(1.,comparisons[key]['two_sided_model_cluster_signflip_p']*(len(order)-i)))
        comparisons[key]['primary_holm_adjusted_p']=previous
    supported=all(comparisons[k]['model_cluster_bootstrap_95_ci'][0]>0 and comparisons[k]['primary_holm_adjusted_p']<.05 for k in primary)
    common.save(root/'analysis.json',{'comparisons':comparisons,'shared_rgbd_native_candidate_cases':shared,
        'positive_evidence_for_added_experience_over_both_controls':supported,
        'all_100_denominator_includes_common_frontend_failures':True,'experience_bank_frozen':True})


def run(root):
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (root/'started.json').open('x') as stream: stream.write(core.now()+'\n')
        reg=common.read(root/'registration.json'); guard(reg)
        inputs=Path(reg['inputs'])
        try:
            aggregate(root,'preparing_scenes')
            for partition in ['experience','test']: data.generate(inputs/partition,GPUS)
            # Admission may change physical placements, never partition membership.
            for partition in ['experience','test']:
                plan=common.read(inputs/partition/'manifest.json')
                assert not set(plan['test_object_ids']) & set(plan['excluded_old_object_ids'])
            aggregate(root,'collecting_experience')
            for partition in ['experience','test']:
                manifest=common.read(inputs/partition/'manifest.json')
                for path,digest in manifest['asset_sha256'].items():
                    if common.sha(path)!=digest: raise RuntimeError('Frozen object asset changed: '+path)
            for name in ['verification.json','shared_input_verification.json']:
                validation=CODE/'outputs/arx_graspcl_unseen_validation_20261119_01'/name
                assert common.read(validation)['passed'], 'Service integration validation required before claiming source grasps'
            group,arm=prepare_group(root,'collection',inputs/'experience')
            run_group(group,arm)
            aggregate(root,'building_frozen_banks')
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
            env['PYTHONPATH']=str(common.UPSTREAM)+':'+str(CODE)
            with (root/'bank_build.log').open('x') as stream:
                subprocess.run([str(canonical_path(CODE.parent/'tmp/arx-graspcl-env/bin/python')),str(CODE/'graspcl_unseen_bank.py'),
                    '--output',str(root)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            assert common.read(root/'bank_provenance_audit.json')['passed']
            for name,bank in [(PHASES[1],None),(PHASES[2],root/'banks/no_experience'),(PHASES[3],root/'banks/experience50')]:
                group,arm=prepare_group(root,name,inputs/'test',bank)
                aggregate(root,'testing_'+name)
                run_group(group,arm)
            analyze(root)
            common.save(root/'final_verification.json',{'passed':True,'registered_and_completed':350,
                'all_services_exited':all(common.read(root/g/'suite_timing.json')['all_owned_services_exited'] for g in PHASES),
                'experience_test_category_model_mesh_disjoint':True,'test_feedback_updates':0})
            aggregate(root,'complete')
        except BaseException:
            common.save(root/'halt.json',{'error':traceback.format_exc(),'time':core.now(),'replay_allowed':False})
            aggregate(root,'halted')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','execute','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args(); root=args.output.resolve()
    if args.stage=='prepare': prepare(root,args.inputs.resolve())
    elif args.stage=='run': run(root)
    elif args.stage=='analyze': analyze(root)
    else: execute(root,args.entry.resolve())
