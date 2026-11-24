"""Registered continuation after 49 completed source cases and one interruption.

No source grasp is replayed. The interrupted case has an unknown grasp outcome
and is excluded from feedback. The original 100-scene A/B/C protocol is reused.
"""
from arx_project.layout import canonical_path
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback

import graspcl_unseen_ready_resume_20261119 as driver
import graspcl_unseen_shared_backends_20261119 as shared
import graspcl_unseen_resident_resume_20261119 as resident
from arx_service_readiness import wait_ready
from graspcl_execute import guard
from resident_backend import rpc

common=resident.common
bench=resident.bench
CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
EPOCH='continuation_epoch07'
INTERRUPTED='E0050_S2_edge_004_sugar_box'
ORIGINAL_PREPARE=bench.prepare_group
ORIGINAL_AGGREGATE=bench.aggregate


def aggregate(root,phase=None):
    ORIGINAL_AGGREGATE(root,phase)
    value=common.read(root/'progress.json')
    value.update(runtime_epoch=7,source_infrastructure_interruptions=1,
        source_grasps_with_completed_results=49,source_verified_successes=39,
        source_interrupted_case_replayed=False,
        historical_halts_retained=['epoch01','epoch02','epoch03','epoch04','epoch05','epoch06'])
    if resident.POOL is not None:value['simulation_workers']=resident.POOL.status()
    if shared.MANAGER is not None:
        value['model_services']={name:{'pid':r['process'].pid,'alive':r['process'].poll() is None,
            'kind':r['kind']} for name,r in shared.MANAGER.all_processes.items()}
    common.save(root/'progress.json',value)


def source_evidence(root):
    arm=root/'collection/fixed_zerograsp'
    reg=common.read(arm/'registration.json');guard(reg)
    completed={}
    for entry in reg['schedule']:
        claim=arm/'claims'/(entry['id']+'.json');record=common.read(claim)
        trial=arm/'trials'/entry['id']
        if entry['id']==INTERRUPTED:
            assert record['state']=='claimed' and not (trial/'result.json').exists()
            continue
        assert record['state']=='complete'
        assert record['result_sha256']==common.sha(trial/'result.json')
        assert common.read(trial/'independent_evaluation.json')['passed']
        completed[entry['id']]={str(p):common.sha(p) for p in [claim,trial/'result.json',trial/'independent_evaluation.json']}
    assert len(completed)==49
    assert sum(common.read(arm/'trials'/name/'result.json')['success'] for name in completed)==39
    trial=arm/'trials'/INTERRUPTED
    evidence={str(p):common.sha(p) for p in trial.rglob('*') if p.is_file()}
    evidence[str(arm/'claims'/(INTERRUPTED+'.json'))]=common.sha(arm/'claims'/(INTERRUPTED+'.json'))
    return completed,evidence


def prepare(root):
    guard(common.read(root/'registration.json'))
    assert common.read(root/'continuation_epoch06/model_services_stopped.json')['all_owned_model_services_exited']
    assert common.read(root/'continuation_epoch06/simulation_stopped.json')['all_owned_simulators_exited']
    assert not any((root/name).exists() for name in bench.PHASES[1:])
    assert not (root/'banks').exists() and not (root/'bank_build.log').exists()
    completed,evidence=source_evidence(root)
    inputs=Path(common.read(root/'registration.json')['inputs'])
    manifests={str(inputs/p/'manifest.json'):common.sha(inputs/p/'manifest.json') for p in ['experience','test']}
    for p,n in [('experience',50),('test',100)]:
        manifest=common.read(inputs/p/'manifest.json');assert len(manifest['schedule'])==n
        for path,digest in manifest['asset_sha256'].items():assert common.sha(path)==digest
    epoch=root/EPOCH;epoch.mkdir(exist_ok=False)
    sources=[SELF,Path(shared.__file__),Path(resident.__file__),Path(driver.__file__),
        CODE/'graspcl_unseen_shared_services_20261119.py',CODE/'arx_service_readiness.py',
        CODE/'graspcl_unseen_simulator_20261119.py',driver.SIMULATOR,CODE/'arx_runtime_imports.py',
        CODE/'graspcl_unseen_bank.py']
    common.save(epoch/'registration.json',{'runtime_epoch':7,'created_at':bench.core.now(),
        'original_registration_sha256':common.sha(root/'registration.json'),
        'source_sha256':{str(p):common.sha(p) for p in sources},
        'frozen_initialization_manifest_sha256':manifests,'retained_completed_source_results':completed,
        'interrupted_source_evidence_sha256':evidence,'interrupted_case':INTERRUPTED,
        'interrupted_outcome':'unknown_infrastructure_interruption_during_approach',
        'interrupted_case_replay':False,'interrupted_case_feedback':False,
        'remaining_registered_grasp_slots':300,'unchanged_total_registered_slots':350,
        'startup_after_explicit_halted_epoch':True,'automatic_worker_restart_or_fallback':False,
        'lifecycle':'Preload simulator runtimes and one real ZeroGrasp/cuRobo pair for all remaining A/B/C slots; B/C load their own frozen bank once.',
        'exit_diagnostics':'Record simulator return code on RPC errors and shutdown; preserve the failing request without replay.',
        'algorithm_robot_camera_collision_success_rules_unchanged':True,
        'failure_cause_status':'Unknown native process exit. User cgroup OOM and OOM-kill counters are both zero; no return code was preserved by epoch06.'})
    for p in sources:shutil.copy2(p,epoch/p.name)
    common.save(epoch/'progress.json',{'phase':'prepared','completed_source_cases':49,
        'source_infrastructure_interruptions':1,'remaining_test_slots':300})


def recover_interruption(root):
    epoch=root/EPOCH;reg=common.read(epoch/'registration.json')
    arm=root/'collection/fixed_zerograsp';trial=arm/'trials'/INTERRUPTED
    for path,digest in reg['interrupted_source_evidence_sha256'].items():assert common.sha(path)==digest
    for files in reg['retained_completed_source_results'].values():
        for path,digest in files.items():assert common.sha(path)==digest
    claim_path=arm/'claims'/(INTERRUPTED+'.json');claim=common.read(claim_path)
    common.save(epoch/'interrupted_claim_before_recovery.json',claim)
    rows=[json.loads(line) for line in (trial/'execution.jsonl').open()]
    assert rows and rows[-1]['stage']=='approach'
    assert not {'close','lift','hold'} & {row['stage'] for row in rows}
    assert not (trial/'closed_state.json').exists() and not (trial/'final_state.json').exists()
    plan=common.read(trial/'plan_result.json');assert plan['success']
    assert common.read(trial/'approach_plan_result.json')['success']
    result={**common.read(arm/'entries'/(INTERRUPTED+'.json')),
        'camera':'right_wrist','success':False,'success_status':'not_verified',
        'grasp_outcome':'unknown','planning_success':True,'physical_attempt':True,
        'grasp_closure_executed':False,'observation_executed':True,
        'reason':'infrastructure_interrupted_during_approach',
        'stages':common.read(trial/'stages.json'),'execution_trace_complete':False,
        'clean_feedback_eligible':False,'infrastructure_interruption':True,
        'recovery_record':str(epoch/'source_terminal_recovery.json')}
    common.save(trial/'result.json',result)
    _,baseline=bench.core.batch.original_modules();bench.lift.install_verifier(baseline)
    baseline.verify_trial(trial)
    evaluation=common.read(trial/'independent_evaluation.json')
    assert evaluation['passed'] and not evaluation['independently_successful']
    evaluation.update(grasp_outcome='unknown',execution_trace_complete=False,
        clean_feedback_eligible=False,scope='Partial executed-trace readback confirms no verified success. Infrastructure interruption is neither a clean grasp failure nor an IK rejection.')
    common.save(trial/'independent_evaluation.json',evaluation)
    claim.update(state='complete',finished_at=bench.core.now(),result_sha256=common.sha(trial/'result.json'),
        completion_kind='terminal_infrastructure_interruption',physical_execution_complete=False,
        recovery_registration=str(epoch/'registration.json'),replayed=False)
    common.save(claim_path,claim)
    summary=bench.base.existing.summarize(arm,'complete',{})
    assert summary['completed']==50 and summary['successes']==39
    summary.update(success_scope='Source collection only: 49 completed outcomes and one interrupted unknown outcome.',
        completed_uninterrupted_cases=49,infrastructure_interruptions=1,unknown_grasp_outcomes=1,
        verified_successes=39,verified_success_fraction_of_registered=39/50,
        success_fraction_of_uninterrupted_cases=39/49)
    common.save(arm/'summary.json',summary)
    progress=common.read(arm/'progress.json')
    progress.update(completed_uninterrupted_cases=49,infrastructure_interruptions=1,unknown_grasp_outcomes=1)
    common.save(arm/'progress.json',progress);common.save(root/'collection/progress.json',progress)
    recovery={'at':bench.core.now(),'passed':True,'registered_source_slots':50,
        'terminal_source_slots':50,'completed_uninterrupted_cases':49,'verified_successes':39,
        'unknown_infrastructure_outcomes':1,'clean_feedback_for_interrupted_case':False,
        'interrupted_id':INTERRUPTED,'replayed_source_cases':0,'original_evidence_unchanged':True,
        'partial_trace_rows':len(rows),'last_recorded_stage':rows[-1]['stage'],
        'retained_49_result_claim_evaluation_hashes_unchanged':True}
    for path,digest in reg['interrupted_source_evidence_sha256'].items():
        if path!=str(claim_path):assert common.sha(path)==digest
    for files in reg['retained_completed_source_results'].values():
        for path,digest in files.items():assert common.sha(path)==digest
    common.save(epoch/'source_terminal_recovery.json',recovery)
    common.save(root/'collection/final_verification.json',recovery)


class Simulators(resident.Simulators):
    def start(self):
        for i,gpu in enumerate(bench.GPUS):
            key=hashlib.sha256((str(self.root)+EPOCH+str(i)).encode()).hexdigest()[:14]
            sock=CODE/'tmp'/('sim7_'+key+'.sock');assert not sock.exists()
            directory=self.root/EPOCH/'simulation'/str(i)
            stream=(self.root/EPOCH/('simulator_'+str(i)+'.log')).open('x');self.streams.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            self.children[i]=subprocess.Popen(['bash',str(CODE/'run.sh'),str(driver.SIMULATOR),
                '--root',str(directory),'--socket',str(sock),'--owner-pid',str(os.getpid())],
                env=env,stdout=stream,stderr=subprocess.STDOUT)
            self.sockets[i]=sock
        health={str(i):wait_ready(self.sockets[i],self.root/EPOCH/'simulation'/str(i)/'ready.json',
            child,'maniskill_simulation',timeout=1800) for i,child in self.children.items()}
        assert all(h['runtime_import_count']==1 for h in health.values())
        common.save(self.root/EPOCH/'simulation_all_ready.json',{'ready_at':bench.core.now(),
            'workers':health,'all_libraries_loaded_before_first_test_grasp':True,
            'claimed_test_grasp_slots':0,'retained_source_slots':50})

    def request(self,index,message,timeout=1800):
        try:return super().request(index,message,timeout)
        except BaseException:
            child=self.children[index]
            try:code=child.wait(timeout=1)
            except subprocess.TimeoutExpired:code=None
            common.save(self.root/EPOCH/('simulator_'+str(index)+'_rpc_failure.json'),{
                'at':bench.core.now(),'pid':child.pid,'returncode':code,'alive':child.poll() is None,
                'request':message,'error':traceback.format_exc(),'restart_or_replay':False})
            raise

    def stop(self):
        before={str(i):c.poll() for i,c in self.children.items()}
        super().stop()
        path=self.root/EPOCH/'simulation_stopped.json';receipt=common.read(path)
        for i,c in self.children.items():
            receipt['workers'][str(i)].update(returncode=c.poll(),returncode_before_shutdown=before[str(i)])
        common.save(path,receipt)


def prepare_group(root,name,inputs,bank=None):
    group,arm=ORIGINAL_PREPARE(root,name,inputs,bank)
    receipt=common.read(root/EPOCH/'registration.json')
    for directory in [group,arm]:
        path=directory/'registration.json';reg=common.read(path)
        reg['source_hashes'].update(receipt['source_sha256'])
        reg['source_hashes'][str(root/EPOCH/'registration.json')]=common.sha(root/EPOCH/'registration.json')
        reg.update(model_gpu_by_component={'zerograsp':shared.NEURAL_GPU,'curobo':shared.PLANNER_GPU,'graspcl':'CPU'},
            shared_model_lifecycle='One warmed pair for the remaining three test arms; no per-case model reload.',
            simulation_lifecycle='Two warmed scene processes; original per-case environment and reset.',
            component_startup_wait_seconds=1800,source_infrastructure_interruptions=1,
            source_interruption_excluded_from_feedback=True)
        common.save(path,reg)
    guard(common.read(group/'registration.json'))
    return group,arm


def run(root):
    receipt=common.read(root/EPOCH/'registration.json')
    for path,digest in receipt['source_sha256'].items():assert common.sha(path)==digest
    for path,digest in receipt['frozen_initialization_manifest_sha256'].items():assert common.sha(path)==digest
    assert common.sha(root/'registration.json')==receipt['original_registration_sha256']
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (root/EPOCH/'started.json').open('x') as stream:stream.write(bench.core.now()+'\n')
        resident.EPOCH=EPOCH;resident.aggregate=aggregate
        bench.aggregate=aggregate;bench.Campaign=driver.Campaign
        bench.Backends=shared.Backends;bench.GPU0=shared.PLANNER_GPU
        shared.MANAGER=shared.Manager(root,EPOCH)
        resident.POOL=Simulators(root)
        success=False
        try:
            recover_interruption(root)
            aggregate(root,'building_frozen_banks')
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
            env['PYTHONPATH']=str(common.UPSTREAM)+':'+str(CODE)
            with (root/'bank_build.log').open('x') as stream:
                subprocess.run([str(canonical_path(CODE.parent/'tmp/arx-graspcl-env/bin/python')),str(CODE/'graspcl_unseen_bank.py'),
                    '--output',str(root)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            audit=common.read(root/'bank_provenance_audit.json');assert audit['passed']
            trained=common.read(root/'banks/experience50/bank_complete.json')
            assert INTERRUPTED not in trained['source_ids']
            assert common.read(root/'banks/no_experience/bank_complete.json')['online_points']==0
            aggregate(root,'preloading_simulation');resident.POOL.start()
            inputs=Path(common.read(root/'registration.json')['inputs'])
            for name,bank in [(bench.PHASES[1],None),(bench.PHASES[2],root/'banks/no_experience'),(bench.PHASES[3],root/'banks/experience50')]:
                group,arm=prepare_group(root,name,inputs/'test',bank)
                aggregate(root,'testing_'+name);bench.run_group(group,arm)
            bench.analyze(root);success=True
        except BaseException:
            common.save(root/EPOCH/'halt.json',{'at':bench.core.now(),'error':traceback.format_exc(),'replay_allowed':False})
            aggregate(root,'halted');raise
        finally:
            resident.POOL.stop();shared.MANAGER.stop()
            stopped=common.read(root/EPOCH/'model_services_stopped.json')
            for name in bench.PHASES[1:]:
                path=root/name/'suite_timing.json'
                if path.exists():
                    value=common.read(path);value.update(model_pair_shared_until_suite_shutdown=True,
                        all_owned_services_exited=stopped['all_owned_model_services_exited'])
                    common.save(path,value)
            if success:
                common.save(root/'final_verification.json',{'passed':True,'runtime_epoch':7,
                    'registered_slots':350,'terminal_slots':350,'uninterrupted_results':349,
                    'source_infrastructure_interruptions':1,'test_cases_completed':300,
                    'all_services_exited':stopped['all_owned_model_services_exited'],
                    'all_simulators_exited':common.read(root/EPOCH/'simulation_stopped.json')['all_owned_simulators_exited'],
                    'test_feedback_updates':0,'physical_grasps_replayed':0,
                    'epoch07_zerograsp_model_loads':stopped['zero_model_loads'],
                    'epoch07_curobo_pool_loads':stopped['curobo_pool_loads'],
                    'prior_epoch_model_loads_recorded_separately':True})
            aggregate(root,'complete' if success else 'halted')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)(args.output.resolve())
