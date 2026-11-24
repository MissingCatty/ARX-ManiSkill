"""Epoch 3: preload persistent simulators and reuse the frozen 350-slot protocol."""
from arx_project.layout import canonical_path
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import fcntl
import hashlib
import inspect
import os
import shutil
import subprocess
import threading
import textwrap
import time
import traceback
import types

import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_pipeline as common
import graspcl_unseen_bench as bench
import graspcl_unseen_resume_20261119 as previous
from benchmark_lifecycle import require_ready
from graspcl_execute import guard
from resident_backend import rpc

SELF=Path(__file__).resolve()
CODE=SELF.parent
EPOCH='continuation_epoch03'
SIMULATOR=CODE/'graspcl_unseen_simulator_20261119.py'
ORIGINAL_AGGREGATE=bench.aggregate
POOL=None


def aggregate(root,phase=None):
    ORIGINAL_AGGREGATE(root,phase)
    value=common.read(root/'progress.json')
    inputs=Path(common.read(root/'registration.json')['inputs'])
    value['scene_initialization']={p:common.read(inputs/p/'admission_progress.json') for p in ['experience','test']}
    value['runtime_epoch']=3
    value['historical_halts_retained']=['epoch01','epoch02']
    if POOL is not None:value['simulation_workers']=POOL.status()
    common.save(root/'progress.json',value)


def adopt(directory,entry,epoch):
    out=directory/'admission'/entry['id']
    failures=[]
    for variant in range(12):
        receipts=sorted(out.glob(f'variant_{variant:02d}*/admission.json'))
        if not receipts:return False
        rows=[common.read(p) for p in receipts]
        assert len({r['passed'] for r in rows})==1,'Conflicting completed initialization receipts'
        receipt=rows[0]
        if receipt['passed']:
            previous.commit_admission(directory,entry,receipts[0].parent,receipt,failures)
            common.save(directory/'admission_claims'/(entry['id']+'.json'),{
                'state':'complete','passed':True,'runtime_epoch':3,'adopted_existing_complete_receipt':True,'grasp_attempt':False})
            common.save(epoch/'adopted_receipts'/(entry['id']+'.json'),{
                'source':str(receipts[0]),'source_sha256':common.sha(receipts[0]),
                'top_level_receipt_sha256':common.sha(out/'admission.json'),'physical_grasps':0})
            return True
        failures.extend(receipt.get('rejected_variants',[]))
    return False


def prepare(root):
    assert not list(root.glob('*/fixed_*/claims/*.json'))
    # Reuse the audited 118-scene carry-over and preserve the epoch 2 claims.
    previous.EPOCH=EPOCH
    previous.prepare(root)
    epoch=root/EPOCH
    receipt=common.read(epoch/'registration.json')
    receipt.update(runtime_epoch=3,
        change='Persistent ManiSkill scene processes preload SciPy, PyTorch and SAPIEN once; scene initialization and the unchanged episode executor use local RPC.',
        simulator_startup_wait_seconds=1800,component_startup_wait_seconds=1800,per_initialization_request_seconds=600,
        per_grasp_inference_planning_execution_budgets_unchanged=True,
        retained_receipts_do_not_require_reexecution=True,
        source_sha256={str(p):common.sha(p) for p in [SELF,SIMULATOR,Path(previous.__file__),
            CODE/'arx_runtime_imports.py',CODE/'source/mani_skill/envs/tasks/mobile_manipulation/__init__.py']},
        arx_import_scope='Omit only unused RoboCasa automatic task registration; all actual ARX/YCB/physics code remains original.')
    inputs=Path(common.read(root/'registration.json')['inputs'])
    adopted=[]
    for partition in ['experience','test']:
        directory=inputs/partition;plan=common.read(directory/'preparation.json')
        for entry in plan['schedule']:
            if not (directory/'admission'/entry['id']/'admission.json').exists():
                if adopt(directory,entry,epoch):adopted.append(entry['id'])
    receipt['additional_completed_initializations_adopted']=adopted
    common.save(epoch/'registration.json',receipt)
    for path in [SELF,SIMULATOR]:shutil.copy2(path,epoch/path.name)
    common.save(epoch/'progress.json',{'phase':'prepared','retained_admissions':118+len(adopted),'physical_grasps_claimed':0})


class Simulators:
    def __init__(self,root):
        self.root=root;self.children={};self.sockets={};self.streams=[]

    def status(self):
        states={}
        for i,child in self.children.items():
            file=self.root/EPOCH/'simulation'/str(i)/'stage.json'
            states['scene_'+str(i)]={'pid':child.pid,'alive':child.poll() is None,
                **(common.read(file) if file.exists() else {'stage':'starting'})}
        return states

    def start(self):
        for i,gpu in enumerate(bench.GPUS):
            key=hashlib.sha256((str(self.root)+str(i)).encode()).hexdigest()[:12]
            sock=CODE/'tmp'/('sim3_'+key+'.sock')
            assert not sock.exists()
            directory=self.root/EPOCH/'simulation'/str(i)
            stream=(self.root/EPOCH/('simulator_'+str(i)+'.log')).open('x');self.streams.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            command=['bash',str(CODE/'run.sh'),str(SIMULATOR),'--root',str(directory),'--socket',str(sock),
                '--owner-pid',str(os.getpid())]
            self.children[i]=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            self.sockets[i]=sock
        deadline=time.monotonic()+1800
        while True:
            if any(c.poll() is not None for c in self.children.values()):raise RuntimeError('Simulator preload failed')
            if all(s.exists() for s in self.sockets.values()):break
            if time.monotonic()>deadline:raise RuntimeError('Simulator preload timed out; no grasp was claimed')
            aggregate(self.root,'preloading_simulation')
            time.sleep(5)
        health={str(i):rpc(s,{'command':'health'},timeout=30) for i,s in self.sockets.items()}
        assert all(h['status']=='ready' and h['runtime_import_count']==1 for h in health.values())
        common.save(self.root/EPOCH/'simulation_all_ready.json',{'ready_at':bench.core.now(),'workers':health,
            'grasp_slots_claimed':0,'all_libraries_loaded_before_first_grasp':True})

    def request(self,index,message,timeout=1800):
        child=self.children[index]
        if child.poll() is not None:raise RuntimeError('Simulator exited; no cold fallback or replay')
        result=rpc(self.sockets[index],message,timeout=timeout)
        if result['status']!='ok':raise RuntimeError('Resident simulation failed: '+str(result))
        assert result['runtime_import_count']==1
        return result

    def stop(self):
        receipts={}
        for i,child in self.children.items():
            forced=False
            if child.poll() is None:
                try:
                    rpc(self.sockets[i],{'command':'shutdown'},timeout=10)
                    child.wait(timeout=20)
                except Exception:
                    forced=True;child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            receipts[str(i)]={'pid':child.pid,'exited':child.poll() is not None,'forced_shutdown':forced}
        for stream in self.streams:stream.close()
        common.save(self.root/EPOCH/'simulation_stopped.json',{'workers':receipts,'all_owned_simulators_exited':all(v['exited'] for v in receipts.values())})


def admit_one(directory,entry,env,epoch):
    out=directory/'admission'/entry['id'];out.mkdir(exist_ok=True)
    binding=out/'entry.json'
    if not binding.exists():
        prior=[]
        for p in (directory/'admission').glob('*/admission.json'):
            row=common.read(p)
            if row['passed'] and row['entry']['model_id']==entry['model_id']:prior.append(row['entry']['scene'])
        common.save(binding,{**entry,'avoid_scenes':prior})
    failures=[]
    index=bench.GPUS.index(env['CUDA_VISIBLE_DEVICES'])
    for variant in range(12):
        receipts=sorted(out.glob(f'variant_{variant:02d}*/admission.json'))
        if receipts:
            rows=[common.read(p) for p in receipts]
            assert len({r['passed'] for r in rows})==1
            receipt=rows[0];destination=receipts[0].parent
        else:
            destination=out/f'variant_{variant:02d}_{EPOCH}'
            assert not destination.exists(),'Unknown initialization in this epoch must not replay silently'
            message={'command':'admit','inputs':str(directory),'entry':str(binding),'output':str(destination),
                'variant':variant,'log':str(out/f'variant_{variant:02d}_{EPOCH}.log')}
            started=time.monotonic()
            response=POOL.request(index,message,timeout=600)
            common.save(epoch/'initialization_requests'/(entry['id']+f'_v{variant:02d}.json'),{
                'request':message,'response':response,'wall_seconds':time.monotonic()-started,'physical_grasp':False})
            receipt=common.read(destination/'admission.json')
        for error in receipt.get('rejected_variants',[]):
            if any(w in error['error'].lower() for w in ['out of memory','vk_error','resource descriptor','supported physical device']):
                raise RuntimeError('Render infrastructure failure: '+str(error))
        if receipt['passed']:
            previous.commit_admission(directory,entry,destination,receipt,failures)
            return True
        failures.extend(receipt.get('rejected_variants',[]))
    common.save(out/'admission.json',{'passed':False,'entry':{**entry,'admission_failed':True},'rejected_variants':failures})
    return False


class Campaign(bench.Campaign):
    def start(self):
        assert (self.suite.parent/EPOCH/'simulation_all_ready.json').exists()
        # This import is shared by all episodes. The independent trace verifier
        # stays in the controller, outside the execution process.
        _,baseline=bench.core.batch.original_modules();bench.lift.install_verifier(baseline)
        super().start()

    def worker(self,index,gpu):
        worker='scene_'+str(index)
        _,baseline=bench.core.batch.original_modules();bench.lift.install_verifier(baseline)
        try:
            while not self.failed.is_set():
                if (self.suite.parent/'STOP').exists():raise RuntimeError('Explicit STOP')
                if not self.resources(worker,gpu):break
                guard(self.reg)
                entry,claim=self.next_entry(worker,gpu)
                if entry is None:break
                trial=self.out/'trials'/entry['id']
                self.progress(worker,phase='executing',current=entry['id'],gpu=gpu,child_pid=POOL.children[index].pid)
                reason='independent_scene_admission_failed' if entry['admission_failed'] else None
                if self.reg.get('shared_input_role')=='consumer' and not (Path(self.reg['shared_input_cache'])/entry['id']/'manifest.json').exists():
                    prior=common.read(self.suite.parent/'A_zerograsp/fixed_zerograsp/trials'/entry['id']/'result.json')
                    assert not prior['planning_success'] and not prior['physical_attempt']
                    reason='common_observation_failed_no_grasp_input'
                if reason:
                    trial.mkdir(exist_ok=False)
                    common.save(trial/'result.json',{**entry,'success':False,'planning_success':False,'physical_attempt':False,'reason':reason})
                    bench.base.placeholder(trial,reason)
                else:
                    response=POOL.request(index,{'command':'execute','output':str(trial),
                        'entry':str(self.out/'entries'/(entry['id']+'.json')),
                        'log':str(self.out/'entries'/(entry['id']+'.log'))})
                    common.save(trial/'resident_simulation_receipt.json',response)
                baseline.verify_trial(trial)
                result=common.read(trial/'result.json');guard(self.reg)
                claim.update(state='complete',finished_at=bench.core.now(),result_sha256=common.sha(trial/'result.json'))
                common.save(self.out/'claims'/(entry['id']+'.json'),claim)
                self.progress(worker,phase='between_scenes',current=None)
                print('RESULT',self.suite.name,entry['id'],result['success'],result['reason'],flush=True)
        except BaseException:
            self.failed.set();error={'worker':worker,'error':traceback.format_exc(),'time':bench.core.now()}
            self.errors.append(error);common.save(self.out/'workers'/(worker+'_halt.json'),error)
            self.progress(worker,phase='halted',error=error['error'])


def run(root):
    global POOL
    epoch=root/EPOCH;receipt=common.read(epoch/'registration.json')
    for path,digest in receipt['source_sha256'].items():assert common.sha(path)==digest
    assert common.sha(root/'registration.json')==receipt['original_registration_sha256']
    assert common.sha(receipt['validation'])==receipt['validation_sha256']
    assert common.read(receipt['validation'])['passed']
    assert not list(root.glob('*/fixed_*/claims/*.json'))
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (epoch/'started.json').open('x') as stream:stream.write(bench.core.now()+'\n')
        POOL=Simulators(root)
        previous.EPOCH=EPOCH;previous.admit_one=admit_one;previous.actual_progress=aggregate
        bench.Campaign=Campaign;bench.aggregate=aggregate
        # First imports under the observed disk contention belong to startup,
        # never to a claimed episode. Keep all per-request budgets unchanged.
        original_start=bench.Backends.start
        source=textwrap.dedent(inspect.getsource(original_start))
        assert source.count('600')==3
        source=source.replace('600','1800')
        namespace=dict(original_start.__globals__)
        exec(compile(source,str(SELF)+':startup_wait_only','exec'),namespace)
        revised=namespace['start']
        bench.Backends.start=types.FunctionType(revised.__code__,original_start.__globals__,'start')
        original_prepare=bench.prepare_group
        def prepare_group(*args,**kwargs):
            group,arm=original_prepare(*args,**kwargs)
            for directory in [group,arm]:
                path=directory/'registration.json';reg=common.read(path)
                reg['source_hashes'].update(receipt['source_sha256'])
                reg['source_hashes'][str(epoch/'registration.json')]=common.sha(epoch/'registration.json')
                reg['simulation_lifecycle']='Two preloaded scene processes, reused across initialization and all 350 slots; fresh env/reset each episode; no runtime reload or replay.'
                reg['component_startup_wait_seconds']=1800
                common.save(path,reg)
            return group,arm
        inputs=Path(common.read(root/'registration.json')['inputs'])
        success=False
        try:
            aggregate(root,'preloading_simulation')
            POOL.start()
            for partition in ['experience','test']:previous.resume_admission(root,partition)
            # Fix inherited scheduler provenance labels; every original receipt
            # is pinned in the epoch registration and is left byte-for-byte intact.
            for path in (inputs/'test/admission_claims').glob('*.json'):
                claim=common.read(path)
                if claim.get('runtime_epoch')==2:
                    claim['runtime_epoch']=3;common.save(path,claim)
            for partition in ['experience','test']:
                path=inputs/partition/'admission_progress.json';value=common.read(path)
                if value.get('runtime_epoch')==2:value['runtime_epoch']=3;common.save(path,value)
                manifest=common.read(inputs/partition/'manifest.json')
                for path,digest in manifest['asset_sha256'].items():
                    if common.sha(path)!=digest:raise RuntimeError('Frozen asset changed: '+path)
            aggregate(root,'collecting_experience')
            group,arm=prepare_group(root,'collection',inputs/'experience');bench.run_group(group,arm)
            aggregate(root,'building_frozen_banks')
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
            env['PYTHONPATH']=str(common.UPSTREAM)+':'+str(CODE)
            with (root/'bank_build.log').open('x') as stream:
                subprocess.run([str(canonical_path(CODE.parent/'tmp/arx-graspcl-env/bin/python')),str(CODE/'graspcl_unseen_bank.py'),
                    '--output',str(root)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            assert common.read(root/'bank_provenance_audit.json')['passed']
            for name,bank in [(bench.PHASES[1],None),(bench.PHASES[2],root/'banks/no_experience'),(bench.PHASES[3],root/'banks/experience50')]:
                group,arm=prepare_group(root,name,inputs/'test',bank)
                aggregate(root,'testing_'+name);bench.run_group(group,arm)
            bench.analyze(root);success=True
        except BaseException:
            common.save(epoch/'halt.json',{'time':bench.core.now(),'error':traceback.format_exc(),'replay_allowed':False})
            aggregate(root,'halted');raise
        finally:
            POOL.stop()
            if success:
                common.save(root/'final_verification.json',{'passed':True,'registered_and_completed':350,
                    'all_services_exited':all(common.read(root/g/'suite_timing.json')['all_owned_services_exited'] for g in bench.PHASES),
                    'all_simulators_exited':common.read(epoch/'simulation_stopped.json')['all_owned_simulators_exited'],
                    'test_feedback_updates':0,'physical_grasps_replayed':0,'runtime_epoch':3})
                aggregate(root,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve())
