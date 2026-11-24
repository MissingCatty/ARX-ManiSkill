"""Continue outcome-blind scene admission after epoch 1 initialization timeouts.

Existing receipts and all frozen scene IDs/models/seeds are retained. Only
initialization subprocess waiting changes from 240 to 600 seconds. No claimed
grasp may be resumed or replayed by this entry.
"""
from arx_project.layout import canonical_path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse
import fcntl
import os
import shutil
import subprocess
import threading
import time
import traceback

import graspcl_pipeline as common
import graspcl_unseen_bench as bench
from graspcl_execute import guard

CODE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
EPOCH = 'continuation_epoch02'
INIT_TIMEOUT = 600


def actual_progress(root, phase=None):
    bench.aggregate(root, phase)
    state = common.read(root/'progress.json')
    inputs = Path(common.read(root/'registration.json')['inputs'])
    state['scene_initialization'] = {name: common.read(inputs/name/'admission_progress.json')
        for name in ['experience','test']}
    state['runtime_epoch'] = 2
    state['epoch01_halt_preserved_as_history'] = True
    common.save(root/'progress.json', state)


def prepare(root):
    reg = common.read(root/'registration.json')
    guard(reg)
    assert common.read(root/'halt.json')['replay_allowed'] is False
    assert not list(root.glob('*/fixed_*/claims/*.json')), 'This continuation cannot replay any claimed grasp'
    assert not any((root/name).exists() for name in bench.PHASES)
    inputs = Path(reg['inputs'])
    epoch = root/EPOCH
    epoch.mkdir(exist_ok=False)
    proofs = {}
    for partition in ['experience','test']:
        directory = inputs/partition
        plan = common.read(directory/'preparation.json')
        schedule = {e['id']:e for e in plan['schedule']}
        terminal, incomplete = {}, []
        archive = epoch/'original_admission_claims'/partition
        shutil.copytree(directory/'admission_claims', archive)
        for path in (directory/'admission_claims').glob('*.json'):
            receipt = directory/'admission'/path.stem/'admission.json'
            claim = common.read(path)
            assert path.stem in schedule
            if claim['state']=='complete':
                result = common.read(receipt)
                assert all(result['entry'][k]==schedule[path.stem][k] for k in ['id','model_id','layout','seed'])
                terminal[str(receipt)] = common.sha(receipt)
            else:
                assert claim['state']=='initialization_only'
                assert not receipt.exists()
                incomplete.append(path.stem)
        proofs[partition] = {'completed_admission_sha256':terminal,'interrupted_initialization_ids':incomplete,
            'frozen_preparation_sha256':common.sha(directory/'preparation.json')}
    assert len(proofs['experience']['completed_admission_sha256'])==50
    assert len(proofs['test']['completed_admission_sha256'])==68
    receipt = {'created_at':bench.core.now(),'original_registration_sha256':common.sha(root/'registration.json'),
        'original_halt_sha256':common.sha(root/'halt.json'),'initialization_only':True,
        'claimed_grasp_slots':0,'original_350_grasp_slots_unchanged':True,
        'scene_models_categories_seeds_visibility_rules_unchanged':True,
        'change':'Scene initialization subprocess timeout 240 -> 600 seconds; component preload already uses 600 seconds.',
        'per_grasp_inference_planning_execution_budgets_unchanged':True,
        'retained_scene_admission':proofs,'runtime_scene_gpus':bench.GPUS,'runtime_backend_gpu':bench.GPU0,
        'source_sha256':{str(SELF):common.sha(SELF)},'retry_policy':'Only unfinished initialization, same seed and variant, separate epoch directory. No completed admission or grasp replay.'}
    common.save(epoch/'registration.json', receipt)
    shutil.copy2(SELF, epoch/SELF.name)
    common.save(epoch/'progress.json',{'phase':'prepared','retained_admissions':118,'grasp_slots_claimed':0})


def commit_admission(directory, entry, destination, receipt, failures):
    out = directory/'admission'/entry['id']
    for name in ['wrist.png','overview.png','observation_probe_state.json']:
        shutil.copy2(destination/name,out/name)
    receipt = deepcopy(receipt)
    receipt['prior_variants'] = failures
    initial = common.read(directory/'common_initial_conditions.json')
    common.save(directory/'declared_resets'/entry['id']/'declared_reset.json',{
        'receipt_kind':'new_declared_reset_not_historical_execution','spawn':receipt['entry']['scene'],
        'base_pose':initial['base_pose'],'initial_q':initial['initial_q'],
        'scene_admission_sha256':common.sha(destination/'admission.json')})
    common.save(out/'admission.json',receipt)


def admit_one(directory, entry, env, epoch):
    out = directory/'admission'/entry['id']
    out.mkdir(exist_ok=True)
    if (out/'admission.json').exists():
        raise RuntimeError('Completed scene must be carried without replay')
    binding = out/'entry.json'
    if not binding.exists():
        prior=[]
        for p in (directory/'admission').glob('*/admission.json'):
            row=common.read(p)
            if row['passed'] and row['entry']['model_id']==entry['model_id']:
                prior.append(row['entry']['scene'])
        common.save(binding,{**entry,'avoid_scenes':prior})
    else:
        assert all(common.read(binding)[k]==entry[k] for k in ['id','model_id','layout','seed'])
    failures=[]
    for variant in range(12):
        previous=out/f'variant_{variant:02d}'
        if (previous/'admission.json').exists():
            # An interrupted controller can adopt a completed render receipt.
            destination=previous
            receipt=common.read(previous/'admission.json')
        else:
            destination=out/f'variant_{variant:02d}_{EPOCH}'
            assert not destination.exists(), 'Do not silently restart this continuation'
            existed = previous.exists() or (out/f'variant_{variant:02d}.log').exists()
            command=['bash',str(CODE/'run.sh'),str(CODE/'graspcl_unseen_inputs.py'),'admit_variant',
                '--inputs',str(directory),'--entry',str(binding),'--output',str(destination),'--variant',str(variant)]
            common.save(epoch/'initialization_requests'/(entry['id']+f'_v{variant:02d}.json'),{
                'scene':entry['id'],'variant':variant,'seed':entry['seed'],'old_unfinished_initialization':existed,
                'grasp_attempt':False,'destination':str(destination),'timeout_seconds':INIT_TIMEOUT,
                'started_at':bench.core.now()})
            with (out/f'variant_{variant:02d}_{EPOCH}.log').open('x') as stream:
                child=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=INIT_TIMEOUT)
            if child.returncode or not (destination/'admission.json').exists():
                raise RuntimeError('Initialization infrastructure failed: '+entry['id'])
            receipt=common.read(destination/'admission.json')
        for error in receipt.get('rejected_variants',[]):
            if any(word in error['error'].lower() for word in ['out of memory','vk_error','resource descriptor','supported physical device']):
                raise RuntimeError('Render infrastructure failure: '+str(error))
        if receipt['passed']:
            commit_admission(directory,entry,destination,receipt,failures)
            return True
        failures.extend(receipt.get('rejected_variants',[]))
    common.save(out/'admission.json',{'passed':False,'entry':{**entry,'admission_failed':True},'rejected_variants':failures})
    return False


def resume_admission(root, partition):
    inputs=Path(common.read(root/'registration.json')['inputs'])
    directory=inputs/partition; epoch=root/EPOCH
    plan=common.read(directory/'preparation.json')
    pinned=common.read(epoch/'registration.json')['retained_scene_admission'][partition]
    assert common.sha(directory/'preparation.json')==pinned['frozen_preparation_sha256']
    for path,digest in pinned['completed_admission_sha256'].items():
        assert common.sha(path)==digest, 'Completed scene changed'
    if (directory/'manifest.json').exists():
        assert common.read(directory/'admission_progress.json')['phase']=='complete'
        return
    mutex=threading.Lock(); failed=threading.Event(); active=set()
    claimed={p.parent.name for p in (directory/'admission').glob('*/admission.json')}
    def progress():
        rows=[common.read(p) for p in (directory/'admission').glob('*/admission.json')]
        common.save(directory/'admission_progress.json',{'phase':'halted' if failed.is_set() else 'generating',
            'completed':len(rows),'passed':sum(r['passed'] for r in rows),'registered':len(plan['schedule']),
            'active':sorted(active),'runtime_epoch':2,'updated_at':bench.core.now()})
        actual_progress(root,'preparing_scenes')
    def worker(index,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        try:
            while not failed.is_set():
                with mutex:
                    options=[e for e in plan['schedule'] if e['id'] not in claimed and e['model_id'] not in active]
                    entry=options[0] if options else None
                    if entry:
                        claimed.add(entry['id']);active.add(entry['model_id'])
                        path=directory/'admission_claims'/(entry['id']+'.json')
                        common.save(path,{'state':'initialization_only','worker':index,'runtime_epoch':2,'grasp_attempt':False})
                        progress()
                    elif len(claimed)==len(plan['schedule']): break
                if entry is None:
                    failed.wait(.5); continue
                while not failed.is_set():
                    free=bench.core.batch.subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
                    memory={line.split(',')[0].strip():int(line.split(',')[1]) for line in free.splitlines()}
                    if memory[gpu]>=3500: break
                    failed.wait(15)
                if failed.is_set(): break
                passed=admit_one(directory,entry,env,epoch)
                with mutex:
                    active.remove(entry['model_id'])
                    common.save(directory/'admission_claims'/(entry['id']+'.json'),{
                        'state':'complete','worker':index,'passed':passed,'runtime_epoch':2,'grasp_attempt':False})
                    progress()
                print('ADMISSION RESUMED',entry['id'],passed,flush=True)
        except BaseException:
            failed.set()
            common.save(epoch/('admission_worker_'+str(index)+'_halt.json'),{'error':traceback.format_exc()})
            with mutex:progress()
    with mutex:progress()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i,gpu) for i,gpu in enumerate(bench.GPUS)]
        for future in futures:future.result()
    if failed.is_set(): raise RuntimeError('Admission continuation infrastructure halted')
    rows=[common.read(directory/'admission'/e['id']/'admission.json') for e in plan['schedule']]
    allowed=set(plan['test_object_ids'])
    for row in rows:
        if row['passed']:
            assert {a['model_id'] for a in row['entry']['scene']['actors']}<=allowed
            assert row['entry']['admission']['occlusion_fraction']<=.40
    for path,digest in pinned['completed_admission_sha256'].items(): assert common.sha(path)==digest
    manifest=deepcopy(plan)
    manifest.update(frozen_at=bench.core.now(),admitted=sum(r['passed'] for r in rows),admission_failed=sum(not r['passed'] for r in rows),
        admission_sha256={str(p):common.sha(p) for p in (directory/'admission').glob('*/admission.json')},
        declared_reset_sha256={str(p):common.sha(p) for p in (directory/'declared_resets').glob('*/declared_reset.json')},
        input_sha256={str(directory/n):common.sha(directory/n) for n in ['preparation.json','common_initial_conditions.json','storage_policy.json']})
    common.save(directory/'manifest.json',manifest)
    common.save(directory/'admission_progress.json',{'phase':'complete','completed':len(rows),'registered':len(rows),'passed':manifest['admitted'],'runtime_epoch':2})
    actual_progress(root,'preparing_scenes')


def run(root):
    epoch=root/EPOCH
    epoch_reg=common.read(epoch/'registration.json')
    for path,digest in epoch_reg['source_sha256'].items(): assert common.sha(path)==digest
    assert common.sha(root/'registration.json')==epoch_reg['original_registration_sha256']
    reg=common.read(root/'registration.json');guard(reg)
    assert not list(root.glob('*/fixed_*/claims/*.json'))
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (epoch/'started.json').open('x') as stream:stream.write(bench.core.now()+'\n')
        original_prepare=bench.prepare_group
        def prepare_group(*args,**kwargs):
            group,arm=original_prepare(*args,**kwargs)
            for target in [group,arm]:
                path=target/'registration.json'; value=common.read(path)
                value['source_hashes'].update({str(SELF):common.sha(SELF),str(epoch/'registration.json'):common.sha(epoch/'registration.json')})
                value['admission_runtime_epoch']=2
                common.save(path,value)
            return group,arm
        inputs=Path(reg['inputs'])
        try:
            for partition in ['experience','test']:resume_admission(root,partition)
            mapping=common.read(reg['semantic_map'])
            actual_models={}
            for partition in ['experience','test']:
                manifest=common.read(inputs/partition/'manifest.json')
                actual_models[partition]=set()
                for path in manifest['admission_sha256']:
                    row=common.read(path)
                    if row['passed']:actual_models[partition].update(a['model_id'] for a in row['entry']['scene']['actors'])
                for path,digest in manifest['asset_sha256'].items():
                    if common.sha(path)!=digest:raise RuntimeError('Frozen asset changed: '+path)
            assert not actual_models['experience']&actual_models['test']
            categories={k:{mapping[m]['category'] for m in v} for k,v in actual_models.items()}
            assert not categories['experience']&categories['test']
            common.save(epoch/'completed_scene_isolation.json',{'passed':True,'actual_all_actor_model_intersection':0,
                'actual_all_actor_category_intersection':0,'completed_scenes_preserved':118,'physical_grasps_replayed':0})
            for name in ['verification.json','shared_input_verification.json','feedback_action_verification.json']:
                assert common.read(CODE/'outputs/arx_graspcl_unseen_validation_20261119_01'/name)['passed']
            actual_progress(root,'collecting_experience')
            group,arm=prepare_group(root,'collection',inputs/'experience')
            bench.run_group(group,arm)
            actual_progress(root,'building_frozen_banks')
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
            env['PYTHONPATH']=str(common.UPSTREAM)+':'+str(CODE)
            with (root/'bank_build.log').open('x') as stream:
                subprocess.run([str(canonical_path(CODE.parent/'tmp/arx-graspcl-env/bin/python')),str(CODE/'graspcl_unseen_bank.py'),
                    '--output',str(root)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            assert common.read(root/'bank_provenance_audit.json')['passed']
            for name,bank in [(bench.PHASES[1],None),(bench.PHASES[2],root/'banks/no_experience'),(bench.PHASES[3],root/'banks/experience50')]:
                group,arm=prepare_group(root,name,inputs/'test',bank)
                actual_progress(root,'testing_'+name)
                bench.run_group(group,arm)
            bench.analyze(root)
            common.save(root/'final_verification.json',{'passed':True,'registered_and_completed':350,
                'all_services_exited':all(common.read(root/g/'suite_timing.json')['all_owned_services_exited'] for g in bench.PHASES),
                'experience_test_category_model_mesh_disjoint':True,'test_feedback_updates':0,'physical_grasps_replayed':0,'runtime_epoch':2})
            actual_progress(root,'complete')
        except BaseException:
            common.save(epoch/'halt.json',{'time':bench.core.now(),'error':traceback.format_exc(),'replay_allowed':False})
            actual_progress(root,'halted')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve())
