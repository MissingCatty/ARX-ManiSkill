"""Resume the existing A/B/C slots after the explicitly recorded native crash."""
from pathlib import Path
import argparse
import fcntl
import inspect
import json
import shutil
import textwrap
import time
import traceback
import types

import graspcl_unseen_postcollection_20261119 as previous
import graspcl_unseen_shared_backends_20261119 as shared
from graspcl_execute import guard

common,bench,resident=previous.common,previous.bench,previous.resident
CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
EPOCH='continuation_epoch08'
INTERRUPTED='T0022_S2_edge_032_knife'
SCENE_GPUS=[shared.PLANNER_GPU,shared.PLANNER_GPU]
ORIGINAL_PREPARE=bench.prepare_group
ORIGINAL_AGGREGATE=bench.aggregate


def aggregate(root,phase=None):
    ORIGINAL_AGGREGATE(root,phase)
    value=common.read(root/'progress.json')
    value.update(runtime_epoch=8,source_infrastructure_interruptions=1,
        test_infrastructure_interruptions=1,source_grasps_with_completed_results=49,
        source_verified_successes=39,physical_grasps_replayed=0,
        interrupted_test_ids=[INTERRUPTED],historical_halts_retained=list(range(1,8)))
    if resident.POOL is not None:value['simulation_workers']=resident.POOL.status()
    if shared.MANAGER is not None:
        value['model_services']={name:{'pid':r['process'].pid,'alive':r['process'].poll() is None,
            'kind':r['kind']} for name,r in shared.MANAGER.all_processes.items()}
    if 'A_zerograsp' in value['groups']:
        v=value['groups']['A_zerograsp'];v.update(infrastructure_interruptions=1,
            completed_uninterrupted_cases=max(0,v['completed']-1))
    common.save(root/'progress.json',value)


def snapshot(paths):
    return {str(p):common.sha(p) for p in paths if p.is_file()}


def verify_hashes(mapping):
    for path,digest in mapping.items():
        assert common.sha(path)==digest,'Frozen evidence changed: '+path


def prepare(root):
    guard(common.read(root/'registration.json'))
    for name,key in [('model_services_stopped.json','all_owned_model_services_exited'),
                     ('simulation_stopped.json','all_owned_simulators_exited')]:
        assert common.read(root/'continuation_epoch07'/name)[key]
    assert not any((root/name).exists() for name in bench.PHASES[2:])
    a=root/'A_zerograsp/fixed_zerograsp';claims=list((a/'claims').glob('*.json'))
    assert len(claims)==22
    retained={}
    for p in list((root/'collection/fixed_zerograsp/claims').glob('*.json'))+claims:
        row=common.read(p);trial=p.parent.parent/'trials'/p.stem
        if p.stem==INTERRUPTED:
            assert row['state']=='claimed' and not (trial/'result.json').exists()
            continue
        assert row['state']=='complete' and row['result_sha256']==common.sha(trial/'result.json')
        assert common.read(trial/'independent_evaluation.json')['passed']
        retained.update(snapshot([p,trial/'result.json',trial/'independent_evaluation.json']))
    assert common.read(root/'bank_provenance_audit.json')['passed']
    banks=snapshot(list((root/'banks').rglob('*'))+[root/'bank_provenance_audit.json'])
    b=common.read(root/'banks/no_experience/bank_complete.json')
    c=common.read(root/'banks/experience50/bank_complete.json')
    assert b['online_points']==0 and c['positive_demonstrations']==39 and c['negative_feedback']==2
    t=a/'trials'/INTERRUPTED
    rows=[json.loads(line) for line in (t/'execution.jsonl').open()]
    assert rows[-1]['stage']=='pregrasp' and not {'close','lift','hold'}&{v['stage'] for v in rows}
    assert (root/'shared_test_inputs'/INTERRUPTED/'manifest.json').exists()
    assert (t/'shared_input_receipt.json').exists()
    epoch=root/EPOCH;epoch.mkdir(exist_ok=False)
    for name in ['suite_timing.json','progress.json','test_started.json','registration.json']:
        p=root/'A_zerograsp'/name
        if p.exists():shutil.copy2(p,epoch/('A_epoch07_'+name))
    shutil.copytree(a/'workers',epoch/'A_epoch07_workers')
    files=[SELF,Path(previous.__file__),Path(shared.__file__),Path(resident.__file__),
        CODE/'arx_quiet_diagnostics_20261119.py',CODE/'graspcl_unseen_simulator_quiet_20261119.py',
        CODE/'graspcl_unseen_services_quiet_20261119.py',CODE/'graspcl_unseen_simulator_pinned_20261119.py',
        CODE/'graspcl_unseen_simulator_20261119.py',CODE/'graspcl_unseen_shared_services_20261119.py',
        CODE/'arx_service_readiness.py',CODE/'arx_runtime_imports.py']
    inputs=Path(common.read(root/'registration.json')['inputs'])
    common.save(epoch/'registration.json',{'created_at':bench.core.now(),'runtime_epoch':8,
        'authorized_by':'User explicitly requested immediate resumption after the halt.',
        'source_sha256':snapshot(files),'original_registration_sha256':common.sha(root/'registration.json'),
        'A_registration_sha256':common.sha(root/'A_zerograsp/registration.json'),
        'retained_terminal_evidence_sha256':retained,'frozen_bank_sha256':banks,
        'frozen_input_manifest_sha256':snapshot([inputs/p/'manifest.json' for p in ['experience','test']]),
        'interrupted_evidence_sha256':snapshot(list(t.rglob('*'))+[a/'claims'/(INTERRUPTED+'.json')]),
        'interrupted_test_id':INTERRUPTED,'interrupted_test_outcome':'unknown_infrastructure_failure',
        'completed_source_slots':50,'completed_A_results':21,'remaining_test_executions':278,
        'total_registered_slots_unchanged':350,'claimed_slots_replayed':0,'test_feedback_updates':0,
        'diagnostic_mitigation':'Both native exits coincided with an unfinished periodic traceback. Disable periodic stack walking only; retain fatal and SIGUSR1 reports. Causality remains unconfirmed.',
        'model_gpu_by_component':{'zerograsp':shared.NEURAL_GPU,'curobo':shared.PLANNER_GPU,'graspcl':'CPU'},
        'simulator_gpus':SCENE_GPUS,'minimum_scene_free_mib_unchanged':3500,
        'old_A_resident_archive':str(epoch/'A_resident_epoch07'),
        'paired_analysis':'Unknown interrupted outcomes are excluded per comparison and explicitly counted, never treated as grasp failures. Shared-input verification still includes all 100 slots.',
        'algorithm_camera_robot_collision_success_rules_unchanged':True})
    for p in files:shutil.copy2(p,epoch/p.name)
    common.save(epoch/'progress.json',{'phase':'prepared','remaining_test_executions':278})


def terminalize_interruption(root):
    epoch=root/EPOCH;reg=common.read(epoch/'registration.json')
    verify_hashes(reg['retained_terminal_evidence_sha256']);verify_hashes(reg['interrupted_evidence_sha256'])
    a=root/'A_zerograsp/fixed_zerograsp';t=a/'trials'/INTERRUPTED
    claim_path=a/'claims'/(INTERRUPTED+'.json');claim=common.read(claim_path)
    common.save(epoch/'interrupted_claim_before_recovery.json',claim)
    result={**common.read(a/'entries'/(INTERRUPTED+'.json')),'camera':'right_wrist',
        'success':False,'success_status':'not_verified','grasp_outcome':'unknown',
        'planning_success':True,'physical_attempt':True,'grasp_closure_executed':False,
        'observation_executed':True,'reason':'infrastructure_interrupted_during_pregrasp',
        'stages':common.read(t/'stages.json'),'execution_trace_complete':False,
        'clean_feedback_eligible':False,'infrastructure_interruption':True,
        'recovery_registration':str(epoch/'registration.json')}
    common.save(t/'result.json',result)
    _,baseline=bench.core.batch.original_modules();bench.lift.install_verifier(baseline)
    baseline.verify_trial(t)
    evaluation=common.read(t/'independent_evaluation.json')
    assert evaluation['passed'] and not evaluation['independently_successful']
    evaluation.update(grasp_outcome='unknown',execution_trace_complete=False,
        scope='Partial trace: no verified success. Native crash outcome is unknown and excluded from paired outcome analysis.')
    common.save(t/'independent_evaluation.json',evaluation)
    claim.update(state='complete',finished_at=bench.core.now(),result_sha256=common.sha(t/'result.json'),
        completion_kind='terminal_infrastructure_interruption',physical_execution_complete=False,replayed=False,
        recovery_registration=str(epoch/'registration.json'))
    common.save(claim_path,claim)
    verify_hashes(reg['retained_terminal_evidence_sha256'])
    verify_hashes({p:h for p,h in reg['interrupted_evidence_sha256'].items() if p!=str(claim_path)})
    common.save(epoch/'interruption_terminal_record.json',{'passed':True,'id':INTERRUPTED,
        'outcome':'unknown','replayed':False,'original_evidence_unchanged':True,
        'retained_completed_results_unchanged':True,'excluded_from_paired_outcome_analysis':True})
    bench.base.existing.summarize(a,'resuming',{})
    common.save(root/'A_zerograsp/progress.json',common.read(a/'progress.json'))
    (root/'A_zerograsp/resident').rename(epoch/'A_resident_epoch07')


def install_backends():
    # Change only the entry wrapper; all actual model and planning code is reused.
    original=shared.Manager.launch
    source=textwrap.dedent(inspect.getsource(original))
    old='graspcl_unseen_shared_services_20261119.py'
    assert source.count(old)==1
    namespace=dict(original.__globals__)
    exec(compile(source.replace(old,'graspcl_unseen_services_quiet_20261119.py'),str(SELF)+':service_entry','exec'),namespace)
    class Manager(shared.Manager):pass
    Manager.launch=types.FunctionType(namespace['launch'].__code__,original.__globals__,'launch')
    return Manager


def prepare_group(root,name,inputs,bank):
    group,arm=ORIGINAL_PREPARE(root,name,inputs,bank)
    receipt=common.read(root/EPOCH/'registration.json')
    for directory in [group,arm]:
        p=directory/'registration.json';reg=common.read(p)
        reg['source_hashes'].update(receipt['source_sha256'])
        reg['source_hashes'][str(root/EPOCH/'registration.json')]=common.sha(root/EPOCH/'registration.json')
        reg.update(component_startup_wait_seconds=1800,
            model_gpu_by_component=receipt['model_gpu_by_component'],simulation_gpus=SCENE_GPUS,
            shared_model_lifecycle='One real warmed ZeroGrasp/cuRobo pair across remaining A/B/C slots; each GraspCL bank loads once.')
        common.save(p,reg)
    return group,arm


def analyze(root):
    from scipy.stats import binomtest
    np=bench.np;rng=np.random.default_rng(2026111950100)
    results={}
    for name in bench.PHASES[1:]:
        arm='fixed_zerograsp' if name==bench.PHASES[1] else 'fixed_graspcl'
        reg=common.read(root/name/'registration.json')
        results[name]={e['id']:common.read(root/name/arm/'trials'/e['id']/'result.json') for e in reg['schedule']}
    ids=list(results[bench.PHASES[1]]);comparisons={}
    for newer,older in [(bench.PHASES[3],bench.PHASES[2]),(bench.PHASES[3],bench.PHASES[1]),(bench.PHASES[2],bench.PHASES[1])]:
        eligible=[i for i in ids if not any(results[n][i].get('infrastructure_interruption') for n in [newer,older])]
        models=sorted({results[newer][i]['model_id'] for i in eligible})
        diffs=np.array([int(results[newer][i]['success'])-int(results[older][i]['success']) for i in eligible])
        sums=np.array([sum(diffs[k] for k,i in enumerate(eligible) if results[newer][i]['model_id']==m) for m in models])
        counts=np.array([sum(results[newer][i]['model_id']==m for i in eligible) for m in models])
        ix=rng.integers(0,len(models),(20000,len(models)));boot=sums[ix].sum(1)/counts[ix].sum(1)
        signs=rng.choice([-1,1],(50000,len(models)));null=(signs*sums).sum(1)
        gains=int((diffs==1).sum());losses=int((diffs==-1).sum())
        comparisons[newer+' vs '+older]={'paired_cases':len(eligible),
            'excluded_unknown_infrastructure_outcomes':[i for i in ids if i not in eligible],
            'gain_scenes':gains,'loss_scenes':losses,'paired_difference':float(diffs.mean()),
            'model_cluster_bootstrap_95_ci':np.quantile(boot,[.025,.975]).tolist(),
            'two_sided_model_cluster_signflip_p':float((1+(np.abs(null)>=abs(diffs.sum())).sum())/50001),
            'mcnemar_exact_descriptive_p':float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.}
    shared_cases=0
    for slot in ids:
        receipts=[]
        for name in bench.PHASES[1:]:
            arm='fixed_zerograsp' if name==bench.PHASES[1] else 'fixed_graspcl'
            p=root/name/arm/'trials'/slot/'shared_input_receipt.json'
            if p.exists():receipts.append(common.read(p)['sha256'])
        if receipts:
            assert len(receipts)==3 and receipts[0]==receipts[1]==receipts[2]
            shared_cases+=1
    primary=list(comparisons)[:2];order=sorted(primary,key=lambda k:comparisons[k]['two_sided_model_cluster_signflip_p'])
    adjustment=0.
    for index,key in enumerate(order):
        adjustment=max(adjustment,min(1.,comparisons[key]['two_sided_model_cluster_signflip_p']*(len(order)-index)))
        comparisons[key]['primary_holm_adjusted_p']=adjustment
    supported=all(comparisons[k]['model_cluster_bootstrap_95_ci'][0]>0 and comparisons[k]['primary_holm_adjusted_p']<.05 for k in primary)
    common.save(root/'analysis.json',{'comparisons':comparisons,'shared_rgbd_native_candidate_cases':shared_cases,
        'positive_evidence_for_added_experience_over_both_controls':supported,
        'common_frontend_failures_retained':True,'unknown_infrastructure_outcomes_excluded_per_comparison':True,
        'experience_bank_frozen':True,'registered_test_cases_per_arm':100})


def run(root):
    epoch=root/EPOCH;receipt=common.read(epoch/'registration.json')
    verify_hashes(receipt['source_sha256']);verify_hashes(receipt['frozen_input_manifest_sha256'])
    verify_hashes(receipt['frozen_bank_sha256'])
    assert common.sha(root/'registration.json')==receipt['original_registration_sha256']
    assert common.sha(root/'A_zerograsp/registration.json')==receipt['A_registration_sha256']
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (epoch/'started.json').open('x') as f:f.write(bench.core.now()+'\n')
        Manager=install_backends();shared.MANAGER=Manager(root,EPOCH)
        resident.EPOCH=EPOCH;resident.aggregate=aggregate
        previous.EPOCH=EPOCH;previous.driver.SIMULATOR=CODE/'graspcl_unseen_simulator_quiet_20261119.py'
        bench.GPUS=SCENE_GPUS;bench.GPU0=shared.PLANNER_GPU
        bench.aggregate=aggregate;bench.Campaign=previous.driver.Campaign;bench.Backends=shared.Backends
        resident.POOL=previous.Simulators(root)
        success=False
        try:
            terminalize_interruption(root)
            aggregate(root,'preloading_simulation');resident.POOL.start()
            aggregate(root,'testing_A_zerograsp')
            bench.run_group(root/'A_zerograsp',root/'A_zerograsp/fixed_zerograsp')
            inputs=Path(common.read(root/'registration.json')['inputs'])
            for name,bank in [(bench.PHASES[2],root/'banks/no_experience'),(bench.PHASES[3],root/'banks/experience50')]:
                verify_hashes(receipt['frozen_bank_sha256'])
                group,arm=prepare_group(root,name,inputs/'test',bank)
                aggregate(root,'testing_'+name);bench.run_group(group,arm)
            analyze(root);success=True
        except BaseException:
            common.save(epoch/'halt.json',{'at':bench.core.now(),'error':traceback.format_exc(),'replay_allowed':False})
            aggregate(root,'halted');raise
        finally:
            resident.POOL.stop();shared.MANAGER.stop()
            stopped=common.read(epoch/'model_services_stopped.json')
            for name in bench.PHASES[1:]:
                p=root/name/'suite_timing.json'
                if p.exists():
                    v=common.read(p);v.update(model_pair_shared_until_suite_shutdown=True,
                        all_owned_services_exited=stopped['all_owned_model_services_exited']);common.save(p,v)
            if success:
                verify_hashes(receipt['frozen_bank_sha256']);verify_hashes(receipt['retained_terminal_evidence_sha256'])
                common.save(root/'final_verification.json',{'passed':True,'runtime_epoch':8,'registered_slots':350,
                    'terminal_slots':350,'uninterrupted_results':348,'source_infrastructure_interruptions':1,
                    'test_infrastructure_interruptions':1,'test_terminal_slots':300,'test_uninterrupted_results':299,
                    'test_feedback_updates':0,'physical_grasps_replayed':0,'experience_banks_unchanged':True,
                    'all_services_exited':stopped['all_owned_model_services_exited'],
                    'all_simulators_exited':common.read(epoch/'simulation_stopped.json')['all_owned_simulators_exited'],
                    'epoch08_zerograsp_model_loads':stopped['zero_model_loads'],
                    'epoch08_curobo_pool_loads':stopped['curobo_pool_loads']})
            aggregate(root,'complete' if success else 'halted')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)(args.output.resolve())
