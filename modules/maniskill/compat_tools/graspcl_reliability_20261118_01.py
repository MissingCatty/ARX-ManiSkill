"""Frozen-bank leakage audit, paired statistics, and outcome-blind repeat validation."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import types

CODE=Path('/home/zhaoyechong/projects/ARX-ManiSkill/modules/maniskill')
SELF=Path(__file__).resolve()
MAIN=CODE/'outputs/arx_physical_table_parallel_20261118_01'
INPUTS=CODE/'benchmark_inputs/arx_independent_20261117_01'
GPU1='GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e'
RUNTIME=CODE/'tmp/physical_table_parallel_20261118_01.py'
sys.path.insert(0,str(CODE))


def read(p):return json.loads(Path(p).read_text())


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.pending',dir=p.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(p)
    finally:
        temporary.unlink(missing_ok=True)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_runtime():
    spec=importlib.util.spec_from_file_location('unchanged_parallel_runtime',RUNTIME)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def digest_order(value):return hashlib.sha256(('graspcl-repeat-validation-20261118:'+value).encode()).hexdigest()


def family(model):return model.split('_')[0].split('-')[0]


def select_cases(schedule):
    # 30 unique models, all 17 YCB numeric families, six scenes per layout.
    models=sorted({e['model_id'] for e in schedule},key=digest_order)
    families=sorted({family(m) for m in models},key=digest_order)
    first=[next(m for m in models if family(m)==f) for f in families]
    selected=first+[m for m in models if m not in first][:30-len(first)]
    selected=sorted(selected,key=lambda m:digest_order('assignment:'+m))
    layouts=sorted({e['layout'] for e in schedule},key=lambda x:digest_order('layout:'+x))
    cases=[]
    for i,model in enumerate(selected):
        layout=layouts[i%5]
        candidates=[e for e in schedule if e['model_id']==model and e['layout']==layout]
        cases.append(min(candidates,key=lambda e:digest_order(e['id'])))
    assert len(cases)==len({e['id'] for e in cases})==len({e['model_id'] for e in cases})==30
    assert set(Counter(e['layout'] for e in cases).values())=={6}
    assert len({family(e['model_id']) for e in cases})==17
    return cases


def configure(reg):
    import short_lift_bench as base
    original=base.verify
    source=textwrap.dedent(inspect.getsource(original))
    old="assert len(files)==200"
    assert source.count(old)==1
    source=source.replace(old,"assert len(files)==common.read(arm/'registration.json')['attempt_budget']")
    scope=dict(original.__globals__);exec(compile(source,str(SELF)+':registered_repeat_budget','exec'),scope)
    generated=scope[original.__name__]
    base.verify=types.FunctionType(generated.__code__,original.__globals__,original.__name__,original.__defaults__)
    base=load_runtime().configure(reg)
    base.__file__=str(SELF)
    return base


def prepare(out):
    original=read(MAIN/'graspcl/registration.json')
    manifest=read(INPUTS/'manifest.json')
    cases=select_cases(manifest['schedule'])
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    shutil.copy2(SELF,out/'source/reliability.py')
    selection={'created_at':datetime.now().astimezone().isoformat(),'selected_cases':cases,
        'selection_uses_outcomes':False,'selection_rule':'SHA256 model/family/layout ordering, cover 17 numeric model families, 30 unique models and 6 cases per layout; no success, failure, candidate or score used.',
        'repeats_per_method':3,'new_slots':180,'repeat_initial_conditions':'Same declared recipe, original reset seed, observation pose, physics and algorithm RNG behavior; fresh reset/capture/inference each trial.',
        'primary_pair':str(MAIN),'offline_learning_updates':False,'online_learning_updates':False}
    save(out/'selection.json',selection)
    inputs=out/'inputs';inputs.mkdir()
    for f in ['common_initial_conditions.json','storage_policy.json']:shutil.copy2(INPUTS/f,inputs/f)
    schedule=[];sources=dict(original['source_hashes'])
    for repetition in range(1,4):
        for canonical in cases:
            identity=canonical['id']+f'_R{repetition}'
            admission=deepcopy(read(INPUTS/'admission'/canonical['id']/'admission.json'))
            entry=admission['entry'];entry.update(id=identity,canonical_case_id=canonical['id'],repeat_index=repetition)
            admitted=inputs/'admission'/identity/'admission.json';save(admitted,admission)
            reset=inputs/'declared_resets'/identity/'declared_reset.json';reset.parent.mkdir(parents=True)
            shutil.copy2(INPUTS/'declared_resets'/canonical['id']/'declared_reset.json',reset)
            schedule.append(entry)
            sources[str(admitted)]=sha(admitted);sources[str(reset)]=sha(reset)
    arms=[]
    for r in range(1,4):
        profiles=['zerograsp_200','graspcl_200'] if r%2 else ['graspcl_200','zerograsp_200']
        arms.extend([[f'R{r}_'+('zerograsp' if p=='zerograsp_200' else 'graspcl'),p,'fixed_orientation'] for p in profiles])
    source_files=[SELF,RUNTIME,CODE/'tmp/physical_table_pair_20261118_01.py',out/'selection.json',INPUTS/'independence_preflight.json']
    source_files.extend([inputs/'common_initial_conditions.json',inputs/'storage_policy.json'])
    sources.update({str(p):sha(p) for p in source_files})
    reg=deepcopy(original)
    reg.update(source_hashes=sources,created_at=datetime.now().astimezone().isoformat(),inputs=str(inputs),
        schedule=schedule,arms=arms,total_slots=180,attempt_budget_by_arm={a:30 for a,_,_ in arms},
        required_components=['zerograsp','graspcl','curobo'],runtime_backend_gpu=GPU1,runtime_scene_gpus=[GPU1],
        backend_lock=str(out/'backend.lock'),scene_workers=1,common_reset_source=str(inputs/'common_initial_conditions.json'),
        inherited_completed=0,source_evaluation_results_carried=0,prior_test_results_carried=0,
        repeats_per_method=3,unique_canonical_cases=30,repeat_target_models=30,
        admitted=sum(read(inputs/'admission'/e['id']/'admission.json')['passed'] for e in schedule),
        admission_failed=sum(not read(inputs/'admission'/e['id']/'admission.json')['passed'] for e in schedule),
        algorithm_changed=False,current_test_feedback_added=0,
        test_design_scope='Preregistered repeatability validation on an outcome-blind 30-scene subset of the existing held-out set; not a newly sampled unseen-scene generalization test. Three additional repeats per method, alternating method order, one GPU and unchanged policy.',
        statistical_plan={'primary':'Paired success difference on all 200 registered primary scenes',
            'repeat':'Per-canonical-scene mean success difference across three additional repeats',
            'confidence_interval':'20000 bootstrap replicates clustered by numeric YCB family; seed 20261118',
            'tests':['exact paired McNemar, descriptive under scene independence','exact family-cluster sign-flip randomization'],
            'decision':'Positive effect, family-cluster 95% interval lower bound above zero and family sign-flip p <= .05; require primary and pooled repeat validation to pass and all three repeat directions positive for a consistent improvement verdict. Otherwise insufficient or inconsistent evidence.',
            'no_optional_stopping':True,'no_failed_slot_replay':True})
    save(out/'registration.json',reg)
    for arm,profile,_ in arms:
        repetition=int(arm[1]);directory=out/arm;directory.mkdir()
        for f in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:(directory/f).mkdir()
        shutil.copy2(INPUTS/'common_initial_conditions.json',directory/'reference/common_initial_conditions.json')
        shutil.copy2(INPUTS/'storage_policy.json',directory/'storage_policy.json')
        (directory/'resident').symlink_to(out/'resident',target_is_directory=True)
        a=deepcopy(reg);a.update(suite=str(out),experiment_arm=arm,pipeline_profile=profile,
            orientation_mode='fixed_orientation',attempt_budget=30,schedule=[e for e in schedule if e['repeat_index']==repetition],
            common_reset_source=str(directory/'reference/common_initial_conditions.json'))
        a['source_hashes'][str(directory/'reference/common_initial_conditions.json')]=sha(directory/'reference/common_initial_conditions.json')
        save(directory/'registration.json',a)
        save(directory/'progress.json',{'phase':'prepared','completed':0,'successes':0,'total':30})
    import short_lift_bench as base
    base.update_suite(out,'prepared')
    save(out/'audit_status.json',{'phase':'waiting_for_primary_zerograsp','new_physical_slots_started':0})
    print('PREREGISTERED 30 scenes x 3 additional repeats x 2 methods = 180 slots',flush=True)


def ready(root):
    try:
        p=read(root/'progress.json');clock=read(root/'suite_timing.json')
        if p['phase']=='halted':raise RuntimeError('Primary ZeroGrasp halted; do not start extra trials')
        return p['phase']=='complete' and p['completed']==200 and clock.get('all_owned_services_exited') and read(root/'final_verification.json')['passed']
    except FileNotFoundError:return False


def leakage_audit():
    import numpy as np
    reg=read(MAIN/'graspcl/registration.json');bank=Path(reg['bank']);complete=read(bank/'bank_complete.json')
    origin=read(CODE/'outputs/arx_zerograsp_graspcl_demo_20261117_02/registration.json')
    labelled={e['id']:e for e in origin['training_trials'] if e.get('label') is not None}
    ids=complete['source_ids'];assert ids==reg['source_feedback_ids_in_memory_order']
    with np.load(bank/'model/score_cl/online_data.npz',allow_pickle=False) as f:
        labels=f['y'].reshape(-1).astype(int).tolist();feature_hash=hashlib.sha256(f['X'].tobytes()).hexdigest()
    assert labels==[labelled[i]['label'] for i in ids]
    tests={e['id'] for e in reg['schedule']};test_models=set(reg['test_object_ids'])
    training_models={labelled[i]['model_id'] for i in ids}
    negative_ids={i for i,label in zip(ids,labels) if label==0}
    feedback_overlap=sorted(tests&set(ids));negative_overlap=sorted(tests&negative_ids)
    model_overlap=sorted(test_models&training_models)
    preflight=read(INPUTS/'independence_preflight.json');assert preflight['passed']
    runtime=MAIN/'graspcl/resident/graspcl_model/model/score_cl/online_data.npz'
    expected_bank_sha=complete['model_sha256'][str(bank/'model/score_cl/online_data.npz')]
    bank_unchanged=sha(runtime)==sha(bank/'model/score_cl/online_data.npz')==expected_bank_sha
    guard=read(MAIN/'graspcl/resident/independent_graspcl_guard_ready.json')
    assert guard['updates_disabled'] and guard['feedback_points']==25 and set(guard['source_ids'])==set(ids)
    checks=Counter();violations=[]
    for f in (MAIN/'graspcl/fixed_graspcl/claims').glob('*.json'):
        claim=read(f)
        if claim['state']!='complete':continue
        trial=MAIN/'graspcl/fixed_graspcl/trials'/f.stem
        assert sha(trial/'result.json')==claim['result_sha256']
        for filename in ['frozen_memory_inference.json','frozen_memory_rescore.json']:
            p=trial/filename
            if p.exists():
                m=read(p);checks['memory_receipts']+=1
                if not (m['points']==25 and m['test_feedback_added']==0 and m['before_sha256']==m['after_sha256']==guard['memory_hash']):violations.append(str(p))
        p=trial/'self_case_exclusion.json'
        if p.exists():
            m=read(p);checks['exclusion_receipts']+=1
            if set(m['retained_feedback_ids'])!=set(ids) or m['excluded_feedback_ids'] or m['retained_feature_sha256']!=feature_hash:violations.append(str(p))
        p=trial/'graspcl/candidate_pool.json'
        if p.exists():
            for g in read(p).get('recalled_before_filter',[]):
                source=g['metadata'].get('source_trial_id');checks['recalled_grasps']+=1
                if source not in ids or source in tests or labelled[source]['label']!=1:violations.append(str(p)+':'+str(source))
        checks['completed_queries']+=1
    passed=not feedback_overlap and not negative_overlap and not model_overlap and bank_unchanged and not violations
    return {'passed':passed,'feedback_points':25,'positive_feedback':labels.count(1),'negative_feedback':labels.count(0),
        'same_test_scene_feedback_overlap':feedback_overlap,'same_test_scene_negative_feedback_overlap':negative_overlap,
        'feedback_object_model_overlap':model_overlap,'current_bank_file_unchanged':bank_unchanged,
        'updates_disabled':guard['updates_disabled'],'checks':dict(checks),'violations':violations,
        'preflight':preflight,'source_feedback_ids':ids,'negative_feedback_ids':sorted(negative_ids),
        'scope':'Local frozen ARX feedback vs current held-out benchmark. No guarantee about upstream pretraining exposure. No model, inference, simulation or physical action performed by this audit.'}


def paired_statistics(zero,cl,case_info):
    import numpy as np
    ids=sorted(zero);assert set(ids)==set(cl)
    a=np.asarray([zero[i] for i in ids],dtype=float);b=np.asarray([cl[i] for i in ids],dtype=float);d=b-a
    groups={}
    for i,key in enumerate(ids):groups.setdefault(family(case_info[key]['model_id']),[]).append(i)
    sums=np.array([d[idx].sum() for idx in groups.values()]);sizes=np.array([len(idx) for idx in groups.values()])
    rng=np.random.default_rng(20261118);draw=rng.integers(0,len(sums),size=(20000,len(sums)))
    bootstrap=sums[draw].sum(-1)/sizes[draw].sum(-1);ci=np.quantile(bootstrap,[.025,.975])
    nonzero=sums[np.abs(sums)>1e-12];g=len(nonzero)
    # All 2^g family label swaps; at most 17 semantic groups in this dataset.
    bits=np.arange(1<<g,dtype=np.uint32)[:,None]
    signs=((bits>>np.arange(g,dtype=np.uint32))&1).astype(float)*2-1
    null=signs@nonzero if g else np.zeros(1)
    p=float(np.mean(np.abs(null)>=abs(float(sums.sum()))-1e-10))
    binary=bool(np.isin(a,[0,1]).all() and np.isin(b,[0,1]).all())
    mc=None;contingency=None
    if binary:
        wins=int(((a==0)&(b==1)).sum());losses=int(((a==1)&(b==0)).sum());n=wins+losses
        mc=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
        contingency={'both_success':int(((a==1)&(b==1)).sum()),'graspcl_only_success':wins,'zerograsp_only_success':losses,'both_fail':int(((a==0)&(b==0)).sum())}
    return {'paired_canonical_cases':len(ids),'semantic_family_clusters':len(groups),
        'zerograsp_mean_success':float(a.mean()),'graspcl_mean_success':float(b.mean()),
        'graspcl_minus_zero_success_rate':float(d.mean()),'family_cluster_bootstrap_95ci':ci.tolist(),
        'family_cluster_sign_flip_p':p,'exact_mcnemar_p_descriptive':mc,'paired_counts':contingency,
        'positive_evidence_under_preregistered_rule':bool(d.mean()>0 and ci[0]>0 and p<=.05),
        'interpretation':'Full frozen recall-plus-reranking chain, not a scoring-only or negative-feedback-only ablation; failure to meet the rule means insufficient evidence, not proof of no effect.'}


def result_map(root):
    reg=read(root/'registration.json');rows={}
    for arm,profile,_ in reg['arms']:
        for f in (root/arm/'claims').glob('*.json'):
            c=read(f)
            if c['state']!='complete':continue
            trial=root/arm/'trials'/f.stem
            assert sha(trial/'result.json')==c['result_sha256'] and read(trial/'independent_evaluation.json')['passed']
            r=read(trial/'result.json')
            rows[(profile,r['id'])]=bool(r['success'])
    return rows


def analyze(out):
    save(out/'provenance_audit.json',leakage_audit())
    info={e['id']:e for e in read(MAIN/'zerograsp/registration.json')['schedule']}
    za=read(MAIN/'zerograsp/progress.json');ga=read(MAIN/'graspcl/progress.json')
    status={'primary_zero_completed':za['completed'],'primary_graspcl_completed':ga['completed']}
    if za['phase']=='complete' and ga['phase']=='complete':
        z=result_map(MAIN/'zerograsp');g=result_map(MAIN/'graspcl')
        primary=paired_statistics({i:z[('zerograsp_200',i)] for i in info},{i:g[('graspcl_200',i)] for i in info},info)
        save(out/'primary_effectiveness.json',primary);status['primary_analysis_complete']=True
    if read(out/'progress.json')['phase']=='complete':
        selection=read(out/'selection.json');repeat=result_map(out)
        cases=selection['selected_cases'];zero={};cl={}
        for case in cases:
            key=case['id'];zero[key]=sum(repeat[('zerograsp_200',key+f'_R{r}')] for r in range(1,4))/3
            cl[key]=sum(repeat[('graspcl_200',key+f'_R{r}')] for r in range(1,4))/3
        validation=paired_statistics(zero,cl,{c['id']:c for c in cases})
        validation['individual_repeats']=[{'repeat':r,
            'zerograsp_successes':sum(repeat[('zerograsp_200',case['id']+f'_R{r}')] for case in cases),
            'graspcl_successes':sum(repeat[('graspcl_200',case['id']+f'_R{r}')] for case in cases),
            'graspcl_minus_zero_success_rate':sum(int(repeat[('graspcl_200',case['id']+f'_R{r}')])-int(repeat[('zerograsp_200',case['id']+f'_R{r}')]) for case in cases)/30} for r in range(1,4)]
        validation['all_repeat_directions_positive']=all(r['graspcl_minus_zero_success_rate']>0 for r in validation['individual_repeats'])
        validation['repeats_per_method']=3;validation['successes_by_case']={i:{'zerograsp':[repeat[('zerograsp_200',i+f'_R{r}')] for r in range(1,4)],'graspcl':[repeat[('graspcl_200',i+f'_R{r}')] for r in range(1,4)]} for i in zero}
        save(out/'repeat_effectiveness.json',validation);status['repeat_analysis_complete']=True
    if status.get('primary_analysis_complete') and status.get('repeat_analysis_complete'):
        p=read(out/'primary_effectiveness.json');v=read(out/'repeat_effectiveness.json');clean=read(out/'provenance_audit.json')['passed']
        verdict='consistent_positive_evidence_on_this_benchmark' if clean and p['positive_evidence_under_preregistered_rule'] and v['positive_evidence_under_preregistered_rule'] and v['all_repeat_directions_positive'] else 'insufficient_or_inconsistent_evidence' if clean else 'data_leakage_or_integrity_violation'
        save(out/'conclusion.json',{'verdict':verdict,'leakage_audit_passed':clean,'primary':p,'repeat_validation':v,
            'scope':'Evidence on the current held-out local benchmark and its preregistered repeatability subset, not proof of general validity or contribution of the scorer alone.'})
        status['phase']='complete'
    else:status['phase']='running_or_waiting_for_fixed_sample_sizes'
    save(out/'audit_status.json',status)
    return status


def wait_run(out):
    while not ready(MAIN/'zerograsp'):
        save(out/'queue_status.json',{'phase':'waiting_for_primary_zerograsp','updated_at':datetime.now().astimezone().isoformat(),'new_physical_slots_started':0})
        time.sleep(10)
    clean=leakage_audit();save(out/'provenance_audit.json',clean)
    assert clean['passed'],'Do not run repeat validation with contaminated or inconsistent memory'
    with (out/'queue_start.json').open('x') as f:json.dump({'started_at':datetime.now().astimezone().isoformat(),'primary_zero_verified_complete':True,'source_feedback_overlap':0},f)
    save(out/'queue_status.json',{'phase':'released','new_trial_budget':180})
    reg=read(out/'registration.json');base=configure(reg);base.run(out)


def watch(out):
    while not ready(MAIN/'zerograsp'):time.sleep(10)
    while True:
        try:status=analyze(out)
        except json.JSONDecodeError:
            time.sleep(1);continue
        print(json.dumps(status),flush=True)
        if status['phase']=='complete':return
        if read(out/'progress.json')['phase']=='halted':return
        time.sleep(60)


def self_test():
    schedule=read(INPUTS/'manifest.json')['schedule'];selected=select_cases(schedule)
    assert selected==select_cases(list(reversed(schedule)))
    ids={str(i):{'model_id':f'{i:03d}_fixture'} for i in range(12)}
    z={i:False for i in ids};c={i:True for i in ids}
    positive=paired_statistics(z,c,ids);assert positive['positive_evidence_under_preregistered_rule']
    equal=paired_statistics(z,z,ids);assert equal['family_cluster_sign_flip_p']==1 and not equal['positive_evidence_under_preregistered_rule']
    negative=paired_statistics(c,z,ids);assert negative['graspcl_minus_zero_success_rate']==-1 and not negative['positive_evidence_under_preregistered_rule']
    return {'passed':True,'outcome_blind_selection_order_invariant':True,'statistical_controls_passed':3,'physical_attempts':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','wait-run','run','execute','watch','self-test'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.stage=='self-test':print(json.dumps(self_test()));sys.exit(0)
    out=args.output.resolve()
    if args.stage=='prepare':prepare(out)
    elif args.stage=='watch':watch(out)
    elif args.stage=='wait-run':wait_run(out)
    else:
        reg=read(out/'registration.json') if args.stage=='run' else read(Path(read(args.entry)['campaign_registration']))
        base=configure(reg)
        if args.stage=='run':base.run(out)
        else:base.execute(out,args.entry.resolve())
