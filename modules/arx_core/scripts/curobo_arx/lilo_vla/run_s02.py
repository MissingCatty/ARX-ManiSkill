"""Registered S02 single-attempt runner. Preparation never sends RPC or loads CUDA.

Execute the registered trials in order. No automatic retries, scene changes,
goal substitution, old campaign continuation, or policy-service replacement.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'scripts/curobo_arx')]
from interaction_contract import read,write,prepare_environment,check_health,reject_overrides,file_sha
from s02_protocol import registration,verify_sources,check_setup,check_runtime
from s02_gates import VERSION as GATE_VERSION, trial_gate, pilot_coverage


def prepare(design,out):
    value=registration(design)
    ancestor=out.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<=1073741824:
        fallback=Path('/coolas/data/arx_s02')/out.name
        fallback.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(fallback.parent).free<=1073741824:raise OSError('No storage above the 1GiB reserve')
        print('S02 storage fallback:',fallback,flush=True);out=fallback
    out.mkdir(parents=True,exist_ok=False)
    write(out/'registration.json',value)
    storage=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if not storage.exists():storage=ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/storage_policy.json'
    shutil.copy2(storage,out/'storage_policy.json')
    for source in value['algorithms']:
        path=Path(source)
        relative=path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('curobo')/path.relative_to(ROOT.parent/'curobo')
        destination=out/'source_snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    write(out/'state.json',dict(status='prepared',physical_attempts=0,next_trial=value['schedule'][0]['trial_id']))
    return value


def preflight(value,row,fixture_registration,policy_url):
    from collect_batch import preflight as fixture_preflight
    from gpu_policy import verify_health,POLICY_GPU
    from lilo_vla.trace import VERSION
    from client import Client
    import httpx
    verify_sources(value)
    case=value['cases'][row['case_id']];fixture=fixture_registration['fixture']
    if fixture['renderer']!=case['renderer']:raise ValueError('Registered historical renderer differs from supplied fixture')
    if 'lilo_egad49_20261026_eval01' in str(fixture['mailbox']):raise ValueError('Stopped EGAD49 mailbox is excluded')
    health=httpx.get(policy_url.rstrip('/')+'/health',timeout=15,trust_env=False);health.raise_for_status();health=health.json()
    check_health(case['resolved'][row['arm']],health);verify_health(health,POLICY_GPU)
    if health.get('trace_version')!=VERSION:raise ValueError('Restart the owned policy service with S02 instrumentation')
    evidence=fixture_preflight(fixture_registration)
    expected={name:file_sha(ROOT/'arx-sim/arx_sim/diagnostics'/name) for name in
        ('curobo_runner.py','lilo_observation.py','research_fingerprint.py','collection_status.py')}
    if evidence['status'].get('instrumentation_sha256')!=expected:
        raise ValueError('Simulator startup source identity is stale; use a freshly started instrumented fixture')
    session=evidence['status']['session_id']
    capabilities=Client(fixture['mailbox'],session=session,explicit_root=True).request('grasp_lilo_status',timeout=15)
    if any(capabilities.get(k)!=VERSION for k in ('trace_version','trajectory_trace_version')):
        raise ValueError('Simulator trace capability missing')
    return dict(policy=health,fixture=evidence,capabilities=capabilities,session=session)


def execute(value,row,out,fixture_registration,policy_url,evidence):
    from batch_runtime import Runtime,run_owned,confirm_idle
    from batch_manifest import pipeline_command
    from client import Client
    from egad_evaluate import evaluate
    fixture=fixture_registration['fixture'];case=value['cases'][row['case_id']]
    trial=out/'trials'/row['trial_id'];trial.mkdir(parents=True,exist_ok=False)
    (trial/'environment_evidence').mkdir();write(trial/'attempt.json',row)
    began=time.monotonic();deadline=began+900;error=None;setup=None;cleanup_ok=False;setup_ok=False
    client=Client(fixture['mailbox'],journal=trial/'requests.jsonl',deadline=deadline,session=evidence['session'],explicit_root=True)
    runtime=Runtime(fixture_registration,evidence['session']);runtime.trial=trial;runtime.heartbeat=lambda:None
    scenario=read(ROOT/case['scenario_source']);write(trial/'scenario.json',scenario)
    try:
        setup=client.request('dynamic_setup',scenario=scenario,timeout=240);write(trial/'setup.json',setup)
        checked=check_setup(read(ROOT/case['setup_source']),setup)
        write(trial/'setup_verification.json',{**checked,'pass':checked['pass_check'],
            'historical_physical_properties_equal':checked['pass_check']})
        if not checked['pass_check']:raise ValueError('Frozen initial conditions or physics differ: '+str(checked['mismatches'][:12]))
        setup_ok=True
        write(trial/'fingerprint_before.json',setup['fingerprint']);write(trial/'active_asset.json',setup['asset'])
        view=read(ROOT/case['source_trial']/'viewport_initial.json');client.request('grasp_viewport_state',restore=view)
        actual=client.request('grasp_viewport_state');write(trial/'viewport_initial.json',actual)
        if any(actual[k]!=view[k] for k in ('camera_path','attributes')):raise ValueError('Historical viewport differs')
        for key in ('world_matrix','local_matrix'):
            if any(abs(a-b)>1e-8 for ar,br in zip(actual[key],view[key]) for a,b in zip(ar,br)):raise ValueError('Historical viewport pose differs')
        write(trial/'guard.json',dict(invariant_sha256='Before/after runtime evidence mandatory',file_sha256=value['dependencies']))
        env={k:v for k,v in os.environ.items() if not k.startswith(('ARX_LILO_','GRASP_','ARX_REQUEST_','ARX_COLLECTION_','ARX_ATTEMPT_'))}
        env.pop('ARX_RESEARCH_GUARD_FILE',None)
        env.update(ARX_LILO_POLICY_URL=policy_url,ARX_LILO_TRACE='1',ARX_CUROBO_VALIDATION_DIR=fixture['mailbox'],
            ARX_OPS_URL=fixture['ops_url'],ARX_OBJECT_LABEL=case['asset'],ARX_BATCH_MANAGED_CLEANUP='1',
            ARX_REQUEST_JOURNAL=str(trial/'requests.jsonl'),ARX_COLLECTION_SESSION=evidence['session'],
            ARX_ATTEMPT_DEADLINE=str(deadline),ARX_RESEARCH_AUDIT='0',PYTHONUNBUFFERED='1')
        env=prepare_environment(env,trial,case['resolved'][row['arm']])
        command=pipeline_command(fixture_registration,trial)+['--fixed-selection',str(ROOT/case['fixed_goal_source'])]
        write(trial/'execution_registration.json',dict(command=command,policy=evidence['policy'],
            trace_enabled=True,expected_rule=case['resolved'][row['arm']]['contract']['handoff_rule'],
            fixture=fixture,source_case=case['case_id'],profile='Explicit S02 research path; original P3 profile unchanged'))
        code=run_owned(command,env,trial/'pipeline.log',deadline)
        if code:raise RuntimeError('Policy pipeline exited '+str(code))
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        try:
            cleanup_deadline=time.monotonic()+60
            runtime.cleanup(cleanup_deadline);cleanup_ok=True
            final=Client(fixture['mailbox'],journal=trial/'requests.jsonl',deadline=cleanup_deadline,
                session=evidence['session'],explicit_root=True).request('grasp_research_fingerprint',timeout=60)
            write(trial/'fingerprint_after.json',final)
            if setup:write(trial/'runtime_property_check.json',check_runtime(setup['fingerprint'],final))
        except Exception as exc:
            cleanup_ok=False;write(trial/'cleanup_error.json',dict(error=str(exc)))
    try:physical=evaluate(trial)
    except Exception as exc:physical=dict(pass_check=False,**{'pass':False},error=str(exc))
    write(trial/'physical_evaluation.json',physical)
    audits=[]
    for name,python,flags in (('motion',ROOT.parent/'curobo/.venv/bin/python',[]),
                              ('pixels',ROOT.parent/'lilo-vla/.venv/bin/python',['--pixels'])):
        report=trial/('trace_'+name+'.json')
        try:
            code=run_owned([python,HERE/'audit_s02.py','--trial',trial,'--output',report,*flags],
                {**os.environ,'CUDA_VISIBLE_DEVICES':'','TF_CPP_MIN_LOG_LEVEL':'3'},trial/('audit_'+name+'.log'),time.monotonic()+120)
            receipt=read(report)
            if code!=0:receipt['record_integrity']='failed'
            audits.append(receipt)
        except Exception as exc:
            audits.append(dict(record_integrity='failed',errors=[str(exc)]))
            write(trial/('audit_'+name+'_error.json'),dict(error=str(exc)))
    runtime_ok=(trial/'runtime_property_check.json').exists() and read(trial/'runtime_property_check.json')['pass_check']
    result=dict(**row,error=error,cleanup_ok=cleanup_ok,physical_pass=physical['pass'],
        **trial_gate(trial,error,audits,physical,cleanup_ok,setup_ok,runtime_ok),
        elapsed_seconds=time.monotonic()-began,attempted=True)
    write(trial/'result.json',result)
    return result


def next_trial(value,out,trial_id):
    remaining=[r for r in value['schedule'] if not (out/'trials'/r['trial_id']).exists()]
    if not remaining or remaining[0]['trial_id']!=trial_id:raise ValueError('Execute only the next registered trial; no automatic repeats')
    for row in value['schedule']:
        trial=out/'trials'/row['trial_id']
        if trial.exists():
            if not (trial/'result.json').exists():raise ValueError('Previous attempt is unresolved')
            saved=read(trial/'result.json')
            if saved.get('gate_version')!=GATE_VERSION or not saved.get('cleanup_ok') or saved.get('halt') or saved.get('record_integrity')!='complete' or saved.get('infrastructure_failure'):
                raise ValueError('Previous attempt instrumentation/infrastructure is unresolved; do not send a new command')
    row=remaining[0]
    if row['phase']=='main':
        results={r['trial_id']:read(out/'trials'/r['trial_id']/'result.json') for r in value['schedule'] if r['phase']=='pilot'}
        coverage=pilot_coverage(value['schedule'],results)
        if not coverage['pass_check']:raise ValueError('Pilot critical-event coverage incomplete: '+str(coverage['missing']))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--design',type=Path,default=ROOT/'research_plan/configs/s02_design.json')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true');mode.add_argument('--preflight',action='store_true')
    parser.add_argument('--trial-id');parser.add_argument('--fixture-registration',type=Path);parser.add_argument('--policy-url',default='http://127.0.0.1:8008')
    args=parser.parse_args();out=args.output.resolve()
    reject_overrides(os.environ)
    if args.prepare:
        prepare(args.design,out);print('S02 registered: 4 instrumentation + 32 main trials; no physical execution.');return
    if not args.trial_id or not args.fixture_registration:parser.error('An explicit trial and isolated fixture registration are required')
    from batch_ledger import lock
    value=read(out/'registration.json');verify_sources(value)
    fixture=read(args.fixture_registration)
    with lock(out/'campaign.lock'),lock(Path(fixture['fixture']['mailbox'])/'p0_execution.lock'):
        row=next_trial(value,out,args.trial_id)
        if shutil.disk_usage(out).free<=1073741824:raise OSError('Preserve the 1GiB reserve; use dedicated /coolas/data storage for a new registration')
        try:
            evidence=preflight(value,row,fixture,args.policy_url)
            write(out/('preflight_'+row['trial_id']+'.json'),evidence)
        except Exception as exc:
            write(out/('preflight_'+row['trial_id']+'_failure_'+str(time.time_ns())+'.json'),
                dict(error=str(exc),physical_attempt_started=False))
            raise
        if args.preflight:return
        result=execute(value,row,out,fixture,args.policy_url,evidence)
        write(out/'state.json',dict(status='halted' if result['halt'] else 'trial_completed',last=result,
            physical_attempts=sum((out/'trials'/r['trial_id']).exists() for r in value['schedule'])))


if __name__=='__main__':main()
