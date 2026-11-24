"""Single-fixture bounded teacher collection. Validation/rehearsal never sends RPC."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from batch_manifest import ROOT, FLOOR, resolve_manifest, verify_registration, rehearsal
from batch_ledger import Ledger, lock, previous, receipt, write
from batch_runtime import Runtime, status, confirm_idle, execute_attempt, run_owned
from interaction_contract import read, reject_overrides, validate_episode


def preflight(registration,session=None):
    from gpu_policy import resolve, verify_process
    fixture=registration['fixture'];mailbox=Path(fixture['mailbox'])
    value=status(mailbox,fixture,session)
    resolve(registration['resources']['gpu_uuid'])
    lines=subprocess.check_output(['podman','top',fixture['startup']['container'],'hpid','pid'],text=True,timeout=15).splitlines()[1:]
    matches=[int(fields[0]) for line in lines if len(fields:=line.split())==2 and fields[1]==str(value['pid'])]
    if len(matches)!=1:raise RuntimeError('Cannot map the declared simulator PID to its owned host process')
    gpu=verify_process(matches[0],registration['resources']['gpu_uuid'])
    imports={}
    for key,modules in (('neural_python',['numpy','scipy','PIL','yaml','torch','pydantic','httpx','imageio_ffmpeg']),
                        ('planner_python',['numpy','scipy','yaml','torch','curobo','trimesh'])):
        command=[registration['resources'][key],'-c',
            'import importlib.util,json,sys; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in sys.argv[1:]}))',*modules]
        import json
        imports[key]=json.loads(subprocess.check_output(command,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=20))
        if not all(imports[key].values()):raise RuntimeError('Missing explicit Python dependencies: '+str(imports[key]))
    confirm_idle(mailbox,fixture,value['session_id'],time.monotonic()+5)
    return {'status':value,'gpu':gpu,'python_dependencies':imports,'read_only':True}


def select_storage(output,fallback_root):
    output=Path(output).resolve();ancestor=output.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free>FLOOR:return output
    fallback=Path(fallback_root)/('batch_'+uuid.uuid4().hex)
    if not fallback.parent.is_dir() or shutil.disk_usage(fallback.parent).free<=FLOOR:raise OSError('Both local and dedicated storage lack the 1GiB reserve')
    print('ARX_STORAGE',str(output),'->',str(fallback),flush=True)
    return fallback


def collect(ledger,runtime_factory,session):
    registration=ledger.registration;done={r['case_id'] for r in ledger.results};goals={};seen=set()
    for r in ledger.results:
        if r.get('demonstration_ready'):seen.add(r['quality']['content_sha256'])
        source=Path(r['source']);selection=source/'selected_grasp.json'
        if r.get('goal_group_id') and selection.exists() and (source/'resolved_interaction.json').exists():
            resolved=read(source/'resolved_interaction.json')
            if resolved.get('grasp_world'):
                validate_episode(source,resolved['contract'])
                goals.setdefault(r['goal_group_id'],str(selection))
    consecutive=0;last_error=None;stop=None
    for index,case in enumerate(registration['cases']):
        if case['case_id'] in done:continue
        if stop:
            ledger.unexecuted.append({'case_id':case['case_id'],'reason':stop});continue
        if len(ledger.results)>=registration['budget']['max_attempts']:
            stop='cumulative_attempt_budget';ledger.unexecuted.append({'case_id':case['case_id'],'reason':stop});continue
        goal=goals.get(case.get('goal_group_id'))
        if case['goal_mode']=='external':goal=case['external_goal']['selection']
        if case['goal_mode']=='group' and not goal:
            ledger.unexecuted.append({'case_id':case['case_id'],'reason':'group_goal_unavailable'});ledger.aggregate();continue
        verify_registration(registration)
        if shutil.disk_usage(ledger.output).free<=FLOOR:
            write(ledger.output/'storage_continuation_required.json',{'resume_from':str(ledger.output),'recommended_output':str(Path(registration['resources']['storage_fallback_root'])/('batch_'+uuid.uuid4().hex)),'preserve_existing_evidence':True})
            stop='storage_reserve_reached';ledger.unexecuted.append({'case_id':case['case_id'],'reason':stop});continue
        result=execute_attempt(ledger,case,runtime_factory(registration,session),goal,seen)
        done.add(case['case_id'])
        if result.get('planned_goal') and case.get('goal_group_id'):goals.setdefault(case['goal_group_id'],result['planned_goal'])
        if result['demonstration_ready']:seen.add(result['quality']['content_sha256'])
        if result['stop_campaign']:stop='unsafe_or_frozen_boundary'
        if result['status']=='infrastructure_failure':
            key=result['error'] or result['cleanup_errors']
            consecutive=consecutive+1 if key==last_error else 1;last_error=key
            if consecutive>=3:stop='three_consecutive_identical_infrastructure_errors'
        else:consecutive=0;last_error=None
        ledger.progress('running',last_case=case['case_id'])
    ledger.aggregate();ledger.progress('stopped' if stop else 'completed',stop_reason=stop,unexecuted=len(ledger.unexecuted))


def encode_results(ledger):
    # No fixture lock is held here. Each encoder has its own 120s process budget.
    for result in ledger.results:
        trial=Path(result['source'])
        if trial.parent.parent!=ledger.output:continue
        if not (trial/'frames/frames.json').exists():
            write(trial/'video_status.json',{'status':'no_recorded_frames'});continue
        started=time.monotonic();error=None;encoding_id=uuid.uuid4().hex
        video=trial/(trial.name+'.mp4')
        if video.exists() and not (trial/'video_receipt.json').exists():
            video.rename(trial/(trial.name+'.partial_'+encoding_id+'.mp4'))
        command=[ledger.registration['resources']['neural_python'],'-c',
            'import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from summarize_chunks import encode;encode(Path(sys.argv[2]))',
            str(ROOT/'scripts/curobo_arx/lilo_vla'),str(trial)]
        try:
            code=run_owned(command,{**os.environ,'CUDA_VISIBLE_DEVICES':''},trial/('encoding_'+encoding_id+'.log'),started+120,
                lambda:print('ARX_ENCODING',result['case_id'],flush=True),kill_grace=0)
            if code:raise RuntimeError('Encoder exited '+str(code))
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        video_status={'status':'failed' if error else 'completed','seconds':time.monotonic()-started,'error':error,'raw_frames_preserved':True,'encoding_id':encoding_id}
        write(trial/('encoding_'+encoding_id+'.json'),video_status);write(trial/'video_status.json',video_status)
    write(ledger.output/'video_index.json',[{'case_id':r['case_id'],'source':r['source'],
          'video':read(Path(r['source'])/'video_status.json') if (Path(r['source'])/'video_status.json').exists() else {'status':'unprocessed'}} for r in ledger.results])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--validate-only',action='store_true');modes.add_argument('--preflight-only',action='store_true')
    modes.add_argument('--encode-only',action='store_true',help='Offline video retry for an existing campaign, without RPC or new attempts')
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    if args.encode_only:
        from types import SimpleNamespace
        out=args.output.resolve()
        with lock(out/'campaign.lock'):
            results=[receipt(p) for p in sorted((out/'attempts').glob('*/result.json'))]
            encode_results(SimpleNamespace(output=out,registration=read(out/'registration.json'),results=results))
        return
    if not args.manifest:parser.error('--manifest is required except for --encode-only')
    registration=resolve_manifest(args.manifest)
    inherited=previous(args.resume_from,registration) if args.resume_from else []
    if args.validate_only or args.preflight_only:
        rehearsal(registration,args.output)
        if args.resume_from:write(args.output/'resume_review.json',{'inherited_attempts':inherited,'never_started_cases':[c['case_id'] for c in registration['cases'] if c['case_id'] not in {r['case_id'] for r in inherited}]})
        if args.preflight_only:
            with lock(Path(registration['fixture']['mailbox'])/'p0_execution.lock'):write(args.output/'preflight.json',preflight(registration))
        print('ARX_BATCH_VALIDATED',len(registration['cases']),'cases; no physical execution',flush=True);return
    reject_overrides(os.environ)
    out=select_storage(args.output,registration['resources']['storage_fallback_root']);ledger=Ledger(out,registration,inherited)
    shutil.copy2(Path(registration['fixture']['bundle'])/'storage_policy.json',out/'storage_policy.json')
    write(out/'storage_selection.json',{'requested':str(args.output.resolve()),'actual':str(out),'reserve_bytes':FLOOR})
    # Archive only once at campaign start; no repeated runtime-property scans.
    snapshot=out/'registered_source_snapshot';snapshot.mkdir()
    for name in registration['algorithm_sha256']:
        relative=Path(name).relative_to(ROOT.parent);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    shutil.copytree(registration['fixture']['bundle'],out/'registered_inputs/fixture')
    for model in ('model','model_closed'):
        target=out/'registered_inputs'/model;target.mkdir()
        for name in ('arx.yml','arx.urdf','model_manifest.json'):
            shutil.copy2(Path(registration['fixture']['model_reference'])/model/name,target/name)
    with lock(out/'campaign.lock'):
        try:
            with lock(Path(registration['fixture']['mailbox'])/'p0_execution.lock'):
                ready=preflight(registration);write(out/'preflight.json',ready)
                collect(ledger,Runtime,ready['status']['session_id'])
        except BaseException as exc:
            ledger.aggregate();ledger.progress('stopped_on_error',error_type=type(exc).__name__,error=str(exc));raise
        encode_results(ledger)


if __name__=='__main__':main()
