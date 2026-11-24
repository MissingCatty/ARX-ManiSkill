"""CPU-only recovery of a finished campaign's cross-environment report audit."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,verify_files
from .prepare_v2 import predecessors
from .main48_report import report

CHANGES=('training/nearfield_arx/receipt_numbers.py','training/nearfield_arx/stage_verification.py',
         'training/nearfield_arx/test_settling_receipts.py','scripts/curobo_arx/nearfield/main48_report_recovery.py')


def prepare(campaign,out):
    campaign=Path(campaign).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve each offline report registration')
    old=read(campaign/'registration.json');summary=read(campaign/'summary.json')
    if read(campaign/'state.json')['status']!='complete' or (campaign/'STOP').exists():raise ValueError('Physical execution must already be complete')
    if summary['incomplete_accounting'] or summary['unexecuted'] or len(summary['trials'])!=19:
        raise ValueError('Expected all19 authorized native slots, without a new unknown')
    if [t['trial_id'] for t in summary['trials']]!=[t['trial_id'] for t in old['seed_trials']]:raise ValueError('Native schedule changed')
    if not (campaign/'coordinator.log').read_text().rstrip().endswith('RuntimeError: C continuation conceals original arrival outcome'):
        raise ValueError('Unexpected original reporting fault')
    idle=predecessors([campaign]);out.mkdir(parents=True)
    write(out/'state.json',{'status':'preparing','physical_trials':0})
    write(out/'original_exit_verification.json',idle)
    cmd=read(campaign/'offline_checks.json')['command'];cmd=cmd[:-1]+['training.nearfield_arx.test_settling_receipts','-v']
    check=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=180)
    (out/'host_tests.log').write_text(check.stdout+check.stderr)
    write(out/'host_tests.json',{'pass':check.returncode==0,'command':cmd,'log_sha256':sha(out/'host_tests.log'),'physical_trials':0})
    if check.returncode:raise RuntimeError('Offline regression checks failed')
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for n in ('tmp','curobo'):(source.parent/n).symlink_to(ROOT.parent/n,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    changed={}
    for n in CHANGES:
        dest=source/n;changed[n]={'before':sha(dest) if dest.exists() else None,'after':sha(ROOT/n)}
        shutil.copy2(ROOT/n,dest);execution[str(dest)]=sha(dest)
    for p,h in old['execution_source_files'].items():
        rel=str(Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p
        dest=source/rel if Path(p).is_relative_to(prior) else Path(p)
        if rel not in CHANGES and sha(dest)!=h:raise ValueError('Unapproved offline-source change')
    def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before=funcs(prior/'training/nearfield_arx/receipt_numbers.py');after=funcs(source/'training/nearfield_arx/receipt_numbers.py')
    if any(after[k]!=v for k,v in before.items()):raise ValueError('Prior numerical comparison rules changed')
    original=(prior/'training/nearfield_arx/stage_verification.py').read_text()
    expected=original.replace('import same_pose_diagnostics\n','import same_pose_diagnostics, same_settling_diagnostics\n').replace(
        "        if digest(original) != digest(read(episode / (label + '_settling.json'))) or original['accepted']:\n",
        "        recorded = read(episode / (label + '_settling.json'))\n"
        "        if digest(proof['original_settling']) != digest(recorded):\n"
        "            raise RuntimeError('Continuation shadow copy differs from original settling receipt')\n"
        "        if not same_settling_diagnostics(original, recorded) or original['accepted']:\n")
    if (source/'training/nearfield_arx/stage_verification.py').read_text()!=expected:raise ValueError('Audit changed beyond the registered scalar comparison')
    proof={'changes':changed,'scope':'Read-only original19/native plus22/carried outcome and video audit',
        'only_recomputed_settling_orientation_scalar_allows_one_adjacent_float':True,
        'original_receipt_and_shadow_copy_exact':True,'all_decisions_other_measurements_requests_hashes_and_actions_exact':True,
        'execution_evaluator_and_all_physical_thresholds_byte_identical':True,
        'physical_trials':0,'new_training_updates':0,'old_failed_report_and_six_partial_videos_retained':True}
    write(out/'source_change_receipt.json',proof)
    command=['podman','run','--pull=never','--rm','-i','--network=none','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import os,json,sys,unittest;p=json.load(sys.stdin);os.environ["ARX_TEST_FIXTURE_ROOT"]=p["fixture_root"];'
        't=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromName("training.nearfield_arx.test_settling_receipts"));'
        'print(json.dumps({"tests":t.testsRun,"pass":t.wasSuccessful()}));sys.exit(not t.wasSuccessful())']
    probe=subprocess.run(command,input=json.dumps({'fixture_root':str(ROOT)}),capture_output=True,text=True,timeout=120)
    write(out/'isaac_cpu_tests.json',{'pass':probe.returncode==0,'command':command,'stdout':probe.stdout,'stderr':probe.stderr,
        'GPU_or_World_started':False,'image_downloaded':False})
    if probe.returncode:raise RuntimeError('Isaac CPU-only record regression failed')
    dependencies={str(campaign/n):sha(campaign/n) for n in ('registration.json','state.json','summary.json','coordinator.log')}
    for p in out.iterdir():
        if p.is_file() and p.name!='state.json':dependencies[str(p)]=sha(p)
    reg={'created_unix':time.time(),'campaign':str(campaign),'source_snapshot':str(source),'source_files':execution,
        'dependencies':dependencies,'physical_trials':0,'scope':'Offline report only; no scheduling, service or World',
        'original_execution_registration_sha256':sha(campaign/'registration.json'),'source_change':proof}
    write(out/'registration.json',reg);verify_files({**execution,**dependencies})
    write(out/'state.json',{'status':'prepared','physical_trials':0});return out


def run(out):
    out=Path(out).resolve();reg=read(out/'registration.json');campaign=Path(reg['campaign'])
    if read(out/'state.json')['status']!='prepared' or (out/'started.json').exists():raise ValueError('Never overwrite a prior report recovery')
    verify_files({**reg['source_files'],**reg['dependencies']});predecessors([campaign])
    write(out/'started.json',{'wall':time.time(),'registration_sha256':sha(out/'registration.json'),'physical_trials':0})
    write(out/'state.json',{'status':'verifying','physical_trials':0})
    from training.nearfield_arx import stage_recovery
    stage_recovery.install();result=report(campaign,out/'results')
    old=read(campaign/'registration.json');summary=read(campaign/'summary.json')
    expected={d['trial_id']:(d['outcome'],d['physical_success']) for d in old['carried_details']}
    expected.update({t['trial_id']:(t['outcome'],t['success']) for t in summary['trials']})
    actual={d['trial_id']:(d['outcome'],d['physical_success']) for d in result['trials']}
    if actual!=expected or not result['record_integrity_pass'] or result['unknown_or_incomplete']:
        raise ValueError('Offline report altered a native/carried result or remains unverified')
    if result['unexecuted']!=[t['trial_id'] for t in old['deferred_trials']]:raise ValueError('Deferred scope changed')
    videos=[]
    for d in result['trials']:
        video=d['video']
        if video:
            if not video['decode_pass'] or sha(video['path'])!=video['sha256']:raise ValueError('Missing, changed or undecodable wrist video')
            videos.append({'trial_id':d['trial_id'],**video})
    write(out/'video_verification.json',{'pass':True,'videos':videos,
        'scope':'Fresh videos decoded in this report; inherited decoded videos verified byte-identical to their recorded hashes'})
    idle=predecessors([campaign]);verify_files({**reg['source_files'],**reg['dependencies']})
    receipt={'pass':True,'physical_trials':0,'original_results_unchanged':True,'original_auto_report_failure_preserved':True,
        'document':str(out/'results/ACCEPTANCE.md'),'document_sha256':sha(out/'results/ACCEPTANCE.md'),
        'verification':str(out/'results/main48_verification.json'),'verification_sha256':sha(out/'results/main48_verification.json'),
        'physical_successes':result['physical_successes'],'physical_failures':result['physical_failures'],
        'infrastructure_interruptions':result['infrastructure_interruptions'],'deferred_unexecuted':result['deferred_unexecuted'],
        'complete_original48':len(result['trials'])==48,'authorized_scope_accounted':len(result['trials'])+len(result['deferred_unexecuted'])==48,
        'videos':len(videos),'old_execution_exit':idle}
    write(out/'completion.json',receipt)
    pointer=campaign/'recovered_report.json'
    if pointer.exists():raise FileExistsError('Preserve previous report-recovery pointer')
    write(pointer,{'recovery':str(out),'completion_sha256':sha(out/'completion.json'),**receipt})
    write(out/'state.json',{'status':'complete','physical_trials':0,'finished_unix':time.time()});return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('out');p.add_argument('--campaign');a=p.parse_args()
    print(prepare(a.campaign,a.out) if a.action=='prepare' else run(a.out))
