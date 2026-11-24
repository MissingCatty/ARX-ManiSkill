"""One bounded cold-start continuation of the completely unissued36slots."""
import argparse
import copy
from pathlib import Path
import shutil
import sqlite3
import time
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from .prepare_v2 import predecessors
from . import campaign as launcher


def unissued(campaign):
    campaign=Path(campaign);reg=read(campaign/'registration.json')
    if read(campaign/'state.json')['status']!='fault' or reg.get('startup_continuation_of'):
        raise ValueError('Only the original unused cold start can continue once')
    if read(campaign/'STOP')['reason']!='runtime startup exceeded 600 seconds':
        raise ValueError('This is not the registered cold-start timeout')
    counts={}
    for t in reg['seed_trials']:
        path=campaign/'trial_ledgers'/(t['trial_id']+'.sqlite3')
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        try:counts[t['trial_id']]=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        finally:db.close()
    if len(counts)!=36 or any(counts.values()) or list((campaign/'groups').glob('*/trials/*/episodes/*/requests.json')):
        raise ValueError('An issued, claimed or uncertain slot cannot be replayed')
    return {'pass':True,'slot_claim_counts':counts,'physical_attempts':0,'commands':0,
        'scope':'All36slots remain unissued; preserves original global budget and deadline'}


def startup_expired(group,elapsed,limit):
    if limit!=1800:raise ValueError('Only this registered1800second cold start is supported')
    path=Path(group)/'initialization.json'
    return elapsed>limit and (read(path).get('phase') if path.exists() else None)!='ready_for_uniform_warmup'


def prepare(previous,out,checks):
    previous,out,checks=map(lambda p:Path(p).resolve(),(previous,out,checks))
    if out.exists():raise FileExistsError('Preserve old STOP and registration')
    check=read(checks);assert check['pass'];verify_files(check['sources'])
    remaining=unissued(previous);idle=predecessors([previous]);old=read(previous/'registration.json')
    verify_files({**old['execution_source_files'],**old['dependencies']})
    if time.time()>=old['deadline_unix']:raise RuntimeError('Original collection deadline expired')
    out.mkdir(parents=True);space(out)
    write(out/'state.json',{'status':'preparing','physical_trials':0})
    source=out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    old_source=Path(old['source_snapshot'])
    shutil.copytree(old_source,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
    extra=source/'scripts/curobo_arx/nearfield/physical_startup.py';shutil.copy2(__file__,extra)
    reg=copy.deepcopy(old)
    execution={str(source/Path(p).relative_to(old_source) if Path(p).is_relative_to(old_source) else p):h for p,h in old['execution_source_files'].items()}
    execution[str(extra)]=sha(extra)
    for name in ('storage_policy.json','advisor20.md','catalog.json','offline_checks.json'):
        shutil.copy2(previous/name,out/name)
    for trial in reg['seed_trials']:
        source_db=previous/'trial_ledgers'/(trial['trial_id']+'.sqlite3')
        dest=out/'trial_ledgers'/source_db.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source_db,dest)
    for name,path in old['reference_files'].items():
        dest=out/'references'/(name+'.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
        reg['reference_files'][name]=str(dest);reg['dependencies'][str(dest)]=sha(dest)
    write(out/'startup_continuation_receipt.json',{'predecessor':str(previous),'remaining':remaining,'idle':idle,
        'startup_seconds_before':600,'startup_seconds_after':1800,'attempt_seconds_unchanged':old['attempt_seconds'],
        'deadline_unix_unchanged':old['deadline_unix'],'global_physical_budget':36,'consumed_physical_budget':0,
        'old_STOP_sha256':sha(previous/'STOP'),'scope':'Waiting for unchanged scene collision cooking; no physics, source runtime, masks, trials or success changes'})
    shutil.copy2(checks,out/'startup_offline_checks.json')
    for p in [previous/n for n in ('registration.json','STOP','state.json','summary.json','FAULT.json','startup_timeout_no_dispatch.json')]+[out/'startup_continuation_receipt.json',out/'startup_offline_checks.json']:
        reg['dependencies'][str(p)]=sha(p)
    reg.update(created_unix=time.time(),source_snapshot=str(source),execution_source_files=execution,
        source_identity=digest(execution),startup_continuation_of=str(previous),startup_timeout_seconds=1800,
        coordinator_module='scripts.curobo_arx.nearfield.physical_startup',
        global_physical_budget=36,consumed_physical_budget=0,automatic_startup_retries=0)
    assert digest(reg['seed_trials'])==digest(old['seed_trials'])
    write(out/'registration.json',reg)
    for n in ('sources.json','research.md','research_log.md'):shutil.copy2(previous/n,out/n)
    with (out/'research_log.md').open('a') as f:f.write('\nSingle cold-start continuation: old36slots allunissued, originaldeadline and36globalbudget preserved; startupwait600->1800s only.\n')
    verify_files({**execution,**reg['dependencies']})
    write(out/'state.json',{'status':'prepared','scheduled_slots':36,'physical_trials':0})
    return str(out)


def run(campaign):
    campaign=Path(campaign).resolve();reg=read(campaign/'registration.json')
    if not reg.get('startup_continuation_of') or reg['startup_timeout_seconds']!=1800:
        raise ValueError('Missing explicit one-time cold-start registration')
    unissued(Path(reg['startup_continuation_of']))
    previous=launcher.startup_expired
    launcher.startup_expired=lambda group,elapsed:startup_expired(group,elapsed,reg['startup_timeout_seconds'])
    try:launcher.run(campaign,reg['requested_devices']['compute_uuid'],reg['requested_devices']['sim_uuid'])
    finally:launcher.startup_expired=previous


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('prepare');q.add_argument('--previous',required=True);q.add_argument('--out',required=True);q.add_argument('--checks',required=True)
    q=s.add_parser('run');q.add_argument('campaign');a=p.parse_args()
    if a.action=='prepare':print(prepare(a.previous,a.out,a.checks))
    else:run(a.campaign)
