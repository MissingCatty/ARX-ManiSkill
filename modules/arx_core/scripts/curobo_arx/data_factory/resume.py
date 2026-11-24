"""Append a production registration after proving the previous virtual World exited.

Never remove old STOPs, modify a parent ledger, or replay an interrupted command.
Only a copied ledger receives recovery transitions; successes keep original paths.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from .common import read,sha,verify_files,write
from .ledger import Ledger
from .prepare import prepare


def require_exit_proof(proof):
    if any(proof.get(k) is not True for k in ('old_process_exited','environments_idle','containers_absent')) or not proof.get('evidence'):
        raise ValueError('Recovery requires verified process exit and destroyed/idle virtual environments')


def exit_proof(parent, *, reconcile_late_shutdown=False):
    parent=Path(parent).resolve();state=read(parent/'state.json')
    if state['status'] not in ('fault','stopped','budget_exhausted','quota_complete') or (state.get('unresolved_groups') and not reconcile_late_shutdown):
        raise ValueError('Parent coordinator is active or its child exit is unresolved')
    processes=[]
    paths=list((parent/'workers').glob('*/process.json'))+list((parent/'group_registrations').glob('*.process.json'))
    if (parent/'coordinator_process.json').exists():paths.append(parent/'coordinator_process.json')
    if not paths:raise ValueError('No parent execution registration to verify')
    for path in paths:
        meta=read(path);proc=Path('/proc')/str(meta['pid'])/'cmdline'
        args=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else ''
        owned=any(module in args for module in ('data_factory','sdg_stage1')) and parent.name in args
        if owned:raise ValueError('Parent execution process still exists: '+str(meta['pid']))
        processes.append({'registration':str(path),'pid':meta['pid'],'owned_process_alive':False,
                          'pid_present_but_not_parent':bool(args)})
    containers=[]
    for path in (parent/'group_registrations').glob('*.process.json'):
        name=read(path)['container']
        result=subprocess.run(['podman','container','exists',name],capture_output=True,text=True,timeout=15)
        if result.returncode!=1:raise ValueError('Parent container is present or absence cannot be verified: '+name)
        containers.append({'name':name,'absent':True})
    late_receipts=[]
    for group in state.get('unresolved_groups',[]):
        if not any(c['name']=='arx-data-factory-'+group for c in containers):
            raise ValueError('Late shutdown group has no registered container absence proof: '+group)
        directory=parent/'groups'/group
        block,shutdown=directory/'block_result.json',directory/'shutdown_receipt.json'
        if not block.is_file() or not shutdown.is_file():raise ValueError('Late shutdown has missing durable receipts: '+group)
        b,s=read(block),read(shutdown)
        if b.get('idle') is not True or any(s.get(k) is not True for k in ('all_trajectories_idle','raw_writer_drained_and_closed','ledger_closed')):
            raise ValueError('Late shutdown did not confirm idle and drained writers: '+group)
        late_receipts.append({'group':group,'block_result_sha256':sha(block),'shutdown_receipt_sha256':sha(shutdown),
            'shutdown_wall':s['wall'],'old_unresolved_flag_preserved':True})
    no_world_evidence=None
    if not containers:
        # A coordinator may fail before its first simulator dispatch. Require
        # positive launch/exit evidence, no group registration at all and no
        # possibly orphaned factory container; absence of receipts alone is
        # never sufficient to infer an idle World.
        if not (parent/'coordinator_process.json').exists() or not (parent/'launch.json').exists():
            raise ValueError('No coordinator/launch evidence for a pre-simulator failure')
        if list((parent/'group_registrations').glob('*.json')):
            raise ValueError('Simulator dispatch exists without virtual environment ownership evidence')
        result=subprocess.run(['podman','ps','-a','--format','{{.Names}}'],capture_output=True,text=True,timeout=15)
        if result.returncode or any(n.startswith('arx-data-factory-group_') for n in result.stdout.splitlines()):
            raise ValueError('Cannot exclude an orphaned factory simulator container')
        no_world_evidence={'no_simulator_group_registration':True,'owned_coordinator_exited':True,
            'no_factory_simulator_container':True,'launch_sha256':sha(parent/'launch.json')}
    return {'checked_unix':time.time(),'old_process_exited':True,'environments_idle':True,'containers_absent':True,
        'idle_basis':('No virtual World was dispatched; coordinator/workers exited and no factory simulator container exists.'
            if no_world_evidence else 'Original virtual World is destroyed: registered processes exited and all owned containers absent. Pre-crash convergence is unverified.'),
        'evidence':{'parent_state':state,'state_sha256':sha(parent/'state.json'),'STOP_sha256':sha(parent/'STOP'),
                    'processes':processes,'containers':containers,'pre_simulator_exit':no_world_evidence,
                    'late_shutdown_receipts':late_receipts}}


def copy_and_recover(source,destination,proof):
    require_exit_proof(proof)
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Recovery must use a new ledger')
    if destination.exists():
        check=sqlite3.connect('file:'+str(destination)+'?mode=ro',uri=True)
        try:
            if check.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] or check.execute('SELECT COUNT(*) FROM goals').fetchone()[0]:
                raise ValueError('Refuse to overwrite a nonempty destination ledger')
        finally:check.close()
    original=sha(source)
    src=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True)
    dest=sqlite3.connect(destination)
    try:src.backup(dest)
    finally:dest.close();src.close()
    ledger=Ledger(destination)
    try:
        before=ledger.rows()
        owners={r['owner'] for r in before if r['state'] in ('claimed','executing','writing','interrupted')}
        owners|={r[0] for r in ledger.db.execute('SELECT owner FROM object_leases')}
        for owner in sorted(owners):ledger.recover(owner,proof)
        after=ledger.rows()
    finally:ledger.close()
    if sha(source)!=original:raise RuntimeError('Parent ledger changed during read-only recovery')
    return {'parent_ledger_sha256':original,'inherited_attempt_ids':[r['id'] for r in before],
        'inherited_success_ids':[r['id'] for r in before if r['state']=='committed'],
        'inherited_physical_attempts':sum(r['physical'] for r in before),
        'recovered':[{'id':a['id'],'prior_state':b['state'],'state':a['state'],'physical':a['physical'],
                      'meaning':'execution_outcome_unknown' if a['physical'] else 'interrupted_before_physical_dispatch'}
                     for b,a in zip(before,after) if b['state']!=a['state']]}


def resume(parent,out,*,envs=1,block=4):
    parent,out=Path(parent).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Create a new continuation; never overwrite old results')
    lock=(parent/'coordinator.lock').open('r')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        proof=exit_proof(parent)
        old=read(parent/'registration.json');launch=read(parent/'launch.json')
        deadline=old.get('deadline_unix',launch['stop_unix'])
        if time.time()>=deadline:raise ValueError('Original wall budget expired; a new explicitly budgeted campaign is required')
        verify_files(old['dependencies'])
        source=sqlite3.connect('file:'+str(parent/'ledger.sqlite3')+'?mode=ro',uri=True);source.row_factory=sqlite3.Row
        try:
            accepted=[dict(r) for r in source.execute("SELECT * FROM attempts WHERE state='committed'")]
        finally:source.close()
        for row in accepted:
            episode=Path(row['episode']);receipt=read(episode/'quality_receipt.json');committed=__import__('json').loads(row['receipt'])
            if sha(episode/'quality_receipt.json')!=committed['quality_sha256'] or sha(episode/'physical_receipt.json')!=committed['physical_sha256']:
                raise ValueError('Committed receipt changed: '+row['id'])
            verify_files({str(episode/p):s for p,s in receipt['files'].items()})
        reg=prepare(out,envs=envs,wall_hours=old['wall_hours'],block=block,
            max_attempts=old['quotas']['max_physical_attempts'],first_object=old.get('first_object'))
        for key in ('catalog','quotas','render_configs','dependencies'):
            if reg[key]!=old[key]:raise ValueError('Continuation changed frozen data or budget: '+key)
        inherited=copy_and_recover(parent/'ledger.sqlite3',out/'ledger.sqlite3',proof)
        write(out/'exit_proof.json',proof);write(out/'recovery_receipt.json',inherited)
        reg.update(deadline_unix=deadline,campaign_started_unix=old.get('campaign_started_unix',launch['started_unix']),
            lineage=old.get('lineage',[])+[{'campaign':str(parent),'registration_sha256':sha(parent/'registration.json'),
                'state_sha256':sha(parent/'state.json'),'status':read(parent/'state.json')['status']}],
            resume={'parent':str(parent),'proof_sha256':sha(out/'exit_proof.json'),
                'recovery_receipt_sha256':sha(out/'recovery_receipt.json'),**inherited,
                'runtime_change':f'{envs} environment(s) per GPU and {block}-attempt object blocks; asynchronous timed frame walking replaced by simulator-main-thread diagnostic snapshots. Real ARX pressure comparison reproduces a crash only with the asynchronous sampler; long-run stability remains to be measured.',
                'frozen_unchanged':'Robot, physics, all initial conditions, cameras/renderers, weights, success and quality gates'},
            wall_budget_scope='Original campaign start/deadline retained across continuation; no counter or budget reset')
        write(out/'registration.json',reg)
        for row in accepted:
            source=parent/'videos'/(row['id']+'.mp4')
            if source.exists():
                target=out/'videos'/source.name;target.parent.mkdir(exist_ok=True)
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
        write(out/'state.json',{'status':'prepared','successes':len(accepted),'new_successes':0,'quota_complete':False})
        return {'campaign':str(out),'inherited_successes':len(accepted),
                'retained_interrupted':len(inherited['recovered']),'deadline_unix':deadline}
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from',dest='parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--envs-per-gpu',type=int,default=1);p.add_argument('--object-block-attempts',type=int,default=4)
    args=p.parse_args();print(resume(args.parent,args.out,envs=args.envs_per_gpu,block=args.object_block_attempts))
