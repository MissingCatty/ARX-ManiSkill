"""Bounded rolling seed batches, with immutable recovery and no command replay."""
import argparse
import collections
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import traceback

from scripts.curobo_arx.data_factory.common import read,write,sha,digest,verify_files,space,FLOOR
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.watchdog_recovery import inspect_records
from scripts.curobo_arx.data_factory.storage import register_bulk_storage
from .contract import source_seed,perturbed_entry,POLICY,ANCHOR_WAIT,audit_episode

OFFSETS=(([.01,0.,0.],[0.,0.,5.]),([0.,-.02,0.],[5.,0.,0.]))
TERMINAL={'committed','failed','visual_rejected','duplicate_excluded'}


def trial_rows(path):
    if not Path(path).is_file():return []
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Augmentation trial ledger is corrupt')
        return [dict(r) for r in db.execute('SELECT * FROM attempts')]
    finally:db.close()


def plan_trials(seeds):
    trials=[]
    for seed_id,seed in seeds.items():
        cases=[('control',i,[0.,0.,0.],[0.,0.,0.]) for i in range(2)]
        for index,(delta,omega) in enumerate(OFFSETS):
            for mode in (('stitch','direct') if index==0 else ('direct','stitch')):
                cases.append((mode,index,delta,omega))
        for mode,index,delta,omega in cases:
            trials.append({'trial_id':f'{len(trials):04d}__{seed["object_id"]}__{mode}__{index}',
                'seed_id':seed_id,'object_id':seed['object_id'],'renderer':seed['renderer'],
                'policy':mode,'repeat':index,'placement':seed['placement'],
                'entry_world':perturbed_entry(seed,delta,omega),
                'perturbation':{'source_eef_translation_m':delta,'source_eef_rotation_vector_degrees':omega}})
    return trials


def select_seeds(parent,excluded,limit):
    db=sqlite3.connect('file:'+str(parent/'ledger.sqlite3')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:rows=[dict(r) for r in db.execute("SELECT * FROM attempts WHERE state='committed' ORDER BY started,id")]
    finally:db.close()
    selected={};objects=set();rejected=[];deferred=[]
    for row in rows:
        if row['id'] in excluded or 'egad' in row['object_id'].lower():continue
        try:seed=source_seed(row)
        except ValueError as error:
            message=str(error)
            if not any(t in message for t in ('approach start differs from actual source entry',
                                               'Choose a measured stationary source entry')):
                raise RuntimeError('Accepted source record integrity fault: '+message)
            rejected.append({'id':row['id'],'reason':message});continue
        if seed['object_id'] in objects:deferred.append(seed);continue
        selected[seed['seed_id']]=seed;objects.add(seed['object_id'])
        if len(selected)>=limit:break
    if len(selected)<limit:
        for seed in deferred:
            selected[seed['seed_id']]=seed
            if len(selected)>=limit:break
    return selected,rejected


def freeze_runtime(parent,envelope):
    """Freeze once before service launch; never import experimental worker code."""
    parent,envelope=Path(parent),Path(envelope);old=read(parent/'registration.json')
    verify_files({**old['execution_source_files'],**old['dependencies']})
    project=Path(__file__).resolve().parents[3]
    source=envelope/'source_snapshot/COOLAS-ARX';source.parent.mkdir(parents=True,exist_ok=True)
    if source.exists():raise FileExistsError('Never replace augmentation runtime snapshot')
    previous=Path(old['source_snapshot'])
    shutil.copytree(previous,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(project.parent/'curobo',target_is_directory=True)
    package=Path('scripts/curobo_arx/seed_augmentation')
    if (source/package).exists():shutil.rmtree(source/package)
    shutil.copytree(project/package,source/package,ignore=shutil.ignore_patterns('__pycache__'))
    extras=[Path('scripts/curobo_arx/data_factory/'+name) for name in
            ('watchdog.py','augmentation_watchdog.py','watchdog_recovery.py')]
    for relative in extras:shutil.copy2(project/relative,source/relative)
    execution={str(source/Path(p).relative_to(previous)) if Path(p).is_relative_to(previous) else p:v
               for p,v in old['execution_source_files'].items()}
    template=copy.deepcopy(old)
    for relative in [*(p.relative_to(source) for p in (source/package).glob('*.py')),*extras]:
        value=sha(source/relative);execution[str(source/relative)]=value
        template['source_files'][str(project/relative)]=value
    for k in ('resume','lineage','bulk_storage','controlled_resource_pause','resource_restart'):template.pop(k,None)
    template.update(source_snapshot=str(source),execution_source_files=execution,
                    source_identity=digest(template['source_files']),project_root=str(project))
    for renderer,path in list(template['reference_files'].items()):
        target=envelope/'references'/(renderer+'.json');target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
        template['reference_files'][renderer]=str(target);template['dependencies'][str(target)]=sha(target)
    write(envelope/'template.json',template)
    return template


def seal_batch(cfg,out,seeds,trials,parent,controls=None,recovery=None):
    if out.exists():raise FileExistsError('Never replace an augmentation registration')
    template=read(Path(cfg['home'])/'template.json');verify_files(template['execution_source_files'])
    reg=copy.deepcopy(template);out.mkdir(parents=True)
    for seed in seeds.values():reg['dependencies'].update(seed['files'])
    for seed in seeds.values():
        episode=Path(seed['episode'])
        verify_files({str(episode/relative):expected
            for relative,expected in read(episode/'quality_receipt.json')['files'].items()})
    quota={**reg['quotas'],'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,
           'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1}
    for trial in trials:
        catalog=copy.deepcopy(reg['catalog'])
        for asset in catalog['objects']:
            asset['collection_enabled']=asset['object_id']==trial['object_id']
            if asset['collection_enabled']:asset['collection_placements']=[trial['placement']]
        path=out/'trial_ledgers'/(trial['trial_id']+'.sqlite3');path.parent.mkdir(exist_ok=True)
        ledger=Ledger(path);ledger.initialize(catalog,{**quota,'first_object':trial['object_id']});ledger.close()
    policy=Path(cfg['storage_policy']);shutil.copy2(policy,out/'storage_policy.json')
    reg['dependencies'][str(policy)]=sha(policy)
    reg.update(created_unix=time.time(),deadline_unix=cfg['deadline_unix'],
        wall_hours=max(0.,(cfg['deadline_unix']-time.time())/3600),stage='success_seed_augmentation_production',
        production=True,production_adoption=False,envs_per_gpu=1,object_block_attempts=1,
        candidate_backend='success_seed_replay',seeds=seeds,seed_trials=trials,join_policy=POLICY,
        anchor_wait_policy=ANCHOR_WAIT,inherited_control_results=controls or [],
        max_physical_trials=len(trials),max_seed_model_replay_checks=2*len(seeds),
        no_automatic_retry=True,process_recovery=True,worker_startup_timeout_seconds=600,
        minimum_available_host_memory_bytes=40*1024**3,preparation_complete=True,
        source_parent=str(parent),augmentation_envelope=str(Path(cfg['home']).resolve()),
        global_slot_cap=cfg['max_slots'],global_budget_not_reset=True,
        seed_control_activation='Two admitted controls per seed, including verified inherited controls; never infer unknown controls',
        existing_production_remains_running=True,
        workflow='Perturb H; stitch H to recorded S then replay the unchanged successful tail, or compare direct H to fixed G; fresh ARX execution and observations')
    if recovery:
        reg['augmentation_recovery']=recovery
        for path,value in recovery.get('dependencies',{}).items():reg['dependencies'][path]=value
    # Routing estimate does not change the 1-GiB admission floor.
    free=shutil.disk_usage(out).free
    if free<FLOOR+3*1024**3+len(trials)*256*1024**2:
        reg['bulk_storage']=register_bulk_storage(out,Path(cfg['bulk_parent'])/out.name)
    verify_files({**reg['execution_source_files'],**reg['dependencies']})
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    write(out/'state.json',{'status':'prepared','scheduled_slots':len(trials),'physical_trials':0})
    write(out/'production_registration_receipt.json',{'new_authorization':cfg['authorized_by'],
        'execution_sources_pinned':True,'fixed_source_tail_and_task_gates_unchanged':True,
        'source_seed_ids':list(seeds),'inherited_controls':len(controls or []),
        'model_checks_are_new_in_this_registration':True,'deadline_not_extended':cfg['deadline_unix']})
    return {'status':'prepared','campaign':str(out),'seed_ids':list(seeds),'slots':len(trials),
            'new_allocated_slots':0 if recovery else len(trials)}


def prepare(request,cfg):
    native=read(cfg['native_watchdog_state']);parent=Path(native['campaign'])
    current=read(parent/'registration.json');template=read(Path(cfg['home'])/'template.json')
    for k in ('catalog','render_configs','render_camera_clipping','frozen','visual_geometry_policy'):
        if current.get(k)!=template.get(k):raise ValueError('Frozen native configuration changed: '+k)
    count=min(cfg['seeds_per_batch'],request['remaining_slots']//6)
    if count<1:return {'status':'budget_complete'}
    seeds,rejected=select_seeds(parent,set(request['excluded_seed_ids']),count)
    if not seeds:return {'status':'waiting_seeds','excluded_inputs':rejected}
    result=seal_batch(cfg,Path(request['out']),seeds,plan_trials(seeds),parent)
    result['excluded_inputs']=rejected
    return result


def gather(campaign):
    """Reconcile only durable real records after all old Worlds have exited."""
    reg=read(campaign/'registration.json');done={}
    for p in campaign.glob('groups/*/seed_results.json'):
        for row in read(p)['trials']:
            if row['trial_id'] in done:raise ValueError('Duplicate trial result')
            done[row['trial_id']]=row
    additions=[];started=set();physical=0;unknown=[]
    for trial in reg['seed_trials']:
        ident=trial['trial_id'];dbfile=campaign/'trial_ledgers'/(ident+'.sqlite3');rows=trial_rows(dbfile)
        if len(rows)>1:raise ValueError('A seed slot dispatched more than one attempt')
        began=ident in done or bool(rows) or bool(list(campaign.glob('groups/*/trials/'+ident)))
        if began:started.add(ident)
        physical+=sum(row['physical'] for row in rows)
        with tempfile.TemporaryDirectory(prefix='arx_seed_recovery_check_') as tmp:
            (Path(tmp)/'ledger.sqlite3').symlink_to(dbfile)
            inspect_records(Path(tmp))
        terminal=bool(rows) and rows[0]['state'] in TERMINAL
        if terminal and rows[0].get('episode'):
            episode=Path(rows[0]['episode'])
            audit_episode(episode,reg['seeds'][trial['seed_id']],trial['policy'])
            if ident not in done:
                evidence=read(episode/'physical_receipt.json')
                row={**trial,'completed':True,'success':bool(evidence['pass']),
                    'admitted':rows[0]['state']=='committed','physical_dispatches':sum(r['physical'] for r in rows),
                    'outcome':rows[0]['outcome'],'attempts':rows,'physical_evidence':[evidence],
                    'recovered_from_terminal_ledger':True,'source_ledger_sha256':sha(dbfile)}
                done[ident]=row;additions.append(row)
        elif rows and rows[0]['state']=='rejected' and not rows[0]['physical'] and ident not in done:
            # A durable pre-dispatch rejection is a known outcome, never a
            # reason to demand post-execution files or to replay this slot.
            row={**trial,'completed':True,'success':False,'admitted':False,
                 'physical_dispatches':0,'outcome':rows[0]['outcome'],'attempts':rows,
                 'recovered_from_terminal_ledger':True,'source_ledger_sha256':sha(dbfile)}
            done[ident]=row;additions.append(row)
        elif began and ident not in done:
            unknown.append(ident)
    if additions:
        target=campaign/'reconciled_results.json'
        value={'trials':additions,'no_success_inferred_from_unknown':True}
        if target.exists() and read(target)!=value:raise ValueError('Prior reconciliation changed')
        if not target.exists():write(target,value)
    return reg,done,started,physical,unknown


def finish(request,cfg):
    from scripts.curobo_arx.data_factory.augmentation_watchdog import prove_exited
    campaign=Path(request['campaign'])
    proof=prove_exited(campaign)
    if not proof:raise RuntimeError('Old augmentation processes or Worlds have not exited')
    verify_files({**read(campaign/'registration.json')['execution_source_files'],
                  **read(campaign/'registration.json')['dependencies']})
    reg,done,started,physical,unknown=gather(campaign)
    from .report import export
    exported=export(campaign,videos=False)
    controls={r['trial_id']:r for r in reg.get('inherited_control_results',[])}
    for ident,row in done.items():
        if row['policy']=='control':controls[ident]=row
    for ident in unknown:
        trial=next(t for t in reg['seed_trials'] if t['trial_id']==ident)
        if trial['policy']=='control':controls[ident]={**trial,'completed':False,'admitted':False,'success':False,'status':'unknown'}
    remaining=[t for t in reg['seed_trials'] if t['trial_id'] not in started]
    state=read(campaign/'state.json') if (campaign/'state.json').exists() else {}
    if state.get('model_activation_rejected'):remaining=[]
    # A clean completion must account for every slot. Missing physical/record
    # outcomes remain unknown; only never-started slots may be continued.
    for ident in done:
        if ident not in {t['trial_id'] for t in reg['seed_trials']}:raise ValueError('Unregistered seed result')
    result={'status':'finished','campaign':str(campaign),'physical_attempts':physical,
        'admitted':exported['total_admitted'],'admitted_by_policy':exported['admitted_by_policy'],
        'unknown_slots':unknown,'remaining_slots':len(remaining),
        'dataset_views':exported['dataset_campaign_views'],'never_replay_started_slots':True}
    if remaining and request.get('resume') and time.time()<cfg['deadline_unix']:
        needed={t['seed_id'] for t in remaining};seeds={k:v for k,v in reg['seeds'].items() if k in needed}
        inheritance=[r for r in controls.values() if r['seed_id'] in needed]
        proof={'parent':str(campaign),'started_slots_not_replayed':sorted(started),
               'unknown_slots_not_credited':unknown,'dependencies':{str(campaign/'registration.json'):sha(campaign/'registration.json')}}
        proof['dependencies'][str(campaign/'watchdog_exit_proof.json')]=sha(campaign/'watchdog_exit_proof.json')
        for p in campaign.glob('groups/*/seed_results.json'):proof['dependencies'][str(p)]=sha(p)
        for p in campaign.glob('trial_ledgers/*.sqlite3'):proof['dependencies'][str(p)]=sha(p)
        child=seal_batch(cfg,Path(request['out']),seeds,remaining,reg['source_parent'],inheritance,proof)
        result['continuation']=child
    return result


def job(path):
    request=read(path);cfg=read(request['config']);result=Path(request['result'])
    try:
        if sha(Path(cfg['home'])/'template.json')!=cfg['template_sha256']:
            raise ValueError('Registered input/source changed: augmentation template')
        if time.time()>=cfg['deadline_unix'] and request['kind']=='prepare':value={'status':'budget_complete'}
        elif request['kind']=='prepare':value=prepare(request,cfg)
        elif request['kind']=='finish':value=finish(request,cfg)
        else:raise ValueError('Unknown augmentation helper job')
    except BaseException as error:
        value={'status':'blocked','error':repr(error),'traceback':traceback.format_exc()}
    write(result,value)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    job(Path(parser.parse_args().request))
