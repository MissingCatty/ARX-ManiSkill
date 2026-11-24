"""Per-object static acquisition can overlap unrelated seed preflights.

Each object has16 preregistered interface observations within its1024 budget,
then the remaining1008. Training is deliberately absent from this coordinator.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices,environment
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.watchdog import process_info,exclusive_stop
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from .advisor34_launch import owned_process,wait_ready,healthy,container_command
from .advisor36_progress import collect
from .advisor36_samples import generate,grouped_split,audit
from .advisor36_export import export,load_lines
from .resources import probe_cuda
from .physical_startup import startup_expired


def pilot_ids(rows):
    pairs={}
    for r in rows:pairs.setdefault(r['pair_id'],[]).append(r)
    chosen=[]
    for layer in ('00_05','15_30','50_65','65_90'):
        chosen.append(next(p for p,v in pairs.items() if v[0]['category']=='funnel' and v[0]['axial_layer_mm']==layer))
    for kind,count in (('boundary_position',1),('boundary_rotation',1),('grasp',2)):
        chosen.extend([p for p,v in pairs.items() if v[0]['category']==kind][:count])
    result=[r['sample_id'] for r in rows if r['pair_id'] in chosen]
    if len(result)!=16:raise ValueError('Interface probe must use exactly8 complete pairs')
    return result


def prior_observations(predecessor, rows):
    """Carry terminal ARX observations; never replay started/missing members."""
    reg=read(predecessor/'registration.json')
    groups=[Path(p) for p in reg['static_observations'].get('prior_capture_groups',[])]
    groups += sorted(g for g in (predecessor/'groups').glob('*') if (g/'completion.json').exists())
    groups=list(dict.fromkeys(groups));observed=set();files={}
    invalidated=dict(reg['static_observations'].get('prior_admission_exclusions',{}))
    table={r['sample_id']:r for r in rows}
    for group in groups:
        completion=read(group/'completion.json');shutdown=read(group/'shutdown_receipt.json')
        results=read(group/'results.json')['observations']
        if (not completion['pass'] or completion['processed']!=completion['registered'] or
            len(results)!=completion['processed'] or not shutdown['all_trajectories_idle']):
            raise RuntimeError('Prior static group lacks complete terminal/idle evidence')
        expected={str(Path(r['observation']).parent) if r.get('observation') else str(group/'observations'/r['sample_id']) for r in results}
        actual={str(p.parent) for p in group.glob('observations/*/capture_receipt.json')}
        if actual!=expected:raise RuntimeError('Prior static group/receipt union mismatch')
        for r in results:
            ident=r['sample_id'];receipt=group/'observations'/ident/'capture_receipt.json'
            if ident in observed or r['candidate']!=table[ident] or read(receipt)!=r:
                raise RuntimeError('Prior static observation identity mismatch/replay')
            if not r.get('record_integrity_pass') or r.get('physical_trajectory_executed') is not False:
                raise RuntimeError('Prior static observation lacks intact non-trajectory evidence')
            verify_files(r.get('files',{}));files.update(r.get('files',{}));observed.add(ident)
            files[str(receipt)]=sha(receipt)
            folder=receipt.parent
            for name in ('STARTED.json','IK_request.json','static_IK_receipt.json'):
                files[str(folder/name)]=sha(folder/name)
            ik=read(folder/'static_IK_receipt.json')
            if (read(folder/'STARTED.json')['sample_id']!=ident or
                read(folder/'IK_request.json')['sample_id']!=ident or
                (bool(ik['pass'])!=bool(r.get('observation')))):
                raise RuntimeError('Prior static IK/request/image event chain disagrees')
            if r.get('pass_') and not r.get('target_surface_check'):
                invalidated[ident]='Old capture lacks current actual-surface and coherent scene-base verification; retained as observed, never replayed'
        for path in [group/'completion.json',group/'shutdown_receipt.json',group/'results.json',
                     *group.glob('objects/*/frozen_*.json')]:
            if path.name.startswith('frozen_') and not read(path)['pass']:
                raise RuntimeError('Prior static frozen configuration check failed')
            files[str(path)]=sha(path)
    uncarried=[p for p in predecessor.glob('groups/*/observations/*/capture_receipt.json')
               if read(p)['sample_id'] not in observed]
    if uncarried:raise RuntimeError('Partial static groups need independent reconciliation before continuation')
    started={read(p)['sample_id'] for p in predecessor.glob('groups/*/observations/*/STARTED.json')}
    affected={table[s]['pair_id'] for s in started-observed}
    inherited=set(reg['static_observations'].get('excluded_without_replay',[]))
    excluded=[r['sample_id'] for r in rows if (r['pair_id'] in affected or r['sample_id'] in inherited)
              and r['sample_id'] not in observed]
    return {'groups':list(map(str,groups)), 'observed':sorted(observed), 'files':files,
            'started_without_terminal':sorted(started-observed), 'excluded':excluded,
            'prior_admission_exclusions':invalidated}


def prepare(top,obj,compute_index,sim_index,*,name=None,predecessor=None):
    top=Path(top).resolve();out=top/'static_objects'/(name or obj)
    if out.parent!=top/'static_objects':raise ValueError('Invalid static continuation name')
    if out.exists():raise FileExistsError('No repeated static candidate registration')
    data=collect(top)
    if obj not in data['objects']:raise RuntimeError('Object has not passed7/7 physical preflight')
    entry=data['objects'][obj];verify_files(entry['physical_evidence'])
    oldroot=Path(entry['seed_campaign']);old=read(oldroot/'registration.json')
    # The source World may still be testing other objects. This static capture
    # owns a different GPU/World and consumes only immutable terminal evidence.
    if old['requested_devices']['sim_uuid']==next(d['uuid'] for d in devices() if d['index']==sim_index):
        from .advisor31_campaign import exited
        exited(oldroot)
    out.mkdir(parents=True);space(out)
    registry={'objects':{obj:entry},'released_objects':1,'training_authorized':False}
    write(out/'canonical_registry.json',registry)
    selected=entry['geometry']['selected'];rows=generate(obj,selected['Gcanonical_object'])
    split=grouped_split(rows,evaluation_only=entry['source']['source_split']!='train')
    write(out/'candidate_quality.json',{**audit(rows),**split})
    with (out/'candidates.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    excluded=[];recovery=None;carried=[];prior_groups=[];invalidated={}
    deadline=min(time.time()+8*3600,old['deadline_unix'])
    if predecessor:
        from .advisor31_campaign import exited
        predecessor=Path(predecessor).resolve();absence=exited(predecessor)
        if not (predecessor/'STOP').exists():raise RuntimeError('Preserve old startup fault STOP')
        previous=load_lines(predecessor/'candidates.jsonl')
        if previous!=rows:raise RuntimeError('Static continuation changed the original candidate table')
        prior=prior_observations(predecessor,rows)
        excluded=prior['excluded'];carried=prior['observed'];prior_groups=prior['groups']
        invalidated=prior['prior_admission_exclusions']
        ancestor=predecessor
        while ancestor:
            prev=read(ancestor/'registration.json');deadline=min(deadline,prev['deadline_unix'])
            ancestor=Path(prev['static_continuation_of']) if prev.get('static_continuation_of') else None
        recovery={'predecessor':str(predecessor),'predecessor_absent':absence,
            'zero_completed_observations':not carried,'started_but_not_admitted':prior['started_without_terminal'],
            'completed_observations_carried':carried,'prior_capture_groups':prior_groups,
            'excluded_complete_pairs':excluded,'original1024_candidate_budget_retained':True,
            'prior_admission_exclusions':invalidated,
            'replayed_started_samples':0,'physical_grasp_attempts':0,
            'reason':'Retain original grasp-target planner treatment; add actual enabled-collider target penetration check, bind scene to post-reset base, carry terminal observations without replay',
            'files':{**prior['files'],**{str(p):sha(p) for p in [predecessor/'registration.json',predecessor/'STOP',
                      *predecessor.glob('groups/*/FAULT.json'),*predecessor.glob('groups/*/observations/*/STARTED.json')]}}}
        write(out/'source_change_continuation_receipt.json',recovery)
    eligible=[r for r in rows if r['sample_id'] not in set(excluded+carried)]
    pilot=pilot_ids(eligible);pilot_set=set(pilot)
    scheduled=[{'name':'pilot','ids':pilot},{'name':'remainder','ids':[r['sample_id'] for r in eligible if r['sample_id'] not in pilot_set]}]
    write(out/'sample_manifest.json',{'total':1024,'remaining_scheduled':len(eligible),'excluded_without_replay':excluded,
        'carried_observations':carried,'prior_admission_exclusions':invalidated,
        'phases':scheduled,'replays':0,'additional_probe_candidates':0})
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    added=[*(ROOT/'scripts/curobo_arx/nearfield').glob('advisor36_*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('runtime_advisor36_*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('test_advisor36*.py')]
    for p in added:
        dest=source/p.relative_to(ROOT);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    ds=devices();by={r['index']:r for r in ds};roles={'compute_uuid':by[compute_index]['uuid'],'sim_uuid':by[sim_index]['uuid']}
    if roles['compute_uuid']==roles['sim_uuid']:raise ValueError('Separate static planner and renderer GPUs')
    deps=dict(old['dependencies']);deps.update(entry['physical_evidence'])
    # Add an independent CPU geometry query library without changing existing
    # planner packages. Pin its exact imported binary in the new registration.
    fcl_info=subprocess.run([str(ROOT.parent/'curobo/.venv/bin/python'),'-c',
        'import fcl,json,pathlib;print(json.dumps({"version":fcl.__version__,"binary":fcl.fcl.__file__,"bundled_libraries":[str(p) for p in sorted((pathlib.Path(fcl.__file__).parent.parent/"python_fcl.libs").glob("*.so*"))]}))'],
        capture_output=True,text=True,check=True,timeout=30)
    fcl_info=json.loads(fcl_info.stdout)
    if fcl_info['version']!='0.7.0.11':raise RuntimeError('Unregistered static surface dependency')
    deps[fcl_info['binary']]=sha(fcl_info['binary'])
    for library in fcl_info['bundled_libraries']:deps[library]=sha(library)
    for p in (top/'scope_amendment_01.json',out/'canonical_registry.json',out/'candidates.jsonl',out/'sample_manifest.json',out/'candidate_quality.json'):
        deps[str(p)]=sha(p)
    if recovery:
        deps.update(recovery['files']);deps[str(out/'source_change_continuation_receipt.json')]=sha(out/'source_change_continuation_receipt.json')
    reg=copy.deepcopy(old);reg.update(stage='advisor36_static_observations',project_root=str(ROOT),source_snapshot=str(source),
        execution_source_files=execution,source_identity=digest(execution),dependencies=deps,
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor36_static',
        worker_module='scripts.curobo_arx.nearfield.advisor36_static_teacher',
        coordinator_module='scripts.curobo_arx.nearfield.advisor36_static_launch',
        created_unix=time.time(),deadline_unix=deadline,requested_devices=roles,
        max_physical_trials=0,global_physical_budget=0,seed_trials=[],training_authorized_here=False,
        no_automatic_retry=True,automatic_attempt_retries=0,automatic_startup_retries=0,
        static_continuation_of=str(predecessor) if predecessor else None,
        static_observations={'object_id':obj,'canonical_registry':str(out/'canonical_registry.json'),
            'candidate_table':str(out/'candidates.jsonl'),'candidate_count':1024,'sample_phases':scheduled,
            'remaining_scheduled':len(eligible),'excluded_without_replay':excluded,
            'prior_capture_groups':prior_groups,'carried_observations':carried,
            'prior_admission_exclusions':invalidated,
            'physics_state_setters_not_trajectories':True,'full_physics_unchanged':True,
            'sample_identity_can_advance_with_fixed_physics_time':True,'mask_source':'same-render native instance IDs',
            'IK_and_actual_FK_fit_m':.0005,'IK_and_actual_FK_fit_degrees':.2,
            'target_inclusive_open_collision_check':False,
            'original_grasp_target_exclusion_preserved':True,
            'actual_target_surface_check_required':True,'surface_query_dependency':fcl_info,
            'surface_check_scope':'Read-only actual collider vertices; conservative convex envelopes; not PhysX cooked-shape or trajectory certification',
            'GRASP_original_closed_model_check':True,
            'mirror_pair_admission':True,'training_requires_later_user_acceptance':True})
    Ledger(out/'ledger.sqlite3').close()
    shutil.copy2(oldroot/'storage_policy.json',out/'storage_policy.json')
    write(out/'registration.json',reg);write(out/'resource_plan.json',{'inventory':ds,'roles':roles})
    verify_files({**execution,**deps})
    write(out/'full_preflight.json',{'pass':True,'checked_files':len({**execution,**deps}),'observations':0,'grasp_trials':0})
    write(out/'state.json',{'status':'prepared','observations':0,'training_started':False})
    lineage=read(top/'static_lineage.json') if (top/'static_lineage.json').exists() else {}
    ancestors=lineage.get(obj,{}).get('predecessors',[])
    lineage[obj]={'current':str(out),'predecessors':list(dict.fromkeys(ancestors+([str(predecessor)] if predecessor else []))),
        'registration_sha256':sha(out/'registration.json')}
    write(top/'static_lineage.json',lineage)
    return str(out)


def run(root):
    root=Path(root).resolve();reg=read(root/'registration.json');roles=reg['requested_devices'];static=reg['static_observations']
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STARTED.json').exists() or (root/'STOP').exists():raise RuntimeError('No in-place restart/replay')
    write(root/'STARTED.json',{'wall':time.time(),'registration_sha256':sha(root/'registration.json')})
    planner=None;active=None;name=None;error=None;unresolved=False;phases=[]
    try:
        verify_files({**reg['dependencies'],**reg['execution_source_files']})
        inventory={r['uuid']:r for r in devices()}
        for role,floor in (('compute',6500),('sim',12000)):
            d=inventory[roles[role+'_uuid']]
            if d['free_mb']<floor:raise RuntimeError(role+' static GPU has insufficient free memory')
            proof=probe_cuda(d,environment(d));write(root/(role+'_cuda_preflight.json'),proof)
            if not proof['pass']:raise RuntimeError(role+' static CUDA preflight failed')
        if available_memory()<reg['minimum_available_host_memory_bytes']:raise RuntimeError('Host memory floor')
        directory=root/'workers/planner';env=environment(inventory[roles['compute_uuid']]);env.update(PYTHONPATH=reg['source_snapshot'],PYTHONUNBUFFERED='1')
        planner=owned_process([str(Path(reg['project_root']).parent/'curobo/.venv/bin/python'),'-m',reg['worker_module'],
            '--directory',str(directory),'--registration',str(root/'registration.json')],reg['source_snapshot'],env,directory)
        wait_ready(planner,directory,root)
        renderer=read(static['canonical_registry'])['objects'][static['object_id']]['source']['renderer']
        for phase in static['sample_phases']:
            healthy(root,{'planner':planner});space(root)
            if next(d['free_mb'] for d in devices() if d['uuid']==roles['sim_uuid'])<12000:
                raise RuntimeError('Static rendering GPU memory changed before new World')
            ident=phase['name'];name='arx-a36-static-'+digest(str(root))[:10]+'-'+ident
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:
                raise RuntimeError('Prior static World absence unconfirmed')
            launch=root/'group_registrations'/(ident+'.json');group=root/'groups'/ident
            write(launch,{'group_id':ident,'renderer':renderer,'gpu_uuid':roles['sim_uuid'],'env_count':1,
                'planner_directory':str(directory),'candidate_directory':str(root/'no_neural_requests'),
                'sample_ids':phase['ids'],'stop_unix':reg['deadline_unix'],'registration_sha256':sha(root/'registration.json')})
            with launch.with_suffix('.log').open('x') as log:
                active=subprocess.Popen(container_command(root,reg,roles['sim_uuid'],launch,name),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(launch.with_suffix('.process.json'),{'pid':active.pid,'container':name,'owned':True,**(process_info(active.pid) or {})})
            start=time.monotonic()
            while active.poll() is None:
                healthy(root,{'planner':planner})
                if (root/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('Static capture stopped/deadline')
                if startup_expired(group,time.monotonic()-start,1800):raise RuntimeError('Static World startup timeout')
                current=read(group/'state.json') if (group/'state.json').exists() else {'processed':0}
                write(root/'state.json',{'status':'capturing','phase':ident,**current,'training_started':False})
                time.sleep(3)
            code=active.returncode;active=None
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:
                raise RuntimeError('Static World removal unconfirmed')
            if code or (group/'FAULT.json').exists():raise RuntimeError('Static simulator/configuration/record failure')
            if not read(group/'shutdown_receipt.json')['all_trajectories_idle']:raise RuntimeError('Static idle receipt failed')
            phases.append(ident)
            if ident=='pilot':
                export(root,root/'pilot_export',phases)
                admitted=[json.loads(s) for s in (root/'pilot_export/admission.jsonl').read_text().splitlines() if s]
                labels={r['category'] for r in admitted if r['admitted']}
                if 'grasp' not in labels or 'funnel' not in labels:
                    write(root/'PILOT_NOT_ACCEPTED.json',{'admitted_categories':sorted(labels),
                        'remainder_not_dispatched':True,'criteria':'at least one complete admitted GRASP pair and one full-funnel REFINE pair'})
                    raise RuntimeError('Static interface probe lacks admitted motion/GRASP pairs')
        export(root,root/'dataset')
        verify_files({**reg['dependencies'],**reg['execution_source_files']})
        write(root/'completion_receipt.json',{'pass':True,'candidate_count':1024,'dataset':str(root/'dataset'),
            'remaining_scheduled':static['remaining_scheduled'],'excluded_without_replay':static['excluded_without_replay'],
            'training_started':False,'physical_grasp_trials':0,'finished_unix':time.time()})
    except BaseException:
        error=traceback.format_exc();write(root/'coordinator_fault.json',{'error':error,'wall':time.time()})
        exclusive_stop(root/'STOP',{'reason':'static_capture_or_record_or_admission_failure','error':error})
    finally:
        if active is not None and active.poll() is None:
            subprocess.run(['podman','kill','--signal','TERM',name],capture_output=True,timeout=30)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['podman','kill','--signal','KILL',name],capture_output=True,timeout=30)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:unresolved=True
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:unresolved=True
        if planner:
            write(root/'workers/planner/STOP',{'reason':'bounded static collection ended'})
            try:planner.wait(timeout=30)
            except subprocess.TimeoutExpired:planner.terminate();planner.wait(timeout=30)
            write(root/'workers/planner/process_exit.json',{'pid':planner.pid,'exit_code':planner.returncode})
        write(root/'state.json',{'status':'fault' if error or unresolved else 'complete','error':error,
            'unconfirmed_world':unresolved,'finished_unix':time.time(),'training_started':False})
    if error:raise RuntimeError(error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('root')
    p.add_argument('--object');p.add_argument('--compute-index',type=int,default=1);p.add_argument('--sim-index',type=int,default=3)
    a=p.parse_args();print(prepare(a.root,a.object,a.compute_index,a.sim_index) if a.mode=='prepare' else run(a.root))
