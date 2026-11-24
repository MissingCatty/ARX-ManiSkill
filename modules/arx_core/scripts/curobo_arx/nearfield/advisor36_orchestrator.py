"""Bounded data-only supervisor. Never launches a model training process."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,verify_files
from scripts.curobo_arx.data_factory.launch import devices
from .advisor36_progress import collect,write_progress,select_next
from .advisor36_seed import prepare as prepare_seed
from .advisor36_static_launch import prepare as prepare_static
from .advisor31_campaign import exited


PYTHON=ROOT.parent/'tmp/arx-show-harness-env/bin/python'


def registration_name(generation):
    return 'data_supervisor_registration'+('' if generation==1 else f'_{generation:02d}')+'.json'


def register(top,generation=1):
    top=Path(top).resolve();path=top/registration_name(generation)
    if path.exists():raise FileExistsError('Supervisor registration exists')
    if generation>1:
        previous=read(top/'data_supervisor_state.json')
        if previous['status']!='scheduling_stopped':raise RuntimeError('Prior supervisor was not stopped for maintenance')
        write(top/f'data_supervisor_previous_state_{generation:02d}.json',previous)
    base=read(top/'seed_preflight/registration.json')
    authority=read(top/'scope_amendment_01.json')
    if authority['training_authorized_this_turn'] is not False:raise ValueError('Data-only user boundary missing')
    sources={str(p):sha(p) for p in (ROOT/'scripts/curobo_arx/nearfield').glob('*advisor36*.py')}
    reg={'created_unix':time.time(),'generation':generation,'deadline_unix':base['deadline_unix'],
        'authority':str(top/'scope_amendment_01.json'),'authority_sha256':sha(top/'scope_amendment_01.json'),
        'initial_seed_campaign':str(top/'seed_preflight'),'maximum_canonical_candidates_per_object':8,
        'initial_seed_systemd_unit':'arx-advisor36-seed-20261112-01.service',
        'maximum_seed_slots_including_initial':448,'one_seed_attempt_per_case_per_candidate':True,
        'ordinary_seed_failure':'register a geometrically distinct normal-qualified candidate, retain old results',
        'infrastructure_failure':'stop dependent scheduling; no automatic retry or STOP removal',
        'static_sim_index':3,'static_compute_index':1,'seed_sim_index':2,'seed_compute_index':1,
        'static_observation_budget_per_object':1024,'maximum_static_observations':8192,
        'interface_probes_are_within_budget':True,'training_authorized':False,
        'source_files':sources,'physical_preflight_and_static_use_separate_Worlds':True}
    write(path,reg);write(top/'data_supervisor_state.json',{'status':'prepared','training_started':False})
    return str(path)


def launch(root,module,unit):
    reg=read(root/'registration.json');cwd=reg['source_snapshot']
    command=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+cwd,
        '--setenv=PYTHONUNBUFFERED=1',str(PYTHON),'-u','-m',module,'run',str(root)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    write(root/'supervisor_service.json',{'unit':unit,'command':command,'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'wall':time.time()})
    if result.returncode:raise RuntimeError('Could not start registered bounded service')


def static_current(top,obj):
    lineage=read(top/'static_lineage.json') if (top/'static_lineage.json').exists() else {}
    if obj not in lineage:return top/'static_objects'/obj
    entry=lineage[obj];path=Path(entry['current'])
    if sha(path/'registration.json')!=entry['registration_sha256']:raise RuntimeError('Static lineage registration changed')
    if entry['predecessors']:
        receipt=read(path/'source_change_continuation_receipt.json');verify_files(receipt['files'])
        if (receipt['replayed_started_samples'] or
            (not receipt['zero_completed_observations'] and not receipt.get('completed_observations_carried'))):
            raise RuntimeError('Invalid static source-change continuation')
    return path


def dataset_summary(top,data):
    objects=[]
    for obj in data['objects']:
        root=static_current(top,obj)
        if (root/'dataset/summary.json').exists():
            s=read(root/'dataset/summary.json')
            objects.append({'object_id':obj,'directory':str(root/'dataset'),
                'admitted':s['admitted_observations'],'observed':s['observed_candidates'],
                'splits':s['split_counts'],'detail':s['objects']})
    return {'completed_object_datasets':len(objects),'objects':objects,
        'observed_candidates':sum(r['observed'] for r in objects),'admitted_observations':sum(r['admitted'] for r in objects),
        'training_started':False,'training_requires_later_user_acceptance':True}


def run(top,generation=1):
    top=Path(top).resolve();lock=(top/'data_supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    regpath=top/registration_name(generation);reg=read(regpath)
    rounds=sorted((p for p in top.glob('seed_preflight*') if (p/'registration.json').exists()),
                  key=lambda p:read(p/'registration.json')['created_unix'])
    current=rounds[-1];round_index=len(rounds)
    started=top/('data_supervisor_STARTED'+('' if generation==1 else f'_{generation:02d}')+'.json')
    if started.exists():raise RuntimeError('No automatic supervisor restart')
    write(started,{'wall':time.time(),'registration_sha256':sha(regpath)})
    active_static=None
    try:
        while True:
            if (top/'STOP').exists():raise RuntimeError('User/campaign STOP')
            if time.time()>=reg['deadline_unix']:raise TimeoutError('Original seed/data deadline reached')
            verify_files(reg['source_files'])
            if sha(reg['authority'])!=reg['authority_sha256']:raise RuntimeError('User execution boundary changed')
            state=read(current/'state.json')
            if state['status']=='fault':raise RuntimeError('Physical preflight infrastructure fault; dependent scheduling stopped')
            if state['status']!='complete':
                service=read(current/'supervisor_service.json')['unit'] if (current/'supervisor_service.json').exists() else reg['initial_seed_systemd_unit']
                alive=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True,timeout=15)
                if alive.stdout.strip() not in ('active','activating'):
                    raise RuntimeError('Physical supervisor no longer active; do not infer terminal outcome or retry')
                if (current/'STOP').exists():raise RuntimeError('Physical preflight STOP must be preserved')
            data=collect(top);progress=write_progress(top,data)
            existing=[]
            for folder in [static_current(top,o) for o in data['objects'] if static_current(top,o).exists()]:
                if not (folder/'state.json').exists():raise RuntimeError('Incomplete static preparation; do not skip it')
                s=read(folder/'state.json')
                if s['status']=='fault':raise RuntimeError('Existing static campaign fault; no automatic replay')
                if s['status']!='complete':existing.append(folder)
            if len(existing)>1:raise RuntimeError('More than one registered static World owner')
            if active_static is None and existing:active_static=existing[0]
            if active_static:
                s=read(active_static/'state.json')
                if s['status']=='fault':raise RuntimeError('Static capture infrastructure/admission fault; no retry')
                if s['status']=='complete':
                    exited(active_static);active_static=None
                else:
                    service=read(active_static/'supervisor_service.json')['unit']
                    alive=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True,timeout=15)
                    if alive.stdout.strip() not in ('active','activating'):
                        raise RuntimeError('Static supervisor no longer active; preserve unfinished observations')
            if active_static is None:
                pending=[o for o in data['objects'] if not static_current(top,o).exists()]
                if pending:
                    inventory={d['index']:d for d in devices()}
                    if (inventory[reg['static_sim_index']]['free_mb']>=12000 and
                        inventory[reg['static_compute_index']]['free_mb']>=6500):
                        obj=pending[0]
                        verify_files(reg['source_files'])
                        path=Path(prepare_static(top,obj,reg['static_compute_index'],reg['static_sim_index']))
                        unit='arx-a36-static-'+obj.replace('_','-')+'-01'
                        launch(path,'scripts.curobo_arx.nearfield.advisor36_static_launch',unit)
                        active_static=path
            if state['status']=='complete' and data['released_objects']<8:
                exited(current)
                if round_index>=reg['maximum_canonical_candidates_per_object']:
                    raise RuntimeError('Finite canonical search envelope exhausted; no standard relaxed')
                candidates,exhausted=select_next(top,current)
                if exhausted:
                    write(top/'geometry_search_exhausted.json',{'objects':exhausted,'existing_results_retained':True,
                        'all_geometry_and_physics_limits_unchanged':True})
                    raise RuntimeError('Registered local-geometry candidate table exhausted for '+','.join(exhausted))
                next_index=round_index+1;folder=top/'candidate_reselection'/f'{next_index:02d}';folder.mkdir(parents=True,exist_ok=False)
                file=folder/'canonical_candidates.json';write(file,candidates)
                path=Path(prepare_seed(top,reg['seed_compute_index'],reg['seed_sim_index'],'coupled_xyz',
                    name=f'seed_preflight_{next_index:02d}',candidate_file=file,objects=[r['object_id'] for r in candidates],previous_campaign=current))
                launch(path,'scripts.curobo_arx.nearfield.advisor36_seed',f'arx-a36-seed-search-{next_index:02d}')
                current=path;round_index=next_index
            dataset=dataset_summary(top,data);write(top/'dataset_progress.json',dataset)
            signature=[data['executed'],data['released_objects'],dataset['completed_object_datasets'],dataset['admitted_observations']]
            marker=top/'data_report_signature.json'
            if not marker.exists() or read(marker)!=signature:
                from .advisor36_data_report import report
                report(top,data);write(marker,signature)
            write(top/'data_supervisor_state.json',{'status':'running_data_only','wall':time.time(),
                'current_seed_campaign':str(current),'canonical_round':round_index,
                'released_objects':data['released_objects'],'executed_seed_tests':data['executed'],
                'successful_seed_tests':data['successful_seed_tests'],
                'active_static_campaign':str(active_static) if active_static else None,
                'completed_object_datasets':dataset['completed_object_datasets'],
                'admitted_observations':dataset['admitted_observations'],'training_started':False})
            if data['released_objects']==8 and dataset['completed_object_datasets']==8 and active_static is None:
                from .advisor36_progress import register_final
                register_final(top)
                write(top/'data_completion_receipt.json',{'pass':True,'seed_objects_7_of_7':8,**dataset,
                    'files':{str(top/n):sha(top/n) for n in ('seed_progress.json','dataset_progress.json')},
                    'training_started':False,'finished_unix':time.time()})
                write(top/'data_supervisor_state.json',{'status':'data_complete_waiting_user_training_acceptance',
                    **dataset,'wall':time.time()});return
            time.sleep(20)
    except BaseException:
        error=traceback.format_exc();write(top/('data_supervisor_FAULT'+('' if generation==1 else f'_{generation:02d}')+'.json'),{'error':error,'wall':time.time(),
            'training_started':False,'already_running_bounded_services_retained':True})
        write(top/'data_supervisor_state.json',{'status':'scheduling_stopped','error':error,'training_started':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','run']);p.add_argument('root')
    p.add_argument('--generation',type=int,default=1)
    a=p.parse_args();print(register(a.root,a.generation) if a.mode=='register' else run(a.root,a.generation))
