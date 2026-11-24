"""Authorized advisor37 seed registration; all prior cases remain immutable."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices,environment
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop,process_info
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from training.nearfield_arx.lateral_campaign import ledgers
from .advisor31_campaign import exited
from .advisor34_launch import owned_process,wait_ready,healthy,container_command
from .advisor36_contract import boundary_poses,canonical_release
from .physical_startup import startup_expired
from .resources import probe_cuda

PREVIOUS=ROOT/'runs/nearfield_advisor34_20261111_01'


def import_probe(out, source, timeout_s=600):
    """Read-only import with an owned container and explicit cleanup evidence."""
    out, source = Path(out), Path(source)
    name = 'arx-a37-import-' + digest(str(out.resolve()))[:16]
    if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
        raise RuntimeError('Import probe container already exists; inspect before continuation')
    command = ['podman', 'run', '--pull=never', '--name', name, '--user', '0',
        '--security-opt=label=disable', '-v', str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-w', str(source), '--entrypoint', '/isaac-sim/python.sh',
        'localhost/arx-s02-isaac:6.0.1', '-c',
        'import scripts.curobo_arx.nearfield.runtime_advisor37_seed; print("seed_runtime_import_ok")']
    receipt = {'command': command, 'container': name, 'timeout_s': timeout_s,
               'image_download': False, 'simulator_World_started': False, 'pass': False}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        receipt.update(returncode=process.returncode, stdout=stdout[-4000:], stderr=stderr[-4000:])
        receipt['pass'] = process.returncode == 0 and 'seed_runtime_import_ok' in stdout
    except subprocess.TimeoutExpired:
        receipt['timed_out'] = True
        process.kill()
        stdout, stderr = process.communicate(timeout=30)
        receipt.update(stdout=stdout[-4000:], stderr=stderr[-4000:])
    finally:
        try:
            cleanup = subprocess.run(['podman', 'rm', '--force', name], capture_output=True, text=True, timeout=60)
            absent = subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30)
            receipt.update(cleanup_returncode=cleanup.returncode, cleanup_stderr=cleanup.stderr[-2000:],
                           container_absent=absent.returncode == 1)
            receipt['pass'] = receipt['pass'] and receipt['container_absent']
        except BaseException:
            receipt['cleanup_error'] = traceback.format_exc()
            receipt['pass'] = False
            raise
        finally:
            write(out / 'container_preflight.json', receipt)
    if not receipt['pass']:
        raise RuntimeError('Frozen container import/cleanup failed; no automatic retry')
    return receipt


def prepare(top,compute_index,sim_index,boundary_mode,*,name='seed_preflight',candidate_file=None,objects=None,previous_campaign=None):
    top=Path(top).resolve();out=top/name
    authority=read(top/'user_authority.json')
    if not authority['remove_nominal_closed_preflight_gate'] or authority['automatic_training_authorized']:
        raise RuntimeError('Advisor37 explicit seed-only authority mismatch')
    if Path(name).name!=name:raise ValueError('One new round directory required')
    candidate_file=Path(candidate_file).resolve() if candidate_file else top/'canonical_candidates.json'
    previous_campaign=Path(previous_campaign).resolve() if previous_campaign else PREVIOUS
    if out.exists():raise FileExistsError('Never overwrite seed registration')
    old=read(PREVIOUS/'registration.json');inventory=read(top/'source_inventory.json')
    candidates={r['object_id']:r for r in read(candidate_file)}
    selected_objects=set(objects) if objects is not None else {r['object_id'] for r in inventory}
    if len(inventory)!=8 or not selected_objects or not selected_objects<=set(candidates) or not all(candidates[o]['pass_local_geometry'] for o in selected_objects):
        raise RuntimeError('Every scheduled object needs a local-normal canonical candidate')
    source_map={r['object_id']:r for r in inventory}
    order=['household_hope_milk_carton','raisin_box','cheez_it','mustard_bottle',
           'ycb_056_tennis_ball','household_hot3d_potato_masher',
           'household_hot3d_bbq_sauce_bottle','household_ycb_spam_can']
    order=[o for o in order if o in selected_objects]
    trials=[];seeds={}
    for obj in order:
        row=source_map[obj];chosen=candidates[obj]['selected'];seed=copy.deepcopy(row['seed'])
        g=np.asarray(row['source_object_world'])@np.asarray(chosen['Gcanonical_object'])
        goal=copy.deepcopy(seed['goal']);goal.update(grasp_world=g.tolist(),raw_grasp_world=g.tolist(),
            goal_id=digest(['advisor37',obj,g.tolist()])[:32],candidate_source='advisor37_local_surface_canonical',
            target_contact_world=(np.asarray(row['source_object_world'])@np.r_[chosen['pad_center_object'],1])[:3].tolist(),
            width_m=chosen['contact_width_m'],qualification={})
        seed['goal']=goal;seed['source_seed_id']=row['seed_id'];seeds[row['seed_id']]=seed
        for case in boundary_poses(g,boundary_mode):
            ident=f'a37_s{len(trials):02d}_{obj}_{case["case"]}'
            trials.append({k:copy.deepcopy(row[k]) for k in ('seed_id','object_id','source_family','source_split','geometry_group','renderer','placement')})
            trials[-1].update(trial_id=ident,method='recovery',policy='advisor37_canonical_seed_preflight',
                boundary_case=case['case'],boundary_mode=boundary_mode,boundary_translation_m=case['translation_m'],
                boundary_rotation_degrees=case['rotation_degrees'],Gcanonical_world=g.tolist(),
                Gcanonical_object=chosen['Gcanonical_object'],source_Gstar_world=g.tolist(),G0_world=g.tolist(),
                entry_world=case['goal_world'],offset_goal_world=case['goal_world'],target_delta_world_m=[0.,0.,0.],
                no_retry=True,learned_inference=False,physical_entry_kind='canonical_seed_target',
                perturbation={'type':boundary_mode,'case':case['case']})
    assert len(trials)==7*len(selected_objects)
    out.mkdir();space(out)
    command=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_advisor37_motion','-v']
    test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
    (out/'offline_checks.log').write_text(test.stdout+test.stderr)
    write(out/'offline_checks.json',{'pass':test.returncode==0,'command':command,'physical_trials':0,'log_sha256':sha(out/'offline_checks.log')})
    if test.returncode:raise RuntimeError('Seed contract offline checks failed')
    write(out/'predecessor_exit_receipt.json',exited(previous_campaign))
    verify_files(old['execution_source_files'])
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    added=[*(ROOT/'scripts/curobo_arx/nearfield').glob('advisor36_*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('runtime_advisor36_*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('test_advisor36*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('advisor37_*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('runtime_advisor37_*.py'),
           *(ROOT/'scripts/curobo_arx/nearfield').glob('test_advisor37*.py')]
    for p in added:
        dest=source/p.relative_to(ROOT);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    # Geometry analysis is complete; include its pure math dependency without
    # changing any inherited runtime implementation.
    for name in ('rulefree_diagnose.py','advisor35_diagnose.py'):
        p=ROOT/'scripts/curobo_arx/nearfield'/name;dest=source/p.relative_to(ROOT);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    ds=devices();by={r['index']:r for r in ds};roles={'compute_uuid':by[compute_index]['uuid'],'sim_uuid':by[sim_index]['uuid']}
    if roles['compute_uuid']==roles['sim_uuid']:raise ValueError('Separate planner and simulator GPUs')
    deps=dict(old['dependencies'])
    for seed in seeds.values():deps.update(seed['files'])
    for p in (top/'advisor37.md',top/'scope_registration.json',top/'source_inventory.json',candidate_file,top/'geometry_completion.json',
              top/'user_authority.json',PREVIOUS/'registration.json',PREVIOUS/'completion_receipt.json'):
        deps[str(p)]=sha(p)
    if (top/'scope_amendment_01.json').exists():deps[str(top/'scope_amendment_01.json')]=sha(top/'scope_amendment_01.json')
    for candidate in candidates.values():
        if candidate.get('refinement_geometry_file'):
            deps[candidate['refinement_geometry_file']]=sha(candidate['refinement_geometry_file'])
    if previous_campaign!=PREVIOUS:
        for file in ('registration.json','summary.json','state.json'):
            deps[str(previous_campaign/file)]=sha(previous_campaign/file)
    deps.update(read(top/'geometry_completion.json')['files'])
    now=time.time();reg=copy.deepcopy(old)
    deadline=min(read(top/'scope_registration.json')['deadline_unix'],read(previous_campaign/'registration.json')['deadline_unix'])
    if deadline<=now:raise RuntimeError('Original canonical search deadline expired')
    for key in ('carried_results','prior_results','prerequisite_results','carried_terminal_ids','ancestor_campaigns'):reg[key]=[]
    reg.update(stage='advisor37_canonical_seed_preflight',project_root=str(ROOT),source_snapshot=str(source),
        execution_source_files=execution,source_identity=digest(execution),dependencies=deps,seeds=seeds,seed_trials=trials,
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor37_seed',worker_module='scripts.curobo_arx.nearfield.advisor37_teacher',
        coordinator_module='scripts.curobo_arx.nearfield.advisor37_seed',created_unix=now,campaign_started_unix=now,
        deadline_unix=deadline,wall_hours=(deadline-now)/3600,requested_devices=roles,envs_per_gpu=1,object_block_attempts=1,
        max_physical_trials=len(trials),global_physical_budget=len(trials),training_authorized_here=False,
        learned_inference_authorized_here=False,no_automatic_retry=True,supplemental_interruptions={},
        automatic_attempt_retries=0,automatic_startup_retries=0,
        preparation_import_timeout_s=600,
        execution_authority={'file':str(top/'advisor37.md'),'sha256':sha(top/'advisor37.md')},
        seed_preflight={'objects':len(selected_objects),'cases_per_object':7,'boundary_mode':boundary_mode,
            'skip_remaining_object_cases_after_first_failure':True,'release_requires_7_of_7_success':True,
            'finite_samples_do_not_prove_every_point_in_continuous_core':True,
            'closure_source':'direct_seed_not_VLM','external_geometric_closure_gate':False},
        advisor37_seed_protocol={'arrival_translation_m':.003,'arrival_rotation_degrees':1.5,
            'remove_nominal_closed_preflight_gate':True,'approach_prepose_backoff_m':.060,
            'target_native_convex_required':True,'Cartesian_knot_spacing_m':.001,
            'closed_microstep_preserves_original_collision_model':True,'training_authorized':False},
        canonical_candidate_file=str(candidate_file),
        continuation_of=str(previous_campaign),remaining_trial_ids=[t['trial_id'] for t in trials])
    shutil.copy2(PREVIOUS/'storage_policy.json',out/'storage_policy.json')
    write(out/'resource_plan.json',{'inventory':ds,'roles':roles,'other_users_untouched':True})
    write(out/'case_manifest.json',{'trials':trials,'maximum_physical_slots':len(trials),'retries':0})
    write(out/'source_change_receipt.json',{'new_modules':{str(p):sha(p) for p in added},
        'inherited_execution_sources_unchanged':True,'robot_physics_success_unchanged':True,
        'changes':['fixed canonical/boundary teacher targets','authorized seed closure without nominal closed prediction; unchanged measured open-model and physical checks',
                   'no learned service or old geometric closure admission','skip object remainder after a failed seed case']})
    ledgers(out,reg);write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    import_probe(out, source, timeout_s=reg['preparation_import_timeout_s'])
    verify_files({**execution,**deps});write(out/'full_preflight.json',{'pass':True,'checked':len({**execution,**deps}),'physical_trials':0})
    write(out/'state.json',{'status':'prepared','scheduled':len(trials),'executed':0})
    return str(out)


def summary(root):
    root=Path(root);reg=read(root/'registration.json');rows=[]
    for p in sorted(root.glob('groups/*/results.json')):rows+=read(p)['trials']
    if len({r['trial_id'] for r in rows})!=len(rows):raise RuntimeError('Repeated seed slot')
    objects=[]
    for obj in dict.fromkeys(t['object_id'] for t in reg['seed_trials']):
        sub=[r for r in rows if r['object_id']==obj]
        objects.append({'object_id':obj,'processed':len(sub),'executed':sum(bool(r.get('executed')) for r in sub),
            'success':sum(r['success'] for r in sub),'unexecuted':sum(not r['completed'] for r in sub),
            'canonical_released':canonical_release(sub),'results':[{'case':r['boundary_case'],'outcome':r['outcome'],'success':r['success']} for r in sub]})
    value={'scheduled':len(reg['seed_trials']),'processed':len(rows),'executed':sum(bool(r.get('executed')) for r in rows),
        'successes':sum(r['success'] for r in rows),'objects':objects,'trials':rows,'canonical_released':sum(r['canonical_released'] for r in objects)}
    write(root/'summary.json',value);return value


def run(root):
    root=Path(root).resolve();reg=read(root/'registration.json');roles=reg['requested_devices'];project=Path(reg['project_root'])
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STARTED.json').exists() or (root/'STOP').exists():raise RuntimeError('No in-place restart or STOP bypass')
    write(root/'STARTED.json',{'wall':time.time(),'registration_sha256':sha(root/'registration.json')})
    planner=None;active=None;name=None;error=None;unresolved=False
    try:
        verify_files({**reg['dependencies'],**reg['execution_source_files']});exited(reg['continuation_of'])
        inventory={r['uuid']:r for r in devices()}
        for role,floor in (('compute',6500),('sim',9000)):
            d=inventory[roles[role+'_uuid']]
            if d['free_mb']<floor:raise RuntimeError(role+' GPU has insufficient free memory')
            proof=probe_cuda(d,environment(d));write(root/(role+'_cuda_preflight.json'),proof)
            if not proof['pass']:raise RuntimeError(role+' CUDA preflight failed')
        if available_memory()<reg['minimum_available_host_memory_bytes']:raise RuntimeError('Host memory floor')
        directory=root/'workers/planner';env=environment(inventory[roles['compute_uuid']]);env.update(PYTHONPATH=reg['source_snapshot'],PYTHONUNBUFFERED='1')
        planner=owned_process([str(project.parent/'curobo/.venv/bin/python'),'-m',reg['worker_module'],
            '--directory',str(directory),'--registration',str(root/'registration.json')],reg['source_snapshot'],env,directory)
        wait_ready(planner,directory,root)
        groups={}
        for t in reg['seed_trials']:groups.setdefault(t['renderer'],[]).append(t['trial_id'])
        for index,(renderer,ids) in enumerate(groups.items()):
            healthy(root,{'planner':planner});space(root)
            if (root/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('Stopped/deadline')
            if next(r['free_mb'] for r in devices() if r['uuid']==roles['sim_uuid'])<9000:raise RuntimeError('Simulator free memory dropped')
            ident=f'seed_{index:02d}_{renderer.lower()}';name='arx-'+root.parent.name+'-'+root.name+'-'+ident
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:raise RuntimeError('Old World not absent')
            path=root/'group_registrations'/(ident+'.json');group=root/'groups'/ident
            write(path,{'group_id':ident,'renderer':renderer,'gpu_uuid':roles['sim_uuid'],'env_count':1,
                'planner_directory':str(directory),'candidate_directory':str(root/'no_neural_requests'),
                'trial_ids':ids,'stop_unix':reg['deadline_unix'],'registration_sha256':sha(root/'registration.json')})
            command=container_command(root,reg,roles['sim_uuid'],path,name)
            with path.with_suffix('.log').open('x') as log:active=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(path.with_suffix('.process.json'),{'pid':active.pid,'container':name,'command':command,'owned':True,**(process_info(active.pid) or {})})
            started=time.monotonic()
            while active.poll() is None:
                healthy(root,{'planner':planner})
                if (root/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('Stopped during seed execution')
                if startup_expired(group,time.monotonic()-started,1800):raise RuntimeError('Simulator startup timeout')
                s=summary(root);write(root/'state.json',{'status':'running_seed_preflight','wall':time.time(),
                    'processed':s['processed'],'executed':s['executed'],'successes':s['successes'],'scheduled':len(reg['seed_trials'])})
                time.sleep(2)
            code=active.returncode;active=None
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:raise RuntimeError('World removal unconfirmed')
            if code or (group/'FAULT.json').exists():raise RuntimeError('Seed simulator/record failure; stop subsequent dispatch')
            if not read(group/'shutdown_receipt.json')['all_trajectories_idle']:raise RuntimeError('Idle receipt failed')
            if len(read(group/'results.json')['trials'])!=len(ids):raise RuntimeError('Unaccounted seed slots')
        verify_files({**reg['dependencies'],**reg['execution_source_files']})
    except BaseException:
        error=traceback.format_exc();write(root/'coordinator_fault.json',{'error':error,'wall':time.time()})
        exclusive_stop(root/'STOP',{'reason':'seed_preflight_infrastructure_configuration_record_or_idle_failure','error':error})
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
            write(root/'workers/planner/STOP',{'reason':'bounded seed preflight ended'})
            try:planner.wait(timeout=30)
            except subprocess.TimeoutExpired:planner.terminate();planner.wait(timeout=30)
            write(root/'workers/planner/process_exit.json',{'pid':planner.pid,'exit_code':planner.returncode})
        s=summary(root);write(root/'state.json',{'status':'fault' if error or unresolved else 'complete',
            'processed':s['processed'],'executed':s['executed'],'successes':s['successes'],
            'canonical_released':s['canonical_released'],'unconfirmed_world':unresolved,'error':error,'finished_unix':time.time()})
    if error:raise RuntimeError(error)
    return s['canonical_released']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','summary']);p.add_argument('root')
    p.add_argument('--compute-index',type=int,default=1);p.add_argument('--sim-index',type=int,default=2)
    p.add_argument('--boundary-mode',choices=['coupled_xyz','lateral_xy_yaw'],default='coupled_xyz');a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.root,a.compute_index,a.sim_index,a.boundary_mode))
    elif a.mode=='run':print(run(a.root))
    else:print({k:v for k,v in summary(a.root).items() if k not in ('trials','objects')})
