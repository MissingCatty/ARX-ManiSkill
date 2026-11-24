"""Finite independent ARX recovery observations under the original static gates."""
import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices, environment
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.watchdog import process_info, exclusive_stop
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from .advisor34_launch import owned_process, wait_ready, healthy, container_command
from .advisor36_contract import token
from .advisor36_export import export, load_lines
from .advisor37_export import independence
from .resources import probe_cuda
from .physical_startup import startup_expired

PARENT=ROOT/'runs/nearfield_advisor40_20261112_01'
OBJECTS=('household_hope_milk_carton','raisin_box','cheez_it')


def entrypoint_preflight(reg):
    result={}
    for key in ('runtime_module','worker_module'):
        path=Path(reg['source_snapshot'])/(reg[key].replace('.','/')+'.py')
        if not path.is_file():raise RuntimeError('Missing registered '+key+': '+str(path))
        actual=sha(path)
        if reg['execution_source_files'].get(str(path))!=actual:
            raise RuntimeError('Execution entry point not pinned: '+str(path))
        result[key]={'module':reg[key],'path':str(path),'sha256':actual}
    return {'pass':True,'entries':result}


def pool(obj,canonical,previous):
    rng=np.random.default_rng(int(digest(['recovery_geometry_v1',obj])[:16],16))
    rows=[];alloc=Counter();anchors=list(previous)
    for index in range(128):
        region='near_recovery' if index<64 else 'mid_rotation' if index<96 else 'entry_rotation'
        if region=='near_recovery':
            p=np.array([rng.uniform(12,35)*rng.choice([-1,1]),rng.uniform(-16,16),rng.uniform(-25,5)])/1000
            if index%2:p[[0,1]]=p[[1,0]]
            r=rng.uniform([-12,-20,-20],[12,20,20])
        elif region=='mid_rotation':
            p=np.r_[rng.uniform(-20,20,2),rng.uniform(15,50)]/1000
            r=rng.uniform(-25,25,3)
        else:
            p=np.r_[rng.uniform(-20,20,2),rng.uniform(65,90)]/1000
            r=rng.uniform([-15,-18,-20],[15,18,20])
        ident=f'rg_{obj}_{index:04d}';members=[]
        for sign in (1,-1):
            dp=p*[sign,sign,1];dr=r*sign;d=np.eye(4)
            d[:3,3]=dp;d[:3,:3]=Rotation.from_rotvec(np.deg2rad(dr)).as_matrix()
            ee=np.asarray(canonical)@np.linalg.inv(d)
            members.append({'sample_id':ident+('_p' if sign==1 else '_m'),'pair_id':ident,'pair_member':sign,
                'object_id':obj,'category':'funnel','axial_layer_mm':region,'recovery_region':region,'wave':0,
                'desired_translation_m':dp.tolist(),'desired_rotation_degrees':dr.tolist(),
                'desired_eef_object':ee.tolist(),'canonical_object':np.asarray(canonical).tolist(),
                'candidate_label':token(ee,canonical),'label_requires_actual_same_frame_FK':True,
                'sampling_contract':'recovery_geometry_fixed_pool_v1','neighborhood_group':ident})
        adjacent=set()
        for a in anchors:
            old=np.asarray(a['pose'])
            for member in members:
                new=np.asarray(member['desired_eef_object'])
                if np.linalg.norm(old[:3,3]-new[:3,3])<=.002 and Rotation.from_matrix(old[:3,:3].T@new[:3,:3]).magnitude()<=np.deg2rad(1):
                    adjacent.add(a['split'])
        if len(adjacent)>1:
            split='excluded'
        elif adjacent:
            split=next(iter(adjacent))
        else:
            ratios={'train':.625,'validation':.1875,'test':.1875}
            split=max(ratios,key=lambda s:(sum(alloc.values())+2)*ratios[s]-alloc[s])
        for row in members:
            row['split']=split
            if split=='excluded':row['proposal_exclusion']='bridges_prior_split_neighborhoods'
            else:anchors.append({'pose':row['desired_eef_object'],'split':split})
        if split!='excluded':alloc[split]+=2
        rows.extend(members)
    assert len(rows)==256 and len({r['sample_id'] for r in rows})==256
    return rows,{'proposed':256,'split_counts':dict(alloc),'proposal_excluded':sum('proposal_exclusion' in r for r in rows)}


def prepare(top,obj,compute_uuid,sim_uuid,name=None):
    top=Path(top).resolve();out=top/'static_objects'/(name or obj)
    if obj not in OBJECTS or out.parent!=top/'static_objects':raise ValueError('Unregistered object or directory')
    if out.exists():raise FileExistsError('No overwrite of capture registration')
    source=next(r for r in read(PARENT/'dataset/manifest.json')['sources'] if r['object_id']==obj)
    donor=Path(source['export']).parent.parent;old=read(donor/'registration.json')
    entrypoint_preflight(old)
    registry=read(old['static_observations']['canonical_registry']);entry=registry['objects'][obj]
    verify_files(entry['physical_evidence']);verify_files(old['execution_source_files'])
    canonical=entry['geometry']['selected']['Gcanonical_object'];previous=[]
    for split in ('train','validation','test'):
        for item in load_lines(PARENT/'dataset'/f'{split}.jsonl'):
            if item['object_id']!=obj:continue
            d=np.eye(4);d[:3,3]=item['actual_translation_m'];d[:3,:3]=Rotation.from_rotvec(np.deg2rad(item['actual_rotation_degrees'])).as_matrix()
            previous.append({'pose':(np.asarray(canonical)@np.linalg.inv(d)).tolist(),'split':split})
    candidates,quality=pool(obj,canonical,previous)
    out.mkdir(parents=True);space(out)
    write(out/'canonical_registry.json',registry);write(out/'candidate_quality.json',quality)
    with (out/'candidates.jsonl').open('x') as f:
        for row in candidates:f.write(json.dumps(row,separators=(',',':'))+'\n')
    prior=Path(old['source_snapshot']);snapshot=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,snapshot,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(snapshot.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(snapshot/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    added=Path(__file__).resolve();dest=snapshot/added.relative_to(ROOT);shutil.copy2(added,dest);execution[str(dest)]=sha(dest)
    deps=copy.deepcopy(old['dependencies'])
    for p in (top/'user_authority.json',top/'scope_registration.json',out/'canonical_registry.json',out/'candidates.jsonl',out/'candidate_quality.json'):
        deps[str(p)]=sha(p)
    reg=copy.deepcopy(old)
    reg.update(project_root=str(ROOT),source_snapshot=str(snapshot),execution_source_files=execution,source_identity=digest(execution),
        dependencies=deps,requested_devices={'compute_uuid':compute_uuid,'sim_uuid':sim_uuid},
        created_unix=time.time(),deadline_unix=min(read(top/'scope_registration.json')['deadline_unix'],time.time()+12*3600),
        coordinator_module='scripts.curobo_arx.nearfield.recovery_geometry_capture',
        worker_module=old['worker_module'],
        runtime_module=old['runtime_module'],
        recovery_geometry_experiment=True,global_physical_budget=0,max_physical_trials=0,seed_trials=[])
    reg['static_observations'].update(canonical_registry=str(out/'canonical_registry.json'),candidate_table=str(out/'candidates.jsonl'),
        candidate_count=256,planned_observations=96,prior_capture_groups=[],excluded_without_replay=[],prior_admission_exclusions={},
        sample_phases=[{'name':'recovery','ids':[r['sample_id'] for r in candidates if not r.get('proposal_exclusion')]}])
    Ledger(out/'ledger.sqlite3').close();shutil.copy2(top/'storage_policy.json',out/'storage_policy.json')
    write(out/'registration.json',reg);verify_files({**execution,**deps})
    write(out/'entrypoint_preflight.json',entrypoint_preflight(reg))
    write(out/'state.json',{'status':'prepared','proposal_count':256,'captured':0,'physical_grasp_trials':0})
    return str(out)


def run(root):
    root=Path(root).resolve();reg=read(root/'registration.json');roles=reg['requested_devices'];static=reg['static_observations']
    lock=(root/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STARTED.json').exists() or (root/'STOP').exists():raise RuntimeError('No capture restart or STOP removal')
    planner=active=None;container=None;error=None;unconfirmed=False;result=None
    try:
        verify_files({**reg['dependencies'],**reg['execution_source_files']});space(root)
        entrypoint_preflight(reg)
        inventory={d['uuid']:d for d in devices()}
        for role,floor in (('compute',6500),('sim',12000)):
            device=inventory[roles[role+'_uuid']]
            if device['free_mb']<floor:raise RuntimeError('Insufficient '+role+' GPU memory')
            proof=probe_cuda(device,environment(device));write(root/(role+'_cuda_preflight.json'),proof)
            if not proof['pass']:raise RuntimeError('CUDA preflight failed')
        if available_memory()<reg['minimum_available_host_memory_bytes']:raise RuntimeError('Host memory floor')
        write(root/'STARTED.json',{'wall':time.time(),'registration_sha256':sha(root/'registration.json')})
        directory=root/'workers/planner';env=environment(inventory[roles['compute_uuid']]);env.update(PYTHONPATH=reg['source_snapshot'],PYTHONUNBUFFERED='1')
        planner=owned_process([str(ROOT.parent/'curobo/.venv/bin/python'),'-m',reg['worker_module'],
            '--directory',str(directory),'--registration',str(root/'registration.json')],reg['source_snapshot'],env,directory)
        wait_ready(planner,directory,root)
        renderer=read(static['canonical_registry'])['objects'][static['object_id']]['source']['renderer']
        container='arx-recovery-geometry-'+digest(str(root))[:12]
        if subprocess.run(['podman','container','exists',container],capture_output=True,timeout=30).returncode!=1:raise RuntimeError('Old World absence unconfirmed')
        launch=root/'group_registrations/recovery.json';group=root/'groups/recovery'
        write(launch,{'group_id':'recovery','renderer':renderer,'gpu_uuid':roles['sim_uuid'],'env_count':1,
            'planner_directory':str(directory),'candidate_directory':str(root/'no_neural_requests'),
            'sample_ids':static['sample_phases'][0]['ids'],'stop_unix':reg['deadline_unix'],
            'registration_sha256':sha(root/'registration.json')})
        with launch.with_suffix('.log').open('x') as log:
            active=subprocess.Popen(container_command(root,reg,roles['sim_uuid'],launch,container),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(launch.with_suffix('.process.json'),{'pid':active.pid,'container':container,'owned':True,**(process_info(active.pid) or {})})
        began=time.monotonic()
        while active.poll() is None:
            healthy(root,{'planner':planner})
            if (root/'STOP').exists() or (root.parent.parent/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('STOP or deadline')
            if startup_expired(group,time.monotonic()-began,1800):raise RuntimeError('Static startup timeout')
            state=read(group/'state.json') if (group/'state.json').exists() else {'processed':0}
            write(root/'state.json',{**state,'status':'capturing','physical_grasp_trials':0,'wall':time.time()})
            time.sleep(3)
        code=active.returncode
        if subprocess.run(['podman','container','exists',container],capture_output=True,timeout=30).returncode!=1:raise RuntimeError('World removal unconfirmed')
        if code or (group/'FAULT.json').exists():raise RuntimeError('Static simulator/configuration/record failure')
        shutdown=read(group/'shutdown_receipt.json')
        if not shutdown['all_trajectories_idle'] or not shutdown['raw_writer_drained_and_closed']:raise RuntimeError('Incomplete shutdown')
        write(root/'state.json',{'status':'independent_admission','physical_grasp_trials':0})
        geometry_files={}
        for path in group.glob('objects/*/static_collision_surfaces.json'):
            geometry=read(path)
            if (geometry['schema']!='arx_static_target_cooked_convex_v2'
                or not geometry['target_all_pieces_native'] or geometry['whole_target_hull_fallback']
                or not all(p['native_convex'] for p in geometry['target'])):
                raise RuntimeError('Original native target collision geometry requirement failed')
            geometry_files[str(path)]=sha(path)
        if not geometry_files:raise RuntimeError('Missing native target geometry evidence')
        write(root/'native_geometry_check.json',{'pass':True,'files':geometry_files})
        exported=root/'exports/recovery';result=export(root,exported,['recovery'])
        records=load_lines(exported/'admission.jsonl');independent=independence(records)
        write(exported/'independent_split_check.json',independent)
        if not independent['pass']:raise RuntimeError('New observed pose split leakage')
        # Check actual captured poses against the original three frozen splits.
        canonical=np.asarray(read(static['canonical_registry'])['objects'][static['object_id']]['geometry']['selected']['Gcanonical_object'])
        old=[]
        for split in ('train','validation','test'):
            for row in load_lines(PARENT/'dataset'/f'{split}.jsonl'):
                if row['object_id']!=static['object_id']:continue
                d=np.eye(4);d[:3,3]=row['actual_translation_m'];d[:3,:3]=Rotation.from_rotvec(np.deg2rad(row['actual_rotation_degrees'])).as_matrix()
                old.append((split,canonical@np.linalg.inv(d)))
        conflicts=[]
        for row in records:
            if not row['admitted']:continue
            p=np.asarray(row['actual_eef_world']);obs=read(row['observation']);obj=np.asarray(obs['snapshot']['objects']['test_object']['world'])
            p=np.linalg.inv(obj)@p
            for split,q in old:
                if split!=row['split'] and np.linalg.norm(p[:3,3]-q[:3,3])<=.001 and Rotation.from_matrix(p[:3,:3].T@q[:3,:3]).magnitude()<=np.deg2rad(.6):
                    conflicts.append(row['sample_id']);break
        write(exported/'original_split_check.json',{'pass':not conflicts,'conflicts':conflicts})
        if conflicts:raise RuntimeError('Actual recovery pose overlaps old different split')
        (root/'dataset').symlink_to('exports/recovery',target_is_directory=True)
        verify_files({**reg['dependencies'],**reg['execution_source_files']})
        write(root/'completion_receipt.json',{'pass':True,**result,'physical_grasp_trials':0,
            'failure_counts':next(r for r in read(exported/'summary.json')['objects']
                                 if r['object_id']==static['object_id'])['rejections']})
    except BaseException:
        error=traceback.format_exc();write(root/'FAULT.json',{'error':error,'wall':time.time()})
        exclusive_stop(root/'STOP',{'reason':'capture_infrastructure_configuration_or_record_fault','error':error})
    finally:
        if active is not None and active.poll() is None:
            subprocess.run(['podman','kill','--signal','TERM',container],capture_output=True,timeout=30)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['podman','kill','--signal','KILL',container],capture_output=True,timeout=30)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:unconfirmed=True
        if container and subprocess.run(['podman','container','exists',container],capture_output=True,timeout=30).returncode!=1:unconfirmed=True
        if planner:
            write(root/'workers/planner/STOP',{'reason':'finite capture ended'})
            try:planner.wait(timeout=30)
            except subprocess.TimeoutExpired:planner.terminate();planner.wait(timeout=30)
            write(root/'workers/planner/process_exit.json',{'pid':planner.pid,'exit_code':planner.returncode})
        write(root/'state.json',{'status':'fault' if error or unconfirmed else 'complete','error':error,
            'unconfirmed_world':unconfirmed,'result':result,'physical_grasp_trials':0,'wall':time.time()})
    if error or unconfirmed:raise RuntimeError(error or 'Unconfirmed World')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('root')
    p.add_argument('--object');p.add_argument('--compute-uuid');p.add_argument('--sim-uuid');p.add_argument('--name');a=p.parse_args()
    print(prepare(a.root,a.object,a.compute_uuid,a.sim_uuid,a.name) if a.mode=='prepare' else run(a.root))
