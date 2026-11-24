"""New independent 2-slot lookahead and 24-slot recovery registrations."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.seed_augmentation.contract import source_seed
from training.nearfield_arx.export import ledger_rows,episode_labels

OBJECTS=('household_hope_milk_carton','mustard_bottle','raisin_box','cheez_it','household_hot3d_potato_masher','ycb_056_tennis_ball')


def predecessors(paths):
    from scripts.curobo_arx.data_factory.watchdog import process_info
    proof=[]
    for root in paths:
        state=read(root/'state.json')
        if state['status'] not in ('complete','fault') or state.get('unconfirmed_world'):
            raise RuntimeError('Prior execution not conclusively idle: '+str(root))
        for file in list((root/'workers').glob('*/process.json'))+list((root/'group_registrations').glob('*.process.json')):
            p=read(file);live=process_info(p['pid'])
            if live and live.get('start_ticks')==p.get('start_ticks'):raise RuntimeError('Prior owned process still exists')
            if p.get('container') and subprocess.run(['podman','container','exists',p['container']],timeout=30).returncode!=1:
                raise RuntimeError('Prior World absence unconfirmed')
        for file in (root/'groups').glob('*/shutdown_receipt.json'):
            if not read(file)['all_trajectories_idle']:raise RuntimeError('Prior idle receipt failed')
        proof.append({'path':str(root),'state_sha256':sha(root/'state.json'),
            'registration_sha256':sha(root/'registration.json'),
            'STOP_sha256':sha(root/'STOP') if (root/'STOP').exists() else None,
            'owned_processes_absent':True,'owned_Worlds_absent':True})
    return proof


def choose(parent):
    rows=ledger_rows(parent/'ledger.sqlite3');seeds={};details={}
    prior=read(ROOT/'runs/nearfield_stage1_20261105_01/execution_03/registration.json')
    common=read(ROOT/'runs/nearfield_stage1_20261105_01/dataset_v3/manifest.json')
    eligible={r['id']:r for r in common['episodes'] if r['comparison_eligible'] and r['split']=='train' and r['policy']=='native'}
    for obj in OBJECTS:
        fixed=next((s['seed_id'] for s in prior['seeds'].values() if s['object_id']==obj),None)
        for row in rows:
            if row['object_id']!=obj or row['id'] not in eligible or fixed and row['id']!=fixed:continue
            try:seed=source_seed(row)
            except ValueError:continue
            record=eligible[row['id']]
            raw,star,priors,samples,excluded=episode_labels(record)
            first=min(s['raw_index'] for s in samples);last=next(s['raw_index'] for s in samples if s['labels']['C']=='GRASP')
            indices=list(range(first+1,last+1))
            details[obj]={'star':star,'priors':priors,'raw_indices':indices,
                'waypoints':[raw[i]['snapshot']['link_world']['right_eef_link'] for i in indices],'record':record}
            seed['files'][str(Path(seed['episode'])/'samples.jsonl')]=sha(Path(seed['episode'])/'samples.jsonl')
            seeds[obj]=seed;break
        else:raise RuntimeError('No admitted stationary training seed for '+obj)
    return seeds,details


def prepare(parent,out,kind,mask_reference,replay=None,continuation=None,admission_recheck=None,startup_of=None,resume_window_seconds=None,resume_reason=None,recover_unissued_claim=False,supplement_of=None,authorization_file=None):
    parent,out,mask_reference=Path(parent).resolve(),Path(out).resolve(),Path(mask_reference).resolve()
    if out.exists():raise FileExistsError('Keep all prior registrations')
    if supplement_of and (kind!='replay' or not continuation or not authorization_file or startup_of or recover_unissued_claim):
        raise ValueError('An authorized new validation is separate from a startup continuation')
    if recover_unissued_claim and (not startup_of or not continuation or kind!='replay' or resume_window_seconds is not None):
        raise ValueError('Unissued-claim repair requires the same replay continuation and original window')
    if resume_window_seconds is not None:
        if not startup_of or kind!='replay' or not resume_reason or not 900<=resume_window_seconds<=7200:
            raise ValueError('An explicit user resume may renew only an unclaimed replay window of 900..7200 seconds')
    prior_root=ROOT/'runs/nearfield_stage1_20261105_01'
    proof=predecessors([prior_root/p for p in ('execution','execution_02','execution_03')])
    from .continuation import admitted_carry, remaining_slots, completed_replays, relocated_dependencies, unused_startup, authorized_supplement
    startup=None;startup_dependencies={}
    supplement=None;prior_results=[];supplement_dependencies={}
    if supplement_of:
        supplement_of=Path(supplement_of).resolve();proof+=predecessors([supplement_of])
        supplement,prior,supplement_dependencies=authorized_supplement(supplement_of,authorization_file)
        prior_results=[prior]
    if startup_of:
        startup_of=Path(startup_of).resolve();proof+=predecessors([startup_of])
        startup=unused_startup(startup_of,recover_unissued_claim=recover_unissued_claim)
        old_startup=read(startup_of/'registration.json')
        verify_files(old_startup['execution_source_files'])
        startup_dependencies,_=relocated_dependencies(old_startup['dependencies'])
    carried, carry_files, remaining = [], {}, None
    replay_dependencies={};prerequisite_results=[];replay_registration=None
    if continuation:
        if kind != 'replay' or not admission_recheck:raise ValueError('Replay continuation needs offline admission evidence')
        continuation=Path(continuation).resolve();proof+=predecessors([continuation])
        accepted,carry_files=admitted_carry(continuation,admission_recheck)
        carried=[accepted];remaining=remaining_slots(continuation,carried)
        if len(remaining)!=1 or remaining[0]['object_id']!='mustard_bottle':
            raise ValueError('Only the unused mustard slot can be scheduled')
    if kind=='recovery':
        if replay is None:raise ValueError('Recovery requires completed interface replay evidence')
        replay=Path(replay).resolve();proof+=predecessors([replay])
        prerequisite_results=completed_replays(replay)
        acceptance=read(replay/'interface_capture_acceptance.json')
        if (not acceptance.get('pass') or not acceptance['recovery_release']['preconditions_satisfied']
                or acceptance['recovery_release']['authorized_slots']!=24):
            raise ValueError('Recovery requires the completed interface/capture review')
        verify_files(acceptance['evidence'])
        replay_registration=read(replay/'registration.json')
        verify_files(replay_registration['execution_source_files'])
        replay_dependencies,_=relocated_dependencies(replay_registration['dependencies'])
        for name in ('registration.json','state.json','summary.json','interface_capture_acceptance.json'):
            replay_dependencies[str(replay/name)]=sha(replay/name)
        replay_dependencies.update(acceptance['evidence'])
        for result in prerequisite_results:
            for attempt in result['attempts']:
                for file in Path(attempt['episode']).rglob('*'):
                    if file.is_file():replay_dependencies[str(file)]=sha(file)
    calibration=read(mask_reference/'comparison.json')
    if len(calibration['frames'])!=20:raise RuntimeError('Complete 20-frame independent mask review first')
    old=read(parent/'registration.json')
    dependencies,relocations=relocated_dependencies(old['dependencies'])
    verify_files(old['execution_source_files'])
    if continuation:
        previous_dependencies,old_relocations=relocated_dependencies(read(continuation/'registration.json')['dependencies'])
        relocations+=old_relocations;carry_files.update(previous_dependencies)
    seeds,details=choose(parent);out.mkdir(parents=True);space(out)
    write(out/'state.json',{'status':'preparing','physical_trials':0})
    previous=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    shutil.copytree(previous,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
    for package in ('scripts/curobo_arx/seed_augmentation','scripts/curobo_arx/nearfield','training/nearfield_arx','training/show_harness_arx'):
        shutil.copytree(ROOT/package,source/package,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('recording.py','ledger.py','runtime.py','vector_env.py'):
        shutil.copy2(ROOT/'scripts/curobo_arx/data_factory'/name,source/'scripts/curobo_arx/data_factory'/name)
    shutil.copy2(ROOT/'arx-sim/arx_sim/sensors/camera.py',source/'arx-sim/arx_sim/sensors/camera.py')
    execution={str(source/Path(p).relative_to(previous) if Path(p).is_relative_to(previous) else p):v for p,v in old['execution_source_files'].items()}
    for p in source.rglob('*.py'):execution[str(p)]=sha(p)
    reg=copy.deepcopy(old)
    reg['dependencies']=dependencies
    reg['dependencies'].update(startup_dependencies)
    reg['dependencies'].update(supplement_dependencies)
    reg['dependencies'].update(replay_dependencies)
    reg['dependencies'].update(carry_files)
    for key in ('bulk_storage','resume','lineage','controlled_resource_pause'):reg.pop(key,None)
    trials=[]
    for k,obj in enumerate(OBJECTS if kind=='recovery' else OBJECTS[:2]):
        seed=seeds[obj];d=details[obj];reg['dependencies'].update(seed['files'])
        for slot in (range(4) if kind=='recovery' else range(1)):
            dp=np.zeros(3);dr=np.zeros(3);axis=k%2;sign=1 if k%2==0 else -1
            if kind=='recovery':
                dp[2]=[-.012,-.015,-.012,-.006][slot]
                if slot in (1,3):dp[axis]=sign*(.010 if slot==1 else .006)
                if slot==2:dr[k%3]=8*(1 if k<3 else -1)
            offset=np.eye(4);offset[:3,3]=dp;offset[:3,:3]=Rotation.from_rotvec(np.deg2rad(dr)).as_matrix()
            target=d['star']@offset
            trials.append({'trial_id':f'{len(trials):02d}__{obj}__'+(f'recovery_{slot+1}' if kind=='recovery' else 'C_lookahead'),
                'method':'recovery' if kind=='recovery' else 'C','policy':'recovery' if kind=='recovery' else 'C',
                'seed_id':seed['seed_id'],'object_id':obj,'renderer':seed['renderer'],'placement':seed['placement'],
                'entry_world':seed['anchor_world'],'source_Gstar_world':d['star'].tolist(),'G0_world':d['priors'][1]['world'],
                'C_waypoints':d['waypoints'] if kind=='replay' else [],'C_waypoint_raw_indices':d['raw_indices'] if kind=='replay' else [],
                'offset_goal_world':target.tolist() if kind=='recovery' else None,
                'perturbation':{'type':['retreat','retreat_side','retreat_rotation','near_side'][slot] if kind=='recovery' else 'none',
                    'translation_Gstar_axes_m':dp.tolist(),'rotation_Gstar_axes_degrees':dr.tolist(),
                    'translation_norm_m':float(np.linalg.norm(dp)),'rotation_center':'Gstar TCP origin',
                    'rotation_convention':'right-multiplied rotation vector in source Gstar axes'},
                'truth_replay_only_no_policy_training':kind=='replay','no_retry':True})
    if remaining is not None:
        expected={t['trial_id']:t for t in remaining}
        trials=[t for t in trials if t['trial_id'] in expected]
        for t in trials:
            if digest(t)!=digest(expected[t['trial_id']]):raise ValueError('Remaining slot specification changed')
    if startup and digest(trials)!=digest(startup['seed_trials']):
        raise ValueError('Startup continuation changed original unused slots')
    if supplement:
        if len(trials)!=1 or trials[0]!=supplement:raise ValueError('Supplement changed original scene/action specification')
        trials[0]['trial_id']+='__supplement_01'
    for t in trials:
        catalog=copy.deepcopy(reg['catalog'])
        for asset in catalog['objects']:
            asset['collection_enabled']=asset['object_id']==t['object_id']
            if asset['collection_enabled']:asset['collection_placements']=[t['placement']]
        path=out/'trial_ledgers'/(t['trial_id']+'.sqlite3');path.parent.mkdir(exist_ok=True)
        ledger=Ledger(path);ledger.initialize(catalog,{**reg['quotas'],'first_object':t['object_id'],
            'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1});ledger.close()
    for name,p in list(reg['reference_files'].items()):
        dest=out/'references'/(name+'.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
        reg['reference_files'][name]=str(dest);reg['dependencies'][str(dest)]=sha(dest)
    shutil.copy2(parent/'storage_policy.json',out/'storage_policy.json')
    required=[ROOT/'research_plan/chat'/('gpt回复' if n.startswith('gpt') else 'gemini回复')/n
              for n in ('gpt回应6.md','gpt回应7.md','gemini回复7.md')]
    required +=[mask_reference/n for n in ('selection.json','annotations.json','comparison.json')]
    required +=[ROOT/'runs/nearfield_stage1_20261105_02/baseline_lookahead_v1'/n for n in ('manifest.json','paired_samples.jsonl','pairing_audit.json')]
    for p in required:
        reg['dependencies'][str(p)]=sha(p)
    reg['dependencies'][str(out/'storage_policy.json')]=sha(out/'storage_policy.json')
    reg.update(stage='nearfield_lookahead_closure_v2_'+kind,production=False,preparation_complete=True,
        created_unix=time.time(),deadline_unix=time.time()+(2 if kind=='replay' else 14)*3600,
        seeds={s['seed_id']:s for s in seeds.values() if any(t['seed_id']==s['seed_id'] for t in trials)},
        seed_trials=trials,source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),
        envs_per_gpu=1,object_block_attempts=1,max_physical_trials=len(trials),no_automatic_retry=True,
        minimum_available_host_memory_bytes=40*1024**3,project_root=str(ROOT),
        worker_module='scripts.curobo_arx.nearfield.worker',runtime_module='scripts.curobo_arx.nearfield.runtime_v2',
        C_intermediate_phase='local',recovery_motion_phase='local',C_lookahead={'position_m':.020,'angle_degrees':4.5},
        formal_48_trials_authorized_here=False,training_authorized_here=False,
        recovery_supervision='Only fresh stopped G1 and subsequent recovery; H-to-G1 excluded',
        replay_predecessor=str(replay) if replay else None)
    reg.update(carried_results=carried, synchronized_capture={'version':'arx_nearfield_capture_v1',
        'scope':'right_wrist after physics; render-only refresh with unchanged physical ticks',
        'max_render_refreshes':6,'same_render_product':True,'save_instance_ids_and_roles':True,
        'render_every_update':True,'capture_camera_tick_rate_hz':0,
        'capture_trigger_scope':'Derived RTX camera autotrigger; original mounted camera, optics, rendering algorithm, physical and recorded sample clocks unchanged'},
        replay_slots_total=2,recovery_slots_total=24)
    if supplement:
        reg.update(supplement_predecessor=str(supplement_of),prior_results=prior_results,
            additional_authorized_validation_slots=1,replay_slots_total=3,
            replay_interface_cases=2,authorization_file=str(Path(authorization_file).resolve()))
        write(out/'supplement_registration.json',{'authorization':read(authorization_file),
            'original_trial':supplement,'new_trial_id':trials[0]['trial_id'],'prior_results':prior_results,
            'new_physical_trial_limit':1,'recovery_trial_limit_unchanged':24})
    if replay_registration:
        reg.update(prior_results=replay_registration.get('prior_results',[]),
            replay_slots_total=replay_registration.get('replay_slots_total',2),
            prerequisite_results=prerequisite_results)
        write(out/'prerequisite_receipt.json',{'replay_campaign':str(replay),
            'passed_interface_cases':[r['object_id'] for r in prerequisite_results],
            'raw_evidence_pinned':True,'new_physical_trial_limit':24,
            'preserved_prior_results':reg['prior_results']})
    if startup:
        reg['deadline_unix']=startup['deadline_unix']
        reg['startup_predecessor']=str(startup_of)
        for name in ('registration.json','STOP'):
            reg['dependencies'][str(startup_of/name)]=sha(startup_of/name)
        for claim in startup['unissued_claims']:
            reg['dependencies'].update(claim['files'])
        for file in (startup_of/'trial_ledgers').glob('*.sqlite3'):
            reg['dependencies'][str(file)]=sha(file)
        write(out/'startup_continuation.json',startup)
        if resume_window_seconds is not None:
            resumed_at=time.time();reg['deadline_unix']=resumed_at+resume_window_seconds
            reg['user_resume_window']={'reason':resume_reason,'resumed_at':resumed_at,
                'old_deadline_unix':startup['deadline_unix'],'new_deadline_unix':reg['deadline_unix'],
                'maximum_seconds':resume_window_seconds,'scheduled_slots':len(trials),
                'previous_claimed_attempts':0,'per_attempt_seconds_unchanged':reg['attempt_seconds']}
            write(out/'user_resume_window.json',reg['user_resume_window'])
    write(out/'document_relocations.json',{'entries':relocations})
    if carried:write(out/'carry_forward.json',{'results':carried,'dependencies':carry_files,'new_physical_attempts':0})
    write(out/'old_execution_absence.json',proof)
    changes={str(p.relative_to(source)):{'old':sha(previous/p.relative_to(source)) if (previous/p.relative_to(source)).exists() else None,'new':sha(p)}
        for p in source.rglob('*.py') if not (previous/p.relative_to(source)).exists() or sha(previous/p.relative_to(source))!=sha(p)}
    write(out/'source_change_receipt.json',{'changed_sources':changes,'parent_registration_sha256':sha(parent/'registration.json'),
        'original_evaluator_sha256':sha(source/'scripts/curobo_arx/lilo_vla/egad_evaluate.py'),
        'change_scope':'Existing lookahead/closure protocol; post-physics render-only capture with instance IDs and ReferenceTime; complete raw channel admission; verified single-slot continuation and document relocations',
        'closed_gate':'Original 22mm nominal gripper, 8mm table margin, +12cm lift; actual hold q plus current nominal-state validation',
        'frozen_robot_physics_success_convergence_and_image_gates':True,'old_STOPs_preserved':True})
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    (out/'research.md').write_text('# Near-field explicit GRASP validation\n\n'+
        f'{len(trials)} fixed slots; no retry, no training. Compare action/record coverage separately from physical success. All old outcomes and STOPs retained. '+
        'Preserve ARX configuration, initial placements, renderer, original qualification900s and attempt900s, >6cm every hold sample for3s, bilateral closure and all original gates.\n')
    (out/'research_log.md').write_text('Registered before new physical commands.\n')
    verify_files({**execution,**reg['dependencies']})
    write(out/'state.json',{'status':'prepared','scheduled_slots':len(trials),'physical_trials':0})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=['replay','recovery'],required=True);p.add_argument('--mask-reference',required=True);p.add_argument('--replay')
    p.add_argument('--continuation');p.add_argument('--admission-recheck')
    p.add_argument('--startup-of')
    p.add_argument('--resume-window-seconds',type=int);p.add_argument('--resume-reason')
    p.add_argument('--recover-unissued-claim',action='store_true')
    p.add_argument('--supplement-of');p.add_argument('--authorization-file')
    a=p.parse_args();print(prepare(a.parent,a.out,a.kind,a.mask_reference,a.replay,a.continuation,a.admission_recheck,a.startup_of,a.resume_window_seconds,a.resume_reason,a.recover_unissued_claim,a.supplement_of,a.authorization_file))
