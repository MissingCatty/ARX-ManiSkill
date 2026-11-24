"""Register the six original held-out sources for synchronized ARX capture.

Source stationarity rejection does not dispatch a physical trial. Missing or
changed provenance is a fault. No replacement sources, retries or split changes.
"""
import argparse
import copy
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.seed_augmentation.contract import source_seed,joint_values
from training.nearfield_arx.export import ledger_rows,episode_labels
from .prepare_v2 import predecessors

BASE=ROOT/'runs/nearfield_stage1_20261105_02/baseline_lookahead_v1'
PARENT=ROOT/'runs/nearfield_stage1_20261105_04/recovery_24'
DATASET=ROOT/'runs/nearfield_dataset_20261106_02'


def selected_records():
    m=read(BASE/'manifest.json')
    records=[r for r in m['episodes'] if r['comparison_eligible'] and r['split']=='validation']
    if len(records)!=6 or len({r['family'] for r in records})!=4:
        raise RuntimeError('Original six-source/four-family validation split changed')
    training=read(DATASET/'manifest.json')['episodes']
    if {r['family'] for r in records}&{r['family'] for r in training}:
        raise RuntimeError('Training family leaked into held-out selection')
    if {r['geometry_group'] for r in records}&{r['geometry_group'] for r in training}:
        raise RuntimeError('Training geometry leaked into held-out selection')
    return records


def source_checks(records):
    def check_files(r):
        verify_files({str(Path(r['episode'])/n):h for n,h in r['files'].items()})
        verify_files({r['asset_path']:r['asset_sha256']})
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(check_files,records))
    cache={};checks=[];seeds={};ledgers={}
    for r in records:
        campaign=Path(r['campaign'])
        if str(campaign) not in cache:
            paths=([campaign/'ledger.sqlite3'] if (campaign/'ledger.sqlite3').is_file() else [])
            paths+=list((campaign/'trial_ledgers').glob('*.sqlite3'))
            found={}
            for path in paths:
                for row in ledger_rows(path):
                    if row['id'] in found and found[row['id']][0]!=row:
                        raise RuntimeError('Conflicting source ledger identities')
                    found[row['id']]=(row,path)
            cache[str(campaign)]=found
        row,ledger=cache[str(campaign)][r['id']]
        if row['episode']!=r['episode'] or row['object_id']!=r['object_id']:
            raise RuntimeError('Validation manifest/committed ledger identity mismatch')
        ep=Path(r['episode']);state=read(ep/'actual_entry.json')['snapshot']
        speed=float(np.max(np.abs(joint_values(state,state['right_arm'],'joint_velocities'))))
        item={'source_episode_id':r['id'],'source_family':r['family'],'object_id':r['object_id'],
              'source_policy':r['policy'],'source_episode':r['episode'],'source_ledger':str(ledger),
              'maximum_source_arm_velocity_rad_s':speed,'stationary_source_limit_rad_s':.01,
              'source_file_integrity_pass':True,'physical_dispatches':0}
        try:seed=source_seed(row)
        except ValueError as e:
            if str(e)!='Choose a measured stationary source entry':raise
            item.update(eligible=False,outcome='source_entry_not_stationary',reason=str(e))
        else:
            if read(ep/'episode.json')['split']!='validation':raise RuntimeError('Source split changed')
            seed['source_family']=r['family'];seed['source_policy']=r['policy']
            seed['files'].update({str(ep/n):h for n,h in r['files'].items()})
            seeds[r['id']]=seed;item.update(eligible=True,outcome='source_preflight_pass')
        checks.append(item);ledgers[r['id']]=str(ledger)
    return checks,seeds,ledgers


def prepare(out,compute_uuid,sim_uuid):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve prior validation registration')
    old=read(PARENT/'registration.json')
    proof=predecessors([PARENT,ROOT/'runs/nearfield_stage1_20261105_04/mustard_validation_01'])
    verify_files({**old['execution_source_files'],**old['dependencies']})
    records=selected_records();checks,seeds,source_ledgers=source_checks(records)
    out.mkdir(parents=True);space(out)
    write(out/'state.json',{'status':'preparing','physical_trials':0})
    auth={'user_message':'继续','authorized_source_slots':6,'automatic_retry':False,
          'basis':str(DATASET/'ACCEPTANCE.md'),'basis_sha256':sha(DATASET/'ACCEPTANCE.md'),
          'scope':'Original validation split synchronized recapture, original entries/scenes/gates; no source substitution',
          'model_training_authorized_here':False,'formal_comparison_authorized_here':False}
    write(out/'authorization.json',auth);write(out/'source_preflight.json',{'selected':checks})
    previous=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    shutil.copytree(previous,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
    # Preserve the already executed runtime/planner/capture snapshot byte-for-byte.
    # The new file only selects and registers the held-out sources.
    dest=source/'scripts/curobo_arx/nearfield/prepare_validation.py'
    shutil.copy2(Path(__file__),dest)
    execution={str(source/Path(p).relative_to(previous) if Path(p).is_relative_to(previous) else p):h
               for p,h in old['execution_source_files'].items()}
    execution[str(dest)]=sha(dest)
    reg=copy.deepcopy(old);deps=reg['dependencies'];trials=[]
    for index,r in enumerate(records):
        if r['id'] not in seeds:continue
        seed=seeds[r['id']];raw,star,priors,samples,_=episode_labels(r)
        first=min(s['raw_index'] for s in samples)
        last=next(s['raw_index'] for s in samples if s['labels']['C']=='GRASP')
        indices=list(range(first+1,last+1))
        if not indices:raise RuntimeError('No source motion path for validation')
        trial={'trial_id':f'{index:02d}__{r["object_id"]}__validation_{r["id"][:8]}',
            'seed_id':r['id'],'source_episode_id':r['id'],'source_family':r['family'],
            'source_policy':r['policy'],'source_split':'validation','object_id':r['object_id'],
            'renderer':seed['renderer'],'method':'C','policy':'C','placement':seed['placement'],
            'entry_world':seed['anchor_world'],'source_Gstar_world':star.tolist(),'G0_world':priors[1]['world'],
            'C_waypoints':[raw[i]['snapshot']['link_world']['right_eef_link'] for i in indices],
            'C_waypoint_raw_indices':indices,'offset_goal_world':None,
            'perturbation':{'type':'none','source_entry_unchanged':True},
            'truth_replay_only_no_policy_training':True,'no_retry':True}
        trials.append(trial);deps.update(seed['files'])
        catalog=copy.deepcopy(reg['catalog'])
        for asset in catalog['objects']:
            asset['collection_enabled']=asset['object_id']==r['object_id']
            if asset['collection_enabled']:
                if asset['split']!='validation' or asset['asset_sha256']!=r['asset_sha256']:
                    raise RuntimeError('Frozen catalog split/asset differs from source')
                asset['collection_placements']=[seed['placement']]
        path=out/'trial_ledgers'/(trial['trial_id']+'.sqlite3');path.parent.mkdir(exist_ok=True)
        ledger=Ledger(path);ledger.initialize(catalog,{**reg['quotas'],'first_object':r['object_id'],
            'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,
            'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1});ledger.close()
    for name,path in list(reg['reference_files'].items()):
        dest=out/'references'/(name+'.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
        reg['reference_files'][name]=str(dest);deps[str(dest)]=sha(dest)
    shutil.copy2(PARENT/'storage_policy.json',out/'storage_policy.json')
    for p in (BASE/'manifest.json',BASE/'paired_samples.jsonl',DATASET/'manifest.json',DATASET/'final_verification.json',
              DATASET/'ACCEPTANCE.md',PARENT/'registration.json',PARENT/'state.json',PARENT/'postrun_verification.json',
              out/'authorization.json',out/'source_preflight.json',out/'storage_policy.json',Path(__file__)):
        deps[str(p)]=sha(p)
    newly_pinned_helpers=[]
    for name in ('scripts/curobo_arx/nearfield/campaign.py','scripts/curobo_arx/nearfield/resources.py',
                 'scripts/curobo_arx/data_factory/launch.py','scripts/curobo_arx/data_factory/common.py',
                 'scripts/curobo_arx/data_factory/watchdog.py','scripts/curobo_arx/seed_augmentation/campaign.py'):
        if (previous/name).is_file() and sha(ROOT/name)!=sha(previous/name):
            raise RuntimeError('Coordinator differs from validated execution snapshot: '+name)
        if not (previous/name).is_file():newly_pinned_helpers.append(name)
        deps[str(ROOT/name)]=sha(ROOT/name)
    # Old supplement results remain in their original campaigns. They are
    # dependencies, never counted as this new validation batch's attempts.
    for key in ('carried_results','prior_results','prerequisite_results'):
        reg[key]=[]
    reg.update(stage='nearfield_heldout_synchronized_validation_v1',created_unix=time.time(),
        campaign_started_unix=time.time(),deadline_unix=time.time()+4*3600,wall_hours=4,
        seeds=seeds,seed_trials=trials,source_snapshot=str(source),execution_source_files=execution,
        source_identity=digest(execution),authorized_validation_slots=6,
        preflight_rejected_slots=[c for c in checks if not c['eligible']],
        max_physical_trials=len(trials),max_replay_trials=len(trials),replay_slots_total=len(trials),
        recovery_slots_total=0,no_automatic_retry=True,training_authorized_here=False,
        formal_48_trials_authorized_here=False,replay_predecessor=str(PARENT),
        requested_devices={'compute_uuid':compute_uuid,'sim_uuid':sim_uuid},
        validation_family_policy='Preserve source_family and original geometry split, including direct descendants',
        validation_sources=records,source_ledger_paths=source_ledgers)
    write(out/'predecessor_exit_receipt.json',proof)
    write(out/'source_change_receipt.json',{'parent':str(PARENT),'parent_source_identity':old['source_identity'],
        'execution_runtime_and_planner_unchanged':True,
        'new_file':str(source/'scripts/curobo_arx/nearfield/prepare_validation.py'),
        'new_file_sha256':sha(source/'scripts/curobo_arx/nearfield/prepare_validation.py'),
        'additional_host_helpers_not_archived_in_predecessor':newly_pinned_helpers,
        'changes':'Held-out source selection and bounded registration only','frozen_robot_scene_evaluator_unchanged':True})
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    write(out/'sources.json',{'basis':'Existing accepted ARX near-field capture/action protocol; no new scientific method',
        'files':{str(DATASET/'ACCEPTANCE.md'):sha(DATASET/'ACCEPTANCE.md'),str(PARENT/'registration.json'):sha(PARENT/'registration.json')}})
    (out/'research.md').write_text('# Held-out synchronized capture\n\nSix fixed historical sources; '+
        f'{len(trials)} pass original stationary-source admission and can execute. '+
        'Preflight source rejections stay in the declared six slots; no replacements or retries. '+
        'Preserve original scene, entry, renderer, robot, physical and success gates. '+
        'Same source-waypoint C lookahead teacher, explicit GRASP, same-frame wrist capture. '+
        'No training or formal policy comparison in this campaign.\n')
    (out/'research_log.md').write_text('Registered all six source outcomes before new physical commands.\n')
    verify_files({**execution,**deps})
    write(out/'state.json',{'status':'prepared','authorized_source_slots':6,'scheduled_slots':len(trials),
                          'preflight_rejected':6-len(trials),'physical_trials':0})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compute-uuid',required=True);p.add_argument('--sim-uuid',required=True)
    a=p.parse_args();print(prepare(a.out,a.compute_uuid,a.sim_uuid))
