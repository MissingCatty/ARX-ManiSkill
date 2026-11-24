"""Immutable 24+6 advisor33 registration; preserve predecessor results and STOPs."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices
from training.nearfield_arx.lateral_campaign import ledgers
from .advisor31_campaign import runtime_dependencies,exited
from .advisor32_spec import make_trials,validate_trials
from .visual_grounding_spec import make_trials as shifted_seeds,expected_reset
from .closedloop30_policy import validate_checkpoints

PARENT=ROOT/'runs/nearfield_advisor31_20261110_01'
DONOR=ROOT/'runs/nearfield_closedloop30_20261110_01'
OFFLINE=ROOT/'runs/nearfield_advisor32_20261111_01'
AUTHORITY=ROOT/'research_plan/chat/gemini回复/gemini回复33.md'


def prepare(out,policy_index,compute_index,sim_index):
    out=Path(out).resolve();space(out)
    if (out/'registration.json').exists() or (out/'source_snapshot').exists():raise FileExistsError('Immutable registration exists')
    for name in ('geometry_preflight.json','spam_model_preflight/result.json','spam_path_preflight/result.json'):
        if not read(out/name)['pass']:raise ValueError('Required preflight failed: '+name)
    geometry=read(out/'geometry_preflight.json')
    check=subprocess.run([sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_advisor33',
        'scripts.curobo_arx.nearfield.test_advisor32','training.nearfield_arx.test_closure_records','-v'],cwd=ROOT,capture_output=True,text=True,timeout=180)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json',{'pass':check.returncode==0,'log_sha256':sha(out/'offline_checks.log'),'physical_trials':0})
    if check.returncode:raise RuntimeError('Offline fault tests failed')
    old=read(PARENT/'registration.json');donor=read(DONOR/'registration.json')
    predecessor=exited(PARENT)
    if not (PARENT/'STOP').exists():raise RuntimeError('Preserve previous STOP')
    canonical=read(out/'spam_model_preflight/result.json')['selected']
    trials=make_trials(donor,{'Gstar_canon_world':canonical['goal_world']});_,seeds=shifted_seeds(donor)
    spam_original=next(t['seed_id'] for t in trials if t['phase']=='spam_canonical')
    sid=digest(['advisor33_canonical',spam_original,canonical['goal_world']])[:32]
    seed=copy.deepcopy(seeds[spam_original]);seed.update(seed_id=sid,source_seed_id=spam_original,
        source_policy='Canonical orthogonal model-qualified proposal; not a demonstrated teacher')
    seed['goal'].update(grasp_world=canonical['goal_world'],raw_grasp_world=canonical['goal_world'],
        goal_id=sid,unscoped_goal_id=sid,candidate_source='advisor33_model_qualified_short_axis_canonical',
        qualification={},canonical_screen=str(out/'spam_model_preflight/result.json'))
    seed['goal']['target_contact_world']=(np.array(canonical['goal_world'])@np.array([-.006,0,.0375,1.]))[:3].tolist()
    seeds[sid]=seed
    for trial in trials:
        trial.update(trial_id=trial['trial_id'].replace('a32_','a33_',1),policy='advisor33_registered')
        if trial['phase']=='spam_canonical':
            trial.update(seed_id=sid,case_id='spam_canonical_'+trial['condition'],source_split='historical_training_source',
                canonical_IK_and_collision_preflight_required=False,canonical_preflight_passed=True)
    seeds={k:v for k,v in seeds.items() if k in {t['seed_id'] for t in trials}}
    validate_trials(trials)
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for n in ('tmp','curobo'):(source.parent/n).symlink_to(ROOT.parent/n,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    package=ROOT/'scripts/curobo_arx/nearfield'
    added=sorted(set([*package.glob('advisor33_*.py'),*package.glob('advisor32_*.py'),
        package/'runtime_advisor33.py',package/'test_advisor33.py',package/'test_advisor32.py']))
    for p in added:
        dest=source/p.relative_to(ROOT);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    reg=copy.deepcopy(old);deps=runtime_dependencies(old,seeds)
    donors=read(OFFLINE/'stale_donors/manifest.json')
    for v in donors.values():
        for r in v['records']:deps.update(r['files'])
    inventory=devices();by={d['index']:d for d in inventory}
    roles={r+'_uuid':by[i]['uuid'] for r,i in (('policy',policy_index),('compute',compute_index),('sim',sim_index))}
    if roles['sim_uuid'] in (roles['policy_uuid'],roles['compute_uuid']):raise ValueError('Dedicated simulator GPU required')
    for k in ('carried_results','prior_results','prerequisite_results'):reg[k]=[]
    reg.update(stage='advisor33_zero_prior_finger_box',source_snapshot=str(source),execution_source_files=execution,
        source_identity=digest(execution),seeds=seeds,seed_trials=trials,full_stale_donors=donors,
        created_unix=time.time(),campaign_started_unix=time.time(),deadline_unix=time.time()+12*3600,wall_hours=12,
        max_physical_trials=30,global_physical_budget=30,max_replay_trials=0,replay_slots_total=0,recovery_slots_total=0,
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor33',policy_module='scripts.curobo_arx.nearfield.advisor33_policy',
        worker_module='scripts.curobo_arx.nearfield.advisor33_worker',coordinator_module='scripts.curobo_arx.nearfield.advisor33_launch',
        requested_devices=roles,policy_memory_floor_mib=9500,training_authorized_here=False,
        no_automatic_retry=True,automatic_attempt_retries=0,automatic_startup_retries=0,
        recorded_state_convention='measured_EEF_and_joints_no_G0_codec',training_export=False,
        advisor33_policy={'D_G0_after_H':False,'C_prompt_and_weights_unchanged':True,'D_model_decisions_maximum':2,
            'D_moving_refinements_maximum':2,'local_translation_max_m':.065,'local_rotation_max_degrees':25,
            'finger_box':'frozen pad X, measured jaw Y, EEF origin through mesh fingertip Z',
            'first_axial':'observed surface to actual fingertip plane, within total65mm',
            'stationary_wait_wall_seconds':10,'position_window_s':2,'position_window_diameter_m':.0005,
            'full_stale':'both neural RGB-D pairs, actual safety observations fresh',
            'slot_timeout':'preserve partial evidence, destroy old World, validate next reset, no retry'},
        execution_authority={'file':str(AUTHORITY),'sha256':sha(AUTHORITY),'full_30_authorized':True})
    reg['learned_policy']['gpu_uuid']=roles['policy_uuid'];validate_checkpoints(reg)
    plan=ROOT/'research_plan/docs/nearfield_advisor33_20261111.md'
    shutil.copy2(plan,out/'PLAN.md');shutil.copy2(plan,out/'research.md');shutil.copy2(PARENT/'storage_policy.json',out/'storage_policy.json')
    write(out/'authorization.json',reg['execution_authority']);write(out/'case_manifest.json',{'slots':trials,'maximum_slots':30,'retries':0})
    write(out/'expected_resets.json',{t['trial_id']:expected_reset(seeds[t['seed_id']],t) for t in trials})
    write(out/'predecessor_exit_receipt.json',predecessor)
    write(out/'resource_plan.json',{'inventory':inventory,'roles':roles,'sim_minimum_free_mib':9000,
        'policy_compute_combined_minimum_free_mib':12000,'separate_compute_minimum_free_mib':5000,
        'policy_memory_fraction':.35,'other_users_jobs_untouched':True})
    write(out/'source_change_receipt.json',{'added':[str(p.relative_to(ROOT)) for p in added],
        'model_robot_scene_camera_success_thresholds_unchanged':True,'post_H_D_no_prior':True,
        'stop_rule_changed_as_authorized':reg['advisor33_policy'],'historical_results_STOPs_preserved':True})
    write(out/'sources.json',{'advisor':reg['execution_authority'],'frozen_checkpoints':reg['learned_policy']['checkpoints'],
        'no_training':True,'engineering_method_not_paper_reproduction':True})
    for p in (AUTHORITY,PARENT/'registration.json',PARENT/'STOP',DONOR/'registration.json',OFFLINE/'proposal_manifest.json',OFFLINE/'stale_donors/manifest.json'):
        deps[str(p)]=sha(p)
    for folder in (out/'spam_model_preflight',out/'spam_path_preflight'):
        for p in folder.rglob('*.json'):deps[str(p)]=sha(p)
    for p in out.iterdir():
        if p.is_file() and p.suffix in ('.json','.md') and p.name not in ('state.json','research_log.md'):deps[str(p)]=sha(p)
    command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,sys,pathlib;import scripts.curobo_arx.nearfield.runtime_advisor33; paths=json.load(sys.stdin); '
        'missing=[p for p in paths if not pathlib.Path(p).is_file()];print(json.dumps({"missing":missing,"checked":len(paths)}));sys.exit(bool(missing))']
    probe=subprocess.run(command,input=json.dumps(list({*deps,*execution})),capture_output=True,text=True,timeout=120)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'stdout':probe.stdout,'stderr':probe.stderr,'physical_trials':0,'image_download':False})
    if probe.returncode:raise RuntimeError('Container import/path preflight failed')
    reg['dependencies']=deps;ledgers(out,reg);write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    began=time.time();verify_files({**execution,**deps})
    write(out/'full_preflight.json',{'pass':True,'checked':len({**execution,**deps}),'seconds':time.time()-began,'physical_trials':0})
    (out/'research_log.md').write_text('30 slots registered before physical execution. Predecessors retained.\n')
    write(out/'state.json',{'status':'prepared','scheduled_slots':30,'physical_trials':0})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--policy-index',type=int,required=True)
    p.add_argument('--compute-index',type=int,required=True);p.add_argument('--sim-index',type=int,required=True)
    a=p.parse_args();print(prepare(a.root,a.policy_index,a.compute_index,a.sim_index))
