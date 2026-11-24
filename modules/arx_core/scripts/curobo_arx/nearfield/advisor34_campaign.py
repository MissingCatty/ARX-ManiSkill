"""Freeze advisor34 into a fresh 24+6-slot campaign, retaining every old result."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices
from training.nearfield_arx.lateral_campaign import ledgers
from .advisor31_campaign import exited
from .advisor32_spec import validate_trials
from .closedloop30_policy import validate_checkpoints
from .visual_grounding_spec import expected_reset

PREVIOUS=ROOT/'runs/nearfield_advisor33_native_20261111_07'
AUTHORITY=ROOT/'research_plan/chat/gemini回复/gemini回复34.md'
PLAN=ROOT/'research_plan/docs/nearfield_advisor34_20261111.md'


def prepare(out,policy_index,compute_index,sim_index):
    out=Path(out).resolve()
    if (out/'registration.json').exists() or (out/'source_snapshot').exists() or (out/'STARTED.json').exists():
        raise FileExistsError('Never overwrite a prepared/started campaign')
    offline=read(out/'offline/verification.json')
    if not offline['pass'] or offline['gate_positives']!=5 or offline['gate_false_rejections']:
        raise RuntimeError('Required archived ARX gate preflight incomplete')
    old=read(PREVIOUS/'registration.json');previous=read(PREVIOUS/'advisor33_verification.json')
    if (not previous['complete'] or not previous['record_integrity_pass'] or previous['scheduled']!=30
        or previous['unknown']!=1 or previous['completed']!=29 or previous['unexecuted']):
        raise RuntimeError('Previous campaign accounting not reconciled')
    if read(PREVIOUS/'state.json')['status']!='complete':raise RuntimeError('Previous coordinator incomplete')
    exit_receipt=exited(PREVIOUS)
    verify_files({**old['execution_source_files'],**old['dependencies'],**previous['evidence'],**offline['files']})
    for interrupted in old['supplemental_interruptions'].values():verify_files(interrupted['evidence'])
    out.mkdir(exist_ok=True);space(out)
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    added=sorted((ROOT/'scripts/curobo_arx/nearfield').glob('advisor34_*.py'))
    added += [ROOT/'scripts/curobo_arx/nearfield'/name for name in ('runtime_advisor34.py','test_advisor34.py')]
    for p in added:
        dest=source/p.relative_to(ROOT);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    command=[sys.executable,'-m','unittest',
        'scripts.curobo_arx.nearfield.test_advisor34','scripts.curobo_arx.nearfield.test_advisor33',
        'scripts.curobo_arx.nearfield.test_stop_history','scripts.curobo_arx.nearfield.test_advisor33_boundary',
        'training.nearfield_arx.test_native_mask','-v']
    check=subprocess.run(command,cwd=source,env={**os.environ,'ARX_TEST_PROJECT_ROOT':str(ROOT)},capture_output=True,text=True,timeout=180)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json',{'pass':check.returncode==0,'command':command,
        'log_sha256':sha(out/'offline_checks.log'),'physical_trials':0,'model_inferences':0})
    if check.returncode:raise RuntimeError('Frozen-source offline tests failed')
    inventory=devices();by={d['index']:d for d in inventory}
    roles={r+'_uuid':by[i]['uuid'] for r,i in (('policy',policy_index),('compute',compute_index),('sim',sim_index))}
    if len(set(roles.values()))!=3:raise ValueError('Use separate simulator, planner and policy GPUs')
    reg=copy.deepcopy(old);trials=copy.deepcopy(old['seed_trials']);mapping=[]
    for row in trials:
        before=row['trial_id'];row.update(trial_id=before.replace('a33m_','a34_',1),policy='advisor34_clock_near335_gravity_column')
        mapping.append({'old_trial_id':before,'new_trial_id':row['trial_id']})
    validate_trials(trials)
    for a,b in zip(old['seed_trials'],trials):
        if {k:v for k,v in a.items() if k not in ('trial_id','policy')}!={k:v for k,v in b.items() if k not in ('trial_id','policy')}:
            raise RuntimeError('Benchmark parameters changed')
    for key in ('carried_results','prior_results','prerequisite_results','carried_terminal_ids','ancestor_campaigns'):reg[key]=[]
    reg['supplemental_interruptions']={}
    reg.pop('correction_scope',None)
    reg.pop('advisor33_policy',None)
    now=time.time()
    reg.update(stage='advisor34_clock_near335_column_30',continuation_of=str(PREVIOUS),
        source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor34',
        policy_module='scripts.curobo_arx.nearfield.advisor34_policy',
        coordinator_module='scripts.curobo_arx.nearfield.advisor34_launch',
        seed_trials=trials,remaining_trial_ids=[t['trial_id'] for t in trials],
        created_unix=now,campaign_started_unix=now,deadline_unix=now+12*3600,wall_hours=12,
        requested_devices=roles,global_physical_budget=30,max_physical_trials=30,
        training_authorized_here=False,new_independent_mask_campaign=False,new_independent_runtime_campaign=True,
        execution_authority={'file':str(AUTHORITY),'sha256':sha(AUTHORITY),'full_30_authorized':True},
        advisor34_policy={'D_G0_after_H':False,'C_prompt_weights_unchanged':True,'D_weights_unchanged':True,
            'native_mask_and_model_input_unchanged':True,'D_model_decisions_maximum':2,'D_moving_refinements_maximum':2,
            'C_model_decisions_maximum':12,'preclosure_axial_controller_moves_maximum':1,
            'first_plane_z_EE_m':.0335,'shallow_trigger_z_EE_m':.0365,
            'preclosure_compensation_all_arms':True,'second_prediction_compensation_D':True,
            'local_translation_max_m':.065,'local_rotation_max_degrees':25,
            'gravity_column_max_down_m':.025,'gravity_column_axial_band_m':[.020,.048],
            'direct_visible_gate_retained':True,'inferred_volume_not_model_input':True,
            'position_window_s':2,'position_window_min_samples':11,'position_window_diameter_m':.0005,
            'differential_joint_speed_rad_s':.01,'maximum_wait_sim_s':10,'maximum_wait_wall_s':120,
            'continuous_raw_history_retained':True,'original_lift_candidates_and_range_handler_unchanged':True})
    reg['learned_policy']['gpu_uuid']=roles['policy_uuid']
    reg['legacy_inherited_policy_request_limits']=copy.deepcopy(reg['learned_policy']['max_policy_calls'])
    # Service requests include non-neural final gates. One axial-only move
    # requires one additional final observation, without another neural call.
    reg['learned_policy']['max_policy_calls']={'C':14,'D':4}
    reg['learned_policy']['max_neural_calls']={'C':12,'D':2}
    validate_checkpoints(reg)
    deps=dict(old['dependencies']);deps.update(previous['evidence']);deps.update(offline['files'])
    for p in (AUTHORITY,PLAN,PREVIOUS/'registration.json',PREVIOUS/'advisor33_verification.json',
              PREVIOUS/'completion_receipt.json',out/'offline/verification.json'):
        deps[str(p)]=sha(p)
    for name in ('research.md','PLAN.md'):shutil.copy2(PLAN,out/name)
    shutil.copy2(PREVIOUS/'storage_policy.json',out/'storage_policy.json')
    write(out/'predecessor_exit_receipt.json',exit_receipt)
    write(out/'previous_results_retained.json',{'old_physical_attempts':30,'old_native_terminals':29,
        'old_unknown':1,'old_dual_stop_successes':4,'new_slots':30,'old_results_pooled':False,
        'old_STOPs_retained':True,'interruption_evidence':old['supplemental_interruptions'],
        'exact_matched_case_map':mapping})
    write(out/'case_manifest.json',{'slots':trials,'maximum_slots':30,'retries':0,'matched_case_map':mapping})
    write(out/'expected_resets.json',{t['trial_id']:expected_reset(reg['seeds'][t['seed_id']],t) for t in trials})
    write(out/'resource_plan.json',{'inventory':inventory,'roles':roles,'other_users_jobs_untouched':True})
    write(out/'source_change_receipt.json',{'new_modules':{str(p):sha(p) for p in added},
        'all_previous_snapshot_source_hashes_preserved':True,
        'model_robot_scene_camera_physics_initial_conditions_success_thresholds_unchanged':True,
        'changes':reg['advisor34_policy'],'original_30_retained_new_30_separate':True})
    write(out/'sources.json',{'advisor':reg['execution_authority'],'previous':str(PREVIOUS),
        'method':'Three explicitly requested engineering changes; not a newly trained policy',
        'diagnostic_causal_claims':'Hypotheses; combined intervention cannot isolate three individual effects'})
    for p in out.iterdir():
        if p.is_file() and p.suffix in ('.json','.md'):deps[str(p)]=sha(p)
    probe_command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,sys,pathlib;import scripts.curobo_arx.nearfield.runtime_advisor34; '
        'import scripts.curobo_arx.nearfield.advisor34_policy;paths=json.load(sys.stdin); '
        'missing=[p for p in paths if not pathlib.Path(p).is_file()]; '
        'print(json.dumps({"missing":missing,"checked":len(paths)}));sys.exit(bool(missing))']
    probe=subprocess.run(probe_command,input=json.dumps(list({*deps,*execution})),capture_output=True,text=True,timeout=120)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'stdout':probe.stdout,
        'stderr':probe.stderr,'physical_trials':0,'image_download':False})
    if probe.returncode:raise RuntimeError('Container import/path preflight failed')
    deps[str(out/'container_preflight.json')]=sha(out/'container_preflight.json')
    reg['dependencies']=deps
    ledgers(out,reg);write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps})
    write(out/'full_preflight.json',{'pass':True,'checked':len({**execution,**deps}),'physical_trials':0})
    (out/'research_log.md').write_text('Advisor34: new30 separately registered; previous30 retained, including one unknown. No retries, no training.\n')
    write(out/'state.json',{'status':'prepared','scheduled_slots':30,'physical_trials':0})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root')
    for role in ('policy','compute','sim'):p.add_argument('--'+role+'-index',type=int,required=True)
    a=p.parse_args();print(prepare(a.root,a.policy_index,a.compute_index,a.sim_index))
