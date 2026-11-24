"""Bounded P2 ARX teacher pilot; fresh observations, fixed physics, object split."""
from arx_project.layout import canonical_path
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

# Resolve-only must work with system Python and no numerical/GPU/RPC packages.
if __name__ == '__main__' and '--resolve-only' in sys.argv:
    from interaction_contract import resolve_cli
    resolve_cli(sys.argv[1:], 'teacher')
    raise SystemExit(0)

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(Path(__file__).parent)]
from interaction_contract import load as load_contract, reject_overrides, resolve as resolve_interaction, prepare_environment
from client import Client
from egad49 import compare_configuration,INITIAL_SHA
from egad_evaluate import evaluate
from compare_chunks import run_process
from grasp_handoff import RULE
from summarize_chunks import encode
from gpu_policy import resolve,compute_env


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--neural-gpu',required=True)
    parser.add_argument('--bottle-mailbox',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,default=ROOT/'runs/lilo_egad49_20261026_eval01/runtime/mailbox')
    parser.add_argument('--schedule-file',type=Path,help='Explicit object/split/jitter-index schedule for a separately registered continuation')
    parser.add_argument('--previous-collection',type=Path,help='Preserve prior attempts and successful sources; never rerun them')
    parser.add_argument('--filter-handoff-candidates',action='store_true',help='Continue to another grasp when the unchanged shared-entry gates reject a proposal')
    parser.add_argument('--handoff-candidate-budget',type=int,default=12,choices=range(1,17))
    parser.add_argument('--resolve-only', action='store_true', help='Resolve common semantics without GPU/RPC; use --resolve-only --help for its minimal arguments')
    parser.add_argument('--interaction-contract', type=Path, help='Explicit common ARX interaction configuration; legacy experiments remain unchanged without it')
    args=parser.parse_args()
    contract = load_contract(args.interaction_contract) if args.interaction_contract else None
    if contract:
        reject_overrides(os.environ)
        if args.handoff_candidate_budget != contract['handoff_candidate_budget']:
            raise ValueError('Candidate budget conflicts with interaction contract')
        args.filter_handoff_candidates = True

    args.neural_gpu=resolve(args.neural_gpu)['uuid'];args.planner_gpu=resolve(args.planner_gpu)['uuid']
    os.environ.update(compute_env(os.environ,args.planner_gpu))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.pop('ARX_CUROBO_VALIDATION_DIR',None);os.environ.pop('ARX_RESEARCH_GUARD_FILE',None)
    locks=[]
    for box in sorted({args.mailbox.resolve(),args.bottle_mailbox.resolve()}):
        lock=(box/'p0_execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert Client(box).request('snapshot',timeout=60).get('command_id') is None
    probe=ROOT/'runs/lilo_P2_teacher_probe_20261028_01'
    reused=probe/'bottle_grasp_frame_r1'
    assert evaluate(reused)['pass'] and not read(reused/'teacher_recording.json').get('error')
    initial=ROOT/'runs/tabletop_trials/initial_setup.json';assert sha(initial)==INITIAL_SHA
    catalog_path=ROOT/'runs/lilo_egad49_20261026_eval01/runtime/historical_reference/catalog.json'
    catalog={item['name']:item for item in read(catalog_path)}
    fixtures={
        'bottle':{'mailbox':args.bottle_mailbox.resolve(),'ops':'http://127.0.0.1:8102',
            'reference':read(ROOT/'runs/lilo_execution_20261026_tracking02/setup.json')['fingerprint'],
            'viewport':read(ROOT/'scripts/curobo_arx/static_pick/reference/historical_reference/viewport_initial.json')},
        'other':{'mailbox':args.mailbox.resolve(),'ops':'http://127.0.0.1:8101',
            'reference':read(ROOT/'runs/lilo_egad49_20261026_eval01/runtime/historical_reference/setup.json')['fingerprint'],
            'viewport':read(ROOT/'runs/lilo_egad49_20261026_eval01/original_viewport.json')}}
    jitter=[[0.]*6,[.005,-.005,.005,.04,-.04,.04],[-.005,.005,.005,-.04,.04,-.04]]
    schedule=[('rubiks_cube',0),('lemon_01',0),('bbq_sauce_bottle',1),('bbq_sauce_bottle',2),
              ('rubiks_cube',1),('rubiks_cube',2),('lemon_01',1),('lemon_01',2)]
    roles={'bbq_sauce_bottle':'train','rubiks_cube':'train','lemon_01':'validation'}
    prior=[]
    if args.previous_collection:
        previous=args.previous_collection.resolve()
        if read(previous/'state.json')['status']=='running':raise ValueError('Prior collection is still active')
        prior=read(previous/('all_attempts.json' if (previous/'all_attempts.json').exists() else 'results.json'))
    if args.schedule_file:
        specification=read(args.schedule_file)
        roles=specification['roles']
        if set(roles.values())!={'train','validation'}:raise ValueError('An explicit train/validation object split is required')
        schedule=[(row['object_name'],row['jitter_index']) for row in specification['schedule']]
        if any(name not in roles or index not in range(len(jitter)) for name,index in schedule):raise ValueError('Invalid registered collection schedule')
    qualified={}
    if args.schedule_file:
        for item in specification['schedule']:
            if not item.get('fixed_selection'):continue
            selection=Path(item['fixed_selection']).resolve();source=selection.parent
            if read(source/'scenario.json')['asset']!=item['object_name'] or not evaluate(source)['pass']:
                raise ValueError('A fixed teacher target must come from an actually successful run of the same object')
            qualified[(item['object_name'],item['jitter_index'])]=selection
    frozen=[ROOT/p for p in ('arx-sim/config/sim.yaml','arx-sim/config/runtime.yaml','arx-sim/config/controllers.yaml',
        'configs/calibration.yaml','asset/robots/arx_x5_lift/1.0.0/usd/robot.usd')]
    for name in roles:frozen.append(canonical_path(ROOT/'asset'/catalog[name]['container_path'].removeprefix('/workspace/asset/')))
    source_names=['lilo_vla/collect_shared_teacher.py','static_pick/reproduce_record.py','grasp_execute.py',
        'grasp_input.py','prepare.py','model_archive.py','gripper_update.py','optimize_path.py','settling.py',
        'lilo_vla/grasp_handoff.py','lilo_vla/observations.py','lilo_vla/egad_evaluate.py',
        'lilo_vla/interaction_contract.py','lilo_vla/check_reach_target.py']
    sources=[ROOT/'scripts/curobo_arx'/name for name in source_names]
    if contract:sources.append(ROOT/'configs/lilo_vla/model_compatibility.json')
    snapshot=out/'registered_source_snapshot';snapshot.mkdir()
    for path in sources:shutil.copy2(path,snapshot/path.name)
    rule = contract['handoff_rule'] if contract else RULE
    rule_file=out/'handoff_rule.json';write(rule_file,rule)
    episodes=[]
    for name,index in schedule:
        asset=catalog[name]
        scenario={'name':f'{name}_j{index}','target':'test_object','asset':name,
            'placement':{'position':[4.048,2.08,.679+asset['size_m'][2]/2+.012]}}
        episodes.append({'label':scenario['name'],'object_name':name,'split':roles[name],
            'fixture':'bottle' if name=='bbq_sauce_bottle' else 'other',
            'scenario':scenario,'entry_delta_world':jitter[index]})
        if (name,index) in qualified:episodes[-1]['fixed_selection']=str(qualified[name,index])
    registration={'scope':'P2 teacher pilot; A1 is diagnostic only and excluded from both dataset splits',
        'train_objects':sorted(n for n,role in roles.items() if role=='train'),
        'validation_objects':sorted(n for n,role in roles.items() if role=='validation'),
        'reused_successful_source':None if contract else str(reused),'reused_recording_sha256':None if contract else sha(reused/'teacher_raw/samples.jsonl'),
        'episodes':episodes,'new_attempts':len(episodes),'attempt_budget_seconds':900,'rule':rule,
        'interaction_contract':contract,
        'previous_collection':str(args.previous_collection.resolve()) if args.previous_collection else None,
        'prior_results_sha256':sha(args.previous_collection.resolve()/('all_attempts.json' if (args.previous_collection/'all_attempts.json').exists() else 'results.json')) if args.previous_collection else None,
        'candidate_handoff_filter':args.filter_handoff_candidates,'handoff_candidate_budget':args.handoff_candidate_budget,
        'teacher_qualified_selection_sha256':{str(p):sha(p) for p in qualified.values()},
        'initial_setup_sha256':sha(initial),'catalog_sha256':sha(catalog_path),
        'physical_source_sha256':{str(p):sha(p) for p in frozen},
        'algorithm_sha256':{str(p):sha(p) for p in sources},
        'rule_sha256':sha(rule_file),'neural_gpu':args.neural_gpu,'planner_gpu':args.planner_gpu,
        'renderers':{n:f['reference']['invariant']['configured']['simulation']['renderer'] for n,f in fixtures.items()},
        'declared_initial_placement_rule':'Existing clear-table task: x=4.048, y=2.08, z=0.679+catalog_z_extent/2+0.012; no asset or physics adjustment',
        'entry_augmentation_units':['m','m','m','rad','rad','rad'],'seed':0,'registered_unix':time.time()}
    write(out/'registration.json',registration)
    storage=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if storage.exists():shutil.copy2(storage,out/'storage_policy.json')
    else:write(out/'storage_policy.json',{'local_first':True,'reserve_bytes':1073741824,'historical_file':'Unavailable, not assumed verified'})
    results=[];state={'status':'running','completed':0,'successes':0,'reused_successes':0 if contract else 1};write(out/'state.json',state)
    neural=canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python');planner=ROOT.parent/'curobo/.venv/bin/python'
    def check_sources():
        for name,digest in {**registration['physical_source_sha256'],**registration['algorithm_sha256'],**registration['teacher_qualified_selection_sha256']}.items():
            if sha(name)!=digest:raise RuntimeError('Registered physical source/teacher algorithm changed: '+name)
        assert sha(rule_file)==registration['rule_sha256']
    try:
        for episode in episodes:
            check_sources()
            if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
            trial=out/episode['label'];trial.mkdir();(trial/'environment_evidence').mkdir()
            fixture=fixtures[episode['fixture']];client=Client(fixture['mailbox'])
            write(trial/'scenario.json',episode['scenario']);write(trial/'entry_perturbation.json',{'delta_world':episode['entry_delta_world']})
            write(trial/'teacher_episode_metadata.json',episode)
            state.update(active=episode['label']);write(out/'state.json',state)
            started=time.monotonic();deadline=started+900
            setup=client.request('dynamic_setup',scenario=episode['scenario'],timeout=240);write(trial/'setup.json',setup)
            mismatch=compare_configuration(fixture['reference'],setup['fingerprint'])
            expected=read(initial);bounds=setup['snapshot']['joint_limits']
            clipped=[min(max(q,lo),hi) for q,(lo,hi) in zip(expected['joint_positions'],bounds)]
            if setup['initial_conditions']['joint_positions']!=clipped:mismatch.append('historical initial joints')
            if setup['initial_conditions']['joint_names']!=expected['joint_names']:mismatch.append('historical joint names')
            if setup['physics_steps']!=360:mismatch.append('historical settling steps')
            if abs(setup['captured_sim_time']-setup['reset_sim_time']-3.)>.001:mismatch.append('historical 3s settling')
            actual=setup['initial_conditions']['objects']['asset:'+episode['object_name']]
            if actual['position']!=episode['scenario']['placement']['position'] or actual['quaternion_wxyz']!=[1.,0.,0.,0.]:mismatch.append('declared object placement')
            write(trial/'setup_verification.json',{'pass':not mismatch,'mismatches':mismatch,'historical_physical_properties_equal':not mismatch})
            if mismatch:raise RuntimeError('Frozen physics/reset mismatch: '+str(mismatch[:12]))
            write(trial/'active_asset.json',setup['asset'])
            client.request('grasp_viewport_state',restore=fixture['viewport'])
            write(trial/'viewport_initial.json',client.request('grasp_viewport_state'))
            write(trial/'guard.json',{'invariant_sha256':'Initial setup fingerprint archived','file_sha256':{}})
            env=os.environ.copy()
            for key in ('ARX_LILO_POLICY_URL','ARX_LILO_HANDOFF_OBSERVATION','GRASP_FIXED_SELECTION','GRASP_CAMERA_VIDEO_DIR','ARX_LILO_FILTER_CANDIDATE_HANDOFF'):env.pop(key,None)
            env.update(ARX_CUROBO_VALIDATION_DIR=str(client.root),ARX_LILO_HANDOFF_RULE_FILE=str(rule_file),
                ARX_LILO_ENTRY_PERTURBATION_FILE=str(trial/'entry_perturbation.json'),
                ARX_LILO_TEACHER_DIR=str(trial/'teacher_raw'),ARX_OPS_URL=fixture['ops'],
                ARX_OBJECT_LABEL=episode['object_name'],ARX_RESEARCH_AUDIT='0',PYTHONUNBUFFERED='1')
            if args.filter_handoff_candidates:
                env.update(ARX_LILO_FILTER_CANDIDATE_HANDOFF='1',ARX_LILO_HANDOFF_CANDIDATE_BUDGET=str(args.handoff_candidate_budget))
            if episode.get('fixed_selection'):
                selection=trial/'teacher_qualified_selection.json';shutil.copy2(episode['fixed_selection'],selection)
                env['GRASP_FIXED_SELECTION']=str(selection)
            if contract:
                resolved = resolve_interaction(contract, episode['object_name'], 'teacher',
                    delta=episode['entry_delta_world'], goal_source=episode.get('fixed_selection'))
                env = prepare_environment(env, trial, resolved)
            command=[neural,ROOT/'scripts/curobo_arx/static_pick/reproduce_record.py','--output',trial,
                '--reference-run',ROOT/'scripts/curobo_arx/static_pick/reference','--mailbox',client.root,
                '--curobo-python',planner,'--neural-gpu',args.neural_gpu,'--planner-gpu',args.planner_gpu,
                '--trajopt-policy','interactive','--no-audit','--skip-encoding']
            if episode.get('fixed_selection'):command+=['--fixed-selection',selection]
            print('P2_COLLECT_START',episode['label'],flush=True);error=None
            try:
                code=run_process(command,env,trial/'pipeline.log',deadline)
                if code:raise RuntimeError('Teacher pipeline exited '+str(code))
            except Exception:error=traceback.format_exc();write(trial/'collection_error.json',{'error':error})
            write(trial/'boundary_snapshot.json',client.request('snapshot',timeout=300))
            if (trial/'teacher_raw').exists() and not (trial/'teacher_recording.json').exists():
                write(trial/'teacher_recording.json',client.request('grasp_lilo_record_stop',timeout=120))
            if (trial/'frames').exists() and not (trial/'recording_stop.json').exists():
                write(trial/'recording_stop.json',client.request('grasp_video_stop',timeout=120))
            evaluation=evaluate(trial)
            recording=read(trial/'teacher_recording.json') if (trial/'teacher_recording.json').exists() else {'error':'Recording absent'}
            ready=bool(evaluation['pass'] and not recording.get('error') and recording.get('samples',0)>=8)
            result={**episode,'pass':evaluation['pass'],'demonstration_ready':ready,'elapsed_seconds':time.monotonic()-started,
                'evaluation':evaluation,'recording':recording,'error':error,'source':str(trial)}
            write(trial/'result.json',result);results.append(result);write(out/'results.json',results)
            state.update(completed=len(results),successes=sum(r['pass'] for r in results),
                valid_new_demonstrations=sum(r['demonstration_ready'] for r in results));write(out/'state.json',state)
            print('P2_COLLECT_RESULT',json.dumps({'label':episode['label'],'pass':result['pass'],'demo_ready':ready,
                'hold_min_rise_m':evaluation.get('hold_min_rise_m'),'error':evaluation.get('reported_error')}),flush=True)
            check_sources()
            if (trial/'frames/frames.json').exists():
                try:encode(trial)
                except Exception:write(trial/'encoding_error.json',{'error':traceback.format_exc()})
        combined=prior+results
        if contract:
            from interaction_contract import validate_episode
            for result in combined:
                if result['demonstration_ready']:validate_episode(result['source'], contract)
        if len({r['source'] for r in combined})!=len(combined):raise ValueError('Duplicate physical attempt in continuation')
        train=([] if contract else [str(reused)])+[r['source'] for r in combined if r['demonstration_ready'] and r['split']=='train' and roles.get(r['object_name'])=='train']
        validation=[r['source'] for r in combined if r['demonstration_ready'] and r['split']=='validation' and roles.get(r['object_name'])=='validation']
        write(out/'all_attempts.json',combined)
        write(out/'dataset_sources.json',{'train':train,'validation':validation,'excluded_failures':[r['source'] for r in combined if not r['demonstration_ready']],
            'prior_collection':registration['previous_collection'],
            'A1_excluded':str(probe/'A1_grasp_frame_r1'),'expected_train_objects':registration['train_objects'],
            'expected_validation_objects':registration['validation_objects']})
        state.update(status='completed',active=None);write(out/'state.json',state)
    except Exception:
        state.update(status='stopped_on_error',error=traceback.format_exc());write(out/'state.json',state);raise
    print('P2_COLLECTION_COMPLETED',out,flush=True)


if __name__=='__main__':main()
