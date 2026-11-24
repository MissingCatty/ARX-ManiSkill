"""P0 action chunks or P1 grasp-relative handoff trials in actual ARX scenes.

Uses a new directory and never resumes the stopped EGAD49 campaign. Fixed
historical proposals remove ZeroGrasp sampling as an experimental variable.
"""
from arx_project.layout import canonical_path
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(ROOT/'scripts/curobo_arx/static_pick'),str(Path(__file__).parent)]
from client import Client
from egad49 import compare_configuration, INITIAL_SHA
from egad_evaluate import evaluate
from gpu_policy import resolve,compute_env,verify_health


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run_process(command,env,log,deadline):
    with log.open('x') as stream:
        child=subprocess.Popen([str(v) for v in command],cwd=ROOT,env=env,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:return child.wait(timeout=max(.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise TimeoutError('Registered 900s trial wall-time budget exhausted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--experiment',choices=('chunks','handoff','teacher'),default='chunks')
    parser.add_argument('--mailbox',type=Path,default=ROOT/'runs/lilo_egad49_20261026_eval01/runtime/mailbox')
    parser.add_argument('--bottle-mailbox',type=Path,required=True,
        help='Separate ARX fixture matching the historical PathTracing LiLo bottle success')
    parser.add_argument('--bottle-ops-url',default='http://127.0.0.1:8102')
    parser.add_argument('--ops-url',default='http://127.0.0.1:8101')
    parser.add_argument('--policy-url',default='http://127.0.0.1:8008')
    args=parser.parse_args()
    args.planner_gpu=resolve(args.planner_gpu)['uuid']
    os.environ.update(compute_env(os.environ,args.planner_gpu))
    phase={'chunks':'P0','handoff':'P1','teacher':'P2_TEACHER'}[args.experiment]
    # Explicit fixture CLI arguments take precedence over a shell's old mailbox.
    os.environ.pop('ARX_CUROBO_VALIDATION_DIR',None)
    os.environ.pop('ARX_RESEARCH_GUARD_FILE',None)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    locks=[]
    for mailbox in sorted({args.mailbox.resolve(),args.bottle_mailbox.resolve()}):
        mailbox.mkdir(parents=True,exist_ok=True)
        lock=(mailbox/'p0_execution.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        if (mailbox/'ready.json').exists():
            boundary=Client(mailbox).request('snapshot',timeout=60)
            if boundary.get('command_id') is not None:raise RuntimeError('ARX simulator is not idle')
    import httpx
    health=httpx.get(args.policy_url.rstrip('/')+'/health',timeout=30,trust_env=False)
    health.raise_for_status();health=health.json()
    assert health['arx_finetuned'] and health['adaptation_selected_step']==300
    verify_health(health)
    bundle=ROOT/'scripts/curobo_arx/static_pick/reference'
    historical=ROOT/'runs/lilo_egad49_20261026_eval01'
    initial=ROOT/'runs/tabletop_trials/initial_setup.json'
    assert sha(initial)==INITIAL_SHA
    bottle=read(ROOT/'runs/lilo_execution_20261026_tracking02/selected_grasp.json')
    a1=read(historical/'trials/01_A1_lilo/selected_grasp.json')
    bottle_setup=read(ROOT/'runs/lilo_execution_20261026_tracking02/setup.json')
    bottle_scenario=read(bundle/'historical_reference/registration.json')['scenario']
    a1_scenario=read(historical/'trials/01_A1_lilo/scenario.json')
    teacher=read(ROOT/'runs/lilo_A1_handoff_20261027_01/lilo_arx_teacher_20261026_05_reference.json')
    inputs=out/'registered_inputs';inputs.mkdir()
    write(inputs/'bottle_selection.json',bottle);write(inputs/'A1_selection.json',a1)
    write(inputs/'A1_teacher05.json',teacher)
    rule=None;rule_file=inputs/'handoff_rule.json'
    if args.experiment in ('handoff','teacher'):
        from grasp_handoff import RULE
        rule=copy.deepcopy(RULE);write(rule_file,rule)
    shutil.copy2(initial,inputs/'initial_setup.json')
    scenes={
        'bottle':{'scenario':bottle_scenario,'fingerprint':bottle_setup['fingerprint'],
            'initial_conditions':bottle_setup['initial_conditions'],
            'mailbox':args.bottle_mailbox.resolve(),'ops_url':args.bottle_ops_url,
            'viewport':read(bundle/'historical_reference/viewport_initial.json'),
            'selection':inputs/'bottle_selection.json','handoff':None,
            'instruction':'pick up the bbq sauce bottle'},
        'A1':{'scenario':a1_scenario,'fingerprint':read(historical/'runtime/historical_reference/setup.json')['fingerprint'],
            'mailbox':args.mailbox.resolve(),'ops_url':args.ops_url,
            'initial_conditions':None,'viewport':read(historical/'original_viewport.json'),
            'selection':inputs/'A1_selection.json','handoff':inputs/'A1_teacher05.json',
            'instruction':'pick up the object'}}
    # A1 is already running; overlap its trials with the isolated bottle startup.
    scenes={scene:scenes[scene] for scene in ('A1','bottle')}
    protected=[ROOT/p for p in ('arx-sim/config/sim.yaml','arx-sim/config/runtime.yaml',
        'arx-sim/config/controllers.yaml','configs/calibration.yaml',
        'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd')]
    sources=[Path(__file__),Path(__file__).with_name('rollout.py'),
        Path(__file__).with_name('observations.py'),Path(__file__).with_name('egad_evaluate.py'),
        ROOT/'scripts/curobo_arx/grasp_execute.py',ROOT/'scripts/curobo_arx/prepare.py',
        Path(__file__).with_name('egad_start_sim.py'),
        ROOT/'scripts/curobo_arx/gripper_update.py',ROOT/'scripts/curobo_arx/settling.py',
        ROOT/'scripts/curobo_arx/model_archive.py',ROOT/'scripts/curobo_arx/optimize_path.py']
    if rule:sources.append(Path(__file__).with_name('grasp_handoff.py'))
    snapshot=out/'registered_source_snapshot';snapshot.mkdir()
    for source in sources:shutil.copy2(source,snapshot/source.name)
    checks=ROOT.parent/('tmp/lilo_P1_geometry_checks_20261027.json' if rule else 'tmp/lilo_P0_controller_checks_20261027.json')
    if checks.exists():
        shutil.copy2(checks,out/('geometry_checks.json' if rule else 'controller_checks.json'))
        check_source=checks.with_suffix('.py')
        if check_source.exists():shutil.copy2(check_source,snapshot/check_source.name)
    adapter=Path(health['adapter'])
    adapter_files={str(p):sha(p) for p in adapter.rglob('*') if p.is_file() and p.suffix in ('.pt','.json','.safetensors')}
    registration={'scope':phase+' development experiment; two ARX scenes, stopped EGAD49 results untouched',
        'experiment':args.experiment,
        'changed_factor':'Physically validate the fixed selected grasp with cuRobo teacher; no VLA actions' if args.experiment=='teacher' else ('Grasp-relative entry selection with frozen continuation/frustum admission' if rule else 'Number of sequential actions consumed per policy prediction'),
        'screening_order':[[scene,'grasp_frame'] for scene in scenes] if args.experiment=='teacher' else ([[scene,variant] for scene in scenes for variant in ('baseline','grasp_frame')] if rule else [[scene,k] for scene in scenes for k in (1,4,8)]),
        'confirmation_rule':'No automatic repeats; this is teacher target qualification, not policy evaluation' if args.experiment=='teacher' else ('If grasp_frame passes BOTH scenes, add two repeats of both variants in each scene' if rule else 'For each K in 4,8 passing BOTH screening scenes, add two repeats of K and K=1 per scene; share baseline repeats'),
        'handoff_rule':rule,
        'max_local_actions':90,'max_local_motion_sim_seconds':60.,'trial_wall_seconds':900,
        'closure_seconds':3.,'hold_seconds':3.,'require_verified_closure':True,
        'bottle_control_note':'Historical positive bottle lacked explicit 3s closure gate; current K=1 must be measured again',
        'planner_gpu':args.planner_gpu,'mailboxes':{s:str(d['mailbox']) for s,d in scenes.items()},'policy':health,
        'historical_renderers':{s:d['fingerprint']['invariant']['configured']['simulation']['renderer'] for s,d in scenes.items()},
        'policy_artifact_sha256':adapter_files,'initial_setup_sha256':sha(initial),
        'physical_source_sha256':{str(p):sha(p) for p in protected},
        'algorithm_sha256':{str(p):sha(p) for p in sources},
        'registered_input_sha256':{str(p):sha(p) for p in inputs.iterdir()},
        'scene_instructions':{s:d['instruction'] for s,d in scenes.items()},
        'seed':0,'registered_unix':time.time()}
    write(out/'registration.json',registration)
    policy=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if policy.exists():shutil.copy2(policy,out/'storage_policy.json')
    else:write(out/'storage_policy.json',{'local_first':True,'reserve_bytes':1073741824,
        'historical_policy_file':'Unavailable on this host; not assumed verified'})
    write(out/'state.json',{'status':'running','completed':0,'successes':0})
    results=[]
    neural=canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python')
    planner=ROOT.parent/'curobo/.venv/bin/python'

    def trial(scene,k,repeat,method=False):
        if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
        for path,digest in {**registration['algorithm_sha256'],**registration['registered_input_sha256']}.items():
            if sha(path)!=digest:raise RuntimeError('Registered algorithm changed during comparison: '+path)
        spec=scenes[scene];client=Client(spec['mailbox'])
        variant='grasp_frame' if method else 'baseline'
        label=f'{scene}_{variant}_r{repeat}' if rule else f'{scene}_k{k}_r{repeat}'
        if not (client.root/'ready.json').exists():
            print(phase+'_WAIT_FOR_FIXTURE',scene,flush=True)
            ready_deadline=time.monotonic()+900
            while not (client.root/'ready.json').exists():
                if time.monotonic()>ready_deadline:raise TimeoutError('Historical fixture startup timeout: '+scene)
                time.sleep(2)
        folder=out/label;folder.mkdir();(folder/'environment_evidence').mkdir()
        started=time.monotonic();deadline=started+900
        scenario=copy.deepcopy(spec['scenario']);scenario['name']=label
        write(folder/'scenario.json',scenario)
        write(out/'state.json',{'status':'running','active':label,'completed':len(results),
            'successes':sum(r['pass'] for r in results)})
        setup=client.request('dynamic_setup',scenario=scenario,timeout=min(240,max(1,deadline-time.monotonic())))
        write(folder/'setup.json',setup)
        mismatches=compare_configuration(spec['fingerprint'],setup['fingerprint'])
        expected=read(initial);limits=setup['snapshot']['joint_limits']
        clipped=[min(max(q,lo),hi) for q,(lo,hi) in zip(expected['joint_positions'],limits)]
        if setup['initial_conditions']['joint_names']!=expected['joint_names']:mismatches.append('historical joint names')
        if setup['initial_conditions']['joint_positions']!=clipped:mismatches.append('historical joint reset')
        if spec['initial_conditions'] and setup['initial_conditions']!=spec['initial_conditions']:
            mismatches.append('historical bottle declared initial conditions')
        actual=setup['initial_conditions']['objects']['asset:'+scenario['asset']]
        if actual['position']!=scenario['placement']['position'] or actual['quaternion_wxyz']!=scenario['placement'].get('orientation',[1.,0.,0.,0.]):
            mismatches.append('historical target placement')
        if setup['physics_steps']!=360 or abs(setup['captured_sim_time']-setup['reset_sim_time']-3.)>.001:
            mismatches.append('historical 3s settling')
        write(folder/'setup_verification.json',{'pass':not mismatches,'mismatches':mismatches,
            'historical_physical_properties_equal':not mismatches,'initial_setup_sha256':INITIAL_SHA})
        if mismatches:raise RuntimeError('Frozen configuration/reset mismatch: '+str(mismatches[:15]))
        write(folder/'active_asset.json',setup['asset'])
        client.request('grasp_viewport_state',restore=spec['viewport'])
        write(folder/'viewport_initial.json',client.request('grasp_viewport_state'))
        write(folder/'guard.json',{'invariant_sha256':'initial fingerprint archived in setup.json','file_sha256':{}})
        env=os.environ.copy();env.update(ARX_CUROBO_VALIDATION_DIR=str(client.root),GRASP_OUTPUT_DIR=str(folder),
            GRASP_TARGET='test_object',ARX_OPS_URL=spec['ops_url'],
            GRASP_FIXED_SELECTION=str(spec['selection']),ARX_LILO_POLICY_URL=args.policy_url,
            ARX_LILO_MAX_STEPS='90',ARX_LILO_MAX_SIM_SECONDS='60',ARX_LILO_EXECUTE_STEPS=str(k),
            ARX_LILO_INSTRUCTION=spec['instruction'],ARX_LILO_VERIFY_CLOSURE='1',
            GRASP_PREPARED_CLOSED_MODEL='1',GRASP_PLANNER='prm_trajopt',
            GRASP_BOUND_WEIGHT_MULTIPLIER='100',GRASP_TRAJOPT_POLICY='interactive',
            ARX_RESEARCH_GUARD_FILE=str(folder/'guard.json'),ARX_RESEARCH_AUDIT='0',
            GRASP_VIDEO_DIR=str(folder/'frames'),CUDA_VISIBLE_DEVICES=args.planner_gpu,PYTHONUNBUFFERED='1')
        for name in ('ARX_LILO_TEACHER_DIR','ARX_LILO_HANDOFF_OBSERVATION','ARX_LILO_HANDOFF_RULE_FILE','GRASP_CAMERA_VIDEO_DIR'):env.pop(name,None)
        if method:env['ARX_LILO_HANDOFF_RULE_FILE']=str(rule_file)
        elif spec['handoff']:env['ARX_LILO_HANDOFF_OBSERVATION']=str(spec['handoff'])
        if args.experiment=='teacher':
            env.pop('ARX_LILO_POLICY_URL',None)
            env['ARX_LILO_TEACHER_DIR']=str(folder/'teacher_raw')
        status='algorithm_failure';error=None
        print(phase+'_START',label,flush=True)
        try:
            for name,python,script,argv in [
                ('capture',neural,'grasp_input.py',[]),
                ('prepare',planner,'prepare.py',['--output',folder,'--model-source',bundle]),
                ('execution',planner,'grasp_execute.py',[])]:
                code=run_process([python,ROOT/'scripts/curobo_arx'/script,*argv],env,folder/(name+'.log'),deadline)
                if code:raise RuntimeError(f'{name} exited {code}; retained in {name}.log')
                if name=='prepare':
                    import yaml
                    path=folder/'model_closed/arx.yml';config=yaml.safe_load(path.read_text())
                    for joint in ('right_joint27','right_joint28'):config['robot_cfg']['kinematics']['lock_joints'][joint]=.022
                    path.write_text(yaml.safe_dump(config,sort_keys=False))
            status='completed'
        except Exception as exc:
            error=traceback.format_exc();write(folder/'runner_error.json',{'error':error})
            if isinstance(exc,TimeoutError):status='timeout'
            if any('OutOfMemoryError' in p.read_text(errors='replace') for p in folder.glob('*.log')):
                status='infrastructure_failure'
        finally:
            write(folder/'boundary_snapshot.json',client.request('snapshot',timeout=300))
            if args.experiment=='teacher' and (folder/'teacher_raw').exists():
                write(folder/'teacher_recording.json',client.request('grasp_lilo_record_stop',timeout=120))
            if (folder/'frames').exists():write(folder/'recording_stop.json',client.request('grasp_video_stop',timeout=120))
        score=evaluate(folder)
        report=read(folder/'grasp_report.json') if (folder/'grasp_report.json').exists() else {}
        if args.experiment=='teacher':
            write(folder/'verification.json',{'pass':score['pass'],'physical_grasp_success':score['pass'],
                'historical_physics_matched_before_execution':True,'configuration_audit_enabled':False,
                'teacher_source':'Actual frozen cuRobo teacher, fixed ZeroGrasp goal + shared P1 entry',
                'held_min_rise_m':score.get('hold_min_rise_m'),'error':score.get('reported_error'),
                'evaluation':score})
        result={'label':label,'scene':scene,'execute_steps':k,'repeat':repeat,'pass':score['pass'],
            'variant':variant if rule else None,
            'status':status,'elapsed_seconds':time.monotonic()-started,'runner_error':error,
            'policy_calls':report.get('local_policy_calls',0),'executed_actions':report.get('local_action_count',0),
            'attempted_actions':report.get('local_action_attempts',0),'local_sim_seconds':report.get('local_sim_seconds',0),
            'evaluation':score}
        write(folder/'result.json',result);results.append(result);write(out/'results.json',results)
        print(phase+'_RESULT',json.dumps({k:v for k,v in result.items() if k not in ('runner_error','evaluation')}),flush=True)
        for path,digest in {**registration['physical_source_sha256'],**registration['policy_artifact_sha256']}.items():
            if sha(path)!=digest:raise RuntimeError('Protected physical source or policy changed: '+path)

    try:
        if args.experiment=='teacher':
            for scene in scenes:trial(scene,1,1,True)
            candidates=[]
        elif rule:
            for scene in scenes:
                for method in (False,True):trial(scene,1,1,method)
            candidates=[rule['version']] if all(any(r['scene']==scene and r['variant']=='grasp_frame' and r['pass'] for r in results) for scene in scenes) else []
        else:
            for scene in scenes:
                for k in (1,4,8):trial(scene,k,1)
            candidates=[k for k in (4,8) if all(any(r['scene']==scene and r['execute_steps']==k and r['pass'] for r in results) for scene in scenes)]
        write(out/'confirmation_selection.json',{'qualifying_variants' if rule else 'qualifying_execute_steps':candidates,
            'rule':registration['confirmation_rule'],'screening_attempts':len(results)})
        if candidates:
            for scene in scenes:
                for repeat in (2,3):
                    if rule:
                        for method in (False,True):trial(scene,1,repeat,method)
                    else:
                        for k in (1,*candidates):trial(scene,k,repeat)
        write(out/'state.json',{'status':'completed','completed':len(results),'successes':sum(r['pass'] for r in results),
            'qualifying_variants' if rule else 'qualifying_execute_steps':candidates,'default_execute_steps':1,
            'default_handoff_rule':None,
            'promotion':'Not automatic; summarize repeated results and bottle regressions first'})
    except Exception:
        write(out/'state.json',{'status':'stopped_on_error','completed':len(results),
            'successes':sum(r['pass'] for r in results),'error':traceback.format_exc()})
        raise
    print(phase+'_COMPLETED',out,flush=True)


if __name__=='__main__':main()
