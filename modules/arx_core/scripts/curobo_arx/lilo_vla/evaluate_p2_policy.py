"""Two actual ARX checks of one explicitly pinned P2 adapter and state codec."""
from arx_project.layout import canonical_path
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(Path(__file__).parent)]
from client import Client
from compare_chunks import run_process
from egad49 import compare_configuration,INITIAL_SHA
from egad_evaluate import evaluate
from grasp_handoff import RULE,validate_rule
from summarize_chunks import encode
from handoff_profile import verify_policy,verify_case
from gpu_policy import resolve,compute_env,verify_health


def selected_mailboxes(scene,mailbox,bottle_mailbox):
    """A single-scene replay must not contact the other paused fixture."""
    return sorted({Path(p).resolve() for p in
        ([mailbox] if scene=='A1' else [bottle_mailbox] if scene=='bottle' else [mailbox,bottle_mailbox])})


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy-url',required=True)
    parser.add_argument('--expected-adapter',type=Path,required=True)
    parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--bottle-mailbox',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,default=ROOT/'runs/lilo_egad49_20261026_eval01/runtime/mailbox')
    parser.add_argument('--handoff-rule',type=Path,help='Explicit registered entry rule; the original 4/6/8cm rule remains default')
    parser.add_argument('--handoff-profile',type=Path,help='Require policy/case/goal identity and actual-entry admission before the first NN request')
    parser.add_argument('--scene',choices=('A1','bottle'),help='Optional single development case; default checks both')
    args=parser.parse_args();args.planner_gpu=resolve(args.planner_gpu)['uuid']
    os.environ.update(compute_env(os.environ,args.planner_gpu))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile=read(args.handoff_profile) if args.handoff_profile else None
    rule=read(args.handoff_rule) if args.handoff_rule else (profile['rule'] if profile else RULE);validate_rule(rule)
    if profile and rule!=profile['rule']:raise ValueError('Requested rule differs from the qualified profile')
    os.environ.pop('ARX_CUROBO_VALIDATION_DIR',None);os.environ.pop('ARX_RESEARCH_GUARD_FILE',None)
    locks=[]
    for mailbox in selected_mailboxes(args.scene,args.mailbox,args.bottle_mailbox):
        lock=(mailbox/'p0_execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert Client(mailbox).request('snapshot',timeout=60).get('command_id') is None
    import httpx
    health=httpx.get(args.policy_url.rstrip('/')+'/health',timeout=30,trust_env=False)
    health.raise_for_status();health=health.json()
    adapter=args.expected_adapter.resolve()
    assert Path(health['adapter']).resolve()==adapter and health['checkpoint_verified']
    verify_health(health)
    convention=health.get('state_convention','object_relative_v1')
    assert convention in ('object_relative_v1','grasp_goal_world_error_v1')
    if profile:verify_policy(profile,health)
    probe=ROOT/'runs/lilo_P2_teacher_probe_20261028_01'
    scenes=[('A1',args.mailbox.resolve(),'http://127.0.0.1:8101','pick up the object'),
            ('bottle',args.bottle_mailbox.resolve(),'http://127.0.0.1:8102','pick up the bbq sauce bottle')]
    if args.scene:scenes=[item for item in scenes if item[0]==args.scene]
    initial=ROOT/'runs/tabletop_trials/initial_setup.json';assert sha(initial)==INITIAL_SHA
    rule_file=out/'handoff_rule.json';write(rule_file,rule)
    profile_file=out/'qualified_handoff_profile.json'
    if profile:write(profile_file,profile)
    physical=[ROOT/p for p in ('arx-sim/config/sim.yaml','arx-sim/config/runtime.yaml','arx-sim/config/controllers.yaml',
        'configs/calibration.yaml','asset/robots/arx_x5_lift/1.0.0/usd/robot.usd')]
    algorithms=[Path(__file__),ROOT/'scripts/curobo_arx/grasp_execute.py',
        ROOT/'scripts/curobo_arx/gpu_policy.py',
        *[Path(__file__).with_name(n) for n in ('rollout.py','observations.py','grasp_handoff.py','egad_evaluate.py')]]
    if profile:algorithms.extend([Path(__file__).with_name('handoff_profile.py'),
        Path(__file__).with_name('egad49.py'),ROOT/'scripts/curobo_arx/static_pick/reproduce_setup.py'])
    archive=out/'registered_source_snapshot';archive.mkdir()
    for path in algorithms:shutil.copy2(path,archive/path.name)
    registration={'scope':'Two ARX development cases, same teacher-qualified grasp goals and shared entry; no EGAD49 batch restart',
        'policy':health,'state_convention':convention,'policy_artifact_sha256':{str(adapter/n):sha(adapter/n) for n in ('action_head.pt','dataset_statistics.json','adapter_manifest.json')},
        'physical_source_sha256':{str(p):sha(p) for p in physical},'algorithm_sha256':{str(p):sha(p) for p in algorithms},
        'teacher_qualified_goal_sources':{scene:str(probe/(scene+'_grasp_frame_r1')/'selected_grasp.json') for scene,_,_,_ in scenes},
        'max_actions':90,'max_local_sim_seconds':60,'trial_wall_seconds':900,'execute_steps':1,
        'admission_profile_sha256':sha(profile_file) if profile else None,
        'initial_sha256':INITIAL_SHA,'rule':rule,'rule_file_sha256':sha(rule_file),'planner_gpu':args.planner_gpu,'registered_unix':time.time()}
    write(out/'registration.json',registration)
    write(out/'storage_policy.json',{'local_first':True,'reserve_bytes':1073741824,'historical_file':'Unavailable on this host; not assumed verified'})
    results=[];write(out/'state.json',{'status':'running','completed':0,'successes':0})
    neural=canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python');planner=ROOT.parent/'curobo/.venv/bin/python'
    for scene,mailbox,url,instruction in scenes:
        for name,digest in {**registration['physical_source_sha256'],**registration['algorithm_sha256'],**registration['policy_artifact_sha256']}.items():
            if sha(name)!=digest:raise RuntimeError('Registered evaluation source changed: '+name)
        if sha(rule_file)!=registration['rule_file_sha256']:raise RuntimeError('Registered entry rule changed')
        if profile and sha(profile_file)!=registration['admission_profile_sha256']:raise RuntimeError('Qualified profile changed')
        if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
        source=probe/(scene+'_grasp_frame_r1');assert read(source/'verification.json')['pass']
        trial=out/scene;trial.mkdir();(trial/'environment_evidence').mkdir()
        client=Client(mailbox);started=time.monotonic();deadline=started+900
        scenario=read(source/'scenario.json');scenario['name']=scene
        if profile:
            _,qualified=verify_case(profile,scenario,read(source/'selected_grasp.json'),check_rule=False)
            if qualified['instruction'].strip().lower()!=instruction.strip().lower():raise ValueError('Qualified instruction differs')
        write(trial/'scenario.json',scenario)
        setup=client.request('dynamic_setup',scenario=scenario,timeout=240);write(trial/'setup.json',setup)
        mismatch=compare_configuration(read(source/'setup.json')['fingerprint'],setup['fingerprint'])
        expected=read(initial);bounds=setup['snapshot']['joint_limits']
        clipped=[min(max(q,lo),hi) for q,(lo,hi) in zip(expected['joint_positions'],bounds)]
        if setup['initial_conditions']['joint_positions']!=clipped or setup['initial_conditions']['joint_names']!=expected['joint_names']:mismatch.append('historical robot reset')
        actual=setup['initial_conditions']['objects']['asset:'+scenario['asset']]
        if actual['position']!=scenario['placement']['position'] or actual['quaternion_wxyz']!=scenario['placement'].get('orientation',[1.,0.,0.,0.]):mismatch.append('historical object placement')
        if setup['physics_steps']!=360:mismatch.append('historical settling')
        write(trial/'setup_verification.json',{'pass':not mismatch,'mismatches':mismatch})
        if mismatch:raise RuntimeError('Frozen evaluation conditions differ: '+str(mismatch[:12]))
        write(trial/'active_asset.json',setup['asset'])
        view=read(source/'viewport_initial.json');client.request('grasp_viewport_state',restore=view)
        write(trial/'viewport_initial.json',client.request('grasp_viewport_state'))
        write(trial/'guard.json',{'invariant_sha256':'Initial fingerprint archived','file_sha256':{}})
        selection=trial/'fixed_selection.json';shutil.copy2(source/'selected_grasp.json',selection)
        env=os.environ.copy()
        for key in ('ARX_LILO_TEACHER_DIR','ARX_LILO_HANDOFF_OBSERVATION','ARX_LILO_ENTRY_PERTURBATION_FILE','GRASP_CAMERA_VIDEO_DIR','ARX_LILO_HANDOFF_PROFILE'):env.pop(key,None)
        env.update(ARX_CUROBO_VALIDATION_DIR=str(client.root),GRASP_OUTPUT_DIR=str(trial),GRASP_TARGET='test_object',
            ARX_OPS_URL=url,GRASP_FIXED_SELECTION=str(selection),ARX_LILO_HANDOFF_RULE_FILE=str(rule_file),
            ARX_LILO_POLICY_URL=args.policy_url,ARX_LILO_MAX_STEPS='90',ARX_LILO_MAX_SIM_SECONDS='60',
            ARX_LILO_EXECUTE_STEPS='1',ARX_LILO_VERIFY_CLOSURE='1',ARX_LILO_INSTRUCTION=instruction,
            GRASP_PREPARED_CLOSED_MODEL='1',GRASP_PLANNER='prm_trajopt',GRASP_BOUND_WEIGHT_MULTIPLIER='100',
            GRASP_TRAJOPT_POLICY='interactive',ARX_RESEARCH_GUARD_FILE=str(trial/'guard.json'),
            ARX_RESEARCH_AUDIT='0',GRASP_VIDEO_DIR=str(trial/'frames'),CUDA_VISIBLE_DEVICES=args.planner_gpu,PYTHONUNBUFFERED='1')
        if profile:env['ARX_LILO_HANDOFF_PROFILE']=str(profile_file)
        write(out/'state.json',{'status':'running','active':scene,'completed':len(results),'successes':sum(r['pass'] for r in results)})
        print('P2_POLICY_START',scene,convention,flush=True);error=None
        try:
            for name,python,script,argv in [('capture',neural,'grasp_input.py',[]),
                ('prepare',planner,'prepare.py',['--output',trial,'--model-source',ROOT/'scripts/curobo_arx/static_pick/reference']),
                ('execution',planner,'grasp_execute.py',[])]:
                code=run_process([python,ROOT/'scripts/curobo_arx'/script,*argv],env,trial/(name+'.log'),deadline)
                if code:raise RuntimeError(name+' exited '+str(code))
                if name=='prepare':
                    import yaml
                    path=trial/'model_closed/arx.yml';config=yaml.safe_load(path.read_text())
                    for joint in ('right_joint27','right_joint28'):config['robot_cfg']['kinematics']['lock_joints'][joint]=.022
                    path.write_text(yaml.safe_dump(config,sort_keys=False))
        except Exception:error=traceback.format_exc();write(trial/'runner_error.json',{'error':error})
        finally:
            write(trial/'boundary_snapshot.json',client.request('snapshot',timeout=300))
            if (trial/'frames').exists():write(trial/'recording_stop.json',client.request('grasp_video_stop',timeout=120))
        score=evaluate(trial);report=read(trial/'grasp_report.json') if (trial/'grasp_report.json').exists() else {}
        result={'scene':scene,'pass':score['pass'],'elapsed_seconds':time.monotonic()-started,'evaluation':score,
            'state_convention':convention,'policy_calls':report.get('local_policy_calls',0),
            'executed_actions':report.get('local_action_count',0),'runner_error':error}
        if (trial/'frames/frames.json').exists():result['video']=encode(trial)
        write(trial/'result.json',result);results.append(result);write(out/'results.json',results)
        print('P2_POLICY_RESULT',json.dumps({'scene':scene,'pass':score['pass'],'calls':result['policy_calls'],'error':score.get('reported_error')}),flush=True)
    write(out/'state.json',{'status':'completed','completed':len(results),'successes':sum(r['pass'] for r in results)})
    write(out/'final_integrity.json',{'physical_and_policy_sources_unchanged':all(sha(p)==v for p,v in {**registration['physical_source_sha256'],**registration['policy_artifact_sha256']}.items()),
        'algorithm_unchanged':all(sha(p)==v for p,v in registration['algorithm_sha256'].items())})


if __name__=='__main__':main()
