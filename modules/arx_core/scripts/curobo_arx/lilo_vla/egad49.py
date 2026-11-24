"""Run one frozen LiLo-VLA attempt for each of the 49 historical EGAD cases."""
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
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(ROOT/'scripts/curobo_arx/static_pick')]
from client import Client
from reproduce_setup import differences,physical
from gpu_policy import verify_health
if __package__:
    from .egad_evaluate import evaluate
else:
    from egad_evaluate import evaluate

INITIAL_SHA='aa52dc17e576422d7aa9f679d1085464623cf4aed05835dab3831ca6b0801768'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def compare_configuration(historical,current):
    mismatches=differences(physical({'invariant':historical['invariant']},True),
        physical({'invariant':current['invariant']},True))
    a,b=historical['invariant']['configured'],current['invariant']['configured']
    # These contain source mounts, optics, drives, limits and all physical layout.
    protected=('controllers','calibration','motion','chassis','lift','robot','arms','objects','camera','task_layout')
    for key in protected:mismatches+=differences(a[key],b[key],'configured/'+key)
    permitted={'active_gpu','physics_gpu','camera_roles'}
    for key in a['simulation'].keys()|b['simulation'].keys():
        if key not in permitted:mismatches+=differences(a['simulation'].get(key),b['simulation'].get(key),'configured/simulation/'+key)
    return mismatches


def install_initial(campaign,source):
    if sha(source)!=INITIAL_SHA:raise ValueError('Historical initial setup SHA256 differs; do not substitute a bottle snapshot')
    destination=campaign/'runtime/historical_reference/initial_setup.json'
    if destination.exists() and sha(destination)!=INITIAL_SHA:raise ValueError('Existing campaign setup differs')
    shutil.copyfile(source,destination)
    return destination


def run_trial(campaign,scenario):
    out=campaign/'trials'/scenario['name'];out.mkdir(parents=True,exist_ok=False)
    (out/'historical_reference').mkdir()
    shutil.copyfile(campaign/'original_viewport.json',out/'historical_reference/viewport_initial.json')
    write(out/'scenario.json',scenario);(out/'environment_evidence').mkdir()
    env=os.environ.copy();mailbox=campaign/'runtime/mailbox'
    env.update(ARX_CUROBO_VALIDATION_DIR=str(mailbox),ARX_LILO_POLICY_URL='http://127.0.0.1:8008',
        ARX_LILO_MAX_STEPS='90',ARX_LILO_INSTRUCTION='pick up the object',
        ARX_LILO_VERIFY_CLOSURE='1',ARX_OBJECT_LABEL=scenario['asset'],ARX_RESEARCH_AUDIT='0')
    client=Client(mailbox);started=time.time();status='failed';error=None;setup_ok=False
    try:
        prepared=client.request('dynamic_setup',scenario=scenario,timeout=240)
        write(out/'setup.json',prepared)
        historical=read(campaign/'runtime/historical_reference/setup.json')['fingerprint']
        mismatch=compare_configuration(historical,prepared['fingerprint'])
        expected=read(campaign/'runtime/historical_reference/initial_setup.json')
        bounds=prepared['snapshot']['joint_limits']
        # The legacy reset clips only to the unchanged recorded articulation limits.
        reset=[min(max(v,lo),hi) for v,(lo,hi) in zip(expected['joint_positions'],bounds)]
        if prepared['initial_conditions']['joint_names']!=expected['joint_names']:
            mismatch.append('initial_conditions/joint_names')
        if prepared['initial_conditions']['joint_positions']!=reset:
            mismatch.append('initial_conditions/joint_positions')
        actual=prepared['initial_conditions']['objects']['asset:'+scenario['asset']]
        if actual['position']!=scenario['placement']['position'] or actual['quaternion_wxyz']!=scenario['placement'].get('orientation',[1.,0.,0.,0.]):
            mismatch.append('initial_conditions/target_pose')
        if prepared['physics_steps']!=360 or abs(prepared['captured_sim_time']-prepared['reset_sim_time']-3.)>.001:
            mismatch.append('initial_conditions/settle')
        verification={'pass':not mismatch,'historical_physical_properties_equal':not mismatch,
            'mismatches':mismatch,'exact_historical_initial_setup_sha256':INITIAL_SHA,
            'initial_conditions_and_3s_settle_checked':True,'extra_calibrated_render_sensor':'right_wrist'}
        write(out/'setup_verification.json',verification)
        if mismatch:raise RuntimeError('Historical configuration/reset mismatch: '+str(mismatch[:20]))
        setup_ok=True;write(out/'active_asset.json',prepared['asset'])
        client.request('grasp_viewport_state',restore=read(campaign/'original_viewport.json'))
        write(out/'viewport_initial.json',client.request('grasp_viewport_state'))
        # Runtime scans are not repeated for every local trajectory.
        write(out/'guard.json',{'invariant_sha256':'initial fingerprint archived in setup.json','file_sha256':{}})
        command=[env['ARX_NEURAL_PYTHON'],str(ROOT/'scripts/curobo_arx/static_pick/reproduce_record.py'),
            '--output',str(out),'--reference-run',str(ROOT/'scripts/curobo_arx/static_pick/reference'),
            '--mailbox',str(mailbox),'--curobo-python',env['CUROBO_PYTHON'],
            '--neural-gpu',env['ARX_NEURAL_GPU'],'--planner-gpu',env['ARX_PLANNER_GPU'],
            '--trajopt-policy','interactive','--no-audit']
        with (out/'pipeline.log').open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=max(1,900-(time.time()-started)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                status='timeout';raise TimeoutError('Original 900s pipeline budget exhausted; counted attempt, no retry')
        status='completed' if code==0 else 'algorithm_failure'
        stderr=out/'network/logs/stderr.log'
        if code!=0 and stderr.exists() and 'torch.OutOfMemoryError' in stderr.read_text():
            status='infrastructure_failure'
    except Exception:
        error=traceback.format_exc();write(out/'runner_error.json',{'error':error})
    score=evaluate(out)
    # A timed-out process can leave an owned native trajectory/video finishing.
    # A snapshot request is serviced only after the original bridge is idle.
    try:
        write(out/'boundary_snapshot.json',client.request('snapshot',timeout=180))
        if (out/'frames').exists() and not (out/'recording_stop.json').exists():
            write(out/'recording_stop.json',client.request('grasp_video_stop',timeout=120))
    except Exception as exc:
        write(out/'boundary_error.json',{'error':str(exc)})
        raise RuntimeError('Simulator not safely idle; stop before another case') from exc
    result={'egad_id':scenario['egad_id'],'success':bool(score['pass']),'status':status,
        'elapsed_s':time.time()-started,'setup_verified':setup_ok,'error':error,'evaluation':score}
    write(out/'result.json',result)
    if not setup_ok:raise RuntimeError('Initial-state/configuration mismatch; no additional cases authorized under a different setup')
    return result


def summarize(campaign):
    scenarios=read(campaign/'scenarios.json');results=[]
    for scenario in scenarios:
        path=campaign/'trials'/scenario['name']/'result.json'
        if path.exists():results.append(read(path))
    successes=sum(r['success'] for r in results)
    summary={'declared_attempts':49,'completed_attempts':len(results),'successes':successes,
        'failures':len(results)-successes,'pending':49-len(results),'success_rate_over_49':successes/49,
        'complete':len(results)==49,'rows':results,
        'scope':'One frozen method, one attempt per EGAD ID, no retries; unfinished cases remain pending'}
    write(campaign/'summary.json',summary);return summary


def run(campaign,initial):
    os.chdir(ROOT);campaign=campaign.resolve()
    lock=(campaign/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if initial:install_initial(campaign,initial)
    initial_path=campaign/'runtime/historical_reference/initial_setup.json'
    if not initial_path.exists() or sha(initial_path)!=INITIAL_SHA:
        raise FileNotFoundError('Need exact historical initial_setup.json before any scored attempts')
    (campaign/'trials').mkdir(exist_ok=True)
    if (campaign/'registration.json').exists():raise RuntimeError('Do not restart/retry an already registered 49-case suite')
    # Infrastructure checks happen before the first registered object attempt.
    # No test-object grasp proposals or model actions are queried here.
    deadline=time.monotonic()+1200
    while not (campaign/'runtime/mailbox/ready.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Benchmark simulator did not become ready')
        time.sleep(2)
    client=Client(campaign/'runtime/mailbox');client.request('grasp_lilo_status',timeout=60)
    probe=read(ROOT/'scripts/curobo_arx/static_pick/reference/historical_reference/registration.json')['scenario']
    prepared=client.request('dynamic_setup',scenario=probe,timeout=240)
    write(campaign/'runtime/preflight_existing_asset_reset.json',prepared)
    current=prepared['fingerprint']
    historical=read(campaign/'runtime/historical_reference/setup.json')['fingerprint']
    mismatch=compare_configuration(historical,current)
    write(campaign/'preflight_fingerprint.json',current)
    write(campaign/'preflight_configuration.json',{'pass':not mismatch,'mismatches':mismatch})
    if mismatch:raise RuntimeError('Historical physical configuration differs before registration: '+str(mismatch[:20]))
    import httpx
    import numpy as np
    from PIL import Image
    cameras={}
    with httpx.Client(base_url='http://127.0.0.1:8101',timeout=120,trust_env=False) as http:
        for camera in ('head','right_wrist'):
            directory=campaign/'runtime'/f'preflight_{camera}_{time.time_ns()}'
            serial=next(k for k,v in current['invariant']['configured']['controllers']['camera']['logical_serials'].items() if v==camera)
            response=http.post('/camera/capture-rgbd',json={'serial':serial,'output_dir':str(directory)})
            response.raise_for_status()
            pixels=np.asarray(Image.open(directory/'color.png').convert('RGB'))
            cameras[camera]={'directory':str(directory),'nonzero':bool(np.any(pixels)),
                'std':float(pixels.std()),'width':pixels.shape[1],'height':pixels.shape[0]}
            if not np.any(pixels) or pixels.std()<1:
                write(campaign/'preflight_cameras.json',cameras)
                raise RuntimeError('Historical renderer/camera produces invalid RGB; no scored attempts started')
    write(campaign/'preflight_cameras.json',cameras)
    model=ROOT/'runs/lilo_pilot_20261026_1835/adapter'
    with httpx.Client(timeout=30,trust_env=False) as http:
        health=http.get('http://127.0.0.1:8008/health');health.raise_for_status();policy_health=health.json()
    if Path(policy_health.get('adapter','')).resolve()!=model.resolve():
        raise RuntimeError('Policy service is not the declared frozen ARX adapter')
    verify_health(policy_health)
    write(campaign/'policy_health.json',policy_health)
    sources=[*Path(__file__).parent.glob('*.py'),ROOT/'scripts/curobo_arx/grasp_execute.py',
        ROOT/'scripts/curobo_arx/settling.py',ROOT/'scripts/curobo_arx/optimize_path.py',
        ROOT/'scripts/curobo_arx/static_pick/reproduce_record.py',ROOT/'scripts/curobo_arx/prepare.py',
        ROOT/'arx-sim/config/runtime.yaml',ROOT/'arx-sim/config/controllers.yaml',
        ROOT/'configs/calibration.yaml',ROOT/'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd']
    frozen={str(p):sha(p) for p in sources}
    registration={'method':'ZeroGrasp + cuRobo reaching + frozen bottle-adapted LiLo-VLA local policy',
        'attempts':49,'no_retry':True,'test_set_training':False,'instruction':'pick up the object',
        'max_local_actions':90,'trial_timeout_s':900,'closing_dwell_s':3.,
        'simulation_physical_gpu':3,'simulator_visible_gpu':0,'policy_a100_gpu':1,
        'reaching_optimizer':'Existing interactive cuRobo, bound weight multiplier100',
        'hold_control_target_m':.07,'success_minimum_rise_m':.06,'hold_s':3.,
        'scenario_sha256':sha(campaign/'scenarios.json'),'initial_setup_sha256':sha(initial_path),
        'head_sha256':sha(model/'action_head.pt'),'statistics_sha256':sha(model/'dataset_statistics.json'),
        'source_sha256':frozen,'registered_unix':time.time()}
    write(campaign/'registration.json',registration)
    archive=campaign/'registered_source_snapshot';archive.mkdir()
    for name,digest in frozen.items():
        source=Path(name);destination=archive/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        if sha(destination)!=digest:raise RuntimeError('Source changed while archiving registration: '+name)
    write(archive/'manifest.json',{'copied_unix':time.time(),'all_bytes_match_preregistered_hashes':True,
        'registered_source_sha256':frozen})
    storage=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if storage.exists():shutil.copyfile(storage,campaign/'storage_policy.json')
    else:write(campaign/'storage_policy_unverified.json',{'source':str(storage),
        'status':'historical file missing; reserve the explicitly required 1073741824 bytes'})
    policy=read(model/'adapter_manifest.json');write(campaign/'adapter_manifest.json',policy)
    state={'status':'running','pid':os.getpid(),'completed':0,'successes':0};write(campaign/'state.json',state)
    try:
        for scenario in read(campaign/'scenarios.json'):
            if shutil.disk_usage(campaign).free<=1073741824:raise RuntimeError('Preserve 1 GiB free')
            if any(sha(p)!=h for p,h in frozen.items()):raise RuntimeError('Method changed during frozen benchmark')
            available=int(subprocess.check_output(['nvidia-smi','--id='+os.environ['ARX_NEURAL_GPU'],
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if available<4096:
                raise RuntimeError(f'ZeroGrasp GPU has only {available} MiB free; stop before a new scored attempt')
            state['active']=scenario['name'];write(campaign/'state.json',state)
            print('EGAD49_START',scenario['egad_id'],flush=True)
            result=run_trial(campaign,scenario);summary=summarize(campaign)
            state.update(completed=summary['completed_attempts'],successes=summary['successes']);write(campaign/'state.json',state)
            print('EGAD49_RESULT',scenario['egad_id'],result['success'],summary['completed_attempts'],'/49',flush=True)
            if result['status']=='infrastructure_failure':
                raise RuntimeError('Shared GPU out of memory; keep failed attempt and stop before another case')
        state.update(status='completed',finished_unix=time.time());write(campaign/'state.json',state)
    except Exception:
        state.update(status='stopped_error',error=traceback.format_exc());write(campaign/'state.json',state);summarize(campaign);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--initial-setup',type=Path)
    args=parser.parse_args();run(args.campaign,args.initial_setup)
