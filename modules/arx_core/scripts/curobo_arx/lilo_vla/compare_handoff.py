"""Isolated A1 start-pose comparison; never resumes the stopped EGAD49 suite."""
from arx_project.layout import canonical_path
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(Path(__file__).parent)]
from client import Client
from egad49 import compare_configuration, INITIAL_SHA
from egad_evaluate import evaluate
from gpu_policy import resolve,compute_env,verify_health


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--planner-gpu',required=True)
    args=parser.parse_args()
    args.planner_gpu=resolve(args.planner_gpu)['uuid']
    os.environ.update(compute_env(os.environ,args.planner_gpu))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    historical=ROOT/'runs/lilo_egad49_20261026_eval01'
    original=historical/'trials/01_A1_lilo'
    client=Client(historical/'runtime/mailbox')
    dataset=read(ROOT/'runs/lilo_dataset_20261026_train03_05_val06_v2/dataset.json')
    references=[]
    for ep in dataset['episodes']:
        if ep['split']!='train':continue
        rows=[json.loads(line) for line in (Path(ep['source'])/'teacher_raw/samples.jsonl').read_text().splitlines()]
        row=next(row for row in rows if row.get('stage')=='approach')
        dest=out/(Path(ep['source']).name+'_reference.json');write(dest,row)
        references.append(dest)
    variants=[('original_start',None),*[(f'teacher_start_{i+1}',p) for i,p in enumerate(references)]]
    assert len(variants)==3
    import httpx
    health=httpx.get('http://127.0.0.1:8008/health',timeout=30,trust_env=False).json()
    assert health['arx_finetuned'] and health['adaptation_selected_step']==300
    verify_health(health)
    initial=ROOT/'runs/tabletop_trials/initial_setup.json';assert sha(initial)==INITIAL_SHA
    frozen_files=[ROOT/p for p in ('arx-sim/config/runtime.yaml','arx-sim/config/controllers.yaml',
        'configs/calibration.yaml','asset/robots/arx_x5_lift/1.0.0/usd/robot.usd')]
    write(out/'registration.json',{'scope':'A1 development diagnosis, separate from stopped 49-case results',
        'changed_factor':'pregrasp motion goal only','variants':[n for n,p in variants],
        'reference_observations':[str(p) for p in references],
        'policy':health,'instruction':'pick up the object','max_local_actions':90,
        'planner_gpu':args.planner_gpu,'physical_configuration_sha256':{str(p):sha(p) for p in frozen_files},
        'initial_setup_sha256':sha(initial),'registered_unix':time.time(),
        'sources':{str(p):sha(p) for p in (Path(__file__),ROOT/'scripts/curobo_arx/grasp_execute.py',Path(__file__).with_name('rollout.py'))}})
    storage=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if storage.exists():shutil.copy2(storage,out/'storage_policy.json')
    else:write(out/'storage_policy.json',{'local_first':True,'reserve_bytes':1073741824,
        'historical_policy_file':'unavailable on this host; not assumed verified'})
    baseline=read(historical/'runtime/historical_reference/setup.json')['fingerprint']
    initial_state=read(initial);results=[]
    neural=canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python');planner=ROOT.parent/'curobo/.venv/bin/python'
    for label,reference in variants:
        if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
        trial=out/label;trial.mkdir();(trial/'environment_evidence').mkdir()
        scenario=read(original/'scenario.json');scenario['name']=label
        write(trial/'scenario.json',scenario)
        setup=client.request('dynamic_setup',scenario=scenario,timeout=240);write(trial/'setup.json',setup)
        mismatch=compare_configuration(baseline,setup['fingerprint'])
        limits=setup['snapshot']['joint_limits']
        expected=[min(max(q,lo),hi) for q,(lo,hi) in zip(initial_state['joint_positions'],limits)]
        if setup['initial_conditions']['joint_positions']!=expected:mismatch.append('historical initial joints')
        if setup['physics_steps']!=360:mismatch.append('historical settling steps')
        actual=setup['initial_conditions']['objects']['asset:'+scenario['asset']]
        if actual['position']!=scenario['placement']['position'] or actual['quaternion_wxyz']!=scenario['placement']['orientation']:
            mismatch.append('historical target placement')
        write(trial/'setup_verification.json',{'pass':not mismatch,
            'historical_physical_properties_equal':not mismatch,'mismatches':mismatch})
        if mismatch:raise RuntimeError(f'Frozen configuration mismatch: {mismatch[:10]}')
        write(trial/'active_asset.json',setup['asset'])
        write(trial/'guard.json',{'invariant_sha256':'setup fingerprint archived','file_sha256':{}})
        client.request('grasp_viewport_state',restore=read(historical/'original_viewport.json'))
        env=os.environ.copy();env.update(ARX_CUROBO_VALIDATION_DIR=str(client.root),GRASP_OUTPUT_DIR=str(trial),
            GRASP_TARGET='test_object',ARX_OPS_URL='http://127.0.0.1:8101',
            GRASP_FIXED_SELECTION=str(original/'selected_grasp.json'),
            ARX_LILO_POLICY_URL='http://127.0.0.1:8008',ARX_LILO_MAX_STEPS='90',
            ARX_LILO_INSTRUCTION='pick up the object',ARX_LILO_VERIFY_CLOSURE='1',
            GRASP_PREPARED_CLOSED_MODEL='1',GRASP_PLANNER='prm_trajopt',
            GRASP_BOUND_WEIGHT_MULTIPLIER='100',GRASP_TRAJOPT_POLICY='interactive',
            ARX_RESEARCH_GUARD_FILE=str(trial/'guard.json'),ARX_RESEARCH_AUDIT='0',
            GRASP_VIDEO_DIR=str(trial/'frames'),CUDA_VISIBLE_DEVICES=args.planner_gpu,
            PYTHONUNBUFFERED='1')
        env.pop('ARX_LILO_TEACHER_DIR',None);env.pop('ARX_LILO_HANDOFF_OBSERVATION',None)
        if reference:env['ARX_LILO_HANDOFF_OBSERVATION']=str(reference)
        started=time.time();error=None
        try:
            for name,python,script,argv in [
                ('capture',neural,'grasp_input.py',[]),
                ('prepare',planner,'prepare.py',['--output',str(trial),'--model-source',str(ROOT/'scripts/curobo_arx/static_pick/reference')]),
                ('execution',planner,'grasp_execute.py',[])]:
                with (trial/(name+'.log')).open('x') as log:
                    subprocess.run([str(python),str(ROOT/'scripts/curobo_arx'/script),*argv],
                        env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                if name=='prepare':
                    # Match reproduce_record's original nominal prewarm state.
                    import yaml
                    path=trial/'model_closed/arx.yml';config=yaml.safe_load(path.read_text())
                    for joint in ('right_joint27','right_joint28'):
                        config['robot_cfg']['kinematics']['lock_joints'][joint]=.022
                    path.write_text(yaml.safe_dump(config,sort_keys=False))
        except Exception:error=traceback.format_exc();write(trial/'runner_error.json',{'error':error})
        finally:
            write(trial/'boundary_snapshot.json',client.request('snapshot',timeout=300))
            if (trial/'frames').exists():write(trial/'recording_stop.json',client.request('grasp_video_stop',timeout=120))
        result=evaluate(trial);result.update(label=label,elapsed_s=time.time()-started,runner_error=error)
        write(trial/'result.json',result);results.append(result);write(out/'results.json',results)
        print(json.dumps({'label':label,'pass':result['pass'],'error':result.get('reported_error'),
            'local_actions':len(list(trial.glob('local_*_prediction.json')))}),flush=True)
        if any(sha(p)!=read(out/'registration.json')['physical_configuration_sha256'][str(p)] for p in frozen_files):
            raise RuntimeError('Protected physical source changed during comparison')
    print('COMPLETED',out,flush=True)


if __name__=='__main__':main()
