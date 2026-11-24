"""Run the unchanged two 200-case pipelines at the preregistered selected lift."""
from arx_project.layout import canonical_path
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
from 公共 import ROOT, MS, read, save, sha, now

SELF=Path(__file__).resolve()
PRIOR=MS/'outputs/arx_physical_table_parallel_20261118_01'
POLICY=MS/'tmp/physical_table_pair_20261118_01.py'
GPU='GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
ARMS=[('fixed_zerograsp','zerograsp_200','zerograsp'),('fixed_graspcl','graspcl_200','graspcl')]


def imports():
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS/'tmp')]
    import arx_runtime_imports
    arx_runtime_imports.install()


def policy():
    spec=importlib.util.spec_from_file_location('height_frozen_physical_table_policy',POLICY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(root,selection):
    imports()
    from graspcl_execute import guard
    chosen=read(selection/'高度选择.json');assert chosen['passed'] and chosen['physical_attempts']==0
    height=chosen['selected_lift_m'];assert 0<=height<=.46
    original=read(PRIOR/'registration.json');guard(original)
    reg=deepcopy(original)
    derived=root/'高度初态';derived.mkdir()
    (root/'source').mkdir()
    inputs=Path(original['inputs'])
    initial=read(inputs/'common_initial_conditions.json')
    historical=next((PRIOR/'zerograsp/fixed_zerograsp/trials').glob('*/state.json'))
    joint_names=read(historical)['joint_names'];index=joint_names.index('joint4')
    original_height=initial['initial_q'][index]
    shifted_initial=deepcopy(initial);shifted_initial['initial_q'][index]=float(np.float32(height))
    save(derived/'common_initial_conditions.json',shifted_initial)
    (derived/'declared_resets').mkdir()
    checks=[];additional=[SELF,selection/'登记.json',selection/'高度选择.json',derived/'common_initial_conditions.json',POLICY]
    for row in original['schedule']:
        slot=row['id'];src=inputs/'declared_resets'/slot/'declared_reset.json'
        if src.exists():
            reset=read(src);before=deepcopy(reset);reset['initial_q'][index]=float(np.float32(height))
            save(derived/'declared_resets'/slot/'declared_reset.json',reset)
            original_reset=deepcopy(reset);original_reset['initial_q'][index]=before['initial_q'][index]
            assert original_reset==before
            additional += [src,derived/'declared_resets'/slot/'declared_reset.json']
    for arm,profile,prior_group in ARMS:
        old=PRIOR/prior_group/arm
        folder=root/arm
        for name in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:(folder/name).mkdir(parents=True)
        save(folder/'reference/common_initial_conditions.json',shifted_initial)
        shutil.copy2(inputs/'storage_policy.json',folder/'storage_policy.json')
        (folder/'resident').symlink_to(root/'resident',target_is_directory=True)
        oldreg=read(old/'registration.json')
        child=deepcopy(oldreg)
        child.update(suite=str(root),inputs=str(derived),common_reset_source=str(derived/'common_initial_conditions.json'),
            source_campaign=str(PRIOR),source_evaluation_results_carried=0,prior_test_results_carried=0,
            inherited_completed=0,selected_lift_m=height,only_robot_reset_joint_changed='joint4',
            required_components=['zerograsp','graspcl','curobo'],runtime_backend_gpu=GPU,runtime_scene_gpus=[GPU],
            backend_lock=str(root/'backend.lock'),report_path=str(root/'结果.md'))
        child['frozen_algorithm']='Original native postprocessing, fixed-orientation TCP43 truncation, frozen feedback, pregrasp/approach/closure/lift/hold, collisions and evaluator. Only joint4 initial state changes; observation goal world-Z follows the same arm-joint pose.'
        for entry_path in sorted((old/'entries').glob('*.json')):
            if entry_path.name.endswith('_binding.json'):continue
            entry=read(entry_path);slot=entry['id']
            ref=folder/'reference'/slot;ref.mkdir()
            oldinitial=old/'trials'/slot/'initial_conditions.json'
            if oldinitial.exists():
                init=read(oldinitial);old_q=deepcopy(init['initial_q']);init['initial_q'][index]=float(np.float32(height))
                assert [a for i,a in enumerate(init['initial_q']) if i!=index]==[a for i,a in enumerate(old_q) if i!=index]
                save(ref/'initial_conditions.json',init)
            else:save(ref/'initial_conditions.json',shifted_initial)
            if 'observation' in entry:
                old_observation=deepcopy(entry['observation'])
                entry['observation']['goal_world'][2][3]+=float(np.float32(height))-original_height
                assert entry['observation']['q']==old_observation['q']
            admission={'entry':{'observation':entry.get('observation'),'scene':entry.get('scene')}}
            save(ref/'admission.json',admission)
            entry.update(campaign_registration=str(folder/'registration.json'),historical_trial=str(ref),
                source_admission=str(ref/'admission.json'),selected_lift_m=height,
                original_reference_entry=str(entry_path),original_reference_entry_sha256=sha(entry_path))
            save(folder/'entries'/entry_path.name,entry)
            evidence={str(entry_path):sha(entry_path)}
            for file in [ref/'initial_conditions.json',ref/'admission.json']:
                evidence[str(file)]=sha(file);additional.append(file)
            save(folder/'entries'/(slot+'_binding.json'),{'entry_sha256':sha(folder/'entries'/entry_path.name),
                'source_evidence_sha256':evidence,'height_only_reference_derivation':True})
            checks.append({'arm':arm,'id':slot,'same_scene_recipe':entry.get('scene')==read(entry_path).get('scene'),
                'same_observation_arm_joints':entry.get('observation',{}).get('q')==read(entry_path).get('observation',{}).get('q'),
                'original_initial_height_m':original_height,'new_initial_height_m':height})
        child['source_hashes']=deepcopy(original['source_hashes'])
        save(folder/'registration.json',child)
    assert len(checks)==400 and all(c['same_scene_recipe'] and c['same_observation_arm_joints'] for c in checks)
    save(root/'仅高度差异预检.json',{'passed':True,'records':checks,'changed_initial_joint':'joint4',
        'observation_world_pose_z_is_derived_from_same_arm_q':True,'camera_mount_intrinsics_unchanged':True,
        'physics_collision_grasping_and_success_rules_unchanged':True,'physical_attempts':0})
    reg.update(created_at=now(),selected_lift_m=height,selection_record=str(selection/'高度选择.json'),
        selected_height_frozen_before_test_claims=True,source_campaign=str(PRIOR),
        source_evaluation_results_carried=0,prior_test_results_carried=0,inherited_completed=0,
        total_slots=400,physical_slot_budget=400,inputs=str(derived),common_reset_source=str(derived/'common_initial_conditions.json'),
        runtime_backend_gpu=GPU,runtime_scene_gpus=[GPU,GPU],scene_workers=2,
        backend_lock=str(root/'backend.lock'),required_components=['zerograsp','graspcl','curobo'],
        height_only_change=True,only_robot_reset_joint_changed='joint4',
        pipeline_scope='Fresh observations and full original fixed-orientation ZeroGrasp+cuRobo / ZeroGrasp+GraspCL+cuRobo. No MPC, no changes to perception/planning/feedback/physics/success settings.',
        test_design_scope='Same registered 200 scenes; height selected from their archived geometry. Old 22cm results are historical references, not a new concurrent height-control arm.',
        prior_test_outcomes_used_for_training=False,testing_feedback_added=0,
        one_attempt_per_arm_scene=True,automatic_grasp_retries=False,
        user_authorization='2026-11-20: search for the highest reachability height, select one height and test the two pipelines changing only lift height.',
        report_path=str(root/'结果.md'),
        runtime_persistence='One ZeroGrasp, one frozen GraspCL and one cuRobo service; two simulation workers. All warm before the first slot.')
    reg['frozen_algorithm']='Only joint4 initial state changes; all original two-pipeline methods, fixed orientation, physics, camera mount/K, TCP43, feedback and physical success gates retained.'
    reg.pop('parallel_suites',None)
    for path in additional:reg['source_hashes'][str(path)]=sha(path)
    reg['source_hashes'][str(root/'仅高度差异预检.json')]=sha(root/'仅高度差异预检.json')
    for name,_,_ in ARMS:
        child=read(root/name/'registration.json');child['source_hashes']=reg['source_hashes']
        save(root/name/'registration.json',child)
    save(root/'registration.json',reg)
    shutil.copy2(SELF,root/'source'/SELF.name)
    for path in [POLICY,MS/'short_lift_bench.py',MS/'short_lift_protocol.py',MS/'short_lift_services.py',
                 MS/'native_pair_timing.py',MS/'table_truncation_bench.py',MS/'postprocess_bench.py']:
        shutil.copy2(path,root/'source'/path.name)
    summarize(root,'prepared')
    print('FORMAL PREPARED',height,'400 fresh slots',flush=True)


def summarize(root,phase):
    reg=read(root/'registration.json');groups={}
    for arm,_,_ in ARMS:
        rows=[];unknown=0
        for claim in (root/arm/'claims').glob('*.json'):
            c=read(claim)
            if c.get('state')=='complete':rows.append(read(root/arm/'trials'/claim.stem/'result.json'))
            else:unknown+=1
        groups[arm]={'completed':len(rows),'registered':200,'planning_passes':sum(bool(r['planning_success']) for r in rows),
            'physical_attempts':sum(bool(r['physical_attempt']) for r in rows),'successes':sum(bool(r['success']) for r in rows),
            'allocated_unknown':unknown,'failure_reasons':{}}
        for r in rows:
            if not r['success']:groups[arm]['failure_reasons'][r['reason']]=groups[arm]['failure_reasons'].get(r['reason'],0)+1
    value={'phase':phase,'selected_lift_m':reg['selected_lift_m'],'registered':400,
           'completed':sum(g['completed'] for g in groups.values()),'groups':groups,'updated_at':now()}
    save(root/'progress.json',value)
    lines=['# 选定升降高度下的两条抓取链路对照','',f"本轮升降轴 joint4 = {reg['selected_lift_m']*100:.1f} cm。唯一改变的机器人初态关节是升降轴；其余机械臂初态、物体、相机标定、规划与抓取规则均沿用原 200 例。",'',
           '| 链路 | 完成/登记 | 规划通过 | 实际抓取成功 | 未知中断 |','| --- | ---: | ---: | ---: | ---: |']
    for arm,label in [('fixed_zerograsp','ZeroGrasp＋cuRobo'),('fixed_graspcl','ZeroGrasp＋GraspCL＋cuRobo')]:
        g=groups[arm];lines.append(f"| {label} | {g['completed']}/200 | {g['planning_passes']} | {g['successes']} | {g['allocated_unknown']} |")
    lines+=['','实际成功要求目标抬高至少 1 cm 并持续保持 3 秒，同时满足原双指接触、收敛和禁止接触要求。','',
            '本轮高度先用归档几何选择，再重新采集观测并执行。原 22 cm 的结果单独保留为历史参考；本轮没有追加同期 22 cm 试验。',f'','当前状态：'+phase+'。']
    (root/'结果.md').write_text('\n'.join(lines)+'\n')
    return value


def rpc(address,message,timeout=1800):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout);connection.connect(str(address))
        with connection.makefile('rwb') as stream:
            stream.write(json.dumps(message).encode()+b'\n');stream.flush()
            return json.loads(stream.readline(64*1024*1024))


def simulator(root,address,owner):
    import faulthandler
    faulthandler.enable()
    imports()
    import short_lift_bench as bench
    import short_lift_protocol as lift
    import environment
    from graspcl_execute import guard
    reg=read(root/'registration.json');guard(reg)
    before=deepcopy(environment.INITIAL_JOINTS)
    environment.INITIAL_JOINTS={**before,'joint4':reg['selected_lift_m']}
    assert {k for k in before if before[k]!=environment.INITIAL_JOINTS[k]} <= {'joint4'}
    run,baseline=bench.core.batch.original_modules()
    lift.install_execution_policy(run,baseline)
    # Keep one renderer context alive across the original per-trial environments.
    make=run.make;keeper=[]
    def persistent_make(*args,**kwargs):
        env=make(*args,**kwargs)
        if not keeper:
            import sapien
            keeper.append(sapien.render.SapienRenderer(env._render_device))
        return env
    run.make=persistent_make;baseline.make=persistent_make
    training=persistent_make('014_lemon',perception_camera='right_wrist')
    try:
        training.reset(seed=0)
        names=[j.name for j in training.agent.robot.active_joints]
        initial=np.array(training.initial_q)
        expected=np.array(read(root/'高度初态/common_initial_conditions.json')['initial_q'])
        assert np.array_equal(initial,expected)
        snapshot=run.snapshot(training,training.get_obs())
        assert abs(snapshot['qpos'][names.index('joint4')]-reg['selected_lift_m'])<1e-6
        save(root/'resident'/(address.stem+'_训练预检.json'),{'passed':True,'training_object':'014_lemon',
            'test_slots_claimed':0,'physical_grasps':0,'selected_lift_m':reg['selected_lift_m'],
            'initial_q_matches_height_only_reference':True,'snapshot':snapshot,'renderer_context_preloaded':True})
    finally:training.close()
    ready={'status':'ready','pid':os.getpid(),'owner_pid':owner,'runtime_import_count':1,
           'height_m':reg['selected_lift_m'],'only_initial_joint_changed':'joint4','requests_completed':0}
    address.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    assert not address.exists()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(address));os.chmod(address,0o600);server.listen(8)
        while True:
            with server.accept()[0] as connection,connection.makefile('rwb') as stream:
                message=json.loads(stream.readline(1024*1024));command=message.get('command','health')
                if command=='health':response=ready
                elif command=='shutdown':
                    stream.write(json.dumps({'status':'stopped','pid':os.getpid()}).encode()+b'\n');stream.flush();break
                else:
                    os.kill(owner,0);guard(reg)
                    try:
                        trial=Path(message['output']);entry=Path(message['entry']);start=time.perf_counter()
                        assert root in trial.resolve().parents
                        with Path(message['log']).open('x') as log,redirect_stdout(log),redirect_stderr(log):
                            bench.execute(trial,entry)
                        response={'status':'ok','returncode':0,'pid':os.getpid(),'wall_seconds':time.perf_counter()-start}
                    except BaseException:response={'status':'error','error':traceback.format_exc(),'pid':os.getpid()}
                    ready['requests_completed']+=1
                stream.write(json.dumps(response).encode()+b'\n');stream.flush()
    address.unlink()


def run(root):
    imports()
    from graspcl_execute import guard
    from benchmark_lifecycle import mark_ready
    from arx_service_readiness import wait_ready
    import short_lift_bench as bench
    reg=read(root/'registration.json');guard(reg)
    resident=root/'resident';resident.mkdir()
    sockets=Path('/tmp')/('arxheight_'+hashlib.sha256(str(root).encode()).hexdigest()[:10])
    sockets.mkdir(mode=0o700,exist_ok=False)
    streams=[];children={};simulators={};config={'sockets':{}}
    start=time.perf_counter();failed=False
    try:
        for kind in ['zerograsp','graspcl','curobo']:
            python=canonical_path(MS.parent/{'zerograsp':'tmp/arx-fullchain/env/bin/python','graspcl':'tmp/arx-graspcl-env/bin/python','curobo':'curobo/.venv/bin/python'}[kind])
            address=sockets/(kind+'.sock');config['sockets'][kind]=str(address)
            cmd=['bash','-c','source "$1"; shift; exec "$@"','height-frozen-service',
                 str(MS.parent/'arx_core/scripts/curobo_arx/environment.sh'),str(python),str(POLICY),'serve',
                 '--kind',kind,'--socket',str(address),'--root',str(resident),'--warmup-trial',reg['warmup_trial']]
            if kind=='graspcl':cmd+=['--bank',reg['bank'],'--policy-registration',str(root/'registration.json')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if kind=='graspcl' else GPU,ARX_NEURAL_GPU=GPU,
                ARX_PLANNER_GPU=GPU,ARX_SIM_GPU=GPU,LILO_GPU=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GRASP_TRAJOPT_POLICY='interactive')
            env.pop('PYTHONPATH',None)
            stream=(resident/(kind+'.log')).open('x');streams.append(stream)
            children[kind]=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
            health=wait_ready(address,resident/(kind+'_ready.json'),children[kind],kind,timeout=1800)
            print('READY',kind,children[kind].pid,health['startup_seconds'],flush=True)
        save(resident/'backend_config.json',config)
        mark_ready(resident,reg['required_components'],config,children)
        for arm,_,_ in ARMS:
            address=sockets/(arm+'.sock')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,MANISKILL_GPU_UUID=GPU,ARX_SIM_GPU=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            stream=(resident/(arm+'_simulation.log')).open('x');streams.append(stream)
            command=['bash',str(MS/'run.sh'),str(SELF),'simulator','--output',str(root),'--socket',str(address),'--owner-pid',str(os.getpid())]
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT);simulators[arm]=(child,address)
            deadline=time.monotonic()+1800
            while not address.exists():
                assert child.poll() is None,'Simulation import failed'
                if time.monotonic()>deadline:raise TimeoutError('Simulation preload timed out')
                time.sleep(1)
            assert rpc(address,{'command':'health'})['runtime_import_count']==1
        save(resident/'simulation_all_ready.json',{'passed':True,'workers':{k:{'pid':c.pid,'socket':str(a)} for k,(c,a) in simulators.items()},'test_slots_claimed':0})
        save(root/'test_started.json',{'time':now(),'all_components_ready':True,'height_m':reg['selected_lift_m']})
        summarize(root,'running')
        def group(arm):
            child,address=simulators[arm]
            for schedule in reg['schedule']:
                if (root/'STOP.json').exists():return
                guard(reg)
                slot=schedule['id'];folder=root/arm;entry=folder/'entries'/(slot+'.json');trial=folder/'trials'/slot
                claim_path=folder/'claims'/(slot+'.json')
                reserved=claim_path.with_suffix('.reserved')
                assert not trial.exists() and not claim_path.exists() and not reserved.exists()
                claim={'state':'claimed','claimed_at':now(),'id':slot,'arm':arm,'height_m':reg['selected_lift_m']}
                with reserved.open('x') as stream:stream.write(now()+'\n')
                save(claim_path,claim)
                e=read(entry)
                try:
                    if e.get('admission_failed'):
                        trial.mkdir();save(trial/'result.json',{**e,'success':False,'planning_success':False,'physical_attempt':False,'reason':'independent_scene_admission_failed'})
                        bench.placeholder(trial,'Historical registered initialization failure; no scene replacement')
                        save(trial/'independent_evaluation.json',{'passed':True,'independently_successful':False,'physical_attempt':False,'admission_failed':True})
                    else:
                        assert child.poll() is None
                        response=rpc(address,{'command':'execute','output':str(trial),'entry':str(entry),'log':str(folder/'entries'/(slot+'.log'))},timeout=1800)
                        if response['status']!='ok':raise RuntimeError(response['error'])
                        assert read(trial/'independent_evaluation.json')['passed']
                    result=read(trial/'result.json');guard(reg)
                    claim.update(state='complete',finished_at=now(),result_sha256=sha(trial/'result.json'))
                    save(folder/'claims'/(slot+'.json'),claim)
                    print('RESULT',arm,slot,result['success'],result['reason'],flush=True)
                    with (root/'summary.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX);summarize(root,'running')
                except BaseException:
                    save(root/'STOP.json',{'time':now(),'arm':arm,'slot':slot,'error':traceback.format_exc(),'claimed_slot_will_not_be_replayed':True})
                    raise
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures=[workers.submit(group,arm) for arm,_,_ in ARMS]
            for future in futures:future.result()
        result=summarize(root,'complete')
        assert result['completed']==400
    except BaseException:
        failed=True;save(root/'控制器故障.json',{'time':now(),'error':traceback.format_exc()})
        raise
    finally:
        exits={}
        for kind,(child,address) in {**{k:(c,Path(config['sockets'][k])) for k,c in children.items()},**simulators}.items():
            if child.poll() is None:
                try:rpc(address,{'command':'shutdown'},timeout=10);child.wait(timeout=20)
                except Exception:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            exits[kind]={'pid':child.pid,'returncode':child.returncode,'exited':child.poll() is not None}
        for stream in streams:stream.close()
        save(root/'运行结束.json',{'time':now(),'services':exits,'all_owned_services_exited':all(c['exited'] for c in exits.values()),'failed':failed,'wall_seconds':time.perf_counter()-start})
        if (root/'registration.json').exists():summarize(root,'halted' if failed else 'complete')


def chain(root,selection):
    root.mkdir(parents=True,exist_ok=False)
    save(root/'等待高度选择.json',{'created_at':now(),'selection':str(selection),'physical_budget':400,
        'user_authorization':'Select the highest reachability height first, then run the two 200-scene pipelines, changing height only.',
        'source_sha256':{str(SELF):sha(SELF)},'physical_attempts':0})
    while not (selection/'高度选择.json').exists():
        if (selection/'故障.json').exists():raise RuntimeError('Height selection failed; physical tests did not start')
        time.sleep(3)
    while True:
        state=subprocess.check_output(['systemctl','--user','show','arx-height-candidates-20261120-01','-p','MainPID','--value'],text=True).strip()
        if state=='0':break
        time.sleep(1)
    prepare(root,selection)
    run(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['chain','prepare','run','simulator'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path)
    p.add_argument('--socket',type=Path);p.add_argument('--owner-pid',type=int)
    a=p.parse_args();root=a.output.resolve()
    if a.stage=='chain':chain(root,a.selection.resolve())
    elif a.stage=='prepare':prepare(root,a.selection.resolve())
    elif a.stage=='run':run(root)
    else:simulator(root,a.socket,a.owner_pid)
