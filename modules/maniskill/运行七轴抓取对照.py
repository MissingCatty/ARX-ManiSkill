"""Fresh 2x160 execution with joint4+right arm, reusing the frozen TCP43 bank."""
from arx_project.layout import canonical_path
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import graspcl_兼容修复与160场景 as prior
from 公共 import read,save,sha,now

CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
PREVIOUS=CODE/'outputs/arx_graspcl_compat160_20261120_01'
GROUPS=['fixed_zerograsp','fixed_graspcl']
GPU=prior.GPU

def prepare(root):
    prior.imports()
    from graspcl_execute import guard
    old=read(PREVIOUS/'registration.json');guard(old)
    assert read(PREVIOUS/'progress.json')['test_completed']==320
    assert read(PREVIOUS/'运行结束.json')['all_owned_services_exited']
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    (root/'bank_tcp43').symlink_to(PREVIOUS/'bank_tcp43',target_is_directory=True)
    (root/'高度初态').symlink_to(PREVIOUS/'高度初态',target_is_directory=True)
    files=[SELF,CODE/'七轴升降规划.py',Path(prior.__file__),Path(prior.legacy.__file__),
        CODE.parent/'reachability/工具/验证升降联合规划.py',CODE.parent/'reachability/工具/机器人几何.py',
        canonical_path(CODE.parent/'reachability/实验/升降联合规划_20261120_01/七关节.yml'),
        canonical_path(CODE.parent/'reachability/实验/升降联合规划_20261120_01/含原升降关节.urdf'),
        PREVIOUS/'progress.json',PREVIOUS/'运行结束.json',PREVIOUS/'bank_frozen_before_tests.json']
    reg=deepcopy(read(PREVIOUS/GROUPS[0]/'registration.json'))
    reg.update(created_at=now(),suite=str(root),arms=[(g,'graspcl_200' if g==GROUPS[1] else 'zerograsp_200','fixed_orientation') for g in GROUPS],
        source_campaign=str(PREVIOUS),total_slots=320,physical_slot_budget=320,test_slot_budget=320,source_collection_budget=0,
        bank=str(root/'bank_tcp43'),initial_lift_m=.452,planning_dof=7,
        experimental_variable='Add authored lift joint4 to the six arm joints for reach-to-pregrasp; chosen lift held for approach/closure/lift. Fixed initial base.',
        scope='Same original non-S5 160 scenes and frozen 24-positive/4-negative bank; fresh wrist observations and inference; historical fixed6 reference, no concurrent fixed6 reruns.',
        motion_policy='7-DOF reach, 6-DOF approach and measured-closure short lift with selected lift held.',
        extra_collision_coverage='Exact translational convex sweep of original torso/left-arm colliders against all native scene colliders; original right-arm cuRobo collision policy retained.',
        success_gate='Original >=1cm object lift for 3 seconds, bilateral closure, arm settling and forbidden contacts. Added 1mm lift settling, no relaxed gate.',
        automatic_grasp_retries=False,old_results_carried=0,prior_test_results_carried=0,
        source_feedback_ids_in_memory_order=read(PREVIOUS/'bank_tcp43/bank_complete.json')['source_ids'],
        user_authorization='2026-11-21: test success with the lift joint added; same two chains and 160 scenes. The later nine-DOF base test is separate.',
        startup_motion_budget='Two 1cm lift motions per warm simulator on independent lemon, zero startup grasps; return/reset to original 45.2cm initial conditions.')
    for g in GROUPS:
        folder=root/g
        for n in ['entries','trials','claims','backend_requests','workers','resource_wait']: (folder/n).mkdir(parents=True,exist_ok=True)
        (folder/'resident').symlink_to(root/'resident',target_is_directory=True)
        (folder/'reference').symlink_to(PREVIOUS/g/'reference',target_is_directory=True)
        shutil.copy2(PREVIOUS/g/'storage_policy.json',folder/'storage_policy.json')
        child=deepcopy(reg);child.update(experiment_arm=g,pipeline_profile='graspcl_200' if g==GROUPS[1] else 'zerograsp_200')
        for row in child['schedule']:
            e=read(PREVIOUS/g/'entries'/(row['id']+'.json'));e.update(campaign_registration=str(folder/'registration.json'),
                planning_dof=7,initial_lift_m=.452,experiment_arm=g)
            path=folder/'entries'/(row['id']+'.json');save(path,e);files.append(PREVIOUS/g/'entries'/(row['id']+'.json'))
        save(folder/'registration.json',child)
    hashes={**reg['source_hashes'],**{str(p.resolve()):sha(p) for p in set(files)}}
    for folder in [root,*[root/g for g in GROUPS]]:
        d=reg if folder==root else read(folder/'registration.json');d['source_hashes']=hashes;save(folder/'registration.json',d)
    for p in [SELF,CODE/'七轴升降规划.py']:shutil.copy2(p,root/'source'/p.name)
    save(root/'progress.json',{'phase':'prepared','test_completed':0,'test_registered':320,'groups':{}})
    print('PREPARED 7DOF, 2x160 slots, frozen bank retained',flush=True)

def progress(root,phase):
    groups={}
    for g in GROUPS:
        claims=[read(p) for p in (root/g/'claims').glob('*.json')];rows=[read(root/g/'trials'/c['id']/'result.json') for c in claims if c['state']=='complete']
        from collections import Counter
        groups[g]={'registered':160,'completed':len(rows),'planning_passes':sum(x['planning_success'] for x in rows),
            'successes':sum(x['success'] for x in rows),'physical_attempts':sum(x['physical_attempt'] for x in rows),
            'in_progress_or_unknown':len(claims)-len(rows),'failure_reasons':dict(Counter(x['reason'] for x in rows if not x['success']))}
    save(root/'progress.json',{'phase':phase,'test_completed':sum(g['completed'] for g in groups.values()),'test_registered':320,'groups':groups,'updated_at':now()})

def serve(arguments):
    prior.imports()
    if arguments[arguments.index('--kind')+1]=='curobo':
        prior.legacy.policy().install_collision_policy()
        from graspcl_示教兼容 import install
        install()
        import 七轴升降规划 as joint
        joint.install_planner()
    prior.serve(arguments)

def simulator(root,address,owner):
    prior.imports()
    import short_lift_bench as bench
    import 七轴升降规划 as joint
    run,baseline=bench.core.batch.original_modules();joint.install_execution(run)
    original_make=run.make
    def make(*args,**kwargs):
        env=original_make(*args,**kwargs)
        if args and args[0]=='014_lemon' and not (root/'test_started.json').exists():
            reset=env.reset;done=False
            def checked_reset(*a,**kw):
                nonlocal done
                result=reset(*a,**kw)
                if not done:
                    for _ in range(80):env._step_action(run.torch.tensor(env.initial_q[None],device=env.device))
                    joint.training_probe(env,run,root/'startup_validation'/(address.stem+'_升降执行接口.json'))
                    done=True;result=reset(*a,**kw)
                return result
            env.reset=checked_reset
        return env
    run.make=make;baseline.make=make
    original=bench.execute
    def checked(directory,entry):
        original(directory,entry);reg=read(read(entry)['campaign_registration']);prior.validate_trial(directory,reg)
        result=read(directory/'result.json')
        if result['planning_success']:
            plan=read(directory/'plan_result.json');height=plan['selected']['selected_lift_m']
            assert plan['selected']['reach_joint_names']==joint.NAMES
            if (directory/'pregrasp_state.json').exists():
                state=read(directory/'pregrasp_state.json');actual=state['qpos'][state['joint_names'].index('joint4')]
                assert abs(actual-height)<.0015
            save(directory/'七轴执行核验.json',{'passed':True,'planning_dof':7,'selected_lift_m':height,
                'reach_joint_names':joint.NAMES,'initial_conditions_preserved':True})
    bench.execute=checked;prior.legacy.simulator(root,address,owner)

def run(root):
    prior.imports()
    from graspcl_execute import guard
    from benchmark_lifecycle import mark_ready
    from arx_service_readiness import wait_ready
    reg=read(root/'registration.json');guard(reg)
    resident=root/'resident';resident.mkdir();sockets=Path('/tmp')/('arxjoint7_'+sha(root/'registration.json')[:10]);sockets.mkdir(mode=0o700)
    children={};sims={};streams=[];config={'sockets':{}};began=time.perf_counter();failed=False
    def launch(kind):
        python=canonical_path(CODE.parent/{'zerograsp':'tmp/arx-fullchain/env/bin/python','curobo':'curobo/.venv/bin/python','graspcl':'tmp/arx-graspcl-env/bin/python'}[kind])
        address=sockets/(kind+'.sock');config['sockets'][kind]=str(address)
        command=['bash','-c','source "$1"; shift; exec "$@"','joint7-service',str(CODE.parent/'arx_core/scripts/curobo_arx/environment.sh'),
            str(python),str(SELF),'serve','--kind',kind,'--socket',str(address),'--root',str(resident),'--warmup-trial',reg['warmup_trial']]
        if kind=='graspcl':command+=['--bank',str(root/'bank_tcp43'),'--policy-registration',str(root/GROUPS[1]/'registration.json')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if kind=='graspcl' else GPU,ARX_NEURAL_GPU=GPU,ARX_PLANNER_GPU=GPU,ARX_SIM_GPU=GPU,
            LILO_GPU=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GRASP_TRAJOPT_POLICY='interactive');env.pop('PYTHONPATH',None)
        stream=(resident/(kind+'.log')).open('x');streams.append(stream)
        children[kind]=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        health=wait_ready(address,resident/(kind+'_ready.json'),children[kind],kind,timeout=1800);print('READY',kind,children[kind].pid,health['startup_seconds'],flush=True)
    def start_sim(g):
        address=sockets/(g+'.sock');env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,MANISKILL_GPU_UUID=GPU,ARX_SIM_GPU=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        stream=(resident/(g+'_simulation.log')).open('x');streams.append(stream)
        command=['bash',str(CODE/'run.sh'),str(SELF),'simulator','--output',str(root),'--socket',str(address),'--owner-pid',str(os.getpid())]
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT);sims[g]=(child,address)
        deadline=time.monotonic()+1800
        while not address.exists():
            assert child.poll() is None,'Seven-joint simulation preload failed'
            if time.monotonic()>deadline:raise TimeoutError('Simulation preload exceeded budget')
            time.sleep(1)
        assert prior.legacy.rpc(address,{'command':'health'})['runtime_import_count']==1
    def run_group(g):
        folder=root/g;group=read(folder/'registration.json')
        for entry in group['schedule']:
            if (root/'STOP.json').exists():return
            guard(group);assert shutil.disk_usage(root).free>1073741824
            slot=entry['id'];path=folder/'claims'/(slot+'.json');reserved=path.with_suffix('.reserved');trial=folder/'trials'/slot
            assert not path.exists() and not reserved.exists() and not trial.exists()
            with reserved.open('x') as f:f.write(now()+'\n')
            claim={'state':'claimed','id':slot,'group':g,'claimed_at':now()};save(path,claim)
            try:
                assert sims[g][0].poll() is None
                response=prior.legacy.rpc(sims[g][1],{'command':'execute','output':str(trial),'entry':str(folder/'entries'/(slot+'.json')),'log':str(folder/'entries'/(slot+'.log'))},timeout=1800)
                assert response['status']=='ok',response
                assert read(trial/'independent_evaluation.json')['passed']
                guard(group);result=read(trial/'result.json');claim.update(state='complete',finished_at=now(),result_sha256=sha(trial/'result.json'));save(path,claim)
                with (root/'summary.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);progress(root,'testing_joint7_pair')
                print('RESULT',g,slot,result['success'],result['reason'],flush=True)
            except BaseException:
                save(root/'STOP.json',{'time':now(),'group':g,'id':slot,'error':traceback.format_exc(),'automatic_replay':False});raise
    try:
        progress(root,'starting_components')
        for kind in ['zerograsp','curobo','graspcl']:launch(kind)
        save(resident/'backend_config.json',config);mark_ready(resident,['zerograsp','graspcl','curobo'],config,children)
        for g in GROUPS:start_sim(g)
        save(root/'bank_frozen_before_tests.json',{'time':now(),'bank_complete_sha256':sha(root/'bank_tcp43/bank_complete.json'),'test_claims':0,'source':str(PREVIOUS/'bank_tcp43')})
        save(root/'test_started.json',{'time':now(),'all_components_ready':True,'test_registered':320,'startup_probes_passed':True})
        with ThreadPoolExecutor(max_workers=2) as workers:
            jobs=[workers.submit(run_group,g) for g in GROUPS]
            for job in jobs:job.result()
        progress(root,'complete')
    except BaseException:
        failed=True;save(root/'控制器故障.json',{'time':now(),'error':traceback.format_exc()});progress(root,'halted');raise
    finally:
        exits={}
        for name,(child,address) in {**{k:(v,Path(config['sockets'][k])) for k,v in children.items()},**sims}.items():
            if child.poll() is None:
                try:prior.legacy.rpc(address,{'command':'shutdown'},timeout=10);child.wait(timeout=20)
                except Exception:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            exits[name]={'pid':child.pid,'returncode':child.returncode,'exited':child.poll() is not None}
        for stream in streams:stream.close()
        save(root/'运行结束.json',{'time':now(),'failed':failed,'all_owned_services_exited':all(x['exited'] for x in exits.values()),'services':exits,'wall_seconds':time.perf_counter()-began})

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='serve':serve(sys.argv[2:])
    else:
        p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','simulator']);p.add_argument('--output',type=Path,required=True);p.add_argument('--socket',type=Path);p.add_argument('--owner-pid',type=int)
        a=p.parse_args();root=a.output.resolve()
        if a.stage=='prepare':prepare(root)
        elif a.stage=='run':run(root)
        else:simulator(root,a.socket,a.owner_pid)
