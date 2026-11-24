"""Retest the twelve fixed-blocker CLARO failures with continuous grasp MPC."""
from arx_project.layout import canonical_path
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
from types import SimpleNamespace

CODE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
CLARO = CODE.parent/'claro'
sys.path[:0] = [str(CODE), str(CODE/'arx_pipeline'), str(CLARO/'工具')]
os.environ.setdefault('MS_ASSET_DIR', str(CODE/'assets'))
from arx_quiet_diagnostics_20261119 import install as quiet
quiet()
import arx_runtime_imports
arx_runtime_imports.install()
import numpy as np
import graspcl_pipeline as common
import 做配对验证 as pairing
from resident_backend import rpc

read, save, sha = common.read, common.save, common.sha
SOURCE = CLARO/'docs/06_做配对验证/resources'
OLD = SOURCE/'S5_固定挡物接续01/配对合并验收.json'
PILOT = CODE/'outputs/arx_direct_grasp_pilot_20261120_05'
PLANNER_GPU = 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'
NEURAL_GPU = 'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
REQUIRED = ['zerograsp','curobo']


def guard(reg):
    for filename, digest in reg['source_hashes'].items():
        assert sha(Path(filename)) == digest, 'Frozen file changed: '+filename


def prepare(root):
    original = read(OLD)
    rows = [row for row in original['rows'] if not row['success']]
    assert len(rows) == 12 and {row['case'] for row in rows} == set(pairing.CASE_IDS)-{'I0069'}
    prior = read(PILOT/'registration.json')
    sources = {filename: digest for filename,digest in prior['source_hashes'].items()
        if Path(filename).suffix == '.py' or any(token in filename for token in [
            'assets/arx/', 'static_pick/reference/', 'arx-sim/config/', 'zerograsp_synthetic.ckpt'])}
    files = [SELF, OLD, Path(pairing.__file__), Path(pairing.removal.__file__),
             Path(pairing.observation.__file__), Path(pairing.regrasp.__file__), pairing.removal.POLICY,
             CODE/'arx_direct_terminal_20261120.py', CODE/'arx_direct_verifier_20261120.py']
    schedule = []
    for old in rows:
        archived = SOURCE/old['run_directory']
        record = read(archived/'配对结果.json')
        assert record['status'] == 'completed' and not record['success']
        grasp = archived/'01_抓挡';trial = Path(old['source_trial'])
        entry = {k:old[k] for k in ['case','id','target_model','blocker','blocker_model','source_trial']}
        entry.update(archived_trial=str(archived), archived_blocker_grasp=str(grasp),
            original_phase=old['phase'], original_reason=old['reason'])
        schedule.append(entry)
        files += [archived/'配对结果.json']
        files += [trial/name for name in ['result.json','state.json','initial_conditions.json',
            'observation_request.json','observation_plan_result.json']]
        files += [grasp/name for name in ['state.json','raw_grasps.npy','graspcl/unrefined_candidate_pool.json']]
        files += list((grasp/'camera').glob('*'))+list((grasp/'network/raw_outputs').glob('*.grasp.npy'))
        for geometry in read(grasp/'state.json')['scene_meshes'].values():files.append(Path(geometry['file_path']))
    sources.update({str(p.resolve()):sha(p) for p in files})
    root.mkdir(exist_ok=False);(root/'source').mkdir();(root/'trials').mkdir();(root/'runtime_01').mkdir()
    reg = {'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'schedule':schedule,
        'total_slots':12,'required_components':REQUIRED, 'direct_grasp':True,
        'source_hashes':sources,'prior_summary':str(OLD),'prior_full_chain_successes_in_selected_cases':0,
        'pipeline':'Fixed blocker -> native ZeroGrasp -> insertion truncation -> continuous cuRobo route/MPC -> unchanged placement -> new wrist observation -> native ZeroGrasp -> continuous MPC regrasp',
        'graspcl_enabled':False,'test_feedback_updates':0,'fixed_orientation':True,'tcp_offset_m':.043,
        'grasp_lift_target_m':.03,'minimum_object_hold_lift_m':.01,'hold_seconds':3,
        'blocker_inputs':'Reuse the exact archived wrist frame and native predictions; current measured robot and scene remain planning inputs.',
        'target_inputs':'Fresh post-removal wrist capture and native inference in the same simulator instance.',
        'motion_policy':'Same frozen continuous grasp MPC and terminal corridor as pilot 05; no pregrasp target, IK or stage.',
        'placement_policy':'Original carry/lower/release/retreat planning, geometry and physical criteria unchanged; not converted to MPC.',
        'attempt_budget':'One clearance attempt and at most one target grasp per registered scene. Allocated cases are never replayed.',
        'runtime_backend_gpu':PLANNER_GPU,'runtime_sim_gpu':PLANNER_GPU,'runtime_neural_gpu':NEURAL_GPU,
        'scope':'Targeted retest of twelve known failures, not an estimate on all scenes.'}
    save(root/'registration.json',reg)
    for p in files:
        if p.suffix=='.py':shutil.copy2(p,root/'source'/p.name)
    save(root/'preflight.json',{'passed':True,'original_case_count':13,'selected_failed_cases':12,
        'excluded_previously_successful_case':'I0069','physical_executions':0,
        'same_scene_and_blocker':True,'no_demonstration_or_feedback_bank':True})
    guard(reg);report(root,'prepared')
    print('PREPARED',root,[e['case'] for e in schedule],flush=True)


def bind_guard(directory,root):
    save(directory/'source_guard.json',{'registration':str(root/'registration.json'),
                                      'sha256':sha(root/'registration.json')})


def planner(args):
    import resident_backend as service
    import resident_planner
    import table_truncation_model as truncation
    from checked_native_services import checked_pool
    from fast_resident_planner import FastPlannerPool
    import short_lift_planner
    import arx_direct_grasp_20261120 as direct
    import arx_direct_terminal_20261120 as terminal
    spec = importlib.util.spec_from_file_location('claro_continuous_physical_policy',pairing.removal.POLICY)
    policy = importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    policy.install_collision_policy();terminal.install();direct.install_resident()
    resident_planner.GraspCLPlanner = short_lift_planner.ShortLiftPlanner
    methods = {'placement':pairing.removal.placement,'measured_lower':pairing.removal.plan_measured_lower,
               'retreat':pairing.removal.plan_retreat}
    class Planner(truncation.make_pool(checked_pool(FastPlannerPool))):
        def request(self,directory,**request):
            import torch
            directory = Path(directory)
            receipt = read(directory/'source_guard.json')
            assert receipt['sha256']==sha(Path(receipt['registration']))
            action = request.pop('action',None)
            if action in methods:return methods[action](self,directory)
            if action=='validate':
                state = read(directory/'state.json');samples = pairing.removal.dense_path(request['path'])
                if np.abs(samples[0]-pairing.removal.arm_q(state)).max()>=.005:
                    return {'success':False,'reason':'measured_path_start_differs'}
                context = self.bind(directory/'全场景轨迹复核',state,include_target=True)
                tensor = torch.as_tensor(samples,device='cuda',dtype=torch.float32)
                valid = all(bool(context.checker.validate(q[:,None]).all()) for q in tensor.split(64))
                return {'success':valid,'points':len(samples),'reason':None if valid else 'full_scene_path_collision'}
            assert action is None
            return super().request(directory,**request)
    resident_planner.PlannerPool = Planner
    service.serve(SimpleNamespace(kind='curobo',root=args.output,socket=args.socket,warmup_trial=pairing.removal.WARMUP))


def observed(env,directory,address,request,archived=None):
    import run
    import arx_route_mpc as route
    root = Path(os.environ['ARX_CLARO_BATCH_ROOT'])
    directory.mkdir();bind_guard(directory,root)
    save(directory/'state.json',run.snapshot(env));save(directory/'observation_request.json',request)
    path = archived if archived is not None else pairing.call(address,directory,'观察规划',observe=True)['paths']['observe']
    save(directory/'observation_plan_result.json',{'success':True,'paths':{'observe':path}})
    pairing.call(address,directory,'观察轨迹复核',action='validate',path=path)
    with pairing.recorded(env,directory) as execution:route.follow_route(execution,'observe',request['goal_world'])
    state = run.snapshot(env,env.get_obs());save(directory/'observed_state.json',state)
    return state


def reuse_blocker(directory,root):
    entry = next(e for e in read(root/'registration.json')['schedule'] if e['case']==directory.parent.name)
    source = Path(entry['archived_blocker_grasp']);actual = read(directory/'state.json');old = read(source/'state.json')
    pairing.removal.require_view_match(directory,actual,old)
    live = directory/'live_capture';live.mkdir()
    shutil.move(directory/'camera',live/'camera');shutil.copy2(directory/'state.json',live/'state.json')
    shutil.copytree(source/'camera',directory/'camera')
    shutil.copytree(source/'network/raw_outputs',directory/'network/raw_outputs')
    shutil.copy2(source/'raw_grasps.npy',directory/'raw_grasps.npy')
    actual['camera_world_optical'] = old['camera_world_optical'];actual['intrinsics'] = old['intrinsics']
    save(directory/'state.json',actual)
    save(directory/'archived_input_receipt.json',{'original':str(source),'outcomes_used_for_inference':False,
        'current_robot_and_collision_state_retained':True,'camera_and_native_predictions_reused':True,
        'raw_sha256':sha(directory/'raw_grasps.npy')})
    return {'count':len(np.load(directory/'raw_grasps.npy')),'wall_seconds':0.,'cached':True}


def grasp(env,directory,sockets):
    import run
    import arx_pipeline.baseline as baseline
    import short_lift_protocol as lift
    import arx_direct_grasp_20261120 as direct
    import arx_direct_verifier_20261120 as verifier
    from native_pair_timing import native_candidates
    from postprocess_pipeline import render
    root = Path(os.environ['ARX_CLARO_BATCH_ROOT']);directory.mkdir();bind_guard(directory,root)
    result = {'id':directory.parent.name,'campaign_registration':str(root/'registration.json'),
              'success':False,'physical_attempt':False,'close_started':False,'planning_success':False}
    began = time.perf_counter()
    try:
        pairing.capture(env,directory)
        inference = reuse_blocker(directory,root) if directory.name=='01_抓挡' else pairing.call(sockets['zerograsp'],directory,'原生推理')
        counts = native_candidates(directory)['counts'];result.update(counts=counts,inference_seconds=inference['wall_seconds'])
        render(directory/'candidates',directory/'zg_outputs')
        if not counts['scored']:raise pairing.removal.StageStopped('no_native_candidates_fit_arx_opening')
        if directory.name=='01_抓挡':
            entry = next(e for e in read(root/'registration.json')['schedule'] if e['case']==directory.parent.name)
            old = read(Path(entry['archived_blocker_grasp'])/'graspcl/unrefined_candidate_pool.json')['candidates']
            new = read(directory/'graspcl/candidate_pool.json')['candidates']
            assert len(old)==len(new)
            assert all(np.array_equal(a['pose'],b['pose']) and a['score']==b['score'] for a,b in zip(old,new))
            save(directory/'same_native_candidates.json',{'passed':True,'count':len(new),'before_refinement':True})
        refinement = pairing.call(sockets['curobo'],directory,'插入复核',refine=True,orientation_mode='fixed_orientation')
        result['retained_candidates'] = refinement['retained']
        plan = pairing.call(sockets['curobo'],directory,'抓取规划',selector='graspcl')
        result.update(planning_seconds=plan['wall_seconds'],planning_success=True)
        selected = plan['selected'];lift.validate_selected(selected)
        lift.install_execution_policy(run,baseline)
        with pairing.recorded(env,directory) as execution:
            result['physical_attempt'] = True
            try:
                with direct.execution_hook(run,True):selected = run.finish_continuous_mpc_grasp(execution,selected)
                result['close_started'] = True
                pairing.segment(execution,'close',pairing.removal.until_still(selected['q_grasp'],3.),finger=0.,goal=selected['grasp_world'])
                result['bilateral_fingers'] = sorted(execution.bilateral)
                if len(execution.bilateral)!=2:raise pairing.removal.StageStopped('no_bilateral_object_contact')
                save(directory/'closed_state.json',run.snapshot(env))
                path = pairing.call(sockets['curobo'],directory,'实测短抬',lift=True)['paths']['lift']
                pairing.segment(execution,'lift',path,finger=0.,goal=selected['lift_world'])
                pairing.segment(execution,'hold',pairing.removal.until_still(path['q'][-1],lift.HOLD_SECONDS),finger=0.,goal=selected['lift_world'])
                result.update(minimum_hold_lift_m=min(execution.heights)-execution.initial_z,hold_seconds=len(execution.heights)/120)
                result['success'] = bool(result['minimum_hold_lift_m']>=lift.MIN_HOLD_LIFT_M and result['hold_seconds']>=lift.HOLD_SECONDS)
                result['reason'] = 'success' if result['success'] else 'insufficient_hold_height'
            finally:result.update(stages=execution.stages,forbidden_contact_events=len(execution.forbidden))
    except pairing.removal.StageStopped as error:result['reason'] = str(error)
    except RuntimeError as error:
        text = str(error)
        if text=='forbidden_robot_contact' or text.endswith('_not_converged') or text.startswith('continuous_mpc_'):
            result['reason'] = text
        else:raise
    finally:
        result['wall_seconds'] = time.perf_counter()-began;save(directory/'result.json',result)
    verifier.verify_trial(directory)
    return result


def report(root,phase=None):
    reg = read(root/'registration.json');rows=[]
    for entry in reg['schedule']:
        p = root/'trials'/entry['case']/'result.json'
        if p.exists():rows.append(read(p))
    done = [r for r in rows if r['status']=='completed']
    result = {'phase':phase or 'running','registered':12,'completed':len(done),
        'blocker_planned':sum(r.get('blocker_grasp',{}).get('planning_success',False) for r in done),
        'blocker_lifted':sum(r.get('blocker_grasp',{}).get('success',False) for r in done),
        'placed':sum(r.get('placed',False) for r in done),'removed':sum(r.get('removed',False) for r in done),
        'reobserved':sum(r.get('reobserved',False) for r in done),'rescued':sum(r.get('success',False) for r in done),
        'infrastructure_faults':sum(r['status']=='infrastructure_fault' for r in rows),
        'rows':rows,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save(root/'summary.json',result)
    save(root/'progress.json',{k:v for k,v in result.items() if k!='rows'})
    return result


def simulation(args):
    import run
    import sapien
    import arx_mpc_guarded_bench as guarded
    from short_lift_protocol import install_execution_policy
    import arx_pipeline.baseline as baseline
    import arx_direct_verifier_20261120 as verifier
    guarded.install_variant();install_execution_policy(run,baseline);verifier.install(baseline)
    root = args.output.resolve();state_dir = root/'runtime_01/simulation';state_dir.mkdir()
    os.environ['ARX_CLARO_BATCH_ROOT'] = str(root)
    original_make = run.make;keepers=[]
    def make(*a,**kw):
        env = original_make(*a,**kw)
        if not keepers:keepers.append(sapien.render.SapienRenderer(env._render_device))
        return env
    run.make = make
    # Load renderer, articulation and camera dependencies on a training object.
    warm = make('014_lemon',perception_camera='right_wrist')
    try:warm.reset(seed=0)
    finally:warm.close()
    pairing.observe = observed;pairing.grasp = grasp
    original_call = pairing.call
    def call(address,directory,name,**kw):
        if not (directory/'source_guard.json').exists():bind_guard(directory,root)
        return original_call(address,directory,name,**kw)
    pairing.call = call
    ready = {'status':'ready','kind':'maniskill_simulation','pid':os.getpid(),
        'runtime_import_count':1,'renderer_context_owner_count':1,'training_only_preload':True,
        'test_claims_during_startup':0,'requests_completed':0}
    args.socket.parent.mkdir(parents=True,exist_ok=True)
    assert not args.socket.exists()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(args.socket));server.listen(4);save(state_dir/'ready.json',ready)
        while True:
            with server.accept()[0] as connection, connection.makefile('rwb') as stream:
                message = json.loads(stream.readline(1024*1024));command = message['command']
                if command=='health':response=ready
                elif command=='shutdown':
                    stream.write(json.dumps({'status':'stopped','pid':os.getpid()}).encode()+b'\n');stream.flush();break
                else:
                    assert command=='execute';os.kill(args.owner_pid,0)
                    entry = message['entry'];directory = root/'trials'/entry['case']
                    result = {**entry,'status':'completed','phase':'进场观测','placed':False,
                              'removed':False,'reobserved':False,'success':False}
                    began=time.perf_counter()
                    try:
                        with (directory/'simulator.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
                            pairing.run_case(directory,{k:Path(v) for k,v in message['sockets'].items()},entry,result)
                        response = {'status':'ok','pid':os.getpid(),'result':result}
                    except BaseException as error:
                        result.update(status='infrastructure_fault',reason=str(error),error=traceback.format_exc())
                        response={'status':'error','pid':os.getpid(),'error':result['error']}
                    finally:
                        result['simulator_request_seconds']=time.perf_counter()-began
                        save(directory/'result.json',result);save(directory/'配对结果.json',result)
                    ready['requests_completed']+=1;save(state_dir/'ready.json',ready)
                stream.write(json.dumps(response).encode()+b'\n');stream.flush()
    args.socket.unlink();save(state_dir/'stopped.json',{'pid':os.getpid(),'requests_completed':ready['requests_completed']})


def training_check(root,address):
    source = PILOT/'B_fixed_pregrasp/startup_validation/direct/B0069_S3_sparse_014_lemon'
    directory = root/'startup_validation/B0069_S3_sparse_014_lemon';directory.mkdir(parents=True)
    for name in ['state.json','raw_grasps.npy']:shutil.copy2(source/name,directory/name)
    save(directory/'graspcl/candidate_pool.json',read(source/'graspcl/candidate_pool.json'))
    bind_guard(directory,root)
    response = rpc(address,{'directory':str(directory)},timeout=900)
    save(directory/'response.json',response)
    assert response['status']=='ok' and response['result']['success'],response
    assert set(response['result']['paths'])=={'approach'}
    assert read(directory/'continuous_grasp_plan.json')['passed']
    save(root/'training_planning_verification.json',{'passed':True,'physical_executions':0,
                                                   'test_cases_used':False,'models_reloaded':0})


@contextmanager
def services(root):
    from arx_service_readiness import wait_ready
    from benchmark_lifecycle import mark_ready
    resident = root/'resident';resident.mkdir()
    children={};streams=[];sockets={};healthy=False
    epoch=root/'runtime_01';begin=time.perf_counter()
    try:
        for kind in ['zerograsp','curobo','simulation']:
            key=hashlib.sha256((str(root)+kind).encode()).hexdigest()[:16]
            sockets[kind]=CODE/'tmp'/('claro20_'+key+'.sock')
            env=dict(os.environ,MS_ASSET_DIR=str(CODE/'assets'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                CUDA_VISIBLE_DEVICES=NEURAL_GPU if kind=='zerograsp' else PLANNER_GPU,
                ARX_NEURAL_GPU=NEURAL_GPU,ARX_PLANNER_GPU=PLANNER_GPU,ARX_SIM_GPU=PLANNER_GPU,
                GRASP_TRAJOPT_POLICY='interactive',PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH',None)
            script_env=CODE.parent/'arx_core/scripts/curobo_arx/environment.sh'
            if kind=='simulation':
                command=['bash',str(CODE/'run.sh'),str(SELF),'simulator','--output',str(root),
                         '--socket',str(sockets[kind]),'--owner-pid',str(os.getpid())]
            else:
                python=canonical_path(CODE.parent/('tmp/arx-fullchain/env/bin/python' if kind=='zerograsp' else 'curobo/.venv/bin/python'))
                command=['bash','-c','source "$1"; shift; exec "$@"','claro20',str(script_env),str(python),
                         str(SELF),'worker','--output',str(resident),'--kind',kind,'--socket',str(sockets[kind])]
            log=(resident/(kind+'.log')).open('x');streams.append(log)
            children[kind]=subprocess.Popen(command,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
        def wait(kind):
            path = root/'runtime_01/simulation/ready.json' if kind=='simulation' else resident/(kind+'_ready.json')
            return wait_ready(sockets[kind],path,children[kind],
                              'maniskill_simulation' if kind=='simulation' else kind,timeout=1800)
        with ThreadPoolExecutor(max_workers=3) as pool:
            ready={kind:future.result() for kind,future in {k:pool.submit(wait,k) for k in children}.items()}
        config={'owner_pid':os.getpid(),'sockets':{k:str(sockets[k]) for k in REQUIRED},'required_components':REQUIRED}
        save(resident/'backend_config.json',config)
        mark_ready(resident,REQUIRED,config,{k:children[k] for k in REQUIRED})
        assert read(resident/'planner_warmup/mpc_ready.json')['passed']
        training_check(root,sockets['curobo'])
        save(epoch/'all_ready.json',{'passed':True,'startup_seconds':time.perf_counter()-begin,
            'components':ready,'mpc':read(resident/'planner_warmup/mpc_ready.json'),'claims_before_ready':0})
        healthy=True;yield sockets
    finally:
        stopped={}
        for kind,process in children.items():
            forced=False
            if process.poll() is None:
                try:rpc(sockets[kind],{'command':'shutdown'},timeout=15);process.wait(timeout=30)
                except Exception:
                    forced=True;process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            stopped[kind]={'pid':process.pid,'exited':process.poll() is not None,'returncode':process.returncode,'forced':forced}
        for stream in streams:stream.close()
        save(epoch/'services_stopped.json',{'services':stopped,'all_owned_services_exited':all(x['exited'] for x in stopped.values()),
                                           'reached_ready':healthy})


def run(root):
    from benchmark_lifecycle import require_ready
    reg=read(root/'registration.json');guard(reg)
    assert not list((root/'trials').glob('*/claim.json'))
    started=time.perf_counter();success=False
    try:
        report(root,'preloading')
        with services(root) as sockets:
            for entry in reg['schedule']:
                guard(reg);require_ready(root/'resident',REQUIRED)
                directory=root/'trials'/entry['case'];directory.mkdir()
                save(directory/'claim.json',{'state':'claimed','id':entry['case'],
                    'claimed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'replayed':False})
                progress=report(root);progress['current']=entry['case'];progress.pop('rows',None);save(root/'progress.json',progress)
                print('START',entry['case'],entry['original_reason'],flush=True)
                response=rpc(sockets['simulation'],{'command':'execute','entry':entry,
                              'sockets':{k:str(sockets[k]) for k in REQUIRED}},timeout=1800)
                save(directory/'simulation_receipt.json',response)
                if response['status']!='ok':raise RuntimeError(response.get('error',str(response)))
                result=read(directory/'result.json')
                import arx_direct_verifier_20261120 as verifier
                for sub in ['01_抓挡','04_重抓']:
                    trial=directory/sub
                    if (trial/'result.json').exists():verifier.verify_trial(trial)
                claim=read(directory/'claim.json');claim.update(state='complete',result_sha256=sha(directory/'result.json'))
                save(directory/'claim.json',claim);report(root)
                print('RESULT',entry['case'],result['phase'],result.get('reason'),flush=True)
            guard(reg);summary=report(root,'verifying');assert summary['completed']==12
            assert len(list((root/'trials').glob('*/claim.json')))==12
            success=True
    except BaseException:
        save(root/'runtime_01/halt.json',{'error':traceback.format_exc(),'replayed':False});raise
    finally:
        stopped=read(root/'runtime_01/services_stopped.json')
        save(root/'suite_timing.json',{'wall_seconds':time.perf_counter()-started,
            'all_owned_services_exited':stopped['all_owned_services_exited']})
        report(root,'complete' if success else 'halted')
        if success:save(root/'final_verification.json',{'passed':True,'completed':12,
            'all_owned_services_exited':stopped['all_owned_services_exited'],'physical_replays':0,'test_feedback_updates':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','worker','simulator'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--socket',type=Path)
    parser.add_argument('--kind',choices=['zerograsp','curobo'])
    parser.add_argument('--owner-pid',type=int)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='run':run(args.output)
    elif args.stage=='simulator':simulation(args)
    elif args.kind=='curobo':planner(args)
    else:
        import resident_backend as service
        from postprocess_backend import NativePostNeural
        service.NeuralWorker=NativePostNeural
        service.serve(SimpleNamespace(kind='zerograsp',root=args.output,socket=args.socket,warmup_trial=pairing.removal.WARMUP))
