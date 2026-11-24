"""Fresh native-post benchmark: ZeroGrasp 200, then ZeroGrasp+GraspCL 200."""
from arx_project.layout import canonical_path
from copy import deepcopy
from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import graspcl_batch as batch
import graspcl_pipeline as common
import postprocess_bench as post
import resident_bench as resident
from graspcl_execute import guard
from resident_backend import rpc
from benchmark_lifecycle import mark_ready, require_ready
from native_pair_timing import timing_summary, execute, native_candidates

CODE = Path(__file__).resolve().parent
PROFILES = ['zerograsp_200','graspcl_200']


def now():
    return datetime.now().astimezone().isoformat()


def prepare(out, inputs):
    manifest = common.read(inputs / 'manifest.json')
    guard({'source_hashes':{**manifest['source_hashes'],**manifest['input_sha256']}})
    validation = CODE / 'outputs/arx_native_post_validation_20261117_04/verification.json'
    tested = common.read(validation)
    if not tested['passed']:
        raise RuntimeError('Resident native-post validation is missing')
    for path,digest in tested['validated_source_sha256'].items():
        if common.sha(path)!=digest: raise RuntimeError('Validated service changed: '+path)
    adapter_validation=CODE / 'outputs/arx_native_pair_validation_20261117_02/verification.json'
    adapter_check=common.read(adapter_validation)
    if not adapter_check['passed'] or not adapter_check['timing_step_matches_original_command_and_recording_cadence']:
        raise RuntimeError('Native-only adapter/timing validation is incomplete')
    for path,digest in adapter_check['validated_source_sha256'].items():
        if common.sha(path)!=digest:raise RuntimeError('Validated timing/adapter source changed: '+path)
    schedule = manifest['schedule']
    assert len(schedule)==len({e['id'] for e in schedule})==200
    assert len({e['model_id'] for e in schedule})==40
    out.mkdir(parents=True,exist_ok=False)
    sources = {**manifest['source_hashes'],**manifest['input_sha256']}
    sources[str(inputs / 'manifest.json')] = common.sha(inputs / 'manifest.json')
    for name in ['graspcl_batch.py','graspcl_pipeline.py','graspcl_execute.py','graspcl_planner.py','grasp_visuals.py',
        'resident_backend.py','resident_planner.py','resident_bench.py','postprocess_bench.py','postprocess_backend.py',
        'postprocess_pipeline.py','postprocess_planner.py','native_resident_services.py','fast_resident_planner.py',
        'benchmark_lifecycle.py','native_pair_bench.py','native_pair_timing.py','native_pair_setup.py']:
        path = CODE / name
        sources[str(path)] = common.sha(path)
        (out / 'source').mkdir(exist_ok=True)
        shutil.copy2(path,out / 'source' / name)
    reg = {'created_at':now(),'profiles':PROFILES,'attempt_budget_per_profile':200,'total_slots':400,
        'schedule':schedule,'layouts':manifest['layouts'],'source_hashes':sources,'inputs':str(inputs),
        'bank':manifest['bank'],'warmup_trial':manifest['warmup_trial'],
        'required_components':['zerograsp','graspcl','curobo'],
        'candidate_policy':{'zero_input':'network/raw_outputs/*.grasp.npy','zero_top_k':None,
            'add_finger_symmetry':False,'custom_geometry_filters':[],
            'recall_max_matches_per_template':2,'patch_time_ms':2000},
        'success':manifest['success'],'camera':manifest['camera'],'table_height_m':manifest['table_height_m'],
        'recording_camera':manifest['recording_camera'],'source_feedback_ids_in_memory_order':manifest['source_feedback_ids_in_memory_order'],
        'gpu':{'scene_simulators':batch.SCENES,'neural_and_planner':batch.BACKEND,'graspcl':'CPU, four threads'},
        'scene_workers':2,'backend_concurrency':1,'backend_lock':str(out / 'backend.lock'),
        'gpu_backend_free_mib_floor':2000,'common_reset_source':str(inputs / 'common_initial_conditions.json'),
        'candidate_protocol':post.PROTOCOL,'old_results_carried':0,'online_updates_during_evaluation':False,
        'user_authorized_full_rerun':True,'adapter_and_timing_validation':str(adapter_validation),
        'startup_model_validation_requests':1,'startup_physical_attempts':0,
        'round_order':PROFILES,'component_lifecycle':'Preload and warm all services before the first slot; one set shared by the two sequential rounds; release on exit; no cold fallback/replay.',
        'observation_protocol':manifest['observation_protocol'],
        'timing_protocol':'Wall-clock each full 200-slot round with two parallel scene workers. Separately record shared startup/warmup, queue, RPC, simulation/contact trace, recording, visualization and residual setup. Record total 400-slot suite including shutdown.',
        'validation':str(validation)}
    common.save(out / 'registration.json',reg)
    common.save(out / 'progress.json',{'phase':'prepared','registered':400,'completed':0,'active_round':None})
    for profile in PROFILES:
        arm = out / profile
        arm.mkdir()
        for folder in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:
            (arm / folder).mkdir()
        shutil.copy2(inputs / 'common_initial_conditions.json',arm / 'reference/common_initial_conditions.json')
        shutil.copy2(inputs / 'storage_policy.json',arm / 'storage_policy.json')
        (arm / 'resident').symlink_to(out / 'resident',target_is_directory=True)
        armreg = deepcopy(reg)
        armreg.update(pipeline_profile=profile,attempt_budget=200,suite=str(out),
            chain='right wrist RGB-D -> ZeroGrasp native post -> '+('frozen GraspCL recall and reranking -> ' if profile=='graspcl_200' else 'network-score ordering -> ')+'frozen ARX cuRobo -> ARX execution',
            scope='Fresh independent 200-slot round; zero prior claims or outcomes reused. Two current rounds use the same fixed scene/viewpoint/reset inputs.',
            required_components=reg['required_components'])
        armreg['source_hashes'][str(arm / 'reference/common_initial_conditions.json')] = common.sha(arm / 'reference/common_initial_conditions.json')
        common.save(arm / 'registration.json',armreg)
        common.save(arm / 'progress.json',{'phase':'prepared','completed':0,'successes':0,'total':200})
    print('PREPARED TWO FRESH ROUNDS',out,flush=True)


class Backends(resident.Backends):
    def start(self):
        root = self.campaign / 'resident'
        root.mkdir(exist_ok=False)
        key = hashlib.sha256(str(self.campaign).encode()).hexdigest()[:12]
        sockets = CODE / 'tmp' / ('pair_'+key)
        sockets.mkdir(mode=0o700,exist_ok=False)
        self.config = {'gpu':batch.BACKEND,'owner_pid':os.getpid(),'candidate_protocol':post.PROTOCOL,
            'required_components':self.reg['required_components'],
            'sockets':{k:str(sockets / (k+'.sock')) for k in self.reg['required_components']}}
        common.save(root / 'backend_config.json',self.config)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=batch.BACKEND,ARX_NEURAL_GPU=batch.BACKEND,
            ARX_PLANNER_GPU=batch.BACKEND,ARX_SIM_GPU=batch.BACKEND,LILO_GPU=batch.BACKEND,
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GRASP_TRAJOPT_POLICY='interactive')
        env.pop('PYTHONPATH',None)
        for kind in self.reg['required_components']:
            python = canonical_path(batch.ROOT / {'zerograsp':'tmp/arx-fullchain/env/bin/python',
                'graspcl':'tmp/arx-graspcl-env/bin/python','curobo':'curobo/.venv/bin/python'}[kind])
            command = ['bash','-c','source "$1"; shift; exec "$@"','native-pair-service',
                str(batch.ROOT / 'arx_core/scripts/curobo_arx/environment.sh'),str(python),
                str(CODE / 'native_resident_services.py'),'--kind',kind,'--socket',self.config['sockets'][kind],
                '--root',str(root),'--warmup-trial',self.reg['warmup_trial']]
            child_env = dict(env)
            if kind=='graspcl':
                child_env['CUDA_VISIBLE_DEVICES']=''
                command.extend(['--bank',self.reg['bank'],'--policy-registration',str(self.campaign / 'registration.json')])
            stream = (root / (kind+'.log')).open('x')
            self.streams.append(stream)
            child = subprocess.Popen(command,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
            self.children[kind]=child
            deadline = time.monotonic()+240
            while not Path(self.config['sockets'][kind]).exists():
                if child.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Resident startup failed: '+kind)
                time.sleep(.2)
            health=rpc(Path(self.config['sockets'][kind]),{'command':'health'})
            if health['status']!='ready':raise RuntimeError('Resident is not ready: '+kind)
            print('RESIDENT READY',kind,child.pid,health['startup_seconds'],flush=True)
        mark_ready(root,self.reg['required_components'],self.config,self.children)


def suite_progress(out,phase='running',active=None):
    arms={}
    for profile in PROFILES:
        arm=out / profile
        state=common.read(arm / 'progress.json')
        arms[profile]={k:state.get(k) for k in ['phase','completed','successes','total']}
        if (arm / 'timing_summary.json').exists():
            arms[profile]['timing']=common.read(arm / 'timing_summary.json')
    common.save(out / 'progress.json',{'phase':phase,'active_round':active,'registered':400,
        'completed':sum(a['completed'] for a in arms.values()),'rounds':arms,'updated_at':now()})


def startup_validation(out,backends,reg):
    """One source-frame ARX planning check; never commands a simulator or robot."""
    import numpy as np
    source=Path(reg['warmup_trial'])
    directory=out / 'startup_validation' / source.name
    directory.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source / 'camera',directory / 'camera')
    shutil.copytree(source / 'network/raw_outputs',directory / 'network/raw_outputs')
    for name in ['state.json','raw_grasps.npy']:shutil.copy2(source / name,directory / name)
    native_candidates(directory)
    pool=common.read(directory / 'candidates/candidate_pool.json')
    rows,_=post.post.processed_rows(directory)
    used=[c['metadata']['postprocessed_row_index'] for c in pool['candidates']]
    assert sorted(used)==list(range(len(rows)))
    assert pool['graspcl_invoked'] is False and pool['custom_geometry_filters']==[]
    for c in pool['candidates']:
        row=rows[c['metadata']['postprocessed_row_index']]
        assert c['score']==row[0] and c['width']==row[1]
    response=rpc(Path(backends.config['sockets']['curobo']),{'directory':str(directory),'selector':'graspcl'},timeout=900)
    common.save(directory / 'curobo_response.json',response)
    if response['status']!='ok':raise RuntimeError('Native-only startup integration failed: '+str(response))
    bind=common.read(directory / 'plan/resident_rebind.json')
    assert bind['runtime_robot_configuration_rebuilds']==0
    require_ready(out / 'resident',reg['required_components'])
    common.save(out / 'startup_validation.json',{'passed':True,'physical_attempts':0,
        'model_requests':1,'native_candidates_used_once':len(rows),'native_network_score_order':True,
        'custom_geometry_filters':[],'no_graspcl_in_zero_arm':True,
        'planning_success':common.read(directory / 'plan_result.json')['success'],
        'runtime_robot_rebuilds':0,'all_components_preloaded':True})


class Campaign(resident.ResidentCampaign):
    def __init__(self,out):
        super().__init__(out)
        self.inputs=Path(self.reg['inputs'])
        self.suite=Path(self.reg['suite'])

    def next_entry(self,worker,gpu):
        require_ready(self.suite / 'resident',self.reg['required_components'])
        with self.mutex:
            for scheduled in self.reg['schedule']:
                slot=scheduled['id']
                if slot in self.claimed:continue
                admission=self.inputs / 'admission' / slot / 'admission.json'
                bound=common.read(admission)
                assert all(bound['entry'][k]==scheduled[k] for k in ['id','model_id','layout','seed'])
                entry=deepcopy(bound['entry'])
                entry.update(campaign_registration=str(self.out / 'registration.json'),
                    historical_trial=str(self.inputs / 'resets' / slot),source_admission=str(admission),
                    admission_failed=not bound['passed'],parallel_worker=worker,
                    pipeline_profile=self.reg['pipeline_profile'],candidate_protocol=post.PROTOCOL)
                evidence=[admission]
                reset=self.inputs / 'resets' / slot / 'initial_conditions.json'
                if reset.exists():evidence.append(reset)
                receipt=self.out / 'entries' / (slot+'.json')
                common.save(receipt,entry)
                common.save(self.out / 'entries' / (slot+'_binding.json'),{
                    'bound_at':now(),'source_evidence_sha256':{str(p):common.sha(p) for p in evidence},
                    'entry_sha256':common.sha(receipt),'scene_and_observation_copied_exactly':True,
                    'old_outcomes_reused':False})
                claim={'slot':slot,'worker':worker,'gpu':gpu,'state':'claimed','claimed_at':now()}
                with (self.out / 'claims' / (slot+'.json')).open('x') as stream:json.dump(claim,stream,indent=2)
                self.claimed.add(slot)
                return entry,claim
        return None,None

    def resources(self,worker,gpu):
        if (self.suite / 'STOP').exists():
            (self.out / 'STOP').write_text('Suite STOP requested\n')
            return False
        start=time.perf_counter()
        result=super().resources(worker,gpu)
        with (self.out / 'resource_wait' / (worker+'.jsonl')).open('a') as stream:
            stream.write(json.dumps({'checked_at':now(),'seconds':time.perf_counter()-start,'ready':result})+'\n')
        return result

    def progress(self,worker,**state):
        super().progress(worker,**state)
        with self.mutex:
            timing_summary(self.out)
            suite_progress(self.suite,active=self.reg['pipeline_profile'])


def run_suite(out):
    reg=common.read(out / 'registration.json')
    guard(reg)
    batch.__file__=str(Path(__file__).resolve())
    with (out / 'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out / 'STOP').exists():raise RuntimeError('Explicit STOP; do not resume')
        if any(list((out / p / 'claims').glob('*.json')) for p in PROFILES):
            raise RuntimeError('Suite already started; do not replay slots')
        backends=Backends(out)
        started=time.perf_counter()
        clock={'started_at':now(),'phase':'starting_components','shared_startup_seconds':None,'rounds':{}}
        common.save(out / 'suite_timing.json',clock)
        try:
            suite_progress(out,phase='starting_components')
            backends.start()
            startup_validation(out,backends,reg)
            clock['shared_startup_seconds']=time.perf_counter()-started
            clock['phase']='running'
            common.save(out / 'suite_timing.json',clock)
            for profile in PROFILES:
                if (out / 'STOP').exists():break
                arm=out / profile
                phase_clock={'started_at':now(),'started_monotonic':time.perf_counter(),'phase':'running',
                    'shared_suite_startup_seconds':clock['shared_startup_seconds']}
                common.save(arm / 'round_timing.json',phase_clock)
                try:
                    Campaign(arm).start()
                finally:
                    phase_clock.update(phase=common.read(arm / 'progress.json')['phase'],
                        ended_at=now(),round_wall_seconds=time.perf_counter()-phase_clock['started_monotonic'])
                    common.save(arm / 'round_timing.json',phase_clock)
                    timing_summary(arm)
                    clock['rounds'][profile]=phase_clock
                    common.save(out / 'suite_timing.json',clock)
                if common.read(arm / 'progress.json')['phase']!='complete':break
                health={k:rpc(Path(backends.config['sockets'][k]),{'command':'health'}) for k in reg['required_components']}
                common.save(arm / 'resident_health_after_round.json',health)
                if profile=='zerograsp_200' and health['graspcl']['requests_completed']!=0:
                    raise RuntimeError('ZeroGrasp-only arm unexpectedly invoked GraspCL')
            phase='complete' if all(common.read(out / p / 'progress.json')['phase']=='complete' for p in PROFILES) else 'stopped'
            suite_progress(out,phase=phase)
            clock['phase']=phase
        except BaseException:
            clock['phase']='halted'
            common.save(out / 'halt.json',{'time':now(),'error':traceback.format_exc(),'replay_allowed':False})
            suite_progress(out,phase='halted')
            raise
        finally:
            shutdown=time.perf_counter()
            backends.stop()
            clock.update(ended_at=now(),shutdown_seconds=time.perf_counter()-shutdown,
                total_suite_wall_seconds=time.perf_counter()-started,
                all_owned_services_exited=all(p.poll() is not None for p in backends.children.values()))
            common.save(out / 'suite_timing.json',clock)
            print('SUITE',clock['phase'],clock['total_suite_wall_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','execute'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve(),args.inputs.resolve())
    elif args.stage=='run':run_suite(args.output.resolve())
    else:execute(args.output.resolve(),args.entry.resolve())
