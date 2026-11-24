"""Matched table-rejected cohort, TCP43, fixed/free orientation physical tests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np
import graspcl_pipeline as common
from graspcl_execute import guard
import native_pair_bench as core
import native_pair_timing as timing
from benchmark_lifecycle import require_ready
from resident_backend import rpc

CODE = Path(__file__).resolve().parent
AUDIT = CODE / 'outputs/arx_ik_workspace_audit_20261117_01'
PRIOR = CODE / 'outputs/arx_native_post_pair_20261117_02'
INPUTS = CODE / 'benchmark_inputs/arx_native_post_20261117_01'
TCP = CODE.parent / 'curobo_adapter/research/20261117_07_action_tcp/action_tcp_43mm.json'
ARMS = [('fixed_zerograsp','zerograsp_200','fixed_orientation'),
        ('fixed_graspcl','graspcl_200','fixed_orientation'),
        ('free_zerograsp','zerograsp_200','free_orientation'),
        ('free_graspcl','graspcl_200','free_orientation')]


def prepare(out):
    manifest = common.read(INPUTS / 'manifest.json')
    guard(manifest)
    counter = common.read(AUDIT / 'closed_margin_fullworld_countercheck.json')
    cohorts = {};evidence = [AUDIT / 'closed_margin_fullworld_countercheck.json',TCP]
    for profile in core.PROFILES:
        chosen = {}
        for path in sorted((AUDIT / 'cases' / profile).glob('*/audit.json')):
            item = common.read(path)
            if item['all_found_ik_solutions_table_blocked']:
                chosen[item['id']] = 'all_found_pose_IK_table_blocked';evidence.append(path)
        for item in counter['results'][profile]['cases']:
            if item['margin_excludes_all_found_open_clear_branches']:
                chosen[item['id']] = 'nominal_closed_table_margin';evidence.append(AUDIT / 'cases' / profile / item['id'] / 'audit.json')
        expected = 72 if profile=='zerograsp_200' else 55
        assert len(chosen)==expected,(profile,len(chosen))
        cohorts[profile] = chosen
        for slot in chosen:
            old = PRIOR / profile / 'trials' / slot
            assert common.read(old / 'independent_evaluation.json')['passed']
            result = common.read(old / 'result.json')
            assert not result['success'] and not result['physical_attempt']
    out.mkdir(parents=True,exist_ok=False)
    (out / 'source').mkdir()
    sources = {**manifest['source_hashes'],**manifest['input_sha256']}
    for path in [INPUTS / 'manifest.json',*evidence]:sources[str(path)] = common.sha(path)
    for name in ['table_truncation_bench.py','table_truncation_model.py','table_truncation_services.py',
        'native_pair_bench.py','native_pair_timing.py','postprocess_bench.py','postprocess_pipeline.py',
        'postprocess_planner.py','native_resident_services.py','checked_native_services.py',
        'resident_backend.py','resident_planner.py','resident_bench.py','fast_resident_planner.py','benchmark_lifecycle.py']:
        path = CODE / name;sources[str(path)] = common.sha(path);shutil.copy2(path,out / 'source' / name)
    base = common.read(PRIOR / 'registration.json')
    reg = {k:deepcopy(base[k]) for k in ['bank','warmup_trial','required_components','candidate_policy',
        'success','camera','table_height_m','recording_camera','layouts','source_feedback_ids_in_memory_order']}
    reg.update(created_at=core.now(),source_hashes=sources,inputs=str(INPUTS),
        common_reset_source=str(INPUTS / 'common_initial_conditions.json'),backend_lock=str(out / 'backend.lock'),
        gpu_backend_free_mib_floor=2000,scene_workers=2,backend_concurrency=1,
        total_slots=254,cohorts=cohorts,round_order=[a[0] for a in ARMS],
        action_tcp_profile=str(TCP),action_tcp_eef_z_m=.043,
        candidate_protocol='arx_tcp43_native_post_observed_insertion_truncation_v1',
        scope='Previously table-blocked subset only. No inference about success rate of all 200 scenes. Separate fresh attempts; historical results retained.',
        truncation='Fixed orientation: 2mm axial retraction grid, at most original native/recall insertion depth, first ARX-gripper/table-clear pose with observed target in finger closing sweep. Full original arm/scene/trajectory/physical gates follow.',
        free_orientation='Same cohort and TCP43/truncation. Additional rotations from 32-seed position-only IK at the actual virtual TCP43 goal frame at each source contact position; convert via TCP43 and re-solve original full-pose/collision IK. Observed local width is recomputed. This is finite orientation proposal search, not exhaustive orientation-free reachability. All original collision/trajectory/physical gates remain.',
        demonstration_conversion='Frozen TCP40 demo contact centers shifted +3mm along their own EEF Z to represent the same saved EEF at TCP43; bank is not trained or updated.',
        preserved='Robot/scene/reset/camera/physics and cuRobo collision geometry, table and nominal-closed 8mm margin, limits, paths, optimizer, bilateral-contact/lift/hold success gates.',
        old_results_carried=0,online_updates_during_evaluation=False,source_cohort=str(PRIOR),
        user_authorized_changes=['insertion truncation','separate free-orientation physical ablation','43mm action TCP'],
        attempt_budget_by_arm={arm:len(cohorts[profile]) for arm,profile,_ in ARMS})
    common.save(out / 'registration.json',reg)
    for name,profile,mode in ARMS:
        arm = out / name;arm.mkdir()
        for folder in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:(arm / folder).mkdir()
        shutil.copy2(INPUTS / 'common_initial_conditions.json',arm / 'reference/common_initial_conditions.json')
        shutil.copy2(INPUTS / 'storage_policy.json',arm / 'storage_policy.json')
        (arm / 'resident').symlink_to(out / 'resident',target_is_directory=True)
        schedule = [e for e in manifest['schedule'] if e['id'] in cohorts[profile]]
        armreg = deepcopy(reg)
        armreg.update(suite=str(out),experiment_arm=name,pipeline_profile=profile,orientation_mode=mode,
            schedule=schedule,attempt_budget=len(schedule),
            chain='wrist RGB-D -> native ZeroGrasp post'+(' -> frozen GraspCL recall' if profile=='graspcl_200' else '')+
                ' -> TCP43 '+mode+' insertion truncation'+(' -> GraspCL rescore' if profile=='graspcl_200' else '')+' -> unchanged cuRobo gates -> ARX execution')
        armreg['source_hashes'][str(arm / 'reference/common_initial_conditions.json')] = common.sha(arm / 'reference/common_initial_conditions.json')
        common.save(arm / 'registration.json',armreg)
        common.save(arm / 'progress.json',{'phase':'prepared','completed':0,'successes':0,'total':len(schedule)})
    update_suite(out,'prepared')
    print('PREPARED',out,'254 fresh slots, 4 arms',flush=True)


class Backends(core.Backends):
    def start(self):
        original = core.subprocess.Popen
        def launch(command,*args,**kwargs):
            command = [str(CODE / 'table_truncation_services.py') if str(v)==str(CODE / 'native_resident_services.py') else v for v in command]
            return original(command,*args,**kwargs)
        with patch.object(core.subprocess,'Popen',launch):super().start()


def checked_response(response):
    if response['status']!='ok':raise RuntimeError('Resident request failed: '+str(response))
    return response


def refine_client(directory,reg):
    config = common.read(Path(reg['suite']) / 'resident/backend_config.json')
    started = time.perf_counter()
    # Refinement owns the same GPU serialization gate as inference and planning.
    with core.batch.locked(reg['backend_lock']):
        response = checked_response(rpc(Path(config['sockets']['curobo']),{
            'directory':str(directory),'refine':True,'orientation_mode':reg['orientation_mode']},timeout=900))
    if reg['pipeline_profile']=='graspcl_200':
        checked_response(rpc(Path(config['sockets']['graspcl']),{'directory':str(directory),
            'policy':{**reg['candidate_policy'],'rescore_only':True}},timeout=900))
    else:
        from table_truncation_services import write_display
        write_display(directory,common.read(directory / 'graspcl/candidate_pool.json'))
    common.save(directory / 'refinement_client_timing.json',{'wall_seconds':time.perf_counter()-started,
        'mode':reg['orientation_mode'],'includes_serialization_wait_and_optional_rescore':True})
    return response


def execute(directory,entry_path):
    entry = common.read(entry_path);reg = common.read(entry['campaign_registration'])
    original_native = timing.native_candidates
    original_rpc = timing.real_rpc
    def native(path):
        result = original_native(path)
        refine_client(path,reg)
        return result
    def routed(path,message,timeout=1800):
        response = original_rpc(path,message,timeout=timeout)
        if Path(path).stem=='graspcl' and response['status']=='ok':
            refine_client(Path(message['directory']),reg)
        return response
    with patch.object(timing,'native_candidates',native),patch.object(timing,'real_rpc',routed):
        timing.execute(directory,entry_path)


def summarize(arm,phase='running',workers=None):
    reg = common.read(arm / 'registration.json');rows = []
    for path in sorted((arm / 'claims').glob('*.json')):
        claim = common.read(path)
        if claim['state']!='complete':continue
        trial = arm / 'trials' / path.stem
        assert common.read(trial / 'independent_evaluation.json')['passed']
        rows.append(common.read(trial / 'result.json'))
    total = reg['attempt_budget'];n = len(rows)
    plans = sum(r['planning_success'] for r in rows);executed = sum(r['physical_attempt'] for r in rows)
    successes = sum(r['success'] for r in rows)
    result = {'registered':total,'completed':n,'planning_passes':plans,'physical_attempts':executed,'successes':successes,
        'planning_pass_rate_of_completed':plans/n if n else None,
        'execution_entry_rate_of_completed':executed/n if n else None,
        'grasp_success_rate_of_completed':successes/n if n else None,
        'grasp_success_rate_of_execution_entries':successes/executed if executed else None,
        'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success'])),
        'by_layout':{layout:{'completed':sum(r['layout']==layout for r in rows),
            'execution_entries':sum(r['physical_attempt'] for r in rows if r['layout']==layout),
            'successes':sum(r['success'] for r in rows if r['layout']==layout)} for layout in reg['layouts']},
        'results':rows,'success_scope':'Table-blocked subset; not overall 200-scene success rate.'}
    common.save(arm / 'summary.json',result)
    common.save(arm / 'progress.json',{'phase':phase,'completed':n,'successes':successes,'total':total,
        'planning_passes':plans,'physical_attempts':executed,'workers':workers or {},'updated_at':core.now()})
    return result


def update_suite(out,phase='running',active=None):
    progress = {name:common.read(out / name / 'progress.json') for name,_,_ in ARMS}
    common.save(out / 'progress.json',{'phase':phase,'active_round':active,'registered':254,
        'completed':sum(p['completed'] for p in progress.values()),'rounds':progress,'updated_at':core.now()})


class Campaign(core.Campaign):
    def next_entry(self,worker,gpu):
        entry,claim = super().next_entry(worker,gpu)
        if entry:
            entry.update(orientation_mode=self.reg['orientation_mode'],candidate_protocol=self.reg['candidate_protocol'],
                action_tcp_profile=self.reg['action_tcp_profile'],experiment_arm=self.reg['experiment_arm'])
            receipt = self.out / 'entries' / (entry['id']+'.json')
            common.save(receipt,entry)
            binding = receipt.with_name(entry['id']+'_binding.json');bound = common.read(binding)
            bound['entry_sha256'] = common.sha(receipt);common.save(binding,bound)
        return entry,claim

    def progress(self,worker,**state):
        with self.mutex:
            self.states[worker] = {**self.states.get(worker,{}),**state,'updated_at':core.now()}
            common.save(self.out / 'workers' / (worker+'.json'),self.states[worker])
            summarize(self.out,'draining_after_error' if self.failed.is_set() else 'running',self.states)
            update_suite(self.suite,active=self.reg['experiment_arm'])

    def worker(self,index,gpu):
        worker = 'scene_'+str(index)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        _,baseline = core.batch.original_modules()
        try:
            while not self.failed.is_set() and not (self.out / 'STOP').exists():
                if not self.resources(worker,gpu):break
                guard(self.reg)
                entry,claim = self.next_entry(worker,gpu)
                if entry is None:break
                slot = entry['id'];trial = self.out / 'trials' / slot
                assert not entry['admission_failed'],'Selected table cohort must have original admission'
                self.progress(worker,phase='executing',current=slot,gpu=gpu)
                command = ['bash',str(CODE / 'run.sh'),str(Path(__file__).resolve()),'execute',
                    '--output',str(trial),'--entry',str(self.out / 'entries' / (slot+'.json'))]
                with (self.out / 'entries' / (slot+'.log')).open('x') as stream:
                    child = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    self.progress(worker,current=slot,child_pid=child.pid,phase='executing',gpu=gpu)
                    child.wait()
                if child.returncode:raise RuntimeError('Execution failed; never replay claimed slot: '+slot)
                baseline.verify_trial(trial)
                result = common.read(trial / 'result.json');guard(self.reg)
                claim.update(state='complete',finished_at=core.now(),result_sha256=common.sha(trial / 'result.json'))
                common.save(self.out / 'claims' / (slot+'.json'),claim)
                self.progress(worker,phase='between_scenes',current=None,child_pid=None)
                print('RESULT',self.reg['experiment_arm'],slot,result['success'],result['reason'],flush=True)
        except BaseException:
            self.failed.set();error = {'worker':worker,'error':traceback.format_exc(),'time':core.now()}
            self.errors.append(error);common.save(self.out / 'workers' / (worker+'_halt.json'),error)
            self.progress(worker,phase='halted',error=error['error'])

    def start(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.worker,i,gpu) for i,gpu in enumerate(core.batch.SCENES)]
            for future in futures:future.result()
        result = summarize(self.out,'halted' if self.failed.is_set() else 'complete',self.states)
        if self.failed.is_set():raise RuntimeError('Campaign stopped without replay: '+str(self.errors))
        assert result['completed']==self.reg['attempt_budget']


def validate(out,backends,reg):
    for mode in ['fixed_orientation','free_orientation']:
        directory = out / 'startup_validation' / mode / Path(reg['warmup_trial']).name
        source = Path(reg['warmup_trial']);directory.mkdir(parents=True)
        shutil.copytree(source / 'camera',directory / 'camera')
        shutil.copytree(source / 'network/raw_outputs',directory / 'network/raw_outputs')
        for filename in ['state.json','raw_grasps.npy']:shutil.copy2(source / filename,directory / filename)
        timing.native_candidates(directory)
        local = {**reg,'suite':str(out),'orientation_mode':mode,'pipeline_profile':'zerograsp_200'}
        checked_response(refine_client(directory,local))
        pool = common.read(directory / 'graspcl/candidate_pool.json')
        for c in pool['candidates']:
            m = c['metadata'];assert 0<=m['insertion_retraction_m']<=m['original_source_insertion_m']+1e-8
            assert m['observed_target_points_in_closing_sweep']>0
            assert m['gripper_table_clearance_after_m']>=0
            if mode=='fixed_orientation':assert m['orientation_change_rad']<1e-8
        checked_response(rpc(Path(backends.config['sockets']['graspcl']),{'directory':str(directory),
            'policy':{**reg['candidate_policy'],'rescore_only':True}},timeout=900))
        response = checked_response(rpc(Path(backends.config['sockets']['curobo']),{'directory':str(directory),'selector':'graspcl'},timeout=900))
        common.save(directory / 'startup_response.json',response)
    require_ready(out / 'resident',reg['required_components'])
    common.save(out / 'startup_validation.json',{'passed':True,'physical_attempts':0,
        'all_components_preloaded':True,'modes_validated':['fixed_orientation','free_orientation'],
        'action_tcp_profile':str(TCP),'T_eef_action':common.read(TCP)['T_eef_action'],
        'collision_models_and_table_margin_unchanged':True})


def report(out):
    lines = ['# 桌面拦截场景截断测试（TCP 43 mm）','',
        '| 测试 | 完成/总数 | 规划通过 | 进入执行 | 抓取成功 | 成功/场景 | 成功/进入执行 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels = {'fixed_zerograsp':'固定朝向 · ZeroGrasp','fixed_graspcl':'固定朝向 · GraspCL',
        'free_zerograsp':'放开朝向 · ZeroGrasp','free_graspcl':'放开朝向 · GraspCL'}
    for name,_,_ in ARMS:
        arm = out / name
        if not (arm / 'summary.json').exists():continue
        s = common.read(arm / 'summary.json');n = s['completed'];e = s['physical_attempts'];success = s['successes']
        lines.append(f"| {labels[name]} | {n}/{s['registered']} | {s['planning_passes']} | {e} | {success} | {success/n:.1%} | {success/e:.1%} |" if n and e else
            f"| {labels[name]} | {n}/{s['registered']} | {s['planning_passes']} | {e} | {success} | {success/n:.1%} | — |" if n else f"| {labels[name]} | 0/{s['registered']} | 0 | 0 | 0 | — | — |")
    lines += ['', '仅统计此前因桌面拦截的 ZeroGrasp 72 例、GraspCL 55 例，不代表全部 200 例的成功率。',
        '两种朝向设置均使用 43 mm TCP 和插入截断；这是新 TCP 下的独立试验，不能将与历史 40 mm 结果的变化全部归因于截断。',
        '规划通过为初次完整抓取方案通过；进入执行为开始预抓取运动，尚不保证完成闭爪；抓取成功需双指接触、无禁止接触并抬升超过 6 cm 连续保持 3 秒。',
        '原生后处理和冻结示教库保持；示教位姿按同一历史 EEF 换算到新 TCP。截断和新增朝向后的 GraspCL 位姿重新评分。',
        '位姿图及录像见每组 trials/<场景>/zg_outputs/ 和该场景录像文件。',
        '各模型/编码器/规划器及自由朝向提议求解器在领取第一例前加载、预热，整个批次常驻，结束统一销毁。','']
    (out / '测试结果.md').write_text('\n'.join(lines))


def run(out):
    reg = common.read(out / 'registration.json');guard(reg)
    with (out / 'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out / 'suite_timing.json').exists():raise RuntimeError('Already started; no replay')
        backends = Backends(out);started = time.perf_counter()
        clock = {'started_at':core.now(),'phase':'starting_components','rounds':{}}
        common.save(out / 'suite_timing.json',clock)
        try:
            update_suite(out,'starting_components');backends.start();validate(out,backends,reg)
            clock['shared_startup_seconds'] = time.perf_counter()-started
            for name,_,_ in ARMS:
                arm = out / name;round_started = time.perf_counter()
                phase = {'started_at':core.now(),'phase':'running','started_monotonic':round_started,
                    'shared_suite_startup_seconds':clock['shared_startup_seconds']}
                common.save(arm / 'round_timing.json',phase);update_suite(out,active=name)
                try:Campaign(arm).start()
                finally:
                    phase.update(ended_at=core.now(),round_wall_seconds=time.perf_counter()-round_started,
                        phase=common.read(arm / 'progress.json')['phase'])
                    common.save(arm / 'round_timing.json',phase);clock['rounds'][name] = phase
                    common.save(out / 'suite_timing.json',clock);report(out)
                require_ready(out / 'resident',reg['required_components'])
                health = {k:rpc(Path(backends.config['sockets'][k]),{'command':'health'}) for k in reg['required_components']}
                common.save(arm / 'resident_health_after_round.json',health)
            clock['phase'] = 'complete';update_suite(out,'complete')
            checks = []
            for name,_,_ in ARMS:
                arm = out / name
                for path in (arm / 'claims').glob('*.json'):
                    trial = arm / 'trials' / path.stem
                    checks.append(common.read(path)['result_sha256']==common.sha(trial / 'result.json') and common.read(trial / 'independent_evaluation.json')['passed'])
            assert len(checks)==254 and all(checks)
            common.save(out / 'final_verification.json',{'passed':True,'verified_trials':len(checks),'physical_attempts_not_replayed':True})
        except BaseException:
            clock['phase'] = 'halted';common.save(out / 'halt.json',{'error':traceback.format_exc(),'time':core.now(),'replay_allowed':False})
            update_suite(out,'halted');raise
        finally:
            shutdown = time.perf_counter();backends.stop()
            clock.update(ended_at=core.now(),shutdown_seconds=time.perf_counter()-shutdown,
                total_suite_wall_seconds=time.perf_counter()-started,
                all_owned_services_exited=all(child.poll() is not None for child in backends.children.values()))
            common.save(out / 'suite_timing.json',clock);report(out)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','execute','report','validate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--entry',type=Path)
    args = parser.parse_args();out = args.output.resolve()
    if args.stage=='prepare':prepare(out)
    elif args.stage=='run':run(out)
    elif args.stage=='report':report(out)
    elif args.stage=='validate':
        backend = Backends(out)
        try:
            backend.start();validate(out,backend,common.read(out / 'registration.json'))
        finally:backend.stop()
    else:execute(out,args.entry.resolve())
