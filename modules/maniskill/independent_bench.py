"""Preregistered paired ARX evaluation on wholly separate local objects/scenes."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from collections import Counter
import argparse
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
from benchmark_lifecycle import require_ready
from resident_backend import rpc
import native_pair_bench as core
import table_truncation_bench as existing

CODE=Path(__file__).resolve().parent


def prepare(out,inputs,modes):
    data=common.read(inputs/'manifest.json')
    assert common.read(inputs/'independence_preflight.json')['passed']
    sources={**data['physics_source_hashes'],**data['input_sha256'],**data['admission_sha256'],**data['declared_reset_sha256']}
    for p in [inputs/'manifest.json',inputs/'independence_preflight.json']:sources[str(p)]=common.sha(p)
    for p,digest in data['asset_sha256'].items():
        if common.sha(p)!=digest:raise RuntimeError('Held-out asset changed before registration: '+p)
    arms=[]
    for mode in modes:
        for profile in ['zerograsp_200','graspcl_200']:
            prefix='fixed' if mode=='fixed_orientation' else 'free'
            arms.append((prefix+'_'+('zerograsp' if profile=='zerograsp_200' else 'graspcl'),profile,mode))
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    names=['independent_inputs.py','independent_services.py','independent_bench.py','table_truncation_bench.py',
        'table_truncation_model.py','table_truncation_services.py','native_pair_bench.py','native_pair_timing.py',
        'postprocess_bench.py','postprocess_pipeline.py','postprocess_planner.py','native_resident_services.py',
        'checked_native_services.py','resident_backend.py','resident_planner.py','resident_bench.py',
        'fast_resident_planner.py','benchmark_lifecycle.py']
    for name in names:
        path=CODE/name;sources[str(path)]=common.sha(path);shutil.copy2(path,out/'source'/name)
    tcp=existing.TCP;sources[str(tcp)]=common.sha(tcp);shutil.copy2(tcp,out/'source/action_tcp_43mm.json')
    reg={k:deepcopy(data[k]) for k in ['schedule','layouts','bank','source_feedback_ids_in_memory_order',
        'warmup_trial','success','camera','table_height_m','recording_camera','test_object_ids','excluded_old_object_ids',
        'independence_scope','observation_protocol']}
    reg.update(created_at=core.now(),source_hashes=sources,inputs=str(inputs),arms=arms,
        attempt_budget_by_arm={a:200 for a,_,_ in arms},total_slots=200*len(arms),
        required_components=['zerograsp','graspcl','curobo'],
        candidate_policy=common.read(CODE/'outputs/arx_native_post_pair_20261117_02/registration.json')['candidate_policy'],
        common_reset_source=str(inputs/'common_initial_conditions.json'),backend_lock=str(out/'backend.lock'),
        gpu_backend_free_mib_floor=2000,scene_workers=2,backend_concurrency=1,
        action_tcp_profile=str(tcp),action_tcp_eef_z_m=.043,
        candidate_protocol='arx_tcp43_native_post_observed_insertion_truncation_v1',
        train_test_disjoint=True,local_object_mesh_scene_seed_disjoint=True,online_updates_during_evaluation=False,
        source_outcomes_used_in_test=False,source_evaluation_results_carried=0,
        frozen_bank_before_test=True,input_generation_before_first_test=True,
        admission_failed=data['admission_failed'],admitted=data['admitted'],asset_sha256=data['asset_sha256'],
        bootstrap_scope='Only the old training/development lemon frame is used for service warmup. No held-out observation or result is used for learning or startup tuning.',
        pipeline_scope='Matched complete-chain comparison. Each arm executes a new reset and captures fresh wrist RGB-D; candidate pools are not held identical. GraspCL adds frozen positive recall and scores with 12 positives and 13 negatives from the separate old development set.',
        frozen_algorithm='Use the previously registered 43mm insertion truncation implementation unchanged. No changes to robot/reset/camera/physics/collision including 8mm closed-table margin/IK/path/optimizer/success gates.',
        outcome_policy='One physical test per allocated arm/scene. No replay or scene replacement. Admission failures remain allocated failures in both arms. Halt on infrastructure error; never update models or tune parameters from held-out results.')
    common.save(out/'registration.json',reg)
    for name,profile,mode in arms:
        arm=out/name;arm.mkdir()
        for folder in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:(arm/folder).mkdir()
        shutil.copy2(inputs/'common_initial_conditions.json',arm/'reference/common_initial_conditions.json')
        shutil.copy2(inputs/'storage_policy.json',arm/'storage_policy.json')
        (arm/'resident').symlink_to(out/'resident',target_is_directory=True)
        r=deepcopy(reg);r.update(suite=str(out),experiment_arm=name,pipeline_profile=profile,orientation_mode=mode,
            attempt_budget=200,chain='new held-out wrist RGB-D -> native ZeroGrasp'+(' + frozen development GraspCL recall' if profile=='graspcl_200' else '')+
                ' -> unchanged TCP43 '+mode+' truncation'+(' -> frozen GraspCL rescore' if profile=='graspcl_200' else '')+' -> frozen cuRobo -> original ARX physical evaluator')
        r['source_hashes'][str(arm/'reference/common_initial_conditions.json')]=common.sha(arm/'reference/common_initial_conditions.json')
        common.save(arm/'registration.json',r)
        common.save(arm/'progress.json',{'phase':'prepared','completed':0,'successes':0,'total':200})
    update_suite(out,'prepared');report(out)
    print('INDEPENDENT BENCHMARK REGISTERED',len(arms),'arms x 200',flush=True)


class Backends(core.Backends):
    def start(self):
        original=core.subprocess.Popen
        def launch(command,*args,**kwargs):
            command=[str(CODE/'independent_services.py') if str(v)==str(CODE/'native_resident_services.py') else v for v in command]
            return original(command,*args,**kwargs)
        with patch.object(core.subprocess,'Popen',launch):super().start()


def update_suite(out,phase='running',active=None):
    reg=common.read(out/'registration.json');states={a:common.read(out/a/'progress.json') for a,_,_ in reg['arms']}
    common.save(out/'progress.json',{'phase':phase,'active_round':active,'registered':reg['total_slots'],
        'completed':sum(v['completed'] for v in states.values()),'rounds':states,'updated_at':core.now()})


def placeholder(trial,reason):
    if (trial/'zg_outputs/grasp_top10.json').exists():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=trial/'zg_outputs';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6));ax.axis('off')
    ax.text(.5,.6,'No grasp inference',ha='center',fontsize=22)
    ax.text(.5,.4,reason,ha='center',wrap=True,fontsize=10)
    for name in ['grasp_top10_rgb.png','grasp_top10.png']:fig.savefig(out/name,dpi=160)
    plt.close(fig);common.save(out/'grasp_top10.json',{'displayed_count':0,'candidate_count':0,
        'reason':reason,'inference_performed':False,'no_synthetic_grasps_drawn':True})


def execute(trial,entry_path):
    entry=common.read(entry_path);reg=common.read(entry['campaign_registration'])
    heldout=set(reg['test_object_ids']);forbidden=set(reg['excluded_old_object_ids'])
    assert all(a['model_id'] in heldout and a['model_id'] not in forbidden for a in entry['scene']['actors'])
    existing.execute(trial,entry_path)
    result=common.read(trial/'result.json')
    initial=trial/'initial_conditions.json'
    if initial.exists():
        got=common.read(initial);expected=common.read(Path(reg['inputs'])/'declared_resets'/entry['id']/'declared_reset.json')
        assert got['spawn']==expected['spawn']==entry['scene'],'New held-out scene reset differs'
        assert got['base_pose']==expected['base_pose']
        assert np.array_equal(got['initial_q'],expected['initial_q'])
        common.save(trial/'independent_reset_match.json',{'passed':True,'new_scene_not_historical_replay':True,
            'scene_equal_to_preregistered_recipe':True,'robot_initial_q_equal':True,'base_pose_equal':True,
            'evaluation_scene_in_feedback_bank':False,'evaluation_objects_in_old_training_or_test':False})
    placeholder(trial,result['reason'])


class Campaign(existing.Campaign):
    def next_entry(self,worker,gpu):
        entry,claim=super().next_entry(worker,gpu)
        if entry:
            assert entry['id'] not in self.reg['source_feedback_ids_in_memory_order']
            assert entry['model_id'] in self.reg['test_object_ids']
        return entry,claim

    def worker(self,index,gpu):
        worker='scene_'+str(index)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        _,baseline=core.batch.original_modules()
        try:
            while not self.failed.is_set() and not (self.out/'STOP').exists():
                if not self.resources(worker,gpu):break
                guard(self.reg);entry,claim=self.next_entry(worker,gpu)
                if entry is None:break
                slot=entry['id'];trial=self.out/'trials'/slot
                self.progress(worker,phase='executing',current=slot,gpu=gpu)
                if entry['admission_failed']:
                    trial.mkdir(exist_ok=False)
                    common.save(trial/'result.json',{**entry,'success':False,'planning_success':False,
                        'physical_attempt':False,'reason':'independent_scene_admission_failed'})
                    placeholder(trial,'Declared scene admission failed; no grasp inference or execution')
                else:
                    command=['bash',str(CODE/'run.sh'),str(Path(__file__).resolve()),'execute','--output',str(trial),
                        '--entry',str(self.out/'entries'/(slot+'.json'))]
                    with (self.out/'entries'/(slot+'.log')).open('x') as stream:
                        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                        self.progress(worker,current=slot,child_pid=child.pid,phase='executing',gpu=gpu);child.wait()
                    if child.returncode:raise RuntimeError('Held-out execution failure; preserve without replay: '+slot)
                baseline.verify_trial(trial);result=common.read(trial/'result.json');guard(self.reg)
                claim.update(state='complete',finished_at=core.now(),result_sha256=common.sha(trial/'result.json'))
                common.save(self.out/'claims'/(slot+'.json'),claim)
                self.progress(worker,phase='between_scenes',current=None,child_pid=None)
                print('RESULT',self.reg['experiment_arm'],slot,result['success'],result['reason'],flush=True)
        except BaseException:
            self.failed.set();error={'worker':worker,'error':traceback.format_exc(),'time':core.now()}
            self.errors.append(error);common.save(self.out/'workers'/(worker+'_halt.json'),error)
            self.progress(worker,phase='halted',error=error['error'])


def report(out):
    reg=common.read(out/'registration.json')
    lines=['# 独立物体与场景测试','',
        f"目标及干扰物只使用 {len(reg['test_object_ids'])} 个新物体模型（含同类尺寸和部件变体）。与旧 40 个模型及其碰撞/视觉网格、示教场景 ID 和采样种子分离。每组同一批 200 例，五类场景各 40 例。",'',
        '示教和反馈只取旧开发集，12 个成功、13 个失败，测试前冻结；本轮所有测试结果禁止入库。组件入口检查物体隔离，评分/召回更新接口被禁用。','',
        '| 组别 | 完成/总数 | 规划通过 | 完成接近和闭爪 | 抓取成功 | 闭爪执行率 | 抓取成功率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,profile,mode in reg['arms']:
        arm=out/name;completed=[]
        for f in (arm/'claims').glob('*.json'):
            if common.read(f)['state']=='complete':completed.append(common.read(arm/'trials'/f.stem/'result.json'))
        n=len(completed);plans=sum(r['planning_success'] for r in completed);success=sum(r['success'] for r in completed)
        closed=sum(r.get('stages',{}).get('approach',{}).get('settled',False) and r.get('stages',{}).get('close',{}).get('settled',False) for r in completed)
        if n:lines.append(f'| {name} | {n}/200 | {plans} | {closed} | {success} | {closed/n:.1%} | {success/n:.1%} |')
        else:lines.append(f'| {name} | 0/200 | 0 | 0 | 0 | — | — |')
    lines+=['','未结束时比率为临时结果。初始化失败保留在两组分母中，不替换困难场景。','',
        '沿用 43 mm TCP、原生 ZeroGrasp 后处理、既有插入截断和原碰撞/物理成功条件，包括闭爪模型 8 mm 桌面余量。GraspCL 增加冻结示教召回并对修改位姿重评分；ZeroGrasp 新朝向沿用父候选分数。比较完整链路，不作为共享候选池的评分消融。','',
        '模型预训练数据是否包含这些 YCB 模型未作保证；这里的独立性指本地 ARX 示教/反馈与旧 benchmark 的物体网格、场景和种子隔离。观察位沿用原目标包围盒中心定位策略，实例分割掩码也沿用原评测条件。','',
        '[登记](registration.json) · [进度](progress.json) · [耗时](suite_timing.json)。每例图像、录像和独立执行判定见各组 `trials/<场景>/`，抓取位姿图在 `zg_outputs/`。','']
    (out/'独立测试结果.md').write_text('\n'.join(lines))


def verify(out):
    reg=common.read(out/'registration.json');guard(reg);bank=Path(reg['bank'])
    models=set(reg['test_object_ids']);old=set(reg['excluded_old_object_ids']);ids=set(reg['source_feedback_ids_in_memory_order'])
    assert not models&old and not ({e['id'] for e in reg['schedule']}&ids)
    checks=Counter()
    for name,profile,_ in reg['arms']:
        arm=out/name;files=list((arm/'claims').glob('*.json'));assert len(files)==200
        for f in files:
            c=common.read(f);assert c['state']=='complete';trial=arm/'trials'/f.stem
            assert c['result_sha256']==common.sha(trial/'result.json')
            assert common.read(trial/'independent_evaluation.json')['passed'];checks['terminal_results']+=1
            for resource in ['grasp_top10_rgb.png','grasp_top10.png','grasp_top10.json']:assert (trial/'zg_outputs'/resource).exists()
            if (trial/'capture.json').exists():
                assert common.read(trial/'capture.json')['fresh_rgbd'];assert common.read(trial/'independent_reset_match.json')['passed']
                checks['fresh_captures_and_exact_resets']+=1
            if profile=='graspcl_200':
                for resource in ['frozen_memory_inference.json','frozen_memory_rescore.json']:
                    p=trial/resource
                    if p.exists():
                        m=common.read(p);assert m['test_feedback_added']==0 and m['before_sha256']==m['after_sha256'];checks['unchanged_memory_receipts']+=1
    assert checks['terminal_results']==reg['total_slots']
    assert common.sha(out/'resident/graspcl_model/model/score_cl/online_data.npz')==common.sha(bank/'model/score_cl/online_data.npz')
    for p,digest in reg['asset_sha256'].items():assert common.sha(p)==digest
    common.save(out/'final_verification.json',{'passed':True,**dict(checks),'object_and_scene_overlap':0,
        'training_updates_during_test':0,'all_registered_cases_preserved':True,'no_started_slots_replayed':True})


def run(out):
    reg=common.read(out/'registration.json');guard(reg)
    existing.update_suite=update_suite
    with (out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'suite_timing.json').exists():raise RuntimeError('Already started; never replay')
        backends=Backends(out);started=time.perf_counter();clock={'started_at':core.now(),'phase':'starting_components','rounds':{}}
        common.save(out/'suite_timing.json',clock)
        try:
            update_suite(out,'starting_components');backends.start();existing.validate(out,backends,reg)
            clock['shared_startup_seconds']=time.perf_counter()-started
            common.save(out/'test_started.json',{'at':core.now(),'training_closed':True,'inputs_frozen':True})
            for name,_,_ in reg['arms']:
                arm=out/name;start=time.perf_counter();phase={'started_at':core.now(),'phase':'running',
                    'started_monotonic':start,'shared_suite_startup_seconds':clock['shared_startup_seconds']}
                common.save(arm/'round_timing.json',phase);update_suite(out,active=name)
                try:Campaign(arm).start()
                finally:
                    phase.update(ended_at=core.now(),round_wall_seconds=time.perf_counter()-start,phase=common.read(arm/'progress.json')['phase'])
                    common.save(arm/'round_timing.json',phase);clock['rounds'][name]=phase;common.save(out/'suite_timing.json',clock);report(out)
                require_ready(out/'resident',reg['required_components'])
                common.save(arm/'resident_health_after_round.json',{k:rpc(Path(backends.config['sockets'][k]),{'command':'health'}) for k in reg['required_components']})
            verify(out);clock['phase']='complete';update_suite(out,'complete')
        except BaseException:
            clock['phase']='halted';common.save(out/'halt.json',{'at':core.now(),'error':traceback.format_exc(),'replay_allowed':False})
            update_suite(out,'halted');raise
        finally:
            stopped=time.perf_counter();backends.stop();clock.update(ended_at=core.now(),shutdown_seconds=time.perf_counter()-stopped,
                total_suite_wall_seconds=time.perf_counter()-started,all_owned_services_exited=all(p.poll() is not None for p in backends.children.values()))
            common.save(out/'suite_timing.json',clock);report(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','execute','report'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inputs',type=Path)
    parser.add_argument('--entry',type=Path);parser.add_argument('--modes',choices=['fixed','free','both'],default='fixed')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':
        modes=['fixed_orientation','free_orientation'] if args.modes=='both' else ['fixed_orientation' if args.modes=='fixed' else 'free_orientation']
        prepare(out,args.inputs.resolve(),modes)
    elif args.stage=='run':run(out)
    elif args.stage=='report':report(out)
    else:execute(out,args.entry.resolve())
