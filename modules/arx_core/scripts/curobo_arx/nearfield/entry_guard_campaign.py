"""Advisor25: reuse registered models, admit inward targets, exactly four new ARX slots."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.launch import devices, environment, allocate
from scripts.curobo_arx.data_factory.watchdog import process_info, exclusive_stop
from .prepare_v2 import predecessors
from .resources import probe_cuda
from training.nearfield_arx.entry_guard import VERSION

MODULE = 'scripts.curobo_arx.nearfield.entry_guard_campaign'
NEW_SOURCES = ('training/nearfield_arx/entry_guard.py', 'training/nearfield_arx/entry_guard_policy.py',
    'training/nearfield_arx/entry_guard_diagnosis.py', 'training/nearfield_arx/test_entry_guard.py',
    'scripts/curobo_arx/nearfield/runtime_entry_guard.py', 'scripts/curobo_arx/nearfield/entry_guard_campaign.py')
IMAGE = 'localhost/arx-s02-isaac:6.0.1'


def prepare(parent, diagnosis, out):
    parent, diagnosis, out = (Path(p).resolve() for p in (parent, diagnosis, out))
    if out.exists():raise FileExistsError('Keep all previous registrations')
    root = parent.parent; old=read(parent/'registration.json')
    previous=read(root/'final_verification.json')
    if not previous['pass'] or not previous['record_integrity_pass']:raise RuntimeError('Parent verification incomplete')
    idle=predecessors([parent]);verify_files(previous['evidence'])
    verify_files({**old['execution_source_files'], **old['dependencies']})
    result=read(diagnosis/'verification.json')
    if not result['pass']:raise RuntimeError('Historical diagnosis failed')
    for name in ('sources','evidence','outputs'):verify_files(result[name])
    inventory=devices();selection=old['requested_devices']
    allocate(inventory,1,selection['compute_uuid'],[selection['sim_uuid']])
    if next(d['free_mb'] for d in inventory if d['uuid']==selection['policy_uuid'])<18000:
        raise RuntimeError('Insufficient free memory for both frozen policies')
    proofs=[probe_cuda(d,environment(d)) for d in inventory if d['uuid'] in selection.values()]
    if len(proofs)!=3 or not all(p['pass'] for p in proofs):raise RuntimeError('Fresh GPU context preflight failed')
    out.mkdir(parents=True);space(out)
    check_command=[sys.executable,'-m','unittest','training.nearfield_arx.test_entry_guard',
        'training.nearfield_arx.test_eef_fit','training.nearfield_arx.test_boundary_fit',
        'training.nearfield_arx.test_generation_boundary','scripts.curobo_arx.nearfield.test_learned','-v']
    check=subprocess.run(check_command,cwd=ROOT,capture_output=True,text=True,timeout=180)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    sources={str(ROOT/n):sha(ROOT/n) for n in NEW_SOURCES}
    write(out/'offline_checks.json',{'pass':check.returncode==0,'command':check_command,
        'sources':sources,'log_sha256':sha(out/'offline_checks.log'),'no_service_or_World_started':True})
    if check.returncode:raise RuntimeError('Offline checks failed; no execution')
    source=out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    prior=Path(old['source_snapshot'])
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    for name in NEW_SOURCES:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target);execution[str(target)]=sha(target)
    deps=copy.deepcopy(old['dependencies'])
    deps.update({str(diagnosis/'verification.json'):sha(diagnosis/'verification.json'),**result['outputs'],**result['evidence'],**result['sources']})
    for p in (root/'final_verification.json',parent/'registration.json',parent/'learned_verification.json',parent/'final_idle_verification.json'):
        deps[str(p)]=sha(p)
    for origin,name in ((ROOT/'research_plan/chat/gemini回复/gemini回复25.md','advisor25.md'),(parent/'storage_policy.json','storage_policy.json')):
        shutil.copy2(origin,out/name);deps[str(out/name)]=sha(out/name)
    write(out/'predecessor_exit_receipt.json',{'pass':True,'predecessors':idle})
    image_id=subprocess.check_output(['podman','image','inspect',IMAGE,'--format','{{.Id}}'],text=True,timeout=30).strip()
    write(out/'resource_preflight.json',{'pass':True,'inventory':inventory,'CUDA':proofs,'image':IMAGE,'image_id':image_id,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'wall':time.time(),'minimum_storage_bytes':1073741824})
    probe_command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh',IMAGE,'-c',
        'import json,sys,pathlib; import scripts.curobo_arx.nearfield.runtime_entry_guard; '
        'paths=json.load(sys.stdin); missing=[p for p in paths if not pathlib.Path(p).is_file()]; '
        'print(json.dumps({"missing":missing,"checked":len(paths)})); sys.exit(bool(missing))']
    probe=subprocess.run(probe_command,input=json.dumps(list({*execution,*deps})),capture_output=True,text=True,timeout=90)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'stdout':probe.stdout,'stderr':probe.stderr,
        'command':probe_command,'no_GPU_or_World_started':True,'no_image_download':True})
    if probe.returncode:raise RuntimeError('Frozen container import/dependency preflight failed')
    reg=copy.deepcopy(old)
    for t in reg['seed_trials']:
        catalog=copy.deepcopy(reg['catalog'])
        for a in catalog['objects']:
            a['collection_enabled']=a['object_id']==t['object_id']
            if a['collection_enabled']:a['collection_placements']=[t['placement']]
        path=out/'trial_ledgers'/(t['trial_id']+'.sqlite3');path.parent.mkdir(exist_ok=True)
        ledger=Ledger(path);ledger.initialize(catalog,{**reg['quotas'],'first_object':t['object_id'],
            'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,
            'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1});ledger.close()
    change={'version':VERSION,'parent':str(parent),'only_new_sources':list(NEW_SOURCES),
        'checkpoints_inputs_prompts_robot_physics_reset_render_success_unchanged':True,
        'D':'Full EEF residual unchanged (200mm/90deg serialization). Target allowed inside original40mm G0 region OR strictly closer to G0 than current pose; target orientation still within30deg of G0. Two refines and three calls.',
        'C':'Reject only if last measured move increased distance to G0, predicted next move increases it again, and predicted target median optical depth is below35mm and decreasing. Single outward correction and explicit GRASP remain possible.',
        'rejected_alternative':'Pure35mm depth guard rejects a known successful mustard action, so it is not deployed.',
        'limitation':'Conservative repeated-divergence stop, not an oracle action correction or proof of graspability.',
        'Gstar_use':'Historical entry setup and offline diagnostic evaluation only; never local guard/action generation.',
        'new_training_or_collection':False,'new_physical_slots':4,'automatic_retry':False}
    write(out/'source_change_receipt.json',change)
    for name in ('offline_checks.json','offline_checks.log','predecessor_exit_receipt.json','resource_preflight.json','container_preflight.json','source_change_receipt.json'):
        deps[str(out/name)]=sha(out/name)
    reg.update(stage='advisor25_entry_admission_four_new_slots',created_unix=time.time(),campaign_started_unix=time.time(),
        deadline_unix=time.time()+3*3600,wall_hours=3,source_snapshot=str(source),execution_source_files=execution,
        source_identity=digest(execution),dependencies=deps,unified_registration=str(out/'registration.json'),
        runtime_module='scripts.curobo_arx.nearfield.runtime_entry_guard',coordinator_module=MODULE,
        policy_module='training.nearfield_arx.entry_guard_policy',action_guard=change,
        execution_authority={'authorization':str(out/'advisor25.md'),'authorization_sha256':sha(out/'advisor25.md'),
            'performance_admission_required':False,'scope':'Exactly four NEW diagnostic slots, reused advisor24 checkpoints, no data collection or training'},
        max_physical_trials=4,global_physical_budget=4,automatic_startup_retries=0,automatic_attempt_retries=0,
        training_authorized_here=False,carried_results=[],prior_results=[],prerequisite_results=[])
    for name in ('seed_trials','learned_policy','frozen','render_configs','render_camera_clipping','seeds'):
        if reg[name]!=old[name]:raise ValueError('Unexpected change to frozen comparison: '+name)
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    verify_files({**execution,**deps})
    write(out/'state.json',{'status':'prepared','physical_trials':0,'scheduled_slots':4,'new_training':False})
    (out/'PLAN.md').write_text('# 回复25四槽复测登记\n\n复用回复24的C、D权重、输入、预热和四个初始状态，顺序为芥末瓶C、D，BBQ瓶D、C。只新增四次实测，不补采、不训练、不自动重试。\n\nD放行向固定粗目标靠近的中间目标，保留两次修正预算和原朝向限制。C同时出现连续背离粗目标与预测深度饱和时停止；不会替模型生成修正或闭合动作。\n\n机器人、物理、历史渲染和成功判据冻结。资源、配置、记录或空闲确认故障停止调度；普通抓取失败继续。结果分别报告任务、记录、事件适用性和覆盖，所有槽位保留视频。\n')
    return out


def verify_guard_records(campaign):
    import numpy as np
    from training.nearfield_arx.projection import Projector
    from training.nearfield_arx.entry_guard import action_admission, previous_motion, validate_receipt
    from training.nearfield_arx.learned_policy import read_bound_reply
    reg=read(campaign/'registration.json');projector=Projector(reg['learned_policy']['robot_usd'])
    records=[]
    for trial in read(campaign/'summary.json')['trials']:
        for attempt in trial['attempts']:
            ep=Path(attempt['episode']);events=read(ep/'interpreter_events.json') if (ep/'interpreter_events.json').exists() else []
            ledger=read(ep/'policy_requests.json') if (ep/'policy_requests.json').exists() else []
            for item in ledger:
                path=ep/'policy'/(item['request_id']+'_request.json');req,rep=read_bound_reply(path,path.with_name(item['request_id']+'_reply.json'))
                scale=reg['learned_policy']['visual_contract']['depth_scale']
                proof=validate_receipt(path,req,rep,scale['near_m'])
                if proof.get('reason')=='invalid_model_action':
                    records.append({'trial_id':trial['trial_id'],'request_id':item['request_id'],'invalid_model_action':True});continue
                expected=action_admission(req,rep['prediction'],projector,scale,lambda p:np.load(p,allow_pickle=False),read,
                    previous_motion(path) if req['method']=='C' else None)
                expected.update(request_id=req['request_id'],request_sha256=sha(path),prediction=rep['prediction'].strip())
                if digest(proof)!=digest(expected):raise RuntimeError('Recomputed action admission differs from executed receipt')
                dispatched=[e for e in events if e.get('policy_request_id')==req['request_id'] and e['kind']=='target_request']
                if not proof['pass'] and dispatched:raise RuntimeError('Rejected action was dispatched')
                records.append({'trial_id':trial['trial_id'],'request_id':req['request_id'],'pass':True,
                    'action_admitted':proof['pass'],'local_target_dispatched':bool(dispatched),'admission':proof})
    result={'pass':True,'applicable_policy_calls':len(records),'rows':records}
    write(campaign/'guard_verification.json',result);return result


def run(campaign):
    campaign=Path(campaign).resolve();reg=read(campaign/'registration.json')
    if (campaign/'entry_guard_launch.json').exists() or (campaign/'STOP').exists():raise RuntimeError('No repeats or STOP bypass')
    write(campaign/'entry_guard_launch.json',{'pid':os.getpid(),**(process_info(os.getpid()) or {}),
        'registration_sha256':sha(campaign/'registration.json'),'wall':time.time()})
    try:
        from training.nearfield_arx import entry_guard, eef_fit, advisor22_report
        from . import learned_campaign, learned_report
        learned_report.decode=entry_guard.decode
        physical=learned_campaign.run(campaign)
        if not physical['record_integrity_pass'] or physical['completed']!=4 or physical['unexecuted'] or physical['unknown_or_incomplete']:
            raise RuntimeError('Four-slot accounting incomplete; preserve all outcomes')
        guard=verify_guard_records(campaign)
        advisor22_report.encode=eef_fit.encode;advisor22_report.analyze(campaign)
        result={'pass':True,'record_integrity_pass':True,'new_training':False,'new_collection':False,
            'physical':{k:v for k,v in physical.items() if k!='evidence'},'guard':guard,
            'event_coverage_trials':physical['event_coverage_trials'],'owned_processes_and_Worlds_absent':predecessors([campaign]),
            'evidence':{str(campaign/n):sha(campaign/n) for n in ('registration.json','learned_verification.json','guard_verification.json','trajectory_analysis.json','final_idle_verification.json')}}
        write(campaign/'final_verification.json',result)
        lines=['# 回复25四槽验收','', '复用上一轮模型；D门控允许向内的中间目标，C增加连续偏离且深度趋于饱和的停止护栏。机器人、场景和成功标准不变。','',
            '| 物体 | 方法 | 任务结果 | 模型调用 | 录像 |','|---|---|---|---:|---|']
        for t in physical['trials']:
            video=t['video'];link=f"[视频]({video['path']})" if video else '未录制'
            lines.append(f"| {t['object_id']} | {t['method']} | {t['outcome']} | {t['policy_calls']} | {link} |")
        lines+=['',f"登记4次，实际{physical['actual_attempts']}次，成功{physical['physical_successes']}次，抓取失败{physical['physical_failures']}次，未执行{len(physical['unexecuted'])}次，未知{len(physical['unknown_or_incomplete'])}次。",'',
            '记录完整性：通过。原始模型输出、动作、门控回执已逐项核对，门控也从实际观测重新计算。事件适用性逐槽保留，未闭合的槽位不记为通过闭合检查。','',
            '| 关键事件 | 实际发生槽位数 |','|---|---:|']
        for k,v in physical['event_coverage_trials'].items():lines.append(f'| {k} | {v} |')
        lines+=['','[逐步轨迹分析](TRAJECTORY_ANALYSIS.md) · [完整记录](final_verification.json)']
        (campaign/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
        write(campaign/'entry_guard_completion.json',{'pass':True,'wall':time.time(),'physical_trials':physical['actual_attempts']})
        return result
    except BaseException:
        error=traceback.format_exc();exclusive_stop(campaign/'STOP',{'reason':'advisor25_execution_or_verification_fault','error':error})
        write(campaign/'entry_guard_fault.json',{'error':error,'automatic_retry':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('prepare');q.add_argument('parent');q.add_argument('diagnosis');q.add_argument('out')
    q=s.add_parser('run');q.add_argument('campaign');a=p.parse_args()
    if a.action=='prepare':print(prepare(a.parent,a.diagnosis,a.out))
    else:run(a.campaign)
