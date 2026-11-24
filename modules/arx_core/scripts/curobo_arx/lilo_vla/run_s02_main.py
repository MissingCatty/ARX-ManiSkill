"""One authorized S02 main campaign; immutable pilot receipts, exactly 32 slots.

Uses the corrected S02 executor without modifying policy or physical semantics.
An explicit, hash-bound carryforward resolves old infrastructure faults; it does
not turn those old terminal receipts into successful/admissible trials.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from interaction_contract import ROOT, read, write, file_sha, reject_overrides
from s02_protocol import registration, verify_sources
from s02_gates import pilot_coverage
from run_s02 import preflight, execute, next_trial

VERSION = 'arx_s02_main_v1'
RESOLUTIONS = {'cold_start_timeout', 'recorder_status_race', 'setup_summary_field'}


def verify_carryforward(value):
    carry = value['main_carryforward']
    if carry['main_authorized'] is not True or carry['automatic_retries'] != 0:
        raise ValueError('Explicit main authorization without retries is required')
    if carry['original_terminal_states_rewritten'] is not False:
        raise ValueError('Old terminal states must remain unchanged')
    if set(carry['resolutions']) != RESOLUTIONS:
        raise ValueError('All three historical infrastructure faults need resolutions')
    for path, expected in carry['evidence_sha256'].items():
        if file_sha(path) != expected:
            raise ValueError('Carryforward evidence changed: ' + path)
    rows = carry['pilots']
    if len(rows) != 4 or len({r['trial_id'] for r in rows}) != 4:
        raise ValueError('Exactly four original pilot receipts are required')
    results = {}
    for row in rows:
        if file_sha(row['result_path']) != row['sha256']:
            raise ValueError('Original pilot receipt changed')
        result = read(row['result_path'])
        if result['trial_id'] != row['trial_id'] or result['case_id'] != row['case_id'] or result['arm'] != row['arm']:
            raise ValueError('Pilot identity differs')
        results[row['trial_id']] = result
    if {(r['case_id'], r['arm']) for r in rows} != {(c,a) for c in ('core_egad_A1','core_bbq_bottle') for a in ('R10','R40')}:
        raise ValueError('Pilot arms/cases differ')
    coverage = pilot_coverage([{**r, 'phase':'pilot'} for r in rows], results)
    if not coverage['pass_check']:
        raise ValueError('Historical critical event coverage incomplete: ' + str(coverage))
    checks = read(carry['offline_checks'])
    if checks['exit_code'] != 0 or checks['passed'] is not True:
        raise ValueError('Offline repair/failure checks have not passed')
    warm = read(carry['warmup_receipt'])
    if warm['status'] != 'passed' or warm['simulation_commands'] != 0:
        raise ValueError('Uniform no-motion component warmup missing')
    for fault, resolution in carry['resolutions'].items():
        if not resolution['evidence'] or not resolution['explanation']:
            raise ValueError('Empty infrastructure resolution: ' + fault)
        if any(p not in carry['evidence_sha256'] for p in resolution['evidence']):
            raise ValueError('Unbound resolution evidence: ' + fault)
    expected = [{**r,'phase':'main'} for r in read(ROOT/'research_plan/configs/s02_design.json')['planned_main_schedule']]
    if value['schedule'] != expected or len(value['schedule']) != 32:
        raise ValueError('Main schedule differs from the original 32 trials')
    return coverage


def prepare_main(design, out, carry, fixtures):
    value = registration(design)
    value['version'] = VERSION
    value['schedule'] = [r for r in value['schedule'] if r['phase'] == 'main']
    value['budget'] = {**value['budget'], 'instrumentation_validation_trials':0,
                       'maximum_total':32, 'main_total':32}
    value['main_carryforward'] = read(carry)
    value['fixtures'] = {k:read(v) for k,v in fixtures.items()}
    value['fixture_source_sha256'] = {str(v.resolve()):file_sha(v) for v in fixtures.values()}
    for renderer, fixture in value['fixtures'].items():
        if renderer != fixture['fixture']['renderer']:
            raise ValueError('Historical renderer mismatch')
    value['devices'] = dict(policy='GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834',
        perception_and_planner={k:v['resources']['gpu_uuid'] for k,v in value['fixtures'].items()},
        simulators={k:v['fixture']['gpu_uuid'] for k,v in value['fixtures'].items()})
    verify_carryforward(value)
    if shutil.disk_usage(out.parent).free <= 1073741824:
        raise OSError('Preserve 1GiB storage reserve')
    out.mkdir(exist_ok=False)
    write(out/'registration.json',value)
    storage=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if not storage.exists():storage=ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/storage_policy.json'
    shutil.copy2(storage,out/'storage_policy.json')
    for name in value['algorithms']:
        path=Path(name)
        dest=out/'source_snapshot'/path.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in value['dependencies']:
        path=Path(name)
        dest=out/'registered_inputs'/path.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(out/'state.json',dict(status='prepared',attempts=0,unexecuted=32))
    return value


def resource_check(value):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    free={r.split(',')[0].strip():int(r.split(',')[1]) for r in output.splitlines()}
    # Same observed stable loads; reserve a modest extra 1GiB before dispatch.
    devices=(set(value['devices']['simulators'].values()) |
             set(value['devices']['perception_and_planner'].values()) | {value['devices']['policy']})
    for uuid in devices:
        if free.get(uuid,0) <= 1024:
            raise RuntimeError('Insufficient current GPU headroom: ' + uuid)
    return dict(free_mib=free,wall_time=time.time())


def summarize(out, status, stop=None):
    value=read(out/'registration.json');records=[]
    for row in value['schedule']:
        d=out/'trials'/row['trial_id']
        result=read(d/'result.json') if (d/'result.json').exists() else None
        report=read(d/'grasp_report.json') if (d/'grasp_report.json').exists() else {}
        physical=read(d/'physical_evaluation.json') if (d/'physical_evaluation.json').exists() else {}
        records.append(dict(**row,status=result['task_result'] if result else 'unknown' if d.exists() else 'unexecuted',
            result=result,reason=report.get('error') or (result or {}).get('error'),
            physical=physical,path=str(d)))
    counts={k:sum(r['status']==k for r in records) for k in ('success','failed','unknown','unexecuted')}
    preserved=all(file_sha(r['result_path'])==r['sha256'] for r in value['main_carryforward']['pilots'])
    receipt=dict(status=status,stop=stop,counts=counts,registered=32,automatic_retries=0,
        old_pilots_in_main=0,trials=records,old_receipts_preserved=preserved,
        original_success_criteria='Original evaluator; lift >6cm for every sample of at least 3s hold')
    write(out/'acceptance.json',receipt)
    text=['# S02 正式主对照验收','',f'状态：{status}。固定G，R10/R40，原定32条；旧4条补录单独保留。',
        f"成功 {counts['success']}，已执行抓取/算法失败 {counts['failed']}，已开始但终态未知 {counts['unknown']}，未执行 {counts['unexecuted']}。",'',
        '模型、机器人、场景物理、初始条件和成功判据沿用登记基线；BBQ使用PathTracing，其余场景使用RaytracedLighting。',
        '两组使用相同的无动作组件预热流程；旧超时/清理/接口异常通过独立修复证据承接，未改写旧停止标记。','',
        '| 物体 | R10：成功/已开始（登记） | R40：成功/已开始（登记） | 主要失败或缺失 |',
        '|---|---:|---:|---|']
    for cid,case in value['cases'].items():
        arms=[];reasons=[]
        for arm in ('R10','R40'):
            rows=[r for r in records if r['case_id']==cid and r['arm']==arm]
            arms.append(f"{sum(r['status']=='success' for r in rows)}/{sum(r['status']!='unexecuted' for r in rows)}（{len(rows)}）")
            for r in rows:
                if r['status']=='failed':
                    reason=str(r['reason'] or r['physical'].get('outcome','原物理评估未通过'))
                    if reason not in reasons:reasons.append(reason)
                elif r['status']=='unknown':reasons.append('基础设施/记录故障，终态未知')
        if any(r['case_id']==cid and r['status']=='unexecuted' for r in records):reasons.append('含未执行项')
        text.append('| '+case['asset']+' | '+' | '.join(arms)+' | '+('; '.join(reasons) or '无')+' |')
    if stop:text+=['','停止原因：`'+str(stop).replace('`',"'")+'`。资源/配置/记录链或空闲状态故障后未继续调度。']
    text+=['','下一步：'+('先处理本轮停止原因，再单独决定未执行项的承接；本轮不自动重试。' if stop else '按物体区分到达拒绝、闭合偏位和起抬保持失败；用本轮同期对照决定入口对齐或抓取目标诊断，不直接宣称泛化提升。'),
        '',f'[完整登记、逐条结果和证据](../{out.relative_to(ROOT)}/acceptance.json)。未执行项未计为已观察到的抓取失败。']
    (ROOT/'research_plan/S02_ACCEPTANCE.md').write_text('\n'.join(text)+'\n')
    return receipt


def run_main(out, policy_url):
    from batch_ledger import lock
    from contextlib import ExitStack
    value=read(out/'registration.json')
    if (out/'run_started.json').exists():raise ValueError('Campaign already started; no automatic retry/resume')
    with ExitStack() as stack:
        stack.enter_context(lock(out/'campaign.lock'))
        for mailbox in sorted({v['fixture']['mailbox'] for v in value['fixtures'].values()}):
            stack.enter_context(lock(Path(mailbox)/'p0_execution.lock'))
        write(out/'run_started.json',dict(wall_time=time.time(),pid=os.getpid()))
        stop=None
        try:
            verify_sources(value);verify_carryforward(value)
            # Check every registered case/renderer before the first motion.
            for cid,case in value['cases'].items():
                evidence=preflight(value,dict(case_id=cid,arm='R10'),value['fixtures'][case['renderer']],policy_url)
                write(out/('initial_preflight_'+cid+'.json'),evidence)
            for index,row in enumerate(value['schedule']):
                verify_carryforward(value);verify_sources(value)
                next_trial(value,out,row['trial_id'])
                if shutil.disk_usage(out).free<=1073741824:raise OSError('1GiB storage reserve reached')
                write(out/('resources_'+row['trial_id']+'.json'),resource_check(value))
                fixture=value['fixtures'][value['cases'][row['case_id']]['renderer']]
                evidence=preflight(value,row,fixture,policy_url)
                write(out/('preflight_'+row['trial_id']+'.json'),evidence)
                print('S02_MAIN_BEGIN',index+1,32,row['trial_id'],flush=True)
                write(out/'state.json',dict(status='running',current=row['trial_id'],slot=index+1,total=32))
                result=execute(value,row,out,fixture,policy_url,evidence)
                print('S02_MAIN_END',row['trial_id'],result['task_result'],'halt',result['halt'],flush=True)
                summarize(out,'running')
                if result['halt']:
                    stop=dict(trial_id=row['trial_id'],reason='Execution infrastructure/configuration/record/cleanup gate',result=result)
                    break
        except BaseException as exc:
            stop=dict(type=type(exc).__name__,message=str(exc))
            write(out/'scheduling_stop.json',stop)
        status='stopped' if stop else 'completed'
        receipt=summarize(out,status,stop)
        write(out/'state.json',dict(status=status,counts=receipt['counts'],stop=stop))
        print('S02_MAIN_FINISHED',status,json.dumps(receipt['counts']),flush=True)
        return 1 if stop else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--carryforward',type=Path);p.add_argument('--fixture-rt',type=Path);p.add_argument('--fixture-pt',type=Path)
    p.add_argument('--policy-url',default='http://127.0.0.1:8008')
    a=p.parse_args();reject_overrides(os.environ)
    if a.prepare:
        if not all((a.carryforward,a.fixture_rt,a.fixture_pt)):p.error('Carryforward and both fixtures required')
        prepare_main(ROOT/'research_plan/configs/s02_design.json',a.output.resolve(),a.carryforward,
            {'RaytracedLighting':a.fixture_rt,'PathTracing':a.fixture_pt})
        print('Registered exactly 32 main trials, no pilot repeats',flush=True)
    else:raise SystemExit(run_main(a.output.resolve(),a.policy_url))


if __name__=='__main__':main()
