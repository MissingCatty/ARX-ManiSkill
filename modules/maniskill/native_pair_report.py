"""Read verified current-round results and render round-time/paired reports."""
import argparse
from datetime import datetime
from pathlib import Path
import time

from graspcl_pipeline import read,save,sha

PROFILES=['zerograsp_200','graspcl_200']
LABELS={'zerograsp_200':'ZeroGrasp＋cuRobo','graspcl_200':'ZeroGrasp＋GraspCL＋cuRobo'}


def duration(value):
    if value is None:return '—'
    if value<60:return f'{value:.1f} 秒'
    if value<3600:return f'{int(value//60)} 分 {value%60:.0f} 秒'
    return f'{int(value//3600)} 小时 {int(value%3600//60)} 分'


def completed(arm,cache):
    rows={}
    for path in (arm / 'claims').glob('*.json'):
        claim=read(path)
        if claim['state']!='complete':continue
        key=(str(arm),path.stem,claim['result_sha256'])
        if key not in cache:
            trial=arm / 'trials' / path.stem
            assert read(trial / 'independent_evaluation.json')['passed']
            assert sha(trial / 'result.json')==claim['result_sha256']
            result=read(trial / 'result.json')
            cache[key]={k:result[k] for k in ['id','layout','success','physical_attempt','planning_success','reason']}
        rows[path.stem]=cache[key]
    return rows


def report(root,cache):
    reg=read(root / 'registration.json')
    progress=read(root / 'progress.json')
    clock=read(root / 'suite_timing.json') if (root / 'suite_timing.json').exists() else {}
    arms={};tables=[];timings={}
    for profile in PROFILES:
        arm=root / profile;rows=completed(arm,cache);state=read(arm / 'progress.json')
        timing=read(arm / 'timing_summary.json') if (arm / 'timing_summary.json').exists() else {}
        arms[profile]={'registered':200,'completed':len(rows),'physical_attempts':sum(r['physical_attempt'] for r in rows.values()),
            'successes':sum(r['success'] for r in rows.values()),'phase':state['phase'],'results':rows}
        timings[profile]=timing
        elapsed=timing.get('round_elapsed_seconds_excluding_shared_startup')
        mean=(timing.get('controller_slot_seconds') or {}).get('mean')
        eta=timing.get('estimated_total_round_seconds')
        tables.append(f"| {LABELS[profile]} | {len(rows)}/200 | {arms[profile]['physical_attempts']} | {arms[profile]['successes']} | {duration(elapsed)} | {duration(mean)} | {duration(eta)} |")
    paired=[]
    for entry in reg['schedule']:
        slot=entry['id']
        if all(slot in arms[p]['results'] for p in PROFILES):
            base=arms[PROFILES[0]]['results'][slot];cl=arms[PROFILES[1]]['results'][slot]
            paired.append({'id':slot,'layout':entry['layout'],'zerograsp_success':base['success'],'graspcl_success':cl['success']})
    tally={'paired_completed':len(paired),'both_successful':sum(r['zerograsp_success'] and r['graspcl_success'] for r in paired),
        'zerograsp_failed_graspcl_succeeded':sum(not r['zerograsp_success'] and r['graspcl_success'] for r in paired),
        'zerograsp_succeeded_graspcl_failed':sum(r['zerograsp_success'] and not r['graspcl_success'] for r in paired),
        'both_failed':sum(not r['zerograsp_success'] and not r['graspcl_success'] for r in paired)}
    save(root / 'comparison.json',{'phase':progress['phase'],'old_results_carried':0,'rounds':arms,
        'paired':tally,'results':paired,'updated_at':datetime.now().astimezone().isoformat()})
    lines=['# 本次两条链路耗时统计','',f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",'',
        '| 链路 | 已完成 | 实际抓取 | 成功 | 本轮已用时 | 每例平均用时 | 预计整轮用时 |',
        '|---|---:|---:|---:|---|---|---|',*tables,'',
        f"- 两轮共用的组件加载、预热及启动核验：{duration(clock.get('shared_startup_seconds'))}。",
        f"- 两轮合计总耗时（含启动和服务关闭）：{duration(clock.get('total_suite_wall_seconds'))}。",
        '- 每轮为 200 个新测试槽，两个场景并行；整轮用时直接测墙钟时间，不把逐例用时相加。',
        '- 每例平均用时包含进程启动和独立结果核验；本轮用时不包含两轮共用的组件启动时间。','',
        '## 每例耗时分解','',
        '| 链路 | GPU 后端排队 | ZeroGrasp | GraspCL | cuRobo | 仿真与接触记录 | 录像采集编码及收尾 | 位姿出图 | 其他 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for profile in PROFILES:
        m=timings[profile].get('mean_trial_breakdown_seconds',{})
        values=[m.get('gpu_backend_queue_seconds'),m.get('zerograsp_client_seconds'),m.get('graspcl_client_seconds'),
            m.get('curobo_client_seconds'),m.get('simulation_and_contact_trace_seconds'),
            m.get('recording_capture_encode_seconds',0)+m.get('recording_finish_seconds',0) if m else None,
            m.get('visualization_seconds'),m.get('other_trial_seconds',0)+m.get('execution_setup_seconds',0)+m.get('native_candidate_adapter_client_seconds',0) if m else None]
        lines.append('| '+LABELS[profile]+' | '+' | '.join(duration(x) for x in values)+' |')
    lines.extend(['','分解取已实际进入执行流程的试例平均值；场景准入失败单独计数。',''])
    target=root / '耗时统计.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(lines));tmp.replace(target)
    return progress['phase'] in ['complete','stopped','halted'] and 'ended_at' in clock


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=args.output.resolve();cache={}
    save(root / 'report_registration.json',{'source_sha256':sha(Path(__file__)),
        'read_only_observer':True,'selection_or_outcome_updates':False})
    while True:
        done=report(root,cache)
        if done or not args.watch:break
        time.sleep(15)
