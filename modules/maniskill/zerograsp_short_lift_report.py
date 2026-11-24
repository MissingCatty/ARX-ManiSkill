"""Independent evaluation reporting; never calls learning or simulator services."""
from collections import Counter
from datetime import datetime
import argparse
import fcntl
import json
from pathlib import Path
import time

import graspcl_pipeline as common

CODE=Path(__file__).resolve().parent


def analyze(root):
    reg=common.read(root/'registration.json');arms={};checks=Counter();details=[]
    for name,profile,mode in reg['arms']:
        directory=root/name;rows=[];origins=Counter();closes=0;bilateral=0
        for f in sorted((directory/'claims').glob('*.json')):
            claim=common.read(f)
            if claim['state']!='complete':continue
            trial=directory/'trials'/f.stem;r=common.read(trial/'result.json')
            assert claim['result_sha256']==common.sha(trial/'result.json')
            assert common.read(trial/'independent_evaluation.json')['passed'];checks['verified_terminal_results']+=1
            rows.append(r);stages=r.get('stages',{})
            closed=bool(stages.get('approach',{}).get('settled') and stages.get('close',{}).get('settled'))
            closes+=closed;bilateral+=int(len(r.get('bilateral_fingers',[]))==2)
            assert r['model_id'] in reg['test_object_ids'] and r['id'] not in reg['source_feedback_ids_in_memory_order']
            for file in ['grasp_top10_rgb.png','grasp_top10.png','grasp_top10.json']:assert (trial/'zg_outputs'/file).exists()
            checks['complete_visualization_trials']+=1;checks['videos']+=len(list(trial.glob('*.mp4')))
            if (trial/'capture.json').exists():
                assert common.read(trial/'capture.json')['fresh_rgbd']
                assert common.read(trial/'independent_reset_match.json')['passed'];checks['fresh_captures_and_matching_resets']+=1
            for file in ['frozen_memory_inference.json','frozen_memory_rescore.json']:
                if (trial/file).exists():
                    m=common.read(trial/file)
                    assert m['points']==25 and m['test_feedback_added']==0 and m['before_sha256']==m['after_sha256']
                    checks['unchanged_training_memory_receipts']+=1
            plan=common.read(trial/'plan_result.json') if (trial/'plan_result.json').exists() else {}
            if plan.get('selected'):
                m=plan['selected']['candidate_metadata'];origins[m['origin']]+=1
                source=m.get('source_trial_id')
                assert source is None or source in reg['source_feedback_ids_in_memory_order']
                assert source!=r['id']
            details.append({'arm':name,'id':r['id'],'model_id':r['model_id'],'layout':r['layout'],
                'planning_success':r['planning_success'],'closure_completed':closed,'success':r['success'],'reason':r['reason']})
        n=len(rows)
        arms[name]={'profile':profile,'orientation':mode,'registered':200,'completed':n,
            'initial_plan_passes':sum(r['planning_success'] for r in rows),'execution_entries':sum(r['physical_attempt'] for r in rows),
            'approach_and_closure_completed':closes,'bilateral_contact_passes':bilateral,'successes':sum(r['success'] for r in rows),
            'closure_execution_rate':closes/n if n else None,'grasp_success_rate':sum(r['success'] for r in rows)/n if n else None,
            'planning_pass_rate':sum(r['planning_success'] for r in rows)/n if n else None,
            'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success'])),
            'selected_origins':dict(origins),
            'by_layout':{layout:{'registered':40,'completed':sum(r['layout']==layout for r in rows),
                'planning_passes':sum(r['planning_success'] for r in rows if r['layout']==layout),
                'successes':sum(r['success'] for r in rows if r['layout']==layout)} for layout in reg['layouts']}}
        clock=directory/'round_timing.json'
        if clock.exists():arms[name]['timing']=common.read(clock)
        # The reused ledger writer has a legacy table-subset description. Only
        # correct this descriptive field once that round has stopped writing it.
        if clock.exists() and common.read(clock)['phase']=='complete' and (directory/'summary.json').exists():
            summary=common.read(directory/'summary.json')
            summary['success_scope']='New 200-scene object/mesh/scene/seed-disjoint local evaluation; full registered denominator, including admission failures.'
            summary['scope_label_corrected_by_reporter']=True
            common.save(directory/'summary.json',summary)
    progress=common.read(root/'progress.json');clock=common.read(root/'suite_timing.json') if (root/'suite_timing.json').exists() else {}
    metrics={'updated_at':datetime.now().astimezone().isoformat(),'phase':progress['phase'],
        'registered':reg['total_slots'],'completed':sum(a['completed'] for a in arms.values()),'arms':arms,
        'test_model_count':38,'test_vs_old_object_overlap':0,'test_feedback_added':0,
        'admission_passed':reg['admitted'],'admission_failed_preserved_per_arm':reg['admission_failed'],
        'checks':dict(checks),'timing':clock,'train_test_scope':reg['independence_scope']}
    common.save(root/'independent_metrics.json',metrics)
    common.save(root/'independent_case_metrics.json',details)
    lines=['# ZeroGrasp 独立测试统计（3 cm 抬升）','',
        '本批只运行 ZeroGrasp＋cuRobo，不加载或调用 GraspCL，不使用示教或反馈。目标与干扰物取旧 40 个模型之外的 38 个模型；场景、网格内容和采样种子与旧集合分离。本批复用上一轮冻结的 200 个场景。','',
        '| 链路 / 朝向 | 完成 | 规划通过 | 完成接近和闭爪 | 抓取成功 | 闭爪执行率 | 抓取成功率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    pct=lambda x:'—' if x is None else f'{x:.1%}'
    for name,a in arms.items():
        lines.append(f"| {name} | {a['completed']}/200 | {a['initial_plan_passes']} | {a['approach_and_closure_completed']} | {a['successes']} | {pct(a['closure_execution_rate'])} | {pct(a['grasp_success_rate'])} |")
    lines+=['',f"初始化通过 {reg['admitted']} 例，失败 {reg['admission_failed']} 例在每条链路的 200 例中均保留，不替换。完成前的百分比以已完成数量为分母，仅为临时结果。",'',
        '成功条件仍为双指接触、无禁止接触、动作收敛、规划抬升 3 cm，物体实际抬高至少 1 cm 并连续保持至少 3 秒。43 mm TCP、既有插入截断及闭爪碰撞模型 8 mm 桌面余量均保留。',
        '本轮禁止测试反馈入库；逐次记录物体隔离。两个场景并行，ZeroGrasp 和 cuRobo 在首例前预热并全批常驻。','',
        '本批按用户要求修改抬升与成功条件，复用上一批的冻结场景，并非重新抽取的盲测。独立性限于本地 ARX 示教和旧 benchmark；未宣称上游预训练从未见过 YCB。物体模型包含尺寸/部件变体。每例重新采图和推理，不继承上一轮结果。','',
        '[逐例统计](independent_case_metrics.json) · [机器统计与耗时](independent_metrics.json) · [登记](registration.json)。每例的 `zg_outputs/` 保存前 10 位姿图；相机、录像及执行记录在相同场景目录。','']
    if clock.get('total_suite_wall_seconds') is not None:
        seconds=round(clock['total_suite_wall_seconds']);lines.append(f'整批耗时 {seconds//3600} 小时 {seconds%3600//60} 分 {seconds%60} 秒；服务全部退出：'+str(clock['all_owned_services_exited'])+'。')
    (root/'独立测试统计.md').write_text('\n'.join(lines))
    if progress['phase'] in ['complete','halted'] and clock.get('all_owned_services_exited'):
        record=CODE/'抓取基线批次记录.md';content=record.read_text();heading='## '+root.name
        if heading in content:
            before,section=content.split(heading,1);parts=[]
            for line in section.split('\n'):
                if line.startswith('- 结果：'):
                    parts.append('- 结果：'+('全部完成，' if progress['phase']=='complete' else '批次停止，')+f"{metrics['completed']}/{reg['total_slots']} 例；已有结果独立判定及散列核查通过，常驻服务均已退出，测试反馈新增 0。")
                    for name,a in arms.items():parts.append(f"- {name}：规划通过 {a['initial_plan_passes']}/{a['registered']}；完成闭爪 {a['approach_and_closure_completed']}（{pct(a['closure_execution_rate'])}）；抓取成功 {a['successes']}（{pct(a['grasp_success_rate'])}）。")
                else:parts.append(line)
            record.write_text(before+heading+'\n'.join(parts))
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    while True:
        with (root/'report.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);data=analyze(root)
        print(json.dumps({'phase':data['phase'],'completed':data['completed'],
            'arms':{k:{x:v[x] for x in ['completed','initial_plan_passes','approach_and_closure_completed','successes']} for k,v in data['arms'].items()}}),flush=True)
        done=data['phase'] in ['complete','halted'] and data['timing'].get('all_owned_services_exited')
        if not args.watch or done:break
        time.sleep(30)
