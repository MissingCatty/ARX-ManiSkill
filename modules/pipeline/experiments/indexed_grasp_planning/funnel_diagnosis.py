"""Measure the completed 69-pair funnel from saved records; no solver calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from arx_project.layout import ROOT,ensure_run_path,now,sha256,write_json

PASSED={'native_full_world_ik','bounded_ik_with_local_table_check','verified_map_kinematic_witness'}
TABLE_REASONS={'nonfinger_table_collision','real_finger_depth_exceeds_tolerance','finger_body_contact'}


def raw(path):return json.loads(Path(path).read_text())


def endpoint_category(event):
    if event['kind'] in PASSED:return 'endpoint_passed'
    if event['kind']!='no_local_grasp_endpoint':raise ValueError('Unknown endpoint evidence kind')
    count=event['kinematic_solutions'];reasons=event['reasons']
    if type(count) is not int or count<0:raise ValueError('Invalid kinematic solution count')
    if count==0:
        if reasons:raise ValueError('No returned solution cannot have checked branch reasons')
        return 'no_kinematic_solution_returned'
    if len(reasons)!=count:raise ValueError('Missing returned-branch rejection reasons')
    classes=set()
    for reason in reasons:
        if reason in TABLE_REASONS:classes.add('table')
        elif reason=='ground_or_other_object_collision':classes.add('other_world')
        elif reason in {'joint_limits','independent_fk_error'}:classes.add('limits_or_FK')
        else:raise ValueError('Unknown branch rejection reason: '+reason)
    return 'solutions_rejected_'+(next(iter(classes)) if len(classes)==1 else 'mixed')


def summarize(rows):
    keys=['postprocessing_seconds','planning_seconds','opening_seconds','candidate_search_seconds',
        'decode_seconds','other_request_seconds','postprocess_plus_plan_seconds',
        'batched_grasp_IK_GPU_seconds','batched_grasp_IK_wall_seconds',
        'saved_trajopt_solve_seconds','saved_trajopt_validation_seconds','candidate_checks']
    return {key:{'mean':float(np.mean([r[key] for r in rows])),
        'median':float(np.median([r[key] for r in rows])),'total':float(sum(r[key] for r in rows))}
        for key in keys}


def run(root,source):
    if root.exists():raise ValueError('Preserve earlier diagnosis registrations')
    accepted=raw(source/'acceptance.json');review=raw(source/'planning_review.json')
    if (not accepted['task_complete'] or not accepted['record_integrity'] or not review['passed']
            or accepted['planning_requests']!=138 or len(accepted['cases'])!=69):
        raise ValueError('Use the fully accepted 69-pair planning comparison')
    paths=[source/name for name in ['registration.json','acceptance.json','planning_review.json']]
    for case in accepted['cases']:
        for mode in ['control','nms15']:
            d=source/'plans'/case['id']/mode
            paths += [d/name for name in ['metrics.json','candidate_order.json','candidate_io.json','interleaved_search.json']]
            for side in ['right','left']:
                arm=d/'arms'/side
                paths += [arm/'plan_result.json',arm/'opening_sweep.json']
                paths += sorted((arm/'search_windows').glob('*.json'))
                paths += sorted((arm/'plan').glob('*_optimization.json'))
    sources=[Path(__file__).resolve(),Path(__file__).with_name('test_funnel_diagnosis.py'),
        ROOT/'modules/curobo/fingertip_contact/planning.py',ROOT/'modules/curobo/fingertip_contact/validator.py',
        ROOT/'modules/curobo/indexed_planning/batches.py',ROOT/'modules/curobo/indexed_planning/pool.py',
        ROOT/'modules/curobo/dual_arm/interleaved_adapter.py']
    inputs={str(p):sha256(p) for p in sorted(set(paths))};hashes={str(p):sha256(p) for p in sources}
    root.mkdir(parents=True)
    for p in sources:
        target=root/'source_registered'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'registration.json',{'at':now(),'version':'V8','phase':'full69_funnel_diagnosis',
        'authorization':'2026-11-26：可以，然后你说的v8什么时候做；继续完整69例定位与后续批量化。',
        'source_run':str(source),'paired_cases':69,'source_hashes':hashes,'input_hashes':inputs,
        'new_IK_calls':0,'planning_requests':0,'simulation_instances':0,'physical_grasps':0})
    rows={mode:[] for mode in ['control','nms15']};counts={mode:Counter() for mode in rows}
    branch_counts={mode:Counter() for mode in rows};examples={mode:{} for mode in rows}
    for case in accepted['cases']:
        for mode in rows:
            d=source/'plans'/case['id']/mode;metric=raw(d/'metrics.json');manifest=raw(d/'candidate_order.json')
            schedule=raw(d/'interleaved_search.json')['schedule'];search=sum(v['wall_seconds'] for v in schedule)
            opening=0.;events=[];calls=[];optimizer=[];rejected=Counter()
            for side in ['right','left']:
                arm=d/'arms'/side;opening+=raw(arm/'opening_sweep.json')['wall_seconds']
                plan=raw(arm/'plan_result.json');events+=plan.get('candidate_events',[]);rejected.update(plan.get('rejected',{}))
                for event in plan.get('candidate_events',[]):
                    category=endpoint_category(event);counts[mode][category]+=1
                    branch_counts[mode].update(event.get('reasons',[]))
                    examples[mode].setdefault(category,[])
                    if len(examples[mode][category])<5:
                        index=event['candidate_index']
                        examples[mode][category].append({'id':case['id'],'arm':side,'search_rank':index,
                            'original_source_index':manifest['priority_to_original_source'][index],
                            'returned_kinematic_solutions':event.get('kinematic_solutions')})
                for path in sorted((arm/'search_windows').glob('*.json')):
                    calls+=raw(path)['result'].get('batch_ik',{}).get('solver_calls',[])
                optimizer += [raw(path) for path in sorted((arm/'plan').glob('*_optimization.json'))]
            if len(events)!=metric['candidate_checks'] or sum(w['candidate_checks'] for w in schedule)!=len(events):
                raise ValueError('Candidate event/window totals disagree')
            failures=sum(e['kind']=='no_local_grasp_endpoint' for e in events)
            if failures!=rejected['grasp_ik']:raise ValueError('Grasp rejection ledger mismatch')
            decoded=raw(d/'candidate_io.json')['load_seconds'];wall=metric['planning_request_seconds']
            rest=wall-opening-search-decoded
            if rest < -1e-6:raise ValueError('Recorded timing intervals overlap')
            row={'id':case['id'],'candidate_checks':len(events),'postprocessing_seconds':metric['postprocessing_seconds'],
                'planning_seconds':wall,'opening_seconds':opening,'candidate_search_seconds':search,
                'decode_seconds':decoded,'other_request_seconds':rest,
                'postprocess_plus_plan_seconds':metric['postprocessing_plus_planning_seconds'],
                'batched_grasp_IK_GPU_seconds':sum(v['gpu_seconds'] for v in calls),
                'batched_grasp_IK_wall_seconds':sum(v['wall_seconds'] for v in calls),
                'batched_grasp_IK_calls':len(calls),'batched_grasp_IK_targets':sum(v['targets'] for v in calls),
                'batched_grasp_IK_solver_stages':dict(Counter(v['stage'] for v in calls)),
                'saved_trajopt_solve_seconds':sum(v.get('phase_seconds',{}).get('solve',0.) for v in optimizer),
                'saved_trajopt_validation_seconds':sum(v.get('phase_seconds',{}).get('validation',0.) for v in optimizer),
                'saved_trajopt_reports':len(optimizer),'candidate_rejected':dict(rejected)}
            if abs(row['postprocess_plus_plan_seconds']-(row['postprocessing_seconds']+wall))>1e-8:
                raise ValueError('Recorded total differs from measured postprocessing plus planning')
            rows[mode].append(row)
    for name,digest in {**inputs,**hashes}.items():
        if sha256(name)!=digest:raise RuntimeError('A diagnosis input or source changed')
    summary={mode:summarize(values) for mode,values in rows.items()}
    result={'at':now(),'passed':True,'record_integrity':True,'paired_cases':69,'source_run':str(source),
        'summaries':summary,'endpoint_candidates':{k:dict(v) for k,v in counts.items()},
        'rejected_solution_branches':{k:dict(v) for k,v in branch_counts.items()},'examples':examples,'cases':rows,
        'new_IK_calls':0,'planning_requests':0,'simulation_instances':0,'physical_grasps':0,
        'missing_timing_scope':'PRM, sequential pre/lift IK and all per-window binding times are not separately accumulated in the saved records',
        'grasp_batch_IK_time_is_inside_candidate_search':True,
        'rejected_branch_counts_are_not_unique_candidate_counts':True}
    write_json(root/'acceptance.json',result)
    write_report(root,result)
    print(json.dumps({'paired_cases':69,'summaries':summary,'endpoint_candidates':result['endpoint_candidates']},ensure_ascii=False),flush=True)


def write_report(root,result):
    summaries=result['summaries'];lines=['# V8：完整69例的剩余耗时与终点拒绝定位','',
        'V8后续阶段已开始。本轮读取V7完整69个配对场景、138次已完成规划的记录，拆分耗时与抓取终点拒绝原因。新增IK、规划、仿真和物理抓取均为0。','',
        '## 整次请求耗时','','各项使用69个相同场景的平均数，便于逐项相加。','',
        '| 环节 | 原候选＋B16 | NMS15＋后备＋B16 |','| --- | ---: | ---: |']
    labels={'postprocessing_seconds':'候选后处理','opening_seconds':'双臂开爪碰撞检查',
        'candidate_search_seconds':'候选搜索，含IK、碰撞、路径和窗口内绑定','decode_seconds':'共享候选读取和解码',
        'other_request_seconds':'其余状态准备与请求记录','postprocess_plus_plan_seconds':'合计'}
    for key,label in labels.items():
        lines.append(f"| {label} | {summaries['control'][key]['mean']:.3f}秒 | {summaries['nms15'][key]['mean']:.3f}秒 |")
    lines+=['','## 抓取终点的候选分类','','下表每个候选只进入一类；一份原抓法被两只手检查时分别计数。','',
        '| 终点检查结果 | 原候选＋B16 | NMS15＋后备＋B16 |','| --- | ---: | ---: |']
    categories={'no_kinematic_solution_returned':'本次未返回合格运动学解',
        'solutions_rejected_table':'有解，全部分支被桌面碰撞拒绝',
        'solutions_rejected_other_world':'有解，全部分支被非桌面场景碰撞拒绝',
        'solutions_rejected_limits_or_FK':'有解，全部分支被限位或独立FK拒绝',
        'solutions_rejected_mixed':'有解，各分支被不同原因拒绝','endpoint_passed':'抓取终点通过'}
    for key,label in categories.items():
        lines.append(f"| {label} | {result['endpoint_candidates']['control'].get(key,0)} | {result['endpoint_candidates']['nms15'].get(key,0)} |")
    lines+=['','“本次未返回解”描述本次求解结果，不作为位姿不可达的证明。','',
        '## 候选搜索内部已有的计时','','批量抓取IK的GPU事件计时属于上方候选搜索，不重复加入总耗时。','',
        '| 指标，逐场景平均 | 原候选＋B16 | NMS15＋后备＋B16 |','| --- | ---: | ---: |']
    for key,label in [('batched_grasp_IK_GPU_seconds','批量抓取IK的GPU事件耗时'),
                      ('batched_grasp_IK_wall_seconds','批量抓取IK接口耗时'),
                      ('saved_trajopt_solve_seconds','保存的轨迹优化求解耗时'),
                      ('saved_trajopt_validation_seconds','保存的轨迹优化后验证耗时')]:
        lines.append(f"| {label} | {summaries['control'][key]['mean']:.3f}秒 | {summaries['nms15'][key]['mean']:.3f}秒 |")
    lines+=['','现有记录没有把每次PRM、逐候选预抓取/抬升IK、全部窗口绑定逐项累计。下一阶段在新登记中补齐这些计时，再比较联合批量检查；旧138次规划不重跑。','',
        '## 下一步实施重点','','1. 以NMS15＋后备＋B16为对照，新增各阶段计时与调用计数。',
        '2. 批量验证抓取关节分支的独立FK与当前场景碰撞，再把同一抓法的预抓取、抓取、抬升终点和局部连接关联起来。',
        '3. 同时比较减少重复开爪/场景绑定的接口开销，完整检查与当前V2规则保持。',
        '4. 通过后再比较4、8个已知预抓取关节目标的并行路径；保持原目标、候选来源及默认链路。','',
        f'完整逐例统计：`{root.relative_to(ROOT)}/acceptance.json`。','']
    text='\n'.join(lines);(root/'report.md').write_text(text)
    (ROOT/'modules/pipeline/docs/mainline_versions/V8_完整候选检查耗时定位.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);args=parser.parse_args();root=ensure_run_path(args.root)
    try:run(root,ensure_run_path(args.source))
    except BaseException:
        if root.exists():write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()})
        raise
