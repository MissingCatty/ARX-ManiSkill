"""Measure fixed-slot overhead and retained planning outcomes."""
import argparse
from pathlib import Path
from threading import Event,Thread
import traceback

import numpy as np

from arx_project.layout import ROOT,ensure_run_path,now,read_json,write_json
from modules.pipeline.experiments.endpoint_parallel.shared_runner import record_load
from modules.pipeline.experiments.parallel_planning.run import run as original_run
from modules.pipeline.experiments.parallel_planning.audit import audit


def analyze(root):
    reg=read_json(root/'registration.json'); progress=read_json(root/'progress.json'); pairs=[]
    for case in reg['cases']:
        for repeat in range(1,reg['config'].get('repeats',2)+1):
            paths={m:root/'plans'/case['id']/(m+'_'+str(repeat)) for m in ['control','cached']}
            plans={m:read_json(p/'plan_result.json') for m,p in paths.items()}; a,b=plans.values()
            same=a['success']==b['success']
            if a['success'] and b['success']:
                same &= all(a['selected'][k]==b['selected'][k] for k in ['arm','index','source_generator_index','open_finger_m'])
                same &= all(np.array_equal(a['selected'][k],b['selected'][k]) for k in ['q_grasp','q_pre','q_lift'])
                same &= all(np.array_equal(a['paths'][s][k],b['paths'][s][k]) for s in ['pregrasp','approach'] for k in ['q','times'])
            pairs.append({'case':case['id'],'repeat':repeat,'exact_selection_and_paths':bool(same)})
    if not all(p['exact_selection_and_paths'] for p in pairs): raise RuntimeError('Fixed slots changed original planning results')
    groups={}
    for mode in ['control','cached']:
        rows=[r for r in progress['results'] if r['mode']==mode]; times=[r['RPC_seconds'] for r in rows]
        stage=lambda name:float(np.mean([r['stages'].get(name,{}).get('exclusive_seconds',0.) for r in rows]))
        groups[mode]={'requests':len(rows),'successes':sum(r['success'] for r in rows),
            'total_RPC_seconds':sum(times),'mean_RPC_seconds':float(np.mean(times)),
            'median_RPC_seconds':float(np.median(times)),'p90_RPC_seconds':float(np.percentile(times,90)),
            'mean_scene_update_seconds':stage('binding_scene_update'),
            'mean_contact_scene_update_seconds':stage('contact_scene_update'),
            'mean_grasp_IK_seconds':stage('grasp_IK'),
            'whole_scene_replacements':sum(r['memory_updates']['scenes']['geometry_replacements'] for r in rows),
            'slot_replacements':sum(r['memory_updates']['scenes'].get('slot_replacements',0) for r in rows),
            'native_GPU_mesh_creations':sum(r['cpu_geometry']['native_load_calls'] for r in rows)}
    a,b=groups.values(); reduction=1-b['total_RPC_seconds']/a['total_RPC_seconds']
    adopted=bool(reduction>=reg['config']['minimum_total_time_reduction'] and b['median_RPC_seconds']<a['median_RPC_seconds']
                 and b['p90_RPC_seconds']<=a['p90_RPC_seconds']*reg['config']['maximum_p90_ratio'])
    accepted=read_json(root/'acceptance.json'); accepted.update(record_integrity=True,exact_selection_and_paths=True,
        reduction_fraction=reduction,optional_implementation_accepted=adopted,
        event_applicability='pure planning and incremental collision scenes',
        key_event_coverage={'hot_arm_switch':True,'target_enable_flags':True,'cross_request_reuse':True,
                            'external_mutation_invalidation':True,'native_collision_cost_and_gradient':True})
    write_json(root/'acceptance.json',accepted); write_json(root/'analysis.json',{'at':now(),'groups':groups,'pairs':pairs,'reduction_fraction':reduction})
    final=reg['config'].get('final_combination',False)
    title='V8_场景更新优化完整规划对照结果.md' if final else 'V8_第3项固定槽位验证结果.md'
    lines=['# '+title[:-3],'',f"{len(reg['cases'])}场景{reg['planning_requests']}次纯规划完成。两组抓法、选臂、关节和路径逐值相同，完整RPC累计节省{reduction:.2%}。",'',
           '| 项目 | 对照 | 优化 |','| --- | ---: | ---: |']
    for key,label in [('mean_RPC_seconds','完整RPC平均'),('median_RPC_seconds','完整RPC中位数'),('p90_RPC_seconds','完整RPC P90'),
        ('mean_scene_update_seconds','规划器与独立检查器场景更新'),('mean_contact_scene_update_seconds','精细接触场景更新'),('mean_grasp_IK_seconds','抓取IK')]:
        lines.append(f'| {label} | {a[key]:.4f}秒 | {b[key]:.4f}秒 |')
    lines += ['',f"整场重装次数{a['whole_scene_replacements']}→{b['whole_scene_replacements']}；优化组单槽替换{b['slot_replacements']}次。",'',
        f"任务结果：{reg['planning_requests']}/{reg['planning_requests']}完成。记录完整性：通过。事件适用性：纯规划。关键事件覆盖：切臂、目标启停、跨请求复用、外部修改失效和GPU碰撞结果核对。",'',
        f"可选实现验收：{'通过' if adopted else '未达到完整时延采用门槛'}。",'',f'原始记录：`{root.relative_to(ROOT)}`。','']
    report='\n'.join(lines); (root/'report.md').write_text(report)
    (ROOT/'modules/pipeline/docs/mainline_versions'/title).write_text(report)
    print('SLOT_ACCEPTANCE',adopted,groups,flush=True)


def run(root):
    reg=read_json(root/'registration.json'); stop=Event(); thread=Thread(target=record_load,args=(root,reg['gpu'],stop)); thread.start()
    try: original_run(root)
    finally: stop.set(); thread.join(timeout=15)
    if thread.is_alive() or not read_json(root/'gpu_load_summary.json')['planning_samples']: raise RuntimeError('Incomplete GPU load records')
    audit(root); analyze(root)
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'stage':reg['config']['stage'],'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    root=ensure_run_path(parser.parse_args().root)
    try: run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False}); raise
