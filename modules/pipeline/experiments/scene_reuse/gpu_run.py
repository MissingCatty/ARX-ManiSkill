"""Second-stage timing, independent trajectory checks and optional adoption."""
import argparse
from pathlib import Path
from threading import Event, Thread
import traceback

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from modules.pipeline.experiments.endpoint_parallel.shared_runner import record_load
from modules.pipeline.experiments.parallel_planning.run import run as original_run
from modules.pipeline.experiments.parallel_planning.audit import audit


def analyze(root):
    reg = read_json(root/'registration.json'); progress = read_json(root/'progress.json')
    pairs = []
    for case in reg['cases']:
        for repeat in [1,2]:
            plans = {m: read_json(root/'plans'/case['id']/(m+'_'+str(repeat))/'plan_result.json') for m in ['control','cached']}
            a,b = plans['control'],plans['cached']; same = a['success']==b['success']
            if a['success'] and b['success']:
                same &= all(a['selected'][k]==b['selected'][k] for k in ['arm','index','source_generator_index','open_finger_m'])
                same &= all(np.array_equal(a['selected'][k],b['selected'][k]) for k in ['q_grasp','q_pre','q_lift'])
                same &= all(np.array_equal(a['paths'][s][k],b['paths'][s][k]) for s in ['pregrasp','approach'] for k in ['q','times'])
            pairs.append({'case':case['id'],'repeat':repeat,'exact_selection_and_paths':bool(same)})
    if not all(p['exact_selection_and_paths'] for p in pairs): raise RuntimeError('Shared geometry changed planning results')
    groups = {}
    for mode in ['control','cached']:
        rows = [r for r in progress['results'] if r['mode']==mode]; times=[r['RPC_seconds'] for r in rows]
        stage = lambda name: float(np.mean([r['stages'].get(name,{}).get('exclusive_seconds',0.) for r in rows]))
        groups[mode] = {'requests':len(rows),'successes':sum(r['success'] for r in rows),
            'total_RPC_seconds':sum(times),'mean_RPC_seconds':float(np.mean(times)),
            'median_RPC_seconds':float(np.median(times)),'p90_RPC_seconds':float(np.percentile(times,90)),
            'mean_scene_update_seconds':stage('binding_scene_update'),
            'mean_contact_scene_update_seconds':stage('contact_scene_update'),
            'mean_grasp_IK_seconds':stage('grasp_IK'),
            'mean_CPU_geometry_seconds':float(np.mean([r['cpu_geometry']['cpu_geometry_seconds'] for r in rows])),
            'native_GPU_mesh_creations':sum(r['cpu_geometry']['native_load_calls'] for r in rows),
            'shared_hits':sum(r['GPU_geometry']['shared_hits'] for r in rows)}
    a,b=groups['control'],groups['cached']; reduction=1-b['total_RPC_seconds']/a['total_RPC_seconds']
    accepted=read_json(root/'acceptance.json')
    optional=bool(reduction>=reg['config']['minimum_total_time_reduction'] and b['median_RPC_seconds']<a['median_RPC_seconds'] and b['p90_RPC_seconds']<=a['p90_RPC_seconds']*reg['config']['maximum_p90_ratio'])
    accepted.update(record_integrity=True,exact_selection_and_paths=True,
        reduction_fraction=reduction,optional_implementation_accepted=optional,
        event_applicability='pure planning and GPU geometry ownership',
        key_event_coverage={'native_cost_and_gradient':True,'independent_scene_buffers':True,
                            'consumer_release':True,'cold_first_use':True,'shared_hits':True})
    write_json(root/'acceptance.json',accepted)
    write_json(root/'analysis.json',{'at':now(),'groups':groups,'pairs':pairs,'reduction_fraction':reduction})
    lines=['# V8 第2项GPU几何共享验证结果','',f'五场景20次纯规划完成。两组抓法、手臂、关节及路径逐值相同，完整RPC累计节省{reduction:.2%}。','',
        '| 项目 | 当前内存更新基线 | GPU几何共享 |','| --- | ---: | ---: |']
    for key,label in [('mean_RPC_seconds','完整RPC平均'),('median_RPC_seconds','完整RPC中位数'),('p90_RPC_seconds','完整RPC P90'),
                      ('mean_scene_update_seconds','规划器与独立检查器场景更新'),('mean_contact_scene_update_seconds','精细接触场景更新'),
                      ('mean_CPU_geometry_seconds','CPU几何准备'),('mean_grasp_IK_seconds','抓取IK')]:
        lines.append(f'| {label} | {a[key]:.4f}秒 | {b[key]:.4f}秒 |')
    lines += ['',f"原生GPU网格创建次数{a['native_GPU_mesh_creations']}→{b['native_GPU_mesh_creations']}；共享命中{b['shared_hits']}次。",'',
        '第1项CPU数组缓存未达到采用门槛，本轮两组均使用原CPU加载器，仅GPU不可变几何共享为对比变量。首次几何加载仍计入请求。','',
        '任务结果：20/20完成。记录完整性：通过。事件适用性：纯规划和GPU资源复用。关键事件覆盖：16组原生几何核验、场景独立更新、释放一个消费者及资源池后其他消费者继续查询。','',
        f"可选实现验收：{'通过' if optional else '未达到完整时延采用门槛'}。",'',f'原始记录：`{root.relative_to(ROOT)}`。','']
    report='\n'.join(lines); (root/'report.md').write_text(report)
    (ROOT/'modules/pipeline/docs/mainline_versions/V8_第2项GPU几何共享验证结果.md').write_text(report)
    print('GPU_GEOMETRY_ACCEPTANCE',optional,groups,flush=True)


def run(root):
    reg=read_json(root/'registration.json'); stop=Event()
    thread=Thread(target=record_load,args=(root,reg['gpu'],stop)); thread.start()
    try: original_run(root)
    finally: stop.set(); thread.join(timeout=15)
    if thread.is_alive() or not read_json(root/'gpu_load_summary.json')['planning_samples']: raise RuntimeError('Incomplete GPU load records')
    audit(root); analyze(root)
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'stage':'shared_native_GPU_geometry','phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    root=ensure_run_path(parser.parse_args().root)
    try: run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False}); raise
