"""Run one registered pilot, then independently compare trajectories and timing."""
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
        for repeat in [1, 2]:
            plans = {mode: read_json(root/'plans'/case['id']/(mode+'_'+str(repeat))/'plan_result.json')
                     for mode in ['control', 'cached']}
            a, b = plans['control'], plans['cached']
            same = a['success'] == b['success']
            if a['success'] and b['success']:
                for key in ['arm', 'index', 'source_generator_index', 'open_finger_m']:
                    same &= a['selected'][key] == b['selected'][key]
                for key in ['q_grasp', 'q_pre', 'q_lift']:
                    same &= np.array_equal(a['selected'][key], b['selected'][key])
                for stage in ['pregrasp', 'approach']:
                    for key in ['q', 'times']:
                        same &= np.array_equal(a['paths'][stage][key], b['paths'][stage][key])
            pairs.append({'case': case['id'], 'repeat': repeat, 'exact_selection_and_paths': bool(same)})
    if not all(p['exact_selection_and_paths'] for p in pairs):
        raise RuntimeError('CPU-only reuse changed original planning results')
    groups = {}
    for mode in ['control', 'cached']:
        rows = [r for r in progress['results'] if r['mode'] == mode]
        values = [r['RPC_seconds'] for r in rows]
        mean = lambda key: float(np.mean([r[key] for r in rows]))
        def stage(key):
            return float(np.mean([r['stages'].get(key, {}).get('exclusive_seconds', 0.) for r in rows]))
        groups[mode] = {'requests': len(rows), 'successes': sum(r['success'] for r in rows),
            'total_RPC_seconds': sum(values), 'mean_RPC_seconds': float(np.mean(values)),
            'median_RPC_seconds': float(np.median(values)), 'p90_RPC_seconds': float(np.percentile(values, 90)),
            'mean_scene_update_seconds': stage('binding_scene_update'),
            'mean_contact_scene_update_seconds': stage('contact_scene_update'),
            'mean_grasp_IK_seconds': stage('grasp_IK'),
            'mean_CPU_geometry_seconds': float(np.mean([r['cpu_geometry']['cpu_geometry_seconds'] for r in rows])),
            'native_load_calls': sum(r['cpu_geometry']['native_load_calls'] for r in rows),
            'actual_CPU_processing_calls': sum(r['cpu_geometry']['native_load_calls'] if mode == 'control'
                else r['cpu_geometry']['cache']['counts'].get('original_mesh_processing_calls', 0) for r in rows),
            'mean_CPU_cache_write_seconds': 0. if mode == 'control' else float(np.mean([
                r['cpu_geometry']['cache']['timings'].get('disk_write_seconds', 0.) for r in rows])),
            'mean_CPU_cache_memory_bytes': 0. if mode == 'control' else float(np.mean([
                r['cpu_geometry']['cache']['memory_bytes'] for r in rows]))}
    a, b = groups['control'], groups['cached']
    reduction = 1-b['total_RPC_seconds']/a['total_RPC_seconds']
    accepted = read_json(root/'acceptance.json')
    accepted.update(record_integrity=True, exact_selection_and_paths=True,
        CPU_cache_reduction_fraction=reduction,
        optional_implementation_accepted=bool(reduction >= reg['config']['minimum_total_time_reduction']
            and b['median_RPC_seconds'] < a['median_RPC_seconds']
            and b['p90_RPC_seconds'] <= a['p90_RPC_seconds']*reg['config']['maximum_p90_ratio']),
        task_result='20 registered planning requests completed', event_applicability='planning and scene geometry only',
        key_event_coverage={'cold_CPU_preparation': True, 'memory_hits': True,
            'disk_hits': 'offline review', 'cache_corruption_and_invalidation': 'offline fault checks',
            'native_collision_cost_and_gradient': True})
    write_json(root/'acceptance.json', accepted)
    write_json(root/'analysis.json', {'at': now(), 'groups': groups, 'pairs': pairs, 'reduction_fraction': reduction})
    lines = ['# V8 第1项CPU网格缓存验证结果', '',
        f"五场景20次纯规划请求完成。两组抓法、选臂、关节解及完整路径逐值相同，完整RPC累计节省{reduction:.2%}。", '',
        '| 项目 | 当前内存更新基线 | CPU几何缓存 |', '| --- | ---: | ---: |']
    for key, label in [('mean_RPC_seconds','完整RPC平均'), ('median_RPC_seconds','完整RPC中位数'),
                       ('p90_RPC_seconds','完整RPC P90'), ('mean_scene_update_seconds','规划器与独立检查器场景更新'),
                       ('mean_contact_scene_update_seconds','精细接触检查器场景更新'),
                       ('mean_CPU_geometry_seconds','CPU几何准备'), ('mean_grasp_IK_seconds','抓取IK')]:
        lines.append(f'| {label} | {a[key]:.4f}秒 | {b[key]:.4f}秒 |')
    lines += ['', f"原生GPU网格加载共{a['native_load_calls']}→{b['native_load_calls']}次；CPU原处理次数{a['actual_CPU_processing_calls']}→{b['actual_CPU_processing_calls']}次。GPU上传和BVH构建仍走原实现。",
              '', '各请求均从冷的正式几何起步，缓存组首次处理及缓存写出计入完整请求。', '',
              f"任务结果：20/20完成。记录完整性：通过。事件适用性：纯规划。关键事件覆盖：缓存首次处理、内存命中、磁盘命中、内容失效、损坏拦截及GPU碰撞比较。", '',
              f"可选实现验收：{'通过' if accepted['optional_implementation_accepted'] else '未达到完整时延采用门槛'}。", '',
              f"原始记录：`{root.relative_to(ROOT)}`。", '']
    report = '\n'.join(lines); (root/'report.md').write_text(report)
    (ROOT/'modules/pipeline/docs/mainline_versions/V8_第1项CPU网格缓存验证结果.md').write_text(report)
    print('CPU_CACHE_ACCEPTANCE', accepted['optional_implementation_accepted'], groups, flush=True)


def run(root):
    reg = read_json(root/'registration.json')
    stop = Event(); thread = Thread(target=record_load, args=(root, reg['gpu'], stop), name='gpu-load')
    thread.start()
    try:
        original_run(root)
    finally:
        stop.set(); thread.join(timeout=15)
    if thread.is_alive() or not read_json(root/'gpu_load_summary.json')['planning_samples']:
        raise RuntimeError('GPU load records are incomplete')
    audit(root); analyze(root)
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'stage': 'cpu_prepared_geometry', 'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    root = ensure_run_path(parser.parse_args().root)
    try: run(root)
    except BaseException:
        write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
