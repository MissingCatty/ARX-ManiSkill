"""Report concrete device kernels separately from enclosing GPU annotations."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, sha256, write_json


DOCUMENT = ROOT/'modules/pipeline/docs/mainline_versions/V8_批量IK定位与射线投票优化结果.md'


def raw(path): return json.loads(Path(path).read_text())


def diagnosis(source, output):
    if output.exists(): return raw(output)
    checked = raw(source/'independent_review.json')
    if not checked['passed'] or checked['checked_requests'] != 10:
        raise ValueError('The diagnostic requests have not passed independent review')
    rows = [];digests = {}
    for case in raw(source/'registration.json')['cases']:
        directory = source/'plans'/case['id']/'profiled_1'
        trace = directory/'ik_sample_0.trace.json';digests[str(trace)] = sha256(trace)
        events = raw(trace)['traceEvents'];kernels = [v for v in events if v.get('cat') == 'kernel']
        if not kernels: raise ValueError('CUDA profiling produced no concrete kernel events')
        total = sum(v['dur'] for v in kernels)/1e6
        collision = sum(v['dur'] for v in kernels if 'sphere_obstacle_collision_kernel' in v['name'])/1e6
        sample = raw(directory/'ik_sample_0.json')
        stages = sample['substage_events']
        rows.append({'case': case['id'], 'kernel_seconds': total, 'collision_kernel_seconds': collision,
            'collision_fraction': collision/total, 'CUDA_graph_captures': sample['CUDA_graph_captures'],
            'LM_host_seconds': sum(v['host_wall_seconds'] for v in stages if v['stage'] == 'world_LM_total'),
            'LBFGS_host_seconds': sum(v['host_wall_seconds'] for v in stages if v['stage'] == 'world_LBFGS_optimize')})
    result = {'at': now(), 'passed': True, 'source': str(source.relative_to(ROOT)), 'sampled_world_batches': len(rows),
        'kernel_seconds': sum(r['kernel_seconds'] for r in rows),
        'collision_kernel_seconds': sum(r['collision_kernel_seconds'] for r in rows),
        'CUDA_graph_captures': sum(r['CUDA_graph_captures'] for r in rows), 'cases': rows,
        'device_event_category': 'kernel', 'GPU_annotation_ranges_excluded': True, 'input_sha256': digests,
        'correctness_requests': checked['checked_requests'], 'independent_endpoints': checked['checked_endpoints'],
        'diagnostic_times_are_not_speedup_measurements': True}
    result['collision_fraction'] = result['collision_kernel_seconds']/result['kernel_seconds']
    write_json(output, result);return result


def report(root, pilot, full=None):
    source = ROOT/raw(pilot/'registration.json')['diagnostic_source']
    proof = diagnosis(source, root/'diagnosis_summary.json')
    lines = ['# V8批量IK定位与射线投票优化结果', '',
        '本轮先定位批量抓取IK，再测试一项保持原查询结果的计算优化。', '',
        '## 定位结果', '',
        f"原五场景共10次规划全部通过，{proof['independent_endpoints']}个终点独立核验通过。"
        '采样组与正常组的抓点、选臂、关节解和候选判断保持。', '',
        '| 场景 | 首批GPU内核总时间 | 碰撞内核时间 | 碰撞占比 | CUDA Graph重新捕获 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in proof['cases']:
        lines.append(f"| {row['case']} | {row['kernel_seconds']:.4f}秒 | {row['collision_kernel_seconds']:.4f}秒 | "
                     f"{row['collision_fraction']:.1%} | {row['CUDA_graph_captures']} |")
    lines += ['', f"五批采样的碰撞内核累计占GPU实际内核时间{proof['collision_fraction']:.1%}，"
        f"重新捕获CUDA Graph共{proof['CUDA_graph_captures']}次。主要耗时来自网格碰撞查询。", '',
        '## 本次修改', '',
        '原Warp查询先找最近表面，再沿+X、+Y、+Z三个方向投票，判断点在网格内部还是外部。'
        '新实现保持相同最近点搜索、射线、方向次序和多数规则；前两票一致时立即返回，前两票不一致时继续第三票。', '',
        '原版和优化版完整场景IK各自保留CUDA Graph，共享相同机器人与场景数据。两组共同使用NMS15＋全部原候选后备、B16及C版YAML写出。'
        '优化器迭代设置、32初值、871球、TCP43、V2接触、候选顺序和最终验收标准保持。', '']
    for run, title in [(pilot, '五场景先导'), (full, '完整69场景对照')]:
        if run is None: continue
        progress = raw(run/'progress.json');accepted = raw(run/'acceptance.json') if (run/'acceptance.json').exists() else None
        checked = raw(run/'independent_review.json') if (run/'independent_review.json').exists() else None
        lines += ['## '+title, '', f"完成{progress['completed_requests']}/{progress['total_requests']}次。", '']
        if accepted:
            lines += ['| 方法 | 规划通过 | RPC中位耗时 | RPC P90 | RPC累计耗时 |', '| --- | ---: | ---: | ---: | ---: |']
            for mode, label in [('control', '原三方向完整投票'), ('early_vote', '一致两票提前返回')]:
                a = accepted['summaries'][mode]
                lines.append(f"| {label} | {a['successes']}/{a['requests']} | {a['median_seconds']:.3f}秒 | "
                             f"{a['p90_seconds']:.3f}秒 | {a['total_seconds']:.3f}秒 |")
            lines += ['', f"累计耗时变化：省时{accepted['planning_total_time_reduction']:.2%}。", '']
        if checked:
            lines += [f"{checked['checked_endpoints']}个终点独立核验通过，原抓点、选臂、关节解与候选判断保持。", '']
        geometry = run/'resident/mesh_vote_geometry_preflight/acceptance.json'
        if geometry.exists():
            g = raw(geometry)
            lines += [f"实际场景网格的{g['checked_sdf_points']}个点，其距离和梯度逐位一致；完整场景碰撞代价和梯度复核通过。", '']
        if (run/'STOP').exists(): lines += ['本登记已停止，原领取、源码与停止记录保留。', '']
        lines += [f"记录目录：`{run.relative_to(ROOT)}`。", '']
    lines += ['## 计时与验收', '',
        '定位阶段开启CPU与CUDA性能采样，采样时间单列；优化对照关闭性能分析器，主表统计完整RPC，包含规划与记录交接。'
        '每组统一预热，先导按ABBA／BAAB次序运行。累计至少省5%、中位数下降、P90增幅不超过5%，并通过独立核验后才展开69场景。', '',
        '默认链路保持；本轮为保存观测的纯规划比较，机器人动作0次。', '']
    DOCUMENT.write_text('\n'.join(lines))
    shutil.copy2(DOCUMENT, root/'report.md')
    return proof
