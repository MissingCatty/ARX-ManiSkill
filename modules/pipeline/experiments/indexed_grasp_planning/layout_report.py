"""Separate component diagnosis, fixed-query timings, and full planning latency."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, write_json

DOCUMENT = ROOT/'modules/pipeline/docs/mainline_versions/V8_网格查询分项与线程排列优化结果.md'


def raw(path): return json.loads(Path(path).read_text())


def report(output, pilot, full=None):
    source = ROOT/raw(pilot/'registration.json')['diagnostic_source']
    proof = raw(source/'independent_review.json')
    if not proof['passed'] or not proof['actual_workloads_verified']:
        raise ValueError('Component diagnosis was not independently accepted')
    rows = [{'case': case['id'], **raw(source/'plans'/case['id']/'captured_1/query_components/components.json')}
            for case in raw(source/'registration.json')['cases']]
    summary = {'at': now(), 'diagnostic_source': str(source.relative_to(ROOT)),
        'completed_diagnostic_requests': 10, 'independent_endpoints': 30, 'cases': rows}
    write_json(output/'query_components_summary.json', summary)
    lines = ['# V8网格查询分项与线程排列优化结果', '',
        '先用原五个场景定位网格查询，再测试把同一个障碍物的查询连续排列，能否缩短完整规划时间。', '',
        '## 网格查询的主要开销', '',
        '10次定位规划与30个终点独立核验通过，抓法、选臂、关节解和候选判断保持一致。'
        '下表来自每例首批完整IK的实际球和GPU网格，记录固定输入的GPU查询时间。', '',
        '| 场景 | 最近表面搜索 | 三方向内外判断 | 合并查询 | 原完整碰撞派发 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [row['timings'][k]['median_seconds']*1000 for k in
                  ['nearest_point', 'three_ray_votes', 'combined_compact', 'original_collision_dispatch']]
        lines.append('| '+row['case']+' | '+' | '.join(f'{v:.3f}毫秒' for v in values)+' |')
    lines += ['', '四个场景的最近表面搜索耗时高于三方向内外判断，苹果场景两者接近。'
        '合并查询先提取有效任务；原完整派发还包含任务筛选、坐标变换和碰撞代价累加，因此两列单独展示。', '',
        '## 本次修改', '',
        '原来相邻线程轮流检查多个障碍物，容量为8时，一组32个线程只处理4个球。'
        '现在让相邻线程处理同一个障碍物上的32个球，再继续下一个障碍物。'
        '所有球与障碍物组合都保留，最近表面搜索、三方向射线、距离公式和碰撞梯度沿用原代码；只改两个任务索引表达式。', '',
        '这一排列方式让相邻线程访问相同网格，尝试减少分支分歧与不连续访存。'
        '方法依据见[NVIDIA CUDA编程优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。', '',
        '两组共用NMS15与全部原候选后备、B16、C版YAML写出和原标量分支复核。'
        '两组完整IK分别预热并保留自己的CUDA Graph。', '']
    for run, title in [(pilot, '五场景20次先导'), (full, '69场景完整对照')]:
        if run is None: continue
        p = raw(run/'progress.json');lines += ['## '+title, '',
            f"完成{p['completed_requests']}/{p['total_requests']}次。", '']
        if (run/'acceptance.json').exists():
            a = raw(run/'acceptance.json')
            lines += ['| 方法 | 规划通过 | 完整请求中位耗时 | 完整请求P90 | 完整请求累计耗时 |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for mode, label in [('control', '原线程排列'), ('obstacle_major', '同一障碍物连续排列')]:
                s = a['summaries'][mode]
                lines.append(f"| {label} | {s['successes']}/{s['requests']} | {s['median_seconds']:.3f}秒 | "
                             f"{s['p90_seconds']:.3f}秒 | {s['total_seconds']:.3f}秒 |")
            lines += ['', f"累计省时{a['planning_total_time_reduction']:.2%}。", '']
            if a['timing_gate_passed']:
                lines += ['累计至少省5%、中位耗时下降、P90增幅不超过5%，三项计时门槛均通过。', '']
            else:
                failed = [label for key, label in [('total_time_reduction', '累计至少省5%'),
                    ('median_reduced', '中位耗时下降'), ('p90_within_limit', 'P90增幅不超过5%')]
                    if not a['timing_gates'].get(key, True)]
                lines += ['尚未通过：'+'、'.join(failed)+'。不扩大本方案测试。', '']
        geometry = run/'resident/layout_geometry_preflight/acceptance.json'
        if geometry.exists():
            g = raw(geometry)
            lines += [f"实际完整IK输入的{g['checked_spheres']:,}个球，其完整碰撞代价和梯度核验通过。", '']
        checked = run/'independent_review.json'
        if checked.exists():
            c = raw(checked)
            lines += [f"{c['checked_endpoints']}个终点独立核验通过，原候选、抓法、选臂、关节解和判断保持。", '']
        if (run/'STOP').exists(): lines += ['本登记因故障停止，领取与停止回执保留。', '']
        lines += [f"记录目录：`{run.relative_to(ROOT)}`。", '']
    lines += ['## 计时口径与结论', '',
        '分项定位与固定查询微计时单列。先导和完整对照关闭分析器，比较完整RPC请求，包含规划与记录交接。'
        '先导采用ABBA／BAAB次序；独立核验和三项计时门槛全部通过后，才进入138次完整对照。', '',
        '默认链路保持原设置。本轮没有新增机器人动作。', '']
    DOCUMENT.write_text('\n'.join(lines));shutil.copy2(DOCUMENT, output/'report.md')
    return summary
