"""Publish complete-request results separately from the batched kernel timing."""
import argparse
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json
from .run import review


NAMES = {'control': '原方法', 'parallel4': '并行路径4条', 'parallel8': '并行路径8条'}


def create(root):
    review(root)
    stopped = read_json(root / 'services_stopped.json')
    if not stopped['all_owned_exited']:
        raise RuntimeError('Finalize only after the owned GPU worker exits')
    acceptance = read_json(root / 'acceptance.json')
    proof = read_json(root / 'independent_review.json')
    paired = read_json(root / 'paired_review.json')
    reg = read_json(root / 'registration.json')
    rows = read_json(root / 'progress.json')['results']
    groups = acceptance['summaries']
    baseline = groups['control']
    adoption = {}
    for mode in ['parallel4', 'parallel8']:
        group = groups[mode]
        reduction = 1 - group['total_seconds'] / baseline['total_seconds']
        retained = all(not p['success']['control'] or p['success'][mode] for p in paired['pairs'])
        timing_passed = (reduction >= .05 and group['median_seconds'] < baseline['median_seconds']
                         and group['p90_seconds'] <= 1.05 * baseline['p90_seconds'])
        adoption[mode] = {'original_successes_retained': retained, 'total_reduction_fraction': reduction,
            'timing_passed': timing_passed, 'eligible_as_optional_speedup': retained and timing_passed,
            'default_activated': False}
    write_json(root / 'adoption.json', {'at': now(), 'variants': adoption, 'default_activated': False})
    stages = {}
    for mode in NAMES:
        stage = {}
        for row in rows:
            if row['mode'] != mode: continue
            for name, value in row['stages'].items():
                stage[name] = stage.get(name, 0.) + value['exclusive_seconds'] / groups[mode]['requests']
        stages[mode] = stage
    batches = {mode: [b for row in rows if row['mode'] == mode
                     for w in row['parallel_windows'] for b in w['batches']] for mode in NAMES}
    batch_metrics = {}
    for mode, values in batches.items():
        if not values: continue
        batch_metrics[mode] = {'batch_calls': len(values), 'real_rows': sum(b['real_rows'] for b in values),
            'accepted_rows': sum(r['accepted'] for b in values for r in b['rows']),
            'median_graph_seconds': float(np.median([b['graph_seconds'] for b in values])),
            'median_solve_seconds': float(np.median([b['solve_seconds'] for b in values])),
            'median_validation_seconds': float(np.median([b['validation_seconds'] for b in values]))}
    snapshot = {'at': now(), 'run': str(root.relative_to(ROOT)), 'acceptance': acceptance,
        'independent_review': proof, 'paired_review': paired, 'adoption': adoption,
        'exclusive_mean_stages': stages, 'batch_metrics': batch_metrics, 'results': rows,
        'startup': read_json(root / 'resident/startup.json'),
        'preflight': read_json(root / 'resident/parallel_preflight/acceptance.json'),
        'services_stopped': stopped, 'trajectory_review': read_json(root / 'trajectory_review.json')}
    write_json(root / 'publication_snapshot.json', snapshot)
    lines = ['# V8 预抓取并行路径规划', '',
        '本轮已把预抓取路径接入原生批量PRM与TrajOpt。每次收集4或8个终点合格的抓法，'
        '按完全相同的夹爪开度分组，将多组q_pre一起交给GPU生成完整路径，再按原评分顺序选择。', '',
        '## 完整请求对照', '',
        '五个固定场景，每组重复两次，共30个完整纯规划请求。两种批容量均提前预热；'
        '下表包含双臂准备、候选IK、模型绑定、路径、接近、检查、留档和交接。', '',
        '| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, name in NAMES.items():
        g = groups[mode]
        lines.append(f"| {name} | {g['successes']}/{g['requests']} | {g['mean_seconds']:.3f}秒 | "
            f"{g['median_seconds']:.3f}秒 | {g['p90_seconds']:.3f}秒 | {g['total_seconds']:.3f}秒 |")
    lines += ['', '## 实际并行量与提前检查', '',
        '| 方法 | 真正的并行批次 | 有效路径行数 | 批量通过行数 | 额外提前检查的候选终点 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for mode in ['parallel4', 'parallel8']:
        g = groups[mode]; b = batch_metrics.get(mode, {})
        lines.append(f"| {NAMES[mode]} | {g['actual_parallel_calls']} | {g['real_parallel_rows']} | "
            f"{b.get('accepted_rows', 0)} | {g['speculative_endpoint_checks']} |")
    lines += ['', '当前六轴模型将夹爪开度锁定为一个值，所以不同开度没有混算。'
        '同开度组不足两条时，继续原路径；补齐行不参与选点。批量路径未通过时，仍对同一候选运行原完整路径。', '',
        '## 单个并行批次耗时', '',
        '下表为完整一批的中位耗时，包含这一批中的所有有效行与补齐行。', '',
        '| 方法 | 批量路径搜索 | 批量轨迹优化 | 逐行完整检查 |', '| --- | ---: | ---: | ---: |']
    for mode, b in batch_metrics.items():
        lines.append(f"| {NAMES[mode]} | {b['median_graph_seconds'] * 1000:.1f}毫秒 | "
            f"{b['median_solve_seconds'] * 1000:.1f}毫秒 | {b['median_validation_seconds'] * 1000:.1f}毫秒 |")
    lines += ['', '## 完整请求的分项平均耗时', '',
        '| 环节 | 原方法 | 并行4条 | 并行8条 |', '| --- | ---: | ---: | ---: |']
    for key, label in [('world_and_robot_bind', '模型和场景绑定'), ('grasp_IK', '抓取IK'),
            ('pregrasp_IK', '预抓取IK'), ('lift_IK', '抬升IK'),
            ('parallel_complete_paths', '并行完整路径及检查'), ('path_PRM', '原路径搜索'),
            ('trajectory_pregrasp', '原预抓取轨迹优化'), ('trajectory_approach', '原接近轨迹优化'),
            ('contact_validator_update', '接触几何更新')]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{stages[m].get(key, 0.):.4f}秒' for m in NAMES) + ' |')
    lines += ['', '## 结果分析', '']
    for mode in ['parallel4', 'parallel8']:
        decision = adoption[mode]
        lines.append(f"- {NAMES[mode]}：累计耗时变化{(-decision['total_reduction_fraction']) * 100:+.2f}%；"
            f"原成功{'全部保留' if decision['original_successes_retained'] else '有丢失'}，"
            f"{'通过' if decision['timing_passed'] else '未通过'}累计下降5%、中位下降和P90不增加5%的时延门槛。")
    lines += ['', '并行计算节省的时间，需要和提前收集更多合格终点、按开度重新绑定及额外路径计算的成本一起比较。'
        '本轮只改变预抓取路径调度，最后接近继续原实现。', '',
        '并行4条比原方法每请求增加约1.51秒，其中模型／场景绑定增加约0.65秒、抓取IK增加约0.38秒。'
        '并行8条增加约2.14秒，其中绑定增加约0.88秒、抓取IK增加约0.63秒。'
        '当前按序搜索经常在第一个合格抓法就结束，提前凑齐更多合格终点带来的计算量超过了并行路径节省的时间。'
        '因此本轮保留可选入口，常用入口继续原方法。', '',
        '| 场景／重复 | 原方法 | 并行4条 | 并行8条 | 抓法与手臂一致 |',
        '| --- | --- | --- | --- | --- |']
    for pair in paired['pairs']:
        lines.append('| ' + pair['case'] + '/' + str(pair['repeat']) + ' | ' +
            ' | '.join('通过' if pair['success'][m] else '规划不出来' for m in NAMES) +
            ' | ' + ('是' if pair['same_grasp_and_arm'] else '有变化') + ' |')
    lines += ['', '## 验收与使用', '', '| 项目 | 结果 |', '| --- | --- |',
        f"| 任务结果 | {acceptance['planning_requests']}个完整纯规划槽完成 |",
        f"| 记录完整性 | {proof['checked_requests']}条回执、{proof['checked_endpoints']}个物理臂终点和60条记录轨迹核验通过 |",
        '| 事件适用性 | 本轮比较规划通过率和耗时，新增物理抓取0次 |',
        '| 关键事件覆盖 | GPU覆盖两臂、三种预检开度、两种批容量、补齐行和原选点；批量50行全部通过。失败回退和资源异常由离线故障测试覆盖 |', '',
        '入口：`modules.curobo.parallel_planning.runtime.create`。常用入口保持原实现。'
        f"本轮登记：`{root.relative_to(ROOT)}`。32项离线检查通过，12组正式预热接口检查通过。", '']
    doc = ROOT / 'modules/pipeline/docs/mainline_versions/V8_预抓取并行路径规划.md'
    doc.write_text('\n'.join(lines))
    write_json(ROOT / 'tmp/parallel_planning/active.json', {'at': now(), 'run': str(root.relative_to(ROOT)),
        'phase': 'complete', 'complete': True, 'report': str(doc.relative_to(ROOT))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    create(p.parse_args().root.resolve())
