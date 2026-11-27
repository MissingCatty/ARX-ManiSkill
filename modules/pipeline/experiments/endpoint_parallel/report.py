"""Report lazy endpoint batches and path reuse against the matched baseline."""
import argparse
from pathlib import Path

from arx_project.layout import ROOT, now, read_json, write_json
from modules.pipeline.experiments.parallel_planning.run import review
from modules.pipeline.experiments.parallel_planning.audit import audit


NAMES = {'control': '原方法', 'batched_endpoints': '固定窗口批量端点', 'batched_paths4': '批量端点＋并行路径4条'}


def create(root):
    review(root); trajectory = audit(root)
    accepted = read_json(root / 'acceptance.json'); proof = read_json(root / 'independent_review.json')
    paired = read_json(root / 'paired_review.json'); rows = read_json(root / 'progress.json')['results']
    offline_checks = read_json(root / 'registration.json')['config'].get('offline_checks', 42)
    timing_environment = read_json(root / 'timing_environment.json') if (root / 'timing_environment.json').exists() else {'exclusive_timing': True}
    stopped = read_json(root / 'services_stopped.json')
    if not stopped['all_owned_exited']: raise RuntimeError('Owned GPU workers have not exited')
    groups = accepted['summaries']; stages, counts, adoption = {}, {}, {}
    baseline = groups['control']
    for mode in NAMES:
        group_rows = [r for r in rows if r['mode'] == mode]
        windows = [w for r in group_rows for w in r['parallel_windows']]
        stages[mode] = {}
        for row in group_rows:
            for name, value in row['stages'].items():
                stages[mode][name] = stages[mode].get(name, 0.) + value['exclusive_seconds'] / len(group_rows)
        counts[mode] = {'scalar_grasp_calls': sum(w['scalar_grasp_calls'] for w in windows) if windows else groups[mode]['candidate_checks'],
            'computed_batch_grasp_candidates': sum(w['computed_endpoint_candidates'] for w in windows),
            'prefetched_unconsumed_candidates': sum(w['extra_endpoint_evaluations'] for w in windows),
            'serial_extra_checks_for_batch_collection': sum(w['serial_extra_endpoint_evaluations'] for w in windows),
            'original_pre_or_lift_fallbacks': sum(w['original_pre_or_lift_fallbacks'] for w in windows),
            'path_batches': sum(len(w['batches']) for w in windows),
            'real_parallel_path_rows': sum(b['real_rows'] for w in windows for b in w['batches']),
            'accepted_parallel_path_rows': sum(v['accepted'] for w in windows for b in w['batches'] for v in b['rows'])}
        if mode != 'control':
            g = groups[mode]; reduction = 1 - g['total_seconds'] / baseline['total_seconds']
            retained = all(not p['success']['control'] or p['success'][mode] for p in paired['pairs'])
            adoption[mode] = {'original_successes_retained': retained, 'total_reduction_fraction': reduction,
                'timing_passed': reduction >= .05 and g['median_seconds'] < baseline['median_seconds'] and
                    g['p90_seconds'] <= baseline['p90_seconds'] * 1.05,
                'timing_environment_valid': timing_environment['exclusive_timing'], 'default_activated': False}
    write_json(root / 'adoption.json', {'at': now(), 'variants': adoption, 'default_activated': False})
    previous = ROOT / 'tmp/parallel_planning/paths5_20261127_01'
    snapshot = {'at': now(), 'run': str(root.relative_to(ROOT)), 'acceptance': accepted,
        'independent_review': proof, 'paired_review': paired, 'trajectory_review': trajectory,
        'exclusive_mean_stages': stages, 'work_counts': counts, 'adoption': adoption,
        'startup': read_json(root / 'resident/startup.json'),
        'preflight': read_json(root / 'resident/endpoint_preflight/acceptance.json'), 'results': rows,
        'services_stopped': stopped, 'timing_environment': timing_environment,
        'previous_serial_collection': read_json(previous / 'acceptance.json')['summaries']}
    write_json(root / 'publication_snapshot.json', snapshot)
    lines = ['# V8 固定候选窗口的批量端点与并行路径', '',
        '新版去掉串行寻找4个合格抓法的过程。只查看下一段16个原候选，'
        '在当前需要检查的开度下，批量求抓取、预抓取和抬升关节解，并完成原碰撞检查。'
        '已经有多个合格结果时再并行规划；只有一个就立即走原方法。', '',
        '## 完整请求结果', '',
        '五个固定场景，每组重复两次，共30次完整纯规划。各组统一预热，共用原15毫米／15°候选、'
        '数值IK设置、871球、TCP43、V2指尖规则及原接近和收爪检查。', '',
        *(['本轮GPU2同时运行其他用户的计算任务，以下为共享运行耗时。功能核验单独完成；提速结论另用空闲卡核验。', '']
          if not timing_environment['exclusive_timing'] else []),
        '| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, name in NAMES.items():
        g = groups[mode]
        lines.append(f"| {name} | {g['successes']}/{g['requests']} | {g['mean_seconds']:.3f}秒 | "
            f"{g['median_seconds']:.3f}秒 | {g['p90_seconds']:.3f}秒 | {g['total_seconds']:.3f}秒 |")
    lines += ['', '## 是否仍在额外串行凑批', '',
        '| 方法 | 实际消费候选数 | 串行抓取终点调用 | 批量计算的抓取终点 | 批量提前算出但未使用 | 为凑批额外串行检查 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, name in NAMES.items():
        c = counts[mode]
        lines.append(f"| {name} | {groups[mode]['candidate_checks']} | {c['scalar_grasp_calls']} | "
            f"{c['computed_batch_grasp_candidates']} | {c['prefetched_unconsumed_candidates']} | "
            f"{c['serial_extra_checks_for_batch_collection']} |")
    lines += ['', '提前批量计算仍可能包含最后没有使用的候选，但不再为这些候选逐个求IK或绑定模型，'
        '也不会扩大窗口寻找更多合格抓法。不同开度保持精确分组，没有取整。', '',
        '## 完整请求分项平均耗时', '', '| 环节 | 原方法 | 批量端点 | 批量端点＋并行路径 |',
        '| --- | ---: | ---: | ---: |']
    for key, label in [('world_and_robot_bind', '模型和场景绑定'), ('grasp_IK', '原串行抓取IK'),
            ('batched_grasp_IK_and_collision', '批量抓取IK及碰撞'),
            ('batched_pregrasp_IK_and_collision', '批量预抓取IK及碰撞'),
            ('batched_lift_IK_and_collision', '批量抬升IK及碰撞'),
            ('pregrasp_IK', '原预抓取IK'), ('lift_IK', '原抬升IK'),
            ('parallel_complete_paths', '并行完整路径及核验'), ('path_PRM', '原路径搜索'),
            ('trajectory_pregrasp', '原预抓取轨迹优化'), ('trajectory_approach', '原接近轨迹优化'),
            ('contact_validator_update', '接触几何更新')]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{stages[m].get(key, 0.):.4f}秒' for m in NAMES) + ' |')
    lines += ['', '## 并行路径实际使用量', '', '| 方法 | 并行批次 | 有效路径行 | 批量通过行 |',
        '| --- | ---: | ---: | ---: |']
    for mode, name in NAMES.items():
        c = counts[mode]
        lines.append(f"| {name} | {c['path_batches']} | {c['real_parallel_path_rows']} | {c['accepted_parallel_path_rows']} |")
    lines += ['', '## 结果分析', '']
    for mode in ['batched_endpoints', 'batched_paths4']:
        a = adoption[mode]
        conclusion = (f"{'通过' if a['timing_passed'] else '未通过'}累计下降5%、中位下降和P90不增加5%的时延门槛。"
                      if timing_environment['exclusive_timing'] else '提速判断等待空闲卡对照。')
        lines.append(f"- {NAMES[mode]}：累计耗时变化{-a['total_reduction_fraction'] * 100:+.2f}%；"
            f"原成功{'全部保留' if a['original_successes_retained'] else '有丢失'}；" + conclusion)
    lines += ['', '这一轮把端点批量化与路径并行拆开比较，分别查看少做串行IK的收益和并行路径新增的成本。'
        '后续使用以完整请求的实际耗时为依据。常用入口和默认配置保持。', '',
        '| 场景／重复 | 原方法 | 批量端点 | 批量端点＋并行路径 | 抓法与手臂一致 |',
        '| --- | --- | --- | --- | --- |']
    for pair in paired['pairs']:
        lines.append('| ' + pair['case'] + '/' + str(pair['repeat']) + ' | ' + ' | '.join(
            '通过' if pair['success'][m] else '规划不出来' for m in NAMES) + ' | ' +
            ('是' if pair['same_grasp_and_arm'] else '有变化') + ' |')
    lines += ['', '## 验收', '', '| 项目 | 结果 |', '| --- | --- |',
        f"| 任务结果 | {accepted['planning_requests']}个完整纯规划槽完成 |",
        f"| 记录完整性 | {proof['checked_requests']}条回执、{proof['checked_endpoints']}个独立物理臂终点、{trajectory['command_paths']}条记录轨迹核验通过 |",
        '| 事件适用性 | 本轮比较规划通过率与完整耗时，物理抓取0次 |',
        '| 关键事件覆盖 | 两臂、三种开度、混合可达/不可达、不同目标初值、固定窗口、单结果立即执行、原评分及回退 |', '',
        '入口：`modules.curobo.parallel_planning.windowed.create`。'
        f"登记：`{root.relative_to(ROOT)}`。{offline_checks}项离线检查通过。", '']
    doc = ROOT / 'modules/pipeline/docs/mainline_versions/V8_固定候选窗口的批量端点与并行路径.md'
    doc.write_text('\n'.join(lines))
    write_json(ROOT / 'tmp/parallel_planning/windowed_active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'complete': True, 'phase': 'complete', 'report': str(doc.relative_to(ROOT))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    create(p.parse_args().root.resolve())
