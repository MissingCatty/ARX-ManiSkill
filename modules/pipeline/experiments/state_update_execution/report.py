"""Keep execution outcomes and latency evidence separate from prior planning-only results."""
from arx_project.layout import ROOT, now, read_json, write_json

REPORT = ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存状态更新完整抓取测试.md'


def report(root):
    reg = read_json(root / 'registration.json'); accepted = read_json(root / 'acceptance.json')
    progress = read_json(root / 'progress.json'); groups = accepted['groups']
    text = ['# V8 内存状态更新完整抓取测试', '',
        '使用同一组74个配对场景，新旧方法每场景各一次，共148个登记槽。69个场景各组重新复位仿真并规划抓取；'
        '5个原站位前提未满足的场景保留为未执行，不把它们写成实际空抓或规划求解失败。', '',
        '两组沿用同一份冻结头部RGB-D、SAM3分割和15毫米／15度NMS后的候选。'
        '本轮实际执行开爪、预抓取、按实测状态重规划接近、闭合、按实测状态重规划抬升与3秒保持。'
        '成功判据沿用目标最低保持抬升至少1厘米、双指有效接触、各段到位且无禁止接触。', '',
        '| 方法 | 完整结束 | 程序中断 | 初始规划通过 | 实际执行 | 抓取成功 | 前提未满足未执行 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode, label in [('baseline', '原方法'), ('optimized', '内存状态更新')]:
        g = groups[mode]
        text.append(f"| {label} | {g['completed']}/74 | {g['interrupted']} | {g['planning_passes']}/74（{g['planning_passes']/74:.1%}） | "
            f"{g['physical_attempts']} | {g['successes']}/74（{g['successes']/74:.1%}） | {g['common_precondition_unexecuted']} |")
    text += ['', '## 各段完整规划耗时', '', '| 环节 | 原方法平均／中位／P90 | 内存更新平均／中位／P90 |', '| --- | ---: | ---: |']
    def fmt(d):
        return '未发生' if not d['samples'] else f"{d['mean_seconds']:.3f}／{d['median_seconds']:.3f}／{d['p90_seconds']:.3f}秒（{d['samples']}次）"
    timing = accepted['matched_phase_timings']
    for stage, label in [('initial', '初始完整规划'), ('approach', '到预抓取后重规划接近'), ('lift', '实测闭合后重规划抬升')]:
        text.append(f"| {label} | {fmt(timing['baseline']['timings'][stage])} | {fmt(timing['optimized']['timings'][stage])} |")
    text.append(f"| 每场景全部规划合计 | {fmt(timing['baseline']['all_planning_per_scene'])} | {fmt(timing['optimized']['all_planning_per_scene'])} |")
    if accepted.get('cross_restart_timing_case'):
        text += ['', f"主耗时表使用同阶段配对。{accepted['cross_restart_timing_case']}的一条初始规划从停止前承接，"
            '该场景仍完整计入任务结果，跨阶段计时另存原始统计。']
    if accepted.get('interrupted_timing_cases_excluded'):
        text += ['', '主耗时表使用两组同一场景、同一实际发生阶段的配对；中断场景单列，不计入配对时延：'
            + '、'.join(accepted['interrupted_timing_cases_excluded']) + '。']
    text += ['', '## 失败分布', '', '| 原因 | 原方法 | 内存更新 |', '| --- | ---: | ---: |']
    reasons = sorted(set(groups['baseline']['failure_reasons']) | set(groups['optimized']['failure_reasons']))
    for reason in reasons:
        text.append(f"| {reason} | {groups['baseline']['failure_reasons'].get(reason,0)} | {groups['optimized']['failure_reasons'].get(reason,0)} |")
    text += ['', '## 核验与对照', '', '| 项目 | 结果 |', '| --- | --- |',
        f"| 任务结果 | 148个登记槽已处理；{accepted['completed_slots']}完整、{accepted['interrupted_slots']}中断、0未开始 |",
        f"| 记录完整性 | {accepted['key_event_coverage']['decoded_videos']}段录像完整解码；完整项命令与回执通过，中断项停止记录单列 |",
        '| 事件适用性 | 发生过的接近、闭合、抬升、保持按实际记录核验；未执行项单列 |',
        f"| 关键事件覆盖 | {accepted['key_event_coverage']} |",
        f"| 初始抓法、手臂与关节相同 | {accepted['all_initial_selections_and_joints_equal']} |",
        f"| 初始命令轨迹相同 | {accepted['all_initial_paths_equal']} |", '',
        f"原成功但新失败：{accepted['baseline_success_losses']}。", '',
        f"新成功但原失败：{accepted['optimized_success_gains']}。", '',
        '## 逐场景结果', '', '| 场景 | 原方法 | 内存更新 |', '| --- | --- | --- |']
    rows = {(r['id'], r['variant']): r for r in progress['results']}
    def result(case, mode):
        r = rows[(case, mode)]
        if r['common_precondition_unexecuted']: return '前提未满足，未执行'
        if r.get('interrupted'): return '程序中断，未重复执行'
        return '抓取成功' if r['success'] else r['reason']
    for case in reg['cases']: text.append(f"| {case['id']} | {result(case['id'],'baseline')} | {result(case['id'],'optimized')} |")
    text += ['', f"运行记录：`{root.relative_to(ROOT)}`。所有视频的路径保存在本轮acceptance.json。", '']
    rendered = '\n'.join(text); REPORT.write_text(rendered); (root / 'report.md').write_text(rendered)
    write_json(root / 'publication_snapshot.json', {'at': now(), 'run': str(root.relative_to(ROOT)),
        'acceptance': accepted, 'progress': progress, 'startup': read_json(root / 'resident/startup.json'),
        'services_stopped': read_json(root / 'services_stopped.json'),
        'preflight': read_json(root / 'preflight/all_ready.json')})
    return REPORT
