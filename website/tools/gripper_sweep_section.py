"""Publish the two V7 gripper/search compute comparisons as standalone files."""
import shutil
import math

from arx_project.layout import ROOT, read_json, sha256


def latency_tables(evidence, postprocessing):
    from modules.pipeline.experiments.gripper_sweep_optimization.report import average
    a, b = evidence['phases']
    old = a['summaries']['legacy']
    current = b['summaries']['batched']
    post = [r['new_complete_postprocessing_seconds'] for r in postprocessing['cases']]
    low, high = min(post), max(post)
    post_range = f'{low:.1f}～{high:.1f}秒'

    def estimated_range(summary):
        mean = summary['total_seconds']/summary['requests']
        return f"约{math.floor(mean+low)}～{math.ceil(summary['median_seconds']+high)}秒"

    lines = ['## 从抓姿输出到准备执行的耗时', '',
        '从GraspGen输出原始抓姿开始，经过候选后处理、共享交接、双臂搜索、IK、碰撞检查和路径规划，'
        '结束于轨迹准备好、可以下发执行。模型已加载并预热。', '',
        '| 环节 | 本轮优化前 | 当前优化后 |', '| --- | ---: | ---: |',
        f'| 候选碰撞筛选、宽深修正和NMS | {post_range} | {post_range} |',
        f"| 候选交接与完整规划，平均值 | {old['total_seconds']/old['requests']:.2f}秒 | "
        f"{current['total_seconds']/current['requests']:.2f}秒 |",
        f"| 候选交接与完整规划，中位数 | {old['median_seconds']:.2f}秒 | {current['median_seconds']:.2f}秒 |",
        f'| 原始抓姿输出到准备执行，分项估算 | {estimated_range(old)} | {estimated_range(current)} |', '',
        '规划耗时来自五个场景各两次的实测；后处理耗时来自苹果、甜甜圈和剪刀三个样本。'
        '合计时间按这两部分估算。', '',
        '### 两轮优化分别省了多少时间', '',
        '| 修改 | 整次规划平均耗时，优化前 → 优化后 | 累计降幅 |', '| --- | ---: | ---: |']
    for phase, label in [(a,'开合爪改为批量碰撞校验'), (b,'加快候选宽度切换时的配置编解码')]:
        sa, sb = [phase['summaries'][mode] for mode in ['legacy','batched']]
        lines.append(f"| {label} | {sa['total_seconds']/sa['requests']:.2f} → "
                     f"{sb['total_seconds']/sb['requests']:.2f}秒 | {phase['review']['total_saving_fraction']:.1%} |")
    lines += ['', '两轮分别做同期对照，每组10次完整规划。', '',
        '### 当前秒级耗时花在哪里', '',
        '以下为最新优化版10次完整规划的平均值。各项已扣除重复包含的子调用，可以相加。', '',
        '| 环节 | 平均耗时 | 具体在做什么 |', '| --- | ---: | --- |']
    stages = [
        ('机器人与场景绑定', ['world_and_robot_bind'], '候选宽度变化时，更新机器人和碰撞场景'),
        ('开合爪校验', ['gripper_opening','gripper_narrowing'], '对全部开度采样计算碰撞球，再批量检查'),
        ('抓取IK与终点几何检查', ['grasp_IK','grasp_endpoint'], '检查多个候选能否到达、终点是否碰撞'),
        ('接触几何更新', ['contact_validator_update'], '更新夹爪与目标接触检查所需的几何'),
        ('路径搜索＋轨迹优化', ['path_PRM','trajectory_approach','trajectory_pregrasp'], '搜索路径，优化运动轨迹')]
    measured = 0.
    for label, names, description in stages:
        seconds = sum(average(b,'batched',name,'exclusive_seconds') for name in names)
        measured += round(seconds,2)
        lines.append(f'| {label} | {seconds:.2f}秒 | {description} |')
    mean = current['total_seconds']/current['requests']
    lines += [f'| 其余IK、交接、调度与记录 | {round(mean,2)-measured:.2f}秒 | 剩余请求工作 |',
        f'| 合计 | {mean:.2f}秒 | 候选交接与完整规划 |', '',
        f'候选后处理另需约{post_range}。下一步优先让固定场景只绑定一次，'
        '候选切换时只更新夹爪开度相关数据，并复用接触几何。', '']
    return lines


def section(website, site):
    active_path = ROOT/'tmp/gripper_sweep_optimization/active.json'
    if not active_path.exists():
        return '', None, []
    active = read_json(active_path)
    if active['phase'] != 'complete':
        return '', None, []
    root = (ROOT/active['candidate_run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/gripper_sweep_optimization'):
        raise ValueError('Unexpected sweep comparison root')
    snapshot = root/'publication_snapshot.json'
    evidence = read_json(snapshot)
    if not evidence['task_complete'] or not evidence['record_integrity']:
        raise ValueError('Sweep comparison is not accepted')

    def attach(source, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise RuntimeError('Sweep publication must use independent copies')
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError('Sweep publication copy differs')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    data_url = attach(snapshot, 'sam3-v7-gripper-sweep-results.json')
    document = ROOT/'modules/pipeline/docs/mainline_versions/V7_开合爪批量校验与候选搜索优化.md'
    report_url = attach(document, 'sam3-v7-gripper-sweep-report.md')
    postprocessing = ROOT/'tmp/graspgenx_native_postprocessing/spatial69_20261127_01/integration.json'
    post_url = attach(postprocessing, 'sam3-v7-postprocessing-runtime-results.json')
    from modules.pipeline.experiments.gripper_sweep_optimization.report import timing_table, average
    a, b = evidence['phases']
    lines = latency_tables(evidence,read_json(postprocessing))
    lines += [f'[三个样本的完整后处理耗时]({post_url}) · [规划分项计时与配对核验]({data_url})', '',
        '## 开合爪与候选搜索的计算优化', '',
        '原来每改变0.5毫米开度就完整绑定一次机器人和场景。现在保留全部开度采样，'
        '先生成每个开度的原生碰撞球，再批量检查。左臂89个开爪采样完整保留。', '',
        '第一轮比较逐采样绑定和批量校验：', '']
    lines += timing_table(a)
    lines += ['', f"完整规划累计省时{a['review']['total_saving_fraction']:.1%}。双臂开爪校验的平均耗时"
              f"由{average(a,'legacy','gripper_opening'):.2f}秒降到{average(a,'batched','gripper_opening'):.2f}秒。", '',
        '第二轮两组都使用批量校验，只加快候选宽度变化时的YAML配置编解码：', '']
    lines += timing_table(b)
    lines += ['', f"第二轮累计再省{b['review']['total_saving_fraction']:.1%}。两轮分别统计，每组均为五个场景各两次规划。", '',
        '40次规划全部通过。抓姿、选臂、开度、关节终点和完整预抓取/接近轨迹完全一致；'
        '258个开合爪采样的碰撞球坐标、半径及碰撞判断一致，中途碰撞能正常拦下。', '',
        '候选搜索仍有三个优化方向：固定场景只绑定一次、候选间复用接触几何、'
        '让批量IK支持每个候选自己的夹爪宽度。详细分项计时和逐场景结果见报告。', '',
        f'[详细方法、分项计时与后续优化]({report_url}) · [完整配对结果与核验]({data_url})', '']
    status = '开合爪与候选配置优化已完成'
    batch = {'id': root.name, 'title': '开合爪批量校验与候选搜索计算优化', 'date': '2026-11-27',
             'status': status, 'summary': f"40/40规划通过；批量开合爪累计省{a['review']['total_saving_fraction']:.1%}，"
                 f"配置编解码再省{b['review']['total_saving_fraction']:.1%}。",
             'outputs': [p['run'] for p in evidence['phases']],
             'artifacts': [{'label': '方法与结果', 'url': report_url}, {'label': '配对数据', 'url': data_url}]}
    return '\n'.join(lines), status, [batch]
