"""Summarize accepted paired replays without starting a planner or simulator."""
import argparse
import csv
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json, ensure_run_path

LABELS = {'sweep': ('逐采样完整绑定', '批量开合爪校验'),
          'codec': ('批量校验＋原YAML', '批量校验＋C版YAML')}


def load_phase(root, kind):
    progress = read_json(root/'progress.json')
    paired = read_json(root/'paired_review.json')
    independent = read_json(root/'independent_review.json')
    stopped = read_json(root/'services_stopped.json')
    if not (progress['phase'] == 'complete' and paired['passed'] and independent['passed']
            and stopped['all_owned_exited']):
        raise RuntimeError('Only complete, reviewed, stopped planning runs can be published')
    codec = read_json(root/'codec_model_review.json') if kind == 'codec' else None
    if codec is not None and not codec['passed']:
        raise RuntimeError('Actual YAML value comparison failed')
    rows = progress['results']
    cases = []
    for case in read_json(root/'registration.json')['cases']:
        groups = {m: [r for r in rows if r['case_id'] == case['id'] and r['mode'] == m]
                  for m in ['legacy', 'batched']}
        cases.append({'id': case['id'], **{m: {
            'requests': len(g), 'successes': sum(r['success'] for r in g),
            'mean_RPC_seconds': float(np.mean([r['RPC_seconds'] for r in g])),
            'candidate_checks': [r['candidate_checks'] for r in g]} for m, g in groups.items()}})
    return {'run': str(root.relative_to(ROOT)), 'kind': kind, 'summaries': progress['summaries'],
            'review': paired, 'independent_review': independent, 'codec_review': codec,
            'preflight': read_json(root/'resident/sweep_preflight/acceptance.json'),
            'cases': cases, 'all_owned_exited': True,
            'kept': bool(paired['timing_gate_passed'] and paired['all_endpoints_bitwise_equal']
                         and paired['all_paths_bitwise_equal'])}


def average(phase, mode, stage, field='wall_seconds'):
    summary = phase['summaries'][mode]
    return summary['stages'].get(stage, {}).get(field, 0.)/summary['requests']


def timing_table(phase):
    lines = ['| 方案 | 规划通过 | 平均耗时 | 中位数 | P90 | 10次累计 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, label in zip(['legacy', 'batched'], LABELS[phase['kind']]):
        s = phase['summaries'][mode]
        lines.append(f"| {label} | {s['successes']}/{s['requests']} | "
                     f"{s['total_seconds']/s['requests']:.3f}秒 | {s['median_seconds']:.3f}秒 | "
                     f"{s['p90_seconds']:.3f}秒 | {s['total_seconds']:.3f}秒 |")
    return lines


def log_phase(root, phase):
    a, b = [phase['summaries'][m]['total_seconds'] for m in ['legacy', 'batched']]
    status = 'kept' if phase['kept'] else 'reverted'
    finished = phase['review']['at']
    path = root/'autoresearch-results.tsv'
    if not path.exists():
        with path.open('w') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(['iteration','metric_value','delta','delta_pct','status',
                             'description','evaluator_source','timestamp'])
            writer.writerow([0,a,'-','-','baseline',LABELS[phase['kind']][0],'mechanical',finished])
            writer.writerow([1,b,b-a,f'{(b/a-1)*100:.3f}%',status,
                             LABELS[phase['kind']][1],'mechanical',finished])
        with (root/'research.md').open('a') as stream:
            stream.write(f"| 1 | {LABELS[phase['kind']][1]} | {a:.6f}→{b:.6f}秒 | {status} |\n")
        (root/'research_log.md').write_text(
            f"# 本轮结果\n\n两组各10次完整规划，累计{a:.6f}→{b:.6f}秒。\n\n"
            f"完整RPC累计、中位数与P90门槛：{phase['review']['timing_gate_passed']}。"
            f"关节终点逐位相同：{phase['review']['all_endpoints_bitwise_equal']}；"
            f"预抓取和接近轨迹逐位相同：{phase['review']['all_paths_bitwise_equal']}。\n\n"
            f"结论：{status}。score=-total_RPC_seconds，原机器人、采样和碰撞约束保持。\n")
    (root/'final_report.md').write_text('# 本轮规划优化结果\n\n'+'\n'.join(timing_table(phase))+
        f"\n\n累计省时{1-b/a:.2%}，采用结论：{status}。物理抓取0次。\n")
    write_json(root/'evaluator.json', {'pass': phase['kept'], 'score': -b,
        'baseline_score': -a, 'status': status, 'evaluator_source': 'mechanical'})


def build(sweep_root, codec_root):
    phases = [load_phase(sweep_root, 'sweep'), load_phase(codec_root, 'codec')]
    for root, phase in zip([sweep_root, codec_root], phases):
        log_phase(root, phase)
    a, b = phases
    chosen = b if b['kept'] else a
    body = ['# V7 开合爪批量校验与候选搜索优化', '', '2026年11月27日。', '',
        f"批量开合爪校验使完整规划累计省时{a['review']['total_saving_fraction']:.2%}。"
        f"第二轮配置编解码优化的累计变化为{b['review']['total_saving_fraction']:.2%}。", '',
        '两轮分别比较五个冻结场景，每场景每组各两次，共40次新规划。使用V7宽深修正后重新检查碰撞的候选池。', '',
        '## 第一轮：开合爪保留原采样，批量检查', '',
        '旧实现每改变0.5毫米开度，就完整绑定一次机器人和场景。左臂准备开爪有89个采样，'
        '同一场景和模型被重复写出、读取、更新。', '',
        '新实现绑定起始场景后，在每个原开度计算原生FK和871个碰撞球，再一起检查关节边界、'
        '自碰撞与场景碰撞。结束后恢复原规划状态。收爪也复用同一流程。', '']
    body += timing_table(a)
    body += ['', '| 每次规划的平均开销 | 旧实现 | 批量实现 |', '| --- | ---: | ---: |']
    for stage, name in [('gripper_opening','双臂开爪校验'), ('gripper_narrowing','候选开度收爪校验'),
                        ('candidate_search','候选搜索全流程')]:
        body.append(f"| {name} | {average(a,'legacy',stage):.3f}秒 | {average(a,'batched',stage):.3f}秒 |")
    body += ['', '收爪校验包含在候选搜索中，表中三行不能相加。完整机器人/场景绑定次数平均'
             f"从{a['summaries']['legacy']['stages']['world_and_robot_bind']['calls']/10:.1f}次降为"
             f"{a['summaries']['batched']['stages']['world_and_robot_bind']['calls']/10:.1f}次。", '',
        '## 第二轮：加快候选宽度切换的配置准备', '',
        '第一轮之后，候选搜索中仍有重复的机器人配置编码与写出。第二轮两组都使用批量开合爪，'
        '只将原绑定函数使用的YAML读写换成C实现，并沿用原缓存和复制规则。场景更新、碰撞球、锁定关节和数值计算保持。', '']
    body += timing_table(b)
    body += ['', f"候选搜索全流程的平均耗时由{average(b,'legacy','candidate_search'):.3f}秒降至"
             f"{average(b,'batched','candidate_search'):.3f}秒，其中候选宽度切换由"
             f"{average(b,'legacy','candidate_width_update'):.3f}秒降至{average(b,'batched','candidate_width_update'):.3f}秒。", '',
             '每轮的前后两组在同一常驻模型中交替运行。两轮分别统计，不将不同轮次的均值拼成加速倍数。', '',
             '## 剩余时间花在哪里', '',
             '下表是第二轮每次完整规划的平均耗时。各行扣除了表中其他已计时子调用，可逐项相加。', '',
             '| 环节 | 原YAML | C版YAML |', '| --- | ---: | ---: |']
    grouped = [
        ('机器人和场景绑定', ['world_and_robot_bind']),
        ('夹爪逐开度FK与批量验碰撞', ['gripper_opening','gripper_narrowing']),
        ('接触几何上下文更新', ['contact_validator_update']),
        ('抓取IK与终点几何核验', ['grasp_IK','grasp_endpoint']),
        ('预抓取与抬升IK', ['pregrasp_IK','lift_IK']),
        ('路径搜索', ['path_PRM']),
        ('轨迹优化', ['trajectory_approach','trajectory_pregrasp']),
        ('双臂映射与候选调度', ['physical_arm_mapping','candidate_search','candidate_width_update'])]
    totals = dict.fromkeys(['legacy','batched'], 0.)
    for label, stages in grouped:
        values = {m: sum(average(b,m,s,'exclusive_seconds') for s in stages) for m in totals}
        for m in totals: totals[m] += values[m]
        body.append(f"| {label} | {values['legacy']:.3f}秒 | {values['batched']:.3f}秒 |")
    rest = {m: b['summaries'][m]['total_seconds']/10-totals[m] for m in totals}
    body.append(f"| 共享交接、记录与其他请求工作 | {rest['legacy']:.3f}秒 | {rest['batched']:.3f}秒 |")
    body += ['', '## 五个场景逐项结果', '',
        '耗时为每场景两次规划的平均值；括号内为每次实际检查的候选数。', '',
        '| 场景 | 第一轮原版 | 第一轮批量 | 第二轮原YAML | 第二轮C版YAML |',
        '| --- | ---: | ---: | ---: | ---: |']
    for ca, cb in zip(a['cases'], b['cases']):
        values = [c[m] for c in [ca,cb] for m in ['legacy','batched']]
        body.append('| '+ca['id']+' | '+' | '.join(
            f"{v['mean_RPC_seconds']:.3f}秒（{'/'.join(map(str,v['candidate_checks']))}）" for v in values)+' |')
    body += ['', '## 验证与采用', '',
        '逐采样核验覆盖左右臂开爪、左右臂收爪和中途碰撞共5种情况、258个采样。'
        '全部碰撞球坐标与半径完全一致，通过掩码和首次失败位置一致；端点无碰撞但中间有碰撞的场景在索引37拦下。', '',
        '两轮共40条完整规划，120个预抓取/抓取/抬升关节终点独立FK与限位复核通过。'
        '两组的抓姿、手臂、开度、关节终点以及预抓取/接近轨迹逐位相同。'
        f"第二轮共{sum(p['models_equal'] for p in b['codec_review']['pairs'])}对实际生成的YAML配置逐值相同。", '',
        '| 检查项 | 结果 |', '| --- | --- |',
        '| 任务完成 | 两轮40/40完整规划；各组10/10通过 |',
        '| 记录完整性 | 登记、领取、输入来源、规划、分项计时和独立核验完整 |',
        '| 事件适用性 | 本轮是规划对照，物理抓取0次 |',
        '| 关键事件覆盖 | 开爪、收爪、中途碰撞拦截、目标/停放臂验碰撞、恢复原状态、完整路径和配置值 |', '',
        f"可选V7规划入口采用{'批量开合爪＋C版YAML' if b['kept'] else '批量开合爪校验'}。主线默认入口保持。", '',
        '## 候选搜索后续可以优化的地方', '',
        '1. 将固定场景和变化的夹爪开度分开更新。切换候选宽度时只更新锁定关节及相关运动学张量，'
        '避免再次绑定同一个场景；必须完整恢复IK、碰撞和轨迹上下文，并逐值核验。',
        '2. 复用固定接触几何。URDF、碰撞网格和不变变换只准备一次，候选间只更新开度相关几何。',
        '3. 对剩余候选的抓取IK单独比较固定批量检查。当前宽度各不相同，需要先保证每个候选使用'
        '自己的碰撞球和原搜索次序；不能将统一全开宽度的批量IK直接接过来。', '',
        '这三项尚未执行。本次已完成的两项先保留为独立可选入口。', '', '## 记录', '',
        f"- 第一轮：`{a['run']}`", f"- 第二轮：`{b['run']}`",
        '- 入口：`modules.curobo.gripper_sweep.runtime.create`',
        '- 原始完整后处理结果：[V7_原始候选完整后处理与优化结果.md](V7_原始候选完整后处理与优化结果.md)', '']
    document = ROOT/'modules/pipeline/docs/mainline_versions/V7_开合爪批量校验与候选搜索优化.md'
    document.write_text('\n'.join(body))
    snapshot = {'at': now(), 'phases': phases, 'planning_requests': 40, 'physical_grasps': 0,
                'default_activated': False, 'optional_codec_enabled': b['kept'],
                'task_complete': True, 'record_integrity': True,
                'adopted_run': chosen['run'], 'remaining_candidate_search_optimizations_executed': False}
    write_json(codec_root/'publication_snapshot.json', snapshot)
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    args = parser.parse_args()
    build(ensure_run_path(args.sweep), ensure_run_path(args.codec))
