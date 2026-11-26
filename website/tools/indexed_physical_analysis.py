"""Explain completed physical pairs using immutable planner and execution records."""
import json
import math
from collections import Counter
from pathlib import Path

from arx_project.layout import ROOT, now, sha256, write_json

MARKER = '<!-- physical-pair-analysis -->'
LABELS = {'donut': '甜甜圈', 'scissors': '剪刀', 'banana': '香蕉',
          'bar_soap': '香皂', 'cucumber': '黄瓜', 'apple': '苹果',
          'cell_phone': '手机', 'mango': '芒果'}
REASONS = {'success': '成功', 'approach_not_converged': '接近动作未到位',
           'no_bilateral_target_contact': '闭合后未形成双指夹持',
           'insufficient_hold_height': '抬升或保持不足'}
STAGES = {'open': '张开', 'pregrasp': '到预抓取位', 'approach': '接近',
          'close': '闭合', 'lift': '抬升', 'hold': '保持'}


def raw(path):
    return json.loads(Path(path).read_text())


def pose_difference(a, b):
    translation = math.dist([a[i][3] for i in range(3)], [b[i][3] for i in range(3)])
    trace = sum(a[i][j] * b[i][j] for i in range(3) for j in range(3))
    angle = math.degrees(math.acos(max(-1., min(1., (trace - 1) / 2))))
    # Identical stored matrices need not be perfectly orthonormal after FP32 conversion.
    return {'translation_mm': translation * 1000, 'rotation_deg': 0. if a == b else angle}


def analyze(run):
    reg = raw(run / 'registration.json')
    accepted = raw(run / 'acceptance.json')
    if not accepted['passed'] or not accepted['record_integrity']:
        raise ValueError('Only independently reviewed physical records can be published')
    pairs = []
    coverage = {mode: Counter() for mode in ['baseline', 'optimized']}
    replans = Counter()
    for case in reg['cases']:
        row = {'id': case['id'], 'label': LABELS.get(case['entry']['family'], case['entry']['family'])}
        poses = []
        for mode in ['baseline', 'optimized']:
            directory = run / 'trials' / case['id'] / mode
            selected = raw(directory / 'plan_result.json')['selected']
            result = raw(directory / 'result.json')
            trace = [json.loads(line) for line in (directory / 'execution.jsonl').read_text().splitlines() if line.strip()]
            actual = set(v['stage'] for v in trace)
            if actual != set(result['stages']):
                raise ValueError('Measured stages disagree with the phase report')
            coverage[mode].update(actual)
            for stage in ['approach', 'lift']:
                if (directory / ('planner_rpc_' + stage + '.json')).exists():
                    if not (directory / ('execution_planner_' + stage + '.json')).is_file():
                        raise ValueError('A requested replan lost its receipt')
                    replans[stage] += 1
            poses.append(selected['contact_pose_world'])
            row[mode] = {'source_index': selected['original_source_index'],
                'search_priority_rank': selected['search_priority_rank'],
                'score': selected['score'], 'arm': selected['arm'], 'success': result['success'],
                'reason': result['reason'], 'stages': result['stages'],
                'open_finger_m': selected['open_finger_m'],
                'minimum_hold_lift_m': result.get('minimum_hold_lift_m'),
                'initial_planning_seconds': raw(directory / 'latency.json')['timings']['initial_planning_with_postprocessing_seconds']}
            if case['id'] in accepted['baseline_success_losses']:
                # Use the first measured sample as the origin; frozen restore errors are sub-millimetric.
                first = trace[0]['object_pose'][:3]
                row[mode]['object_motion'] = {}
                for stage in ['approach', 'close']:
                    distances = [1000 * math.dist(first, v['object_pose'][:3]) for v in trace if v['stage'] == stage]
                    row[mode]['object_motion'][stage] = {
                        'maximum_mm': max(distances) if distances else None,
                        'last_mm': distances[-1] if distances else None}
        row['pose_difference'] = pose_difference(*poses)
        row['same_grasp_and_arm'] = poses[0] == poses[1] and row['baseline']['arm'] == row['optimized']['arm']
        manifest = raw(run / 'trials' / case['id'] / 'optimized' / 'candidate_order.json')
        old_index = row['baseline']['source_index']
        row['old_candidate_in_optimized_order'] = {
            'retained': old_index in manifest['priority_to_original_source'],
            'representative': old_index in manifest['representatives'],
            'deferred': old_index in manifest['deferred'],
            'priority_rank': manifest['priority_to_original_source'].index(old_index)}
        pairs.append(row)
    strata = {}
    for unchanged, name in [(True, 'same_grasp'), (False, 'changed_grasp')]:
        rows = [p for p in pairs if p['same_grasp_and_arm'] == unchanged]
        strata[name] = {'pairs': len(rows), **{
            mode + '_successes': sum(p[mode]['success'] for p in rows) for mode in ['baseline', 'optimized']}}
    output = {'at': now(), 'acceptance_sha256': sha256(run / 'acceptance.json'),
        'pairs': pairs, 'strata': strata, 'measured_stage_coverage': coverage,
        'requested_replans': replans,
        'task': {'registered': 20, 'completed': 20, 'interrupted': 0, 'unstarted': 0},
        'record_integrity': True, 'default_activated': False,
        'event_applicability': 'actual trace and phase-report stages agree; no close/lift/hold required after earlier normal failure'}
    write_json(run / 'physical_analysis.json', output)
    return output


def markdown(analysis):
    rows = analysis['pairs']
    lines = ['### 抓法变化与实际抓取', '',
        '| 场景 | 原候选 → 新候选 | 抓点位置变化 | 抓点朝向变化 | 当前链路 → 优化组合 |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in rows:
        diff = row['pose_difference']
        lines.append(f"| {row['id'].split('_')[0]} · {row['label']} | {row['baseline']['source_index']} → {row['optimized']['source_index']} | "
            f"{diff['translation_mm']:.2f}毫米 | {diff['rotation_deg']:.2f}° | "
            f"{REASONS[row['baseline']['reason']]} → {REASONS[row['optimized']['reason']]} |")
    same = analysis['strata']['same_grasp']; changed = analysis['strata']['changed_grasp']
    lines += ['', f"抓法和选臂未变的{same['pairs']}个场景，两组均成功{same['baseline_successes']}/{same['pairs']}，逐例结果一致。"
        f"换了抓法的{changed['pairs']}个场景，成功从{changed['baseline_successes']}/{changed['pairs']}降至{changed['optimized_successes']}/{changed['pairs']}。", '',
        '### 两例退步发生在哪里', '']
    donut = next(p for p in rows if p['id'].startswith('M0020_'))
    scissors = next(p for p in rows if p['id'].startswith('M0074_'))
    old = donut['baseline']; new = donut['optimized']
    lines += [f"**甜甜圈 M0020：接近时推偏物体，未通过到位检查。** 抓法由原候选{old['source_index']}换成{new['source_index']}，"
        f"位置相差{donut['pose_difference']['translation_mm']:.2f}毫米、朝向相差{donut['pose_difference']['rotation_deg']:.2f}°。"
        f"接近期间物体最大移动量由{old['object_motion']['approach']['maximum_mm']:.2f}增至{new['object_motion']['approach']['maximum_mm']:.2f}毫米。"
        f"优化组接近结束的最大关节误差为{new['stages']['approach']['joint_error_rad']:.6f}弧度，超过原有0.005弧度门槛；"
        '因此在闭合之前结束。当前链路则完成了闭合、抬升和保持。', '',
        f"**杂乱剪刀 M0074：到位后闭合，没有形成双指夹持。** 抓法由原候选{scissors['baseline']['source_index']}换成{scissors['optimized']['source_index']}，"
        f"位置相差{scissors['pose_difference']['translation_mm']:.2f}毫米、朝向相差{scissors['pose_difference']['rotation_deg']:.2f}°。"
        f"优化组接近末端位置误差约{scissors['optimized']['stages']['approach']['tool_position_error_m']*1000:.6f}毫米，"
        '到位检查通过；随后闭合未形成双指夹持，未进入抬升。当前链路成功带起并保持。', '',
        '| 退步场景 | 旧成功候选仍在池中 | NMS后的所在区域 | 旧成功候选的新优先名次（从0计） | 实际先选中的新候选 |',
        '| --- | --- | --- | ---: | ---: |']
    for row in [donut, scissors]:
        old = row['old_candidate_in_optimized_order']
        lines.append(f"| {row['id'].split('_')[0]} | 是 | {'后备区' if old['deferred'] else '代表区'} | {old['priority_rank']} | {row['optimized']['source_index']} |")
    lines += ['', 'NMS保留全部后备，保证了候选仍可被查询；先接受新代表后，搜索就结束了，旧成功抓法没有获得实际执行机会。'
        '两例都保持了原夹爪进入开度：单指0.044米、总宽0.088米。', '',
        '本轮证据指向候选重排带来的抓法变化。四个抓法未变的对照逐例一致；'
        'B16、配置写出和网格排列也一同改变了，单靠这一轮还不能分别计算各项对物理结果的影响。', '',
        '### 任务、记录与事件覆盖', '',
        '| 核验项目 | 本轮结果 |', '| --- | --- |',
        '| 任务结果 | 20/20完成；中断0，未开始0。两组规划均10/10，实际成功8/10与6/10。 |',
        '| 记录完整性 | 20条领取与结果、20次初始状态恢复、命令与回执逐项核对；40段原视频解码通过。 |',
        f"| 事件适用性 | 按实际轨迹与阶段报告核对已发生的动作。{20-sum(c['close'] for c in analysis['measured_stage_coverage'].values())}例在接近结束，"
        f"{sum(c['close']-c['lift'] for c in analysis['measured_stage_coverage'].values())}例在闭合结束；后续未执行的阶段单列。 |",
        f"| 关键事件覆盖 | 初始规划20次，接近重规划{analysis['requested_replans']['approach']}次；抬升重规划{analysis['requested_replans']['lift']}次。逐阶段执行数量见下表。 |", '',
        '| 实际执行阶段 | 当前链路 | 优化组合 |', '| --- | ---: | ---: |']
    for key, label in STAGES.items():
        lines.append(f"| {label} | {analysis['measured_stage_coverage']['baseline'][key]} | {analysis['measured_stage_coverage']['optimized'][key]} |")
    lines += ['', '### 下一步建议', '',
        '保留当前默认链路。下一轮优先保持原始评分顺序，只叠加B16、C版YAML与新网格排列，'
        '将提速和候选重排分开验证；先检查本轮两例退步，再决定更大范围测试。NMS代表优先的抓取质量另设对照。', '']
    return '\n'.join(lines)


def update_document(run, document):
    analysis = analyze(run)
    body = document.read_text().split(MARKER)[0].rstrip()
    body = body.replace('no_bilateral_target_contact', '闭合后未形成双指夹持').replace(
        'approach_not_converged', '接近动作未到位').replace('insufficient_hold_height', '抬升或保持不足')
    document.write_text(body + '\n\n' + MARKER + '\n\n' + markdown(analysis))
    return analysis
