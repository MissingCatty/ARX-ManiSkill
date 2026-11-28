"""Read-only, additive latency accounting for the completed paired grasp suite."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from arx_project.layout import ROOT, now, read_json, write_json, ensure_run_path, sha256


def completed_registration(root):
    """Verify the executed archive even after a later optional-module registration."""
    reg = read_json(root / 'registration.json')
    if not read_json(root / 'acceptance.json')['task_complete']:
        raise ValueError('Completed records are required for archival analysis')
    for path, digest in reg['source_sha256'].items():
        archive = root / 'source_registered' / Path(path).relative_to(ROOT)
        if sha256(archive) != digest:
            raise ValueError('Executed source archive changed: ' + str(archive))
    for path, digest in reg['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Recorded input changed: ' + path)
    return reg

MODES = ['baseline', 'optimized']
PHASES = ['initial', 'approach', 'lift']
LABELS = {
    'robot_scene': '机器人和场景绑定', 'contact': '接触几何更新',
    'grasp_IK': '抓取终点IK', 'endpoint_other': '终点补充检查与回退',
    'other_IK': '其他终点IK', 'path': '路径搜索', 'trajectory': '轨迹优化',
    'gripper': '开爪与收爪扫掠检查', 'selection': '候选调度及开度切换管理',
    'unclassified': '未单列的其余服务开销'}


def category(name):
    if name == 'world_and_robot_bind' or name.startswith('binding_'): return 'robot_scene'
    if name == 'contact_validator_update' or name.startswith('contact_'): return 'contact'
    if name == 'grasp_IK': return 'grasp_IK'
    if name == 'grasp_endpoint': return 'endpoint_other'
    if name in ['pregrasp_IK', 'lift_IK', 'other_IK']: return 'other_IK'
    if name == 'path_PRM': return 'path'
    if name.startswith('trajectory_'): return 'trajectory'
    if name in ['gripper_opening', 'gripper_narrowing']: return 'gripper'
    if name in ['candidate_search', 'candidate_width_update']: return 'selection'
    raise ValueError('An unrecognized timing stage needs an explicit category: ' + name)


def verified_read(path, claim):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != claim['files_sha256'][path.name]:
        raise ValueError('Recorded timing differs from its completed claim: ' + str(path))
    return json.loads(data)


def stats(values):
    if not values: return {'samples': 0, 'total_seconds': 0.}
    ordered = sorted(values); index = .9 * (len(ordered) - 1); lo = math.floor(index)
    return {'samples': len(values), 'total_seconds': sum(values), 'mean_seconds': statistics.mean(values),
        'median_seconds': statistics.median(values),
        'p90_seconds': ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)}


def create(root):
    completed_registration(root)
    accepted = read_json(root / 'acceptance.json'); progress = read_json(root / 'progress.json')
    if not accepted['task_complete'] or not accepted['record_integrity'] or progress['phase'] != 'complete':
        raise ValueError('Only a completed, independently accepted suite can be analyzed')
    excluded = set(accepted['interrupted_timing_cases_excluded'])
    cases = sorted({row['id'] for row in progress['results']
        if not row['common_precondition_unexecuted'] and not row.get('interrupted') and row['id'] not in excluded})
    if len(cases) != 68: raise ValueError('The accepted matched cohort changed')
    result = {'at': now(), 'run': str(root.relative_to(ROOT)), 'paired_cases': cases,
        'excluded_interrupted_cases': sorted(excluded), 'new_planning_requests': 0, 'new_GPU_calls': 0,
        'timing_scope': 'recorded host wall time, mutually exclusive nested stage costs', 'groups': {}}
    for mode in MODES:
        phases = {}; rpc = []; trials = []; remainders = []; scene_planning = []
        for phase in PHASES:
            phases[phase] = {'durations': [], 'group_totals': Counter(), 'named_totals': Counter(),
                'calls': Counter(), 'individual_calls': defaultdict(list)}
        for case in cases:
            folder = root / 'trials' / case / mode
            claim = read_json(root / 'claims' / (case + '_' + mode + '.json'))
            if claim['state'] != 'complete': raise ValueError('A timing case is not complete')
            latency = verified_read(folder / 'latency.json', claim)['timings']
            rpc.append(latency['initial_planning_with_postprocessing_seconds'])
            trials.append(latency['trial_wall_seconds']); planning = 0.
            for phase in PHASES:
                path = folder / (phase + '_metrics.json')
                other = root / 'trials' / case / ('optimized' if mode == 'baseline' else 'baseline') / path.name
                if path.exists() != other.exists():
                    raise ValueError('Phase accounting requires the same occurring phases in both groups')
                if not path.exists(): continue
                metric = verified_read(path, claim); target = phases[phase]
                timing = metric['stages']['stages']; measured = metric['seconds']
                total = 0.
                for name, row in timing.items():
                    value = row['exclusive_seconds']; target['group_totals'][category(name)] += value
                    target['named_totals'][name] += value; target['calls'][name] += row['calls']; total += value
                if total > measured + 1e-7: raise ValueError('Nested timing costs exceed the whole service request')
                target['group_totals']['unclassified'] += measured - total
                target['durations'].append(measured); planning += measured
                for event in metric['stages']['events']:
                    target['individual_calls'][event['stage']].append(event['wall_seconds'])
            scene_planning.append(planning); remainders.append(latency['trial_wall_seconds'] - planning)
        for phase, data in phases.items():
            n = len(data['durations']); whole = sum(data['durations'])
            if abs(sum(data['group_totals'].values()) - whole) > 1e-7:
                raise ValueError('Timing categories must add to the complete service cost')
            expected = accepted['matched_phase_timings'][mode]['timings'][phase]
            if n != expected['samples'] or abs(whole - expected['total_seconds']) > 1e-7:
                raise ValueError('Timing totals differ from the accepted paired results')
            phases[phase] = {'service': stats(data['durations']),
                'mean_per_scene_seconds': whole / len(cases),
                'mean_groups_per_request_seconds': {key: data['group_totals'][key] / n for key in LABELS},
                'mean_named_per_request_seconds': {key: value / n for key, value in data['named_totals'].items()},
                'group_totals_seconds': dict(data['group_totals']), 'calls': dict(data['calls']),
                'individual_call_wall_times': {key: stats(value) for key, value in data['individual_calls'].items()}}
        whole = stats(scene_planning)
        if abs(whole['mean_seconds'] - accepted['matched_phase_timings'][mode]['all_planning_per_scene']['mean_seconds']) > 1e-8:
            raise ValueError('Per-scene planning sums differ from acceptance')
        result['groups'][mode] = {'phases': phases, 'all_planning': whole,
            'initial_RPC': stats(rpc), 'trial_wall': stats(trials), 'trial_other_than_planning': stats(remainders)}
    n = len(cases); a, b = result['groups']['baseline'], result['groups']['optimized']
    result['reduction_fraction'] = 1 - b['all_planning']['total_seconds'] / a['all_planning']['total_seconds']
    result['initial_share_of_net_savings'] = ((a['phases']['initial']['mean_per_scene_seconds'] -
        b['phases']['initial']['mean_per_scene_seconds']) / (a['all_planning']['mean_seconds'] - b['all_planning']['mean_seconds']))
    started = read_json(root / 'started.json')['at']; ready = read_json(root / 'preflight/all_ready.json')['at']
    from datetime import datetime
    result['one_time_startup_and_preflight_seconds'] = (datetime.fromisoformat(ready) - datetime.fromisoformat(started)).total_seconds()
    write_json(root / 'latency_breakdown.json', result)
    resources = ROOT / 'modules/pipeline/docs/mainline_versions/resources/memory_state_execution_latency.json'
    write_json(resources, result)
    lines = ['# V8 内存更新完整测试耗时拆解', '',
        '读取execution74_20261128_04已完成记录，只做离线统计。本次测试复用了已保存的观测和候选，以下拆分规划与抓取试验耗时。', '',
        '排除共同站位未满足和中断场景，比较同一68个完整配对场景。各阶段累计时间统一除以68，未发生阶段贡献为0；接近实际发生67次，抬升59次。', '',
        '## 三次规划在总耗时中的贡献', '', '| 阶段 | 更新前平均每场景 | 更新后平均每场景 |', '| --- | ---: | ---: |']
    for phase, label in [('initial', '初始规划'), ('approach', '实测预抓取后重规划接近'), ('lift', '实测闭合后重规划抬升')]:
        lines.append(f"| {label} | {a['phases'][phase]['mean_per_scene_seconds']:.4f}秒 | {b['phases'][phase]['mean_per_scene_seconds']:.4f}秒 |")
    lines += [f"| 合计 | {a['all_planning']['mean_seconds']:.4f}秒 | {b['all_planning']['mean_seconds']:.4f}秒 |", '',
        f"合计减少{result['reduction_fraction']:.2%}，约{result['initial_share_of_net_savings']:.1%}的净节省来自初始规划。", '',
        '## 初始规划内部', '', '各项互不重叠。开爪期间发生的绑定统一归入绑定项，IK包含原生碰撞约束；终点补充检查包含回退及精细接触校验。', '',
        '| 环节 | 更新前 | 更新后 |', '| --- | ---: | ---: |']
    ia = a['phases']['initial']['mean_groups_per_request_seconds']; ib = b['phases']['initial']['mean_groups_per_request_seconds']
    combined = [('robot_scene', 'contact', '机器人、场景与接触几何更新'), ('path','trajectory','路径搜索与轨迹优化')]
    initial_rows = [('robot_scene_and_contact', combined[0][2], ia['robot_scene']+ia['contact'], ib['robot_scene']+ib['contact'])]
    for key in ['grasp_IK','endpoint_other','other_IK']:
        initial_rows.append((key,LABELS[key],ia[key],ib[key]))
    initial_rows.append(('path_and_trajectory',combined[1][2],ia['path']+ia['trajectory'],ib['path']+ib['trajectory']))
    for key in ['gripper','selection','unclassified']:
        initial_rows.append((key,LABELS[key],ia[key],ib[key]))
    for key,label,x,y in initial_rows: lines.append(f'| {label} | {x:.3f}秒 | {y:.3f}秒 |')
    lines += [f"| 初始服务合计 | {a['phases']['initial']['service']['mean_seconds']:.3f}秒 | {b['phases']['initial']['service']['mean_seconds']:.3f}秒 |", '',
        f"初始完整RPC包含交接与返回，平均{a['initial_RPC']['mean_seconds']:.3f}→{b['initial_RPC']['mean_seconds']:.3f}秒。", '']
    calls = b['phases']['initial']['individual_call_wall_times']['grasp_IK']
    lines += [f"两组抓取终点IK均调用{calls['samples']}次，平均每场景{calls['samples']/n:.1f}次。"
        f"更新后单次墙钟均值{calls['mean_seconds']*1000:.1f}毫秒、中位{calls['median_seconds']*1000:.1f}毫秒，累计贡献{ib['grasp_IK']:.3f}秒/场景。", '',
        '本轮在共享GPU2上运行，以上为实际请求的墙钟耗时。', '',
        '## 当前几何更新的进一步拆分', '',
        '以下各项属于前表的机器人、场景与接触几何更新，按同一68个场景平均。场景更新是将已有场景同步到碰撞检测器。', '',
        '| 环节 | 更新后平均耗时 |', '| --- | ---: |']
    named = b['phases']['initial']['mean_named_per_request_seconds']
    update_rows = [
        ('规划器的场景碰撞模型更新', named['binding_scene_update']),
        ('精细接触检查器的场景更新', named['contact_scene_update']),
        ('实测FK与起始状态碰撞核验', named['binding_live_validation']),
        ('机器人与夹爪动态参数更新', named['binding_dynamic_robot_update'] + named['contact_dynamic_robot_update'])]
    update_total = ib['robot_scene'] + ib['contact']
    update_rows.append(('输入快照、留档入队及其余绑定处理', update_total - sum(value for _, value in update_rows)))
    for label, value in update_rows: lines.append(f'| {label} | {value:.3f}秒 |')
    lines += [f'| 合计 | {update_total:.3f}秒 |', '',
        f"几何更新与抓取终点IK合计{update_total + ib['grasp_IK']:.3f}秒，占初始规划的{(update_total + ib['grasp_IK']) / b['phases']['initial']['service']['mean_seconds']:.1%}。", '',
        '## 接近与抬升的服务内部', '',
        '下表按实际发生的一次请求平均；接近为67次、抬升为59次。', '',
        '| 更新后的环节 | 接近重规划 | 抬升重规划 |', '| --- | ---: | ---: |']
    for label, keys in [('机器人、场景与接触几何更新',['robot_scene','contact']), ('IK',['grasp_IK','other_IK','endpoint_other']),
                        ('路径与轨迹优化',['path','trajectory']), ('其余未单列开销',['unclassified','gripper','selection'])]:
        values = [sum(b['phases'][phase]['mean_groups_per_request_seconds'][key] for key in keys) for phase in ['approach','lift']]
        lines.append(f'| {label} | {values[0]:.3f}秒 | {values[1]:.3f}秒 |')
    lines += [f"| 单次服务合计 | {b['phases']['approach']['service']['mean_seconds']:.3f}秒 | {b['phases']['lift']['service']['mean_seconds']:.3f}秒 |", '',
        '其余服务开销通过完整请求减去已记录的互斥计时得到。现有记录没有给这部分逐函数分配时间。', '',
        '## 整次仿真试验', '', '| 项目 | 更新前 | 更新后 |', '| --- | ---: | ---: |',
        f"| 各段规划服务合计 | {a['all_planning']['mean_seconds']:.3f}秒 | {b['all_planning']['mean_seconds']:.3f}秒 |",
        f"| 复位、动作执行、渲染、录像与结果核验等其余耗时 | {a['trial_other_than_planning']['mean_seconds']:.3f}秒 | {b['trial_other_than_planning']['mean_seconds']:.3f}秒 |",
        f"| 整次试验实际耗时 | {a['trial_wall']['mean_seconds']:.3f}秒 | {b['trial_wall']['mean_seconds']:.3f}秒 |", '',
        f"04批次常驻组件启动与统一预检为{result['one_time_startup_and_preflight_seconds']:.1f}秒，单列为一次性开销。", '',
        '当前初始规划最主要的两块为模型及场景更新、抓取终点IK；后续细分可优先覆盖抬升中约1秒的未分项开销。', '',
        '数据：[聚合JSON](resources/memory_state_execution_latency.json)。入口：`modules.pipeline.experiments.state_update_execution.latency_analysis`。', '']
    document = ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存更新完整测试耗时拆解.md'
    document.write_text('\n'.join(lines))
    print(json.dumps({'document':str(document),'paired_cases':n,'reduction_fraction':result['reduction_fraction'],
        'initial_share_of_net_savings':result['initial_share_of_net_savings'],'new_GPU_calls':0},ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    create(ensure_run_path(p.parse_args().root))
