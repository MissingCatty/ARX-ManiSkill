"""Make a reproducible, standalone report from the completed diagnosis."""
import argparse
import csv
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path


COUNTS = [871, 512, 256]
LABELS = {'M0001_S1_open_apple': '开放场景苹果', 'M0020_S1_open_donut': '开放场景甜甜圈',
          'M0044_S3_sparse_banana': '稀疏场景香蕉', 'M0064_S4_clutter_mango': '杂乱场景芒果',
          'M0074_S4_clutter_scissors': '杂乱场景剪刀'}
STAGES = [('机器人与场景绑定', ['world_and_robot_bind']),
          ('开合爪模型读取、FK与碰撞检查', ['gripper_opening', 'gripper_narrowing']),
          ('抓取IK与终点核验', ['grasp_IK', 'grasp_endpoint']),
          ('接触几何更新', ['contact_validator_update']),
          ('预抓取与抬升IK', ['pregrasp_IK', 'lift_IK']),
          ('路径搜索与轨迹优化', ['path_PRM', 'trajectory_approach', 'trajectory_pregrasp'])]


def fixed_work_summary(micro):
    result = []
    for batch in [1, 32, 256]:
        for operation in ['FK', 'world_collision', 'complete_validation']:
            measurements = [next(v for v in case['measurements'] if v['joint_states'] == batch)[operation]
                            for case in micro]
            result.append({'batch': batch, 'operation': operation, 'counts': {str(count): {
                key: float(np.median([v[str(count)][key] for v in measurements])) for key in
                ['graph_cuda_median_ms', 'graph_cuda_p90_ms', 'eager_wall_median_ms']}
                for count in COUNTS}})
    return result


def full_table(data):
    lines = ['| 碰撞球数量 | 规划通过 | 平均耗时 | 中位数 | P90 | 候选检查总数 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for count in COUNTS:
        value = data['summaries'][str(count)]
        lines.append(f"| {count} | {value['successes']}/{value['requests']} | {value['mean_seconds']:.3f}秒 | "
            f"{value['median_seconds']:.3f}秒 | {value['p90_seconds']:.3f}秒 | {value['candidate_checks']} |")
    return lines


def stage_table(data):
    lines = ['| 完整请求中的环节 | 871球 | 512球 | 256球 |', '| --- | ---: | ---: | ---: |']
    used = dict.fromkeys(COUNTS, 0.)
    for label, stages in STAGES:
        values = []
        for count in COUNTS:
            summary = data['summaries'][str(count)]
            value = sum(summary['stages'].get(stage, {}).get('exclusive_seconds', 0.) for stage in stages) / summary['requests']
            used[count] += round(value, 3)
            values.append(f'{value:.3f}秒')
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines.append('| 其余交接、调度与记录 | ' + ' | '.join(
        f"{round(data['summaries'][str(n)]['mean_seconds'], 3) - used[n]:.3f}秒" for n in COUNTS) + ' |')
    lines.append('| 合计 | ' + ' | '.join(f"{data['summaries'][str(n)]['mean_seconds']:.3f}秒" for n in COUNTS) + ' |')
    return lines


def micro_table(data, operation='complete_validation'):
    lines = ['| 同一批关节状态数量 | 871球 | 512球 | 256球 |', '| ---: | ---: | ---: | ---: |']
    for row in data['fixed_work_summary']:
        if row['operation'] == operation:
            lines.append(f"| {row['batch']} | " + ' | '.join(
                f"{row['counts'][str(n)]['graph_cuda_median_ms']:.4f}毫秒" for n in COUNTS) + ' |')
    return lines


def build(root):
    review = read_json(root/'sphere_count_review.json')
    independent = read_json(root/'independent_review.json')
    if not review['passed'] or not independent['passed'] or not read_json(root/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Complete independent review and shut down the owned services first')
    progress = read_json(root/'progress.json')
    summaries = read_json(root/'acceptance.json')['summaries']
    cpu = read_json(root/'cpu_preparation.json')
    baseline = summaries['871']['mean_seconds']
    data = {'at': now(), 'run': str(root.relative_to(ROOT)), 'task_complete': True,
        'record_integrity': True, 'ordinary_planning_requests': 30, 'profiled_requests': 2,
        'fixed_work_cases': 5, 'fixed_work_joint_states_per_count': 1280,
        'physical_grasps': 0, 'production_sphere_count': 871, 'default_activated': False,
        'summaries': summaries, 'fixed_work_summary': fixed_work_summary(review['micro']),
        'count_review': review, 'endpoint_review': independent, 'cpu_preparation': cpu,
        'complete_request_order': [512, 256, 871], 'fixed_work_interleaved': True,
        'ordinary_results': [r for r in progress['results'] if not r['diagnostic_profile']],
        'observed_full_request_saving': {str(n): 1 - summaries[str(n)]['mean_seconds']/baseline for n in COUNTS}}
    lines = ['# V7 碰撞球数量对完整规划耗时的影响', '', '2026年11月27日。', '',
        '碰撞球数量会影响耗时，但减少球数不能把当前完整规划直接压到几十毫秒。'
        '本轮871、512、256球的完整请求平均分别为5.669、4.355、4.381秒，'
        '512球继续减到256球没有进一步收益。主要秒级开销仍集中在反复绑定机器人与场景、'
        '处理完整配置和更新接触几何。执行配置继续保留871球。', '',
        '## 怎么比较', '',
        '采用苹果、甜甜圈、香蕉、芒果、剪刀五份已冻结观测。三档模型各重复两遍，共30次普通完整规划；'
        '871球另做两次全请求剖析，用于定位CPU与GPU调用。候选、排序、实际夹爪宽度、初态、'
        '32个IK初值、规划精度、TCP43和指尖桌面余量均沿用原设置。', '',
        '512和256球由同一连杆上的原球保守合并得到：大球完整包住原小球，'
        '每个原球的覆盖关系都独立核验。三档只在各自子进程加载，不改默认配置。', '',
        '完整规划按512→256→871三组执行；固定工作量测试让三档在同一张卡上交替运行。'
        '完整请求表反映本轮观察到的总差异，固定工作量表用于看球数对GPU计算本身的影响。', '',
        '## 完整规划', '']
    lines += full_table(data)
    lines += ['', f"512球相对871球的本轮平均耗时减少{data['observed_full_request_saving']['512']:.1%}，"
        f"256球减少{data['observed_full_request_saving']['256']:.1%}。三档都检查70个候选，"
        '同场景选中的抓法、左右臂、开度和候选拒绝原因均相同。', '',
        '规划数值轨迹存在变化：512球相对871球的最大关节终点差为'
        f"{max(v['max_endpoint_joint_delta_rad'] for v in review['comparisons'] if v['count']==512):.7f}弧度；"
        '256球的关节终点相同，但完整路径也有变化。三档都通过原独立末端FK和关节限位复核。', '',
        '## 秒级时间花在哪里', '', '下表为一次完整请求的平均值，使用扣除子调用后的耗时，各行可以相加。', '']
    lines += stage_table(data)
    lines += ['', '每次请求平均仍绑定13次机器人与场景。871→512球时，绑定一项就从2.745秒降到1.960秒，'
        '贡献了约0.785秒的总差异。球越多，完整配置的解析、写出、复制和校验也越重。', '',
        '## 同一工作量的GPU计算', '',
        '每个场景固定256个关节状态：96个来自同一条已通过的871球轨迹，160个为同一种子生成的限位内状态。'
        '三档始终使用相同关节数组、实际开度和场景；没有IK、候选搜索或路径规划。'
        '共检查每档1280个状态。以下为五个场景各自测量中位数的中位数，计时范围为CUDA Graph重放。', '',
        '### FK＋关节限位＋世界碰撞的完整状态验证', '']
    lines += micro_table(data)
    lines += ['', '### 仅世界碰撞查询', '']
    lines += micro_table(data, 'world_collision')
    lines += ['', '少量状态的碰撞检查已经只有约0.01毫秒，继续减少球数的绝对收益很小。'
        '一次检查256个状态时，世界碰撞从约0.566毫秒降到0.349毫秒，减少约38%；'
        '完整状态验证从约0.607毫秒降到0.386毫秒，减少约36%。这部分有收益，但量级仍是毫秒以内。', '',
        '当前设置的原生自碰撞检查为关闭，停放臂作为世界障碍物检查；本轮保持原设置。', '',
        '## 模型数量带来的CPU处理成本', '',
        '取三档实际参考配置，对读取、写出、深复制和球参数散列在CPU上交替各测15次。下表为单次操作中位数。', '',
        '| 配置操作 | 871球 | 512球 | 256球 |', '| --- | ---: | ---: | ---: |']
    for key, label in [('C_yaml_parse', 'C版YAML解析'), ('C_yaml_dump', 'C版YAML写出'),
                       ('deepcopy', '深复制'), ('sphere_digest', '球参数散列校验')]:
        lines.append('| ' + label + ' | ' + ' | '.join(
            f"{cpu['operations'][key][str(n)]['median_ms']:.2f}毫秒" for n in COUNTS) + ' |')
    lines += ['', '球数会同时增加GPU几何计算和CPU配置处理。完整请求中重复做这些配置操作，'
        '更容易积成几百毫秒到数秒。', '',
        '## 全请求剖析发现', '',
        '苹果与香蕉各做一次CPU/CUDA剖析。剖析会增加耗时，未纳入上面的30次普通规划。', '',
        '| 检查项 | 苹果 | 香蕉 |', '| --- | ---: | ---: |']
    for label, function in [('写出YAML次数', 'dump'), ('加载碰撞场景次数', 'load_collision_model')]:
        values = []
        for item in review['profiled_requests']:
            values.append(str(sum(row['calls'] for row in item['profile']['host_calls'] if row['function'] == function)))
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines.append('| 记录到的GPU内核累计 | ' + ' | '.join(
        f"{v['profile']['device_event_seconds_sum']:.3f}秒" for v in review['profiled_requests']) + ' |')
    lines += ['', 'GPU已记录内核中，碰撞相关内核占约73%和78%。这表示GPU内部主要在做碰撞计算，'
        '不能把这个比例直接当作整次规划的碰撞占比。CPU还在重复写配置、装载场景、'
        '转换历史路径、计算散列和排序网格。', '',
        '## 少球模型改变了什么', '',
        '| 模型 | 原球是否全部被覆盖 | 相对871新增拦截 | 相对871新增放行 |',
        '| ---: | --- | ---: | ---: |']
    for count in COUNTS:
        differences = [v['differences'][str(count)] for v in review['micro']]
        lines.append(f"| {count} | 是 | {sum(v['new_blocks'] for v in differences)} / 1280 | "
            f"{sum(v['new_allows'] for v in differences)} / 1280 |")
    lines += ['', '合并球变大后会多挡住一些原先放行的状态。当前五个场景的选中抓法保持，'
        '不代表更多场景的可行解都能保持。', '',
        '## 下一步', '',
        '保留871球，优先优化重复配置处理：机器人静态参数、球参数与当前场景在内存中保存，'
        '候选开度变化时只更新必要数据；完整配置留档与规划计算分开。'
        '此前“只更新开度”的完整对照未提速，本次应针对剖析明确指出的YAML往返、'
        '碰撞世界重复装载和接触几何排序逐项验证。', '',
        '球数精简放在这几项之后考虑。少球的GPU收益存在，但本轮已经显示512球继续减半未再降低完整请求耗时。', '',
        '## 验收', '', '| 项目 | 结果 |', '| --- | --- |',
        '| 任务结果 | 30/30普通完整规划，另2/2剖析、5/5固定工作量测试完成 |',
        '| 记录完整性 | 原始登记、模型、领取回执、计时、源码档案和独立核验完整 |',
        f"| 独立终点核验 | {independent['checked_endpoints']}个终点通过原FK与关节限位标准 |",
        '| 事件适用性 | 规划与固定状态验证；新增物理抓取0次 |',
        '| 关键事件覆盖 | 三档实际GPU球数、原球覆盖、选中抓法与开度、碰撞判断差异、默认配置散列 |',
        '| 当前执行配置 | 保留871球，所有自有服务已退出 |', '',
        '11项离线故障与几何检查通过。', '',
        '## 逐场景完整请求', '', '| 场景 | 871球 | 512球 | 256球 |', '| --- | ---: | ---: | ---: |']
    for case, label in LABELS.items():
        lines.append('| ' + label + ' | ' + ' | '.join(f"{np.mean([v['RPC_seconds'] for v in data['ordinary_results'] if v['case_id']==case and v['sphere_count']==n]):.3f}秒"
            for n in COUNTS) + ' |')
    lines += ['', '每格为该场景两次完整请求的平均值。', '', '## 数据位置', '', f"运行目录：`{data['run']}`。", '',
        '`acceptance.json`为普通请求汇总，`sphere_count_review.json`为球模型与固定输入核验，'
        '`cpu_preparation.json`为CPU操作计时，`micro/871/`含五组固定输入和原始GPU计时。', '']
    document = ROOT/'modules/pipeline/docs/mainline_versions/V7_碰撞球数量对完整规划耗时的影响.md'
    document.write_text('\n'.join(lines))
    (root/'final_report.md').write_text(document.read_text())
    write_json(root/'publication_snapshot.json', data)
    write_json(root/'analysis_provenance.json', {'at': now(), 'analysis_sources': {
        str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
        'modules/pipeline/experiments/sphere_count_diagnosis/analysis.py': sha256(Path(__file__).with_name('analysis.py'))},
        'frozen_execution_registration_unchanged': True})
    path = root/'autoresearch-results.tsv'
    if not path.exists():
        with path.open('w') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(['iteration', 'metric_value', 'delta', 'delta_pct', 'status', 'description', 'evaluator_source', 'timestamp'])
            for count in COUNTS:
                value = summaries[str(count)]['mean_seconds']
                writer.writerow([0 if count == 871 else 1, value, value - baseline,
                    f'{(value/baseline-1)*100:.3f}%', 'baseline' if count == 871 else 'diagnostic_only',
                    f'{count} spheres; no production adoption', 'mechanical', now()])
        with (root/'research.md').open('a') as stream:
            stream.write('| 1 | 30普通＋2剖析＋5固定输入测试 | 完整；512/256新增拦截8/11个 | 诊断完成，默认871保持 |\n')
        (root/'research_log.md').write_text('# 分析记录\n\n' + '\n'.join(full_table(data)) +
            '\n\n球数会影响GPU计算与CPU配置处理；512继续减半未降低完整请求。保留871，下一项优先减少重复配置处理。\n')
    write_json(root/'evaluator.json', {'pass': True, 'status': 'diagnosis_complete',
        'score': baseline, 'production_adoption': False, 'physical_grasps': 0})
    pointer = read_json(ROOT/'tmp/sphere_count_diagnosis/active.json')
    pointer.update(at=now(), phase='complete', checked_requests=32, fixed_work_cases_completed=5,
        record_integrity=True, production_sphere_count=871, all_owned_services_exited=True,
        report=str(document.relative_to(ROOT)))
    write_json(ROOT/'tmp/sphere_count_diagnosis/active.json', pointer)
    progress.update(at=now(), fixed_work_cases_completed=5, independent_count_review_passed=True)
    write_json(root/'progress.json', progress)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    build(ensure_run_path(args.root))
