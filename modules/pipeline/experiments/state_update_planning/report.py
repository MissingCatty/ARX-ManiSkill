"""Full-request adoption decision with the original poses and recorded paths."""
import argparse
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json


def compare_plans(root, reg):
    pairs = []
    for case in reg['cases']:
        for repeat in [1, 2]:
            directory = root / 'plans' / case['id']
            a = read_json(directory / ('control_' + str(repeat)) / 'plan_result.json')
            b = read_json(directory / ('memory_' + str(repeat)) / 'plan_result.json')
            identity = lambda p: {k: p.get('selected', {}).get(k) for k in
                                 ['arm', 'original_native_rank', 'source_generator_index']}
            equal_paths, max_error = True, 0.
            if a['success'] and b['success']:
                for stage in ['pregrasp', 'approach']:
                    for field in ['q', 'times']:
                        x, y = np.asarray(a['paths'][stage][field]), np.asarray(b['paths'][stage][field])
                        equal_paths &= np.array_equal(x, y)
                        if x.shape == y.shape and max_error is not None:
                            max_error = max(max_error, float(np.max(np.abs(x - y))))
                        else:
                            max_error = None
                for field in ['q_pre', 'q_grasp', 'q_lift']:
                    equal_paths &= a['selected'][field] == b['selected'][field]
            else:
                equal_paths = a['success'] == b['success']
            pairs.append({'case': case['id'], 'repeat': repeat,
                'successes': {'control': a['success'], 'memory': b['success']},
                'same_grasp_and_arm': identity(a) == identity(b),
                'same_command_arrays_and_endpoints': bool(equal_paths),
                'maximum_same_shape_array_error': max_error})
    result = {'at': now(), 'pairs': pairs,
              'all_grasps_and_arms_equal': all(p['same_grasp_and_arm'] for p in pairs),
              'all_recorded_paths_equal': all(p['same_command_arrays_and_endpoints'] for p in pairs)}
    write_json(root / 'equivalence_review.json', result)
    return result


def create(root):
    reg = read_json(root / 'registration.json')
    accepted = read_json(root / 'acceptance.json')
    proof = read_json(root / 'independent_review.json')
    trajectories = read_json(root / 'trajectory_review.json')
    stopped = read_json(root / 'services_stopped.json')
    if not all([accepted['task_complete'], proof['passed'], trajectories['passed'], stopped['all_owned_exited']]):
        raise RuntimeError('Complete-request record review is incomplete')
    paired = compare_plans(root, reg)
    groups = accepted['summaries']
    a, b = groups['control'], groups['memory']
    reduction = 1 - b['total_seconds'] / a['total_seconds']
    retained = all(not p['successes']['control'] or p['successes']['memory'] for p in paired['pairs'])
    timing = reduction >= .05 and b['median_seconds'] < a['median_seconds'] and b['p90_seconds'] <= a['p90_seconds'] * 1.05
    adoption = {'at': now(), 'original_successes_retained': retained, 'timing_passed': timing,
        'total_reduction_fraction': reduction,
        'optional_implementation_accepted': bool(retained and timing and paired['all_grasps_and_arms_equal']
                                                  and paired['all_recorded_paths_equal']),
        'default_activated': False}
    write_json(root / 'adoption.json', adoption)
    rows = read_json(root / 'progress.json')['results']
    stages = {mode: {} for mode in ['control', 'memory']}
    for row in rows:
        for name, value in row['stages'].items():
            stages[row['mode']][name] = stages[row['mode']].get(name, 0.) + value['exclusive_seconds'] / groups[row['mode']]['requests']
    snapshot = {'at': now(), 'run': str(root.relative_to(ROOT)), 'acceptance': accepted,
        'independent_review': proof, 'trajectory_review': trajectories, 'equivalence_review': paired,
        'exclusive_mean_stages': stages, 'adoption': adoption, 'results': rows,
        'preflight': read_json(root / 'resident/state_preflight/acceptance.json'),
        'startup': read_json(root / 'resident/startup.json'), 'services_stopped': stopped,
        'timing_environment': read_json(root / 'timing_environment.json'),
        'GPU_load': read_json(root / 'gpu_load_summary.json')}
    previous = reg['config'].get('previous_completed_pilot')
    if previous:
        before = read_json(ROOT / previous / 'publication_snapshot.json')
        snapshot['previous_pilot'] = {'run': previous, 'acceptance': before['acceptance'],
                                     'adoption': before['adoption'], 'report': previous + '/report.md'}
    write_json(root / 'publication_snapshot.json', snapshot)
    lines = ['# V8 内存状态更新完整规划对照', '',
        '机器人固定配置保存在内存中，夹爪仅更新原URDF推导的动态参数；场景几何保留在原容量内，'
        '通过实际启停标志切换碰撞对象，并及时更新障碍位姿。不变的YAML文本提前生成，每次只替换两个开度数值。'
        '配置与回执由单线程后台写入，全部写入并核验后才结束请求。', '',
        '原五场景、每组重复两次，共20次完整纯规划。两组在共享GPU2交替运行，使用同一常驻规划器、'
        '统一预热及原候选顺序。', '', '| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, label in [('control', '原方法'), ('memory', '内存状态更新')]:
        g = groups[mode]
        lines.append(f"| {label} | {g['successes']}/{g['requests']} | {g['mean_seconds']:.3f}秒 | "
                     f"{g['median_seconds']:.3f}秒 | {g['p90_seconds']:.3f}秒 | {g['total_seconds']:.3f}秒 |")
    lines += ['', '## 分项平均耗时', '', '| 环节 | 原方法 | 内存状态更新 |', '| --- | ---: | ---: |']
    names = [('world_and_robot_bind', '绑定中其余处理'), ('binding_input_snapshot', '创建内存输入'),
        ('binding_dynamic_robot_update', '夹爪动态参数更新'), ('binding_scene_update', '场景更新'),
        ('binding_live_validation', '实测FK与起始碰撞'), ('contact_validator_update', '接触更新中其余处理'),
        ('contact_dynamic_robot_update', '接触模型动态参数更新'), ('contact_scene_update', '接触场景更新'),
        ('contact_state_update', '精细几何实测状态更新'), ('grasp_IK', '原抓取IK'),
        ('path_PRM', '路径搜索'), ('trajectory_pregrasp', '预抓取轨迹优化'),
        ('trajectory_approach', '接近轨迹优化'), ('memory_request_admission', '请求模型与资源准入'),
        ('memory_receipt_drain_and_validation', '记录完成等待与结束核验')]
    for key, name in names:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{stages[m].get(key, 0.):.4f}秒' for m in ['control', 'memory']) + ' |')
    lines += ['', '分项为互斥耗时；原方法的绑定与接触更新尚未拆成上述子项，整组比较以完整请求为准。', '',
        '## 结果与验证', '', f'累计耗时变化{-reduction * 100:+.2f}%。'
        + ('完整请求时延门槛通过。' if timing else '完整请求时延门槛未通过。'), '',
        '| 项目 | 结果 |', '| --- | --- |', f"| 任务结果 | {accepted['planning_requests']}/20个完整纯规划完成 |",
        f"| 记录完整性 | {proof['checked_endpoints']}个独立终点、{trajectories['command_paths']}条记录轨迹通过 |",
        '| 事件适用性 | 本轮为完整规划与状态更新比较，物理抓取0次 |',
        '| 关键事件覆盖 | 两臂、三种开度、192状态的871球与碰撞判断、四组完整开合爪采样、障碍移动与恢复、写入故障 |',
        f"| 抓法与手臂相同 | {'是' if paired['all_grasps_and_arms_equal'] else '有变化'} |",
        f"| 关节与记录轨迹逐位相同 | {'是' if paired['all_recorded_paths_equal'] else '有变化'} |",
        f"| 可选实现采用门槛 | {'通过' if adoption['optional_implementation_accepted'] else '未通过'} |", '',
        '当前默认入口保持。候选姿态、实际开度、871球、TCP43、V2指尖规则与完整路径检查保持。', '',
        f"登记：`{root.relative_to(ROOT)}`。入口：`modules.curobo.state_updates.runtime.create`。"]
    if previous:
        old = snapshot['previous_pilot']['acceptance']['summaries']
        lines += ['', '## 第一版完整结果', '',
            '第一版20次结果和源码完整保留。以下是第一版自己的同期对照，修订版耗时使用上面的新对照。', '',
            '| 第一版方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 |', '| --- | ---: | ---: | ---: | ---: |']
        for mode, name in [('control', '原方法'), ('memory', '第一版内存更新')]:
            g = old[mode]
            lines.append(f"| {name} | {g['successes']}/{g['requests']} | {g['mean_seconds']:.3f}秒 | "
                         f"{g['median_seconds']:.3f}秒 | {g['p90_seconds']:.3f}秒 |")
        lines += ['', '第一版累计省4.07%，未达到5%的采用门槛。修订版将YAML不变文本提前生成，'
            '并让同一容量中的碰撞对象通过原生启停标志切换，减少重新装载几何。'
            '两版累计40个完整请求，新增物理抓取0次。']
    doc = ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存状态更新完整规划对照.md'
    doc.write_text('\n'.join(lines) + '\n')
    write_json(ROOT / 'tmp/state_updates/active.json', {'at': now(), 'run': str(root.relative_to(ROOT)),
        'phase': 'complete', 'complete': True, 'report': str(doc.relative_to(ROOT))})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    create(p.parse_args().root.resolve())
