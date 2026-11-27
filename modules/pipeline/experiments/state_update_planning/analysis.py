"""Aggregate mutually exclusive timings from completed, archived-source pilots."""
import argparse
from pathlib import Path

from arx_project.layout import ROOT, now, read_json, write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard


def create(root):
    guard(root, completed=True)
    data = read_json(root / 'publication_snapshot.json')
    complete = {}
    for mode in ['control', 'memory']:
        stages = data['exclusive_mean_stages'][mode]
        mean = data['acceptance']['summaries'][mode]['mean_seconds']
        complete[mode] = {
            'robot_scene_binding': sum(v for k, v in stages.items() if k == 'world_and_robot_bind' or k.startswith('binding_')),
            'contact_geometry_update': sum(v for k, v in stages.items() if k == 'contact_validator_update' or k.startswith('contact_')),
            'grasp_IK': stages.get('grasp_IK', 0.),
            'grasp_endpoint_other': stages.get('grasp_endpoint', 0.),
            'path_and_trajectory': sum(stages.get(k, 0.) for k in ['path_PRM', 'trajectory_pregrasp', 'trajectory_approach']),
            'opening_and_narrowing': sum(stages.get(k, 0.) for k in ['gripper_opening', 'gripper_narrowing']),
            'other_recorded_stages': sum(v for k, v in stages.items() if k in ['pregrasp_IK', 'lift_IK', 'candidate_search',
                'candidate_width_update', 'memory_request_admission', 'memory_receipt_drain_and_validation']),
            'outside_recorded_stages': mean - sum(stages.values()), 'complete_RPC': mean}
        if abs(sum(v for k, v in complete[mode].items() if k != 'complete_RPC') - mean) > 1e-8:
            raise RuntimeError('Timing categories overlap or omit costs')
    analysis = {'at': now(), 'exclusive_complete_groups': complete, 'default_activated': False,
                'first_pilot_requests': 20, 'revised_pilot_requests': 20, 'total_requests': 40,
                'new_physical_grasps': 0}
    write_json(root / 'post_analysis.json', analysis)
    data['work_analysis'] = analysis
    write_json(root / 'publication_snapshot.json', data)
    names = [('robot_scene_binding', '机器人与场景绑定合计'), ('contact_geometry_update', '接触几何更新合计'),
        ('grasp_IK', '原抓取IK'), ('grasp_endpoint_other', '终点补充检查与回退'),
        ('path_and_trajectory', '路径搜索与轨迹优化'), ('opening_and_narrowing', '开爪与原地收爪'),
        ('other_recorded_stages', '其他已分项的IK、调度与核验'),
        ('outside_recorded_stages', '其余未分项请求开销'), ('complete_RPC', '完整请求合计')]
    lines = ['# V8 内存状态更新后的完整耗时与下一步', '',
        '修订版在共享GPU2完成20次同期对照。原方法和新方法各10/10规划通过，'
        '原抓法、手臂、关节终点与记录轨迹逐位一致。完整请求平均7.022→4.403秒，减少37.29%。', '',
        '## 完整平均耗时', '', '| 环节 | 原方法 | 内存更新修订版 |', '| --- | ---: | ---: |']
    for key, name in names:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{complete[m][key]:.3f}秒' for m in ['control', 'memory']) + ' |')
    lines += ['', '上表互斥项相加等于完整请求。机器人绑定和接触几何更新合计3.940→0.979秒，减少75.16%。'
        '其余未分项开销按完整请求减去已记录的互斥分项计算，下一阶段需要继续拆开，不能把它视为已经消失。', '',
        '## 场景工作量', '',
        '两版内存更新都实际消费98个候选。第一版10次请求重新装载124套场景；修订版为60套，'
        '并完成192次障碍位姿更新和154次原生启停标志更新。原碰撞判断和实时关节状态仍然检查。', '',
        '第一版仍逐次把约103KB的模型字典转成YAML。修订版预先生成不变文本，运行时只编码两个夹爪开度，'
        '后台负责写文件，并在结束请求前等到全部完成。', '',
        '## 逐场景完整耗时', '', '| 场景 | 原方法两次平均 | 修订版两次平均 |', '| --- | ---: | ---: |']
    for case in dict.fromkeys(r['case_id'] for r in data['results']):
        values = {m: sum(r['RPC_seconds'] for r in data['results'] if r['case_id'] == case and r['mode'] == m) / 2
                  for m in ['control', 'memory']}
        lines.append(f"| {case} | {values['control']:.3f}秒 | {values['memory']:.3f}秒 |")
    lines += ['', '## 下一步优先级', '',
        '| 顺序 | 内容 | 要解决的耗时 |', '| --- | --- | --- |',
        '| 1 | 把请求内的状态映射、候选交接、记录提交及未分项GPU等待分别计时 | 当前其余未分项开销约1.097秒 |',
        '| 2 | 在已验证的内存更新上，比较首候选直接执行、失败后再按4／16扩批的IK策略；保持原精度、开度、评分顺序与回退 | 当前抓取IK约1.275秒 |',
        '| 3 | 继续减少首次场景装载和多套接触视图的重复工作，保留当前碰撞几何 | 当前绑定与接触更新仍约0.979秒 |', '',
        '当前仍未达到一秒。下一步以整次请求衡量收益，同时保留原成功和完整路径核验。'
        '本阶段两个版本累计40次完整纯规划，120个终点和80条记录轨迹核验通过；默认抓取入口保持。', '',
        f"运行记录：`{root.relative_to(ROOT)}`。"]
    doc = ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存状态更新剩余耗时.md'
    doc.write_text('\n'.join(lines) + '\n')
    write_json(root / 'analysis_complete.json', {'at': now(), 'passed': True,
        'document': str(doc.relative_to(ROOT)), 'new_GPU_calls': 0, 'new_planning_requests': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    create(p.parse_args().root.resolve())
