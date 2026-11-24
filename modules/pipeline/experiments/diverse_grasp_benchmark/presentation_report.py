"""Human-readable reporting from completed receipts, separate from frozen execution code."""
from collections import Counter
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REASONS = {
    'sam3_no_target_mask': ('SAM3目标分割', '没有分割出目标物体'),
    'rayst3r_empty_pointcloud': ('RaySt3R点云生成', '没有生成足够的目标点云'),
    'no_feasible_native_post_or_recalled_candidate': ('cuRobo规划', '规划不出来'),
    'measured_start_state_in_collision': ('cuRobo规划', '规划不出来：机器人当前状态存在碰撞'),
    'measured_pregrasp_approach_unreachable': ('接近动作规划', '规划不出来：无法从实测状态接近目标'),
    'measured_closure_lift_unreachable': ('抬升动作规划', '规划不出来：夹爪闭合后无法规划抬升'),
    'observation_path_failed': ('观察动作规划', '规划不出来：无法到达观察位置'),
    'no_bilateral_target_contact': ('实际抓取', '双指没有形成有效夹持'),
    'insufficient_hold_height': ('实际抓取', '物体抬升或保持高度未达标'),
    'forbidden_robot_contact': ('实际抓取', '发生禁止接触'),
    'Target has fewer than 50 visible pixels': ('图像观察', '目标的可见像素不足50个'),
}


def describe(reason):
    if reason in REASONS:
        return REASONS[reason]
    if reason.endswith('_not_converged'):
        stage = reason.removesuffix('_not_converged')
        name = {'observe': '观察', 'pregrasp': '预抓取', 'approach': '接近',
                'close': '闭爪', 'lift': '抬升', 'hold': '保持'}.get(stage, stage)
        return '动作执行', name + '动作未到位'
    return '执行记录', reason


def render(root):
    root = Path(root)
    reg = json.loads((root / 'registration.json').read_text())
    rows, interrupted = [], []
    for file in sorted((root / 'claims').glob('*.json')):
        claim = json.loads(file.read_text())
        if claim['state'] != 'complete':
            interrupted.append(file.stem)
            continue
        result = root / 'trials' / file.stem / 'result.json'
        assert hashlib.sha256(result.read_bytes()).hexdigest() == claim['result_sha256']
        rows.append(json.loads(result.read_text()))
    reasons = Counter(row['reason'] for row in rows if not row['success'])
    phase = json.loads((root / 'progress.json').read_text())['phase']
    status = {'stopped_by_user': '已按用户要求停止', 'complete': '测试已完成',
              'halted': '发生运行或记录故障，已停止'}.get(phase, '测试进行中')
    n, completed = len(reg['schedule']), len(rows)
    planned = sum(bool(row['planning_success']) for row in rows)
    attempted = sum(bool(row['physical_attempt']) for row in rows)
    successes = sum(bool(row['success']) for row in rows)
    not_started = n - completed - len(interrupted)
    lines = ['# 独立采样200场景抓取测试', '', '当前状态：' + status + '。', '',
             '| 项目 | 数量 |', '| --- | ---: |', f'| 已完成测试 | {completed}/{n} |',
             f'| SAM3没有分割出目标物体 | {reasons["sam3_no_target_mask"]} |',
             f'| RaySt3R没有生成足够的目标点云 | {reasons["rayst3r_empty_pointcloud"]} |',
             f'| 规划出可执行轨迹 | {planned} |',
             f'| 规划不出来 | {reasons["no_feasible_native_post_or_recalled_candidate"]} |',
             f'| 实际执行抓取 | {attempted} |', f'| 实际抓取成功 | {successes} |',
             f'| 实际抓取失败 | {attempted-successes} |',
             f'| 停止时中断、尚无完整结果 | {len(interrupted)} |',
             f'| 尚未开始 | {not_started} |', '',
             '| 失败发生在哪一步 | 具体情况 | 次数 |', '| --- | --- | ---: |']
    for reason, count in reasons.items():
        stage, label = describe(reason)
        lines.append(f'| {stage} | {label} | {count} |')
    lines += ['', '感知没有产出有效输入、规划不出来、实际执行后没抓住，分别记录。中断和尚未开始的场景单独统计。', '',
              '抓取成功要求物体实际抬高至少1厘米并保持3秒，同时双指接触、执行收敛和禁止接触检查达标。', '',
              f'原始记录：`{root}`。']
    text = '\n'.join(lines) + '\n'
    (root / 'report.md').write_text(text)
    (PROJECT / 'docs/diverse_grasp_benchmark/抓取测试结果.md').write_text(text)
    summary = {'registered': n, 'completed': completed, 'planning_passes': planned,
               'physical_attempts': attempted, 'successes': successes, 'interrupted': interrupted,
               'not_started': not_started, 'failure_reasons': dict(reasons)}
    (root / 'presentation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    print(json.dumps(render(parser.parse_args().root), ensure_ascii=False))
