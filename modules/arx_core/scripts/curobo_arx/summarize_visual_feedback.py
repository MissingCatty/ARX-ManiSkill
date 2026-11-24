"""Compare actual ARX feedback attempts to their explicit initial-proposal pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from summarize_failure_scenarios import summarize


def read(path):
    return json.loads(path.read_text())


def compare(out):
    summarize(out)
    report = read(out / 'report.json')
    raw = read(out / 'results.json')
    rows = []
    for value, row in zip(raw, report['rows']):
        folder = Path(value['folder'])
        cfg = read(folder / 'trajopt_effective_config.json')
        row['execution_policy'] = value.get('execution_policy', 'baseline')
        row['bound_weight'] = cfg['rollout']['cost_cfg']['cspace_cfg']['weight'][0]
        row['total_trial_wall_seconds'] = value['finished_unix'] - value['started_unix']
        row['stage_convergence'] = value.get('stage_convergence', {})
        row['integration_failure'] = row['outcome'] == 'infrastructure_or_input_failure'
        if 'outside the validated model range' in (row.get('error') or ''):
            row['original_outcome'] = row['outcome']
            row['outcome'] = 'measured_gripper_model_state_failure'
            row['integration_failure'] = True
        if (folder / 'gripper_model_projection.json').exists():
            row['planning_gripper_state_projection'] = read(folder / 'gripper_model_projection.json')
        if (folder / 'visual_feedback_report.json').exists():
            feedback = read(folder / 'visual_feedback_report.json')
            row['feedback'] = {'observations': feedback['observations'], 'updates': []}
            for update in feedback['refinements']:
                seed, chosen = np.asarray(update['initial_pose']), np.asarray(update['chosen_pose'])
                row['feedback']['updates'].append({
                    'iteration': update['iteration'], 'pose_changed': update['pose_changed'],
                    'translation_update_m': float(np.linalg.norm(chosen[:3, 3] - seed[:3, 3])),
                    'seed_geometric_quality': update['seed_geometric_quality'],
                    'candidate_rejections': dict(Counter(c['reason'] for c in update['checked_candidates'] if c['reason'])),
                    'wall_seconds': update['wall_seconds']})
        if value.get('paired_initial_proposal'):
            path = Path(value['paired_initial_proposal'])
            seed = read(folder / ('selected_grasp_initial.json' if (folder / 'selected_grasp_initial.json').exists() else 'selected_grasp.json'))
            source = read(path)
            same = all(np.allclose(seed[key], source[key], atol=1e-9, rtol=0)
                       for key in ('pregrasp_world', 'grasp_world', 'q_pre', 'network_row'))
            initial = read(folder / 'capture_snapshot.json')
            original = read(path.parent / 'capture_snapshot.json')
            target = row['target']
            first = np.asarray(initial['objects'][target]['world'])
            second = np.asarray(original['objects'][target]['world'])
            first_base = np.asarray(initial['link_world']['body4'])
            second_base = np.asarray(original['link_world']['body4'])
            first_arm = [initial['joint_positions'][initial['joint_names'].index(n)] for n in initial['right_arm']]
            second_arm = [original['joint_positions'][original['joint_names'].index(n)] for n in initial['right_arm']]
            row['paired_initial_proposal'] = {
                'source': str(path), 'identical_initial_proposal': bool(same),
                'initial_object_position_difference_m': float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
                'initial_base_position_difference_m': float(np.linalg.norm(first_base[:3, 3] - second_base[:3, 3])),
                'initial_arm_max_difference_rad': float(np.max(np.abs(np.asarray(first_arm) - second_arm)))}
        rows.append(row)
    by_name = {r['name']: r for r in rows}
    pairs = []
    for value, row in zip(raw, rows):
        source = value['scenario'].get('paired_baseline')
        if not source or row['execution_policy'] == 'baseline':
            continue
        name = Path(source).name
        paired = by_name.get(name)
        if paired:
            pairs.append({'asset': row['asset'], 'baseline_trial': name, 'feedback_trial': row['name'],
                          'baseline_success': paired['success'], 'feedback_success': row['success'],
                          'baseline_outcome': paired['outcome'], 'feedback_outcome': row['outcome'],
                          'same_initial_proposal': row['paired_initial_proposal']['identical_initial_proposal'],
                          'eligible_method_comparison': not row['integration_failure']})
    comparison = {'implementation': 'VFAS-inspired stage-boundary visual feedback with geometric contact proxy',
                  'original_VFAS_evaluator': False, 'GNP_amortized_inference_trained': False,
                  'interpretation': 'Exploratory fixed-proposal comparisons; not a reproduction of the papers or an unbiased success-rate benchmark. Approach splitting/replanning is part of the treatment.',
                  'trials': len(rows), 'rows': rows, 'pairs': pairs}
    (out / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    text = ['ARX 视觉反馈与局部自适应采样试验 2026-10-19', '',
            '方法范围：借鉴 VFAS 的新 RGB-D 反馈、局部 SE3 扰动、自适应采样范围/数量、姿态连续性和噪声敏感性惩罚。',
            '公开仓库未提供要求的评分网络权重，本实验以 ARX 指尖几何和观察点云的两侧表面支持代替；不是原论文网络复现。',
            '在预抓取和靠近中途各观测一次，尚非论文 20 Hz 视觉伺服，也没有使用 GMFlow。',
            '腕部相机按需开启，使用已有序列号接口取图；观测后关闭，动作时仍只启用原有头部 RTX 传感器。',
            'Amortized Inference 的 GNP 编码器/解码器需要预训练；本次没有训练或声称验证该方法。', '',
            '控制：原执行器、轨迹优化、接触/到位检查、质量/摩擦、控制增益、成功判据保持不变，边界惩罚 10000。',
            '改进版增加靠近中途停顿与重新规划；保留现有实验的目标真值辅助分割，只用观测点云评分，不读取质量、摩擦或目标网格评分。',
            '每组复用相同初始 ZeroGrasp 提案，反馈图像重新采集；物体初始实测微小差异见 comparison.json。', '',
            '新版筛选：每轮先检查 ARX 可执行性，仅围绕已通过检查的姿态继续采样；补充 3 mm / 3 度小扰动，并允许绕夹持轴调整接近倾角。',
            '新版建模输入：将不超过 20 微米的夹爪实测数值越界投影回 USD 限位，独立重建验证；原始实测状态保留，实际物理限位和动作命令不改，更大越界拒绝。', '',
            '最后的绿色球接入验证：候选还需通过 9 个开合宽度采样检查，提前拒绝名义宽度可行但实测闭爪宽度碰撞的姿态；桌面仍保留原 8 mm 规划余量。', '',
            '实际试验结果']
    for row in rows:
        hold = row['stages'].get('hold', {})
        peak = row['stages'].get('lift', {}).get('object_max_rise_m')
        text.append(f"{row['name']} / {row['execution_policy']}：{'成功' if row['success'] else '失败'}，{row['outcome']}。")
        if row['outcome'] == 'measured_gripper_model_state_failure':
            text.append('  夹爪建模状态护栏中止，未执行抬升；不作为完整抓取效果对照。')
        elif row['integration_failure']:
            text.append('  接口集成失败，完整保留记录；不计入有效方法对照。')
        if row.get('planning_gripper_state_projection'):
            projection = row['planning_gripper_state_projection']
            text.append(f"  夹爪规划模型输入最大投影 {max(projection['projection_m'].values())*1e6:.2f} 微米；物理限位/执行命令未改。")
        if peak is not None:
            text.append(f"  抬升阶段中心最高增量 {peak*100:.2f} cm；保持阶段最小增量 {hold['object_min_rise_m']*100:.2f} cm。")
        text.append('  视频：' + row['video'])
        if row.get('paired_initial_proposal'):
            check = row['paired_initial_proposal']
            text.append(f"  初始提案完全一致：{check['identical_initial_proposal']}；初始物体位置差 {check['initial_object_position_difference_m']*1000:.4f} mm。")
        if row.get('feedback'):
            text.append('  已保存反馈次数：' + str(len(row['feedback']['updates'])))
    text += ['', '配对结果']
    for pair in pairs:
        text.append(f"{pair['asset']}：基线 {pair['baseline_trial']} {'成功' if pair['baseline_success'] else '失败'} -> 反馈 {pair['feedback_trial']} {'成功' if pair['feedback_success'] else '失败'}。")
    text += ['', '新增成功的基线也完整保留，不能沿用以前两次失败就把物体称为必败。',
             '几何评分不是校准的成功概率，评分提高不等于实际抓取成功；以接触、物体高度和最终保持判断。',
             'comparison.json 含逐试次测量、初始提案一致性、观测输入和候选拒绝原因；method_review.json 含论文与代码来源。',
             '每次 /feedback/ 中保存头部/腕部 RGB-D、掩码、点云/法向量和所有采样候选。',
             '原始视频沿用用户 WebRTC 视角，无文字、标题或音频；只删除规划等待。', '',
             'VFAS 论文：https://arxiv.org/pdf/2310.18459',
             'VFAS 官方代码：https://github.com/kracon7/vfas_code_release',
             'Amortized Inference 论文：https://groups.csail.mit.edu/rrg/papers/noseworthy_shaw_icra24.pdf']
    (out / 'readme_zh.txt').write_text('\n'.join(text) + '\n')
    print(json.dumps({'trials': len(rows), 'pairs': pairs}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    compare(parser.parse_args().output.resolve())
