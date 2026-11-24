"""Summarize a completed ARX EGAD49 run from its original execution artifacts."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(trial,result):
    network=trial/'network/logs/stderr.log'
    stderr=network.read_text(errors='replace') if network.exists() else ''
    error=result['evaluation'].get('reported_error') or result.get('error') or ''
    if 'OutOfMemoryError' in stderr:return 'shared_gpu_out_of_memory',stderr.splitlines()[-1]
    if result['success']:return 'success',''
    if result['status']=='timeout':return 'pipeline_timeout',error
    if 'No bilateral target contact' in error:return 'closure_without_bilateral_contact',error
    if 'exhausted its action budget' in error:return 'local_action_budget_exhausted',error
    if 'Forbidden contact' in error:return 'forbidden_execution_contact',error
    if 'convergence' in error:return 'execution_convergence_rejected',error
    if 'IK rejected' in error or 'collision' in error or 'collision-free' in error:
        return 'local_ik_or_path_rejected',error
    if 'reopened' in error:return 'policy_reopened_after_closure',error
    if result['evaluation'].get('bilateral_closure'):
        return 'lift_or_hold_failed',error
    client=trial/'client_error.json'
    if not error and client.exists():error=str(read(client).get('error',''))
    return 'other_pipeline_failure',error or stderr[-1200:]


def summarize(campaign,historical):
    scenarios=read(campaign/'scenarios.json')
    ids=[r['egad_id'] for r in scenarios]
    expected=[f'{letter}{number}' for letter in 'ABCDEFG' for number in range(7)]
    if sorted(ids)!=expected or len(set(ids))!=49:raise ValueError('Need exactly A0..G6 once')
    registration=read(campaign/'registration.json')
    if digest(campaign/'scenarios.json')!=registration['scenario_sha256']:
        raise ValueError('Registered scenario bytes changed')
    current_changes=[name for name,value in registration['source_sha256'].items() if digest(name)!=value]
    archived_changes=[name for name,value in registration['source_sha256'].items()
        if digest(campaign/'registered_source_snapshot'/Path(name).relative_to(ROOT))!=value]
    if archived_changes:raise ValueError('Registered source archive changed: '+str(archived_changes))
    state=read(campaign/'state.json')
    if state['status']!='completed' or state['completed']!=49:
        raise ValueError('Wait for all 49 scored attempts to finish')
    # Capture the end-of-run check before any later, separately committed fixes.
    completion=campaign/'completion_source_check.json'
    if not completion.exists():
        if current_changes:raise ValueError('Frozen sources changed before finalization: '+str(current_changes))
        completion.write_text(json.dumps({'checked_unix':time.time(),
            'registration_sha256':digest(campaign/'registration.json'),
            'finished_unix':state['finished_unix'],'completed':49,
            'all_registered_source_hashes_match':True},indent=2)+'\n')
    evidence=read(completion)
    if (evidence['registration_sha256']!=digest(campaign/'registration.json')
            or evidence['finished_unix']!=state['finished_unix']
            or evidence['all_registered_source_hashes_match'] is not True):
        raise ValueError('Completion evidence does not match the frozen run')
    spec=importlib.util.spec_from_file_location('registered_egad_evaluator',
        campaign/'registered_source_snapshot/scripts/curobo_arx/lilo_vla/egad_evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    evaluate=module.evaluate
    old=read(historical/'benchmark_results.json')
    pairs={row['id']:row for row in old['pairs']}
    rows=[];inference_times=[];pose_components=0;pose_at_clip_boundary=0
    evaluation_matches=True
    for scenario in scenarios:
        trial=campaign/'trials'/scenario['name']
        result=read(trial/'result.json')
        if result['egad_id']!=scenario['egad_id']:raise ValueError('Mismatched case ID')
        independent=evaluate(trial)
        if independent!=result['evaluation']:evaluation_matches=False
        if bool(independent['pass'])!=result['success']:raise ValueError('Result disagrees with native logs')
        setup=read(trial/'setup_verification.json')
        category,error=classify(trial,result)
        video=trial/'bottle_grasp_no_obstacle_action.mp4'
        history=pairs[scenario['egad_id']]
        queries=list(trial.glob('local_*_prediction.json'))
        for query in queries:
            prediction=read(query)
            inference_times.append(float(prediction['inference_seconds']))
            normalized=prediction['normalized_state'][:6]
            pose_components+=len(normalized)
            pose_at_clip_boundary+=sum(abs(value)>=1.-1e-8 for value in normalized)
        commands=list(trial.glob('*_command.json'))
        observations=trial/'capture_snapshot.json'
        row={'egad_id':scenario['egad_id'],'success':result['success'],'failure_category':category,
            'setup_verified':setup['pass'],'fresh_head_observation':observations.exists(),
            'zerograsp_completed':bool(list((trial/'network').glob('**/*.grasp.npy'))),
            'actual_execution_count':len(list(trial.glob('*_execution.json'))),
            'command_count':len(commands),'lilo_model_calls':len(queries),
            'bilateral_closure':independent.get('bilateral_closure',False),
            'hold_min_rise_m':independent.get('hold_min_rise_m'),
            'recorded_hold_duration_s':independent.get('recorded_hold_duration_s',0.),
            'forbidden_contact_events':independent.get('forbidden_contact_events',0),
            'elapsed_s':result['elapsed_s'],'historical_baseline_success':history['baseline'],
            'historical_h022_success':history['current'],'error':error,
            'video':str(video) if video.exists() else None,'trial':str(trial),
            'result_sha256':digest(trial/'result.json'),'setup_sha256':digest(trial/'setup.json')}
        rows.append(row)
    if not evaluation_matches:raise ValueError('Independent native log re-evaluation changed')
    if not all(r['setup_verified'] for r in rows):raise ValueError('Historical reset mismatch')
    successes=sum(r['success'] for r in rows)
    counts=dict(Counter(r['failure_category'] for r in rows))
    comparisons={}
    for label,field in [('baseline','historical_baseline_success'),('H022','historical_h022_success')]:
        comparisons[label]={'historical_successes':sum(r[field] for r in rows),
            'preserved_successes':[r['egad_id'] for r in rows if r[field] and r['success']],
            'gains':[r['egad_id'] for r in rows if not r[field] and r['success']],
            'losses':[r['egad_id'] for r in rows if r[field] and not r['success']],
            'both_failed':[r['egad_id'] for r in rows if not r[field] and not r['success']]}
    ordered_times=sorted(inference_times)
    summary={'generated_unix':time.time(),'campaign':str(campaign),'complete':True,
        'method':registration['method'],'attempts':49,'successes':successes,'success_rate':successes/49,
        'failure_categories':counts,'native_log_evaluations_identical':evaluation_matches,
        'all_initial_configuration_checks_passed':True,'source_hashes_unchanged_at_completion':True,
        'registered_source_snapshot_verified':True,
        'actual_motion_attempts':sum(r['actual_execution_count']>0 for r in rows),
        'model_calls':sum(r['lilo_model_calls'] for r in rows),
        'model_forward_p50_s':statistics.median(ordered_times) if ordered_times else None,
        'model_forward_p95_s':ordered_times[max(0,math.ceil(.95*len(ordered_times))-1)] if ordered_times else None,
        'normalized_pose_components_at_clip_boundary_fraction':pose_at_clip_boundary/pose_components if pose_components else None,
        'median_trial_elapsed_s':statistics.median(r['elapsed_s'] for r in rows),
        'comparisons_to_transferred_historical_results':comparisons,
        'scope':'One attempt per object; all failures including shared-resource OOM remain in denominator 49. Historical controls were transferred, not rerun locally. Frozen bottle-trained action head; no EGAD training or per-object retries.',
        'rows':rows}
    (campaign/'final_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (campaign/'final_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ARX EGAD49：冻结的 LiLo-VLA 接入测试','',
        f'完成 49/49 个物体，各一次尝试；成功 **{successes}/49（{successes/49:.1%}）**。',
        '没有重试，没有删除失败；规划、超时、空抓、掉落和显存故障均保留在分母。','',
        '| 方法 | 成功 / 尝试 | 成功率 |','|---|---:|---:|',
        '| 历史原始 ZeroGrasp + cuRobo | 26/49 | 53.1% |',
        '| 历史 H022 综合链路（实际均选择 H011 分支） | 28/49 | 57.1% |',
        f'| 本轮 ZeroGrasp + cuRobo 接近 + LiLo-VLA 局部策略 | {successes}/49 | {successes/49:.1%} |','',
        '历史两项来自已传输的 2026-10-20 评测记录，本轮没有重新运行对照组。每物体仅一次，不估计重复稳定性。','',
        '## 失败阶段','', '| 分类 | 次数 |','|---|---:|']
    labels={'shared_gpu_out_of_memory':'共享 A100 显存不足（基础设施）',
        'closure_without_bilateral_contact':'闭合 3 秒后没有双指接触',
        'local_action_budget_exhausted':'90 次局部动作预算耗尽',
        'forbidden_execution_contact':'实际执行出现禁止接触',
        'local_ik_or_path_rejected':'局部 IK / 碰撞路径检查拒绝',
        'pipeline_timeout':'900 秒总预算超时','lift_or_hold_failed':'实际抬升或保持未达标',
        'policy_reopened_after_closure':'闭合后策略重新张开',
        'execution_convergence_rejected':'实际执行未通过原收敛标准',
        'other_pipeline_failure':'其他链路失败','success':'成功'}
    lines.extend(f'| {labels.get(key,key)} | {value} |' for key,value in sorted(counts.items()))
    lines+=['', '显存故障不能当作模型能力结论。发现后将 ZeroGrasp 从物理 A100 GPU0 转到 GPU1，保持同一算法、权重、输入和评测；此前失败计入本轮且没有重跑。该轮含共享资源干扰，不能作为纯算法泛化估计。详细事件见 `resource_continuation.json`。','',
        '## 方法与执行证据','',
        'ZeroGrasp 使用每次重置后新采集的头部 RGB-D 和原模拟器辅助目标隔离，cuRobo 执行接近。LiLo-VLA 每步使用新右腕 RGB、8 维状态和统一指令 `pick up the object`，预测动作片段后执行第一步；每个动作通过原 ARX IK、限位和稠密碰撞检查，再由 Isaac Sim 实际执行。8 维状态为末端相对物体的位置差、相对轴角及双指位置，其中物体位姿取自模拟器；输入沿用瓶子训练时的非目标物体投影遮罩，保存原图和遮罩框。这是带仿真状态辅助的接入。',
        '冻结的是三条 BBQ 酱料瓶教师轨迹得到的动作头适配器（300 步）；骨干没有进行完整 LoRA 训练。没有用这 49 个物体训练或按物体调参，不能据此推断充分训练后的 LiLo-VLA 上限。公开模型预训练与 EGAD 是否重叠未核验。',
        f"实际进入机械臂运动 {summary['actual_motion_attempts']}/49；保存 {summary['model_calls']} 次局部策略预测。规划证书与实际执行、接触和高度单独保存；只有原物理执行判据通过才计成功。",'',
        '成功需要原阶段覆盖、原收敛判据、闭合期间双指接触、无禁止接触和越限，以及最后连续 3 秒所有记录采样抬升均超过 6 cm。原记录覆盖标准不变；7 cm 只是控制目标。没有为失败案例放宽标准。','',
        '## 初始条件与完整性','',
        '49 份重置配置证据全部通过：历史机器人、相机标定、23 关节初始状态（升降约 0.22 m）、物体初始摆放、物理配置和 360 步 / 3 秒静置。49 个 USD 的几何和物理资产字节匹配原目录；不使用重新转换的网格。',
        '原 RaytracedLighting 不变，仅将容器 GPU 序号映射为唯一可见的物理 4090，并激活已有标定的右腕渲染传感器。执行期间不做逐动作全配置扫描；每例初始原生指纹与登记源文件哈希均检查。全部原生执行日志独立重新评分，与保存结果一致。',
        '缺失的历史运行时属性只能报告未核验：本机没有原 storage_policy.json，保留未核验说明并执行用户指定的 1 GiB 空闲空间下限。历史原生指纹未包含的 getter 不推定相同。',
        '首次录像缺少视角引用文件时，只补原视角文件并从相同原始帧编码；未重复执行。原登记源码副本和哈希、基础设施事件、全部失败记录保留。','',
        '## 逐物体结果','', '| ID | 本轮 | 分类 | 原始历史 | H022 历史 |','|---|---|---|---|---|']
    mark=lambda value:'成功' if value else '失败'
    for row in rows:
        lines.append(f"| {row['egad_id']} | {mark(row['success'])} | {labels.get(row['failure_category'],row['failure_category'])} | {mark(row['historical_baseline_success'])} | {mark(row['historical_h022_success'])} |")
    lines+=['','## 配对比较','']
    for label,comparison in comparisons.items():
        lines.append(f"相对历史 {label}：保留成功 {len(comparison['preserved_successes'])}，改善 {len(comparison['gains'])}，退化 {len(comparison['losses'])}，共同失败 {len(comparison['both_failed'])}。")
        lines.append('改善 ID：'+(', '.join(comparison['gains']) or '无')+'。退化 ID：'+(', '.join(comparison['losses']) or '无')+'。')
    if ordered_times:
        lines+=['','## 运行诊断','',
            f"已保存策略调用的模型前向 P50 为 {summary['model_forward_p50_s']:.3f} 秒，P95 为 {summary['model_forward_p95_s']:.3f} 秒；这不含观察、IK、碰撞检查、规划、仿真执行、进程启动及编码。",
            f"归一化后位姿状态分量有 {summary['normalized_pose_components_at_clip_boundary_fraction']:.1%} 位于裁剪边界（±1）。这说明输入经常触及单瓶适配统计范围的边界；当前记录不能单独证明失败因果。该动作头尚未得到跨形状训练，本轮不能提升为默认抓取方法。"]
    lines+=['','## 文件','',
        '- `final_results.csv`：逐物体结果、失败原因、真实视频和试验目录。',
        '- `final_results.json`：完整汇总、配对比较、日志重评分与哈希验证结果。',
        '- `registration.json` 与 `registered_source_snapshot/`：运行前冻结的设置、源代码及权重哈希。',
        '- `trials/*/`：初始指纹、输入观测、模型预测、规划证书、实际命令/执行/接触/高度与原始录像帧。',
        '- 视频沿用旧编码器的 `bottle_grasp_no_obstacle_*` 文件名，内容是对应 EGAD 物体，不能据文件名误认成瓶子测试。','']
    (campaign/'final_report.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--historical',type=Path,default=ROOT/'runs/egad49_arx_20261020')
    args=parser.parse_args()
    result=summarize(args.campaign.resolve(),args.historical.resolve())
    print(json.dumps({key:value for key,value in result.items() if key!='rows'},indent=2))
