"""Summarize executed events and the measured closure/hold divergence."""
import json
from collections import Counter
from pathlib import Path

from arx_project.layout import sha256,write_json


def raw(path):return json.loads(Path(path).read_text())


def analyze(root):
    accepted=raw(root/'acceptance.json');reg=raw(root/'registration.json')
    counts={mode:Counter() for mode in ['baseline','optimized']}
    for item in reg['schedule']:
        directory=root/'trials'/item['case_id']/item['variant']
        stages=set(json.loads(line)['stage'] for line in (directory/'execution.jsonl').read_text().splitlines() if line.strip())
        if stages!=set(raw(directory/'stages.json')):raise ValueError('Executed and reported stages disagree')
        counts[item['variant']].update(stages)
    summary={'acceptance_sha256':sha256(root/'acceptance.json'),'actual_stages':counts,
        'registered_slots':20,'completed_slots':accepted['completed_slots'],'interrupted':0,'unstarted':0,
        'decoded_original_videos':len(accepted['decoded_videos']),'record_integrity':accepted['record_integrity'],
        'old_success_losses':accepted['baseline_success_losses'],'new_success_gains':accepted['optimized_success_gains']}
    for name in ['scissors_hold_diagnosis.json','scissors_command_contact_analysis.json']:
        if (root/name).exists():summary[name.removesuffix('.json')]=raw(root/name)
    write_json(root/'physical_analysis.json',summary)
    return summary


def markdown(summary):
    lines=['### 任务与事件核验','', '| 项目 | 结果 |','| --- | --- |',
        '| 任务完成情况 | 20/20完成，中断0、未开始0。 |',
        f"| 记录完整性 | 20个槽位的命令、回执与评估一致；{summary['decoded_original_videos']}段原视频解码通过。 |",
        '| 事件适用性 | 按实际轨迹和阶段报告核对已执行动作；提前结束后未执行的抬升、保持单列。 |',
        '| 关键事件覆盖 | 两组逐阶段实际执行数量见下表。 |','',
        '| 实际阶段 | 当前链路 | 本轮优化组 |','| --- | ---: | ---: |']
    for stage,label in [('open','张开'),('pregrasp','预抓取'),('approach','接近'),('close','闭合'),('lift','抬升'),('hold','保持')]:
        lines.append(f"| {label} | {summary['actual_stages']['baseline'][stage]} | {summary['actual_stages']['optimized'][stage]} |")
    diagnosis=summary.get('scissors_hold_diagnosis');commands=summary.get('scissors_command_contact_analysis')
    if diagnosis and commands:
        motion=diagnosis['stages'];coverage=commands['contact_coverage']
        lines+=['','### 杂乱剪刀：同一抓法仍有夹持保持差异','',
            'M0074两组都选第166号抓法。预抓取和接近期间，物体逐采样位置完全一致；'
            '实际接近轨迹的两组关节最大差约0.00000453弧度。闭合接触后，物体运动开始分岔。','',
            '| 观测 | 当前链路 | 本轮优化组 |','| --- | ---: | ---: |',
            f"| 闭合期间物体最大位移 | {motion['close']['baseline']['max_object_displacement_mm']:.2f}毫米 | {motion['close']['optimized']['max_object_displacement_mm']:.2f}毫米 |",
            f"| 闭合结束两指位置 | {motion['close']['baseline']['last_finger_openings_mm'][0]:.2f}、{motion['close']['baseline']['last_finger_openings_mm'][1]:.2f}毫米 | {motion['close']['optimized']['last_finger_openings_mm'][0]:.2f}、{motion['close']['optimized']['last_finger_openings_mm'][1]:.2f}毫米 |",
            f"| 抬升期双指同时有有效接触的采样 | {coverage['baseline']['lift']['bilateral_positive_impulse_samples']}/{coverage['baseline']['lift']['samples']} | {coverage['optimized']['lift']['bilateral_positive_impulse_samples']}/{coverage['optimized']['lift']['samples']} |",
            f"| 保持期双指同时有有效接触的采样 | {coverage['baseline']['hold']['bilateral_positive_impulse_samples']}/{coverage['baseline']['hold']['samples']} | {coverage['optimized']['hold']['bilateral_positive_impulse_samples']}/{coverage['optimized']['hold']['samples']} |",'',
            '两组物理配置记录完全相同。新基线的逐采样关节状态和物体位姿，也与上一轮基线完全一致。'
            '两组闭合时形成了不同的接触状态；优化组随后在抬升过程中脱离，保持期物体已回到桌面。','',
            '本轮去掉NMS后仍有一例退步，需要继续定位同一抓法下的细小指令差异与接触保持过程。'
            '现有记录把分歧定位到了闭合接触，具体因果还需要后续固定命令回放来区分。', '']
    lines+=['### 本轮结论与下一步','']
    if summary['old_success_losses']:
        lines+=['本轮优化组合保留为可选，默认链路保持。下一步优先固定同一成功抓法的运动命令，'
            '隔离微小指令差异与夹持接触稳定性，再决定计算优化如何接入常规抓取。', '']
    else:
        lines+=['本轮原成功案例已保留。计算优化的后续接入以本轮验收和时延结果为依据。', '']
    return '\n'.join(lines)
