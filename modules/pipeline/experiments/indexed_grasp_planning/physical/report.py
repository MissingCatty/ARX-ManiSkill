"""Physical pilot report and independent website publication snapshot."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, write_json
from .registration import raw

DOCUMENT = ROOT/'modules/pipeline/docs/mainline_versions/V8_优化组合实际抓取对照.md'


def report(root):
    reg = raw(root/'registration.json');progress = raw(root/'progress.json')
    accepted = raw(root/'acceptance.json') if (root/'acceptance.json').exists() else None
    lines = ['# V8优化组合实际抓取对照', '',
        '比较当前双臂交替链路与经过验证的优化组合，每个场景每组一次，共10个场景、20个槽位。', '',
        '| 项目 | 当前链路 | 优化组合 |', '| --- | --- | --- |',
        '| 双臂搜索 | 首轮16、后续64的交替窗口 | 相同 |',
        '| GPU抓取IK | 原单候选求解 | 每批16个候选 |',
        '| 候选次序 | 原始评分次序 | NMS15代表优先，保留全部原候选后备 |',
        '| 配置写出 | 原YAML方式 | C版YAML |',
        '| 网格碰撞查询 | 原线程排列 | 同一网格连续查询 |', '',
        '两组使用同一份冻结的SAM3、RGB-D与原生GraspGenX候选，并各自重新规划、执行抓取。'
        '每次恢复该场景原有的最终底盘位置、升降状态、物体和关节状态，检查静止及误差后再开始。'
        '新拍摄的头部图像和第三人称画面用于记录本次状态。', '',
        '机器人、相机、物理、871球、TCP43和V2指尖规则保持。计划抬升3厘米，'
        '物体连续保持至少1厘米、3秒，双指接触和禁止接触检查沿用原评估器。', '',
        '主时延为包含NMS和共享内存交接的初始规划请求；恢复场景、实际运动与执行中再规划分别记录。', '',
        '## 固定场景', '', '| 场景 | 纳入原因 |', '| --- | --- |']
    for case in reg['cases']:
        reason = '此前NMS改变了抓法或选臂' if case['selection_changed_by_NMS'] else '补齐布局、短搜索或慢场景'
        lines.append(f"| {case['id']} | {reason} |")
    lines += ['', '## 当前结果', '', f"完成{progress['completed_requests']}/{progress['total_requests']}个槽位。", '']
    if accepted:
        lines += ['| 方法 | 完成 | 规划通过 | 实际成功 | 初始规划中位数 | 初始规划P90 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for mode, label in [('baseline', '当前链路'), ('optimized', '优化组合')]:
            g = accepted['groups'][mode]
            lines.append(f"| {label} | {g['completed']}/10 | {g['planning_passes']}/10 | {g['successes']}/10 | "
                         f"{g['planning_median_seconds']:.3f}秒 | {g['planning_p90_seconds']:.3f}秒 |")
        lines += ['', f"初始规划累计省时{accepted['planning_total_time_reduction']:.2%}。", '',
            f"原成功变失败：{len(accepted['baseline_success_losses'])}例；原失败变成功：{len(accepted['optimized_success_gains'])}例。", '',
            f"全部20个槽位记录已核验，{len(accepted['decoded_videos'])}段视频解码通过。", '',
            '## 失败分布', '', '| 方法 | 原因 | 数量 |', '| --- | --- | ---: |']
        for mode, label in [('baseline', '当前链路'), ('optimized', '优化组合')]:
            for reason, count in accepted['groups'][mode]['failure_reasons'].items():
                lines.append(f'| {label} | {reason} | {count} |')
        lines += ['', '默认链路保持，后续批量测试另行登记。', '']
    else: lines += ['服务预热和仿真预检通过后，按登记次序交替执行，不自动重试。', '']
    DOCUMENT.write_text('\n'.join(lines));shutil.copy2(DOCUMENT, root/'report.md')


def publish(root):
    from modules.pipeline.experiments.indexed_grasp_planning.validation_followup import publish as publish_v8
    publish_v8(root)
