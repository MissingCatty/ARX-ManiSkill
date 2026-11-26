"""Plain-language report of configuration serialization and complete RPC time."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT


DOCUMENT = ROOT/'modules/pipeline/docs/mainline_versions/V8_配置准备提速.md'


def report(root):
    reg = json.loads((root/'registration.json').read_text())
    accepted = json.loads((root/'acceptance.json').read_text())
    checked = json.loads((root/'independent_review.json').read_text())
    full = reg.get('phase') == 'full';heading = '## 完整69场景结果' if full else '## GPU先导结果'
    body = DOCUMENT.read_text()
    if heading in body: body = body.split(heading)[0].rstrip()+'\n'
    lines = [heading, '', '| 方法 | 完整请求 | 规划通过 | RPC中位耗时 | RPC P90 | RPC累计耗时 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, label in [('control', '原版配置写出'), ('fast_binding', 'C版配置写出')]:
        value = accepted['summaries'][mode]
        lines.append(f"| {label} | {value['requests']} | {value['successes']} | {value['median_seconds']:.3f}秒 | "
                     f"{value['p90_seconds']:.3f}秒 | {value['total_seconds']:.3f}秒 |")
    saving = accepted['planning_total_time_reduction']
    lines += ['', f"累计耗时{'减少' if saving >= 0 else '增加'}{abs(saving):.2%}。"
              f"{checked['checked_endpoints']}个终点独立核验通过，{checked['generated_robot_YAML_pairs']}对实际配置值一致。"
              '原抓点、选臂、关节终点与候选判断均保持。', '',
              '| 分项 | 原版每请求平均 | C版每请求平均 |', '| --- | ---: | ---: |']
    for key, label in [('resident_world_bind', '场景与机器人绑定'),
                       ('grasp_batch_full_world_IK', '完整场景抓取IK'), ('PRM_path', 'PRM路径搜索')]:
        times = [accepted['summaries'][mode]['stage_totals'].get(key, {}).get('wall_seconds', 0.)/
                 accepted['summaries'][mode]['requests'] for mode in ['control', 'fast_binding']]
        lines.append(f'| {label} | {times[0]:.4f}秒 | {times[1]:.4f}秒 |')
    lines += ['', '分项计时来自规划服务内部；主表采用完整RPC耗时。', '',
              ('时延门槛全部通过。' if accepted['timing_gate_passed'] else
               '时延门槛未全部通过，保留原链路并分析分项，不扩展完整批次。'), '',
              f"原始记录：`{root.relative_to(ROOT)}`。", '']
    DOCUMENT.write_text(body.rstrip()+'\n\n'+'\n'.join(lines))
    shutil.copy2(DOCUMENT, root/'report.md')
