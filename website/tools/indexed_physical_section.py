"""Copy physical V8 evidence into the independent website resources."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, sha256, write_json


def raw(path): return json.loads(Path(path).read_text())


def section(website, site, active):
    if not active.get('physical_run'): return None
    run = (ROOT/active['physical_run']).resolve()
    if not run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
        raise ValueError('Wrong physical publication root')
    reg = raw(run/'registration.json');p = raw(run/'progress.json')
    accepted = raw(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    stopped = (run/'STOP').exists()
    status = f"实际抓取对照：{p['completed_requests']}/20次"
    status += '，已停止' if stopped else '，独立核验完成' if accepted else '，执行中' if p['completed_requests'] else '，预检与预热中'
    if accepted and accepted['baseline_success_losses']:
        status += '，优化组合未保留全部原成功'

    def attach(path, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Physical evidence must be independent website files')
        if not target.exists() or sha256(target) != sha256(path): shutil.copy2(path, target)
        if target.stat().st_nlink != 1 or sha256(target) != sha256(path):
            raise ValueError('Physical publication copy changed')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    document = ROOT/'modules/pipeline/docs/mainline_versions/V8_优化组合实际抓取对照.md'
    analysis = None
    if accepted:
        from website.tools.indexed_physical_analysis import update_document
        analysis = update_document(run, document)
    report_url = attach(document, 'sam3-v8-physical-pilot.md') if document.exists() else None
    target = website/'resources/media/sam3-v8-physical-pilot.json'
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Physical summary must be an independent file')
    write_json(target, {'run': active['physical_run'], 'progress': p, 'acceptance': accepted, 'analysis': analysis})
    site['localAssets'][target.name] = '../../resources/media/'+target.name
    lines = ['## V8当前实验：优化组合实际抓取', '', status+'。', '',
        '选取10个场景，两组各执行一次。其中包括全部6个此前NMS改变抓法的场景，并覆盖五种布局。'
        '两组恢复相同的最终站位、升降和物体状态，使用相同的冻结感知与原始候选，再各自规划、实际抓取。', '',
        '| 项目 | 当前链路 | 优化组合 |', '| --- | --- | --- |',
        '| 双臂搜索 | 首轮16、后续64交替 | 相同 |',
        '| 抓取IK | 单候选 | 每批16个候选 |',
        '| 候选次序 | 原评分次序 | NMS15代表优先，原候选全部保留 |',
        '| 配置写出 | 原方式 | C版YAML |',
        '| 网格查询 | 原线程排列 | 同一网格连续查询 |', '']
    if accepted:
        old = accepted['groups']['baseline'];new = accepted['groups']['optimized']
        lines += [f"本轮现用链路实际成功{old['successes']}/10，优化组合{new['successes']}/10。"
            +('优化组合更快，但没有保留全部原成功案例，默认链路保持。' if accepted['baseline_success_losses'] else '逐场景结果与时延如下。'), '']
        lines += ['| 方法 | 规划通过 | 实际抓取成功 | 初始规划中位数 | 初始规划P90 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for mode, label in [('baseline', '当前链路'), ('optimized', '优化组合')]:
            g = accepted['groups'][mode]
            lines.append(f"| {label} | {g['planning_passes']}/10（{g['planning_passes']/10:.0%}） | {g['successes']}/10（{g['successes']/10:.0%}） | "
                f"{g['planning_median_seconds']:.3f}秒 | {g['planning_p90_seconds']:.3f}秒 |")
        lines += ['', f"包含NMS及共享内存交接的初始规划，累计省时{accepted['planning_total_time_reduction']:.2%}。"
            f"原成功变失败{len(accepted['baseline_success_losses'])}例，原失败变成功{len(accepted['optimized_success_gains'])}例。", '']
    reasons = {'success': '成功', 'no_bilateral_target_contact': '闭合后未形成双指夹持',
        'no_arm_has_executable_candidate': '规划不出来', 'insufficient_hold_height': '抬升或保持不足',
        'forbidden_robot_contact': '发生禁止接触', 'measured_closure_lift_unreachable': '闭合后抬升规划不出来',
        'measured_pregrasp_approach_unreachable': '实测预抓取状态下接近规划不出来',
        'pregrasp_not_converged': '到预抓取位置未到位', 'approach_not_converged': '接近动作未到位',
        'close_not_converged': '闭合时关节未到位', 'lift_not_converged': '抬升动作未到位',
        'hold_not_converged': '保持时关节未到位', 'open_not_converged': '夹爪张开未到位',
        'gripper_open_not_converged': '夹爪张开未到位'}
    labels = {'donut': '甜甜圈', 'scissors': '剪刀', 'banana': '香蕉', 'bar_soap': '香皂',
              'cucumber': '黄瓜', 'apple': '苹果', 'cell_phone': '手机', 'mango': '芒果'}
    if accepted:
        failures = sorted(set(accepted['groups']['baseline']['failure_reasons']) |
                          set(accepted['groups']['optimized']['failure_reasons']))
        lines += ['### 失败分布', '', '| 原因 | 当前链路 | 优化组合 |', '| --- | ---: | ---: |']
        for reason in failures:
            counts = [accepted['groups'][mode]['failure_reasons'].get(reason, 0) for mode in ['baseline', 'optimized']]
            lines.append(f"| {reasons.get(reason, reason)} | {counts[0]} | {counts[1]} |")
        lines += ['']
    rows = [];media_count = 0
    for case in reg['cases']:
        cells = []
        for mode in ['baseline', 'optimized']:
            out = run/'trials'/case['id']/mode
            claim = run/'claims'/(case['id']+'_'+mode+'.json')
            if not claim.exists() or raw(claim)['state'] != 'complete':
                cells.append('未完成' if out.exists() else '待执行');continue
            result = raw(out/'result.json');links = []
            prefix = 'sam3-v8-physical-'+case['id'].split('_')[0].lower()+'-'+mode
            for filename, name, label in [('execution.mp4', 'overview', '第三人称'), ('head.mp4', 'head', '头部')]:
                if (out/filename).exists():
                    url = attach(out/filename, prefix+'-'+name+'.mp4');links.append(f'[{label}]({url})');media_count += 1
            if (out/'paired_actual_head.png').exists():
                url = attach(out/'paired_actual_head.png', prefix+'-initial.png');links.append(f'[初始图]({url})')
            text = reasons.get(result['reason'], result['reason'].replace('_not_converged', '动作未到位'))
            if (out/'latency.json').exists():
                seconds = raw(out/'latency.json')['timings']['initial_planning_with_postprocessing_seconds']
                text += f'，初始规划{seconds:.2f}秒'
            cells.append(text+(' · '+' · '.join(links) if links else ''))
        target_label = labels.get(case['entry']['family'], case['entry']['family'])
        rows.append(f"| {case['id'].split('_')[0]} · {target_label} | {cells[0]} | {cells[1]} |")
    lines += ['### 逐场景结果与视频', '', '| 场景 | 当前链路 | 优化组合 |', '| --- | --- | --- |', *rows, '']
    if accepted and accepted['baseline_success_losses']:
        from website.tools.indexed_physical_videos import comparisons
        clips = comparisons(website, run, accepted)
        lines += ['### 原成功、新失败的同步对比视频', '',
            '左侧为现用链路，右侧为优化组合。视频标有实际结果和初始规划耗时，较短的一侧结束后停留在最后一帧。', '']
        for case, name in clips:
            site['localAssets'][name] = '../../resources/media/'+name
            family = next(c['entry']['family'] for c in reg['cases'] if c['id'] == case)
            lines += [f'[{labels.get(family, family)}（{case.split("_")[0]}）· 同步对比](/media/{name})', '']
    if analysis:
        from website.tools.indexed_physical_analysis import markdown
        lines += [markdown(analysis)]
    if report_url: lines += [f'[实施设置与结果]({report_url}) · [统计与核验数据](/media/sam3-v8-physical-pilot.json)', '']
    batch = {'id': run.name, 'title': '优化组合20次实际抓取对照', 'date': reg['at'].split('T')[0],
        'status': status, 'summary': status, 'outputs': [active['physical_run']],
        'artifacts': [{'label': '统计与核验', 'url': '/media/sam3-v8-physical-pilot.json'}]}
    return {'body': '\n'.join(lines), 'status': status, 'summary': status, 'batch': batch}
