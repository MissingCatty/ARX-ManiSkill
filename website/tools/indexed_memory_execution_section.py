"""Publish full state-update grasp results and independently copied recordings."""
import json
import shutil
import statistics

from arx_project.layout import ROOT, read_json, sha256


def section(website, site):
    pointer = ROOT / 'tmp/state_updates/execution_active.json'
    if not pointer.exists() or not read_json(pointer).get('complete'): return None
    root = (ROOT / read_json(pointer)['run']).resolve()
    if not root.is_relative_to(ROOT / 'tmp/state_updates'): raise ValueError('Unexpected execution result root')
    data = read_json(root / 'publication_snapshot.json'); accepted = data['acceptance']
    if not accepted['task_complete'] or not accepted['record_integrity'] or not data['services_stopped']['all_owned_exited']:
        raise ValueError('Full execution review or shutdown incomplete')
    prefix = 'sam3-v8-' + root.name.replace('_', '-')
    def attach(source, suffix):
        name = prefix + '-' + suffix; target = website / 'resources/media' / name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Execution evidence must be independent files')
        if source.suffix == '.json':
            target.write_text(json.dumps(read_json(source), ensure_ascii=False, separators=(',', ':')) + '\n')
            if read_json(target) != read_json(source): raise ValueError('Copied result data changed')
        else:
            if not target.exists() or sha256(source) != sha256(target): shutil.copy2(source, target)
            if sha256(source) != sha256(target): raise ValueError('Copied recording/report changed')
        site['localAssets'][name] = '../../resources/media/' + name
        return '/media/' + name
    rows = data['progress']['results']
    excluded = set(accepted['interrupted_timing_cases_excluded'])
    pairs = {}
    for row in rows:
        if not row['common_precondition_unexecuted'] and row['id'] not in excluded:
            pairs.setdefault(row['id'], {})[row['variant']] = row
    successes = {mode: sum(p[mode]['success'] for p in pairs.values()) for mode in ['baseline','optimized']}
    a_time = accepted['matched_phase_timings']['baseline']['all_planning_per_scene']['mean_seconds']
    b_time = accepted['matched_phase_timings']['optimized']['all_planning_per_scene']['mean_seconds']
    savings = 1 - b_time / a_time
    a, b = accepted['groups']['baseline'], accepted['groups']['optimized']
    conclusion = (f"## 本轮结果\n\n{accepted['processed_slots']}个登记槽全部处理，"
        f"{accepted['completed_slots']}个完整结束、{accepted['interrupted_slots']}个程序中断、{accepted['not_started_slots']}个未开始。"
        f"排除共同站位前提未满足和中断场景后，有{len(pairs)}个完整配对场景，"
        f"双方分别成功{successes['baseline']}次和{successes['optimized']}次"
        f"（{successes['baseline']/len(pairs):.1%}／{successes['optimized']/len(pairs):.1%}）。"
        f"各阶段规划服务耗时合计平均{a_time:.2f}→{b_time:.2f}秒，减少{savings:.1%}。\n\n"
        + (f"全登记表里原方法{a['successes']}次成功、内存更新{b['successes']}次，差的一次来自香蕉新方法的程序中断；"
           "该条保留且未重抓。\n\n" if accepted['interrupted_slots'] == 1 else ''))
    rendered = (root / 'report.md').read_text().replace('\n\n', '\n\n' + conclusion, 1)
    rendered = rendered.replace('保留为未执行，不把它们写成实际空抓或规划求解失败。', '单列未执行。')
    rendered = rendered.replace('## 各段完整规划耗时', '## 各段规划计时')
    rendered = rendered.replace('| 初始完整规划 |', '| 初始规划服务内部 |')
    rendered = rendered.replace('| 每场景全部规划合计 |', '| 每场景全部规划服务合计 |')
    rpc_values = {mode: [p[mode]['timings']['initial_planning_with_postprocessing_seconds']
        for p in pairs.values()] for mode in ['baseline','optimized']}
    def rpc_cell(values):
        values = sorted(values); point = .9 * (len(values)-1); lo = int(point)
        p90 = values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(point-lo)
        return f'{statistics.mean(values):.3f}／{statistics.median(values):.3f}／{p90:.3f}秒（{len(values)}次）'
    first = '| 初始规划服务内部 |'
    rendered = rendered.replace(first, f"| 初始规划完整请求（含交接与返回） | {rpc_cell(rpc_values['baseline'])} | {rpc_cell(rpc_values['optimized'])} |\n" + first, 1)
    coverage = accepted['key_event_coverage']; counts = coverage['requested_stages']
    rendered = rendered.replace(str(coverage), f"完整项核查{counts['initial']}次初始规划、{counts['approach']}次接近重规划、"
        f"{counts['lift']}次抬升重规划；独立验算{coverage['independent_FK_endpoints']}个末端点和"
        f"{coverage['checked_command_paths']}条命令路径，解码{coverage['decoded_videos']}段录像")
    rendered = rendered.replace('原成功但新失败：[]。', '完整配对范围内，原方法成功而内存更新失败的场景为0。')
    rendered = rendered.replace('新成功但原失败：[]。', '完整配对范围内，内存更新成功而原方法失败的场景为0。')
    rendered = rendered.replace('| True |', '| 是 |').replace('| False |', '| 否 |')
    document = ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存状态更新完整抓取测试.md'
    document.write_text(rendered)
    report = attach(document, 'report.md')
    results = attach(root / 'publication_snapshot.json', 'results.json')
    reg = read_json(root / 'registration.json'); links = {}
    for video in accepted['decoded_videos']:
        source = ROOT / video; case, mode = source.parent.parent.name, source.parent.name
        view = 'head' if source.name == 'head.mp4' else 'overview'
        url = attach(source, case.split('_')[0].lower() + '-' + mode + '-' + view + '.mp4')
        links.setdefault((case, mode), []).append(f'[{"头部" if view == "head" else "第三人称"}]({url})')
    reasons = {'unable_to_reach_high_workspace': '站位前提未满足，未执行',
        'no_arm_has_executable_candidate': '规划不出来',
        'no_bilateral_target_contact': '闭合后未形成双指夹持',
        'insufficient_hold_height': '物体抬升或保持不足',
        'forbidden_robot_contact': '发生禁止接触',
        'measured_closure_lift_unreachable': '实测闭合后抬升规划不出来',
        'measured_pregrasp_approach_unreachable': '实测预抓取状态下接近规划不出来',
        'measured_start_state_in_collision': '实测起始状态碰撞，规划不出来',
        'gripper_open_not_converged': '夹爪张开未到位', 'open_not_converged': '夹爪张开未到位',
        **{s + '_not_converged': label + '未到位' for s, label in
           [('pregrasp', '预抓取'), ('approach', '接近'), ('close', '闭合'), ('lift', '抬升'), ('hold', '保持')]}}
    for code, label in reasons.items(): rendered = rendered.replace(code, label)
    document.write_text(rendered)
    report = attach(document, 'report.md')
    body = rendered.replace('# V8 内存状态更新完整抓取测试',
        '## 内存状态更新：完整抓取测试', 1).replace('\n## ', '\n### ')
    for code, label in reasons.items(): body = body.replace(code, label)
    for case in reg['cases']:
        prefix_line = '| ' + case['id'] + ' |'
        for line in body.splitlines():
            if line.startswith(prefix_line):
                cells = line.split('|')
                for index, mode in [(2, 'baseline'), (3, 'optimized')]:
                    if (case['id'], mode) in links:
                        cells[index] = cells[index].rstrip() + ' · ' + ' · '.join(links[(case['id'], mode)]) + ' '
                body = body.replace(line, '|'.join(cells), 1)
                break
    body += f'\n[完整测试报告]({report}) · [逐场景统计与核验]({results})\n'
    status = '74场景双组抓取测试结束'
    pair_success = (f"双方均成功{successes['baseline']}次" if successes['baseline'] == successes['optimized'] else
        f"分别成功{successes['baseline']}、{successes['optimized']}次")
    summary = (f"{len(pairs)}个完整配对场景{pair_success}；"
        f"规划各段合计平均{a_time:.2f}→{b_time:.2f}秒，减少{savings:.1%}。"
        f"全登记{accepted['completed_slots']}完整、{accepted['interrupted_slots']}中断。")
    return {'body': body, 'status': status, 'summary': summary,
        'batch': {'id': root.name, 'title': '内存状态更新74场景完整抓取对照',
            'date': reg['at'].split('T')[0], 'status': status, 'summary': summary,
            'outputs': [str(root.relative_to(ROOT))], 'artifacts': [
                {'label': '完整测试报告', 'url': report}, {'label': '统计与核验', 'url': results}]}}
