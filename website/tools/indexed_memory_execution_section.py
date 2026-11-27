"""Publish full state-update grasp results and independently copied recordings."""
import json
import shutil

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
    report = attach(root / 'report.md', 'report.md')
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
    body = (root / 'report.md').read_text().replace('# V8 内存状态更新完整抓取测试',
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
    a, b = accepted['groups']['baseline'], accepted['groups']['optimized']
    status = '74场景双组完整抓取测试完成'
    summary = (f"两组各74槽完成，规划通过{a['planning_passes']}/74、{b['planning_passes']}/74；"
        f"实际抓取成功{a['successes']}/74、{b['successes']}/74，每组5个原站位前提未满足场景单列未执行。")
    return {'body': body, 'status': status, 'summary': summary,
        'batch': {'id': root.name, 'title': '内存状态更新74场景完整抓取对照',
            'date': reg['at'].split('T')[0], 'status': status, 'summary': summary,
            'outputs': [str(root.relative_to(ROOT))], 'artifacts': [
                {'label': '完整测试报告', 'url': report}, {'label': '统计与核验', 'url': results}]}}
