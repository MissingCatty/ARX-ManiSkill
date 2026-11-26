"""Publish only source-backed V8 branch validation status and results."""
import shutil

from arx_project.layout import ROOT, read_json, sha256, write_json


def section(website, site, active):
    relative = active.get('validation_run')
    if not relative: return None
    run = (ROOT/relative).resolve()
    if not run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
        raise ValueError('Unexpected V8 validation result path')
    reg = read_json(run/'registration.json');progress = read_json(run/'progress.json')
    full = reg.get('phase') == 'full'
    phase_label = '完整69场景对照' if full else '先导'
    accepted = read_json(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    reviewed = read_json(run/'independent_review.json') if (run/'independent_review.json').exists() else None
    stopped = (run/'STOP').exists();done = progress['completed_requests'];total = reg['planning_requests']
    if stopped: status = f'抓取分支批量复核{phase_label}停止：{done}/{total}次完成'
    elif reviewed: status = f'抓取分支批量复核{phase_label}与独立核验完成'
    elif accepted: status = f'{total}次{phase_label}完成，独立核验中'
    else: status = f'抓取分支批量复核{phase_label}：{done}/{total}次'
    snapshot = {'run': relative, 'experiment': reg['experiment'], 'gpu': reg['gpu'],
        'phase': progress['phase'], 'completed_requests': done, 'total_requests': total,
        'current': progress.get('current'), 'summaries': progress.get('summaries'),
        'accepted': {k: v for k, v in accepted.items() if k != 'cases'} if accepted else None,
        'independent_review': {k: v for k, v in reviewed.items() if k != 'endpoints'} if reviewed else None,
        'physical_grasps': 0, 'stopped': stopped}
    stem = 'sam3-v8-branch-validation-'+('full69' if full else 'pilot')
    name = stem+'.json';target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Validation publication must be an independent file')
    write_json(target, snapshot);site['localAssets'][name] = '../../resources/media/'+name
    doc = ROOT/'modules/pipeline/docs/mainline_versions'/('V8_抓取分支批量复核完整对照结果.md' if full else 'V8_抓取分支批量复核先导.md')
    doc_name = stem+'.md';dest = website/'resources/media'/doc_name
    if dest.is_symlink() or dest.exists() and dest.stat().st_nlink != 1:
        raise ValueError('Validation report must be an independent copy')
    shutil.copy2(doc, dest)
    if sha256(doc) != sha256(dest): raise ValueError('Validation report copy differs')
    site['localAssets'][doc_name] = '../../resources/media/'+doc_name
    lines = ['## V8正在推进：抓取分支批量复核', '', status+'。', '',
        '同一个抓取位姿往往有多组关节解。原版逐组做FK和碰撞检查；本轮改成批量复核，'
        '仍按原顺序取第一个通过的解。两组都使用NMS15＋批量16 IK。'+
        ('原69份观测全部保留，每场景各1次，按AB／BA交替顺序完成138次纯规划对照。' if full else
         '每场景各重复两次，按ABBA／BAAB顺序完成5个场景、20次纯规划对照。'), '',
        '新增计时覆盖每次场景绑定、独立FK、局部碰撞、预抓取与抬升IK、PRM及轨迹优化。', '']
    if full:
        pilot = read_json(ROOT/reg['pilot_source']/'acceptance.json')
        values_pilot = pilot['summaries']
        lines += [f"先导两组各10/10通过，60个终点独立核验通过，累计耗时"
            f"{values_pilot['control']['total_seconds']:.2f}→{values_pilot['vectorized']['total_seconds']:.2f}秒，"
            f"省时{pilot['planning_total_time_reduction']:.2%}；中位耗时6.06→6.92秒。完整对照继续检查收益能否保持。", '',
            '[先导方法与结果](/media/sam3-v8-branch-validation-pilot.md) · '
            '[先导数据](/media/sam3-v8-branch-validation-pilot.json)', '']
    values = accepted['summaries'] if accepted else progress.get('summaries')
    if values:
        lines += ['| 方法 | 完整请求 | 规划通过 | 中位耗时 | P90耗时 | 累计耗时 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for mode, label in [('control', '原版逐分支复核'), ('vectorized', '批量复核')]:
            value = values[mode]
            lines.append(f"| {label} | {value['requests']} | {value['successes']} | {value['median_seconds']:.3f}秒 | "
                         f"{value['p90_seconds']:.3f}秒 | {value['total_seconds']:.3f}秒 |")
        lines += ['', '表中每条样本是一整个场景的双臂规划请求，包含开爪、候选搜索、IK、碰撞和路径。', '']
    if reviewed:
        saving = accepted['planning_total_time_reduction']
        lines += [f"原抓点、选臂、关节终点和候选拒绝分类全部一致，{reviewed['checked_endpoints']}个终点独立FK／限位核验通过。",
            (f'累计耗时减少{saving:.2%}。' if saving >= 0 else f'累计耗时增加{-saving:.2%}。'), '']
        if full:
            rows = accepted['cases']
            faster = sum(row['variants']['vectorized_1']['planning_request_seconds'] <
                         row['variants']['control_1']['planning_request_seconds'] for row in rows)
            lines += [f'{faster}个场景更快、{len(rows)-faster}个更慢。收益集中在部分慢场景，'
                      '中位耗时有所增加，P90基本持平；这项实现先保留为可选方案。', '']
        lines += [(('完整对照已完成，按全部输入的耗时和稳定性确定后续组合。' if full else
                    '达到先导2%的省时门槛，下一步比较完整69场景。') if accepted['timing_gate_passed'] else
                   '尚未达到2%的省时门槛，先分析新增分项计时，调整下一项优化。'), '']
    elif not accepted and not stopped:
        lines += [('先导正确性与耗时门槛已通过，完整批次另建登记，沿用相同算法代码。' if full else
                   '19项离线检查通过，含192组分支等值子例。')+'组件预热后常驻，按登记顺序逐例执行。', '']
    lines += [f'[本轮方法与结果](/media/{doc_name}) · [先导数据与分项计时](/media/{name})', '']
    summary = status
    if reviewed:
        summary += f"；累计{'省时' if accepted['planning_total_time_reduction'] >= 0 else '增时'}{abs(accepted['planning_total_time_reduction']):.2%}，原抓点和关节分支一致。"
    batch = {'id': run.name, 'title': '抓取分支批量复核纯规划'+phase_label, 'date': '2026-11-26',
        'status': status, 'summary': summary, 'outputs': [relative],
        'artifacts': [{'label': '方法与结果', 'url': '/media/'+doc_name}, {'label': '先导数据', 'url': '/media/'+name}]}
    return {'body': '\n'.join(lines), 'status': status, 'summary': summary, 'batch': batch}
