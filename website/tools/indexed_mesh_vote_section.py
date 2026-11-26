"""Publish IK diagnosis and exact mesh-query comparison as standalone copies."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, sha256, write_json


def raw(path): return json.loads(Path(path).read_text())


def section(website, site, active):
    relative = active.get('mesh_vote_run')
    if not relative: return None
    run = (ROOT/relative).resolve()
    if not run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
        raise ValueError('Unexpected mesh comparison publication source')
    reg = raw(run/'registration.json');p = raw(run/'progress.json')
    accepted = raw(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    reviewed = raw(run/'independent_review.json') if (run/'independent_review.json').exists() else None
    geometry = raw(run/'resident/mesh_vote_geometry_preflight/acceptance.json') \
        if (run/'resident/mesh_vote_geometry_preflight/acceptance.json').exists() else None
    source = ROOT/reg['diagnostic_source'];proof = raw(source/'diagnosis_summary.json')
    status = f"网格查询优化：{p['completed_requests']}/{p['total_requests']}次"
    if (run/'STOP').exists(): status += '，已停止'
    elif reviewed:
        status += '，独立核验完成'
        if reg['phase'] == 'pilot' and not accepted['timing_gate_passed']:
            status += '，未扩展69场景'
    elif accepted: status += '，独立核验中'
    elif geometry and not p['completed_requests']: status += '，几何核验通过，预热中'

    def attach(path, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Mesh-query artifacts must be independent files')
        shutil.copy2(path, target)
        if sha256(path) != sha256(target): raise ValueError('Standalone mesh-query copy differs')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    diagnosis_url = attach(source/'diagnosis_summary.json', 'sam3-v8-ik-hotspot.json')
    document = ROOT/'modules/pipeline/docs/mainline_versions/V8_批量IK定位与射线投票优化结果.md'
    if not document.exists(): document = ROOT/'modules/pipeline/docs/mainline_versions/V8_批量IK耗时定位与优化.md'
    document_url = attach(document, 'sam3-v8-mesh-vote.md')
    analysis = ROOT/'modules/pipeline/docs/mainline_versions/V8_批量IK优化先导分析.md'
    analysis_url = attach(analysis, 'sam3-v8-mesh-vote-analysis.md') if analysis.exists() else None
    name = 'sam3-v8-mesh-vote.json';target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Mesh-query publication snapshot must be independent')
    snapshot = {'run': relative, 'phase': 'stopped' if (run/'STOP').exists() else p['phase'],
        'completed_requests': p['completed_requests'], 'total_requests': p['total_requests'], 'gpu': reg['gpu'],
        'acceptance': {k: v for k, v in accepted.items() if k != 'cases'} if accepted else None,
        'independent_review': {k: v for k, v in reviewed.items() if k != 'endpoints'} if reviewed else None,
        'geometry_preflight': geometry, 'diagnostic_source': reg['diagnostic_source']}
    write_json(target, snapshot);site['localAssets'][name] = '../../resources/media/'+name
    data_url = '/media/'+name
    lines = ['## V8当前实验：网格查询优化', '', status+'。', '',
        f"原五场景10次定位全部通过，30个终点核验一致。五批采样中，碰撞查询占GPU实际内核时间{proof['collision_fraction']:.1%}，"
        f"CUDA Graph重新捕获{proof['CUDA_graph_captures']}次。", '',
        '原实现沿三个方向投票，判断点在网格内部还是外部。新实现当前两票一致时立即返回，意见不一致时继续第三票。'
        '最近点、射线方向、几何和碰撞标准保持。两组分别保留CUDA Graph，共同使用NMS15＋全部原候选后备、B16及C版YAML。', '']
    if geometry:
        lines += [f"{geometry['checked_sdf_points']}个实际网格查询点的距离和梯度逐位一致；完整场景碰撞代价与梯度复核通过。", '']
    if accepted:
        lines += ['| 方法 | 规划通过 | 请求中位耗时 | P90 | 累计耗时 |', '| --- | ---: | ---: | ---: | ---: |']
        for mode, label in [('control', '原三方向完整投票'), ('early_vote', '一致两票提前返回')]:
            a = accepted['summaries'][mode]
            lines.append(f"| {label} | {a['successes']}/{a['requests']} | {a['median_seconds']:.3f}秒 | "
                         f"{a['p90_seconds']:.3f}秒 | {a['total_seconds']:.3f}秒 |")
        lines += ['', f"完整请求累计省时{accepted['planning_total_time_reduction']:.2%}。", '']
    if reviewed:
        lines += [f"{reviewed['checked_endpoints']}个终点独立核验通过，原抓点、选臂、关节解和候选判断保持。", '']
        if reg['phase'] == 'pilot' and not accepted['timing_gate_passed']:
            lines += ['本轮累计省时和中位耗时未同时达标，先导在20次结束，保留原组合。', '']
    lines += ['先导五场景按ABBA／BAAB运行20次，关闭性能分析器。累计至少省5%、中位数下降、P90增幅不超过5%，'
        '并通过独立核验后，才开展69场景完整对照。默认链路保持，机器人动作0次。', '',
        f'[定位与优化方法、结果]({document_url}) · [实际计时和进度]({data_url}) · [GPU内核定位数据]({diagnosis_url})', '']
    if analysis_url: lines += [f'[先导的逐场景效果与下一步分析]({analysis_url})', '']
    batches = [{'id': source.name, 'title': '批量IK的CPU与CUDA内核定位', 'date': '2026-11-26',
        'status': '10/10完成，30终点独立核验通过', 'summary': '碰撞查询占采样内核97.75%，Graph重捕获0次。',
        'outputs': [reg['diagnostic_source']], 'artifacts': [{'label': '内核定位数据', 'url': diagnosis_url}]}]
    if reg.get('pilot_source'):
        pilot = ROOT/reg['pilot_source'];a = raw(pilot/'acceptance.json')
        batches.append({'id': pilot.name, 'title': '网格查询优化五场景先导', 'date': '2026-11-26',
            'status': '20/20完成，独立核验通过', 'summary': f"完整请求累计省时{a['planning_total_time_reduction']:.2%}。",
            'outputs': [reg['pilot_source']], 'artifacts': [{'label': '方法与结果', 'url': document_url}]})
    batches.append({'id': run.name, 'title': '网格查询优化'+('完整69场景对照' if reg['phase'] == 'full' else '五场景先导'),
        'date': '2026-11-26', 'status': status, 'summary': status, 'outputs': [relative],
        'artifacts': [{'label': '方法与结果', 'url': document_url}, {'label': '实际计时和进度', 'url': data_url}]})
    return {'body': '\n'.join(lines), 'status': status, 'summary': status, 'batches': batches}
