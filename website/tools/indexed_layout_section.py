"""Publish component diagnosis and collision-layout results as standalone files."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, sha256, write_json


def raw(path): return json.loads(Path(path).read_text())


def section(website, site, active):
    if not active.get('layout_run'): return None
    run = (ROOT/active['layout_run']).resolve()
    if not run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
        raise ValueError('Unexpected collision-layout publication source')
    reg = raw(run/'registration.json');progress = raw(run/'progress.json')
    source = ROOT/reg['diagnostic_source'];reviewed = raw(source/'independent_review.json')
    if not reviewed['passed'] or not reviewed['actual_workloads_verified']:
        raise ValueError('The actual query diagnosis is not accepted')
    accepted = raw(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    checked = raw(run/'independent_review.json') if (run/'independent_review.json').exists() else None
    geometry_path = run/'resident/layout_geometry_preflight/acceptance.json'
    geometry = raw(geometry_path) if geometry_path.exists() else None
    status = f"线程排列优化：{progress['completed_requests']}/{progress['total_requests']}次"
    if (run/'STOP').exists(): status += '，故障停止'
    elif checked:
        status += '，独立核验完成'
        if reg['phase'] == 'pilot' and not accepted['timing_gate_passed']: status += '，未扩展69场景'
    elif accepted: status += '，独立核验中'
    elif not progress['completed_requests']: status += '，预检与预热中' if (run/'started.json').exists() else '，等待可用显卡'

    def attach(path, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Layout publication must use independent files')
        shutil.copy2(path, target)
        if sha256(path) != sha256(target): raise ValueError('Layout publication copy changed')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    source_reg = raw(source/'registration.json')
    rows = [{'case': case['id'], **raw(source/'plans'/case['id']/'captured_1/query_components/components.json')}
            for case in source_reg['cases']]
    snapshot = {'run': active['layout_run'], 'completed_requests': progress['completed_requests'],
        'total_requests': progress['total_requests'], 'phase': progress['phase'], 'gpu': reg['gpu'],
        'diagnostic_source': reg['diagnostic_source'], 'component_samples': rows,
        'acceptance': {k: v for k, v in accepted.items() if k != 'cases'} if accepted else None,
        'independent_review': {k: v for k, v in checked.items() if k != 'endpoints'} if checked else None,
        'actual_collision_preflight': geometry}
    name = 'sam3-v8-kernel-layout.json';target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Layout snapshot must be independent')
    write_json(target, snapshot);site['localAssets'][name] = '../../resources/media/'+name
    plan = attach(ROOT/'modules/pipeline/docs/mainline_versions/V8_网格碰撞线程排列优化方案.md',
                  'sam3-v8-kernel-layout-plan.md')
    document = ROOT/'modules/pipeline/docs/mainline_versions/V8_网格查询分项与线程排列优化结果.md'
    result_url = attach(document, 'sam3-v8-kernel-layout.md') if document.exists() else None
    lines = ['## V8当前实验：网格查询定位与线程排列优化', '', status+'。', '',
        '原五个场景的10次定位已经完成，30个终点独立核验通过。'
        '用每例首批完整IK的实际球和网格，分开测量最近表面搜索和三方向内外判断。', '',
        '| 场景 | 最近表面搜索 | 三方向内外判断 | 合并查询 | 原完整碰撞派发 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        times = [row['timings'][key]['median_seconds']*1000 for key in
                 ['nearest_point', 'three_ray_votes', 'combined_compact', 'original_collision_dispatch']]
        lines.append('| '+row['case']+' | '+' | '.join(f'{v:.3f}毫秒' for v in times)+' |')
    lines += ['', '四个场景的最近表面搜索更耗时，苹果场景两项接近。'
        '本轮只调整GPU任务顺序，让相邻线程连续处理同一个障碍物上的球。'
        '所有球与障碍物组合、几何、距离公式、三方向射线及精度门槛保持。', '',
        '原排列与新排列共用NMS15＋全部原候选后备、B16及C版YAML。'
        '完整IK分别预热并保存CUDA Graph。上表为固定查询的GPU分项时间，完整规划请求另表比较。', '']
    if geometry:
        lines += [f"预检已核对{geometry['checked_spheres']:,}个实际IK碰撞球，完整碰撞代价与梯度一致。", '']
    if accepted:
        lines += ['| 方法 | 规划通过 | 完整请求中位耗时 | 完整请求P90 | 完整请求累计耗时 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for mode, label in [('control', '原线程排列'), ('obstacle_major', '同一障碍物连续排列')]:
            s = accepted['summaries'][mode]
            lines.append(f"| {label} | {s['successes']}/{s['requests']} | {s['median_seconds']:.3f}秒 | "
                         f"{s['p90_seconds']:.3f}秒 | {s['total_seconds']:.3f}秒 |")
        lines += ['', f"完整请求累计省时{accepted['planning_total_time_reduction']:.2%}。", '']
        lines += ['### 哪个环节变快了', '',
            '| 环节 | 原排列：每次请求平均 | 新排列：每次请求平均 |', '| --- | ---: | ---: |']
        stages = [('grasp_batch_full_world_IK', '成批抓取IK，含场景碰撞'),
                  ('grasp_batch_kinematic_IK', '纯运动学IK回退'),
                  ('resident_world_bind', '机器人与场景绑定'),
                  ('contact_world_bind', '局部接触世界绑定'),
                  ('pregrasp_scalar_IK', '预抓取IK'), ('lift_scalar_IK', '抬升IK'),
                  ('PRM_path', '到预抓取位置的路径搜索')]
        for key, label in stages:
            values = [accepted['summaries'][mode]['stage_totals'].get(key, {}).get('wall_seconds', 0.)
                      /accepted['summaries'][mode]['requests'] for mode in ['control', 'obstacle_major']]
            lines.append(f'| {label} | {values[0]:.4f}秒 | {values[1]:.4f}秒 |')
        lines += ['', '表中记录各环节实际调用时间，其中部分步骤互相包含。完整请求的总耗时以上一表为准。', '']
        lines += ['### 不同场景的耗时', '',
            '| 场景类型 | 配对场景数 | 原排列平均每次 | 新排列平均每次 | 累计耗时变化 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for code, label in [('S1', '单物体'), ('S2', '桌沿'), ('S3', '稀疏摆放'),
                            ('S4', '杂乱摆放'), ('S5', '遮挡')]:
            cases = [row for row in accepted['cases'] if row['id'].split('_')[1] == code]
            if not cases: continue
            values = [[v['planning_rpc_seconds'] for row in cases for variant, v in row['variants'].items()
                       if reg['variant_modes'][variant] == mode] for mode in ['control', 'obstacle_major']]
            saving = 1-sum(values[1])/sum(values[0])
            change = f'减少{saving:.1%}' if saving >= 0 else f'增加{-saving:.1%}'
            means = [sum(v)/len(v) for v in values]
            lines.append(f'| {label} | {len(cases)} | {means[0]:.3f}秒 | {means[1]:.3f}秒 | {change} |')
        lines += ['', '各组统计同一批场景的完整请求。每个场景的原抓点、选臂和关节终点另行逐项核验。', '']
    if checked:
        lines += [f"{checked['checked_endpoints']}个终点独立核验通过，原候选、抓法、选臂、关节解和候选判断保持。", '']
    if checked and reg['phase'] == 'pilot' and not accepted['timing_gate_passed']:
        lines += ['先导未通过全部计时门槛，在20次结束，默认链路保持。', '']
    else:
        lines += ['先导按ABBA／BAAB完成20次。累计至少省5%、中位耗时下降、P90增幅不超过5%，'
            '并通过独立核验后，才运行69场景的138次完整对照。', '']
    links = [f'[实施方案]({plan})', f'[计时与核验数据](/media/{name})']
    if result_url: links.insert(1, f'[方法与结果]({result_url})')
    lines += [' · '.join(links), '']
    batches = [{'id': source.name, 'title': '最近表面搜索与射线分项定位', 'date': source_reg['at'].split('T')[0],
        'status': '10/10完成，30终点独立核验通过', 'summary': '五批实际IK输入已拆分并核验，最近表面搜索在四例中占主要部分。',
        'outputs': [reg['diagnostic_source']], 'artifacts': [{'label': '分项计时数据', 'url': '/media/'+name}]}]
    if reg.get('pilot_source'):
        pilot = ROOT/reg['pilot_source'];a = raw(pilot/'acceptance.json')
        batches.append({'id': pilot.name, 'title': '线程排列优化20次先导', 'date': raw(pilot/'registration.json')['at'].split('T')[0],
            'status': '20/20完成，独立核验通过', 'summary': f"累计省时{a['planning_total_time_reduction']:.2%}。",
            'outputs': [reg['pilot_source']], 'artifacts': [{'label': '方法与结果', 'url': result_url or plan}]})
    batches.append({'id': run.name, 'title': '线程排列优化'+('69场景完整对照' if reg['phase'] == 'full' else '20次先导'),
        'date': reg['at'].split('T')[0], 'status': status, 'summary': status, 'outputs': [active['layout_run']],
        'artifacts': [{'label': '实施方案', 'url': plan}, {'label': '实际计时', 'url': '/media/'+name}]})
    return {'body': '\n'.join(lines), 'status': status, 'summary': status, 'batches': batches}
