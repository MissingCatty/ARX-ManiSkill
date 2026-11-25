"""Publish V4 progress/results and independent compressed website snapshots."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json

LAYOUTS = {'S1_open': '开阔', 'S2_edge': '桌边', 'S3_sparse': '稀疏干扰',
           'S4_clutter': '密集杂物', 'S5_occluded': '部分遮挡'}
REASONS = {'success': '成功', 'coarse_sam3_no_target_mask': '初次观测SAM3未分割出目标',
    'pregrasp_not_converged': '预抓取未到位', 'approach_not_converged': '最后接近未到位',
    'close_not_converged': '闭合未到位', 'lift_not_converged': '抬升未到位', 'hold_not_converged': '保持未到位',
    'sam3_no_target_mask': '到位后SAM3未分割出目标', 'coarse_invalid_depth': '初次目标深度不足',
    'coarse_rayst3r_empty_pointcloud': '初次观测未生成目标点云',
    'rayst3r_empty_pointcloud': '到位后未生成目标点云',
    'no_safe_reachable_station': '站位检查未通过，见逐例阶段记录',
    'no_arm_has_executable_candidate': '到位后左右臂均规划不出来',
    'no_bilateral_target_contact': '闭合后未形成双指夹持',
    'insufficient_hold_height': '物体抬升或保持不足', 'forbidden_robot_contact': '发生禁止接触',
    'measured_closure_lift_unreachable': '实测闭合后抬升规划不出来',
    'measured_pregrasp_approach_unreachable': '预抓取后接近规划不出来',
    'Target has fewer than 50 visible pixels': '目标在新观测中可见像素不足'}


def publish(root):
    website = ROOT/'website'
    progress = read_json(root/'progress.json')
    reg = read_json(root/'registration.json')
    dataset = Path(reg['dataset'])
    summary = read_json(root/'summary.json')
    adjudication = read_json(root/'acceptance.json') if (root/'acceptance.json').exists() else None
    if adjudication:
        complete = {row['id'] for row in adjudication['results'] if row['state'] == 'complete'}
        progress['completed'] = adjudication['completed']
        progress['not_completed'] = adjudication['interrupted']+adjudication['not_started']
        progress['failure_reasons'] = dict(Counter(row['reason'] for row in summary['results']
            if row['id'] in complete and not row['success']))
        for layout, data in progress['by_layout'].items():
            data['completed'] = sum(row['id'] in complete and row['layout'] == layout for row in summary['results'])
    site_path, version_path = website/'内容/网站.json', website/'内容/experiments.json'
    site, versions = read_json(site_path), read_json(version_path)
    copies = []
    media = website/'resources/media'

    def snapshot(source, name):
        destination = media/name
        if destination.is_symlink():
            raise RuntimeError('Website snapshot must be an independent real file')
        shutil.copy2(source, destination)
        if destination.stat().st_nlink != 1 or sha256(source) != sha256(destination):
            raise RuntimeError('Website copy failed')
        site.setdefault('localAssets', {})[name] = '../../resources/media/'+name
        copies.append({'source': str(source.relative_to(ROOT)), 'copy': str(destination.relative_to(website)),
                       'sha256': sha256(destination)})
        return '/media/'+name

    summary_source = root/'summary.json'
    if adjudication:
        exported = {**summary, 'completed': progress['completed'],
            'not_completed': progress['not_completed'], 'by_layout': progress['by_layout'],
            'interrupted': adjudication['interrupted'], 'not_started': adjudication['not_started'],
            'raw_result_record_count': len(summary['results']), 'failure_reasons': progress['failure_reasons'],
            'original_summary_sha256': sha256(summary_source),
            'count_source': 'independent_acceptance_and_claims'}
        states = {row['id']: row['state'] for row in adjudication['results']}
        exported['results'] = [{**row, 'record_state': states[row['id']]} for row in summary['results']]
        summary_source = root/'publication_summary.json'
        write_json(summary_source, exported)
    progress_url = snapshot(summary_source, 'sam3-v4-summary.json')
    snapshot(ROOT/'configs/experiments/active_experiment.json', 'active-configuration.json')
    snapshot(root/'registration.json', 'sam3-v4-registration.json')
    snapshot(dataset/'verification.json', 'sam3-v4-scene-verification.json')
    if (root/'acceptance.json').exists():
        snapshot(root/'acceptance.json', 'sam3-v4-acceptance.json')
    titles = {'registered': '100场景已登记，待组件预检', 'running': '100场景抓取测试进行中',
              'complete': '100场景抓取测试完成', 'stopped': '试验停止，保留现场'}
    status = titles.get(progress['phase'], progress['phase'])
    metric = (f"已完成{progress['completed']}/100例；站位到位{progress['station_ready']}例，"
              f"最终抓取规划通过{progress['planning_passes']}例，抓取成功{progress['successes']}例。")
    if adjudication and adjudication['interrupted']:
        metric += f" 另有{adjudication['interrupted']}例异常中断、{adjudication['not_started']}例未开始。"
    experiment = next(r for r in versions['experiments'] if r['id'] == 'sam3-graspgenx')
    experiment['current'] = 'V4'
    version = next((r for r in experiment['iterations'] if r['code'] == 'V4'), None)
    if version is None:
        version = {'code': 'V4', 'parent': 'V3', 'sourceId': 'sam3-graspgenx',
            'title': '底盘、升降与双臂自主站位抓取', 'date': '2026-11-25',
            'contentFile': 'sam3-graspgenx-v4.md', 'reportLayout': True,
            'authorization': {'kind': 'explicit-user-request', 'request': reg['user_authorization']}}
        experiment['iterations'].append(version)
    version.update(status=status, resultSummary=metric,
        change='由目标观测计算底盘位置、朝向和升降高度，停稳后重新感知，再选一臂抓取。',
        pipeline='初次头部RGB-D与SAM3 → 站位和整机移动检查 → 底盘与升降移动 → 停稳重新观测 → SAM3＋RaySt3R＋GraspGenX → 左右臂cuRobo → 抓取',
        scenes='新采样100例，五类布局各20例，使用共享物体资产。',
        settings=[['底盘', '限速SE2仿真位姿接口，完整机器人路径检查。'],
                  ['升降', '真实joint4原PD控制，0～46厘米。'],
                  ['末段', '重新观测与规划，每臂871球，TCP43和原成功标准。']],
        batches=[{'id': root.name, 'title': 'V4独立100场景', 'date': '2026-11-25',
                  'status': status, 'summary': metric, 'outputs': [str(root.relative_to(ROOT))],
                  'artifacts': [{'label': '逐例原始结果', 'url': progress_url}]}])
    home = next(r for r in site['versions'] if r['id'] == 'sam3-graspgenx')
    home.update(status='V4 '+status, metric=metric)
    text = ['## 本版流程', '',
        '先看目标，计算底盘和升降站位；移动到位并停稳后重新拍摄，再由左右臂规划器选择一条臂完成抓取。', '',
        '| 步骤 | 具体做法 |', '| --- | --- |',
        '| 1. 初次观测 | 头部RGB-D与SAM3获取可见目标的三维测量，生成原生抓姿。 |',
        '| 2. 选站位 | 反推底盘XY、朝向和升降高度，检查整机移动路径及左右臂完整抓取规划。 |',
        '| 3. 移动与停稳 | 底盘限速移动，joint4升降；保持双臂，确认到位和连续0.5秒静止。 |',
        '| 4. 重新观测 | 在新位置重新运行SAM3、RaySt3R与GraspGenX。 |',
        '| 5. 抓取 | 从左右臂选择可执行抓法，接近、闭合、抬升并保持3秒。 |', '',
        '停稳后的感知与抓取模块可点击查看：', '', '[[流程卡片]]', '',
        '底盘使用仿真的根位姿接口沿平滑路径移动；升降由原joint4驱动执行。原模型三只轮子的连接为固定关节。', '',
        '## 100场景测试进度', '', metric, '',
        '| 布局 | 已完成/登记 | 最终规划通过 | 抓取成功 |', '| --- | ---: | ---: | ---: |']
    for layout, label in LAYOUTS.items():
        data = progress['by_layout'][layout]
        text.append(f"| {label} | {data['completed']}/20 | {data['planning_passes']} | {data['successes']} |")
    remaining = (f"异常中断{adjudication['interrupted']}例，未开始{adjudication['not_started']}例。"
                 if adjudication else f"尚未完成{progress['not_completed']}例。")
    text += ['', remaining+f"实际执行抓取{progress['physical_attempts']}例。", '',
             f'[逐例结果]({progress_url}) · [试验登记](/media/sam3-v4-registration.json)', '',
             '## 场景示例', '', '目标放在桌面近侧，每类展示两例。点击图片查看原图。', '',
             '| 布局 | 场景一 | 场景二 |', '| --- | --- | --- |']
    for layout, label in LAYOUTS.items():
        entries = [r for r in reg['schedule'] if r['layout'] == layout]
        cells = []
        for entry in [entries[0], entries[10]]:
            name = 'sam3-v4-'+entry['id'].split('_')[0].lower()+'-initial.jpg'
            source = dataset/'cases'/entry['id']/'head.jpg'
            url = snapshot(source, name)
            cells.append(f"[![{entry['target_text']}]({url})]({url})")
        text.append(f"| {label} | {' | '.join(cells)} |")
    text += ['', '## 失败阶段', '', '| 原因 | 数量 |', '| --- | ---: |']
    for reason, count in progress['failure_reasons'].items():
        label = REASONS.get(reason, reason)
        if reason.endswith('_not_converged'):
            label = {'pregrasp': '预抓取', 'approach': '最后接近', 'close': '闭合', 'lift': '抬升', 'hold': '保持'}.get(reason.removesuffix('_not_converged'), '动作')+'未到位'
        text.append(f'| {label} | {count} |')
    videos = []
    for row in sorted(summary['results'], key=lambda r: (not r['success'], r['id'])):
        trial = root/'trials'/row['id']
        if not (trial/'execution.mp4').exists():
            continue
        prefix = 'sam3-v4-'+row['id'].split('_')[0].lower()
        overview = snapshot(trial/'execution.mp4', prefix+'-grasp.mp4')
        wrist = snapshot(trial/'head.mp4', prefix+'-head.mp4')
        movement = snapshot(trial/'stationing/motion.mp4', prefix+'-station.mp4')
        videos.append(f"| {row['target_text']} | {REASONS.get(row['reason'], row['reason'])} | [移动]({movement}) · [抓取]({overview}) · [头部]({wrist}) |")
        if len(videos) == 4:
            break
    if videos:
        text += ['', '## 实际视频', '', '| 目标 | 结果 | 视频 |', '| --- | --- | --- |', *videos]
    text += ['', '## 主要设置', '',
        '双臂各871碰撞球，TCP43；头部D405向下45°。升降和底盘到位后固定，再执行六轴手臂抓取。物体实际至少抬高1厘米并连续保持3秒，双指接触、到位和禁止接触检查通过。', '',
        '## 参考方法', '', '[SAM3](/papers/sam3/) · [RaySt3R](/papers/rayst3r/) · [GraspGenX](/papers/graspgenx/) · [cuRobo](/papers/curobo/)', '']
    acceleration = read_json(ROOT/'configs/experiments/active_experiment.json').get('acceleration_revision')
    validation = version.get('planningAcceleration', {})
    if acceleration or validation:
        report = ROOT/(acceleration or validation)['validation_report']
        if not report.is_file():
            report = media/'sam3-v4-planning-acceleration.json'
        data = read_json(report)
        url = '/media/sam3-v4-planning-acceleration.json'
        if report != media/'sam3-v4-planning-acceleration.json':
            url = snapshot(report, 'sam3-v4-planning-acceleration.json')
        section = ['## 已验证的规划加速', '',
            '以下是在同一显卡、相同缓存观测和候选上的完整选臂规划回放。'+
            ('新登记默认使用排名截断与批量开爪检查。' if acceleration else '加速实现已验证，供后续登记使用。'), '',
            '| 物体 | 原实现 | 加速实现 | 加速倍数 | 手臂与抓姿 |', '| --- | ---: | ---: | ---: | --- |']
        labels = {'apple': '苹果', 'peach': '桃子', 'spatula': '锅铲'}
        for row in data['paired_cases']:
            name = row['case'].split('_')[-1]
            section.append(f"| {labels.get(name, name)} | {row['original_seconds']:.1f}秒 | {row['accelerated_seconds']:.1f}秒 | {row['speedup']:.2f}倍 | {'一致' if row['passed'] else '有差异'} |")
        section += ['', '另一臂保留能胜出或同名次的候选；开爪保留原有全部宽度采样和871球，批量计算关节限位、自碰撞与场景碰撞。', '',
                    f'[逐点开爪与完整规划回放核验]({url})', '']
        insert = text.index('## 参考方法')
        text[insert:insert] = section
    (website/'内容/迭代/sam3-graspgenx-v4.md').write_text('\n'.join(text))
    write_json(site_path, site)
    write_json(version_path, versions)
    write_json(website/'resources/mobile-v4-publication.json', {'at': now(), 'run': str(root.relative_to(ROOT)),
        'phase': progress['phase'], 'copies': copies})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    publish(parser.parse_args().root.resolve())
