"""Publish V4 progress/results and independent compressed website snapshots."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from grasp_metrics import fraction_text, sam3_failure_phase, summarize

LAYOUTS = {'S1_open': '开阔', 'S2_edge': '桌边', 'S3_sparse': '稀疏干扰',
           'S4_clutter': '密集杂物', 'S5_occluded': '部分遮挡'}
REASONS = {'success': '成功', 'coarse_sam3_no_target_mask': '初次观测SAM3未分割出目标',
    'pregrasp_not_converged': '预抓取未到位', 'approach_not_converged': '最后接近未到位',
    'close_not_converged': '闭合未到位', 'lift_not_converged': '抬升未到位', 'hold_not_converged': '保持未到位',
    'sam3_no_target_mask': '到位后SAM3未分割出目标', 'coarse_invalid_depth': '初次目标深度不足',
    'coarse_rayst3r_empty_pointcloud': '初次观测未生成目标点云',
    'rayst3r_empty_pointcloud': '到位后未生成目标点云',
    'no_safe_reachable_station': '站位检查未通过，见逐例阶段记录',
    'unable_to_plan': '规划不出来（详见站位与抓取阶段记录）',
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
    high_priority = reg.get('station_search_options', {}).get('placement_method') == 'high_workspace_priority'
    continuation = reg.get('continuation')
    batch_label = '高可达空间优先' if high_priority else '地图主动站位'
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
    if not adjudication:
        complete = {row['id'] for row in summary['results']
                    if read_json(root/'claims'/(row['id']+'.json'))['state'] == 'complete'}
    completed_rows = [row for row in summary['results'] if row['id'] in complete]
    statistics = summarize(completed_rows)
    statistics['by_layout'] = {layout: summarize([row for row in completed_rows if row['layout'] == layout])
                               for layout in LAYOUTS}
    statistics['definition'] = 'Completed trials minus terminal SAM3 failures; recovered SAM3 events remain included'
    statistics['source_summary_sha256'] = sha256(root/'summary.json')
    statistics['completion_source'] = 'independent_acceptance' if adjudication else 'completed_claims'
    statistics['published_at'] = now()
    write_json(root/'publication_statistics.json', statistics)
    site_path, version_path = website/'内容/网站.json', website/'内容/experiments.json'
    site, versions = read_json(site_path), read_json(version_path)
    copies = []
    media = website/'resources/media'
    publication_prefix = reg.get('publication_prefix', 'sam3-v4')

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
            'raw_result_record_count': summary.get('result_records', len(summary['results'])),
            'failure_reasons': progress['failure_reasons'],
            'original_summary_sha256': sha256(summary_source),
            'count_source': 'independent_acceptance_and_claims'}
        states = {row['id']: row['state'] for row in adjudication['results']}
        exported['results'] = [{**row, 'record_state': states[row['id']]} for row in summary['results']]
        summary_source = root/'publication_summary.json'
        write_json(summary_source, exported)
    if reg.get('variant'):
        archive_name = 'sam3-v4-initial100-report.html'
        previous_page = website/'dist/iterations/sam3-graspgenx/versions/V4/index.html'
        if not (media/archive_name).exists() and previous_page.is_file():
            snapshot(previous_page, archive_name)
    progress_url = snapshot(summary_source, publication_prefix+'-summary.json')
    statistics_url = snapshot(root/'publication_statistics.json', publication_prefix+'-statistics.json')
    registered_configuration = root/'source_registered/configs/experiments/active_experiment.json'
    snapshot(registered_configuration if registered_configuration.is_file() else
             ROOT/'configs/experiments/active_experiment.json', publication_prefix+'-configuration.json')
    registration_url = snapshot(root/'registration.json', publication_prefix+'-registration.json')
    snapshot(dataset/'verification.json', publication_prefix+'-scene-verification.json')
    if (root/'acceptance.json').exists():
        snapshot(root/'acceptance.json', publication_prefix+'-acceptance.json')
    titles = {'registered': '100场景已登记，待组件预检', 'running': '100场景抓取测试进行中',
              'complete': '100场景抓取测试完成', 'stopped': '试验停止，保留现场'}
    status = titles.get(progress['phase'], progress['phase'])
    if reg.get('variant'):
        status = batch_label+'新版：'+status
    if continuation:
        status = '原100场景续跑：'+titles.get(progress['phase'], progress['phase'])
    metric = (f"已完成{statistics['completed']}/100例，SAM3失败{statistics['sam3_excluded']}例，"
              f"统计分母{statistics['eligible']}例；规划通过"
              f"{fraction_text(statistics['planning_passes'], statistics['eligible'])}，"
              f"抓取成功{fraction_text(statistics['successes'], statistics['eligible'])}。")
    if progress.get('interrupted'):
        metric += f" 另有{progress['interrupted']}例异常中断、{progress['not_started']}例未开始。"
    if continuation:
        inherited = len(continuation['completed_case_ids'])
        metric += f" 承接旧{inherited}例完整结果，本次新增完成{statistics['completed']-inherited}/{continuation['new_execution_slot_budget']}例。"
    experiment = next(r for r in versions['experiments'] if r['id'] == 'sam3-graspgenx')
    if experiment.get('current') in {None, 'V1', 'V2', 'V3', 'V4'}:
        experiment['current'] = 'V4'
    version = next((r for r in experiment['iterations'] if r['code'] == 'V4'), None)
    if version is None:
        version = {'code': 'V4', 'parent': 'V3', 'sourceId': 'sam3-graspgenx',
            'title': '底盘、升降与双臂自主站位抓取', 'date': '2026-11-25',
            'contentFile': 'sam3-graspgenx-v4.md', 'reportLayout': True,
            'authorization': {'kind': 'explicit-user-request', 'request': reg['user_authorization']}}
        experiment['iterations'].append(version)
    batch = {'id': root.name, 'title': batch_label+'100场景' if reg.get('variant') else 'V4独立100场景',
             'date': '2026-11-25', 'status': status, 'summary': metric,
             'outputs': [str(root.relative_to(ROOT))], 'artifacts': [{'label': '逐例原始结果', 'url': progress_url}]}
    if continuation:
        batch['title'] = f"剩余{continuation['new_execution_slot_budget']}场景续跑，承接原结果"
    batches = [b for b in version.get('batches', []) if b['id'] != root.name]+[batch]
    version.update(status=status, resultSummary=metric,
        change='由目标观测计算底盘位置、朝向和升降高度，停稳后重新感知，再选一臂抓取。',
        pipeline='初次头部RGB-D与SAM3 → 站位和整机移动检查 → 底盘与升降移动 → 停稳重新观测 → SAM3＋RaySt3R＋GraspGenX → 左右臂cuRobo → 抓取',
        scenes='沿用同一组100个冻结场景，五类布局各20例，重新采集观测。' if reg.get('variant') else
               '新采样100例，五类布局各20例，使用共享物体资产。',
        settings=[['底盘', '限速SE2仿真位姿接口，完整机器人路径检查。'],
                  ['升降', '真实joint4原PD控制，0～46厘米。'],
                  ['末段', '重新观测与规划，每臂871球，TCP43和原成功标准。']],
        batches=batches)
    home = next(r for r in site['versions'] if r['id'] == 'sam3-graspgenx')
    if experiment['current'] == 'V4':
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
        '规划通过率和抓取成功率均以“已完成例数减去SAM3最终失败例数”为分母。SAM3失败单独列出；途中分割失败但后来恢复并完成抓取的场景继续计入。', '',
        '| 布局 | 已完成/登记 | SAM3失败 | 统计分母 | 最终规划通过 | 抓取成功 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    if reg.get('variant'):
        old_step = '| 2. 选站位 | 反推底盘XY、朝向和升降高度，检查整机移动路径及左右臂完整抓取规划。 |'
        text[text.index(old_step)] = '| 2. 地图主动选站位 | 从已知可达关节解反算底盘与升降站位，核验FK、限位、V2指尖余量及完整移动路径，再进行抓取规划。 |'
    if high_priority:
        text[text.index('| 1. 初次观测 | 头部RGB-D与SAM3获取可见目标的三维测量，生成原生抓姿。 |')] = \
            '| 1. 初次观测 | 用头部SAM3与实测深度估计目标位置。 |'
        text[text.index('| 2. 地图主动选站位 | 从已知可达关节解反算底盘与升降站位，核验FK、限位、V2指尖余量及完整移动路径，再进行抓取规划。 |')] = \
            '| 2. 优先进入高可达空间 | 先让目标进入双臂共同高可达区域；共同区域受限或抓取规划不通过时，再选择单臂高可达区域，并绑定对应手臂。每次移动前检查整机终点和完整路径。 |'
        text[text.index('| 5. 抓取 | 从左右臂选择可执行抓法，接近、闭合、抬升并保持3秒。 |')] = \
            '| 5. 抓取 | 在新位置生成抓姿并由cuRobo规划；执行前复查观测、机器人状态与规划记录，再完成接近、闭合、抬升和保持。 |'
        index = text.index('## 100场景测试进度')
        text[index:index] = ['高可达区域沿用已登记的推荐操作空间：2厘米体素内每臂至少有29组关节采样、26个朝向桶和4个向下朝向桶。共同区域内两臂均满足；单臂区域内对应手臂满足。站位后再检查实际抓姿的IK、碰撞与完整轨迹。', '']
    for layout, label in LAYOUTS.items():
        data = statistics['by_layout'][layout]
        text.append(f"| {label} | {data['completed']}/20 | {data['sam3_excluded']} | {data['eligible']} | "
                    f"{fraction_text(data['planning_passes'], data['eligible'])} | "
                    f"{fraction_text(data['successes'], data['eligible'])} |")
    remaining = (f"异常中断{progress['interrupted']}例，未开始{progress['not_started']}例。"
                 if progress.get('interrupted') else f"尚未完成{progress['not_completed']}例。")
    text += ['', remaining+f"实际执行抓取{progress['physical_attempts']}例。", '',
             f'[逐例结果]({progress_url}) · [统计明细]({statistics_url}) · [试验登记]({registration_url})', '',
             '## 场景示例', '', '目标放在桌面近侧，每类展示两例。点击图片查看原图。', '',
             '| 布局 | 场景一 | 场景二 |', '| --- | --- | --- |']
    for layout, label in LAYOUTS.items():
        entries = [r for r in reg['schedule'] if r['layout'] == layout]
        cells = []
        for entry in [entries[0], entries[10]]:
            name = publication_prefix+'-'+entry['id'].split('_')[0].lower()+'-initial.jpg'
            source = dataset/'cases'/entry['id']/'head.jpg'
            url = snapshot(source, name)
            cells.append(f"[![{entry['target_text']}]({url})]({url})")
        text.append(f"| {label} | {' | '.join(cells)} |")
    text += ['', '## 失败阶段', '', '| 原因 | 数量 |', '| --- | ---: |']
    text.append(f"| SAM3最终分割失败，单独统计 | {statistics['sam3_excluded']} |")
    for reason, count in statistics['eligible_failure_reasons'].items():
        label = REASONS.get(reason, reason)
        if reason.endswith('_not_converged'):
            label = {'pregrasp': '预抓取', 'approach': '最后接近', 'close': '闭合', 'lift': '抬升', 'hold': '保持'}.get(reason.removesuffix('_not_converged'), '动作')+'未到位'
        text.append(f'| {label} | {count} |')
    if high_priority:
        failures = Counter(row.get('stationing', {}).get('failed_stage', 'unknown')
                           for row in completed_rows
                           if row['reason'] == 'unable_to_plan' and not sam3_failure_phase(row))
        labels = {'high_workspace_positioning': '高可达区域内未找到通过整机移动检查的站位',
                  'grasp_preparation_in_high_workspace': '已移动到高可达区域，后续重新感知或抓取规划未通过'}
        if failures:
            text += ['', '| 规划不出来的具体阶段 | 数量 |', '| --- | ---: |']
            text.extend(f'| {labels.get(stage, stage)} | {count} |' for stage, count in failures.items())
        position_review = website/'content_sections/物体摆放与站位碰撞.md'
        if position_review.is_file():
            text += ['', position_review.read_text(encoding='utf-8').strip(), '']
    videos = []
    for row in sorted(summary['results'], key=lambda r: (not r['success'], r['id'])):
        trial = root/'trials'/row['id']
        if not (trial/'execution.mp4').exists():
            continue
        prefix = publication_prefix+'-'+row['id'].split('_')[0].lower()
        overview = snapshot(trial/'execution.mp4', prefix+'-grasp.mp4')
        links = [f'[抓取]({overview})']
        if (trial/'head.mp4').is_file():
            links.append('[头部]('+snapshot(trial/'head.mp4', prefix+'-head.mp4')+')')
        if (trial/'stationing/motion.mp4').is_file():
            links.insert(0, '[移动]('+snapshot(trial/'stationing/motion.mp4', prefix+'-station.mp4')+')')
        videos.append(f"| {row['target_text']} | {REASONS.get(row['reason'], row['reason'])} | {' · '.join(links)} |")
        if len(videos) == 4:
            break
    if videos:
        text += ['', '## 实际视频', '', '| 目标 | 结果 | 视频 |', '| --- | --- | --- |', *videos]
    if reg.get('predecessor'):
        previous = reg['predecessor']
        text += ['', '## 两轮批次记录', '',
            ('新版沿用同一组100场景，优先把目标移入共同或单臂高可达空间，停稳后重新感知并规划。原有实际结果的批次保留。'
             if high_priority else '新版沿用原100场景、物理与成功判据，启用地图主动站位和已验证的非感知加速。原批次保留故障停止状态。'), '',
            '| 批次 | 完整结束 | 中断 | 未开始 | 抓取成功 |', '| --- | ---: | ---: | ---: | ---: |',
            f"| 原版 | {previous['completed']} | {previous['interrupted']} | {previous['not_started']} | {previous['successes']} |",
            f"| {batch_label}新版 | {progress['completed']} | {progress.get('interrupted', 0)} | {progress.get('not_started', 100-progress['completed'])} | {progress['successes']} |", '',
            '[原版页面快照](/media/sam3-v4-initial100-report.html) · [原版逐例记录](/media/sam3-v4-summary.json) · [原版独立核验](/media/sam3-v4-acceptance.json)', '']
        if continuation:
            text += ['### 本次续跑范围', '',
                f"本次只执行原登记中尚未开始的{continuation['new_execution_slot_budget']}例，保留原场景编号和随机种子。"
                f"原{len(continuation['completed_case_ids'])}例完整结果按文件散列核对后承接；"
                f"{len(continuation['interrupted_case_ids'])}例原中断保留，不重抓。上表新版列为承接后的累计数。", '']
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
    active_search = version.get('activeStationSearch')
    if active_search:
        source = ROOT/active_search['validation_report']
        if not source.is_file():
            source = media/'sam3-v4-active-station-search.json'
        data = read_json(source)
        if source != media/'sam3-v4-active-station-search.json':
            snapshot(source, 'sam3-v4-active-station-search.json')
        verification = ROOT/active_search['verification_report']
        if verification.is_file():
            snapshot(verification, 'sam3-v4-active-station-verification.json')
        section = ['## 可达性地图主动选站位', '',
            '手臂按6自由度规划。底盘XY、转向和升降先确定站位，到位后固定。', '',
            '先保持原始抓法，从已有可达性地图中反算底盘和升降站位；再检查实际夹爪、停放臂、桌面和完整移动路径。'+
            '地图提供的终点关节解经独立复算和原生碰撞检查后直接复用，预抓取与接近路径仍由原规划器生成。', '',
            '以下是同一张显卡、同一份移动前观测的离线规划回放。两种搜索均使用双臂排名截断与批量开爪，最多调用3次完整规划。'+
            '用时包含地图查询、碰撞检查和路径规划，模型先统一预热。', '',
            '| 物体 | 几何站位搜索 | 地图主动搜索 | 耗时减少 | 两组规划结果 |',
            '| --- | ---: | ---: | ---: | --- |']
        labels = {'apple': '苹果', 'peach': '桃子', 'spatula': '锅铲'}
        for pair in data['paired_cases']:
            name = pair['case'].split('_')[-1]
            a, b = pair['grid'], pair['map']
            result = '均通过' if a['success'] and b['success'] else f"原搜索{'通过' if a['success'] else '未通过'}，地图搜索{'通过' if b['success'] else '未通过'}"
            section.append(f"| {labels.get(name, name)} | {a['seconds']:.1f}秒 | {b['seconds']:.1f}秒 | {(1-b['seconds']/a['seconds'])*100:.1f}% | {result} |")
        if all('solver_components' in p['map'] for p in data['paired_cases']):
            section += ['', '### 单次求解耗时', '',
                '以下单独统计地图方案的一次求解接口调用，单位为毫秒，取该例调用的中位数。整次搜索还包含多次尝试、碰撞世界准备和轨迹复核。', '',
                '| 物体 | IK | 路径搜索 | 轨迹优化 |', '| --- | ---: | ---: | ---: |']
            for pair in data['paired_cases']:
                name = pair['case'].split('_')[-1]
                parts = pair['map']['solver_components']
                values = [f"{parts[key]['median_ms']:.1f}" if key in parts else '—'
                          for key in ['native_ik', 'graph_path', 'trajectory_optimization']]
                section.append(f"| {labels.get(name, name)} | {' | '.join(values)} |")
        geometry_path = active_search.get('geometry_validation_report')
        if geometry_path:
            geometry_source = ROOT/geometry_path
            if not geometry_source.is_file():
                geometry_source = media/'sam3-v4-hand-geometry-acceleration.json'
            geometry = read_json(geometry_source)
            if geometry_source != media/'sam3-v4-hand-geometry-acceleration.json':
                snapshot(geometry_source, 'sam3-v4-hand-geometry-acceleration.json')
            section += ['', '### 几何检查的增量优化', '',
                '随后对夹爪网格增加保守包围盒粗筛，仅精算可能碰桌的顶点。以下是已有记录上的CPU配对回放；上面的整次搜索计时保留原批次数据。', '',
                '| 物体 | 关节解记录 | 原几何检查 | 新几何检查 |', '| --- | ---: | ---: | ---: |']
            for row in geometry['paired_cases']:
                name = row['case'].split('_')[-1]
                section.append(f"| {labels.get(name, name)} | {row['proposals']} | {row['reference_median_seconds']*1000:.1f}毫秒 | {row['updated_median_seconds']*1000:.1f}毫秒 |")
            section += ['', '三例共478条记录的通过结果、碰撞部位与原因一致，深度差小于2×10⁻¹²米。固定机器人凸几何也已改为启动时处理一次，场景障碍仍逐例更新。', '',
                '下一步优先减少同站位的重复状态/场景绑定，并批量计算地图关节解的独立FK。感知配置保持现状。', '',
                '[几何检查配对核验](/media/sam3-v4-hand-geometry-acceleration.json)', '']
        section += ['', '沿用V2指尖余量：最前端10毫米允许与桌面上表面最多2毫米接触。快速检查与最终检查读取同一份配置。'+
            '按原排名分批查询抓姿，必要时继续向后查询；地图方案没有通过时，剩余预算仍可使用原候选池。', '',
            f"本次新增实际抓取为{data['physical_grasps']}次；原100例结果保留。", '',
            '[完整对照数据](/media/sam3-v4-active-station-search.json) · [独立核验](/media/sam3-v4-active-station-verification.json)', '']
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
