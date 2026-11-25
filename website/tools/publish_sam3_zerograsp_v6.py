"""Publish paired grasp rates and latency using independent website copies."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arx_project.layout import now, read_json, write_json
from modules.pipeline.experiments.sam3_zerograsp_comparison.metrics import distribution

PAGE = ROOT/'website/内容/迭代/sam3-graspgenx-v6.md'
PREFIX = 'sam3-v6-zerograsp-20261125-01'


def ratio(value, total):
    return f'{value}/{total}（{value/total:.1%}）' if total else '待执行'


def seconds(value):
    if not value['samples']:
        return '—'
    return f"{value['median_seconds']:.3f} / {value['p90_seconds']:.3f} 秒（{value['samples']}例）"


def _publish(root):
    root = Path(root).resolve()
    registered = (root/'registration.json').exists()
    reg = read_json(root/'registration.json') if registered else None
    summary = read_json(root/'summary.json') if (root/'summary.json').exists() else None
    n = summary['completed_pairs'] if summary else 0
    phase = summary['phase'] if summary else read_json(root/'queue_status.json').get('state', 'waiting_for_baseline')
    status = ('已完成' if phase == 'complete' else '故障停止，保留记录' if phase == 'stopped'
        else '正在配对测试' if n else '等待模型预检' if registered else '等待V4结束并验收')
    progress = f'已完成 {n}/{reg["cohort"]["pair_count"]} 个配对场景。' if reg else 'V4续跑结束并验收后，冻结有效SAM观测，按顺序执行新分支。'
    body = ['## 本版进度', '', progress, '',
        '## 比较的两条链路', '',
        '| 方案 | 感知与抓姿生成 | 规划与执行 |', '| --- | --- | --- |',
        '| 当前链路 | 头部RGB-D → SAM3 → RaySt3R重建点云 → GraspGenX生成抓姿 | 双臂cuRobo → 物理抓取 |',
        '| 实验链路 | 同一份头部RGB-D与SAM3掩码 → ZeroGrasp重建并生成抓姿 → 原生碰撞检测、宽度深度修正、NMS | 同一套双臂cuRobo → 物理抓取 |', '',
        '每个场景固定为V4最后一次抓取准备时的站位、升降、关节状态和物体摆放。原链路结果作为配对参照，新分支从相同状态执行一次。', '',
        'SAM3最终分割失败的场景剔除；其余规划失败、空抓和掉落都计入分母。原中断场景和未开始项单独登记。底盘无法通过共同站位门控的场景，两组均保留为规划失败。', '',
        '两组沿用871球、TCP43、1毫米物理接触偏移和V2指尖2毫米桌面余量。计划抬升3厘米，物体至少升高1厘米并保持3秒，同时满足双指接触和动作检查。', '',
        '现有对照组各模块的接口说明：', '', '[[流程卡片]]', '',
        '## 规划与抓取结果', '',
        '| 方案 | 规划成功率 | 抓取成功率 |', '| --- | ---: | ---: |']
    for method, title in [('baseline', 'RaySt3R＋GraspGenX'), ('zerograsp', 'ZeroGrasp')]:
        value = summary[method] if summary else {'planning_passes': 0, 'successes': 0}
        body.append(f'| {title} | {ratio(value["planning_passes"], n)} | {ratio(value["successes"], n)} |')
    if reg:
        excluded = reg['cohort']['excluded']
        body += ['', f"配对队列共{reg['cohort']['pair_count']}例；SAM3最终失败剔除{sum(e['kind']=='sam3_failure' for e in excluded)}例，原中断{sum(e['kind']=='interrupted' for e in excluded)}例。当前两组表格使用同样的{n}个已完成配对场景。"]
    if summary and n:
        body += ['', '### 按布局比较', '',
            '| 场景布局 | 配对数 | 当前链路规划成功率 | ZeroGrasp规划成功率 | 当前链路抓取成功率 | ZeroGrasp抓取成功率 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        labels = {'S1_open': '单物体', 'S2_edge': '桌边', 'S3_sparse': '稀疏多物体', 'S4_clutter': '杂乱多物体', 'S5_occluded': '目标遮挡'}
        for layout, title in labels.items():
            rows = [r for r in summary['results'] if r['layout'] == layout]
            values = [ratio(sum(r[k] for r in rows), len(rows)) for k in
                ['baseline_planning_success', 'zerograsp_planning_success', 'baseline_success', 'zerograsp_success']]
            body.append(f'| {title} | {len(rows)} | '+ ' | '.join(values)+' |')
    latency = {'baseline': {}, 'zerograsp': {}}
    samples = []
    if reg and summary:
        for row in summary['results']:
            path = root/'trials'/row['id']/'paired_latency.json'
            if path.exists():
                sample = read_json(path)
                for method, outcome_path in [
                    ('baseline', root/'inputs'/row['id']/'baseline_result.json'),
                    ('zerograsp', root/'trials'/row['id']/'result.json')]:
                    if not read_json(outcome_path).get('physical_attempt'):
                        sample.get(method, {}).pop('execution_with_replanning_seconds', None)
                new_plan = root/'trials'/row['id']/'plan_result.json'
                if new_plan.exists() and read_json(new_plan).get('reason') == 'zerograsp_no_postprocessed_candidate':
                    sample.get('zerograsp', {}).pop('initial_planning_seconds', None)
                samples.append(sample)
    body += ['', '## 延时', '', '表中依次显示中位数 / P90，并标出实际计时例数。规划环节统计实际调用，执行环节统计实际运动；模型加载和预热单列。', '',
        '| 环节 | RaySt3R＋GraspGenX | ZeroGrasp |', '| --- | ---: | ---: |']
    stages = [('generation_seconds', '重建与抓姿生成'), ('initial_planning_seconds', '首次cuRobo抓取规划'),
        ('generation_and_initial_planning_seconds', '生成与首次规划合计'),
        ('execution_with_replanning_seconds', '执行抓取，含到位后的重规划'),
        ('warm_request_to_terminal_seconds', '生成开始至本次抓取结束')]
    for key, title in stages:
        values = []
        for method in ['baseline', 'zerograsp']:
            value = distribution([s.get(method, {}).get(key) for s in samples])
            latency[method][key] = value
            values.append(seconds(value))
        body.append(f'| {title} | '+ ' | '.join(values)+' |')
    sam = distribution([s.get('baseline', {}).get('shared_sam3_seconds') for s in samples])
    body += ['', f'共用SAM3分割耗时：{seconds(sam)}。', '',
        '重建与生成使用模型同步后的实测耗时；执行耗时从首次规划返回计到最终状态落盘，包含开爪、运动、闭合、抬升、保持及过程中重新规划。每个批次保留GPU型号与UUID，便于核对运行设备。']
    startup = []
    previous_latency = root/'latency_summary.json'
    previous_startup = {item['component']: item for item in
        read_json(previous_latency).get('startup', [])} if previous_latency.exists() else {}
    for kind in ['zerograsp', 'curobo_dual_arm']:
        if not reg:
            continue
        module_kind = reg['component_modules'][kind]
        path = root/'resident'/kind/(module_kind+'_ready.json')
        if path.exists():
            record = read_json(path)
            measured = record['startup_seconds']
            timing_source = 'resident service startup and warmup'
            if (kind in previous_startup and previous_startup[kind]['gpu'] == record['gpu']
                    and previous_startup[kind]['startup_seconds'] >= measured):
                measured = previous_startup[kind]['startup_seconds']
                timing_source = previous_startup[kind]['timing_source']
            proc = Path('/proc')/str(record['pid'])/'stat'
            if proc.exists():
                # The explicit adapter factory runs before shared serve(); include it.
                process_start_ticks = int(proc.read_text().rsplit(') ', 1)[1].split()[19])
                boot = next(float(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                            if line.startswith('btime '))
                start_at = boot+process_start_ticks/os.sysconf('SC_CLK_TCK')
                elapsed = datetime.fromisoformat(record['at']).timestamp()-start_at
                if elapsed >= measured:
                    measured, timing_source = elapsed, 'OS process start to resident readiness'
            startup.append({'component': kind, 'startup_seconds': measured, 'gpu': record['gpu'],
                'resident_internal_startup_seconds': record['startup_seconds'], 'timing_source': timing_source})
    if startup:
        body += ['', '| 首次启动与预热 | 耗时 | GPU |', '| --- | ---: | --- |']
        for item in startup:
            body.append(f'| {item["component"]} | {item["startup_seconds"]:.2f} 秒 | `{item["gpu"]}` |')
    site = read_json(ROOT/'website/内容/网站.json')
    attachments = []

    def attach(source, name):
        destination = ROOT/'website/resources/media'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_symlink():
            raise RuntimeError('Website attachment cannot be a symlink')
        if not (destination.exists() and destination.stat().st_size == source.stat().st_size
                and destination.stat().st_mtime_ns == source.stat().st_mtime_ns):
            pending = destination.with_name(destination.name+f'.{os.getpid()}.pending')
            shutil.copy2(source, pending)
            pending.replace(destination)
        if destination.is_symlink() or destination.stat().st_nlink != 1:
            raise RuntimeError('Website attachment must be an independent file')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    if summary:
        attachments.append(('逐例配对结果', attach(root/'summary.json', PREFIX+'-summary.json')))
        write_json(root/'latency_summary.json', {'at': now(), 'latency': latency, 'shared_sam3': sam, 'startup': startup})
        attachments.append(('延时汇总', attach(root/'latency_summary.json', PREFIX+'-latency.json')))
        videos = [r for r in summary['results'] if r['zerograsp_success']][:3]
        if videos:
            body += ['', '## 配对录像', '']
        for row in videos:
            pair = next(p for p in reg['cohort']['pairs'] if p['id'] == row['id'])
            short = row['id'].split('_')[0].lower()
            old = ROOT/pair['baseline_trial']/'execution.mp4'
            new = root/'trials'/row['id']/'execution.mp4'
            body += [f'### {row["id"]}', '', '| 当前链路 | ZeroGrasp |', '| --- | --- |']
            old_url = attach(old, PREFIX+'-'+short+'-baseline.mp4') if old.exists() else None
            new_url = attach(new, PREFIX+'-'+short+'-zerograsp.mp4')
            body.append('| '+(f'<video controls preload="none" src="{old_url}"></video>' if old_url else '未执行抓取')+
                f' | <video controls preload="none" src="{new_url}"></video> |')
    if attachments:
        body += ['', '## 数据文件', '']+[f'- [{title}]({url})' for title, url in attachments]
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text('\n'.join(body)+'\n')
    registry = read_json(ROOT/'website/内容/experiments.json')
    group = next(g for g in registry['experiments'] if g['id'] == 'sam3-graspgenx')
    version = next((v for v in group['iterations'] if v['code'] == 'V6'), None)
    if version is None:
        version = {'code': 'V6', 'parent': 'V4', 'sourceId': 'sam3-graspgenx',
            'title': '同观测ZeroGrasp生成对照', 'date': '2026-11-25',
            'change': 'SAM3后改用ZeroGrasp替换RaySt3R与GraspGenX，固定相同站位和执行配置。',
            'pipeline': '相同SAM3与RGB-D → ZeroGrasp原生重建/抓姿/后处理 → 同一双臂cuRobo与执行',
            'scenes': 'V4原100场景中完整结束且SAM3最终分割成功的全部场景。',
            'settings': [['比较内容', '规划成功率、抓取成功率及分环节延时。'],
                ['配对条件', '相同SAM3掩码、RGB-D、机器人站位与物体状态。'],
                ['候选来源', 'ZeroGrasp原生碰撞、宽度深度修正与NMS后候选。']],
            'contentFile': PAGE.name, 'reportLayout': True,
            'authorization': {'kind': 'explicit-user-request',
                'request': read_json(ROOT/'configs/experiments/sam3_zerograsp_comparison.json')['user_request']},
            'batches': []}
        group['iterations'].append(version)
    version.update(status=status, resultSummary=progress)
    if reg:
        version['batches'] = [{'id': root.name, 'title': '相同SAM观测与站位的逐例对照', 'date': '2026-11-25',
            'status': status, 'summary': progress, 'outputs': [str(root.relative_to(ROOT))],
            'artifacts': [{'label': title, 'url': url} for title, url in attachments]}]
    if int(group.get('current', 'V1')[1:]) <= 6:
        group['current'] = 'V6'
    write_json(ROOT/'website/内容/experiments.json', registry)
    write_json(ROOT/'website/内容/网站.json', site)
    return {'at': now(), 'version': 'V6', 'status': status, 'completed': n}


def publish(root):
    root = Path(root).resolve()
    with (root/'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(publish(parser.parse_args().root))
