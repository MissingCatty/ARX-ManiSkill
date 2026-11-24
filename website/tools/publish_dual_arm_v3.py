"""Publish completed V3 integrations with independent website media copies."""
import argparse
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def publish(runs):
    website = ROOT/'website'
    site_file = website/'内容/网站.json'
    versions_file = website/'内容/experiments.json'
    site, versions = read_json(site_file), read_json(versions_file)
    experiment = next(v for v in versions['experiments'] if v['id'] == 'sam3-graspgenx')
    version = next(v for v in experiment['iterations'] if v['code'] == 'V3')
    media = website/'resources/media'
    copies, reports, rows, batches = [], [], [], []

    def snapshot(source, name):
        target = media/name
        if target.is_symlink():
            raise ValueError('Publication target must be a real file')
        shutil.copy2(source, target)
        if target.stat().st_nlink != 1 or sha256(target) != sha256(source):
            raise ValueError('Publication snapshot mismatch')
        site.setdefault('localAssets', {})[name] = '../../resources/media/'+name
        copies.append({'source': str(source.relative_to(ROOT)), 'copy': str(target.relative_to(website)),
                       'sha256': sha256(target)})
        return '/media/'+name

    snapshot(ROOT/'configs/experiments/active_experiment.json', 'active-configuration.json')

    for index, root in enumerate(runs, 1):
        report = read_json(root/'acceptance.json')
        if not report['passed'] or report['not_executed']:
            raise ValueError('Only completed, accepted integrations may be published')
        reports.append(report)
        url = snapshot(root/'acceptance.json', f'sam3-v3-integration-{index}-acceptance.json')
        for row in report['results']:
            prefix = f"sam3-v3-r{index}-{row['id'].split('_')[0].lower()}"
            trial = root/'trials'/row['id']
            images = snapshot(trial/'camera/color.png', prefix+'-initial.png')
            videos = {}
            for source, label in [('execution.mp4', '第三人称'), ('head.mp4', '头部')]:
                if source in row['videos']:
                    videos[label] = snapshot(trial/source, prefix+'-'+('overview' if source == 'execution.mp4' else 'head')+'.mp4')
            rows.append({**row, 'batch': root.name, 'image': images, 'video_urls': videos})
        batches.append({'id': root.name, 'title': '三项抓取联调' if len(report['results']) == 3 else '左臂芒果执行联调',
            'date': '2026-11-25', 'status': '已完成',
            'summary': f"登记{report['registered']}项，规划通过{report['planning_passes']}项，实际执行{report['physical_attempts']}项，成功{report['grasp_successes']}项。",
            'outputs': [str(root.relative_to(ROOT))], 'artifacts': [{'label': '独立验收与视频记录', 'url': url}]})
    snapshot(runs[-1]/'resident/curobo_dual_arm/dual_arm/readiness.json', 'sam3-v3-planner-loaded.json')
    total = sum(r['registered'] for r in reports)
    physical = sum(r['physical_attempts'] for r in reports)
    successes = sum(r['grasp_successes'] for r in reports)
    planning = sum(r['planning_passes'] for r in reports)
    left_success = any(r['selected_arm'] == 'left' and r['success'] for r in rows)
    status = '左右臂实际抓取联调完成' if left_success else '双臂规划已接通，左臂接近未到位'
    summary = f'{total}项联调：规划通过{planning}项，实际执行{physical}项，成功{successes}项。'
    version.update(status=status, resultSummary=summary, batches=batches)
    version['settings'] = [['选臂', '原生抓姿排名优先，同一抓姿比较关节位移。'],
        ['球模型', '左右臂各871球，对应球心和半径逐项一致。'],
        ['执行', '原地张开后执行选中臂轨迹，停放臂真实网格进入规划场景。'],
        ['本轮结果', summary]]
    entry = next(v for v in site['versions'] if v['id'] == 'sam3-graspgenx')
    entry.update(status='V3'+status, metric=summary)
    reasons = {'success': '成功', 'sam3_no_target_mask': 'SAM3未找到目标，未开展规划与抓取',
               'no_arm_has_executable_candidate': '左右臂均规划不出来',
               'no_bilateral_target_contact': '闭合后未形成双指夹持',
               'insufficient_hold_height': '抬升与保持高度不足',
               'forbidden_robot_contact': '发生禁止接触', 'approach_not_converged': '接近动作未到位，未下发闭合'}
    side = {'right': '右臂', 'left': '左臂', None: '未执行'}
    table = ['| 目标与设置 | 右臂规划 | 左臂规划 | 实际执行 | 结果 | 最低保持抬升 | 视频 |',
             '| --- | --- | --- | --- | --- | ---: | --- |']
    for row in rows:
        mode = side[row['forced_arm']] if row['forced_arm'] else '自动选臂'
        plans = []
        for arm in ['right', 'left']:
            p = row['arm_results'].get(arm)
            plans.append('未开展' if p is None else f"通过，候选{p['candidate_index']+1}" if p['success'] else '规划不出来')
        height = '—' if row['minimum_hold_lift_m'] is None else f"{row['minimum_hold_lift_m']*100:.2f}厘米"
        videos = ' · '.join(f'[{label}]({url})' for label, url in row['video_urls'].items()) or f"[初始观测]({row['image']})"
        label = {'lemon': '柠檬', 'tomato': '番茄', 'mango': '芒果'}.get(row['target'], row['target'])
        table.append(f"| {label} · {mode} | {plans[0]} | {plans[1]} | {side[row['selected_arm']]} | {reasons.get(row['reason'], row['reason'])} | {height} | {videos} |")
    results = ('## 联调结果\n\n'+summary+' 三种物体的初始三项结果保留，芒果另做一次指定左臂的执行联调。\n\n'+
        '\n'.join(table)+'\n\n成功项的物体实际抬升均超过1厘米并保持至少3秒，独立接触、运动学和视频解码检查通过。\n\n'+
        '[实际加载的双臂规划器核验](/media/sam3-v3-planner-loaded.json) · '+
        '[第一批独立验收](/media/sam3-v3-integration-1-acceptance.json)'+
        (' · [左臂专项独立验收](/media/sam3-v3-integration-2-acceptance.json)' if len(runs)>1 else '')+'\n')
    diagnosis = runs[-1]/'left_approach_diagnosis.json'
    if diagnosis.exists():
        snapshot(diagnosis, 'sam3-v3-left-approach-diagnosis.json')
        results += ('\n## 左臂接近阶段的记录\n\n左臂预抓取到位误差约0.0012毫米。最后接近时已与目标发生手指接触，'
            '末端位置误差约1.89毫米，left_joint15的误差在最后1秒稳定为0.0104 rad；到位门槛为0.005 rad，因此停止在接近阶段。\n\n'
            '| 腕部驱动配置 | 左臂left_joint15 | 右臂right_joint25 |\n| --- | ---: | ---: |\n'
            '| 刚度值 | 20000 | 120000 |\n| 阻尼值 | 800 | 2000 |\n\n'
            '两侧球参数相同，驱动配置存在上述差异。接触负载与驱动跟踪是下一步对照的重点。\n\n'
            '[原始关节、接触与驱动记录](/media/sam3-v3-left-approach-diagnosis.json)\n')
    page = website/'内容/迭代/sam3-graspgenx-v3.md'
    prefix = page.read_text().split('## 当前设置')[0]
    page.write_text(prefix+'## 当前设置\n\n沿用当前200场景、TCP43、升降45.2厘米、头部向下45°、1毫米接触偏移和2毫米局部指尖近桌规则。\n\n'+results+
        '\n## 参考方法\n\n[SAM3](/papers/sam3/) · [RaySt3R](/papers/rayst3r/) · [GraspGenX](/papers/graspgenx/) · [cuRobo](/papers/curobo/)\n')
    doc = ROOT/'modules/pipeline/docs/mainline_versions/V3_双臂选择与抓取.md'
    doc.write_text(doc.read_text().split('## 当前进度')[0]+'## 当前进度\n\n'+status+'，37项离线检查通过。\n\n'+results+
        '\n运行目录：\n\n'+'\n'.join('- `'+str(run.relative_to(ROOT))+'`。' for run in runs)+'\n')
    config_doc = website/'内容/实验配置/夹爪碰撞球精细化.md'
    base = config_doc.read_text().split('### V3左右臂统一球模型')[0].rstrip()
    config_doc.write_text(base+'\n\n### V3左右臂统一球模型\n\n'
        'V3左右臂各871球，对应连杆局部坐标系中的球心和半径逐项相同。左侧连杆名归一化后的参数哈希与右侧一致，'
        '16组关节状态下的球心独立运动学核对通过。运动臂使用871球，停放臂的实测真实碰撞网格加入规划。\n\n'
        '[逐球参数核对](/media/sam3-v3-sphere-comparison.json) · [实际加载核验](/media/sam3-v3-planner-loaded.json) · '
        '[V3规划与执行结果](/iterations/sam3-graspgenx/versions/V3/)\n')
    write_json(site_file, site); write_json(versions_file, versions)
    write_json(website/'resources/metadata/sam3-v3-publication.json', {'at': now(), 'summary': summary,
        'copies': copies, 'runtime_sources_required_for_serving': False,
        'registered_slots': total, 'physical_attempts': physical, 'successes': successes})
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    publish([root.resolve() for root in args.runs])
