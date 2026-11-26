"""Standalone V8 configuration-preparation results and bounded run status."""
import json
import shutil

from arx_project.layout import ROOT, sha256, write_json


def raw(path): return json.loads(path.read_text())


def section(website, site, active):
    relative = active.get('binding_serialization_run')
    if not relative: return None
    run = (ROOT/relative).resolve()
    if not run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
        raise ValueError('Unexpected binding serialization publication path')
    reg = raw(run/'registration.json');progress = raw(run/'progress.json')
    accepted = raw(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    reviewed = raw(run/'independent_review.json') if (run/'independent_review.json').exists() else None
    verified = bool(reviewed and reviewed.get('all_generated_configuration_values_equal'))
    full = reg['phase'] == 'full';label = '完整69场景对照' if full else '20次先导'
    count = progress['completed_requests'];total = reg['planning_requests']
    status = f'配置准备提速{label}：{count}/{total}次'
    if (run/'STOP').exists(): status += '，已停止'
    elif verified: status += '，独立核验完成'
    elif accepted: status += '，独立核验中'
    proof = raw(ROOT/reg['cpu_preflight']/'acceptance.json')
    summaries = accepted['summaries'] if accepted else progress.get('summaries')
    snapshot = {'run': relative, 'phase': progress['phase'], 'gpu': reg['gpu'],
        'completed_requests': count, 'total_requests': total, 'current': progress.get('current'),
        'primary_time_metric': 'planning_rpc_seconds', 'summaries': summaries,
        'acceptance': {k: v for k, v in accepted.items() if k not in ['cases', 'summaries']} if accepted else None,
        'independent_review': {k: v for k, v in reviewed.items() if k != 'endpoints'} if reviewed else None,
        'CPU_preparation': {k: v for k, v in proof.items() if k not in ['cases', 'codec']}}
    stem = 'sam3-v8-binding-serialization';files = {}
    for source, name in [(ROOT/'modules/pipeline/docs/mainline_versions/V8_配置准备提速.md', stem+'.md'),
                         (ROOT/reg['cpu_preflight']/'acceptance.json', stem+'-cpu.json')]:
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Serialization publication must be an independent file')
        shutil.copy2(source, target)
        if sha256(source) != sha256(target): raise ValueError('Serialization publication copy differs')
        site['localAssets'][name] = '../../resources/media/'+name;files[name] = '/media/'+name
    name = stem+'.json';target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Serialization snapshot must be independent')
    write_json(target, snapshot);site['localAssets'][name] = '../../resources/media/'+name
    lines = ['## V8当前实验：配置准备提速', '', status+'。', '',
        '上一轮的机器人和场景绑定平均约3秒。本轮把约100KB机器人配置的写出交给C版YAML库，'
        '保留原绑定函数、配置读取缓存、机器人参数、FK、碰撞、IK及路径检查。两组共同使用NMS15＋B16。', '',
        '全部69个场景的138份左右臂配置，加上6组闭爪和目标参与组合，144对CPU核对全部一致。'
        '单次配置准备中位141→42毫秒；完整请求收益由本轮测量。', '']
    if summaries:
        lines += ['| 方法 | 完整请求 | 规划通过 | RPC中位耗时 | RPC P90 | RPC累计耗时 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for group, title in [('control', '原版配置写出'), ('fast_binding', 'C版配置写出')]:
            value = summaries[group]
            lines.append(f"| {title} | {value['requests']} | {value['successes']} | {value['median_seconds']:.3f}秒 | "
                         f"{value['p90_seconds']:.3f}秒 | {value['total_seconds']:.3f}秒 |")
        lines += ['', '主表按从发送规划请求到RPC返回计时，包含规划、组件记录与候选快照提交。', '']
    if verified:
        saving = accepted['planning_total_time_reduction']
        lines += [f"{reviewed['checked_endpoints']}个终点独立核验通过，{reviewed['generated_robot_YAML_pairs']}对实际配置值一致，"
            f"累计耗时{'减少' if saving >= 0 else '增加'}{abs(saving):.2%}。", '']
    lines += ['先导按ABBA／BAAB交替，共20次。累计至少省5%、中位数下降、P90增幅不超过5%，'
        '三项都满足才另登记完整69场景对照；本轮不自动切换默认链路。', '',
        f'[配置准备提速方法与结果](/media/{stem}.md) · [完整请求进度与计时](/media/{stem}.json) · '
        f'[144组CPU核对](/media/{stem}-cpu.json)', '']
    return {'body': '\n'.join(lines), 'status': status, 'summary': status,
        'batch': {'id': run.name, 'title': '配置准备提速'+label, 'date': '2026-11-26',
            'status': status, 'summary': status, 'outputs': [relative],
            'artifacts': [{'label': '方法与结果', 'url': '/media/'+stem+'.md'},
                          {'label': '进度与计时', 'url': '/media/'+stem+'.json'}]}}
