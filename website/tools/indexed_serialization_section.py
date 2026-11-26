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
    continuation = reg.get('continuation')
    history = raw(ROOT/continuation['source_run']/'stop_summary.json') if continuation else None
    accepted = raw(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    partial = raw(run/'stop_summary.json') if (run/'stop_summary.json').exists() else None
    reviewed = raw(run/'independent_review.json') if (run/'independent_review.json').exists() else None
    verified = bool(reviewed and reviewed.get('all_generated_configuration_values_equal'))
    full = reg['phase'] == 'full';label = '完整69场景对照' if full else '20次先导'
    count = progress['completed_requests'];total = reg['planning_requests']
    status = f'配置准备提速{label}：{count}/{total}次'
    if (run/'STOP').exists(): status += '，已停止'
    elif verified: status += '，独立核验完成'
    elif accepted: status += '，独立核验中'
    proof = raw(ROOT/reg['cpu_preflight']/'acceptance.json')
    summaries = (accepted['summaries'] if accepted else partial['paired_summaries'] if partial else
                 progress.get('summaries') or (history['paired_summaries'] if history else None))
    stopped = (run/'STOP').exists()
    snapshot = {'run': relative, 'phase': 'stopped' if stopped else progress['phase'], 'gpu': reg['gpu'],
        'completed_requests': count, 'total_requests': total, 'current': None if stopped else progress.get('current'),
        'stopped_at_request': progress.get('current') if stopped else None, 'stop_summary': partial,
        'primary_time_metric': 'planning_rpc_seconds', 'summaries': summaries,
        'continuation': {k: continuation[k] for k in ['source_run', 'inherited_complete_requests',
            'new_request_budget', 'latency_excluded_case_ids']} if continuation else None,
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
    if partial:
        name = stem+'-stop.json';target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Stopped result must be an independent copy')
        shutil.copy2(run/'stop_summary.json', target)
        if sha256(run/'stop_summary.json') != sha256(target): raise ValueError('Stop summary copy differs')
        site['localAssets'][name] = '../../resources/media/'+name
    history_batches = []
    if history:
        previous = ROOT/continuation['source_run'];name = stem+'-'+previous.name.replace('_', '-')+'-stop.json'
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Inherited stop evidence must be a standalone resource')
        shutil.copy2(previous/'stop_summary.json', target)
        site['localAssets'][name] = '../../resources/media/'+name
        history_batches.append({'id': previous.name, 'title': '配置准备完整对照：显存门控停止',
            'date': '2026-11-26', 'status': '109次完成，1次资源未执行，28次未领取',
            'summary': '54个配对场景累计省15.10%；旧记录与STOP保留，后续另建登记承接。',
            'outputs': [continuation['source_run']],
            'artifacts': [{'label': '停止记录与已完成统计', 'url': '/media/'+name}]})
    name = stem+'.json';target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Serialization snapshot must be independent')
    write_json(target, snapshot);site['localAssets'][name] = '../../resources/media/'+name
    lines = ['## V8当前实验：配置准备提速', '', status+'。', '',
        '上一轮的机器人和场景绑定平均约3秒。本轮把约100KB机器人配置的写出交给C版YAML库，'
        '保留原绑定函数、配置读取缓存、机器人参数、FK、碰撞、IK及路径检查。两组共同使用NMS15＋B16。', '',
        '全部69个场景的138份左右臂配置，加上6组闭爪和目标参与组合，144对CPU核对全部一致。'
        '单次配置准备中位141→42毫秒；完整请求收益由本轮测量。', '']
    if continuation:
        lines += [f"继承{continuation['inherited_complete_requests']}次完成记录，本次只执行原未执行的"
            f"{continuation['new_request_budget']}次，原批次停止标记保留。M0072这一对跨服务重启，"
            '保留完整规划结果，主延时表使用其余68对同次常驻运行的场景。', '']
        if accepted:
            groups = accepted['planning_groups']
            lines += [f"完整规划结果：原版{groups['control']['successes']}/{groups['control']['requests']}，"
                f"C版{groups['fast_binding']['successes']}/{groups['fast_binding']['requests']}。", '']
    if partial:
        counts = partial['counts'];complete = partial['completed_requests']
        blocked = sum(v['resource_gate_not_executed'] for v in counts.values())
        unclaimed = sum(v['unclaimed'] for v in counts.values())
        lines += [f'运行时可用显存低于1536 MiB余量门槛，模型服务已退出。已完成{complete}次，'
            f'{blocked}次领取后未运行规划，{unclaimed}次尚未领取。原停止标记保留，没有自动重试。', '',
            '| 方法 | 已完成 | 规划通过 | 资源拦截，未执行 | 尚未领取 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for mode, title in [('control', '原版配置写出'), ('fast_binding', 'C版配置写出')]:
            value = counts[mode]
            lines.append(f"| {title} | {value['complete']}/69 | {value['planning_passes']} | "
                         f"{value['resource_gate_not_executed']} | {value['unclaimed']} |")
        lines += ['', f"下面的时延只比较{partial['complete_pairs']}个已完成的相同场景；"
            '另有1次控制组已完成，待其优化组对照完成后再纳入。', '']
    if summaries:
        lines += ['| 方法 | 计时请求 | 计时样本规划通过 | RPC中位耗时 | RPC P90 | RPC累计耗时 |',
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
    if partial:
        lines += [f"已完成记录完整，{partial['checked_endpoints']}个终点独立FK／限位核验通过，"
            f"{partial['generated_YAML_pairs']}对配置值一致。当前配对累计省时"
            f"{partial['paired_total_time_reduction']:.2%}。", '',
            f'[停止后的记录核对与分组统计](/media/{stem}-stop.json)', '']
    lines += ['先导按ABBA／BAAB交替，共20次。累计至少省5%、中位数下降、P90增幅不超过5%，'
        '三项都满足才另登记完整69场景对照；本轮不自动切换默认链路。', '',
        f'[配置准备提速方法与结果](/media/{stem}.md) · [完整请求进度与计时](/media/{stem}.json) · '
        f'[144组CPU核对](/media/{stem}-cpu.json)', '']
    return {'body': '\n'.join(lines), 'status': status, 'summary': status, 'history_batches': history_batches,
        'batch': {'id': run.name, 'title': '配置准备提速'+label, 'date': '2026-11-26',
            'status': status, 'summary': status, 'outputs': [relative],
            'artifacts': [{'label': '方法与结果', 'url': '/media/'+stem+'.md'},
                          {'label': '进度与计时', 'url': '/media/'+stem+'.json'}]}}
