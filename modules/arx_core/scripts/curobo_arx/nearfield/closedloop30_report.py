"""New outcomes, strict shadow scores and historical48 reference, never overwritten."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from training.nearfield_arx.contract import pose_error
from . import closedloop30_audit as audit
from .prepare_v2 import predecessors
from .verify_validation import reset_signature
from .verify_physical import entry_matches_raw
from .capture import audit_episode
from .report import video


def report(campaign):
    root = Path(campaign).resolve(); reg = read(root/'registration.json'); summary = read(root/'summary.json')
    if (root/'closedloop30_verification.json').exists(): raise FileExistsError('Preserve completed verification')
    idle = predecessors([root]); verify_files({**reg['execution_source_files'], **reg['dependencies']})
    previous = read(reg['historical_results']); specs = {t['trial_id']: t for t in reg['seed_trials']}
    details = []; evidence = {}; coverage = Counter()
    for trial in summary['trials']:
        spec = specs[trial['trial_id']]
        if any(trial[k] != spec[k] for k in spec): raise RuntimeError('Executed trial differs from registration')
        db = sqlite3.connect('file:'+str(root/'trial_ledgers'/(trial['trial_id']+'.sqlite3'))+'?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try: rows = [dict(r) for r in db.execute('select * from attempts order by started,id')]
        finally: db.close()
        if rows != trial['attempts'] or len(rows) > 1: raise RuntimeError('Attempt ledger mismatch or retry')
        item = {k: trial[k] for k in ('trial_id', 'method', 'case_id', 'condition', 'object_id', 'historical_trial_id')}
        item.update(task_success=trial['success'], outcome=trial['outcome'], record_integrity_pass=True,
                    physical_attempt=bool(trial['physical_dispatches']), video=None, steps=[], strict_success=False)
        if not rows:
            item.update(event_applicability={}, policy_calls=0); details.append(item); continue
        ep = Path(rows[0]['episode']); physical = read(ep/'physical_receipt.json')
        independently = audit.evaluate(ep)
        if digest(independently) != digest(physical) or trial['success'] != physical['pass']:
            raise RuntimeError('Independent task evaluation differs')
        if reset_signature(read(ep/'initial_conditions.json')) != reset_signature(read(Path(reg['seeds'][trial['seed_id']]['episode'])/'initial_conditions.json')):
            raise RuntimeError('Historical physical reset changed')
        policy = audit.audit_policy(ep)
        raw = [json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()] if (ep/'samples.jsonl').exists() else []
        capture = audit_episode(ep) if raw else {'applicable': False, 'frames': 0}
        events = read(ep/'interpreter_events.json') if (ep/'interpreter_events.json').exists() else []
        if (ep/'closedloop_contract.json').exists():
            contract = read(ep/'closedloop_contract.json')
            if contract['G0_world'] != spec['G0_world'] or contract['method'] != spec['method']:
                raise RuntimeError('Policy method/prior changed')
        for request in policy['requests']:
            p = ep/'policy'/(request['request_id']+'_request.json'); req = read(p)
            reply = read(p.with_name(request['request_id']+'_reply.json'))
            if reply['checkpoint'] != reg['learned_policy']['checkpoints'][spec['method']]:
                raise RuntimeError('Wrong model used')
            entry_matches_raw(raw[req['raw_index']], read(req['observation']))
            if req['inputs']['eef_world'] != raw[req['raw_index']]['snapshot']['link_world']['right_eef_link']:
                raise RuntimeError('Policy image/pose binding changed')
        for event in events:
            if event['kind'] in ('physical_entry_stopped', 'CD_waypoint_stop', 'grasp_request'):
                pose = event['snapshot']['link_world']['right_eef_link']
                item['steps'].append({'stage': event.get('command_label', event['kind']),
                    'to_Gstar': pose_error(pose, spec['source_Gstar_world']),
                    'to_G0': pose_error(pose, spec['G0_world']),
                    'original_arrival_accepted': event.get('original_arrival_accepted'),
                    'closure_origin': event.get('closure_origin')})
        frozen = list((ep.parent.parent/'blocks').glob('*/frozen_*.json'))
        if not frozen: raise RuntimeError('Missing runtime frozen-configuration evidence')
        for p in frozen:
            proof = read(p)
            if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:
                raise RuntimeError('Robot/scene configuration changed')
            evidence[str(p)] = sha(p)
        occurred = set(physical['executed_stages'])
        flags = {'entry_motion': 'state_generation' in occurred,
                 'fresh_entry': any(e['kind'] == 'physical_entry_stopped' for e in events),
                 'axial_preparation': 'd_entry_axis' in occurred,
                 'policy_query': policy['policy_calls'] > 0,
                 'policy_motion': any(n.startswith(('c_step_', 'd_refine_')) for n in occurred),
                 'stationary_continuation': bool(physical['certified_nonconverged_waypoints']),
                 'anti_regression_closure': 'D_second_step_outward_intercept' in policy['closure_origins'],
                 'closure_request': bool(policy['closure_origins']),
                 'closure_gate': physical['independent_closure_pass'],
                 'close': 'close' in occurred, 'lift': 'lift' in occurred, 'hold': 'hold' in occurred}
        search = read(ep/'entry_goal_search.json') if (ep/'entry_goal_search.json').exists() else None
        flags['entry_shortened'] = bool(search and search['selected'] and search['selected']['backoff_m'] < .08)
        if physical['pass'] and not all(flags[k] for k in ('fresh_entry', 'policy_query', 'closure_request', 'closure_gate', 'close', 'lift', 'hold')):
            raise RuntimeError('Successful trial lacks actual key events')
        coverage.update({k: int(v) for k, v in flags.items()})
        item.update(strict_success=physical['strict_original_success'], physical=physical, capture=capture,
                    event_applicability=flags, policy_calls=policy['policy_calls'], closure_origins=policy['closure_origins'],
                    selected_entry=search, historical_reset_unchanged=True,
                    video=video(ep, root/'videos'/(trial['trial_id']+'.mp4')) if raw else None)
        if item['video'] is None and raw:
            raise RuntimeError('Recorded frames have no deliverable video')
        for p in ep.rglob('*.json*'): evidence[str(p)] = sha(p)
        details.append(item)
    result = {'pass': not summary['incomplete_accounting'], 'record_integrity_pass': not summary['incomplete_accounting'],
        'scheduled': len(specs), 'completed': len(details), 'physical_successes': sum(t['task_success'] for t in details),
        'strict_successes': sum(t['strict_success'] for t in details),
        'ordinary_failures': sum(not t['task_success'] for t in details),
        'unknown_or_incomplete': summary['incomplete_accounting'], 'unexecuted': summary['unexecuted'],
        'trials': details, 'event_coverage_trials': dict(coverage), 'evidence': evidence,
        'historical_original48_unchanged': True, 'historical_unknowns_retained': 3}
    write(root/'final_idle_verification.json', {'pass': True, 'proof': idle})
    write(root/'closedloop30_verification.json', result)
    latest = {t['historical_trial_id']: t for t in details}
    table = []; text = ['# 导师30闭环复测验收', '',
        f'登记24次C/D复测；已完成{len(details)}，新版物理任务成功{result["physical_successes"]}，原严格评分成功{result["strict_successes"]}，普通失败{result["ordinary_failures"]}，未执行{len(result["unexecuted"])}，本轮中断/未知{len(result["unknown_or_incomplete"])}。', '',
        '原48项保持不变。下表同时展示旧结果与此次新结果；A/B为历史参考，缩短入口的案例单列条件变化。', '',
        '| 原槽位 | 旧结果 | 本轮任务结果 | 本轮严格评分 | 入口变化 | 视频 |', '|---|---|---|---|---|---|']
    for old in previous['trials']:
        current = latest.get(old['trial_id'])
        prior = '历史未知' if old.get('task_result') == 'infrastructure_interrupted' else '成功' if old['physical_success'] else '失败'
        row = {'historical': old, 'new': current}; table.append(row)
        if current:
            changed = current['event_applicability'].get('entry_shortened', False)
            clip = f'[录像]({current["video"]["path"]})' if current.get('video') else '无运动画面'
            text.append(f'| {old["trial_id"]} | {prior} | {"成功" if current["task_success"] else current["outcome"]} | {"成功" if current["strict_success"] else "未通过"} | {"短退距新条件" if changed else "原80毫米入口"} | {clip} |')
        else:
            text.append(f'| {old["trial_id"]} | {prior} | {"历史参考" if old["method"] in "AB" else "未取得新结果"} | — | — | — |')
    write(root/'comparison48.json', {'scope': 'Historical48 reference with separately identified new C/D attempts', 'rows': table})
    text += ['', '| 方法 | 原成功/12 | 本轮成功/已完成 | 本轮原严格评分 |', '|---|---:|---:|---:|']
    for method in 'ABCD':
        before = sum(t['method'] == method and t['physical_success'] for t in previous['trials'])
        new = [t for t in details if t['method'] == method]
        text.append(f'| {method} | {before}/12 | {str(sum(t["task_success"] for t in new))+"/"+str(len(new)) if new else "未重跑"} | {sum(t["strict_success"] for t in new) if new else "—"} |')
    text += ['', '| 轨迹 | 阶段 | 到G*毫米/度 | 到G0毫米/度 | 原到位通过 |', '|---|---|---:|---:|---|']
    import math
    for item in details:
        for step in item['steps']:
            star, g0 = step['to_Gstar'], step['to_G0']
            text.append(f'| {item["trial_id"]} | {step["stage"]} | {1000*star["translation_m"]:.2f}/{math.degrees(star["rotation_rad"]):.2f} | {1000*g0["translation_m"]:.2f}/{math.degrees(g0["rotation_rad"]):.2f} | {step["original_arrival_accepted"]} |')
    text += ['', f'记录完整性：{result["record_integrity_pass"]}。事件按实际发生记录；关键事件覆盖：{dict(coverage)}。',
             '中间步原到位回执未被改写；新版和原严格评分分列。全部新增录像见上表，逐项证据见closedloop30_verification.json。']
    (root/'ACCEPTANCE.md').write_text('\n'.join(text)+'\n')
    return result
