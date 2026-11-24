"""Independent original48 outcome/evidence audit; learned and baseline origins separate."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx.stage_recovery import decode
from training.nearfield_arx.export import state_time
from training.nearfield_arx.stage_verification import audit_policy
from .main48_audit import audit_baseline
from .main48_spec import validate_schedule
from .prepare_v2 import predecessors
from .runtime import audit_actions
from .capture import audit_episode
from .closure import validate_reply
from .verify_validation import reset_signature
from .verify_physical import entry_matches_raw
from .report import video


def task_counts(details):
    """Infrastructure interruptions consume a slot, but are not grasp failures."""
    interrupted=[d for d in details if d.get('task_result')=='infrastructure_interrupted']
    return {'physical_successes':sum(d['physical_success'] for d in details),
        'physical_failures':sum(d['physical_attempt'] and not d['physical_success'] and d.get('task_result')!='infrastructure_interrupted' for d in details),
        'infrastructure_interruptions':len(interrupted),
        'unobserved_task_outcomes':[d['trial_id'] for d in interrupted]}


def unexecuted_partition(reg,summary):
    """User-deferred slots remain unexecuted, never failures or consumed trials."""
    pending=list(summary['unexecuted']);deferred=[t['trial_id'] for t in reg.get('deferred_trials',[])]
    if len(pending)!=len(set(pending)) or set(pending)&set(deferred):raise ValueError('Duplicate unexecuted accounting')
    if not set(pending)<={t['trial_id'] for t in reg['seed_trials']}:raise ValueError('Foreign pending slot')
    ids=set(pending)|set(deferred)
    full=reg.get('registered_trials',reg['seed_trials'])
    return {'unexecuted':[t['trial_id'] for t in full if t['trial_id'] in ids],
            'pending_unexecuted':pending,'deferred_unexecuted':deferred}


def report(campaign,output=None):
    campaign = Path(campaign).resolve()
    output=campaign if output is None else Path(output).resolve()
    if (output / 'main48_verification.json').exists():raise FileExistsError('Preserve finished result')
    output.mkdir(parents=True,exist_ok=True)
    reg = read(campaign / 'registration.json');summary = read(campaign / 'summary.json')
    validate_schedule(reg)
    idle = predecessors([campaign]);verify_files({**reg['execution_source_files'], **reg['dependencies']})
    registered = {t['trial_id']: t for t in reg.get('registered_trials',reg['seed_trials'])}
    details=copy.deepcopy(reg.get('carried_details',[]));seen={t['trial_id'] for t in details};coverage=Counter()
    pins=copy.deepcopy(reg.get('carried_evidence',{}));verify_files(pins)
    for t in details:coverage.update({k:int(v) for k,v in t['event_applicability'].items()})
    for trial in summary['trials']:
        ident = trial['trial_id']
        if ident not in registered or ident in seen:raise ValueError('Unexpected/repeated registered slot')
        seen.add(ident);spec = registered[ident]
        for key in ('method', 'seed_id', 'Gbase_world', 'G0_world', 'entry_world', 'offset_goal_world', 'placement', 'perturbation', 'condition', 'case_id'):
            if trial[key] != spec[key]:raise ValueError('Actual execution differs from registered trial')
        db = sqlite3.connect('file:' + str(campaign / 'trial_ledgers' / (ident + '.sqlite3')) + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:attempts = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY started,id')]
        finally:db.close()
        if attempts != trial['attempts'] or len(attempts) > 1:raise ValueError('Trial ledger mismatch')
        detail = {'trial_id': ident, 'object_id': trial['object_id'], 'condition':trial['condition'], 'case_id':trial['case_id'], 'split': trial['source_split'], 'method': trial['method'],
            'physical_success': trial['success'], 'admitted': trial['admitted'], 'outcome': trial['outcome'],
            'physical_attempt': bool(trial['physical_dispatches']), 'record_integrity_pass': True,
            'wall_seconds': trial['finished_unix'] - trial['started_unix'], 'video': None, 'policy_calls': 0, 'geometry_calls':0}
        flags = {k: False for k in ('entry_motion', 'fresh_entry', 'model_query', 'geometry_query', 'policy_motion', 'baseline_motion', 'GRASP', 'closure_check', 'closure_gate', 'close', 'lift', 'hold')}
        blocks=list((campaign/'groups').glob('*/trials/'+ident+'/blocks/*/frozen_*.json'))
        if not blocks:raise ValueError('Missing live configuration evidence even for pre-dispatch rejection')
        for f in blocks:
            proof=read(f)
            if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:raise ValueError('Frozen configuration changed')
            pins[str(f)]=sha(f)
        if not attempts:
            if trial['success'] or trial['physical_dispatches']:raise ValueError('Credited outcome without physical attempt')
            detail.update(event_applicability=flags);details.append(detail);continue
        episode = Path(attempts[0]['episode']);physical = read(episode / 'physical_receipt.json')
        if physical['pass'] != trial['success']:raise ValueError('Physical result mismatch')
        seed = reg['seeds'][spec['seed_id']]
        if reset_signature(read(episode / 'initial_conditions.json')) != reset_signature(read(Path(seed['episode']) / 'initial_conditions.json')):
            raise ValueError('Historical initial state changed')
        requests = read(episode / 'requests.json');actions = audit_actions(episode, requests)
        sequence = [(label if label in ('pregrasp', 'approach', 'close', 'lift', 'hold') else 'approach', label)
            for r in requests if (label := r['label']) in physical.get('executed_stages', [])]
        chronology = evaluate(episode, sequence=sequence)
        if chronology['pass'] != physical['pass']:raise ValueError('Independent original physical evaluation mismatch')
        write(output/'chronological_evaluations'/(ident+'.json'), {'result': chronology, 'sequence': sequence,
            'original_receipt_sha256': sha(episode / 'physical_receipt.json'), 'original_task_outcome_unchanged': True})
        raw = [json.loads(l) for l in (episode / 'samples.jsonl').read_text().splitlines()] if (episode / 'samples.jsonl').exists() else []
        capture = audit_episode(episode) if raw else {'applicable': False, 'frames': 0}
        events = read(episode / 'interpreter_events.json') if (episode / 'interpreter_events.json').exists() else []
        policy = audit_policy(episode) if spec['method'] in ('C','D') else audit_baseline(episode,spec)
        occurred = set(actions['executed'])
        flags.update(entry_motion='state_generation' in occurred, fresh_entry=any(e['kind'] == 'physical_entry_stopped' for e in events),
            model_query=bool(policy['policy_calls']), policy_motion=any(n.startswith(('c_step_', 'd_refine_')) for n in occurred),
            geometry_query=bool(policy.get('geometry_calls',0)), baseline_motion=any(n.startswith(('a_approach','b_refine_')) for n in occurred),
            GRASP=any(e['kind'] == 'grasp_request' for e in events), closure_check=(episode/'closure_model_validation.json').is_file(),
            close='close' in occurred, lift='lift' in occurred, hold='hold' in occurred)
        if flags['fresh_entry']:
            entry = [e for e in events if e['kind'] == 'physical_entry_stopped']
            if len(entry) != 1 or not flags['entry_motion']:raise ValueError('Entry lacks unique executed arrival')
            e = entry[0];row = raw[e['raw_index']];generation = read(episode / 'state_generation_execution.json')
            if state_time(row) <= generation['final']['sim_time'] + 1e-6:raise ValueError('Stale entry frame')
            if e['registered_G0_world'] != spec['G0_world'] or e['registered_H_world'] != spec['entry_world']:
                raise ValueError('Fixed prior or entry differs')
            entry_matches_raw(row, read(episode / 'physical_entry_observation.json'))
            drift = pose_error(generation['final']['objects']['test_object']['world'], read(episode / 'capture_snapshot.json')['objects']['test_object']['world'])
            target = generation['final']['objects']['test_object']['path']
            if drift['translation_m'] > .001 or drift['rotation_rad'] > .01 or any(c['num_contacts'] > 0 and target in c['a'] + c['b'] for c in generation['contacts']):
                raise ValueError('Physical entry moved the object')
        convergence = []
        for item in policy['requests']:
            identp = item['request_id'];request = read(episode / 'policy' / (identp + '_request.json'))
            reply = read(episode / 'policy' / (identp + '_reply.json'))
            if request['inputs']['G0_world'] != spec['G0_world'] or reply['checkpoint'] != reg['learned_policy']['checkpoints'][spec['method']]:
                raise ValueError('Fixed G0/checkpoint binding changed')
            row = raw[request['raw_index']];entry_matches_raw(row, read(request['observation']))
            if request['inputs']['eef_world'] != row['snapshot']['link_world']['right_eef_link']:raise ValueError('Action axes differ from input image')
            local = [e for e in events if e.get('policy_request_id') == identp and e['kind'] == 'target_request']
            if local:
                goal = decode(spec['method'], request['inputs']['eef_world'], reply['prediction'].strip(), spec['G0_world'])
                np.testing.assert_allclose(goal, local[0]['goal_world'], atol=1e-12, rtol=0)
            error = pose_error(request['inputs']['eef_world'], spec['source_Gstar_world'])
            convergence.append({**item, 'diagnostic_to_historical_Gstar': error})
        if flags['close']:
            gate = read(episode / 'closure_gate_optimization.json');binding = gate['binding']
            if not validate_reply(binding, read(episode / 'closure_model_validation.json'), read(episode / (binding['previous_command_label'] + '_command.json'))):
                raise ValueError('Missing independent explicit-GRASP gate')
            verify_files(binding['files']);flags['closure_gate'] = True
        required=('fresh_entry','GRASP','closure_gate','close','lift','hold') + (('model_query',) if spec['method'] in ('C','D') else ('baseline_motion',))
        if physical['pass'] and not all(flags[k] for k in required):
            raise ValueError('Successful grasp lacks required actual events')
        for query in policy.get('geometry_queries',[]):
            request=read(episode/'geometry'/(query['request_id']+'_request.json'))
            entry_matches_raw(raw[request['raw_index']],read(request['observation']))
        frozen = list((episode.parent.parent / 'blocks').glob('*/frozen_*.json'))
        if not frozen:raise ValueError('Missing live configuration evidence')
        for f in frozen:
            proof = read(f)
            if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:raise ValueError('Frozen robot/scene changed')
            pins[str(f)] = sha(f)
        if trial['admitted']:
            quality = read(episode / 'quality_receipt.json')
            if not quality['pass']:raise ValueError('Admitted episode failed original quality')
            verify_files({str(episode / n): h for n, h in quality['files'].items()})
        coverage.update({k: int(v) for k, v in flags.items()})
        stamps=[e['sim_time'] for e in events if e['kind']=='physical_entry_stopped']
        detail.update(event_applicability=flags, capture=capture, actions=actions, policy_calls=policy['policy_calls'],
            geometry_calls=policy.get('geometry_calls',0),geometry_queries=policy.get('geometry_queries',[]),
            nearfield_sim_seconds=(raw[-1]['capture_sim_time']-stamps[0] if stamps and raw else None),
            nearfield_wall_seconds=(trial['finished_unix']-next(e['wall_unix'] for e in events if e['kind']=='physical_entry_stopped') if stamps else None),
            convergence=convergence, historical_reset_unchanged=True, physical=physical,
            video=video(episode, output / 'videos' / (ident + '.mp4')) if raw else None)
        details.append(detail)
        for f in episode.rglob('*.json*'):pins[str(f)] = sha(f)
    result = {'pass': not summary['incomplete_accounting'], 'record_integrity_pass': not summary['incomplete_accounting'],
        'scheduled': 48, 'completed': len(details), 'actual_attempts': summary['actual_attempts']+sum(t['physical_attempt'] for t in reg.get('carried_details',[])),
        'learned_policy_trials': sum(d['policy_calls'] > 0 for d in details), **task_counts(details),
        'completed_before_physical_dispatch': [d['trial_id'] for d in details if not d['physical_attempt']],
        **unexecuted_partition(reg,summary), 'unknown_or_incomplete': summary['incomplete_accounting'],
        'event_coverage_trials': dict(coverage), 'trials': details, 'evidence': pins, 'formal_48_trials':len(details)}
    result['native_record_completeness_pass']=not summary['incomplete_accounting'] and not any(not d.get('native_runtime_completed',True) for d in details)
    result['native_completed']=sum(d.get('native_runtime_completed',True) for d in details)
    result['supplemented_terminal_failures']=[d['trial_id'] for d in details if not d.get('native_runtime_completed',True)]
    result['record_integrity_scope']='Occurred action/request chains and explicit supplemental accounting; native terminal gaps are reported separately'
    write(output / 'final_idle_verification.json', {'pass': True, 'proof': idle})
    aggregates=[]
    for method in 'ABCD':
        rows=[d for d in details if d['method']==method];entered=[d for d in rows if d['event_applicability']['fresh_entry']]
        aggregates.append({'method':method,'scheduled':12,'completed':len(rows),'success':sum(d['physical_success'] for d in rows),
            'failed_completed':sum(not d['physical_success'] and d.get('task_result')!='infrastructure_interrupted' for d in rows),
            'unexecuted':sum(registered[i]['method']==method for i in result['unexecuted']),
            'user_deferred':sum(registered[i]['method']==method for i in result['deferred_unexecuted']),
            'unknown':sum(t['method']==method for t in summary['incomplete_accounting'])+sum(d.get('task_result')=='infrastructure_interrupted' for d in rows),
            'entered_nearfield':len(entered),'success_after_entry':sum(d['physical_success'] for d in entered),
            'full_pipeline_wall_seconds':[d['wall_seconds'] for d in rows]})
    result['methods']=aggregates
    write(output / 'main48_verification.json', result)
    text = ['# 48次系统主对照验收', '',
        '模型和执行器采用回复42锁定版本。A为固定粗目标规划，B为腕部Graspness局部候选修正，C为多步视觉接近，D为最多两次视觉修正。四组共享原机器人、历史初态、渲染、闭合、抬升与保持判据。', '',
        '| 方法 | 成功／登记 | 已结束失败 | 未执行 | 中断／未知 | 进入近场后成功／入口数 |', '|---|---:|---:|---:|---:|---:|']
    for a in aggregates:text.append(f'| {a["method"]} | {a["success"]}/12 | {a["failed_completed"]} | {a["unexecuted"]} | {a["unknown"]} | {a["success_after_entry"]}/{a["entered_nearfield"]} |')
    text+=['','| 物体 | 条件 | 方法 | 实际运动 | VLM调用 | 几何调用 | 抓取成功 | 结果 |','|---|---|---|---:|---:|---:|---:|---|']
    for d in details:text.append(f'| {d["object_id"]} | {d["condition"]} | {d["method"]} | {int(d["physical_attempt"])} | {d["policy_calls"]} | {d["geometry_calls"]} | {int(d["physical_success"])} | {d["outcome"]} |')
    text += ['', f'登记48；实际运动{result["actual_attempts"]}；VLM参与{result["learned_policy_trials"]}；成功{result["physical_successes"]}；抓取失败{result["physical_failures"]}；基础设施中断{result["infrastructure_interruptions"]}；未执行{len(result["unexecuted"])}；未完整归档{len(result["unknown_or_incomplete"])}。',
        '', f'另有{len(result["completed_before_physical_dispatch"])}槽在物理运动前被拒绝，保留在登记分母中。未知槽不计为普通抓取失败。',
        '', f'动作与请求链完整性：{result["record_integrity_pass"]}；原生终结记录全部完整：{result["native_record_completeness_pass"]}。事后承接的失败槽为{result["supplemented_terminal_failures"]}，旧记录不改写，不补造动作或成功。事件适用性与覆盖见main48_verification.json；未发生事件不记为通过。',
        '每次停稳后的原始模型输入、输出、解码目标及执行回执均保留。抬升超过6厘米并连续保持3秒、所有采样达标的成功判据不变。',
        'BBQ和Spam是留出验证物体，此前已经参与验证或诊断；结果不代表盲测。三条件各一次用于定位差异，不能提供可靠的总体成功率置信区间。完整耗时包含共同准备与入口；近场仿真时间与视频播放时长单独记录。',
        '后续先分析同case配对的失败阶段与耗时；本登记不追加训练、局部调参或失败重试。']
    if result['deferred_unexecuted']:
        text += ['',f'按本轮用户要求暂缓后续D组，共{len(result["deferred_unexecuted"])}个名额；它们属于未执行，保留原48登记分母，不计抓取失败，也不视为已经完成。其余A/B/C沿用原相对顺序。',
                 '暂缓清单：'+', '.join(result['deferred_unexecuted'])]
    for d in details:
        if d['video']:text.append(f'\n- [{d["trial_id"]} 腕部视频]({d["video"]["path"]})')
    (output / 'ACCEPTANCE.md').write_text('\n'.join(text) + '\n');return result


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('campaign');report(p.parse_args().campaign)
