"""Independent physical outcome, record and model-origin evidence for four slots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
from training.nearfield_arx.contract import decode, pose_error
from training.nearfield_arx.export import state_time
from training.nearfield_arx.learned_policy import audit_policy
from .prepare_v2 import predecessors
from .runtime import audit_actions
from .capture import audit_episode
from .closure import validate_reply
from .verify_validation import reset_signature
from .verify_physical import entry_matches_raw
from .report import video


def report(campaign):
    campaign = Path(campaign).resolve()
    if (campaign / 'learned_verification.json').exists():raise FileExistsError('Preserve finished result')
    reg = read(campaign / 'registration.json');summary = read(campaign / 'summary.json')
    idle = predecessors([campaign]);verify_files({**reg['execution_source_files'], **reg['dependencies']})
    registered = {t['trial_id']: t for t in reg['seed_trials']};seen = set();details = [];coverage = Counter();pins = {}
    for trial in summary['trials']:
        ident = trial['trial_id']
        if ident not in registered or ident in seen:raise ValueError('Unexpected/repeated registered slot')
        seen.add(ident);spec = registered[ident]
        for key in ('method', 'seed_id', 'G0_world', 'entry_world', 'offset_goal_world', 'placement', 'perturbation'):
            if trial[key] != spec[key]:raise ValueError('Actual execution differs from registered trial')
        db = sqlite3.connect('file:' + str(campaign / 'trial_ledgers' / (ident + '.sqlite3')) + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:attempts = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY started,id')]
        finally:db.close()
        if attempts != trial['attempts'] or len(attempts) > 1:raise ValueError('Trial ledger mismatch')
        detail = {'trial_id': ident, 'object_id': trial['object_id'], 'split': trial['source_split'], 'method': trial['method'],
            'physical_success': trial['success'], 'admitted': trial['admitted'], 'outcome': trial['outcome'],
            'physical_attempt': bool(trial['physical_dispatches']), 'record_integrity_pass': True,
            'wall_seconds': trial['finished_unix'] - trial['started_unix'], 'video': None, 'policy_calls': 0}
        flags = {k: False for k in ('entry_motion', 'fresh_entry', 'model_query', 'policy_motion', 'GRASP', 'closure_gate', 'close', 'lift', 'hold')}
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
        write(episode.parent.parent / 'chronological_evaluation.json', {'result': chronology, 'sequence': sequence,
            'original_receipt_sha256': sha(episode / 'physical_receipt.json'), 'original_task_outcome_unchanged': True})
        raw = [json.loads(l) for l in (episode / 'samples.jsonl').read_text().splitlines()] if (episode / 'samples.jsonl').exists() else []
        capture = audit_episode(episode) if raw else {'applicable': False, 'frames': 0}
        events = read(episode / 'interpreter_events.json') if (episode / 'interpreter_events.json').exists() else []
        policy = audit_policy(episode);occurred = set(actions['executed'])
        flags.update(entry_motion='state_generation' in occurred, fresh_entry=any(e['kind'] == 'physical_entry_stopped' for e in events),
            model_query=bool(policy['policy_calls']), policy_motion=any(n.startswith(('c_step_', 'd_refine_')) for n in occurred),
            GRASP=any(e['kind'] == 'grasp_request' for e in events), close='close' in occurred, lift='lift' in occurred, hold='hold' in occurred)
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
        if physical['pass'] and not all(flags[k] for k in ('fresh_entry', 'model_query', 'GRASP', 'closure_gate', 'close', 'lift', 'hold')):
            raise ValueError('Successful grasp lacks required actual events')
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
        detail.update(event_applicability=flags, capture=capture, actions=actions, policy_calls=policy['policy_calls'],
            convergence=convergence, historical_reset_unchanged=True, physical=physical,
            video=video(episode, campaign / 'videos' / (ident + '.mp4')) if raw else None)
        details.append(detail)
        for f in episode.rglob('*.json*'):pins[str(f)] = sha(f)
    result = {'pass': not summary['incomplete_accounting'], 'record_integrity_pass': not summary['incomplete_accounting'],
        'scheduled': 4, 'completed': len(details), 'actual_attempts': summary['actual_attempts'],
        'learned_policy_trials': sum(d['policy_calls'] > 0 for d in details), 'physical_successes': sum(d['physical_success'] for d in details),
        'physical_failures': sum(d['physical_attempt'] and not d['physical_success'] for d in details),
        'completed_before_physical_dispatch': [d['trial_id'] for d in details if not d['physical_attempt']],
        'unexecuted': summary['unexecuted'], 'unknown_or_incomplete': summary['incomplete_accounting'],
        'event_coverage_trials': dict(coverage), 'trials': details, 'evidence': pins, 'formal_48_trials': 0}
    write(campaign / 'final_idle_verification.json', {'pass': True, 'proof': idle})
    write(campaign / 'learned_verification.json', result)
    text = ['# 首次学习策略四槽验收', '', '| 物体 | 方法 | 实际运动 | 模型调用 | 抓取成功 | 结果 |', '|---|---|---:|---:|---:|---|']
    for d in details:text.append(f'| {d["object_id"]} | {d["method"]} | {int(d["physical_attempt"])} | {d["policy_calls"]} | {int(d["physical_success"])} | {d["outcome"]} |')
    text += ['', f'登记4；实际运动{result["actual_attempts"]}；模型参与{result["learned_policy_trials"]}；成功{result["physical_successes"]}；实际失败{result["physical_failures"]}；未执行{len(result["unexecuted"])}；未完整归档{len(result["unknown_or_incomplete"])}。',
        '', f'另有{len(result["completed_before_physical_dispatch"])}槽在物理运动前被拒绝，单列保留在四槽总分母中。',
        '', f'记录完整性：{result["record_integrity_pass"]}。各事件是否发生见 learned_verification.json；未发生事件不记为检查通过。',
        '每次停稳后的原始模型输入、输出、解码目标及执行回执均保留。抬升超过6厘米并连续保持3秒、所有采样达标的成功判据不变。',
        '这四次用于确认学习闭环的行为与失败位置。下一步依据逐步误差与闭合结果决定48次主比较登记；历史教师结果不作为同期对照。']
    for d in details:
        if d['video']:text.append(f'\n- [{d["trial_id"]} 腕部视频]({d["video"]["path"]})')
    (campaign / 'ACCEPTANCE.md').write_text('\n'.join(text) + '\n');return result


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('campaign');report(p.parse_args().campaign)
