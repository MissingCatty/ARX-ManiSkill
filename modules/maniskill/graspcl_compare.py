"""Read-only paired reporting for the matched ARX GraspCL benchmark."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def tally(rows):
    return {'paired_completed': len(rows),
        'original_successes': sum(r['original_success'] for r in rows),
        'graspcl_successes': sum(r['graspcl_success'] for r in rows),
        'improved': sum(not r['original_success'] and r['graspcl_success'] for r in rows),
        'regressed': sum(r['original_success'] and not r['graspcl_success'] for r in rows),
        'both_successful': sum(r['original_success'] and r['graspcl_success'] for r in rows),
        'both_unsuccessful': sum(not r['original_success'] and not r['graspcl_success'] for r in rows),
        'original_physical_attempts': sum(r['original_physical_attempt'] for r in rows),
        'graspcl_physical_attempts': sum(r['graspcl_physical_attempt'] for r in rows),
        'executed_recall_grasps': sum(r['graspcl_physical_attempt'] and r.get('selected_origin', '').startswith('demonstration_') for r in rows),
        'successful_recall_grasps': sum(r['graspcl_success'] and r.get('selected_origin', '').startswith('demonstration_') for r in rows)}


def report(campaign, cache):
    import hashlib
    registration = read(campaign / 'registration.json')
    original = Path(registration['source_baseline'])
    source_ids = set(registration['source_feedback_ids_in_memory_order'])
    pending = []
    for scheduled in registration['schedule']:
        slot = scheduled['id']
        if slot in cache:
            continue
        claim_path = campaign / 'claims' / (slot + '.json')
        if not claim_path.exists():
            continue
        claim = read(claim_path)
        if claim['state'] != 'complete':
            pending.append(slot)
            continue
        trial = campaign / 'trials' / slot
        current_path = trial / 'result.json'
        if hashlib.sha256(current_path.read_bytes()).hexdigest() != claim['result_sha256']:
            raise RuntimeError('Terminal result changed: ' + slot)
        result = read(current_path)
        evaluation = read(trial / 'independent_evaluation.json')
        prior = read(original / 'trials' / slot / 'result.json')
        prior_eval = read(original / 'trials' / slot / 'independent_evaluation.json')
        if not evaluation['passed'] or not prior_eval['passed']:
            raise RuntimeError('Independent evaluation disagrees: ' + slot)
        row = {'id': slot, 'layout': scheduled['layout'], 'model_id': scheduled['model_id'],
            'original_success': prior['success'], 'graspcl_success': result['success'],
            'original_reason': prior['reason'], 'graspcl_reason': result['reason'],
            'original_physical_attempt': prior['physical_attempt'],
            'graspcl_physical_attempt': result['physical_attempt'],
            'feedback_source_case': slot in source_ids,
            'outcome': 'both_successful' if prior['success'] and result['success'] else
                       'improved' if result['success'] else 'regressed' if prior['success'] else 'both_unsuccessful'}
        if (trial / 'self_case_exclusion.json').exists():
            exclusion = read(trial / 'self_case_exclusion.json')
            if slot in exclusion['retained_feedback_ids']:
                raise RuntimeError('Self-case feedback leaked: ' + slot)
            row['excluded_self_feedback'] = exclusion['excluded_feedback_ids']
        if (trial / 'plan_result.json').exists():
            plan = read(trial / 'plan_result.json')
            if plan.get('success'):
                metadata = plan['selected']['candidate_metadata']
                if metadata.get('source_trial_id') == slot:
                    raise RuntimeError('Self-case recall leaked: ' + slot)
                row.update(selected_origin=metadata['origin'],
                    selected_source=metadata.get('source_trial_id'), selected_rank=plan['selected']['index'] + 1,
                    selected_novelty=metadata.get('novelty_against_all_zero'))
        cache[slot] = row
    rows = [cache[e['id']] for e in registration['schedule'] if e['id'] in cache]
    progress = read(campaign / 'progress.json')
    comparison = {'updated_at': datetime.now().astimezone().isoformat(), 'registered': 200,
        'campaign_phase': progress['phase'], 'paired_verified_only': True, 'pending_claims': pending,
        'overall': tally(rows), 'by_layout': {k: tally([r for r in rows if r['layout'] == k]) for k in registration['layouts']},
        'by_bank_source': {k: tally([r for r in rows if r['feedback_source_case'] == flag])
                          for k, flag in [('source_case_excluded_from_own_query', True), ('other_cases', False)]},
        'graspcl_failure_reasons': dict(Counter(r['graspcl_reason'] for r in rows if not r['graspcl_success'])),
        'scope': 'Same declared scene/reset and success gates; frozen bank with self-case exclusion. GraspCL branch also omits the original GT center/height selector, so this is not an isolated scorer ablation. Not unseen-object generalization.',
        'results': rows}
    save(campaign / 'comparison.json', comparison)
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    cache = {}
    while True:
        comparison = report(args.campaign.resolve(), cache)
        if not args.watch or comparison['campaign_phase'] in ['complete', 'stopped'] or (args.campaign / 'halt.json').exists():
            break
        time.sleep(20)
