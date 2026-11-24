"""Read-only ARX scoring: retain first registered pairs and ALL repetitions."""
import argparse
import json
from pathlib import Path

from autoresearch import evaluate
from research_guard import invariant_digest, changed_files


def summarize(campaign):
    campaign = Path(campaign).resolve()
    state = json.loads((campaign / 'state.json').read_text())
    declared = json.loads((campaign / 'scenarios.json').read_text())
    guard = json.loads((campaign / 'configuration_guard.json').read_text())
    rows = []; groups = {}; references = []
    reference_file = campaign / 'reference_controls.json'
    if reference_file.exists():
        for ref in json.loads(reference_file.read_text()):
            folder = Path(ref['folder']); score = evaluate(folder)
            ok = all(invariant_digest(json.loads((folder / name).read_text())) == guard['invariant_sha256']
                     for name in ('initial_fingerprint.json', 'final_fingerprint.json'))
            row = {'trial': folder.name, 'case_id': ref['case_id'], 'arm': 'shared_baseline_reference',
                   'pass': score['pass'] and ok, 'execution': score,
                   'configuration_evidence_pass': ok, 'folder': str(folder),
                   'reference_only_not_a_new_attempt': True}
            references.append(row)
            groups[ref['case_id']] = {'historical_case': ref['historical_case'], 'baseline': row}

    for scenario in declared:
        folder = campaign / 'trials' / scenario['name']
        if not (folder / 'result.json').is_file(): continue
        result = json.loads((folder / 'result.json').read_text())
        score = evaluate(folder)
        evidence = []
        for name in ('initial_fingerprint.json', 'final_fingerprint.json'):
            path = folder / name
            evidence.append(path.is_file() and invariant_digest(json.loads(path.read_text())) == guard['invariant_sha256'])
        layout = json.loads((folder / 'initial_layout_check.json').read_text()) if (folder / 'initial_layout_check.json').exists() else None
        ok = all(evidence) and not result.get('configuration_guard_failed') and (layout is None or layout['pass'])
        policy = json.loads((folder / 'shape_policy.json').read_text()) if (folder / 'shape_policy.json').exists() else None
        row = {'trial': scenario['name'], 'case_id': scenario['case_id'],
               'arm': scenario['research_policy'], 'pass': score['pass'] and ok,
               'execution': score, 'configuration_evidence_pass': ok,
               'layout': layout, 'shape_policy_applied': policy.get('applied') if policy else None,
               'shape_policy_reason': policy.get('reason') if policy else None,
               'input_protocol': result.get('input_protocol', 'Fresh control inference'),
               'outcome': result.get('outcome'), 'error': result.get('error'),
               'folder': str(folder), 'recording': result.get('recording')}
        rows.append(row)
        group = groups.setdefault(scenario['case_id'], {'historical_case': scenario['historical_case']})
        group.setdefault('baseline' if scenario['research_policy'] == 'baseline' else 'candidate', row)
    paired = []
    for case_id, group in groups.items():
        if not all(k in group for k in ('baseline', 'candidate')): continue
        old = group['historical_case']
        paired.append({'case_id': case_id, 'historical_success_case': old.get('successes', 0) > 0 or old.get('historical_success') is True,
                       'historical_failure_case': old.get('failures', 0) > 0,
                       'baseline_success': group['baseline']['pass'], 'candidate_success': group['candidate']['pass'],
                       'configuration_pass': group['baseline']['configuration_evidence_pass'] and group['candidate']['configuration_evidence_pass']})
    checks = [json.loads(p.read_text()) for p in (campaign / 'guard_checks').glob('*.json')
              if not p.name.endswith(('_execution.json', '_mismatch.json'))]
    report = {'state': state['status'], 'scheduled_attempts': len(declared), 'completed_attempts': len(rows),
              'actual_motion_attempts': sum(bool(r['execution']['executed_stages']) for r in rows),
              'shared_control_count': len(references), 'shared_controls': references,
              'completed_pairs': len(paired), 'paired_baseline_successes': sum(p['baseline_success'] for p in paired),
              'paired_candidate_successes': sum(p['candidate_success'] for p in paired),
              'newly_successful_cases': [p['case_id'] for p in paired if not p['baseline_success'] and p['candidate_success']],
              'regressions': [p['case_id'] for p in paired if p['baseline_success'] and not p['candidate_success']],
              'historical_success_subset': [p for p in paired if p['historical_success_case']],
              'historical_failure_subset': [p for p in paired if p['historical_failure_case']],
              'configuration_checks': len(checks), 'configuration_failures': sum(not p.get('pass') for p in checks),
              'current_protected_file_changes': changed_files(guard['file_sha256']),
              'default_promoted': False, 'whole_research_goal_complete': False,
              'scope': 'Shared controls are explicitly referenced from H008, not rerun or independent new trials. Remaining controls are fresh. Incomplete until declared cases/repeats run; selected cases do not estimate population success. Legacy native cases remain separate.',
              'pairs': paired, 'trials': rows,
              'replicate_counts': {case: {arm: {'successes': sum(r['pass'] for r in items), 'attempts': len(items), 'trials': [r['trial'] for r in items]} for arm, items in [('baseline', [r for r in references + rows if r['case_id'] == case and r['arm'] in ('baseline', 'shared_baseline_reference')]), ('candidate', [r for r in rows if r['case_id'] == case and r['arm'] != 'baseline'])]} for case in groups},
              'pair_definition': 'First preregistered control/candidate in each case; repetitions never overwrite them and all attempts remain in replicate_counts/trials. Reused controls are references, not new attempts.'}
    (campaign / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('pairs', 'trials', 'shared_controls', 'historical_success_subset', 'historical_failure_subset', 'replicate_counts')}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--campaign', required=True)
    summarize(parser.parse_args().campaign)
