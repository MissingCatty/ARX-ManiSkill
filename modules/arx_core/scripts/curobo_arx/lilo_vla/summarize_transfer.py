"""Summarize every registered transfer attempt without changing saved scores."""
import argparse
import json
from pathlib import Path

from egad_evaluate import evaluate
from summarize_chunks import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaigns = []
    rows = []
    policy_identities = []
    grasp_goals = []
    initial_conditions = []
    for folder in args.campaigns:
        folder = folder.resolve()
        registration = read(folder / 'registration.json')
        if read(folder / 'state.json')['status'] != 'completed':
            raise ValueError('Incomplete campaign: ' + str(folder))
        archive_matches = all(sha(folder / 'registered_source_snapshot' / Path(p).relative_to(ROOT)) == digest
                              for p, digest in registration['algorithm_sha256'].items())
        physical_matches = all(sha(p) == digest for p, digest in registration['physical_source_sha256'].items())
        if not archive_matches or not physical_matches:
            raise ValueError('Execution source archive or frozen physical files differ')
        policy_identities.append(registration['policy_artifact_sha256'])
        campaigns.append({'path': str(folder), 'registration_sha256': sha(folder / 'registration.json'),
                          'registered_reach_algorithm': registration.get('reach_algorithm', 'original_direct_PRM'),
                          'state': read(folder / 'state.json'), 'archive_matches_registration': archive_matches,
                          'frozen_physical_sources_still_equal': physical_matches,
                          'completion_integrity': read(folder / 'completion_integrity.json')})
        for result in read(folder / 'results.json'):
            trial = folder / result['controller']
            score = evaluate(trial)
            if score != result['evaluation']:
                raise ValueError('Saved physical evaluation changed: ' + str(trial))
            entry = read(trial / 'handoff_actual_entry.json') if (trial / 'handoff_actual_entry.json').exists() else None
            cancellation = trial / 'early_diagnostic_cancellation.json'
            target_checks = {name: read(path) for name in ('pregrasp_via', 'pregrasp')
                             if (path := trial / (name + '_target_validation.json')).exists()}
            report = read(trial / 'grasp_report.json') if (trial / 'grasp_report.json').exists() else {}
            selected = read(trial / 'selected_grasp.json') if (trial / 'selected_grasp.json').exists() else None
            if selected:
                grasp_goals.append(selected['grasp_world'])
            if result['controller'] == 'policy':
                initial_conditions.append(read(trial / 'setup.json')['initial_conditions'])
            classification = ('cancelled_before_motion' if cancellation.exists() else
                              'pregrasp_target_contact' if entry and entry.get('pregrasp_target_contact_events', 0) else
                              'optimized_reach_target_veto' if any(not v['pass'] for v in target_checks.values()) else
                              score['outcome'])
            provenance = trial / 'provenance_runtime_check.json'
            video = trial / 'video_receipt.json'
            if video.exists() and sha(read(video)['video']) != read(video)['sha256']:
                raise ValueError('Video identity differs')
            row = {'trial': str(trial), 'controller': result['controller'], 'pass': score['pass'],
                   'classification': classification, 'evaluation': score,
                   'policy_calls': result.get('policy_calls', 0), 'executed_actions': result.get('executed_actions', 0),
                   'actual_entry_pass': entry.get('pass') if entry else None,
                   'pregrasp_target_contacts': entry.get('pregrasp_target_contact_events') if entry else None,
                   'actual_reach_commands': [p.name for p in sorted(trial.glob('pregrasp*_command.json'))],
                   'target_query_checks': {name: {k: v[k] for k in ('pass', 'samples', 'invalid_samples')}
                                           for name, v in target_checks.items()},
                   'error': report.get('error'), 'elapsed_seconds': result['elapsed_seconds'],
                   'video': read(video) if video.exists() else None,
                   'cancellation': read(cancellation) if cancellation.exists() else None}
            row['reach_selection'] = [
                {'file': str(path), 'algorithm': value['algorithm'], 'accepted': value['accepted'],
                 **{name: {k: value[name][k] for k in ('pass', 'samples', 'invalid_samples') if k in value[name]}
                    for name in ('direct', 'via_route') if name in value}}
                for path in sorted((trial / 'reach_target_checks').glob('*.json'))
                for value in [read(path)]]
            if provenance.exists():
                checked = read(provenance)
                row['neural_action_provenance_pass'] = checked['action_provenance_pass']
                row['frozen_planning_models'] = checked['frozen_planning_models']
                row['actual_frame_state_alignment'] = checked['actual_frame_state_alignment']
                row['provenance_environment'] = checked['state_recheck_environment']
            rows.append(row)
    policy_rows = [r for r in rows if r['controller'] == 'policy']
    all_same_goal = bool(grasp_goals) and all(g == grasp_goals[0] for g in grasp_goals)
    if not all_same_goal or not all(p == policy_identities[0] for p in policy_identities):
        raise ValueError('Expected one unchanged policy and one planned goal for this route comparison')
    controls = args.campaigns[0].resolve() / 'known_success_target_checks.json'
    comparison = {'scope': 'ARX mustard development extension and fixed-goal reach comparison; not an EGAD49 benchmark',
                  'campaigns': campaigns, 'attempts': rows, 'all_recorded_attempts_including_cancelled': len(rows),
                  'policy_attempts_including_no_inference_failures': len(policy_rows),
                  'policy_successes': sum(r['pass'] for r in policy_rows),
                  'teacher_diagnosis_attempts_including_cancelled': len(rows) - len(policy_rows),
                  'same_selected_grasp_goal': all_same_goal, 'same_policy_artifacts': True,
                  'same_declared_initial_conditions': all(c == initial_conditions[0] for c in initial_conditions),
                  'frozen_policy_sha256': policy_identities[0],
                  'first_failure_measured_displacement': read(args.campaigns[0] / 'policy/measured_pregrasp_displacement.json'),
                  'first_failure_measured_model_query_components': read(args.campaigns[0] / 'policy/measured_query_components.json'),
                  'extra_query_on_historical_successful_paths': [
                      {'scene': r['scene'], 'supplementary_query_pass': r['pass'],
                       'invalid_samples': r['invalid_samples'], 'last_sample_valid': r['last_sample_valid'],
                       'actual_historical_grasp_success': evaluate(ROOT / 'runs/lilo_P3_guarded_20261028_02/ten_mm_r1' / r['scene'])['pass'],
                       'historical_actual_entry_target_contacts': read(ROOT / 'runs/lilo_P3_guarded_20261028_02/ten_mm_r1' / r['scene'] / 'handoff_actual_entry.json')['pregrasp_target_contact_events']}
                      for r in read(controls)] if controls.exists() else None,
                  'model_retrained': False, 'global_defaults_changed': False, 'qualified_profile_expanded': False,
                  'stopped_EGAD49_state': read(ROOT / 'runs/lilo_egad49_20261026_eval01/state.json'),
                  'limitations': ['One development object with one fixed grasp goal; no unseen-object success-rate claim',
                                  'Extra target query is conservative and rejects some actual historical successes; opt-in research only',
                                  'Planning certificates do not establish physical success; all measured attempts and cancellations remain counted']}
    latest = args.campaigns[-1].resolve()
    comparison['latest_executed_source_matches_current'] = all(
        sha(path) == digest for path, digest in read(latest / 'registration.json')['algorithm_sha256'].items())
    for name, receipt in [('protocol_checks', 'lilo_P4_protocol_checks_20261028.json'),
                          ('evaluator_compatibility', 'lilo_P4_evaluator_compatibility_20261028.json'),
                          ('shared_video_copies', 'lilo_P4_video_copies_20261028.json')]:
        path = ROOT / 'runs' / receipt
        if path.exists():
            comparison[name] = {'source': str(path), 'sha256': sha(path), 'report': read(path)}
    write(args.output.resolve(), comparison)
    print(json.dumps({k: comparison[k] for k in ('all_recorded_attempts_including_cancelled',
                     'policy_attempts_including_no_inference_failures', 'policy_successes', 'same_selected_grasp_goal')}))


if __name__ == '__main__':
    main()
