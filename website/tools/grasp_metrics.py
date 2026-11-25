"""Publication metrics for completed grasp trials after SAM3 failure exclusion."""
from collections import Counter


SAM3_FAILURE_REASONS = {'sam3_no_target_mask', 'coarse_sam3_no_target_mask'}


def sam3_failure_phase(row):
    """Classify terminal SAM3 failures, including stationing's wrapped reason."""
    if row.get('success'):
        return None
    reason = row.get('reason')
    if reason in SAM3_FAILURE_REASONS:
        return 'initial_observation' if reason.startswith('coarse_') else 'after_stationing'
    stationing = row.get('stationing', {})
    history = stationing.get('priority_history', [])
    observed = [event for event in history if event.get('geometry_ready')]
    if (reason == 'unable_to_plan' and not row.get('planning_success')
            and observed and not any(event.get('native_planning_calls', 0) for event in history)
            and all(event.get('failure') in SAM3_FAILURE_REASONS for event in observed)):
        return 'after_stationing'
    return None


def summarize(rows):
    excluded = [row for row in rows if sam3_failure_phase(row)]
    eligible = [row for row in rows if not sam3_failure_phase(row)]
    denominator = len(eligible)
    successes = sum(bool(row.get('success')) for row in eligible)
    planning = sum(bool(row.get('planning_success')) for row in eligible)
    attempts = sum(bool(row.get('physical_attempt')) for row in eligible)
    return {
        'completed': len(rows), 'sam3_excluded': len(excluded), 'eligible': denominator,
        'station_ready': sum(bool(row.get('stationing', {}).get('success')) for row in rows),
        'planning_passes': planning, 'physical_attempts': attempts, 'successes': successes,
        'planning_success_fraction': planning/denominator if denominator else None,
        'grasp_success_fraction': successes/denominator if denominator else None,
        'execution_success_fraction': successes/attempts if attempts else None,
        'sam3_failure_phases': dict(Counter(sam3_failure_phase(row) for row in excluded)),
        'excluded_case_ids': [row['id'] for row in excluded],
        'eligible_case_ids': [row['id'] for row in eligible],
        'eligible_failure_reasons': dict(Counter(row.get('reason', 'unknown')
                                                for row in eligible if not row.get('success'))),
    }


def fraction_text(numerator, denominator):
    if not denominator:
        return f'{numerator}/{denominator}（暂无）'
    return f'{numerator}/{denominator}（{100*numerator/denominator:.1f}%）'
