"""Additional read-only checks for H012 actions; the core evaluator stays fixed."""
import json
from pathlib import Path

import numpy as np

from settling import assess_settling


def audit(folder):
    folder = Path(folder)
    read = lambda name: json.loads((folder / name).read_text())
    snapshot = read('capture_snapshot.json')
    names = snapshot['right_arm']
    limits = np.asarray([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names])
    target = snapshot['objects'][read('scenario.json').get('target', 'test_object')]['path']
    failures = []; rows = []
    paths = sorted(folder.glob('feedback_history*_approach_execution.json'))
    for path in paths:
        label = path.name.removesuffix('_execution.json')
        issues = []
        try:
            command = read(label + '_command.json')
            execution = read(path.name)
            optimization = read(label + '_optimization.json')
            goal = read(label + '_goal.json')
            positions = np.asarray(command['position'])
            times = np.asarray(command['time_from_start'])
            violation = float(np.maximum(limits[:, 0] - positions, positions - limits[:, 1]).clip(0).max())
            if command['joint_names'] != names or positions.shape != (len(times), len(names)):
                issues.append('Joint ordering or shape mismatch')
            if not np.isfinite(positions).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
                issues.append('Nonfinite or nonmonotonic command')
            if command['gripper_target'] != -3.2:
                issues.append('Extra alignment was not commanded with an open hand')
            if violation > 1e-6:
                issues.append('Command exceeds frozen joint limits')
            if not optimization['success'] or optimization['max_limit_exceed_rad'] > 1e-6 or optimization['invalid_samples']:
                issues.append('Optimizer or dense-check certificate rejected')
            convergence = assess_settling(execution, goal)
            if not convergence['accepted']:
                issues.append('Extra approach did not satisfy original convergence criteria')
            unwanted = [c for c in execution['contacts'] if c['num_contacts'] > 0
                        and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])
                        and not (target in c['a'] + c['b'] and any(n in c['a'] + c['b'] for n in ('right_link27', 'right_link28')))]
            if unwanted:
                issues.append('Unwanted physical contact in extra approach')
            rows.append({'stage': label, 'pass': not issues, 'issues': issues,
                         'planned_max_limit_exceed_rad': violation,
                         'convergence': convergence, 'unwanted_contact_events': len(unwanted)})
        except Exception as exc:
            issues.append(repr(exc)); rows.append({'stage': label, 'pass': False, 'issues': issues})
        failures.extend(label + ': ' + issue for issue in issues)
    feedback = read('tracking_feedback.json') if (folder / 'tracking_feedback.json').exists() else {}
    if feedback.get('applied') and (folder / 'close_execution.json').exists():
        alignments = feedback.get('alignments', [])
        if not alignments or not alignments[-1]['measured_closed_check']['pass'] or not alignments[-1]['held_command_valid']:
            failures.append('Closure occurred without an accepted measured alignment')
        elif not np.array_equal(np.asarray(read('close_command.json')['position']),
                                np.repeat([alignments[-1]['held_command_q']], 2, axis=0)):
            failures.append('Closure did not hold the last accepted command target constant')
        if len(paths) != len(alignments):
            failures.append('Extra approach archive count differs from completed alignments')
    result = {'pass': not failures, 'extra_completed_approaches': len(paths), 'stages': rows,
              'violations': failures, 'scope': 'Additional checks only; original five-stage evaluator and thresholds unchanged'}
    (folder / 'extra_action_evaluation.json').write_text(json.dumps(result, indent=2))
    return result
