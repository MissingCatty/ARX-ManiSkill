"""Recompute advisor30 events and task success from unchanged raw ARX records."""
from pathlib import Path
import copy
import math
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.recording import physical as original_physical
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate as original_evaluate
from training.nearfield_arx.learned_policy import audit_policy as original_policy_audit, read_bound_reply
from training.nearfield_arx.receipt_numbers import same_pose_diagnostics
from training.nearfield_arx import closure_recovery as policy
from . import closedloop30_rules as rules
from .runtime import audit_actions


def same_trigger(actual, expected):
    # Only derived angle diagnostics have platform-dependent final bits.
    # Decisions, distances, action vectors and all identities remain exact.
    left, right = copy.deepcopy(actual), copy.deepcopy(expected)
    pairs = [(left.get('current_to_fixed_prior'), right.get('current_to_fixed_prior'))]
    if left.get('anti_regression') is not None and right.get('anti_regression') is not None:
        pairs += [(left['anti_regression'].get(k), right['anti_regression'].get(k))
                  for k in ('current_to_G0', 'predicted_to_G0')]
    for a, b in pairs:
        if a is None or b is None: continue
        if not math.isfinite(a['rotation_rad']) or not math.isfinite(b['rotation_rad']): return False
        if abs(a['rotation_rad']-b['rotation_rad']) > 1e-7: return False
        a['rotation_rad'] = b['rotation_rad']
    return digest(left) == digest(right)


def verify_stops(episode, events):
    episode = Path(episode)
    files = {p.name.removesuffix('_waypoint_stop.json'): p for p in episode.glob('*_waypoint_stop.json')}
    occurred = [e for e in events if e['kind'] == 'CD_waypoint_stop']
    labels = [e['command_label'] for e in occurred]
    if len(labels) != len(set(labels)) or set(labels) != set(files):
        raise RuntimeError('Missing, orphan or repeated measured-stop evidence')
    accepted = []
    requests = read(episode/'requests.json')
    for event in occurred:
        label = event['command_label']; path = files[label]; proof = read(path)
        if str(path) != event['receipt'] or sha(path) != event['receipt_sha256']:
            raise RuntimeError('Measured-stop receipt changed')
        required = {str(episode/(label+s+'.json')) for s in ('_command', '_execution', '_settling')}
        if set(proof['files']) != required: raise RuntimeError('Measured-stop proof omits occurred-action evidence')
        verify_files(proof['files'])
        command = read(episode/(label+'_command.json')); execution = read(episode/(label+'_execution.json'))
        if not any(r['id'] == command['id'] and r['label'] == label for r in requests):
            raise RuntimeError('Measured stop has no actual dispatched command')
        if proof['command_id'] != command['id'] or digest(event['snapshot']) != digest(proof['snapshot']):
            raise RuntimeError('Measured-stop command/state binding differs')
        expected = rules.stationary(execution, proof['snapshot'], active=False, fault=None, label=label)
        if not same_pose_diagnostics({k: proof[k] for k in expected}, expected):
            raise RuntimeError('Measured-stop proof does not match original ARX states')
        settling = read(episode/(label+'_settling.json'))
        if digest(proof['original_settling']) != digest(settling) or event['original_arrival_accepted'] != settling['accepted']:
            raise RuntimeError('Original tracking failure concealed')
        from scripts.curobo_arx.settling import assess_settling
        from training.nearfield_arx.receipt_numbers import same_settling_diagnostics
        if not same_settling_diagnostics(assess_settling(execution, proof['desired_world']), settling):
            raise RuntimeError('Original convergence diagnosis differs')
        targets = [e for e in events if e.get('command_label') == label and
                   e['kind'] in ('target_request', 'D_axial_entry_preparation')]
        if len(targets) != 1:
            raise RuntimeError('Waypoint lacks its unique commanded target')
        np.testing.assert_allclose(targets[0]['goal_world'], proof['desired_world'], atol=1e-12, rtol=0)
        if event['pass'] != proof['pass']: raise RuntimeError('Measured-stop event differs from receipt')
        if proof['pass']: accepted.append(label)
    return accepted


def audit_policy(episode):
    episode = Path(episode); result = original_policy_audit(episode)
    events = read(episode/'interpreter_events.json') if (episode/'interpreter_events.json').exists() else []
    contract = read(episode/'closedloop_contract.json') if (episode/'closedloop_contract.json').exists() else None
    continuations = verify_stops(episode, events)
    executed = 0; decisions = {}; closures = []; axes = []
    for event in events:
        kind = event['kind']
        if kind == 'D_axial_entry_preparation':
            if contract is None or contract['method'] != 'D' or event['G0_world'] != contract['G0_world']:
                raise RuntimeError('Unregistered deterministic axial action')
            expected = rules.axial_goal(event['snapshot']['link_world']['right_eef_link'], contract['G0_world'])
            if not same_pose_diagnostics({k: event[k] for k in expected}, expected):
                raise RuntimeError('Axial action differs from measured state and fixed prior')
            axes.append(event)
            continue
        if kind not in ('policy_interpreter_decision', 'target_request', 'grasp_request'): continue
        ident = event.get('policy_request_id')
        if not ident: raise RuntimeError('Policy event lacks original model request')
        path = episode/'policy'/(ident+'_request.json')
        req, rep = read_bound_reply(path, path.with_name(ident+'_reply.json'))
        method, token = req['method'], rep['prediction'].strip()
        if contract is None or req['inputs']['G0_world'] != contract['G0_world']:
            raise RuntimeError('Policy prior changed')
        if kind == 'policy_interpreter_decision':
            if ident in decisions: raise RuntimeError('Repeated interpreter decision')
            trigger = rules.interpret(method, token, req['inputs'], executed)
            transform = policy.action_transform(method, token)
            if not same_trigger(event['closure_trigger'], trigger) or digest(event['action_transform']) != digest(transform):
                raise RuntimeError('Raw prediction/history differs from interpreter decision')
            decisions[ident] = trigger, transform
            continue
        if ident not in decisions: raise RuntimeError('Missing actual interpreter decision')
        trigger, transform = decisions[ident]
        if kind == 'grasp_request':
            if not trigger['request_closure'] or event['closure_origin'] != trigger['origin']:
                raise RuntimeError('Unregistered closure or blocked motion dispatched')
            if event['model_explicit_GRASP'] != (token == 'GRASP'):
                raise RuntimeError('Executor closure incorrectly attributed to model')
            closures.append(trigger['origin'])
        else:
            if trigger['request_closure'] or trigger['motion_budget_exhausted']:
                raise RuntimeError('Motion after terminal decision')
            goal = policy.decode(method, req['inputs']['eef_world'], token, req['inputs']['G0_world'])
            np.testing.assert_allclose(event['goal_world'], goal, atol=1e-12, rtol=0)
            if digest(event['action_transform']) != digest(transform): raise RuntimeError('Command transform differs')
            if event['command_label'] in continuations: executed += 1
    if len(axes) > 1 or len(closures) > 1: raise RuntimeError('Repeated axial action or closure')
    if any(r['label'] == 'd_entry_axis' for r in read(episode/'requests.json')) and len(axes) != 1:
        raise RuntimeError('Actual axial action lacks registered provenance')
    result.update(closure_origins=closures, interpreter_decisions=len(decisions), axial_preparations=len(axes),
                  stationary_waypoints=continuations, visual_motion_count=executed)
    return result


def task_result(strict, sequence, stops, closure_ok):
    exempt = [label for label in strict.get('executed_stages', [])
              if label not in strict.get('converged_stages', []) and rules.waypoint(label) and label in stops]
    issues = [issue for issue in strict.get('issues', []) if not (
        isinstance(issue, dict) and issue.get('stage') in exempt and
        issue.get('error') == 'Original ARX convergence gate rejected execution')]
    planning = strict.get('planning', {})
    passed = bool(not issues and not strict.get('missing_stages') and closure_ok and
        strict.get('bilateral_closure') and strict.get('all_recorded_hold_samples_above_6cm') and
        not strict.get('forbidden_contact_events') and planning.get('dense_invalid_samples') == 0 and
        planning.get('max_limit_exceed_rad', float('inf')) <= 1e-6 and
        len(strict.get('converged_stages', []))+len(exempt) == len(sequence) and not strict.get('reported_error'))
    return {**strict, 'pass': passed, 'score': float(passed),
            'outcome': 'success' if passed else 'failed_execution_or_planning', 'issues': issues,
            'evaluation_version': rules.VERSION, 'strict_original_success': strict['pass'],
            'strict_original_issues': strict.get('issues', []), 'certified_nonconverged_waypoints': exempt,
            'independent_closure_pass': closure_ok,
            'scope': 'Original contact/height/hold/planning gates; certified open waypoint tracking errors reported separately'}


def evaluate(episode, *, sequence=None):
    episode = Path(episode)
    if sequence is None:
        sequence = read(episode/'execution_roles.json')['sequence']
    audit_actions(episode, read(episode/'requests.json'))
    strict = original_evaluate(episode, sequence=sequence)
    events = read(episode/'interpreter_events.json') if (episode/'interpreter_events.json').exists() else []
    stops = verify_stops(episode, events)
    closure_ok = False
    if 'close' in strict.get('executed_stages', []):
        gate = read(episode/'closure_gate_optimization.json'); binding = gate['binding']
        command = read(episode/(binding['previous_command_label']+'_command.json'))
        closure_ok = rules.validate_closure(binding, read(episode/'closure_model_validation.json'), command)
        fresh = gate['fresh_dispatch_check']
        if not gate['pass'] or not fresh['pass'] or gate['maximum_joint_drift_rad'] > .0005:
            raise RuntimeError('Independent closure dispatch certificate failed')
        close = read(episode/'close_command.json')
        if close['position'] != [binding['hold_q'], binding['hold_q']]:
            raise RuntimeError('Actual close held a different arm configuration')
    return task_result(strict, sequence, stops, closure_ok)


def physical(episode):
    episode = Path(episode)
    strict = original_physical(episode)
    write(episode/'strict_physical_receipt.json', strict)
    result = evaluate(episode)
    write(episode/'physical_receipt.json', result)
    return result
