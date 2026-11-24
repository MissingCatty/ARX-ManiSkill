"""Independent raw-record evaluation for advisor31; strict scores retained."""
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.recording import physical as original_physical
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate as original_evaluate
from training.nearfield_arx.learned_policy import audit_policy as original_policy_audit, read_bound_reply
from training.nearfield_arx.receipt_numbers import same_pose_diagnostics, same_settling_diagnostics
from training.nearfield_arx import closure_recovery as policy
from scripts.curobo_arx.settling import assess_settling
from . import advisor31_rules as rules
from . import closedloop30_rules as old_rules
from .closedloop30_audit import verify_stops, task_result as old_task_result, evaluate as old_evaluate
from .runtime import audit_actions

def registration(episode):
    roots = [p for p in Path(episode).parents if (p/'registration.json').is_file()]
    if not roots: raise RuntimeError('Episode has no registered campaign')
    return read(roots[0]/'registration.json')


def audit_policy(episode):
    episode = Path(episode); result = original_policy_audit(episode)
    from .advisor31_policy import verify_reply
    reg = registration(episode)
    for request in result['requests']:
        path = episode/'policy'/(request['request_id']+'_request.json')
        req, rep = read_bound_reply(path, path.with_name(request['request_id']+'_reply.json'))
        verify_reply(reg, req, rep)
    events = read(episode/'interpreter_events.json') if (episode/'interpreter_events.json').exists() else []
    contract = read(episode/'closedloop_contract.json') if (episode/'closedloop_contract.json').exists() else None
    continuations = verify_stops(episode, events)
    executed = 0; decisions = {}; closures = []; axes = []
    for event in events:
        kind = event['kind']
        if kind == 'D_axial_entry_preparation':
            if contract is None or contract['method'] != 'D' or event['G0_world'] != contract['G0_world']:
                raise RuntimeError('Unregistered deterministic axial action')
            expected = old_rules.axial_goal(event['snapshot']['link_world']['right_eef_link'], contract['G0_world'])
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
            trigger = rules.interpret(method, token, req['inputs'], executed, rep['advisor31_depth'], rep['contact_control'])
            transform = policy.action_transform(method, token)
            if not same_pose_diagnostics(event['closure_trigger'], trigger) or digest(event['action_transform']) != digest(transform):
                raise RuntimeError('Raw prediction/history differs from interpreter decision')
            decisions[ident] = trigger, transform
            continue
        if ident not in decisions: raise RuntimeError('Missing actual interpreter decision')
        trigger, transform = decisions[ident]
        if kind == 'grasp_request':
            if not trigger['request_closure'] or event['closure_origin'] != trigger['origin']:
                raise RuntimeError('Unregistered closure or blocked motion dispatched')
            if event['model_explicit_GRASP'] != bool(rep['model_inference'] and token == 'GRASP'):
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


def verify_post_close(episode, events):
    episode = Path(episode); path = episode/'close_contact_admission.json'
    occurred = [e for e in events if e['kind'] == 'post_close_contact_admission']
    executed = (episode/'close_execution.json').exists()
    if not executed and not occurred and not path.exists(): return None
    if not executed or not path.exists() or len(occurred) != 1:
        raise RuntimeError('Executed close lacks unique admission evidence')
    proof = read(path); event = occurred[0]
    if event['receipt'] != str(path) or event['receipt_sha256'] != sha(path):
        raise RuntimeError('Close admission evidence identity changed')
    required = {str(episode/('close_'+s+'.json')) for s in ('command', 'execution', 'settling')}
    if set(proof['files']) != required: raise RuntimeError('Close admission omits occurred evidence')
    verify_files(proof['files'])
    command = read(episode/'close_command.json'); execution = read(episode/'close_execution.json')
    if not any(r['id'] == command['id'] and r['label'] == 'close' for r in read(episode/'requests.json')):
        raise RuntimeError('Closed action is not in the command ledger')
    reg = registration(episode)
    closed = {float(c['controllers']['gripper']['closed_m']) for c in reg['render_configs'].values()}
    if len(closed) != 1: raise RuntimeError('Frozen gripper closure calibration is ambiguous')
    expected = rules.post_close(execution, proof['snapshot'], command, closed.pop(), active=False, fault=None)
    if not same_pose_diagnostics({k: proof[k] for k in expected}, expected):
        raise RuntimeError('Post-close admission contradicts measured ARX state')
    if digest(event['snapshot']) != digest(proof['snapshot']) or event['pass'] != proof['pass']:
        raise RuntimeError('Post-close event/state changed')
    settling = read(episode/'close_settling.json')
    gate = read(episode/'closure_model_validation.json')
    if not same_settling_diagnostics(assess_settling(execution, gate['hold_world']), settling):
        raise RuntimeError('Original close tracking diagnosis changed')
    if event['original_arrival_accepted'] != settling['accepted']:
        raise RuntimeError('Original close arrival result concealed')
    return proof


def task_result(strict, sequence, stops, closure_ok, close_admission):
    prior = old_task_result(strict, sequence, stops, closure_ok)
    exempt = list(prior['certified_nonconverged_waypoints'])
    close_ok = close_admission is not None and close_admission['pass']
    if close_ok and 'close' in strict['executed_stages'] and 'close' not in strict['converged_stages']:
        exempt.append('close')
    issues = [issue for issue in strict.get('issues', []) if not (
        isinstance(issue, dict) and issue.get('stage') in exempt and
        issue.get('error') == 'Original ARX convergence gate rejected execution')]
    planning = strict.get('planning', {})
    passed = bool(not issues and not strict.get('missing_stages') and closure_ok and close_ok and
        strict.get('bilateral_closure') and strict.get('all_recorded_hold_samples_above_6cm') and
        not strict.get('forbidden_contact_events') and planning.get('dense_invalid_samples') == 0 and
        planning.get('max_limit_exceed_rad', float('inf')) <= 1e-6 and
        len(strict.get('converged_stages', []))+len(exempt) == len(sequence) and not strict.get('reported_error'))
    return {**prior, 'pass': passed, 'score': float(passed),
        'outcome': 'success' if passed else 'failed_execution_or_planning', 'issues': issues,
        'evaluation_version': rules.VERSION, 'advisor30_task_success': prior['pass'],
        'post_close_contact_admission': close_ok,
        'certified_nonconverged_close': 'close' in exempt,
        'scope': 'Original physical success gates; certified stationary open/closed tracking errors reported separately'}


def evaluate(episode, *, sequence=None):
    episode = Path(episode)
    if sequence is None: sequence = read(episode/'execution_roles.json')['sequence']
    audit_actions(episode, read(episode/'requests.json'))
    previous = old_evaluate(episode, sequence=sequence)
    strict = {**previous, 'pass': previous['strict_original_success'],
              'issues': previous['strict_original_issues']}
    events = read(episode/'interpreter_events.json') if (episode/'interpreter_events.json').exists() else []
    close = verify_post_close(episode, events)
    return task_result(strict, sequence, verify_stops(episode, events), previous['independent_closure_pass'], close)


def physical(episode):
    episode = Path(episode); strict = original_physical(episode)
    write(episode/'strict_physical_receipt.json', strict)
    result = evaluate(episode); write(episode/'physical_receipt.json', result)
    return result


