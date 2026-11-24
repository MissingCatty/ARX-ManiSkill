"""Independent policy/command binding plus unchanged physical hold criteria."""
import json
from pathlib import Path

import numpy as np

from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files, digest
from scripts.curobo_arx.data_factory.recording import physical as original_physical
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate as original_evaluate
from training.nearfield_arx.learned_policy import audit_policy as base_audit, read_bound_reply
from .advisor36_contract import interpret, instruction
from .advisor34_audit import bound_stop
from .runtime import audit_actions


def command_files(episode):
    ep = Path(episode); requests = read(ep / 'requests.json')
    expected = {r['label'] for r in requests}
    commands = {p.name.removesuffix('_command.json') for p in ep.glob('*_command.json')}
    executions = {p.name.removesuffix('_execution.json') for p in ep.glob('*_execution.json')}
    if len(expected) != len(requests) or commands != expected or executions != expected:
        raise RuntimeError('Missing/orphan command or execution receipt; scheduling must stop')
    audit_actions(ep, requests)
    return {'pass': True, 'commands': len(commands), 'executions': len(executions)}


def audit_policy(episode):
    ep = Path(episode); result = base_audit(ep)
    ledger = read(ep / 'policy_requests.json')
    present = {p.name.removesuffix('_request.json') for p in (ep / 'policy').glob('*_request.json')}
    if present != {r['request_id'] for r in ledger} or len(present) != len(ledger):
        raise RuntimeError('Policy request ledger/file mismatch')
    events = read(ep / 'interpreter_events.json')
    if len(ledger) > 3:
        raise RuntimeError('Pure D neural call budget exceeded')
    moves = [e for e in events if e['kind'] == 'target_request']
    closes = [e for e in events if e['kind'] == 'grasp_request']
    if len(moves) > 2 or len(closes) > 1:
        raise RuntimeError('Pure D movement/closure budget exceeded')
    for entry in ledger:
        path = Path(entry['request']); req, rep = read_bound_reply(path, path.with_name(req_name(entry) + '_reply.json'))
        if set(req['inputs']) != {'eef_world', 'stopped'} or req['inputs']['stopped'] is not True:
            raise RuntimeError('Forbidden goal/history input or missing idle state')
        observation = read(req['observation'])
        if not np.allclose(req['inputs']['eef_world'], observation['snapshot']['link_world']['right_eef_link'], atol=1e-12, rtol=0):
            raise RuntimeError('Policy composition frame is not the measured observation')
        prompt = read(path.with_name(req['request_id'] + '_input.json'))
        if prompt['instruction'] != instruction({'near_m': .035, 'far_m': .327}) or prompt['input']:
            raise RuntimeError('Pure model prompt changed')
        images = [str(path.with_name(req['request_id'] + '_' + x + '.png')) for x in ('rgb', 'depth')]
        if prompt['images'] != images or rep['visual_control']['images'] != images or not rep['model_inference']:
            raise RuntimeError('Model did not consume the registered fresh two images')
        try:
            expected = interpret(req['inputs']['eef_world'], rep['prediction'])
        except ValueError:
            expected = {'kind': 'reject'}
        if expected['kind'] != rep['action']['kind']:
            raise RuntimeError('Changed pure-model interpretation')
        linked = [e for e in events if e.get('policy_request_id') == req['request_id']]
        for e in linked:
            if e['kind'] == 'target_request':
                if expected['kind'] != 'move' or not np.allclose(expected['goal_world'], e['goal_world'], atol=1e-12, rtol=0):
                    raise RuntimeError('Executed REFINE differs from raw six-dimensional output')
                if e.get('controller_axial_only') or not e.get('all_six_values_preserved'):
                    raise RuntimeError('Controller synthesized or clipped a coordinate')
            if e['kind'] == 'grasp_request':
                if expected['kind'] != 'close' or rep['prediction'].strip() != 'GRASP' or e['closure_origin'] != 'explicit_model_GRASP':
                    raise RuntimeError('Closure was not explicitly requested by the model')
        dispatch = read(path.with_name(req['request_id'] + '_dispatch_check.json'))
        if not dispatch['pass']:
            raise RuntimeError('Dispatch idle proof failed')
    result.update(model_calls=len(ledger), raw_full6d_audit_pass=True,
                  explicit_GRASP_only=True, G0_guidance_after_H=False)
    return result


def req_name(entry):
    return entry['request_id']


def physical(episode, persist=True):
    ep = Path(episode); command_files(ep); policy = audit_policy(ep)
    if persist:
        write(ep / 'strict_physical_receipt.json', original_physical(ep))
    sequence = read(ep / 'execution_roles.json')['sequence']
    strict = original_evaluate(ep, sequence=sequence)
    raw = [json.loads(line) for line in (ep / 'samples.jsonl').read_text().splitlines()]
    accepted, stops = [], []
    for path in ep.glob('*_waypoint_stop.json'):
        proof = read(path); bound_stop(proof, raw)
        if proof['pass']:
            accepted.append(proof['label'])
        stops.append(proof['origin'])
    closure = False
    if (ep / 'close_execution.json').exists():
        gate = read(ep / 'closure_gate_optimization.json'); binding = gate['binding']
        cert = read(ep / 'closure_model_validation.json')
        verify_files(binding['files']); bound_stop(binding['arrival'], raw)
        closure = bool(gate['pass'] and cert['pass'] and cert['binding_sha256'] == digest(binding)
                       and gate['maximum_joint_drift_rad'] <= .0005
                       and binding['closure_source'] == 'explicit_model_GRASP'
                       and binding['target_geometry_closure_gate_applicable'] is False)
        events = read(ep / 'interpreter_events.json')
        linked = [e for e in events if e['kind'] == 'grasp_request'
                  and e['decision_id'] == binding['grasp_request_id']]
        if len(linked) != 1 or linked[0]['token'] != 'GRASP':
            raise RuntimeError('Physical closure not bound to one raw model GRASP')
        if read(ep / 'close_command.json')['position'] != [binding['hold_q'], binding['hold_q']]:
            raise RuntimeError('Wrong measured joint hold during closure')
    # This is the same previously registered dual-route stationary evaluation;
    # the object lift, bilateral contact and 3-second hold criteria are unchanged.
    exempt = [s for s in strict['executed_stages'] if s not in strict['converged_stages'] and s in accepted
              and (s in ('state_generation', 'close') or s.startswith('d_refine_'))]
    issues = [i for i in strict.get('issues', []) if not (isinstance(i, dict)
        and i.get('stage') in exempt and i.get('error') == 'Original ARX convergence gate rejected execution')]
    planning = strict.get('planning', {})
    passed = bool(not issues and not strict['missing_stages'] and closure and strict['bilateral_closure']
        and strict['all_recorded_hold_samples_above_6cm'] and not strict['forbidden_contact_events']
        and planning.get('dense_invalid_samples') == 0 and planning.get('max_limit_exceed_rad', float('inf')) <= 1e-6
        and len(strict['converged_stages']) + len(exempt) == len(sequence) and not strict.get('reported_error'))
    result = {**strict, 'pass': passed, 'score': float(passed), 'issues': issues,
        'strict_original_success': strict['pass'], 'strict_original_issues': strict['issues'],
        'independent_closure_pass': closure, 'certified_stationary_tracking_exemptions': exempt,
        'stop_origins': stops, 'pure_policy': policy,
        'outcome': 'success' if passed else 'failed_execution_or_planning'}
    if persist:
        write(ep / 'physical_receipt.json', result)
    return result
