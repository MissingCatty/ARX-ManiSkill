"""Seed/pure-model audit, retaining original contact/height and clock evaluation."""
from pathlib import Path
import json
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,digest,verify_files
from scripts.curobo_arx.data_factory.recording import physical as original_physical
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
from .runtime import audit_actions
from .advisor34_audit import bound_stop
from .advisor36_contract import interpret


def audit_policy(episode):
    ep=Path(episode);events=read(ep/'interpreter_events.json');ledger=read(ep/'policy_requests.json')
    commands=read(ep/'requests.json')
    if any(r['label'].startswith('axial_align_') for r in commands):raise RuntimeError('Geometric axial fallback in pure pipeline')
    if not ledger:
        if any(e.get('model_inference') for e in events):raise RuntimeError('Model event without request ledger')
        if (ep/'close_execution.json').exists():
            seed=read(ep/'seed_pose_verification.json')
            if not seed['pass']:raise RuntimeError('Seed closure from an unverified target pose')
        return {'pass':True,'model_calls':0,'scope':'direct canonical seed preflight, not a learned policy'}
    from training.nearfield_arx.learned_policy import read_bound_reply
    expected={r['request_id'] for r in ledger}
    present={p.name.removesuffix('_request.json') for p in (ep/'policy').glob('*_request.json')}
    if expected!=present:raise RuntimeError('Policy request/ledger mismatch')
    for item in ledger:
        path=Path(item['request']);req,rep=read_bound_reply(path,path.with_name(item['request_id']+'_reply.json'))
        if set(req['inputs'])!={'eef_world'}:raise RuntimeError('Unexpected policy input; target or budget may leak')
        prompt=read(path.with_name(item['request_id']+'_input.json'))
        if 'budget' in prompt['instruction'] or 'remaining_corrections' in prompt['instruction']:
            raise RuntimeError('Budget leaked into prompt')
        target=interpret(req['inputs']['eef_world'],rep['prediction'])
        if target!=rep['action']:raise RuntimeError('Modified model interpretation')
        for e in (e for e in events if e.get('policy_request_id')==req['request_id']):
            if e['kind']=='grasp_request' and rep['prediction'].strip()!='GRASP':raise RuntimeError('Non-model closure')
            if e['kind']=='target_request' and not np.allclose(e['goal_world'],target['goal_world'],atol=1e-12,rtol=0):
                raise RuntimeError('Model target altered before planning')
    return {'pass':True,'model_calls':len(ledger),'geometry_control_rules':0}


def physical(episode,persist=True):
    ep=Path(episode);audit_actions(ep,read(ep/'requests.json'));audit_policy(ep)
    if persist:write(ep/'strict_physical_receipt.json',original_physical(ep))
    sequence=read(ep/'execution_roles.json')['sequence'];strict=evaluate(ep,sequence=sequence)
    raw=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()] if (ep/'samples.jsonl').exists() else []
    accepted=[];origins=[]
    for path in ep.glob('*_waypoint_stop.json'):
        proof=read(path);bound_stop(proof,raw)
        if proof['pass']:accepted.append(proof['label'])
        origins.append(proof['origin'])
    closure=False
    if (ep/'close_execution.json').exists():
        gate=read(ep/'closure_gate_optimization.json');b=gate['binding'];cert=read(ep/'closure_model_validation.json')
        verify_files(b['files']);bound_stop(b['arrival'],raw)
        closure=bool(gate['pass'] and cert['pass'] and cert['binding_sha256']==digest(b)
            and gate['maximum_joint_drift_rad']<=.0005
            and b['closure_source'] in ('canonical_seed_preflight','explicit_model_GRASP'))
        if read(ep/'close_command.json')['position']!=[b['hold_q'],b['hold_q']]:raise RuntimeError('Closure hold modified')
    exempt=[s for s in strict['executed_stages'] if s not in strict['converged_stages'] and s in accepted
            and (s in ('state_generation','close') or s.startswith('d_refine_'))]
    issues=[i for i in strict.get('issues',[]) if not (isinstance(i,dict) and i.get('stage') in exempt
            and i.get('error')=='Original ARX convergence gate rejected execution')]
    planning=strict.get('planning',{})
    passed=bool(not issues and not strict['missing_stages'] and closure and strict['bilateral_closure']
        and strict['all_recorded_hold_samples_above_6cm'] and not strict['forbidden_contact_events']
        and planning.get('dense_invalid_samples')==0 and planning.get('max_limit_exceed_rad',float('inf'))<=1e-6
        and len(strict['converged_stages'])+len(exempt)==len(sequence) and not strict.get('reported_error'))
    result={**strict,'pass':passed,'score':float(passed),'issues':issues,'strict_original_success':strict['pass'],
        'strict_original_issues':strict['issues'],'independent_closure_pass':closure,
        'certified_stationary_tracking_exemptions':exempt,'stop_origins':origins,
        'external_visual_closure_gate':'removed_by_advisor36','outcome':'success' if passed else 'failed_execution_or_planning'}
    if persist:write(ep/'physical_receipt.json',result)
    return result


def install_quality():
    # Reuse unchanged image/timing/file checks with this explicit-token auditor.
    from . import advisor34_audit
    advisor34_audit.audit_policy=audit_policy
    return advisor34_audit.record_quality
