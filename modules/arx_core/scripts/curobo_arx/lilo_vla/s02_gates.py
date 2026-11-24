"""S02 scheduling gates. Pure file/record checks, no service or simulation access."""
import json
from pathlib import Path
from interaction_contract import read

VERSION = 'arx_s02_gates_v3'
EVENTS = ('observation', 'policy_input', 'command_execution')
TASK_FAILURES = ('Measured handoff rejected:', 'No grasp-relative handoff passed',
    'No bilateral target contact', 'LiLo policy exhausted', 'ARX IK rejected LiLo action',
    'cuRobo collision checking rejected LiLo action', 'No precise collision-free ARX command',
    'Original convergence check rejected', 'Policy reopened the gripper',
    'Lift planning failed', 'Object hold or trajectory convergence check failed',
    'Grasp approach did not converge', 'Pregrasp did not converge', 'Forbidden contact during')


def report(directory):
    p=Path(directory)/'grasp_report.json'
    try:
        value=read(p)
        return value if isinstance(value,dict) else {'_record_error':'Grasp report is not an object'}
    except (OSError,ValueError) as exc:return {'_record_error':str(exc)}


def ordinary_failure(message):
    return isinstance(message,str) and message.startswith(TASK_FAILURES)


def event_inventory(directory):
    """Union of artifacts, completed-stage reports and unlabeled RPC journal facts.

    Legacy journals contain request IDs but no stage label. Count discrepancies
    therefore remain unattributed evidence loss, never invented label matches.
    """
    d=Path(directory);errors=[];sources={};counts={}
    def add(label,source):
        if not isinstance(label,str) or not label or Path(label).name!=label or label in ('.','..'):
            errors.append('Invalid recorded stage label: '+repr(label));return
        sources.setdefault(label,[]).append(source)
    for kind in ('command','execution','settling'):
        paths=list(d.glob('*_'+kind+'.json'));counts[kind]=len(paths)
        for path in paths:
            add(path.name.removesuffix('_'+kind+'.json'),path.name)
            try:
                if not isinstance(read(path),dict):raise ValueError('Expected JSON object')
            except (OSError,ValueError) as exc:errors.append(path.name+': '+str(exc))
    r=report(d)
    for key in ('stage_timing','stage_convergence'):
        stages=r.get(key,{})
        if not isinstance(stages,dict):errors.append('Malformed grasp_report/'+key);continue
        for label in stages:add(label,'grasp_report/'+key)
    for label in sources:
        for kind in ('command','execution','settling'):
            name=label+'_'+kind+'.json'
            if not (d/name).is_file():errors.append('Missing required event record: '+name)
    journal=d/'requests.jsonl';requests={}
    if journal.exists():
        try:
            for line in journal.read_text().splitlines():
                row=json.loads(line)
                if not isinstance(row,dict):raise ValueError('Journal row is not an object')
                if row.get('action')!='trajectory':continue
                rid=row.get('request_id')
                if not isinstance(rid,str) or not rid:raise ValueError('Missing trajectory request ID')
                record=requests.setdefault(rid,set());event=row.get('event');record.add(event)
                if event=='response' and row.get('ok') is True:record.add('success')
                if row.get('label'):add(row['label'],'requests.jsonl/'+rid)
            if len(requests)>counts['command']:errors.append('Trajectory journal has more request IDs than command files: unmapped command loss')
            succeeded=sum('success' in events for events in requests.values())
            if succeeded>counts['execution']:errors.append('Trajectory journal has more successful replies than execution files: unmapped execution loss')
            for rid,events in requests.items():
                if not {'intent','sent','success'}<=events:errors.append('Unresolved or rejected trajectory request: '+rid)
        except (OSError,ValueError,TypeError) as exc:errors.append('Invalid trajectory journal: '+str(exc))
    # Labeled journals (when available) may add stages not found above.
    for label in sources:
        for kind in ('command','execution','settling'):
            name=label+'_'+kind+'.json'
            if not (d/name).is_file() and 'Missing required event record: '+name not in errors:errors.append('Missing required event record: '+name)
    return dict(errors=errors,stages=sources,journal_trajectory_requests=len(requests),
        post_handoff_event=any(k.startswith('local_') or k in ('close','hold') for k in sources))


def absence(directory, event):
    """N/A requires positive failure evidence and no contradictory later event."""
    d=Path(directory);r=report(d);reason=r.get('error')
    inventory=event_inventory(d)
    if inventory['errors'] or inventory['post_handoff_event']:return False
    if not ordinary_failure(reason):return False
    if event=='policy_input':
        entry=d/'handoff_actual_entry.json'
        denied=(entry.exists() and read(entry).get('pass') is False) or reason.startswith(('No grasp-relative handoff passed','Pregrasp did not converge'))
        later=any(d.glob('policy_*_request.json')) or any(d.glob('policy_*_response.json')) or any(d.glob('local_*_command.json')) or any(d.glob('policy_*_prediction.json'))
        return bool(denied and not later and r.get('local_policy_calls',0)==0 and r.get('local_policy_requests',0)==0)
    if event=='command_execution':
        validation=d/'local_0000_validation.json'
        denied=reason.startswith(('ARX IK rejected LiLo action 0','cuRobo collision checking rejected LiLo action 0')) or (reason.startswith('No precise collision-free ARX command for LiLo action 0') and validation.exists() and read(validation).get('accepted') is False)
        return bool(denied and not any(d.glob('local_*_command.json')) and not any(d.glob('local_*_execution.json')))
    return False


def audit_receipt(errors, events, **extra):
    complete=not errors
    return dict(pass_check=complete,record_integrity='complete' if complete else 'failed',
        event_applicability=events,key_event_coverage={k:complete and events.get(k)=='applicable' for k in EVENTS},
        errors=errors,**extra)


def trial_gate(directory, error, audits, physical, cleanup_ok, setup_ok, runtime_ok):
    r=report(directory)
    inventory=event_inventory(directory)
    infrastructure=bool(error and (error.get('type')!='RuntimeError' or not ordinary_failure(r.get('error'))))
    log=Path(directory)/'pipeline.log'
    if log.exists():
        text=log.read_text(errors='replace').lower()
        infrastructure |= any(x in text for x in ('out of memory','connection refused','no module named'))
    intact=not inventory['errors'] and not r.get('_record_error') and len(audits)==2 and all(a.get('record_integrity')=='complete' for a in audits)
    events={};coverage={}
    for k in EVENTS:
        values=[a.get('event_applicability',{}).get(k,'unknown') for a in audits]
        events[k]='applicable' if 'applicable' in values else 'not_applicable' if all(v=='not_applicable' for v in values) else 'unknown'
        coverage[k]=intact and any(a.get('key_event_coverage',{}).get(k) is True for a in audits)
    halt=not (cleanup_ok and setup_ok and runtime_ok and intact) or infrastructure or bool(physical.get('error'))
    return dict(gate_version=VERSION,task_result='unknown' if infrastructure or physical.get('error') else 'success' if physical.get('pass') else 'failed',
        record_integrity='complete' if intact else 'failed',event_applicability=events,key_event_coverage=coverage,
        event_inventory=inventory,infrastructure_failure=infrastructure,halt=halt,instrumentation_pass=intact and runtime_ok and cleanup_ok)


def pilot_coverage(schedule, results):
    """Both arms and both cases need observed, verified critical events, not four successes."""
    missing=[]
    pilots=[r for r in schedule if r['phase']=='pilot']
    for dimension in ('arm','case_id'):
        for group in sorted({r[dimension] for r in pilots}):
            members=[results[r['trial_id']] for r in pilots if r[dimension]==group]
            for event in EVENTS:
                if not any(r.get('key_event_coverage',{}).get(event) is True for r in members):missing.append(f'{dimension}:{group}/{event}')
    return dict(pass_check=not missing,missing=missing)
