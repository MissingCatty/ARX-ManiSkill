"""Transfer complete slots and a proven zero-motion handoff; never repeat a grasp."""
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json

CARRY_FILES = ['plan_result.json', 'execution_planner_initial.json', 'initial_metrics.json',
               'request_ledger.jsonl', 'candidate_order.json', 'candidates/candidate_pool.json']


def interrupted_before_approach(out):
    """Recognize the recorded handoff fault without calling it a grasp failure."""
    import json
    out = Path(out); result = read_json(out / 'result.json')
    text = result['reason']
    if ('Measured-state planning RPC failed' not in text or 'FileNotFoundError' not in text
            or '/arms/right/plan_result.json' not in text):
        raise ValueError('Unknown interruption needs independent diagnosis')
    rows = [json.loads(line) for line in (out / 'execution.jsonl').read_text().splitlines() if line.strip()]
    if not rows or {row['stage'] for row in rows} != {'open', 'pregrasp'}:
        raise ValueError('The interrupted action progressed beyond the known safe boundary')
    if not (out / 'pregrasp_state.json').exists() or not (out / 'infrastructure_error.json').exists():
        raise ValueError('Missing measured-state or explicit failure receipt')
    if any((out / name).exists() for name in ['closed_state.json', 'approach_plan_result.json', 'lift_plan_result.json']):
        raise ValueError('Unexpected later command in interrupted slot')
    if not read_json(out / 'independent_evaluation.json')['passed']:
        raise ValueError('Interrupted physical record has not been independently evaluated')
    return {'interrupted': True, 'executed_stages': ['open', 'pregrasp'],
        'approach_command_generated': False, 'close_lift_hold_executed': False,
        'reason': 'execution_interface_interruption', 'physical_grasps_repeated': 0}


def no_motion(out):
    out = Path(out)
    trace = out / 'execution.jsonl'
    physics = out / 'physics_trace.jsonl'
    if not trace.exists() or trace.stat().st_size or not physics.exists() or physics.stat().st_size:
        raise ValueError('An interrupted slot with motion must not be retried')
    result = read_json(out / 'result.json')
    if result.get('stages') or any((out / name).exists() for name in ['opened_state.json', 'pregrasp_state.json', 'closed_state.json']):
        raise ValueError('An executed stage must not be repeated')
    if result['reason'] != 'V3 opening must match the registered planned finger locks':
        raise ValueError('Only the identified pre-command opening interface fault is transferable')
    from .contract import check_requests
    if check_requests(out) != ['initial']: raise ValueError('Unexpected interrupted planner stage')
    from .opening import validate_opening
    selected = read_json(out / 'plan_result.json')['selected']
    validate_opening(selected, selected['open_finger_m'])
    arm = selected['arm']
    if arm not in ['left', 'right'] or not (out / 'arms' / arm / 'plan_result.json').is_file():
        raise ValueError('The measured-stage handoff requires the selected arm subplan')
    return True


def transfer(previous, root, schedule, hashes):
    if not (previous / 'STOP').exists() or not read_json(previous / 'services_stopped.json')['all_owned_exited']:
        raise ValueError('The previous stopped comparison must have no live workers')
    old = read_json(previous / 'registration.json')
    if old['schedule'] != schedule: raise ValueError('Continuation schedule differs')
    for name, digest in old['source_sha256'].items():
        if sha256(previous / 'source_registered' / Path(name).relative_to(ROOT)) != digest:
            raise ValueError('Previous execution source archive changed')
    progress = read_json(previous / 'progress.json'); inherited = []
    for row in progress['results']:
        item = next(s for s in schedule if s['case_id'] == row['id'] and s['variant'] == row['variant'])
        claim_path = previous / 'claims' / (item['id'] + '.json'); claim = read_json(claim_path)
        if claim['state'] != 'complete': raise ValueError('A completed result lacks its completed claim')
        source = previous / 'trials' / row['id'] / row['variant']
        for name, digest in claim['files_sha256'].items():
            if sha256(source / name) != digest: raise ValueError('Inherited completed result changed')
            hashes[str(source / name)] = digest
        target = root / 'trials' / row['id'] / row['variant']; target.parent.mkdir(exist_ok=True)
        shutil.copytree(source, target); shutil.copy2(claim_path, root / 'claims' / claim_path.name)
        inherited.append({**row, 'directory': str(target.relative_to(ROOT)), 'inherited_complete': True})
        hashes[str(claim_path)] = sha256(claim_path)
    complete = {row['id'] + '_' + row['variant'] for row in inherited}
    interrupted = [p for p in (previous / 'claims').glob('*.json') if p.stem not in complete]
    if len(interrupted) != 1: raise ValueError('Exactly one known zero-motion interruption is expected')
    item = read_json(interrupted[0]); out = previous / 'trials' / item['case_id'] / item['variant']
    hashes[str(interrupted[0])] = sha256(interrupted[0])
    zero_motion = not (out / 'execution.jsonl').stat().st_size
    terminal_ids = list(complete)
    if zero_motion:
        no_motion(out)
        for name in CARRY_FILES: hashes[str(out / name)] = sha256(out / name)
        for path in (out / 'arms').rglob('*'):
            if path.is_file(): hashes[str(path)] = sha256(path)
    else:
        proof = interrupted_before_approach(out)
        target = root / 'trials' / item['case_id'] / item['variant']; target.parent.mkdir(exist_ok=True)
        shutil.copytree(out, target)
        evidence = {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.mp4']}
        hashes.update({str(out / name): digest for name, digest in evidence.items()})
        write_json(target / 'terminal_interruption.json', {**proof, 'source_directory': str(out.relative_to(ROOT))})
        evidence['terminal_interruption.json'] = sha256(target / 'terminal_interruption.json')
        write_json(root / 'claims' / interrupted[0].name, {**item, 'at': now(), 'state': 'interrupted',
            'files_sha256': evidence, 'source_claim_sha256': sha256(interrupted[0])})
        inherited.append({'id': item['case_id'], 'variant': item['variant'], 'reason': proof['reason'],
            'planning_success': True, 'physical_attempt': True, 'success': False, 'interrupted': True,
            'selected_arm': read_json(out / 'plan_result.json')['selected']['arm'],
            'common_precondition_unexecuted': False, 'directory': str(target.relative_to(ROOT)),
            'timings': {}, 'inherited_interrupted': True})
        terminal_ids.append(item['id'])
    for name in ['STOP', 'registration.json', 'services_stopped.json', 'progress.json']:
        hashes[str(previous / name)] = sha256(previous / name)
    hashes[str(out / 'result.json')] = sha256(out / 'result.json')
    return {'run': str(previous.relative_to(ROOT)), 'completed_results': inherited,
        'completed_slot_ids': sorted(complete), 'terminal_slot_ids': sorted(terminal_ids),
        'zero_motion_handoff_slot': item['id'] if zero_motion else None,
        'zero_motion_handoff_directory': str(out.relative_to(ROOT)) if zero_motion else None,
        'initial_plan_reused_without_solver_call': zero_motion,
        'new_physical_attempts_so_far': 0, 'old_STOP_preserved': True, 'actual_grasps_repeated': 0}
