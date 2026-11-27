"""Transfer complete slots and a proven zero-motion handoff; never repeat a grasp."""
from pathlib import Path
import shutil

from arx_project.layout import ROOT, read_json, sha256

CARRY_FILES = ['plan_result.json', 'execution_planner_initial.json', 'initial_metrics.json',
               'request_ledger.jsonl', 'candidate_order.json', 'candidates/candidate_pool.json']


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
    item = read_json(interrupted[0]); out = previous / 'trials' / item['case_id'] / item['variant']; no_motion(out)
    for name in CARRY_FILES: hashes[str(out / name)] = sha256(out / name)
    for name in ['STOP', 'registration.json', 'services_stopped.json', 'progress.json']:
        hashes[str(previous / name)] = sha256(previous / name)
    hashes[str(out / 'result.json')] = sha256(out / 'result.json')
    return {'run': str(previous.relative_to(ROOT)), 'completed_results': inherited,
        'completed_slot_ids': sorted(complete), 'zero_motion_handoff_slot': item['id'],
        'zero_motion_handoff_directory': str(out.relative_to(ROOT)), 'initial_plan_reused_without_solver_call': True,
        'new_physical_attempts_so_far': 0, 'old_STOP_preserved': True, 'actual_grasps_repeated': 0}
