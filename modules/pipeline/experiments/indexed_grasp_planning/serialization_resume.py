"""Resume only unexecuted planning slots while retaining prior complete files."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, sha256, write_json
from .validation_review import check_validation_receipt


def raw(path): return json.loads(Path(path).read_text())


def schedule(cases, variants, claims, authorized):
    inherited, pending = [], []
    expected = {case['id']+'_'+variant for case in cases for variant in variants}
    if len(expected) != len(cases)*len(variants) or set(claims)-expected:
        raise ValueError('Duplicate cohort rows or unknown prior claims')
    consumed = set()
    for case in cases:
        for variant in variants:
            key = case['id']+'_'+variant;claim = claims.get(key)
            entry = {'case': case['id'], 'variant': variant, 'key': key}
            if claim is None: pending.append(entry)
            elif claim['state'] == 'complete':
                if key in authorized: raise ValueError('A completed request cannot be authorized for replay')
                inherited.append(entry)
            elif claim['state'] == 'claimed' and key in authorized:
                consumed.add(key);pending.append(entry)
            else: raise ValueError('Claimed request has no explicit continuation authorization: '+key)
    if consumed != set(authorized): raise ValueError('Resource-gate authorization does not match prior claims')
    return inherited, pending


def install(root, reg):
    spec = reg['continuation'];source = ensure_run_path(ROOT/spec['source_run'])
    review = ensure_run_path(ROOT/spec['stop_review']);proof = raw(review/'summary.json')
    if not proof['completed_record_integrity'] or not proof['resource_gate_before_planner_verified']:
        raise ValueError('Stopped source records have not been independently checked')
    if not (source/'STOP').exists() or sha256(source/'STOP') != raw(review/'registration.json')['original_STOP_sha256']:
        raise ValueError('Original stop receipt changed')
    state = subprocess.check_output(['systemctl', '--user', 'show',
        raw(source/'controller.json')['unit'], '-p', 'MainPID', '--value'], text=True).strip()
    if state != '0' or not (source/'resident/curobo_indexed_planning_stopped.json').exists():
        raise ValueError('Previous resident planner is not confirmed stopped')
    previous = raw(source/'registration.json')
    if previous['cases'] != reg['cases'] or previous['variants'] != reg['variants']:
        raise ValueError('Continuation changed the original cohort or methods')
    for name, digest in previous['input_sha256'].items():
        if sha256(name) != digest: raise ValueError('Original frozen input changed')
    for name, digest in previous['source_sha256'].items():
        relative = str(Path(name).relative_to(ROOT))
        archived = source/'source_registered'/relative
        if sha256(archived) != digest: raise ValueError('Previous source archive changed')
        if relative.startswith(('modules/curobo/', 'modules/maniskill/', 'arx_project/')) or \
                relative.endswith(('/serialization_worker.py', '/validation_worker.py')):
            if sha256(name) != digest: raise ValueError('Planning algorithm changed across continuation')
    claims = {p.stem: raw(p) for p in (source/'claims').glob('*.json')}
    inherited, pending = schedule(reg['cases'], reg['variants'], claims, spec['authorized_resource_gate_requests'])
    if len(inherited) != proof['completed_requests'] or len(pending) != spec['new_request_budget']:
        raise ValueError('Continuation budget does not match checked completed/unexecuted requests')
    for name in ['STOP', 'registration.json', 'progress.json', 'stop_summary.json']:
        reg['input_sha256'][str(source/name)] = sha256(source/name)
    for name in ['registration.json', 'summary.json', 'endpoints.json']:
        reg['input_sha256'][str(review/name)] = sha256(review/name)
    sources = {}
    for entry in inherited:
        case, variant, key = entry['case'], entry['variant'], entry['key']
        directory = source/'plans'/case/variant;claim = claims[key]
        check_validation_receipt(directory, claim)
        target = root/'plans'/case/variant;target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(directory, target_is_directory=True)
        shutil.copy2(source/'claims'/(key+'.json'), root/'claims'/(key+'.json'))
        sources[key] = str(directory)
        for path in directory.rglob('*'):
            if path.is_file(): reg['input_sha256'][str(path)] = sha256(path)
        reg['input_sha256'][str(source/'claims'/(key+'.json'))] = sha256(source/'claims'/(key+'.json'))
    partial = [case['id'] for case in reg['cases']
        if 0 < sum(case['id']+'_'+v in sources for v in reg['variants']) < len(reg['variants'])]
    if partial != spec['latency_excluded_case_ids']:
        raise ValueError('Cross-restart timing exclusions differ from the registered boundary case')
    spec.update(inherited_complete_requests=len(inherited), inherited_sources=sources,
                new_schedule=pending, completed_planning_slots_repeated=0)
    write_json(root/'inheritance.json', spec)


def initial(root, reg):
    spec = reg.get('continuation')
    if not spec: return [], 0
    pending = {entry['key'] for entry in spec['new_schedule']}
    if pending & set(spec['inherited_sources']):
        raise ValueError('Inherited request also appeared in the new schedule')
    for key, source in spec['inherited_sources'].items():
        relative = Path(source).relative_to(ROOT/spec['source_run']/'plans')
        if (root/'plans'/relative).resolve() != Path(source).resolve():
            raise ValueError('Inherited result link changed')
    rows = raw(ROOT/spec['source_run']/'progress.json')['results']
    return rows, spec['inherited_complete_requests']


def timing_rows(rows, reg):
    excluded = set(reg.get('continuation', {}).get('latency_excluded_case_ids', []))
    return [row for row in rows if row['id'] not in excluded]


def follow(output, run):
    from .serialization_followup import wait_for_result, pointer
    from .serialization_report import report
    from .validation_followup import publish
    if output.exists(): raise ValueError('Do not duplicate the continuation follow-up')
    output.mkdir(parents=True)
    write_json(output/'registration.json', {'at': now(), 'run': str(run.relative_to(ROOT)),
        'new_request_budget': raw(run/'registration.json')['continuation']['new_request_budget'],
        'physical_grasps': 0})
    checked = wait_for_result(run, output/'status.json');report(run)
    pointer(run, 'binding_serialization_full_complete');publish(output)
    write_json(output/'status.json', {'at': now(), 'phase': 'complete',
        'checked_requests': checked['checked_requests'], 'checked_endpoints': checked['checked_endpoints']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True);parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args();output = ensure_run_path(args.output)
    try: follow(output, ensure_run_path(args.run))
    except BaseException:
        write_json(output/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
