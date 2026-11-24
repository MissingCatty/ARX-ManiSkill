"""Adopt running advisor38 children and consume registered supplemental pools."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.launch import devices
from .advisor36_progress import collect
from .advisor36_orchestrator import launch
from .advisor38_orchestrator import combined as previous_combined, child_state, select_candidates as initial_select
from .advisor38_prefilter import require_seven
from .advisor38_seed import prepare as prepare_seed
from .advisor38_static_launch import prepare as prepare_static


def combined(top):
    oldtop = Path(read(top / 'scope_registration.json')['source_campaign'])
    old, fresh = previous_combined(oldtop), collect(top)
    if set(old['objects']) & set(fresh['objects']):
        raise RuntimeError('Previously released object physically repeated')
    return {'objects': {**old['objects'], **fresh['objects']},
            'candidate_rounds': old['candidate_rounds'] + fresh['candidate_rounds'],
            'released_objects': old['released_objects'] + fresh['released_objects'],
            'executed': old['executed'] + fresh['executed'],
            'successful_seed_tests': old['successful_seed_tests'] + fresh['successful_seed_tests']}


def selection(top, data, spec, initial, extra, initial_done, extra_progress, extra_done):
    oldtop = Path(read(top / 'scope_registration.json')['source_campaign'])
    proposals, exhausted, blocked = initial_select(oldtop, data, spec, initial, False)
    selected = {r['object_id'] for r in proposals}
    for obj, item in spec['objects'].items():
        if obj in selected or obj in blocked:
            continue
        history = [r for r in data['candidate_rounds'] if r['object_id'] == obj]
        if any(r['processed'] != 7 for r in history):
            raise RuntimeError('Unresolved physical candidate')
        used = {digest(r['geometry']['selected']) for r in history}
        choices = [r for r in extra if r['object_id'] == obj and r['pass'] and r['candidate_sha256'] not in used]
        if choices:
            choice = choices[0]
            if sha(choice['candidate_file']) != choice['candidate_file_sha256']:
                raise RuntimeError('Supplemental candidate changed')
            candidate, proof = read(choice['candidate_file']), read(choice['result_file'])
            require_seven(proof, candidate)
            if proof['object_id'] != obj or proof['candidate_sha256'] != choice['candidate_sha256']:
                raise RuntimeError('Supplemental object/candidate binding mismatch')
            verify_files(proof['evidence_files'])
            row = {k: copy.deepcopy(v) for k, v in read(item['geometry']).items() if k != 'candidates'}
            row.update(selected=candidate, model_prefilter_file=choice['result_file'],
                refinement_geometry_file=choice['candidate_file'], prior_registered_candidate_count=len(history),
                original_cumulative_candidate_cap=15, supplemental_authority=str(top / 'advisor39.md'))
            proposals.append(row)
        elif initial_done:
            had_initial = any(r['object_id'] == obj and r['pass'] for r in initial)
            if had_initial or extra_done or obj in extra_progress.get('exhausted_objects', []):
                exhausted.append(obj)
    return proposals, list(dict.fromkeys(exhausted)), blocked


def alive(unit):
    p = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True, timeout=15)
    return p.stdout.strip() in ('active', 'activating')


def run(top):
    top = Path(top).resolve()
    reg = read(top / 'data_supervisor_registration.json')
    oldtop = Path(reg['predecessor'])
    locks = [(root / 'data_supervisor.lock').open('a+') for root in (oldtop, top)]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (top / 'data_supervisor_STARTED.json').exists():
        raise RuntimeError('No automatic supervisor restart')
    transfer = read(top / 'handoff_receipt.json')
    if not transfer['old_scheduler_absent'] or alive(transfer['old_unit']):
        raise RuntimeError('Two schedulers could dispatch the same physical queue')
    write(top / 'data_supervisor_STARTED.json', {'wall': time.time(), 'registration_sha256': sha(top / 'data_supervisor_registration.json')})
    current = Path(transfer['current_seed_campaign'])
    spec = read(top / 'prefilter_registration.json')
    fingerprint, data = None, None
    try:
        verify_files({**reg['fixed_files'], **reg['source_files']})
        while True:
            if (top / 'STOP').exists() or (oldtop / 'STOP').exists() or time.time() >= reg['deadline_unix']:
                raise RuntimeError('Campaign STOP or original deadline')
            verify_files(reg['source_files'])
            for file, expected in reg['boundary_files'].items():
                if sha(file) != expected:
                    raise RuntimeError('Registered campaign boundary changed')
            pf = read(oldtop / 'prefilter/state.json')
            if pf['status'] == 'fault' or (oldtop / 'prefilter/FAULT.json').exists():
                raise RuntimeError('Initial model prefilter fault')
            if pf['status'] != 'complete' and not alive(read(oldtop / 'prefilter_service.json')['unit']):
                raise RuntimeError('Initial model worker disappeared')
            extra_state_file = top / 'supplement/state.json'
            xs = read(extra_state_file) if extra_state_file.exists() else {'status': 'starting'}
            if xs['status'] == 'fault' or (top / 'supplement/FAULT.json').exists():
                raise RuntimeError('Supplemental model worker fault')
            if xs['status'] != 'complete' and not alive(read(top / 'supplement_service.json')['unit']):
                raise RuntimeError('Supplemental model worker disappeared')
            seed_state = child_state(current)
            files = [p for base in (oldtop, top) for pattern in ('seed_preflight*/registration.json', 'seed_preflight*/groups/*/results.json') for p in base.glob(pattern)]
            signature = [(str(p), p.stat().st_mtime_ns) for p in sorted(files)]
            if signature != fingerprint:
                data, fingerprint = combined(top), signature
            counts = {obj: sum(r['object_id'] == obj for r in data['candidate_rounds']) for obj in spec['objects']}
            control = {'released_objects': list(data['objects']), 'registered_candidates': counts,
                       'used_candidate_sha256': [digest(r['geometry']['selected']) for r in data['candidate_rounds']]}
            write(top / 'supplement_control.json', control)
            write(oldtop / 'prefilter_control.json', {'released_or_budget_exhausted': list(data['objects']) + [o for o, n in counts.items() if n >= 15]})
            lineage = read(top / 'static_lineage.json')
            active_static, datasets = [], []
            for obj, entry in lineage.items():
                root = Path(entry['current'])
                if sha(root / 'registration.json') != entry['registration_sha256']:
                    raise RuntimeError('Static lineage registration changed')
                state = child_state(root)
                if state['status'] == 'complete':
                    pairs = [r for r in read(root / 'dataset/pair_checks.json') if r['pass']]
                    errors = [float(np.max(np.abs(r['actual_pair_translation_residual_m']))) * 1000 for r in pairs]
                    datasets.append({'object_id': obj, 'campaign': str(root), **read(root / 'dataset/quota_status.json'),
                         'actual_pair_maximum_component_residual_mm': max(errors) if errors else None,
                         'pairs_above_advisor_reference_0_0015mm': sum(x >= .0015 for x in errors),
                         'original_admission_unchanged': True})
                else:
                    active_static.append(root)
            if len(active_static) > 1:
                raise RuntimeError('Multiple static World owners')
            inventory = {r['uuid']: r for r in devices()}

            def ready(roles, floor):
                a, b = inventory[roles['compute_uuid']], inventory[roles['sim_uuid']]
                return (a['index'], b['index']) if a['free_mb'] >= 6500 and b['free_mb'] >= floor else None

            if not active_static:
                pending = [o for o in data['objects'] if o not in lineage]
                resource = ready(reg['static_devices'], 12000)
                if pending and resource:
                    obj = pending[0]
                    source = Path(data['objects'][obj]['seed_campaign']).parent
                    root = Path(prepare_static(top, source, obj, *resource))
                    launch(root, 'scripts.curobo_arx.nearfield.advisor38_static_launch', 'arx-a39-static-' + obj.replace('_', '-') + '-01')
                    active_static.append(root)
            initial = read(oldtop / 'prefilter/results.json')['records']
            extra = read(top / 'supplement/results.json')['records'] if (top / 'supplement/results.json').exists() else []
            progress = read(top / 'supplement/progress.json') if (top / 'supplement/progress.json').exists() else {}
            exhausted = []
            if seed_state['status'] == 'complete':
                proposals, exhausted, _ = selection(top, data, spec, initial, extra, pf['status'] == 'complete', progress, xs['status'] == 'complete')
                resource = ready(reg['seed_devices'], 9000)
                if proposals and resource:
                    index = len(list(top.glob('seed_preflight*/registration.json'))) + 1
                    file = top / 'candidate_reselection' / f'{index:02d}' / 'canonical_candidates.json'
                    if file.exists():
                        raise RuntimeError('Prepared selection already exists; no automatic replay')
                    write(file, proposals)
                    write(top / 'data_supervisor_state.json', {'status': 'preparing_screened_physical_candidates',
                          'objects': [r['object_id'] for r in proposals], 'training_started': False})
                    root = Path(prepare_seed(top, *resource, 'coupled_xyz', name=f'seed_preflight_{index:02d}',
                          candidate_file=file, objects=[r['object_id'] for r in proposals], previous_campaign=current))
                    launch(root, 'scripts.curobo_arx.nearfield.advisor38_seed', f'arx-a39-seed-{index:02d}')
                    current, seed_state = root, {'status': 'launch_requested'}
            state = {'status': 'running_data_only', 'wall': time.time(), 'training_started': False,
                'main_training_dataset': 'Version_A', 'training_requires_later_user_confirmation': True,
                'current_seed_campaign': str(current), 'seed_status': seed_state['status'],
                'released_objects': data['released_objects'], 'executed_seed_tests': data['executed'],
                'successful_seed_tests': data['successful_seed_tests'], 'registered_candidates': counts,
                'active_static_campaign': str(active_static[0]) if active_static else None,
                'completed_object_datasets': len(datasets), 'datasets': datasets,
                'admitted_observations_completed_objects': sum(d['admitted_observations'] for d in datasets),
                'model_prefilter_completed': len(initial), 'model_prefilter_qualified': sum(r['pass'] for r in initial),
                'supplemental_progress': progress, 'supplemental_state': xs,
                'canonical_search_exhausted_objects': exhausted}
            write(top / 'data_supervisor_state.json', state)
            write(top / 'current_canonical_releases.json', data)
            if seed_state['status'] == 'complete' and not active_static and len(datasets) == data['released_objects'] and (
                    data['released_objects'] == 8 or len(exhausted) == 8 - data['released_objects']):
                complete = data['released_objects'] == 8 and all(d['minimum_met'] for d in datasets)
                write(top / 'data_completion_receipt.json', {**state, 'pass': complete, 'finished_unix': time.time(),
                      'advisor_reference_symmetry_all_met': all(d['pairs_above_advisor_reference_0_0015mm'] == 0 for d in datasets),
                      'status': 'data_ready_waiting_training_confirmation' if complete else 'bounded_data_stage_incomplete'})
                return
            time.sleep(20)
    except BaseException:
        error = traceback.format_exc()
        write(top / 'data_supervisor_FAULT.json', {'error': error, 'wall': time.time(),
              'already_running_bounded_children_retained': True, 'training_started': False})
        write(top / 'data_supervisor_state.json', {'status': 'scheduling_stopped', 'error': error, 'training_started': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root')
    run(parser.parse_args().root)
