"""Freeze the full cohort without using previous success to select cases."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json, ensure_run_path
from .contract import schedule

CONFIG = ROOT / 'configs/experiments/in_memory_state_execution.json'


def guard(root, full=True):
    if (root / 'STOP').exists(): raise RuntimeError('Preserve the stopped complete comparison')
    reg = read_json(root / 'registration.json')
    values = {**reg['source_sha256'], **reg['input_sha256']} if full else reg['source_sha256']
    for path, digest in values.items():
        if sha256(path) != digest: raise ValueError('Frozen source/input changed: ' + path)
    return reg


def prepare(root, predecessor=None, prestart_stop=None):
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    from modules.pipeline.experiments.sam3_zerograsp_comparison.cohort import check_pair
    cfg = read_json(CONFIG); previous = ROOT / cfg['planning_acceptance']
    if root.exists(): raise ValueError('Create a new registration; do not reuse old slots')
    accepted = read_json(previous / 'adoption.json')
    if not accepted['optional_implementation_accepted'] or not read_json(previous / 'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Accepted planning implementation is not complete and stopped')
    old = read_json(previous / 'registration.json')
    metadata_revisions = []
    for path, digest in old['source_sha256'].items():
        archived = previous / 'source_registered' / Path(path).relative_to(ROOT)
        if sha256(archived) != digest:
            raise ValueError('Accepted state-update source changed: ' + path)
        if sha256(path) != digest:
            if Path(path) != ROOT / 'configs/experiments/active_experiment.json':
                raise ValueError('Accepted runtime source changed: ' + path)
            current_config, old_config = read_json(path), read_json(archived)
            normalized = deepcopy(current_config)
            normalized.pop('memory_state_optional_registration', None)
            key = str(ROOT / 'configs/modules.json')
            normalized['configuration_hashes'][key] = old_config['configuration_hashes'][key]
            if normalized != old_config or current_config['configuration_hashes'][key] != sha256(key):
                raise ValueError('Configuration revision changed more than optional-module registration')
            metadata_revisions.append({'path': path, 'previous_sha256': digest, 'sha256': sha256(path),
                'only_change': 'register the already-accepted optional memory module; active planner unchanged'})
    source = ROOT / cfg['cohort_source']; origin = read_json(source / 'registration.json')
    cohort = origin['cohort']['pairs']
    if len(cohort) != 74 or sum(bool(c.get('common_precondition_failure')) for c in cohort) != 5:
        raise ValueError('The frozen complete cohort changed')
    current = load_current()
    root.mkdir(parents=True)
    for name in ['inputs', 'claims', 'trials', 'preflight', 'warmup', 'source_registered']:
        (root / name).mkdir()
    hashes = {}; cases = []
    def copy(path, target, expected=None):
        digest = sha256(path)
        if expected and digest != expected: raise ValueError('Frozen input mismatch: ' + str(path))
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        hashes[str(path)] = digest; hashes[str(target)] = sha256(target)
    for pair in cohort:
        observation = check_pair(pair); frozen = root / 'inputs' / pair['id']
        for name, digest in pair['input_hashes'].items(): copy(observation / name, frozen / name, digest)
        if not pair.get('common_precondition_failure'):
            copy(ROOT / cfg['candidate_source'] / pair['id'] / 'fine.json', frozen / 'candidates/candidate_pool.json')
            directory = ROOT / cfg['cohort_source'] / 'trials' / pair['id']
            if sha256(directory / 'state.json') != pair['input_hashes']['state.json']:
                raise ValueError('Candidates and frozen handoff differ')
        cases.append({**pair, 'input_directory': str(frozen.relative_to(ROOT))})
    for name in ['object_catalog.json', 'gripper_descriptor.json', 'common_initial_conditions.json']:
        copy(source / name, root / name)
    for name in ['state.json', 'closed_state.json']: copy(previous / 'warmup' / name, root / 'warmup' / name)
    for name in ['registration.json', 'adoption.json', 'equivalence_review.json', 'trajectory_review.json', 'services_stopped.json']:
        hashes[str(previous / name)] = sha256(previous / name)
    sources = {Path(p) for p in old['source_sha256']}
    sources.update(Path(p) for p in current['configuration_hashes'])
    for folder in ['arx_project', 'modules/pipeline/experiments/state_update_execution',
                   'modules/pipeline/experiments/repaired_grasp_benchmark',
                   'modules/pipeline/experiments/diverse_grasp_benchmark',
                   'modules/pipeline/experiments/diverse_object_scenes',
                   'modules/maniskill/arx_pipeline', 'modules/maniskill/planning_common']:
        sources.update((ROOT / folder).glob('*.py'))
    sources.update([CONFIG, ROOT / 'modules/maniskill/short_lift_protocol.py',
        ROOT / 'modules/pipeline/experiments/sam3_zerograsp_comparison/replay.py',
        ROOT / 'modules/pipeline/experiments/indexed_grasp_planning/physical/simulation.py',
        ROOT / 'modules/pipeline/experiments/indexed_grasp_planning/physical/contract.py',
        ROOT / 'modules/pipeline/experiments/indexed_grasp_planning/physical/controller.py'])
    for path in sorted(sources):
        target = root / 'source_registered' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    plan = schedule(cases); continuation = None
    if predecessor:
        from .continuation import transfer
        continuation = transfer(predecessor, root, plan, hashes)
    prestart_ack = None
    if prestart_stop:
        stopped = read_json(prestart_stop / 'registration.json')
        inherited = set(stopped['continuation']['completed_slot_ids'])
        if ({p.stem for p in (prestart_stop / 'claims').glob('*.json')} != inherited
                or read_json(prestart_stop / 'STOP')['new_formal_claims'] != 0
                or not read_json(prestart_stop / 'services_stopped.json')['all_owned_exited']):
            raise ValueError('Prestart predecessor must have zero new formal slots or motion')
        for name in ['STOP', 'registration.json', 'services_stopped.json', 'progress.json']:
            hashes[str(prestart_stop / name)] = sha256(prestart_stop / name)
        prestart_ack = {'run': str(prestart_stop.relative_to(ROOT)), 'new_formal_claims': 0,
            'new_physical_grasps': 0, 'STOP_preserved': True}
    reg = {'at': now(), 'config': cfg, 'cases': cases, 'schedule': plan,
        'configuration_id': current['profile_id'], 'gpu': cfg['fixed_gpu_uuid'],
        'physical_slot_budget': 138, 'registered_slots': 148, 'automatic_retries': False,
        'source_sha256': {str(p): sha256(p) for p in sorted(sources)}, 'input_sha256': hashes,
        'optional_registration_revisions': metadata_revisions, 'continuation': continuation,
        'prestart_stop_acknowledgement': prestart_ack}
    write_json(root / 'registration.json', reg)
    rows = continuation['completed_results'] if continuation else []
    write_json(root / 'progress.json', {'at': now(), 'phase': 'registered', 'completed_requests': len(rows),
        'total_requests': 148, 'results': rows})
    write_json(ROOT / 'tmp/state_updates/execution_active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'phase': 'registered', 'complete': False})
    return reg


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--predecessor', type=Path); p.add_argument('--prestart-stop', type=Path); args = p.parse_args()
    prepare(ensure_run_path(args.root), ensure_run_path(args.predecessor) if args.predecessor else None,
        ensure_run_path(args.prestart_stop) if args.prestart_stop else None)
