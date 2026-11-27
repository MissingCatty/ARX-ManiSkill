"""Register a fresh twenty-request pilot using the existing frozen candidates."""
import argparse
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.pipeline.experiments.parallel_planning.prepare import prepare as original_prepare

CONFIG = ROOT / 'configs/experiments/in_memory_state_planning.json'


def prepare(root):
    cfg = read_json(CONFIG)
    if cfg['planning_requests'] != 20 or cfg['variants'] != ['control', 'memory']:
        raise ValueError('The registered state-update pilot is twenty requests')
    original_prepare(root, cfg['fixed_gpu_uuid'], CONFIG)
    reg = read_json(root / 'registration.json')
    previous = cfg.get('previous_completed_pilot')
    if previous:
        from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard
        prior = ROOT / previous
        guard(prior, completed=True)
        accepted = read_json(prior / 'acceptance.json')
        if not read_json(prior / 'services_stopped.json')['all_owned_exited']:
            raise RuntimeError('The previous pilot still owns GPU workers')
        if accepted['planning_requests'] + cfg['planning_requests'] > cfg['maximum_complete_requests_this_stage']:
            raise RuntimeError('The complete-request stage budget is exhausted')
        for name in ['acceptance.json', 'equivalence_review.json', 'publication_snapshot.json', 'report.md']:
            reg['input_sha256'][str(prior / name)] = sha256(prior / name)
        reg['previous_completed_requests'] = accepted['planning_requests']
    stops = []
    for previous in cfg.get('preflight_stop_acknowledgements', []):
        prior = ROOT / previous
        if list((prior / 'claims').glob('*.json')) or not read_json(prior / 'services_stopped.json')['all_owned_exited']:
            raise RuntimeError('Preflight predecessor has claimed requests or active GPU workers')
        for name in ['STOP', 'preflight_stop_summary.json', 'registration.json']:
            reg['input_sha256'][str(prior / name)] = sha256(prior / name)
        stops.append({'run': previous, 'formal_claims': 0, 'STOP_preserved': True,
                      'active_actual_table_policy_restored': True})
    sources = {ROOT / 'configs/modules.json'}
    for folder in ['modules/curobo/state_updates', 'modules/curobo/gripper_sweep',
                   'modules/pipeline/experiments/state_update_planning']:
        sources.update((ROOT / folder).glob('*.py'))
    for path in sources:
        target = root / 'source_registered' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        reg['source_sha256'][str(path)] = sha256(path)
    reg.update(experiment='V8_in_memory_state_updates',
               authorization=cfg['authorization'], candidate_batching_enabled=False,
               source_and_inputs_frozen=True, preflight_stop_acknowledgements=stops)
    write_json(root / 'registration.json', reg)
    write_json(ROOT / 'tmp/state_updates/active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'phase': 'registered', 'complete': False})
    return reg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    prepare(ensure_run_path(p.parse_args().root))
