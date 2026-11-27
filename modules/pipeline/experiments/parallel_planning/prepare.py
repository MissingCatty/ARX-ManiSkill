"""Freeze one fresh thirty-slot planning comparison; never retry old slots."""
import argparse
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard

CONFIG = ROOT / 'configs/experiments/parallel_path_planning.json'


def prepare(root, gpu):
    if root.exists():
        raise ValueError('Keep previous runs and STOP markers')
    cfg = read_json(CONFIG)
    previous = ROOT / cfg['source_run']
    old = guard(previous, completed=True)
    if not read_json(previous / 'independent_review.json')['passed']:
        raise RuntimeError('Previous frozen inputs did not pass review')
    cases = [c for c in old['cases'] if c['id'] in cfg['cases']]
    if {c['id'] for c in cases} != set(cfg['cases']):
        raise RuntimeError('Pilot fixtures changed')
    root.mkdir(parents=True)
    inputs = {}
    development = ROOT / cfg['development_source'] / 'resident'
    checked = read_json(development / 'parallel_preflight/acceptance.json')
    if not checked['passed'] or checked['batch_queries'] != 12:
        raise RuntimeError('Parallel interface preflight is incomplete')
    for name in ['startup.json', 'parallel_preflight/acceptance.json']:
        path = development / name; inputs[str(path)] = sha256(path)
    for case in cases:
        source = previous / 'offline' / case['id'] / 'fine.json'
        target = root / 'offline' / case['id'] / 'fine.json'
        target.parent.mkdir(parents=True); shutil.copy2(source, target)
        for path in [source, target, Path(case['source_directory']) / 'state.json']:
            inputs[str(path)] = sha256(path)
    for name in ['state.json', 'closed_state.json']:
        target = root / 'warmup' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / 'warmup' / name, target)
        inputs[str(target)] = sha256(target)
    sources = {Path(p) for p in old['source_sha256'] if Path(p).exists()}
    sources.add(CONFIG)
    for folder in ['modules/curobo/parallel_planning', 'modules/pipeline/experiments/parallel_planning']:
        sources.update((ROOT / folder).glob('*.py'))
    for path in sources:
        target = root / 'source_registered' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    schedule = []
    for i, case in enumerate(cases):
        modes = cfg['variants']; modes = modes[i % 3:] + modes[:i % 3]
        for mode, repeat in [(m, 1) for m in modes] + [(m, 2) for m in reversed(modes)]:
            schedule.append({'id': case['id'] + '_' + mode + '_' + str(repeat),
                'case_id': case['id'], 'mode': mode, 'variant': mode + '_' + str(repeat), 'repeat': repeat})
    if len(schedule) != cfg['planning_requests']:
        raise RuntimeError('Planning budget differs from the registered schedule')
    (root / 'claims').mkdir()
    write_json(root / 'registration.json', {'at': now(), 'config': cfg, 'gpu': gpu, 'cases': cases,
        'schedule': schedule, 'source_sha256': {str(p): sha256(p) for p in sources},
        'input_sha256': inputs, 'planning_requests': len(schedule),
        'physical_grasps': 0, 'simulation_instances': 0, 'automatic_retries': False})
    write_json(root / 'progress.json', {'at': now(), 'phase': 'registered', 'completed_requests': 0,
        'total_requests': len(schedule), 'results': []})
    write_json(ROOT / 'tmp/parallel_planning/active.json', {'at': now(), 'run': str(root.relative_to(ROOT)),
        'phase': 'registered'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--gpu', required=True)
    args = p.parse_args(); prepare(ensure_run_path(args.root), args.gpu)
