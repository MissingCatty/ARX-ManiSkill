"""Register a new CPU-cache pilot without changing any previous run."""
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json, ensure_run_path
from .offline import CONFIG


def prepare(root):
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    if root.exists(): raise ValueError('Use a fresh run; preserve old STOP markers')
    current = load_current(); cfg = read_json(CONFIG); prior = ROOT/cfg['source_run']
    accepted = read_json(prior/'acceptance.json')
    if not accepted['record_integrity'] or not read_json(prior/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Previous completed execution was not independently accepted and stopped')
    old = read_json(prior/'registration.json')
    metadata = read_json(ROOT/'configs/experiments/history/scene_reuse_optional_20261128_01.json')
    allowed = {str(ROOT/'configs/modules.json'), str(ROOT/'configs/experiments/active_experiment.json')}
    for path, digest in old['source_sha256'].items():
        archive = prior/'source_registered'/Path(path).relative_to(ROOT)
        if sha256(archive) != digest: raise ValueError('Previous source archive changed: '+path)
        if path in allowed:
            if digest != metadata['previous_sha256'][path] or sha256(path) != metadata['current_sha256'][path]:
                raise ValueError('Optional registry acknowledgement mismatch')
        elif sha256(path) != digest:
            raise ValueError('Previous runtime changed: '+path)
    offline = ROOT/'tmp/state_updates/scene_reuse_cpu_offline_20261128_01/offline_review.json'
    if not read_json(offline)['passed']: raise ValueError('CPU geometry equivalence is incomplete')
    root.mkdir(parents=True); (root/'claims').mkdir()
    inputs = {str(offline): sha256(offline)}
    cases = []
    for cid in cfg['cases']:
        original = next(c for c in old['cases'] if c['id'] == cid)
        source = ROOT/original['input_directory']
        target = root/'inputs'/cid; target.mkdir(parents=True)
        for name in ['state.json', 'candidates/candidate_pool.json']:
            p = source/name; dest = target/name
            if sha256(p) != old['input_sha256'][str(p)]: raise ValueError('Frozen input changed')
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
            inputs[str(p)] = sha256(p); inputs[str(dest)] = sha256(dest)
        dest = root/'offline'/cid/'fine.json'; dest.parent.mkdir(parents=True)
        shutil.copy2(target/'candidates/candidate_pool.json', dest); inputs[str(dest)] = sha256(dest)
        cases.append({'id': cid, 'source_directory': str(target), 'original_input': str(source.relative_to(ROOT))})
    for name in ['state.json', 'closed_state.json']:
        dest = root/'warmup'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prior/'warmup'/name, dest); inputs[str(dest)] = sha256(dest)
    for name in ['registration.json', 'acceptance.json', 'services_stopped.json']:
        p = prior/name; inputs[str(p)] = sha256(p)
    metadata_path = ROOT/'configs/experiments/history/scene_reuse_optional_20261128_01.json'
    inputs[str(metadata_path)] = sha256(metadata_path)
    sources = {Path(p) for p in old['source_sha256']} | {CONFIG}
    for folder in ['modules/curobo/scene_reuse', 'modules/pipeline/experiments/scene_reuse',
                   'modules/pipeline/experiments/parallel_planning']:
        sources.update((ROOT/folder).glob('*.py'))
    for name in ['types.py', 'data/data_mesh.py', 'data/data_scene.py', 'collision/collision_scene.py']:
        sources.add(ROOT/'modules/curobo/source/curobo/_src/geom'/name)
    sources.update(Path(p) for p in current['configuration_hashes'])
    for p in sources:
        dest = root/'source_registered'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    schedule = []
    for index, case in enumerate(cases):
        modes = ['control', 'cached'] if index % 2 == 0 else ['cached', 'control']
        for mode, repeat in [(m, 1) for m in modes] + [(m, 2) for m in reversed(modes)]:
            schedule.append({'id': case['id']+'_'+mode+'_'+str(repeat), 'case_id': case['id'],
                             'mode': mode, 'variant': mode+'_'+str(repeat), 'repeat': repeat})
    if len(schedule) != 20: raise ValueError('Wrong stage-one planning budget')
    write_json(root/'registration.json', {'at': now(), 'experiment': 'V8_CPU_prepared_geometry',
        'config': cfg, 'gpu': cfg['fixed_gpu_uuid'], 'cases': cases, 'schedule': schedule,
        'planning_requests': 20, 'physical_grasps': 0, 'simulation_instances': 0,
        'automatic_retries': False, 'source_sha256': {str(p): sha256(p) for p in sources},
        'input_sha256': inputs, 'prior_source_archives_verified': True})
    write_json(root/'progress.json', {'at': now(), 'phase': 'registered', 'completed_requests': 0,
        'total_requests': 20, 'results': []})
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json', {'at': now(), 'run': str(root.relative_to(ROOT)),
        'stage': 'cpu_prepared_geometry', 'phase': 'registered'})
    return root


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    prepare(ensure_run_path(parser.parse_args().root))
