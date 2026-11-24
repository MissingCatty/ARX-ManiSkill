"""Immutable advisor38 finite model-screen registration; no physical calls."""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, verify_files
from .advisor38_geometry import generate


def prepare(top):
    top = Path(top).resolve()
    output = top / 'prefilter_registration.json'
    if output.exists():
        raise FileExistsError('Model-screen registration is immutable')
    carry = read(top / 'carry_forward_registry.json')
    authority = read(top / 'user_authority.json')
    if authority['training_authorized'] or authority['maximum_registered_physical_candidates_per_object_cumulative'] != 15:
        raise RuntimeError('Advisor38 execution scope mismatch')
    previous = Path(read(top / 'scope_registration.json')['source_campaign'])
    basepath = previous / 'seed_preflight_03/registration.json'
    base = read(basepath)
    old36 = Path(carry['old_source'])
    files = dict(base['dependencies'])
    files[str(basepath)] = sha(basepath)
    for name in ('user_authority.json', 'scope_registration.json', 'source_inventory.json', 'advisor38.md', 'carry_forward_registry.json'):
        files[str(top / name)] = sha(top / name)
    pending = [r for r in read(top / 'source_inventory.json') if r['object_id'] not in carry['objects']]
    requests = []
    for f in sorted((basepath.parent / 'workers/planner/requests').glob('*.done')):
        value = read(f)
        if value['action'] == 'qualify_pool':
            requests.append((f, value))
    objects = {}
    for item in pending:
        obj = item['object_id']
        source_geometry = old36 / 'geometry' / (obj + '.json')
        original = read(source_geometry)
        trials = [r for r in carry['candidate_rounds'] if r['object_id'] == obj]
        tried = [np.array(r['geometry']['selected']['Gcanonical_object']) for r in trials]
        pool = generate(item, original, tried, count=60)
        if pool['registered_pool_count'] < 30:
            raise RuntimeError('Insufficient distinct normal-qualified candidates: ' + obj)
        path = top / 'candidate_pools' / (obj + '.json')
        write(path, pool)
        request_file, request = next((f, r) for f, r in requests if obj in r['scene'])
        files[str(path)] = sha(path)
        files[str(source_geometry)] = sha(source_geometry)
        files[str(request_file)] = sha(request_file)
        files[request['scene']] = sha(request['scene'])
        for mesh in read(request['scene'])['meshes']:
            if mesh.get('geometry_file'):
                files[mesh['geometry_file']] = mesh['geometry_sha256']
        objects[obj] = {'request': str(request_file), 'candidate_pool': str(path),
                        'source_object_world': item['source_object_world'], 'geometry': str(source_geometry),
                        'prior_registered_candidates': len(trials), 'maximum_registered_candidates_cumulative': 15}
        print('POOL', obj, pool['registered_pool_count'], 'past physical registrations', len(trials), flush=True)
    source = top / 'prefilter_source/COOLAS-ARX'
    prior = Path(base['source_snapshot'])
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'):
        (source.parent / name).symlink_to(ROOT.parent / name, target_is_directory=True)
    execution = {str(source / Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
                 for p, h in base['execution_source_files'].items()}
    for pattern in ('advisor38_*.py', 'runtime_advisor38_*.py', 'test_advisor38*.py'):
        for p in (ROOT / 'scripts/curobo_arx/nearfield').glob(pattern):
            destination = source / p.relative_to(ROOT)
            shutil.copy2(p, destination)
            execution[str(destination)] = sha(destination)
    resources = read(top / 'resource_plan.json')
    registration = {'created_unix': time.time(), 'deadline_unix': read(top / 'scope_registration.json')['deadline_unix'],
                    'source_snapshot': str(source), 'source_files': execution, 'files': files,
                    'base_registration': str(basepath), 'objects': objects, 'physical_commands': 0,
                    'maximum_candidates_per_object': 60, 'cpu_candidate_seconds': 180.,
                    'model_candidate_seconds': 600., 'training_authorized': False,
                    'compute_uuid': resources['roles']['compute_uuid'],
                    'CPU_stage': 'original ARX URDF seven-case IK and camera geometry',
                    'CUDA_stage': 'same original cuRobo model for reach and full dense path checks; no SimulationApp or World',
                    'postclosure_state_unknown': True, 'physical_success_requires_fresh_7_of_7': True,
                    'automatic_retries': 0}
    write(output, registration)
    verify_files({**files, **execution})
    write(top / 'prefilter_full_preflight.json', {'pass': True, 'checked_files': len({**files, **execution}),
                                               'physical_commands': 0, 'training_started': False})
    return str(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(prepare(parser.parse_args().root))
