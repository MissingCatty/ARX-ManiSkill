"""Register a scheduler handoff without restarting simulator children."""
import argparse
import shutil
from pathlib import Path
import time
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, verify_files


def prepare(top):
    top = Path(top).resolve()
    oldtop = Path(read(top / 'scope_registration.json')['source_campaign'])
    old = read(oldtop / 'data_supervisor_registration.json')
    spec = read(oldtop / 'prefilter_registration.json')
    if (top / 'supplement_registration.json').exists() or (top / 'data_supervisor_registration.json').exists():
        raise FileExistsError('Keep every registration immutable')
    source = top / 'supplement_source/COOLAS-ARX'
    prior = Path(spec['source_snapshot'])
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'):
        (source.parent / name).symlink_to(ROOT.parent / name, target_is_directory=True)
    sources = {str(source / Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
               for p, h in spec['source_files'].items()}
    additions = list((ROOT / 'scripts/curobo_arx/nearfield').glob('*advisor39*.py'))
    for p in additions:
        dest = source / p.relative_to(ROOT); shutil.copy2(p, dest); sources[str(dest)] = sha(dest)
    files = dict(spec['files'])
    boundary = {str(top / f): sha(top / f) for f in ('user_authority.json', 'scope_registration.json',
                'advisor39.md', 'version_A_selection.json', 'prefilter_registration.json')}
    files.update(boundary)
    s = read(top / 'scope_registration.json')
    reg = {'created_unix': time.time(), 'predecessor': str(oldtop), 'deadline_unix': s['deadline_unix'],
           'compute_uuid': spec['compute_uuid'], 'source_snapshot': str(source),
           'source_files': sources, 'files': files, 'training_authorized': False,
           'maximum_density_levels': s['maximum_supplemental_geometry_levels'],
           'geometry_candidates_per_level': s['maximum_raw_proposals_per_object_per_level'],
           'activation': 'all original sixty terminal and zero original full-model qualification',
           'physical_candidate_limit_cumulative': 15, 'physical_commands': 0,
           'original_deadline_retained': True, 'CPU_front_filter_seconds': 30,
           'CPU_pose_tolerances_m_degrees': [.0005, .2],
           'model_gate': 'same advisor38 full seven paths, original robot/scene/physics/success',
           'unused_qualified_candidate_backpressure': 2}
    write(top / 'supplement_registration.json', reg)
    verify_files({**files, **sources})
    write(top / 'supplement_full_preflight.json', {'pass': True, 'files_checked': len(files) + len(sources),
          'physical_trials': 0, 'existing_children_interrupted': 0})
    root_sources = {**old['source_files'], **{str(p): sha(p) for p in additions}}
    fixed = {**old['fixed_files'], **boundary,
             str(top / 'supplement_registration.json'): sha(top / 'supplement_registration.json')}
    verify_files({**fixed, **root_sources})
    supervisor = {'created_unix': time.time(), 'predecessor': str(oldtop), 'deadline_unix': s['deadline_unix'],
                  'source_files': root_sources, 'fixed_files': fixed, 'boundary_files': boundary,
                  'seed_devices': old['seed_devices'], 'static_devices': old['static_devices'],
                  'max_physical_candidates_cumulative': 15, 'training_authorized': False,
                  'main_training_dataset': 'Version_A', 'version_B_work': False,
                  'existing_children_must_keep_process_identity': True}
    write(top / 'data_supervisor_registration.json', supervisor)
    return str(top / 'data_supervisor_registration.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root')
    print(prepare(parser.parse_args().root))
