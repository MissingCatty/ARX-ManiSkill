"""Immutable variant settings and historical evidence for a new V4 batch."""
from pathlib import Path

from arx_project.layout import ROOT, ensure_run_path, read_json, sha256


def load_variant(path, base_path):
    if path is None:
        return None
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Variant configuration must belong to this project')
    variant = read_json(path)
    base = read_json(base_path)
    if (ROOT/variant['base_experiment']).resolve() != Path(base_path).resolve():
        raise ValueError('Variant uses a different base experiment')
    if variant['dataset'] != base['dataset'] or variant['physical_slot_budget'] != 100:
        raise ValueError('Map comparison must retain the same 100 frozen scenes')
    if variant['component_modules'] != {'curobo_dual_arm': 'curobo_dual_arm_map'}:
        raise ValueError('Variant must retain the original perception components')
    options_path = ROOT/variant['station_search_configuration']
    options = read_json(options_path)
    if options['trajopt_policy'] != 'interactive' or options['grasp_pose_changes']:
        raise ValueError('Map variant must retain production optimization and original grasp poses')
    # Offline validation budgets describe a past diagnostic, not this new run.
    options = {k: v for k, v in options.items() if k not in ['validation', 'activation', 'parent_version']}
    options['registered_planner_module'] = options['planner_module']
    options['planner_module'] = 'curobo_dual_arm'
    return {**variant, 'configuration_file': str(path.relative_to(ROOT)),
            'station_search_options': options}


def preserved_predecessor(variant, dataset_digest):
    previous = ensure_run_path(ROOT/variant['predecessor'])
    registration = read_json(previous/'registration.json')
    acceptance = read_json(previous/'acceptance.json')
    if registration['dataset_freeze_sha256'] != dataset_digest:
        raise ValueError('New comparison scenes differ from the previous batch')
    if not acceptance['record_integrity'] or not (previous/'STOP').is_file():
        raise ValueError('Prior stopped batch lacks accepted historical records')
    if not read_json(previous/'services_stopped.json')['all_owned_exited']:
        raise ValueError('Prior components have not exited')
    for name, expected in registration['source_hashes'].items():
        archived = previous/'source_registered'/Path(name).relative_to(ROOT)
        if sha256(archived) != expected:
            raise ValueError('Historical source archive differs: '+name)
    files = [previous/name for name in ['registration.json', 'acceptance.json', 'STOP', 'services_stopped.json']]
    files += sorted((previous/'claims').glob('*.json'))
    files += sorted((previous/'trials').glob('*/result.json'))
    return {'run': str(previous), 'files': {str(p): sha256(p) for p in files},
            'completed': acceptance['completed'], 'interrupted': acceptance['interrupted'],
            'not_started': acceptance['not_started'], 'successes': acceptance['successes'],
            'original_STOP_retained': True, 'historical_source_archive_verified': True}


def map_resources(options):
    if options.get('placement_method') == 'high_workspace_priority':
        files = [ROOT/options['workspace'], ROOT/options['density_atlas']]
        return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                        'sha256': sha256(p)} for p in files}
    directory = ROOT/options['map_directory']
    meta = read_json(directory/'地图元数据.json')
    index = ROOT/options['index_directory']
    files = [directory/name for name in ['地图元数据.json', '完成.json', meta['witness_file']]]
    files += [index/name for name in ['completion.json', 'order.npy', 'offsets.npy']]
    return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                     'sha256': sha256(p)} for p in files}


def verify_preserved_and_resources(reg, *, full_resources=False):
    for name, expected in (reg.get('predecessor') or {}).get('files', {}).items():
        if not full_resources and Path(name).name not in ['registration.json', 'acceptance.json', 'STOP', 'services_stopped.json']:
            continue
        if sha256(name) != expected:
            raise RuntimeError('Historical batch was modified: '+name)
    for name, value in reg.get('map_resources', {}).items():
        path = Path(name)
        stat = path.stat()
        if (stat.st_size != value['size'] or stat.st_mtime_ns != value.get('mtime_ns', stat.st_mtime_ns)
                or (full_resources and sha256(path) != value['sha256'])):
            raise RuntimeError('Registered reachability resource changed: '+name)
