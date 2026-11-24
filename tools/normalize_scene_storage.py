"""Keep the current 200 scenes and move 3D data into a scene-neutral library."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from arx_project.assets import recipe_sha256, load_object_catalog
from arx_project.layout import ROOT, read_json, write_json, sha256, now, canonical_path

RUN = ROOT / 'tmp/scene_storage/normalize_20261124_01'
ASSETS = ROOT / 'simulation/assets'
SCENES = ROOT / 'simulation/scenes'
LATEST = SCENES / 'latest200'


def remap(path, moves):
    value = str(path)
    for source, destination in sorted(moves, key=lambda pair: len(pair[0]), reverse=True):
        if value == source or value.startswith(source + '/'):
            return destination + value[len(source):]
    return value


def prepare():
    RUN.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT / 'configs/experiments/scene_dataset.json')
    source = Path(config['dataset'])
    catalog = read_json(source / 'object_catalog.json')
    models = catalog['models']
    assert len(models) == 80
    moves = []
    for name, row in models.items():
        destination = ASSETS / 'models' / name
        if row['source'] == 'robocasa':
            moves.append((str(Path(row['source_definition']).parent), str(destination)))
        else:
            files = {row['source_definition'], *row['source_hashes'],
                     *(item['file'] for item in row['visuals'] + row['collisions'])}
            for file in sorted(files):
                moves.append((file, str(destination / Path(file).name)))
    moves += [(str(ASSETS / 'maniskill/arx'), str(ASSETS / 'models/arx_x5_lift')),
              (str(ROOT / 'modules/maniskill/source/mani_skill/utils/scene_builder/table/assets'),
               str(ASSETS / 'models/workspace_table'))]
    for name in ['arx_source', 'easg', 'generated', 'maniskill', 'robot_sources']:
        moves.append((str(ASSETS / name), str(ASSETS / ('models/source_' + name))))
    relative_models = deepcopy(models)
    for row in relative_models.values():
        convert = lambda p: str(Path(remap(p, moves)).relative_to(ASSETS))
        row['source_definition'] = convert(row['source_definition'])
        row['source_hashes'] = {convert(p): digest for p, digest in row['source_hashes'].items()}
        for item in row['visuals'] + row['collisions']:
            item['file'] = convert(item['file'])
    shared = {'schema_version': 1, 'kind': 'shared_3d_asset_catalog',
              'categories': catalog['categories'], 'models': relative_models,
              'resources': {
                  'arx_robot': {'directory': 'models/arx_x5_lift'},
                  'workspace_table': {'file': 'models/workspace_table/table.glb'},
                  'maniskill_library': {'directory': 'models/source_maniskill'},
                  'arx_library': {'directory': 'models/source_arx_source'}},
              'source_catalog_sha256': sha256(source / 'object_catalog.json')}
    reference = {'schema_version': 1, 'kind': 'shared_asset_catalog_reference',
                 'catalog_ref': 'simulation/assets/catalog.json',
                 'asset_ids': list(models), 'categories': catalog['categories'],
                 'recipe_sha256': {name: recipe_sha256(row) for name, row in relative_models.items()}}
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*') if p.is_file()}
    write_json(RUN / 'plan.json', {'at': now(), 'source': str(source), 'destination': str(LATEST),
               'moves': moves, 'shared_catalog': shared, 'scene_reference': reference,
               'original_models': models, 'current_scene_files': before,
               'old_scene_roots': [str(p) for p in SCENES.iterdir() if p != source.parent]})
    for name in ['frozen_dataset.json', 'registration.json', 'manifest.json', 'object_catalog.json']:
        shutil.copy2(source / name, RUN / ('before_' + name))
    shutil.copy2(ROOT / 'configs/experiments/active_experiment.json', RUN / 'before_active_configuration.json')
    print('Prepared:', len(models), 'shared object models; 200 current scenes.')


def apply():
    plan = json.loads((RUN / 'plan.json').read_text())
    for source, destination in plan['moves']:
        source, destination = Path(source), Path(destination)
        if not source.exists():
            if not destination.exists(): raise FileNotFoundError(source)
            continue
        if destination.exists(): raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
    source = Path(plan['source'])
    if source.exists(): shutil.move(source, LATEST)

    aliases_file = ROOT / 'configs/path_aliases.json'
    aliases = json.loads(aliases_file.read_text())
    for source, destination in plan['moves']:
        aliases['aliases'][source] = str(Path(destination).relative_to(ROOT))
    aliases['aliases'][plan['source']] = str(LATEST.relative_to(ROOT))
    write_json(aliases_file, aliases)
    write_json(ASSETS / 'catalog.json', plan['shared_catalog'])

    # Old third-party readers refer to the common library, never to a scene copy.
    library_robot = ASSETS / 'models/source_maniskill/arx'
    if not library_robot.exists(): library_robot.symlink_to('../arx_x5_lift', target_is_directory=True)
    table = ROOT / 'modules/maniskill/source/mani_skill/utils/scene_builder/table/assets'
    if not table.exists(): table.symlink_to(os.path.relpath(ASSETS / 'models/workspace_table', table.parent), target_is_directory=True)
    for base in [ROOT / 'modules', ROOT / 'simulation', ROOT / 'configs']:
        for folder, dirs, files in os.walk(base):
            dirs[:] = [n for n in dirs if n not in {'.git', '__pycache__'}]
            for name in dirs + files:
                link = Path(folder) / name
                if not link.is_symlink(): continue
                old = link.resolve(strict=False)
                target = canonical_path(old)
                if target != old and target.exists():
                    link.unlink(); link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())

    write_json(LATEST / 'object_catalog.json', plan['scene_reference'])
    registration = json.loads((LATEST / 'registration.json').read_text())
    registration['object_catalog_sha256'] = sha256(LATEST / 'object_catalog.json')
    registration['asset_storage_revision'] = {'at': now(), 'catalog': 'simulation/assets/catalog.json',
        'parent_registration_sha256': sha256(RUN / 'before_registration.json'),
        'scene_positions_and_asset_physics_changed': False}
    write_json(LATEST / 'registration.json', registration)
    manifest = json.loads((LATEST / 'manifest.json').read_text())
    manifest['registration_sha256'] = sha256(LATEST / 'registration.json')
    write_json(LATEST / 'manifest.json', manifest)
    frozen = json.loads((LATEST / 'frozen_dataset.json').read_text())
    for name in ['object_catalog.json', 'registration.json', 'manifest.json']:
        assert name in frozen['files']
        frozen['files'][name] = sha256(LATEST / name)
    frozen['storage_revision'] = {'at': now(), 'parent_freeze_sha256': sha256(RUN / 'before_frozen_dataset.json'),
        'changed_files': ['object_catalog.json', 'registration.json', 'manifest.json'],
        'scene_entries_and_observations_changed': False}
    write_json(LATEST / 'frozen_dataset.json', frozen)
    separate_source_assets()
    pointer_path = ROOT / 'configs/experiments/scene_dataset.json'
    pointer = read_json(pointer_path)
    pointer.update(dataset=str(LATEST), object_catalog=str(LATEST / 'object_catalog.json'),
                   shared_asset_catalog='simulation/assets/catalog.json',
                   verification=str(LATEST / 'verification.json'), freeze=str(LATEST / 'frozen_dataset.json'),
                   grasp_tests_authorized=False, grasp_test_budget=0)
    write_json(pointer_path, pointer)
    register_configuration()
    verify()


def register_configuration():
    path = ROOT / 'configs/experiments/active_experiment.json'
    current = read_json(path)
    current.update(dataset=str(LATEST), dataset_freeze_sha256=sha256(LATEST / 'frozen_dataset.json'))
    hashes = {str(canonical_path(p)): sha256(canonical_path(p)) for p in current['configuration_hashes']}
    for path_new in [ROOT / 'arx_project/assets.py', ROOT / 'arx_project/scenes.py', ASSETS / 'catalog.json',
                     ROOT / 'modules/pipeline/experiments/diverse_object_scenes/asset_environment.py']:
        hashes[str(path_new)] = sha256(path_new)
    current['configuration_hashes'] = hashes
    current['asset_storage_revision'] = {'at': now(), 'parent_configuration_sha256': sha256(RUN / 'before_active_configuration.json'),
        'shared_asset_catalog': 'simulation/assets/catalog.json', 'scene_directory': 'simulation/scenes/latest200',
        'robot_geometry_physics_initial_conditions_changed': False, 'physical_grasps': 0}
    write_json(path, current)


def separate_source_assets():
    """Keep the saved kinematic URDF with robot assets, rather than scenes."""
    source = LATEST / 'source/model__arx.urdf'
    if not source.exists(): return
    destination = ASSETS / 'models/arx_kinematic_reference/arx.urdf'
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    shutil.move(source, destination)
    catalog = json.loads((ASSETS / 'catalog.json').read_text())
    catalog['resources']['arx_kinematic_reference'] = {
        'file': str(destination.relative_to(ASSETS)), 'sha256': digest}
    write_json(ASSETS / 'catalog.json', catalog)
    write_json(source.with_name('model__arx.reference.json'), {
        'kind': 'shared_robot_reference', 'catalog_ref': 'simulation/assets/catalog.json',
        'resource': 'arx_kinematic_reference', 'sha256': digest})
    plan = json.loads((RUN / 'plan.json').read_text())
    plan.setdefault('relocated_scene_files', {})['source/model__arx.urdf'] = str(destination)
    write_json(RUN / 'plan.json', plan)
    aliases_file = ROOT / 'configs/path_aliases.json'
    aliases = json.loads(aliases_file.read_text())
    aliases['aliases'][str(source)] = str(destination.relative_to(ROOT))
    write_json(aliases_file, aliases)


def verify():
    plan = json.loads((RUN / 'plan.json').read_text())
    models = load_object_catalog(LATEST / 'object_catalog.json')['models']
    assert len(models) == 80
    files_checked = 0
    for name, row in models.items():
        old = plan['original_models'][name]
        new_plain, old_plain = deepcopy(row), deepcopy(old)
        for plain in [new_plain, old_plain]:
            plain.pop('source_definition'); plain.pop('source_hashes')
            for item in plain['visuals'] + plain['collisions']: item.pop('file')
        assert old_plain == new_plain, 'Geometry/physics recipe changed: ' + name
        for file, expected in row['source_hashes'].items():
            assert sha256(file) == expected, file
            files_checked += 1
    changed = {'object_catalog.json', 'registration.json', 'manifest.json', 'frozen_dataset.json'}
    for name, digest in plan['current_scene_files'].items():
        path = Path(plan.get('relocated_scene_files', {}).get(name, LATEST / name))
        if name not in changed: assert sha256(path) == digest, name
    frozen = json.loads((LATEST / 'frozen_dataset.json').read_text())
    for name, digest in frozen['files'].items(): assert sha256(LATEST / name) == digest, name
    manifest = json.loads((LATEST / 'manifest.json').read_text())
    schedule = manifest['schedule']
    assert len(schedule) == 200
    for entry in schedule:
        saved = json.loads((LATEST / 'cases' / entry['id'] / 'entry.json').read_text())
        assert saved == entry, entry['id']
        for actor in entry['scene']['actors']: assert actor['model_id'] in models
    report = {'passed': True, 'at': now(), 'scenes': 200, 'object_models': 80,
              'asset_source_hashes_checked': files_checked, 'frozen_files': len(frozen['files']),
              'scene_entries_observations_unchanged': True, 'geometry_physics_recipes_unchanged': True,
              'physical_grasps': 0, 'old_scenes_deleted': set(SCENES.iterdir()) == {LATEST}}
    write_json(RUN / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False))


def cleanup():
    separate_source_assets()
    register_configuration()
    verify()
    assert json.loads((RUN / 'runtime_validation.json').read_text())['passed']
    removed = []
    for path in list(SCENES.iterdir()):
        if path == LATEST: continue
        count = sum(p.is_file() for p in path.rglob('*'))
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
        if path.is_symlink(): path.unlink()
        elif path.is_dir(): shutil.rmtree(path)
        else: path.unlink()
        removed.append({'path': str(path.relative_to(ROOT)), 'files': count, 'bytes': size})
    # Remove downloaded layout/episode data, keeping all mesh and material libraries.
    library = ASSETS / 'models/source_maniskill/data/scene_datasets'
    candidates = [library / 'replica_cad_dataset' / name
                  for name in ['rearrange', 'navmeshes', 'navmeshes_default', '.cache', 'configs/scenes']]
    candidates += [library / 'replica_cad_dataset/rearrange' / name
                   for name in ['train', 'val', 'spawn_data', 'grasp_poses', 'task_plans']]
    candidates += list((library / 'ai2thor').glob('*/configs/scenes'))
    meshes = {'.obj', '.glb', '.gltf', '.stl', '.usd', '.usda', '.usdc', '.urdf', '.ply'}
    for path in candidates:
        if not path.exists(): continue
        scene_navigation_only = path == library / 'replica_cad_dataset/configs/scenes'
        if not scene_navigation_only and any(p.suffix.lower() in meshes for p in path.rglob('*') if p.is_file()): continue
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        shutil.rmtree(path)
        removed.append({'path': str(path.relative_to(ROOT)), 'bytes': size})
    for base in [ROOT / 'modules', ROOT / 'configs']:
        for folder, dirs, files in os.walk(base):
            dirs[:] = [n for n in dirs if n not in {'.git', '__pycache__'}]
            for name in dirs + files:
                link = Path(folder) / name
                if link.is_symlink() and not link.exists() and str(SCENES) in str(link.resolve(strict=False)):
                    link.unlink()
    receipt = RUN / 'cleanup.json'
    previous = json.loads(receipt.read_text())['removed'] if receipt.exists() else []
    write_json(receipt, {'at': now(), 'removed': previous + removed, 'remaining_scene_directories': ['latest200']})
    print('Removed old scene collections; retained shared 3D models.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'apply', 'register', 'verify', 'cleanup'])
    args = parser.parse_args()
    {'prepare': prepare, 'apply': apply, 'register': register_configuration,
     'verify': verify, 'cleanup': cleanup}[args.stage]()
