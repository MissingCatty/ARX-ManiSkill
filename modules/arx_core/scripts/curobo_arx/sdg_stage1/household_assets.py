"""Fetch pinned, authored-physics household USDs; never modify source assets.

This is CPU-only asset preparation, not simulated grasp validation. LFS objects
are verified against their upstream pointer, and all referenced assets must be
present before an object can enter the additional catalog.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import urllib.request

import numpy as np

from scripts.curobo_arx.data_factory.common import ROOT, read, sha, space, write
from .asset_contract import source_contract
from .discover_assets import geometry_key
from .sampler import composed_mesh


REPOSITORY = 'NVlabs/RoboLab'
EXCLUDED = {'lizard_figurine', 'birdhouse', 'megaphone', 'foam_roller'}
DATASETS = {'hope', 'hot3d', 'ycb', 'handal'}
FILE_LIMIT = 64 * 1024 * 1024


def candidates(metadata):
    """Broad household coverage; missing physics stays explicitly ineligible."""
    accepted, excluded = [], []
    for row in metadata:
        reason = None
        size = np.asarray(row.get('dims', []), dtype=float)
        if row['dataset'] not in DATASETS:
            reason = 'Outside this authored-mass household source pack'
        elif row['name'] in EXCLUDED:
            reason = 'Not selected for the common tabletop object collection'
        elif not row.get('rigid_body') or row.get('static_body'):
            reason = 'Not a dynamic rigid object'
        elif size.shape != (3,) or not np.isfinite(size).all() or min(size) < .002 or max(size) > .65:
            reason = 'Outside original tabletop dimension envelope; no rescaling'
        elif not 0 < float(row.get('mass') or 0) < 20:
            reason = 'No explicit positive source mass; no guessed replacement'
        if reason:
            excluded.append({'source_path': row['usd_path'], 'name': row['name'], 'reason': reason})
        else:
            accepted.append(row)
    return accepted, excluded


def fetch_pack(metadata_dir, out, workers=6, supplement_tree_only=False, dedup_catalog=None):
    from pxr import Sdf
    metadata_dir, out = Path(metadata_dir).resolve(), Path(out).resolve()
    tree = read(metadata_dir / 'robolab_tree.json')
    metadata = read(metadata_dir / 'upstream_catalog.json')
    revision = tree['sha']
    files = {x['path']: x for x in tree['tree'] if x['type'] == 'blob'}
    selected, excluded = candidates(metadata)
    if supplement_tree_only:
        indexed = {r['usd_path'] for r in metadata}
        selected = [{'usd_path': path, 'name': PurePosixPath(path).stem,
                     'dataset': PurePosixPath(path).parts[2],
                     'metadata_evidence':'Unindexed pinned source USD; mass and dimensions checked directly'}
                    for path in sorted(files) if path.startswith('assets/objects/') and path.endswith('.usd')
                    and PurePosixPath(path).parts[2] in DATASETS and path not in indexed]
    out.mkdir(parents=True, exist_ok=False)
    cache = out / 'upstream'
    cache.mkdir()
    receipts, failures = {}, {}

    def request(url, limit):
        error = None
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'ARX-household-asset-preparation'})
                with urllib.request.urlopen(req, timeout=20) as response:
                    data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError('Source file exceeds declared download bound')
                return data
            except (OSError, ValueError) as exc:
                error = exc
        raise error

    def download(relative):
        try:
            relative = str(PurePosixPath(relative))
            if relative not in files or relative.startswith('/') or '..' in PurePosixPath(relative).parts:
                raise ValueError('Dependency outside pinned repository tree: ' + relative)
            raw_url = f'https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{relative}'
            pointer = request(raw_url, FILE_LIMIT)
            blob_sha = hashlib.sha1(b'blob ' + str(len(pointer)).encode() + b'\0' + pointer).hexdigest()
            if blob_sha != files[relative]['sha']:
                raise ValueError('Git blob identity mismatch')
            data, url, lfs = pointer, raw_url, False
            if pointer.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                fields = dict(line.split(' ', 1) for line in pointer.decode().splitlines()[1:])
                expected = fields['oid'].removeprefix('sha256:')
                length = int(fields['size'])
                if length > FILE_LIMIT:
                    raise ValueError('LFS object exceeds declared per-file bound')
                space(out, length)
                url = f'https://media.githubusercontent.com/media/{REPOSITORY}/{revision}/{relative}'
                data = request(url, length)
                if len(data) != length or hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('LFS size or SHA256 mismatch')
                lfs = True
            target = cache / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            return relative, {'url': url, 'path': str(target), 'sha256': sha(target),
                              'bytes': len(data), 'git_blob_sha1': blob_sha, 'lfs_verified': lfs}, None
        except (OSError, ValueError, KeyError) as exc:
            return relative, None, type(exc).__name__ + ': ' + str(exc)

    def batch(paths):
        pending = sorted(set(paths) - receipts.keys() - failures.keys())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for relative, receipt, error in executor.map(download, pending):
                if error:
                    failures[relative] = error
                else:
                    receipts[relative] = receipt
                print('ASSET_FILE', len(receipts), len(failures), relative, error or 'verified', flush=True)
        write(out / 'download_receipt.json', {'repository': REPOSITORY, 'revision': revision,
              'files': receipts, 'failures': failures, 'bytes': sum(x['bytes'] for x in receipts.values())})

    licenses = [x for x in files if x == 'LICENSE' or
                x.startswith('assets/objects/') and PurePosixPath(x).name.lower().startswith(('license', 'copying'))]
    batch([x['usd_path'] for x in selected] + licenses)
    dependencies = {}
    visited = set()
    while True:
        pending_layers = sorted(x for x in receipts if x.endswith(('.usd', '.usda', '.usdc')) and x not in visited)
        if not pending_layers:
            break
        paths = set()
        for relative in pending_layers:
            visited.add(relative)
            try:
                layer = Sdf.Layer.OpenAsAnonymous(str(cache / relative))
                if layer is None:
                    raise ValueError('USD layer cannot be read')
                targets = []
                for dep in re.findall(r'@([^@\n]+)@', layer.ExportToString()):
                    # NVIDIA built-in material modules are provided by Isaac Sim.
                    if dep.endswith('.mdl') and '/' not in dep:
                        continue
                    normalized = posixpath.normpath(posixpath.join(posixpath.dirname(relative), dep))
                    if normalized not in files:
                        raise ValueError('Unresolved source dependency: ' + dep)
                    targets.append(normalized)
                dependencies[relative] = sorted(set(targets))
                paths.update(targets)
            except (ValueError, RuntimeError) as exc:
                failures[relative] = str(exc)
        batch(paths)

    def dependency_error(relative, seen=None):
        seen = set() if seen is None else seen
        if relative in seen:
            return None
        seen.add(relative)
        if relative in failures:
            return failures[relative]
        if relative not in receipts:
            return 'Dependency was not downloaded: ' + relative
        for dep in dependencies.get(relative, []):
            error = dependency_error(dep, seen)
            if error:
                return error
        return None

    # Preserve historical objects and deduplicate against their actual geometry.
    prior = read(ROOT / 'runs/sdg_stage1_20261031_07/catalog.json')['objects']
    if dedup_catalog:
        prior += [{**a, 'object_id':a['name'], 'available':True, 'usd_path':a['path']} for a in read(dedup_catalog)]
    keys = {}
    for asset in prior:
        if asset['available'] and not asset['object_id'].lower().startswith('egad'):
            mesh, _ = composed_mesh(asset['usd_path'])
            keys[geometry_key(mesh)] = asset['object_id']
    catalog, objects = [], []
    for record in selected:
        relative = record['usd_path']
        row = {'source_path': relative, 'source_name': record['name'], 'dataset': record['dataset'], 'available': False}
        try:
            error = dependency_error(relative)
            if error:
                raise ValueError(error)
            path = cache / relative
            contract = source_contract(path)
            mesh, evidence = composed_mesh(path)
            key = geometry_key(mesh)
            if key in keys:
                row.update(reason='Exact composed render geometry already registered', duplicate_of=keys[key])
            else:
                name = 'household_' + record['dataset'] + '_' + path.stem
                size = np.asarray(mesh.extents)
                if min(size) < .002 or max(size) > .65:
                    raise ValueError('Measured dimensions outside original tabletop envelope; no rescaling')
                catalog.append({'name': name, 'path': str(path), 'container_path': str(path),
                    'source': f'https://github.com/{REPOSITORY}/blob/{revision}/{relative}',
                    'upstream_revision': revision, 'source_name': record['name'],
                    'category': record['dataset'], 'semantic_class': record.get('class'),
                    'description': record.get('description'), 'size_m': size.tolist(),
                    'mass_kg': contract['expected_mass_kg'], 'physics_contract': contract,
                    'placement': {'position': [4.048, 2.08, float(.679 - mesh.bounds[0, 2] + .012)],
                                  'orientation': [1., 0., 0., 0.]},
                    'registered_original_sha256': sha(path), 'exact_visual_geometry_key': key,
                    'new_scene_version': 'common_household_20261101_01',
                    'usage_conditions': 'Research use under upstream repository and dataset-specific licenses, retained in upstream/',
                    'dependency_files': {k: v['sha256'] for k, v in receipts.items()
                                         if k == relative or k in dependencies.get(relative, [])},
                    'near_shape_deduplication': 'Not established; distinct model instances are not necessarily distinct semantic categories'})
                keys[key] = name
                row.update(available=True, object_id=name, size_m=size.tolist(), mass_kg=contract['expected_mass_kg'])
        except (ValueError, RuntimeError, OSError) as exc:
            row['reason'] = str(exc)
        objects.append(row)
        print('OBJECT_CHECK', record['dataset'], record['name'], row.get('object_id') or row.get('reason'), flush=True)
    write(out / 'catalog.json', catalog)
    write(out / 'asset_receipt.json', {'repository': REPOSITORY, 'revision': revision,
        'source_candidates': len(selected), 'additional_usable_objects': len(catalog),
        'objects': objects, 'metadata_exclusions': excluded, 'dependencies': dependencies,
        'physics_mutations': False, 'new_grasp_experiments': 0,
        'validation_scope': 'Static USD/source/dependency checks only; no physical grasp or contact-quality claims'})
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--supplement-tree-only', action='store_true')
    parser.add_argument('--dedup-catalog')
    args = parser.parse_args()
    print('NEW_HOUSEHOLD_OBJECTS', len(fetch_pack(args.metadata_dir, args.out, args.workers,
          args.supplement_tree_only, args.dedup_catalog)))
