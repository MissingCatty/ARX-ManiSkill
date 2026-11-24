"""Extensible catalog independent of the old fixed-67 manifest validator."""
from arx_project.layout import canonical_path
import gzip
from pathlib import Path
import numpy as np
from .common import ROOT, VERSION, digest, read, sha, write

KNOWN_ROLES = {'bbq_sauce_bottle':'train','rubiks_cube':'train','mustard_bottle':'train',
               'apple_01':'train','005_tomato_soup_can':'train','lemon_01':'validation','egad_A1':'development'}


def collection_input(extra, policy):
    """Resolve the active household pack without a silent legacy fallback."""
    if extra is not None or policy != 'common_household':
        return extra, None
    profile_path = ROOT/'configs/lilo_vla/common_household_collection.json'
    profile = read(profile_path)
    if profile['object_policy'] != policy:
        raise ValueError('Active object profile has a different collection policy')
    path = ROOT/profile['extra_catalog']
    if sha(path) != profile['extra_catalog_sha256']:
        raise ValueError('Active household catalog hash mismatch')
    return path, profile_path


def select_collection(catalog, policy):
    """Select collection targets without removing any frozen scene object."""
    import copy
    if policy == 'all':
        return catalog
    if policy != 'common_household':
        raise ValueError('Unknown collection object policy: '+str(policy))
    result = copy.deepcopy(catalog)
    excluded_names = {'lizard_figurine', 'birdhouse', 'megaphone', 'foam_roller'}
    for obj in result['objects']:
        evidence = ' '.join(str(obj.get(k, '')) for k in ('object_id', 'path', 'usd_path', 'source')).lower()
        reason = ('EGAD suspended by user; retain original physical scene object' if 'egad' in evidence else
                  'Outside current common tabletop object selection' if obj.get('source_name', obj['object_id']) in excluded_names else None)
        obj['collection_enabled'] = reason is None
        obj['collection_exclusion_reason'] = reason
    selected = [o['object_id'] for o in result['objects'] if o['collection_enabled'] and o['available']]
    if not selected:
        raise ValueError('Collection policy has no available target objects')
    result['collection_selection'] = {'policy':policy, 'selected_object_ids':selected,
        'excluded_object_ids':[o['object_id'] for o in result['objects'] if not o['collection_enabled']],
        'scope':'Collection targets only; frozen historical scene assets, placements and physics remain intact'}
    result['catalog_sha256'] = digest(result['objects'])
    return result


def asset_path(asset):
    path = Path(asset['path'])
    if path.is_file():
        return path.resolve()
    # Historical transfer paths resolve to the identical project asset subtree.
    for prefix in ('/workspace/asset/', '/home/zhaoyechong/projects/asset/'):
        if str(path).startswith(prefix):
            candidate = canonical_path(ROOT/'asset'/str(path)[len(prefix):])
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(path)


def build(extra=None, seed=20261029):
    bundle = ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1'
    assets = read(bundle/'catalog.json')
    egad = {s['asset']: s['placement'] for s in read(ROOT/'runs/egad49_arx_20261020/scenarios.json')}
    native = {c['object_name']:c['placement'] for c in read(ROOT/'configs/lilo_vla/teacher_pilot20.json')['cases']}
    native['bbq_sauce_bottle'] = read(ROOT/'scripts/curobo_arx/static_pick/reference/historical_reference/registration.json')['scenario']['placement']
    if extra:
        assets += read(extra)
    records, names = [], set()
    for index, asset in enumerate(assets):
        name = asset['name']
        if name in names or not name.replace('_', '').isalnum():
            raise ValueError('Duplicate/unsafe object identifier: '+name)
        names.add(name)
        row = {**asset, 'object_id': name, 'catalog_index': index, 'available': False, 'issues': []}
        try:
            path = asset_path(asset)
            if path.suffix.lower() not in ('.usd', '.usda', '.usdc'):
                raise ValueError('Register converted USD with explicit units/physics before production')
            size = np.asarray(asset['size_m'], dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
                raise ValueError('Invalid or unknown dimensions/units')
            mass = asset.get('mass_kg')
            if mass is None:
                mass = asset.get('measured_mass_kg')
            if mass is not None and not 0 < float(mass) < 20:
                raise ValueError('Missing/invalid registered mass')
            row.update(usd_path=str(path), asset_sha256=sha(path), size_m=size.tolist(), available=True,
                       mass_evidence='catalog' if mass is not None else 'requires composed USD/runtime verification')
            # Exact geometry hash is replaced by composed collider hash in prepare.
            row['geometry_group'] = asset.get('geometry_group') or row['asset_sha256']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row['issues'].append(str(exc))
        row['placement'] = asset.get('placement') or egad.get(name) or native.get(name)
        if row['placement'] is None and row['available']:
            row['placement'] = {'position':[4.048,2.08,.679+float(asset['size_m'][2])/2+.012],
                                'orientation':[1.,0.,0.,0.]}
        row['placement_source'] = ('explicit_asset_manifest' if asset.get('placement') else
            'historical_egad49_scenarios' if name in egad else 'historical_teacher_or_bottle_registration' if name in native else
            'historical_tabletop_batch_rule: .679 + size_z/2 + .012')
        if row['placement']:
            row['placement'].setdefault('orientation', [1., 0., 0., 0.])
        row['renderer'] = asset.get('renderer', 'PathTracing' if name == 'bbq_sauce_bottle' else 'RaytracedLighting')
        group = row.get('geometry_group', name)
        row['split'] = KNOWN_ROLES.get(name) or ('validation' if int(digest([seed, group])[:8], 16)%5 == 0 else 'train')
        row['geometry_dedup_scope'] = 'Exact composed collision geometry hash; near-shape similarity is not established'
        records.append(row)
    return {'version': VERSION, 'seed': seed, 'objects': records, 'initial_count': len(read(bundle/'catalog.json')),
            'catalog_sha256': digest(records), 'source_catalog_sha256': sha(bundle/'catalog.json')}


def historical(renderer):
    if renderer == 'RaytracedLighting':
        return __import__('json').loads(gzip.decompress((ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/setup.json.gz').read_bytes()))
    if renderer == 'PathTracing':
        expanded = ROOT/'runs/research_s02_20261029_04/fixture_pt/historical_reference/setup.json'
        if expanded.is_file():
            return read(expanded)
        path = ROOT/'scripts/curobo_arx/static_pick/reference/historical_reference/setup.json.gz'
        data = __import__('json').loads(gzip.decompress(path.read_bytes()))
        if data['fingerprint']['invariant']['configured']['simulation']['renderer'] != renderer:
            raise ValueError('Historical PathTracing reference missing')
        return data
    raise ValueError('Unknown historical renderer')
