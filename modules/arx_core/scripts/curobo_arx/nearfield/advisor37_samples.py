"""Bounded tapered candidates and leakage-safe replacement pairs for ARX.

Sampling is offline geometry, not a grasp or image-admission certificate.
Every replacement has a fresh identity. Previous proposals, including rejected
ones, continue to reserve their split neighborhoods.
"""
from collections import Counter, defaultdict
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from .advisor36_contract import token
from .advisor36_samples import LAYERS, ball, grouped_split, audit

SEED = 3712026
MAX_WAVES = 4
PLANNED = {**{f'funnel/{lo:02d}_{hi:02d}': count * 2
              for lo, hi, count in LAYERS},
           'boundary_position/core_boundary': 128,
           'boundary_rotation/core_boundary': 128,
           'grasp/core': 256}
MINIMUM = {key: count * 3 // 4 for key, count in PLANNED.items()}


def stratum(row):
    return row['category'] + '/' + row['axial_layer_mm']


def envelope(z_m):
    """C1 smooth convergence to the unchanged 4 mm/3 degree core boundaries.

Bounds are componentwise sampling bounds, not a redefinition of the core:
GRASP still requires Euclidean translation <=4 mm AND rotation angle <=3 deg.
"""
    if not np.isfinite(z_m) or not 0 <= z_m <= .09:
        raise ValueError('Funnel Z outside registered [0,90] mm')
    t = min(z_m / .035, 1.)
    s = t * t * (3 - 2 * t)
    return .004 + .016 * s, np.array([3., 3., 3.]) + np.array([7., 7., 12.]) * s


def generate_wave(object_id, canonical_object, wave):
    if not isinstance(wave, int) or not 0 <= wave < MAX_WAVES:
        raise ValueError('Finite four-wave proposal budget exhausted')
    seed = hashlib.sha256(f'{SEED}/{object_id}/{wave}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], 'little'))
    g = np.asarray(canonical_object, dtype=float)
    pairs = []
    for lower, upper, count in LAYERS:
        for index in range(count):
            z = (lower + (index + rng.random()) * (upper - lower) / count) / 1000
            xy, angles = envelope(z)
            for _ in range(1000):
                dp = np.r_[rng.uniform(-xy, xy, 2), z]
                dr = rng.uniform(-angles, angles)
                if np.linalg.norm(dp) > .004 or np.linalg.norm(dr) > 3:
                    break
            else:
                raise RuntimeError('Finite tapered REFINE proposal budget exhausted')
            pairs.append(('funnel', f'{lower:02d}_{upper:02d}', dp, dr))
    for kind in ('boundary_position', 'boundary_rotation'):
        for _ in range(64):
            dp, dr = ball(rng, .004), ball(rng, 3.)
            if kind == 'boundary_position':
                dp *= rng.uniform(.0040001, .008) / np.linalg.norm(dp)
            else:
                dr *= rng.uniform(3.0001, 6.) / np.linalg.norm(dr)
            pairs.append((kind, 'core_boundary', dp, dr))
    for _ in range(128):
        pairs.append(('grasp', 'core', ball(rng, .004), ball(rng, 3.)))
    rows = []
    for index, (kind, layer, dp, dr) in enumerate(pairs):
        pair_id = f'a37_{object_id}_w{wave:02d}_{index:04d}'
        for sign in (1, -1):
            p, r = dp * [sign, sign, 1], dr * sign
            d = np.eye(4)
            d[:3, 3] = p
            d[:3, :3] = Rotation.from_rotvec(np.deg2rad(r)).as_matrix()
            e = g @ np.linalg.inv(d)
            rows.append({'sample_id': pair_id + ('_p' if sign == 1 else '_m'),
                'pair_id': pair_id, 'pair_member': sign, 'object_id': object_id,
                'category': kind, 'axial_layer_mm': layer, 'wave': wave,
                'desired_translation_m': p.tolist(), 'desired_rotation_degrees': r.tolist(),
                'desired_eef_object': e.tolist(), 'canonical_object': g.tolist(),
                'candidate_label': token(e, g, grasp=kind == 'grasp'),
                'label_requires_actual_same_frame_FK': True,
                'sampling_contract': 'advisor37_smoothstep_taper_v1'})
    return rows


def _neighbors(members, previous):
    if not previous:
        return []
    poses = np.asarray([r['desired_eef_object'] for r in previous])
    hit = np.zeros(len(previous), dtype=bool)
    for member in members:
        m = np.asarray(member['desired_eef_object'])
        indices = np.flatnonzero(np.linalg.norm(poses[:, :3, 3] - m[:3, 3], axis=1) <= .002)
        if len(indices):
            angles = Rotation.from_matrix(np.einsum('ij,njk->nik', m[:3, :3].T,
                                                    poses[indices, :3, :3])).magnitude()
            hit[indices[angles <= np.deg2rad(1.)]] = True
    return [previous[i] for i in np.flatnonzero(hit)]


def assign_replacement_splits(rows, previous, evaluation_only=False):
    """Never move old splits or bridge train/validation/test neighborhoods.

The entire finite proposal pool is assigned before any rendering/outcomes.
Cross-split bridge proposals are retained as excluded records, not reassigned.
"""
    weights = {'train': .75, 'validation': .125, 'test': .125}
    reserved = [r for r in previous if not r.get('proposal_exclusion')]
    allocated = {s: Counter(stratum(r) for r in reserved if r['split'] == s) for s in weights}
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row)
    for pair, members in pairs.items():
        neighbors = _neighbors(members, reserved)
        splits = {r['split'] for r in neighbors}
        if len(splits) > 1:
            for row in members:
                row.update(split='excluded', neighborhood_group=pair,
                           proposal_exclusion='replacement_would_bridge_frozen_splits')
            continue
        key = stratum(members[0])
        total = sum(v[key] for v in allocated.values()) + 2
        split = ('unseen_evaluation' if evaluation_only else next(iter(splits)) if splits
                 else max(weights, key=lambda s: total * weights[s] - allocated[s][key]))
        anchors = sorted({r['neighborhood_group'] for r in neighbors})
        for row in members:
            row.update(split=split, neighborhood_group=pair,
                       same_split_neighbor_groups=anchors)
        if not evaluation_only:
            allocated[split][key] += 2
        reserved.extend(members)
    return {'proposal_excluded': sum(bool(r.get('proposal_exclusion')) for r in rows),
            'split_counts': dict(Counter(r['split'] for r in rows))}


def generate_pool(object_id, canonical_object, evaluation_only=False):
    rows, checks = [], []
    for wave in range(MAX_WAVES):
        new = generate_wave(object_id, canonical_object, wave)
        split = (grouped_split(new, evaluation_only=evaluation_only) if wave == 0
                 else assign_replacement_splits(new, rows, evaluation_only))
        checks.append({'wave': wave, **audit(new), **split})
        rows.extend(new)
    return rows, {'waves': checks, 'planned_observations': 1024,
                  'maximum_fresh_proposals': 4096, 'training_started': False,
                  'all_splits_assigned_before_observation_outcomes': True}


def quota_status(records):
    """Only independent admission of both members can satisfy a quota."""
    pairs, seen = defaultdict(list), set()
    for row in records:
        if row['sample_id'] in seen:
            raise ValueError('Repeated observation in quota ledger')
        seen.add(row['sample_id'])
        pairs[row['pair_id']].append(row)
    counts = Counter()
    for members in pairs.values():
        if not any(r.get('admitted') for r in members):
            continue
        if (len(members) != 2 or not all(r.get('admitted') and r.get('capture_admitted')
                                        and r.get('pair_check', {}).get('pass') for r in members)
            or {r['candidate']['pair_member'] for r in members} != {-1, 1}
            or len({stratum(r['candidate']) for r in members}) != 1
            or len({r['split'] for r in members}) != 1):
            raise ValueError('Broken independently admitted pair in quota ledger')
        key = stratum(members[0]['candidate'])
        if key not in PLANNED:
            raise ValueError('Unregistered stratum in quota ledger')
        counts[key] += 2
    rows = [{'stratum': key, 'planned': plan, 'minimum': MINIMUM[key],
             'admitted': counts[key], 'complete_pairs': counts[key] // 2,
             'minimum_met': counts[key] >= MINIMUM[key],
             'remaining_to_plan': max(0, plan - counts[key])}
            for key, plan in PLANNED.items()]
    return {'minimum_met': all(r['minimum_met'] for r in rows),
            'planned_counts_met': all(r['remaining_to_plan'] == 0 for r in rows),
            'admitted_observations': sum(counts.values()), 'strata': rows,
            'boundary_subtypes_each_require_75_percent': True,
            'training_authorization_implied': False}


def select_wave(pool, wave, records, started_ids=()):
    status = quota_status(records)
    remaining = {r['stratum']: r['remaining_to_plan'] for r in status['strata']}
    used = set(started_ids) | {r['sample_id'] for r in records}
    touched_pairs = {r['pair_id'] for r in pool if r['sample_id'] in used}
    pairs = defaultdict(list)
    for row in pool:
        if row['wave'] == wave:
            pairs[row['pair_id']].append(row)
    selected = []
    for pair, members in pairs.items():
        if (pair in touched_pairs or len(members) != 2 or
            any(r.get('proposal_exclusion') for r in members)):
            continue
        key = stratum(members[0])
        if remaining[key] >= 2:
            selected.extend(r['sample_id'] for r in members)
            remaining[key] -= 2
    return selected
