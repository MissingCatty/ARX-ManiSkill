"""Finite, paired local-pose candidate table. Rendering requires 7/7 release.

These are observation candidates, never successful trajectories. The actual
joint FK and same-frame object pose must be used to label admitted captures.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, write, sha
from .advisor36_contract import canonical_release, residual, token

LAYERS = ((0, 5, 32), (5, 15, 32), (15, 30, 32), (30, 40, 32),
          (40, 50, 32), (50, 65, 48), (65, 90, 48))
SEED = 3612026


def ball(rng, radius):
    direction = rng.normal(size=3)
    return direction / np.linalg.norm(direction) * radius * rng.random() ** (1/3)


def generate(object_id, canonical_object):
    digest = hashlib.sha256((str(SEED) + object_id).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    g = np.asarray(canonical_object, dtype=float)
    pairs = []
    for lower, upper, count in LAYERS:
        for index in range(count):
            # Stratify continuously within each layer; exclude its upper end.
            z = (lower + (index + rng.random()) * (upper-lower)/count) / 1000
            for _ in range(1000):
                dp = np.r_[rng.uniform(-.020, .020, 2), z]
                dr = rng.uniform([-10., -10., -15.], [10., 10., 15.])
                if np.linalg.norm(dp) > .004 or np.linalg.norm(dr) > 3: break
            else: raise RuntimeError('Finite REFINE sampling failed')
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
    assert len(pairs) == 512
    rows = []
    for index, (kind, layer, dp, dr) in enumerate(pairs):
        pair_id = f'{object_id}_{index:04d}'
        for sign in (1, -1):
            translation = dp * [sign, sign, 1]
            rotation = dr * sign
            d = np.eye(4); d[:3, 3] = translation
            d[:3, :3] = Rotation.from_rotvec(np.deg2rad(rotation)).as_matrix()
            e = g @ np.linalg.inv(d)
            rows.append({'sample_id': pair_id + ('_p' if sign == 1 else '_m'),
                'pair_id': pair_id, 'pair_member': sign, 'object_id': object_id,
                'category': kind, 'axial_layer_mm': layer,
                'desired_translation_m': translation.tolist(),
                'desired_rotation_degrees': rotation.tolist(),
                'desired_eef_object': e.tolist(), 'canonical_object': g.tolist(),
                'candidate_label': token(e, g, grasp=kind == 'grasp'),
                'label_requires_actual_same_frame_FK': True})
    return rows


def grouped_split(rows, evaluation_only=False):
    """Mirror pairs and nearby poses remain in one split.

    Link candidate observations within2mm AND1degree before splitting. This
    reserves slack for the0.5mm/0.2degree FK fit limits. Never split a connected
    group just to force nominal counts; report actual allocation instead.
    """
    parent = {r['pair_id']: r['pair_id'] for r in rows}
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        a, b = root(a), root(b)
        if a != b: parent[max(a,b)] = min(a,b)
    positions = np.array([r['desired_eef_object'] for r in rows])[:, :3, 3]
    rotations = Rotation.from_matrix(np.array([r['desired_eef_object'] for r in rows])[:, :3, :3])
    for i, j in sorted(cKDTree(positions).query_pairs(.002)):
        if (rotations[i].inv()*rotations[j]).magnitude() <= np.deg2rad(1.):
            union(rows[i]['pair_id'], rows[j]['pair_id'])
    groups = {}
    for row in rows: groups.setdefault(root(row['pair_id']), []).append(row)
    # Largest components first, stable randomized tie-breaking; deficit-based
    # allocation tracks category/layer coverage as well as the total count.
    totals = Counter((r['category'], r['axial_layer_mm']) for r in rows)
    weights = {'train': .75, 'validation': .125, 'test': .125}
    allocated = {s: Counter() for s in weights}
    counts = Counter()
    ordered = sorted(groups.items(), key=lambda kv:(-len(kv[1]), hashlib.sha256(kv[0].encode()).hexdigest()))
    for group, members in ordered:
        categories = Counter((r['category'], r['axial_layer_mm']) for r in members)
        def score(split):
            return sum(n * (totals[k]*weights[split]-allocated[split][k]) /
                       max(1., totals[k]*weights[split]) for k,n in categories.items()) + \
                   .1 * (len(rows)*weights[split]-counts[split]) / (len(rows)*weights[split])
        split = 'unseen_evaluation' if evaluation_only else max(weights, key=score)
        for row in members: row.update(split=split, neighborhood_group=group)
        if not evaluation_only: allocated[split].update(categories)
        counts[split] += len(members)
    return {'split_counts': dict(counts), 'neighborhood_groups': len(groups),
            'largest_group_observations': max(map(len, groups.values())),
            'nearby_grouping_translation_m': .002, 'nearby_grouping_rotation_degrees': 1.,
            'split_counts_are_candidates_not_admitted': True}


def audit(rows):
    pairs = {}
    maxima = np.zeros(5)
    for row in rows: pairs.setdefault(row['pair_id'], []).append(row)
    for pair in pairs.values():
        if len(pair) != 2 or len({r['split'] for r in pair}) != 1: raise ValueError('Split or broken mirror pair')
        a, b = pair
        dp = np.array(a['desired_translation_m']) + np.array(b['desired_translation_m'])
        dr = np.array(a['desired_rotation_degrees']) + np.array(b['desired_rotation_degrees'])
        maxima = np.maximum(maxima, np.abs(np.r_[dp[:2], dr]))
        if a['desired_translation_m'][2] != b['desired_translation_m'][2]: raise ValueError('Pair axial mismatch')
        for r in pair:
            actual = residual(r['desired_eef_object'], r['canonical_object'])
            if not np.allclose(actual[0], r['desired_translation_m'], atol=1e-12, rtol=0):
                raise ValueError('Candidate residual coordinate mismatch')
            if not np.allclose(actual[1], np.deg2rad(r['desired_rotation_degrees']), atol=1e-12, rtol=0):
                raise ValueError('Candidate rotation coordinate mismatch')
    return {'pass': bool(np.max(maxima) < 1e-12), 'candidate_count': len(rows), 'pair_count': len(pairs),
        'maximum_ideal_pair_sum_xy_m_rot_deg': maxima.tolist(),
        'categories': dict(Counter(r['category'] for r in rows)),
        'funnel_axial_layers_mm': dict(Counter(r['axial_layer_mm'] for r in rows if r['category']=='funnel')),
        'actual_captures': 0, 'actual_FK_symmetry_verified': False}


def require_release(seed_rows, object_id):
    selected = [r for r in seed_rows if r['object_id'] == object_id]
    if (not canonical_release(selected) or
        {r['boundary_case'] for r in selected}!={'center','X_minus','X_plus','Y_minus','Y_plus','Z_minus','Z_plus'} or
        not all(r.get('executed') is True for r in selected)):
        raise RuntimeError('Rendering blocked: canonical lacks7/7 physical release')
    return selected


def prepare(campaign, *, canonical_file=None, name='observation_candidates'):
    import json
    top = Path(campaign).resolve(); out = top/name
    canonical_file=Path(canonical_file).resolve() if canonical_file else top/'canonical_candidates.json'
    out.mkdir(exist_ok=False)
    inventory = {r['object_id']: r for r in read(top/'source_inventory.json')}
    checks = []; all_rows = []
    for canonical in read(canonical_file):
        obj = canonical['object_id']; selected = canonical['selected']
        rows = generate(obj, selected['Gcanonical_object'])
        held = inventory[obj]['source_split'] != 'train'
        split = grouped_split(rows, evaluation_only=held)
        checks.append({'object_id':obj, 'evaluation_only':held, **audit(rows), **split})
        all_rows.extend(rows)
    table = out/'candidates.jsonl'
    with table.open('x') as f:
        for row in all_rows: f.write(json.dumps(row, separators=(',',':'))+'\n')
    receipt = {'candidate_count':len(all_rows), 'train_domain_candidates':sum(not c['evaluation_only'] and c['candidate_count'] or 0 for c in checks),
        'unseen_evaluation_candidates':sum(c['evaluation_only'] and c['candidate_count'] or 0 for c in checks),
        'actual_captures':0, 'training_authorized':False, 'render_requires_7_of_7_per_object':True,
        'objects':checks, 'files':{str(table):sha(table)}, 'source_sha256':sha(__file__),
        'canonical_candidates_file':str(canonical_file),'canonical_candidates_sha256':sha(canonical_file),
        'reselected_canonical_requires_new_candidate_registration':True}
    write(out/'registration.json',receipt)
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('campaign')
    result=prepare(p.parse_args().campaign)
    print({k:result[k] for k in ('candidate_count','train_domain_candidates','unseen_evaluation_candidates','actual_captures')})
