"""Verify saved fit samples, conservative coverage, and native mesh distances."""
import argparse
from pathlib import Path

import numpy as np
import trimesh

from arx_project.assets import resource_path
from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.curobo.sphere_study.fit_metrics import ConvexUnion, weighted_stats
from .analysis import cover_margin
from .geometry_fit import guard, profiles
from .cooked_geometry_fit import loader


def review(root):
    reg = guard(root)
    result = read_json(root/'results.json')
    if not result['passed'] or len(result['links']) != 16 or len(result['refined_fingers']) != 4:
        raise ValueError('Incomplete geometry comparison')
    native = ROOT/reg['native_geometry_source'];load = loader(native)
    source = profiles();coverage = {}
    for name, balls in source[512]['collision_spheres'].items():
        coverage[name] = cover_margin(source[871]['collision_spheres'][name], balls,
            source[512]['coverage'][name]['members'])
    convergence = [];distances = []
    original = {(v['arm'], v['canonical_link']): v for v in result['links']}
    for row in result['refined_fingers']:
        arm, name = row['arm'], row['canonical_link']
        directory = root/'refined/links'/arm/name
        if sha256(directory/'samples.npz') != row['samples_sha256']:
            raise ValueError('Saved refined samples damaged')
        geometry = ConvexUnion(load(resource_path('arx_robot'), arm)[0][name])
        with np.load(directory/'samples.npz', allow_pickle=False) as arrays:
            frame = arrays['eef_from_link'];axis = frame[:3, :3].T @ [0., 0., 1.]
            for count in [871, 512]:
                outward = arrays[f'outward_{count}_m'];points = arrays[f'boundary_{count}']
                idx = np.argsort(outward)[-3:]
                # Brute force every triangle; independent of the R-tree query.
                _, brute, _ = trimesh.proximity.closest_point_naive(geometry.mesh, points[idx])
                error = float(np.max(np.abs(brute-outward[idx])))
                if error > 1e-8:
                    raise ValueError('Independent mesh distance check failed')
                distances.append({'arm': arm, 'link': name, 'count': count, 'points': len(idx),
                                  'maximum_distance_disagreement_m': error})
                tip = points @ axis + frame[2, 3] >= float(arrays['tip_eef_z_m'])-.010
                stats = weighted_stats(outward[tip], arrays[f'weights_{count}'][tip])
                if abs(stats['p95_mm']-row['profiles'][str(count)]['tip_sphere_outward']['p95_mm']) > 1e-10:
                    raise ValueError('Saved fingertip metrics differ from samples')
            added = row['profiles']['512']['tip_additional_envelope_vs_871']
            previous = original[(arm, name)]['profiles']['512']['tip_additional_envelope_vs_871']
            delta = abs(added['p95_mm']-previous['p95_mm'])
            if delta > .1:
                raise ValueError('Fingertip added-envelope P95 did not stabilize with doubled samples')
            if added['sampled_max_mm'] > row['profiles']['512']['tip_additional_envelope_upper_bound_mm'] + 1e-6:
                raise ValueError('Sample exceeds the enclosing-ball distance bound')
            convergence.append({'arm': arm, 'link': name, 'P95_change_mm': delta,
                'sampled_max_change_mm': abs(added['sampled_max_mm']-previous['sampled_max_mm'])})
    guard(root)
    receipt = {'at': now(), 'passed': True, 'record_integrity': True, 'physical_grasps': 0,
        'planning_requests': 0, 'simulation_scenes': 0, 'GPU_used': False,
        'measured_arm_links': 16, 'refined_fingers': 4, 'coverage_margin_m': coverage,
        'additional_fingertip_envelope_convergence': convergence,
        'independent_triangle_distance_checks': distances,
        'native_shape_parts': {arm: {v['canonical_link']: v['parts'] for v in read_json(native/'index.json')['records']
            if v['arm'] == arm and v['canonical_link'] in ['right_link27', 'right_link28']} for arm in ['right','left']},
        'default_geometry_unchanged': True, 'production_spheres': 871,
        'review_source_sha256': sha256(Path(__file__))}
    write_json(root/'independent_review.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args();review(ensure_run_path(args.root))
