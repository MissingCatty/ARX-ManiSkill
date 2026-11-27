"""Measure registered 512/871 ball unions against current physical colliders."""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np

from arx_project.assets import resource_path
from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.curobo.dual_arm.model import ArmModel
from modules.curobo.fingertip_contact.geometry import collision_meshes
from modules.curobo.sphere_study.profiles import digest
from modules.curobo.sphere_study.fit_metrics import (ConvexUnion, ball_array, directions,
    sphere_boundary, sphere_gap, weighted_stats, support_differences)


SOURCE = ROOT/'tmp/sphere_count_diagnosis/counts5_20261127_01'


def expansion_bound(balls, reference):
    bounds = np.linalg.norm(balls[:, None, :3] - reference[None, :, :3], axis=-1)
    return np.maximum((bounds + balls[:, None, 3] - reference[None, :, 3]).min(axis=1), 0)


def profiles():
    result = {n: read_json(SOURCE/'profiles'/str(n)/'profile.json') for n in [871, 512]}
    for count, profile in result.items():
        if digest(profile['collision_spheres']) != profile['sphere_parameter_sha256']:
            raise ValueError('Diagnostic sphere parameters changed')
        if sum(map(len, profile['collision_spheres'].values())) != count:
            raise ValueError('Diagnostic sphere count changed')
    if result[871]['sphere_parameter_sha256'] != read_json(
            ROOT/'simulation/robots/arx/collision/arx_refined_871.json')['sphere_parameter_sha256']:
        raise ValueError('871 comparison differs from production')
    return result


def prepare(root):
    if root.exists():
        raise RuntimeError('Preserve old records; select a fresh geometry-analysis directory')
    root.mkdir(parents=True)
    asset = resource_path('arx_robot')
    paths = []
    for arm in ['right', 'left']:
        _, files = collision_meshes(asset, arm)
        paths.extend(files)
    inputs = {p: sha256(p) for p in set(paths)}
    for count in [871, 512]:
        p = SOURCE/'profiles'/str(count)/'profile.json';inputs[p] = sha256(p)
    p = SOURCE/'plans/M0001_S1_open_apple/s871_r1/state.json';inputs[p] = sha256(p)
    for name in ['simulation/robots/arx/collision/arx_refined_871.json',
                 'configs/modules.json', 'configs/experiments/active_experiment.json',
                 'configs/policies/fingertip_table.json']:
        p = ROOT/name;inputs[p] = sha256(p)
    sources = [Path(__file__), ROOT/'modules/curobo/sphere_study/fit_metrics.py',
        ROOT/'modules/curobo/fingertip_contact/geometry.py', ROOT/'modules/curobo/dual_arm/model.py']
    write_json(root/'registration.json', {'at': now(), 'task': '512_vs_871_physical_geometry_fit',
        'source_diagnosis': str(SOURCE.relative_to(ROOT)), 'physical_grasps': 0, 'planning_requests': 0,
        'GPU_required': False, 'production_sphere_count': 871, 'automatic_activation': False,
        'sphere_samples_per_ball': 1024, 'finger_refinement_samples_per_ball': 2048,
        'mesh_surface_samples_per_link': 16384, 'support_directions': 32768,
        'surface_inside_tolerance_m': 1e-10, 'boundary_exposure_tolerance_m': 1e-9,
        'fingertip_distal_region_m': .010, 'random_seed': 20261127,
        'numerical_coverage_threshold_m': .00005,
        'judgement_reference': 'Compare added fingertip envelope with current 2mm local table-contact allowance; no automatic replacement',
        'inputs': {str(p): value for p, value in inputs.items()},
        'sources': {str(p): sha256(p) for p in sources}})
    write_json(root/'progress.json', {'at': now(), 'phase': 'registered', 'completed': 0, 'total': 16})


def guard(root):
    reg = read_json(root/'registration.json')
    for group in ['inputs', 'sources']:
        for path, value in reg[group].items():
            if sha256(path) != value:
                raise ValueError('Frozen geometry-analysis input/source changed: '+path)
    return reg


def measure(root, arm, name, pieces, spheres, frame, *, surface_samples, per_sphere):
    directory = root/'links'/arm/name;directory.mkdir(parents=True)
    union = ConvexUnion(pieces)
    mesh_points, mesh_weights = union.sample_surface(surface_samples, 20261127 + int(name[-2:]))
    vertices = union.critical_vertices()
    is_finger = name in ['right_link27', 'right_link28']
    axis = frame[:3, :3].T @ [0., 0., 1.]
    tip = float((vertices @ axis + frame[2, 3]).max())
    tip_cut = tip - .010
    result = {'arm': arm, 'canonical_link': name, 'source_convex_parts': len(pieces),
        'planar_parts': sum(p.equations is None for p in union.parts),
        'surface_sampling_points': len(mesh_points), 'physical_vertex_count': len(vertices),
        'per_sphere_samples': per_sphere, 'is_finger': is_finger, 'profiles': {}}
    arrays = {'mesh_points': mesh_points, 'mesh_weights': mesh_weights, 'physical_vertices': vertices,
        'eef_from_link': frame, 'tip_eef_z_m': np.array(tip)}
    reference = ball_array(spheres[871]['collision_spheres'][name])
    for count in [871, 512]:
        balls = ball_array(spheres[count]['collision_spheres'][name])
        gap = np.maximum(sphere_gap(mesh_points, balls), 0)
        vertex_gap = np.maximum(sphere_gap(vertices, balls), 0)
        points, weights = sphere_boundary(balls, per_sphere)
        outward = union.outside_distance(points)
        item = {'spheres': len(balls), 'mesh_to_spheres_gap': weighted_stats(gap, mesh_weights),
            'sphere_boundary_to_mesh_outward': weighted_stats(outward, weights),
            'maximum_vertex_gap_mm': float(vertex_gap.max()*1000)}
        if is_finger:
            mesh_tip = mesh_points @ axis + frame[2, 3] >= tip_cut
            sphere_tip = points @ axis + frame[2, 3] >= tip_cut
            item['tip_mesh_gap'] = weighted_stats(gap[mesh_tip], mesh_weights[mesh_tip])
            item['tip_sphere_outward'] = weighted_stats(outward[sphere_tip], weights[sphere_tip])
            item['forward_extent_beyond_mesh_mm'] = float(((balls[:, :3]@axis + frame[2, 3] + balls[:, 3]).max()-tip)*1000)
        if count == 512:
            added = np.maximum(sphere_gap(points, reference), 0)
            item['additional_envelope_vs_871'] = weighted_stats(added, weights)
            bounds = expansion_bound(balls, reference)
            item['additional_envelope_proven_upper_bound_mm'] = float(bounds.max()*1000)
            if is_finger:
                item['tip_additional_envelope_vs_871'] = weighted_stats(added[sphere_tip], weights[sphere_tip])
                relevant = balls[:, :3] @ axis + frame[2, 3] + balls[:, 3] >= tip_cut
                item['tip_additional_envelope_upper_bound_mm'] = float(bounds[relevant].max()*1000)
            arrays['additional_512_vs_871_m'] = added
        result['profiles'][str(count)] = item
        arrays.update({f'spheres_{count}': balls, f'boundary_{count}': points,
            f'weights_{count}': weights, f'outward_{count}_m': outward,
            f'mesh_gap_{count}_m': gap, f'vertex_gap_{count}_m': vertex_gap})
    normals = np.concatenate([directions(32768), np.eye(3), -np.eye(3)])
    changes = support_differences(ball_array(spheres[512]['collision_spheres'][name]), reference, normals)
    result['support_plane_displacement_mm'] = {'median': float(np.median(changes)*1000),
        'p95': float(np.percentile(changes,95)*1000), 'sampled_max': float(changes.max()*1000)}
    arrays['support_changes_m'] = changes
    np.savez_compressed(directory/'samples.npz', **arrays)
    result['samples_sha256'] = sha256(directory/'samples.npz')
    write_json(directory/'metrics.json', result)
    return result


def run(root):
    reg = guard(root)
    if (root/'started.json').exists():
        raise RuntimeError('Geometry analysis already started; inspect its outputs')
    write_json(root/'started.json', {'at': now(), 'CPU_only': True})
    spheres = profiles();asset = resource_path('arx_robot')
    state = read_json(SOURCE/'plans/M0001_S1_open_apple/s871_r1/state.json')
    frames = ArmModel().frames(state)
    results = [];cache = {};refinements = []
    began = time.perf_counter()
    for arm in ['right', 'left']:
        meshes, _ = collision_meshes(asset, arm)
        for name in spheres[871]['collision_spheres']:
            guard(root)
            physical = name if arm == 'right' else 'left_link'+str(int(name[-2:])-10)
            transform = np.linalg.inv(frames[arm+'_eef_link']) @ frames[physical]
            pieces = meshes[name]
            # A single convex solid is identical when its unique vertices are
            # identical. Different multi-piece finger decompositions are measured.
            key = tuple(map(tuple, np.unique(pieces[0], axis=0))) if len(pieces) == 1 else None
            write_json(root/'progress.json', {'at': now(), 'phase': 'measuring', 'current': arm+'/'+name,
                'completed': len(results), 'total': 16})
            if key is not None and key in cache:
                item = {**cache[key], 'arm': arm, 'reused_identical_convex_geometry': True}
                target = root/'links'/arm/name;target.mkdir(parents=True)
                write_json(target/'metrics.json', item)
            else:
                item = measure(root, arm, name, pieces, spheres, transform,
                    surface_samples=reg['mesh_surface_samples_per_link'], per_sphere=reg['sphere_samples_per_ball'])
                if key is not None:
                    cache[key] = item
            results.append(item)
            print('GEOMETRY_FIT', len(results), arm, name,
                '871/512 outward95',*[round(item['profiles'][str(c)]['sphere_boundary_to_mesh_outward']['p95_mm'],3) for c in [871,512]],
                'additional95', round(item['profiles']['512']['additional_envelope_vs_871']['p95_mm'],3),flush=True)
    for arm in ['right', 'left']:
        meshes, _ = collision_meshes(asset, arm)
        for name in ['right_link27','right_link28']:
            physical = name if arm == 'right' else 'left_link'+str(int(name[-2:])-10)
            transform = np.linalg.inv(frames[arm+'_eef_link']) @ frames[physical]
            write_json(root/'progress.json', {'at': now(), 'phase': 'finger_refinement', 'current': arm+'/'+name,
                'completed': 16, 'total': 16, 'finger_refinements_completed': len(refinements)})
            item = measure(root/'refined', arm, name, meshes[name], spheres, transform,
                surface_samples=reg['mesh_surface_samples_per_link']*2,
                per_sphere=reg['finger_refinement_samples_per_ball'])
            refinements.append(item)
            print('REFINED_FINGER',arm,name,'tip additional',item['profiles']['512']['tip_additional_envelope_vs_871'],flush=True)
    guard(root)
    write_json(root/'results.json', {'at': now(), 'passed': True, 'CPU_only': True, 'physical_grasps': 0,
        'planning_requests': 0, 'seconds': time.perf_counter()-began, 'links': results,
        'refined_fingers': refinements, 'production_sphere_count': 871, 'default_changed': False})
    write_json(root/'progress.json', {'at': now(), 'phase': 'complete', 'completed': 16, 'total': 16,
        'finger_refinements_completed': len(refinements)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args();root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root)
    else:
        try:
            run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
            raise
