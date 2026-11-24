"""Inspect saved grasp geometry and contact traces without loading simulation."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT = PROJECT / 'experiments/diverse_grasp_benchmark/runs/geometry200_fixed452_20261122_01'


def read(p):
    return json.loads(p.read_text())


def surface_mesh(record):
    parts = []
    for item in record['visuals']:
        mesh = trimesh.load(item['file'], process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh = mesh.copy()
        mesh.vertices *= np.asarray(item['scale'])
        mesh.apply_transform(item['transform'])
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


def observed(directory, state):
    depth = np.load(directory / 'camera/depth_m.npy')
    mask = np.asarray(Image.open(directory / 'camera/mask.png')) > 0
    k = np.asarray(state['intrinsics'])
    v, u = np.indices(depth.shape)
    keep = mask & np.isfinite(depth) & (depth > 0)
    pc = np.stack([(u-k[0, 2])*depth/k[0, 0],
                   (v-k[1, 2])*depth/k[1, 1], depth], -1)[keep]
    t = np.asarray(state['camera_world_optical'])
    return pc @ t[:3, :3].T + t[:3, 3]


def local(points, transform):
    return (points - transform[:3, 3]) @ transform[:3, :3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results_20261123')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = load_object_catalog(args.run / 'object_catalog.json')['models']
    descriptor = read(args.run / 'gripper_descriptor.json')
    # Generator +X is ARX +Y; generator +Y is ARX -X.
    generator_from_eef = np.asarray(descriptor['T_eef_generator'])[:3, :3]
    params = descriptor['parameters']
    ext, off = np.asarray(params['extents_open']), np.asarray(params['offset_open'])
    depth_tip = params['fingertip_depth']
    results, hashes, mesh_cache = [], {}, {}

    def corridor(points, eef):
        gen = local(points, eef) @ generator_from_eef
        return np.all(np.abs(gen-off) <= ext/2, axis=1)

    for claim in sorted((args.run / 'claims').glob('*.json')):
        c = read(claim)
        if c['state'] != 'complete':
            continue
        d = args.run / 'trials' / claim.stem
        result = read(d / 'result.json')
        if not result.get('physical_attempt'):
            continue
        digest = hashlib.sha256((d / 'result.json').read_bytes()).hexdigest()
        assert digest == c['result_sha256']
        hashes[str(d / 'result.json')] = digest
        state = read(d / 'state.json')
        entry = read(d / 'entry.json')
        selected = read(d / 'plan_result.json')['selected']
        eef = np.asarray(selected['grasp_world'])
        t_obj = np.asarray(state['object_world'])
        record = catalog[entry['model_id']]
        if entry['model_id'] not in mesh_cache:
            mesh = surface_mesh(record)
            np.random.seed(0)
            sample, _ = trimesh.sample.sample_surface(mesh, 15000)
            mesh_cache[entry['model_id']] = (mesh, sample)
        mesh, sample = mesh_cache[entry['model_id']]
        surface = sample @ t_obj[:3, :3].T + t_obj[:3, 3]
        vertices = np.asarray(mesh.vertices) @ t_obj[:3, :3].T + t_obj[:3, 3]
        obs = observed(d, state)
        completed = np.load(d / 'completed_points_world.npy')
        surface_tree = cKDTree(surface)
        obs_to_surface = surface_tree.query(obs)[0]
        recon_to_surface = surface_tree.query(completed)[0]
        obs_to_recon = cKDTree(completed).query(obs)[0]
        pad_center = eef[:3, 3] + eef[:3, :3] @ (generator_from_eef @ off)
        pad_to_surface = float(surface_tree.query(pad_center)[0])
        pts_in_corridor = int(corridor(surface, eef).sum())
        visible_in_corridor = int(corridor(obs, eef).sum())
        reconstructed_in_corridor = int(corridor(completed, eef).sum())

        traces = [json.loads(s) for s in (d / 'execution.jsonl').open()]
        xyz = np.asarray([r['object_pose'][:3] for r in traces])
        dt = np.diff([r['tick'] for r in traces]) / 120.
        assert np.all(dt > 0)
        speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
        close_rows = [i for i, r in enumerate(traces) if r['stage'] == 'close']
        contacts = [(i, r) for i, r in enumerate(traces) if r['contacts']]
        first = contacts[0] if contacts else None
        max_speed_index = int(np.argmax(speed)) + 1
        max_impulse = max(sum(v['impulse_ns'] for v in r['contacts']) for r in traces)
        target_contacts = Counter()
        for r in traces:
            for contact in r['contacts']:
                if contact.get('allowed'):
                    target_contacts[contact['a'] if 'right_link' in contact['a'] else contact['b']] += 1
        info = read(d / 'graspgenx_raw/inference.json')
        obb = info.get('obb')
        stages = read(d / 'stages.json')
        rows = {
            'id': entry['id'], 'model_id': entry['model_id'], 'reason': result['reason'],
            'branch': selected['candidate_metadata']['source_branch'],
            'selected_score': selected['score'],
            'mass_kg': state['scene_meshes']['target']['mass_kg'],
            'collision_parts': len(record['collisions']),
            'densities': sorted(set(v['density'] for v in record['collisions'])),
            'friction': sorted(set(v['friction'] for v in record['collisions'])),
            'object_world_bounds_m': [vertices.min(0).tolist(), vertices.max(0).tolist()],
            'pad_center_world_m': pad_center.tolist(),
            'pad_center_to_surface_mm': pad_to_surface * 1000,
            'target_surface_samples_in_pad_corridor': pts_in_corridor,
            'observed_points_in_pad_corridor': visible_in_corridor,
            'reconstructed_points_in_pad_corridor': reconstructed_in_corridor,
            'surface_samples': len(surface),
            'observed_surface_error_median_mm': float(np.median(obs_to_surface)*1000),
            'reconstruction_to_surface_median_mm': float(np.median(recon_to_surface)*1000),
            'observed_to_reconstruction_median_mm': float(np.median(obs_to_recon)*1000),
            'max_object_speed_mps': float(speed[max_speed_index-1]),
            'max_speed_tick': traces[max_speed_index]['tick'],
            'max_speed_stage': traces[max_speed_index]['stage'],
            'max_step_impulse_sum_ns': max_impulse,
            'object_max_displacement_m': float(np.linalg.norm(xyz-xyz[0], axis=1).max()),
            'contact_ticks_per_finger': dict(target_contacts),
            'first_contact': None if first is None else {
                'tick': first[1]['tick'], 'stage': first[1]['stage'],
                'position_m': first[1]['object_pose'][:3],
                'events': first[1]['contacts'],
            },
            'object_shift_before_close_mm': None if not close_rows else float(np.linalg.norm(xyz[close_rows[0]]-xyz[0])*1000),
            'approach_eef_error_mm': stages.get('approach', {}).get('tool_position_error_m', float('nan')) * 1000,
        }
        if obb and np.allclose(eef[:3, 2], [0, 0, -1], atol=1e-5):
            top = obb['center'][2] + obb['half_extent'][2]
            rows.update(obb_tip_offset_mm=float((eef[2, 3]-depth_tip-top)*1000),
                        fingertip_above_mesh_top_mm=float((eef[2, 3]-depth_tip-vertices[:, 2].max())*1000),
                        pad_center_above_mesh_top_mm=float((pad_center[2]-vertices[:, 2].max())*1000))
        results.append(rows)
    summary = {
        'run': str(args.run), 'physical_attempts': len(results), 'simulation_started': False,
        'reasons': dict(Counter(r['reason'] for r in results)),
        'selected_branches': dict(Counter(r['branch'] for r in results)),
        'no_surface_sample_in_pad_corridor': sum(r['target_surface_samples_in_pad_corridor'] == 0 for r in results),
        'speed_exceeds_1_mps': sum(r['max_object_speed_mps'] > 1 for r in results),
        'speed_exceeds_5_mps': sum(r['max_object_speed_mps'] > 5 for r in results),
        'median_observed_surface_error_mm': float(np.median([r['observed_surface_error_median_mm'] for r in results])),
        'median_reconstruction_surface_error_mm': float(np.median([r['reconstruction_to_surface_median_mm'] for r in results])),
        'obb_zero_tip_offset': sum(abs(r.get('obb_tip_offset_mm', float('inf'))) < .05 for r in results if r['branch'] == 'obb'),
        'input_result_hashes': hashes,
    }
    (args.output / 'cases.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_result_hashes'}, indent=2), flush=True)
    for row in results:
        if row['id'] in ['D0002_S1_open_bottled_water', 'D0003_S1_open_mushroom', 'D0042_S2_edge_lemon']:
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
