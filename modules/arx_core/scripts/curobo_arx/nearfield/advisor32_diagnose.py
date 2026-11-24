"""Read-only advisor32 geometry, camera-distance and stationary-window review.

No inference, GPU initialization, simulation access, grasp dispatch or asset
mutation. Historical transforms are used for offline diagnosis/proposals only.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha
from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh, frozen_pad_center
from training.nearfield_arx.projection import Projector
from .advisor32_rules import StopSample, settle, backproject, points_in_frame, optical_first_advance


OLD = ROOT/'runs/nearfield_closedloop30_20261110_01'
PARENT = ROOT/'runs/nearfield_advisor31_20261110_01'


def angle(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.rad2deg(np.arccos(np.clip(abs(a @ b)/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1))))


def episode(root, trial_id):
    rows = list(Path(root).glob('groups/*/trials/'+trial_id+'/episodes/*'))
    if len(rows) != 1:
        raise ValueError('Expected one retained physical episode: '+trial_id)
    return rows[0]


def camera_review(projector, reg, root, trial, out):
    ep = episode(root, trial['trial_id'])
    asset = next(a for a in reg['catalog']['objects'] if a['object_id'] == trial['object_id'])
    selected = []
    entry = ep/'physical_entry_observation.json'
    if entry.is_file():
        original = read(entry)
        matches = [json.loads(line) for line in (ep/'samples.jsonl').read_text().splitlines()
                   if line.strip() and abs(json.loads(line)['frame_time']-original['frame_time']) < 1e-9]
        if len(matches) != 1:
            raise ValueError('Entry has no unique synchronized raw frame')
        selected.append(('H', entry, matches[0], None))
    for i, request in enumerate(read(ep/'policy_requests.json')):
        path = Path(request['request']); req = read(path)
        selected.append((f'policy_{i+1}', Path(req['observation']), read(req['observation']), req))
    rows = []
    for name, path, row, req in selected:
        depth_path = Path(req['depth']) if req else ep/row['depth_relative']
        depth = np.load(depth_path, allow_pickle=False)
        masks, proof, _ = projector.project(asset['usd_path'], row, depth)
        current = np.asarray(row['snapshot']['link_world']['right_eef_link'])
        camera = np.asarray(row['camera_world_optical'])
        target = masks['target']; values = depth[target & np.isfinite(depth) & (depth > 0)]
        item = {'trial_id': trial['trial_id'], 'stage': name, 'observation': str(path),
                'depth': str(depth_path), 'files': {str(p): sha(p) for p in (path, depth_path)},
                'projection': proof, 'target_depth_valid_pixels': int(len(values)),
                'camera_in_EE': (np.linalg.inv(current) @ camera).tolist(),
                'optical_axis_to_EE_approach_angle_degrees': angle(camera[:3, 2], current[:3, 2])}
        if len(values):
            points = backproject(depth, target, row['intrinsics'])
            ee_points = points_in_frame(points, camera, current)
            pad = frozen_pad_center()
            item.update(target_optical_depth_percentiles_mm=(np.percentile(values, [0, 10, 50, 90, 100])*1000).tolist(),
                        target_median_EE_xyz_mm=(np.median(ee_points, axis=0)*1000).tolist(),
                        target_median_relative_pad_xyz_mm=((np.median(ee_points, axis=0)-pad)*1000).tolist(),
                        literal_15_to_50mm_depth_gate=bool(.015 <= np.median(values) <= .050))
            neutral = 'REFINE 0 0 0 0 0 0'
            item['literal_23_5mm_insertion_no_lateral_or_rotation'] = optical_first_advance(current, camera, points, neutral)
        if req:
            reply_path = path.with_name(req['request_id']+'_reply.json')
            if reply_path.exists():
                reply = read(reply_path); item['raw_prediction'] = reply['prediction']
        rows.append(item)
    write(out/(trial['trial_id']+'_camera.json'), {'object_id': trial['object_id'], 'observations': rows})
    return rows


def obb_review(reg, trial):
    seed = reg['seeds'][trial['seed_id']]
    asset = next(a for a in reg['catalog']['objects'] if a['object_id'] == trial['object_id'])
    mesh, evidence = composed_mesh(asset['usd_path'])
    to_box, extents = trimesh.bounds.oriented_bounds(mesh, angle_digits=1)
    box_object = np.linalg.inv(to_box)
    obj = np.asarray(seed['anchor_snapshot']['objects']['test_object']['world'])
    box_world = obj @ box_object
    star = np.asarray(trial['source_Gstar_world'])
    prior = np.asarray(trial['G0_world'])
    vertical = int(np.argmax(abs(box_world[2, :3])))
    horizontal = [i for i in range(3) if i != vertical]
    short = min(horizontal, key=lambda i: extents[i])
    axes = box_world[:3, :3]
    record = {'object_id': trial['object_id'], 'source_trial_id': trial['trial_id'],
              'mesh_evidence': evidence, 'object_world_source': 'seed anchor_snapshot',
              'object_world': obj.tolist(), 'OBB_world': box_world.tolist(),
              'OBB_extents_mm': (extents*1000).tolist(), 'vertical_axis': vertical,
              'short_horizontal_axis': short,
              'Gstar_closing_to_each_OBB_normal_degrees': [angle(star[:3, 1], axes[:, i]) for i in range(3)],
              'Gstar_closing_to_short_side_normal_degrees': angle(star[:3, 1], axes[:, short]),
              'G0_closing_to_short_side_normal_degrees': angle(prior[:3, 1], axes[:, short]),
              'frozen_closing_axis_EE': [0, 1, 0], 'frozen_pad_center_EE_m': frozen_pad_center().tolist(),
              'scope': 'OBB/nominal finger-axis diagnostic, not measured contact normals or force closure proof',
              'OBB_fit': 'composed visible USD triangle geometry; trimesh oriented_bounds(angle_digits=1)',
              'source_Gstar_world': star.tolist()}
    if trial['object_id'] == 'household_ycb_spam_can':
        closing = axes[:, short].copy()
        if closing @ star[:3, 1] < 0:
            closing *= -1
        approach = star[:3, 2]-closing*(closing @ star[:3, 2])
        if np.linalg.norm(approach) < 1e-6:
            raise ValueError('Source approach parallel to canonical closure normal')
        approach /= np.linalg.norm(approach)
        across = np.cross(closing, approach); across /= np.linalg.norm(across)
        rotation = np.column_stack([across, closing, approach])
        up = axes[:, vertical].copy()
        if up[2] < 0:
            up *= -1
        # A declared geometric proposal, not optimized against the outcome.
        pad_world = box_world[:3, 3]+.20*extents[vertical]*up
        canonical = np.eye(4); canonical[:3, :3] = rotation
        canonical[:3, 3] = pad_world-rotation @ frozen_pad_center()
        record['canonical_proposal'] = {'Gstar_canon_world': canonical.tolist(),
            'closing_normal_error_degrees': angle(closing, axes[:, short]),
            'pad_height_fraction_from_OBB_bottom': .70,
            'pad_target_world': pad_world.tolist(), 'required_nominal_width_m': float(extents[short]),
            'approach_rule': 'Project the source approach onto the plane perpendicular to the short-side normal',
            'model_IK_checked': False, 'collision_checked': False, 'executed': False,
            'object_or_robot_configuration_changed': False}
    return record


def stop_review():
    ident = 'a31_v03_r1_household_hope_milk_carton_minus15_fresh'
    ep = episode(PARENT, ident)
    rows = [json.loads(s) for s in (ep/'samples.jsonl').read_text().splitlines() if s.strip()]
    samples = []
    for row in rows[-50:]:
        snap = row['snapshot']; indices = [snap['joint_names'].index(n) for n in snap['right_arm']]
        samples.append(StopSample(float(snap['sim_time']), tuple(np.asarray(snap['link_world']['right_eef_link'])[:3, 3]),
            tuple(snap['joint_positions'][i] for i in indices), tuple(snap['joint_velocities'][i] for i in indices)))
    result = settle(samples, active=False, fault=None, elapsed_wall_s=2.)
    return {'episode': str(ep), 'files': {str(ep/'samples.jsonl'): sha(ep/'samples.jsonl')},
            'replayed_window': result, 'original_900s_timeout_retained': True,
            'physical_runtime_rerun': False,
            'sampling_limitation': '5Hz recorded position differences do not prove absence of higher-frequency joint oscillation',
            'velocity_API_false_value_proven': False}


def run(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Do not overwrite a completed diagnostic')
    out.mkdir(parents=True)
    reg = read(OLD/'registration.json'); previous = read(PARENT/'registration.json')
    projector = Projector(reg['learned_policy']['robot_usd'])
    normal = [t for t in reg['seed_trials'] if t['method'] == 'D' and t['condition'] == 'normal']
    geometry, views = [], []
    for t in normal:
        geometry.append(obb_review(reg, t)); write(out/'geometry.json', geometry)
        views += camera_review(projector, reg, OLD, t, out)
    fresh = next(t for t in previous['seed_trials'] if t['trial_id']=='a31_v01_r1_household_hope_milk_carton_plus15_fresh')
    views += camera_review(projector, previous, PARENT, fresh, out)
    stopped = stop_review(); write(out/'stationary_window_review.json', stopped)
    result = {'created_unix': time.time(), 'physical_trials': 0, 'model_inferences': 0,
        'objects_reviewed': len(geometry), 'views_reviewed': len(views), 'geometry': geometry,
        'camera_views': views, 'stationary_review': stopped,
        'original_records_modified': False, 'runtime_rules_installed': False,
        'literal_depth_contract_counterexample': [v for v in views if v['trial_id']==fresh['trial_id'] and v['stage']=='policy_3'],
        'dependencies': {str(p): sha(p) for p in [OLD/'registration.json', PARENT/'registration.json',
            PARENT/'STOP', ROOT/'research_plan/chat/gemini回复/gemini回复32.md', Path(__file__)]}}
    write(out/'review.json', result)
    return {'out': str(out), 'objects': len(geometry), 'views': len(views), 'physical_trials': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    print(json.dumps(run(parser.parse_args().out), ensure_ascii=False))
