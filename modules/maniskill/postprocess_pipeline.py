"""ZeroGrasp native postprocessed proposals plus GraspCL, without geometric priors."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np

import graspcl_pipeline as common


def processed_rows(trial):
    files = sorted((trial / 'network/raw_outputs').glob('*.grasp.npy'))
    if not files:
        raise RuntimeError('Native ZeroGrasp postprocessing output files are missing')
    rows = np.concatenate([np.load(f, allow_pickle=False) for f in files])
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise RuntimeError('Invalid native ZeroGrasp postprocessed rows')
    return rows[np.argsort(-rows[:, 0], kind='stable')], files


def canonical_candidates(rows, camera):
    result = []
    for index, row in enumerate(rows):
        rotation = camera[:3, :3] @ row[4:13].reshape(3, 3)
        center = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3] * rotation[:, 0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation @ common.AXES, center
        result.append({'pose': pose, 'width': float(row[1]),
            'point1': center - row[1] / 2 * pose[:3, 1],
            'point2': center + row[1] / 2 * pose[:3, 1],
            'metadata': {'origin': 'zerograsp', 'postprocessed_row_index': index,
                'source_candidate_stage': 'native collision correction/filtering and NMS',
                'source_network_score': float(row[0]), 'source_network_depth_m': float(row[3]),
                'source_postprocessed_row': row.tolist(), 'finger_flip_added': False}})
    return result


def infer(module, trial, policy):
    import torch
    scene, target, camera = common.clouds(trial)
    rows, files = processed_rows(trial)
    raw = np.load(trial / 'raw_grasps.npy', allow_pickle=False)
    zero = canonical_candidates(rows, camera)
    recalled, counts = [], {}
    for name, recall in [('object', module.grasp_recall_module_object), ('patch', module.grasp_recall_module_patch)]:
        matches = recall.predict(target, scene, debug=False,
            max_matches_per_template=policy['recall_max_matches_per_template'],
            max_total_time_ms=policy['patch_time_ms'])
        counts[name] = len(matches)
        for candidate in matches:
            candidate['metadata']['origin'] = 'demonstration_' + name
        recalled.extend(matches)
    # Novelty is provenance only and never rejects a candidate.
    if len(raw) and recalled:
        common.annotate_novelty(recalled, common.zero_grasps(raw, camera, len(raw)))
    state = common.read(trial / 'state.json')
    maximum_opening = sum(state['joint_limits'][state['joint_names'].index(name)][1]
                          for name in ['right_joint27', 'right_joint28'])
    accepted, rejected = [], []
    for candidate in zero + recalled:
        pose = np.asarray(candidate['pose'])
        reason = None
        if not np.isfinite(pose).all() or not np.isfinite(candidate['width']) or candidate['width'] <= 0:
            reason = 'invalid_candidate_values'
        elif candidate['width'] > maximum_opening + 1e-6:
            reason = 'required_opening_exceeds_unchanged_arx_joint_limits'
        if reason:
            rejected.append({'reason': reason, 'grasp': candidate})
        else:
            accepted.append(candidate)
    encoded = []
    for start in range(0, len(accepted), 64):
        with torch.no_grad():
            encoded.extend(module.encoder.predict(scene, accepted[start:start+64]))
    module.grasp_scoring_module_cl.score_grasps(encoded, estimation_mode='posterior')
    prior = deepcopy(encoded)
    module.grasp_scoring_module_cl.score_grasps(prior, estimation_mode='offline')
    for candidate, offline in zip(encoded, prior):
        candidate['offline_score'] = offline['score']
        candidate.pop('score_thompson', None)
    ranked = sorted(encoded, key=lambda item: -item['score'])
    return {'candidates': ranked, 'recalled_before_filter': recalled, 'rejected': rejected,
        'counts': {'raw_zero_archived_only': len(raw), 'native_postprocessed_zero': len(rows),
                   'zero_scored_input': len(zero), 'recalled': counts, 'scored': len(ranked), 'rejected': len(rejected)},
        'native_postprocessed_source_files': list(map(str, files)),
        'native_postprocessed_source_sha256': {str(f): common.sha(f) for f in files},
        'candidate_protocol': 'zerograsp_native_post_no_custom_geometry_v1',
        'custom_geometry_filters': [], 'zero_top_k_cap': None, 'additional_finger_symmetry_candidates': False,
        'hardware_opening_limit_m': maximum_opening,
        'scene_points': len(scene.points), 'target_points': len(target.points), 'camera': camera}


_original_write = common.write_inference


def write_inference(result, trial, destination):
    _original_write(result, trial, destination)
    if not result['candidates']:
        np.save(destination / 'raw_grasps.npy', np.empty((0, 17)), allow_pickle=False)
    metadata = {'model_name': 'ZeroGrasp post + GraspCL', 'count': len(result['candidates']),
        'candidate_stage': 'Native ZeroGrasp corrected/filtered/NMS proposals + demonstration recall; no custom geometric priors; before ARX planning',
        'score_label': 'GraspCL score', 'physical_attempt': False,
        'candidate_protocol': result['candidate_protocol']}
    common.save(destination / 'network_candidates.json', metadata)


def render(trial, out):
    """Keep an explicit zero-candidate artifact instead of treating it as an error."""
    import grasp_visuals
    rows = np.load(trial / 'raw_grasps.npy', allow_pickle=False)
    if len(rows):
        original = getattr(grasp_visuals, '_native_post_original_render', grasp_visuals.render)
        if original is render:
            raise RuntimeError('Original visualization renderer was not preserved')
        return original(trial, out)
    import matplotlib.pyplot as plt
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(Image.open(trial / 'camera/color.png'))
    ax.set_axis_off()
    ax.set_title('ZeroGrasp post + GraspCL | no feasible candidate values/opening')
    grasp_visuals.figure_save(fig, out / 'grasp_top10_rgb.png')
    params = common.read(trial / 'camera/params.json')
    k = params['intrinsics']
    depth = np.asarray(Image.open(trial / 'camera/depth.png'), dtype=float) * params['depth_scale']
    rgb = np.asarray(Image.open(trial / 'camera/color.png').convert('RGB')) / 255.
    v, u = np.indices(depth.shape)
    points = np.stack([(u-k['ppx'])*depth/k['fx'], (v-k['ppy'])*depth/k['fy'], depth], axis=-1)
    valid = (depth > .12) & (depth < 2.)
    camera = np.asarray(common.read(trial / 'state.json')['camera_world_optical'])
    points = points[valid] @ camera[:3, :3].T + camera[:3, 3]
    colors = rgb[valid]
    step = max(1, len(points)//60000)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*points[::step].T, c=colors[::step], s=.3, depthshade=False)
    ax.set_title('Recorded RGB-D | 0 candidates')
    ax.set_axis_off()
    grasp_visuals.figure_save(fig, out / 'grasp_top10.png')
    metadata = {'displayed_count': 0, 'candidate_count': 0, 'top10': [],
        'candidate_protocol': 'zerograsp_native_post_no_custom_geometry_v1',
        'reason': 'No candidates after native postprocessing/recall and physical opening/value checks',
        'no_synthetic_grasps_drawn': True}
    common.save(out / 'grasp_top10.json', metadata)
    return metadata
