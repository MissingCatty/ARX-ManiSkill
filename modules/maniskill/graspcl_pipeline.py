"""ARX RGB-D / ZeroGrasp / GraspCL adapter, separate from the running baseline.

Run with tmp/arx-graspcl-env/bin/python. No simulator or robot configuration is
changed here. Offline recall and score results are not physical success labels.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import traceback
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'maniskill/outputs/arx_baseline_20261116_parallel_01'
ARCHIVE = ROOT / 'research_dependencies/graspcl_assets/corl_adapted_model_rw.zip'
UPSTREAM = ROOT / 'research_dependencies/graspcl_model_public'
# Existing ARX mapping: GraspNet X approach -> ARX EEF Z approach.
AXES = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
TCP_OFFSET = .040


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(plain(data), indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read(path):
    from arx_project.layout import read_json
    return read_json(path)


def seed():
    import open3d as o3d
    import torch
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    o3d.utility.random.seed(17)


def inputs(trial, require_all=True):
    names = ['state.json', 'camera/color.png', 'camera/depth.png',
             'camera/mask.png', 'camera/params.json', 'raw_grasps.npy']
    return {str(trial / n): sha(trial / n) for n in names if require_all or (trial / n).is_file()}


def prepare(out):
    """Freeze a disjoint, retrospective scene split before any model prediction."""
    out.mkdir(parents=True, exist_ok=False)
    source = read(BASELINE / 'registration.json')
    train, completed = [], []
    for i, entry in enumerate(source['schedule']):
        trial = BASELINE / 'trials' / entry['id']
        if not (trial / 'result.json').exists():
            continue
        result = read(trial / 'result.json')
        item = dict(entry, schedule_index=i + 1, trial=str(trial),
                    original_success=bool(result['success']),
                    original_reason=result['reason'],
                    original_physical_attempt=bool(result['physical_attempt']))
        if i < 40:
            evaluation = read(trial / 'independent_evaluation.json')
            if result['success'] and evaluation.get('independently_successful') and evaluation['passed']:
                item['label'], item['use'] = 1, 'demonstration_and_feedback'
            elif (result['physical_attempt'] and
                  result['reason'] in ['no_bilateral_target_contact', 'insufficient_hold_height'] and
                  result.get('stages', {}).get('close', {}).get('settled') and
                  not read(trial / 'forbidden_contacts.json')):
                item['label'], item['use'] = 0, 'feedback_only'
            else:
                item['label'], item['use'] = None, 'excluded_no_clean_completed_grasp_label'
            item['input_sha256'] = inputs(trial, require_all=item['label'] is not None)
            item['evidence_sha256'] = {str(trial / f): sha(trial / f) for f in
                ['result.json', 'independent_evaluation.json', 'stages.json'] if (trial / f).exists()}
            if item['label'] is not None:
                item['evidence_sha256'][str(trial / 'plan_result.json')] = sha(trial / 'plan_result.json')
                item['evidence_sha256'][str(trial / 'execution.jsonl')] = sha(trial / 'execution.jsonl')
                item['evidence_sha256'][str(trial / 'forbidden_contacts.json')] = sha(trial / 'forbidden_contacts.json')
            train.append(item)
        else:
            completed.append(item)
    if len(train) != 40:
        raise RuntimeError('All first 40 scheduled trials must be terminal before a split')
    queries = []
    for layout in source['layouts']:
        for wanted in [True, False]:
            matches = [e for e in completed if e['layout'] == layout and e['original_success'] == wanted
                       and len(inputs(Path(e['trial']), require_all=False)) == 6]
            if not matches:
                raise RuntimeError(f'Missing held-out diagnostic stratum {layout} {wanted}')
            item = matches[0]
            item['input_sha256'] = inputs(Path(item['trial']))
            queries.append(item)
    # Only encoder and offline scorer components from the author's adapted archive.
    # Author online/recall payloads are never extracted or used as ARX examples.
    components = ['config.json', 'encoder/best.ckpt', 'score_cl/config.json', 'score_cl/offline_data.npz']
    with zipfile.ZipFile(ARCHIVE) as archive:
        component_hashes = {f: hashlib.sha256(archive.read('bootstrapped/' + f)).hexdigest() for f in components}
    sources = dict(source['source_hashes'])
    sources.update({str(Path(__file__).resolve()): sha(Path(__file__)), str(ARCHIVE): sha(ARCHIVE)})
    sources.update({str(f.resolve()): sha(f) for f in sorted((UPSTREAM / 'grasp_learning_cl').rglob('*.py'))})
    registration = {
        'created_at': datetime.now().astimezone().isoformat(),
        'chain': 'right_wrist RGB-D -> ZeroGrasp + GraspCL object/patch demonstration recall -> GraspCL posterior mean -> unchanged ARX cuRobo/execution',
        'phase': 'offline integration and diagnosis; physical validation separately registered',
        'physical_attempt_budget': 0,
        'baseline_registration': str(BASELINE / 'registration.json'),
        'baseline_registration_sha256': sha(BASELINE / 'registration.json'),
        'source_hashes': sources, 'archive_components': component_hashes,
        'author_online_and_recall_data_used': False,
        'training_trials': train, 'query_trials': queries,
        'split': 'First 40 frozen schedule slots for memory; later completed scenes, first success and failure per layout. Retrospective outcome-stratified diagnostics, not unbiased performance estimation; same object models may occur on both sides.',
        'feedback': 'Clean executed closure failure or completed hold failure = 0; independently verified full grasp = 1. Planning, convergence and infrastructure failures excluded. Automatic positives use weight 3, without human-override weighting.',
        'pose_frame': 'World Z up. GraspCL pose x perpendicular, y closing, z approach. Commanded executed ARX EEF grasp + 0.040m along EEF z gives nominal contact center. Measured pre-close pose and errors stored separately. No post-lift pose is attached to the pregrasp image.',
        'width': 'Successful demonstrations retain commanded 0.088m opening; nominal point1/2 are not measured contacts. Measured closed finger coordinates and source network width preserved as provenance. Encoder does not explicitly condition on width.',
        'perception': {'target_roi': 'existing saved instance mask; no GT object pose/center in recall/scoring',
                       'scene_depth_m': [.12, 2.], 'voxel_m': .002,
                       'normals_radius_m': .015, 'normals_max_nn': 30,
                       'normal_orientation': 'towards original optical camera position',
                       'encoder_cloud': 'full observed scene including table, per checkpoint with_table setting'},
        'candidate_policy': {'zero_top_k': 256, 'include_gripper_flip': True,
                             'recall_max_matches_per_template': 2, 'patch_time_ms': 2000,
                             'target_support_distance_m': .055, 'max_width_m': .088,
                             'novel_position_threshold_m': .005, 'novel_rotation_threshold_deg': 10,
                             'novelty': 'SE(3) nearest-neighbor against ALL ZeroGrasp rows with finger symmetry; diagnostic only, not acceptance'},
        'selection': 'GraspCL posterior mean, deterministic stable descending; no Thompson sampling, no claim of calibrated ARX probability',
        'robot_collision': 'Do not invoke GraspCL Panda sampler or Panda collision proxy. Only original ARX cuRobo and original physical evaluator can qualify execution.',
        'device': 'CPU, four threads; no competing GPU model/planner instance',
        'observation': 'Saved historical wrist frames only; original GT bbox-based observation policy remains a disclosed baseline limitation',
    }
    save(out / 'registration.json', registration)
    model = out / 'model'
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in components:
            path = model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read('bootstrapped/' + name))
    shutil.copy2(BASELINE / 'storage_policy.json', out / 'storage_policy.json')
    print(json.dumps({'prepared': str(out), 'demonstrations': sum(e['label'] == 1 for e in train),
                      'negative_feedback': sum(e['label'] == 0 for e in train), 'queries': len(queries)}), flush=True)


def verify(out):
    registration = read(out / 'registration.json')
    if sha(registration['baseline_registration']) != registration['baseline_registration_sha256']:
        raise RuntimeError('Source registration changed')
    for path, digest in registration['source_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Pinned source changed: ' + path)
    for entry in registration['training_trials'] + registration['query_trials']:
        for path, digest in {**entry['input_sha256'], **entry.get('evidence_sha256', {})}.items():
            if sha(path) != digest:
                raise RuntimeError('Pinned evidence changed: ' + path)
    return registration


def clouds(trial):
    """Only image, depth, mask, K and measured optical extrinsics enter features."""
    import open3d as o3d
    params = read(trial / 'camera/params.json')
    k = params['intrinsics']
    camera = np.asarray(read(trial / 'state.json')['camera_world_optical'])
    depth = np.asarray(Image.open(trial / 'camera/depth.png'), dtype=float) * params['depth_scale']
    rgb = np.asarray(Image.open(trial / 'camera/color.png').convert('RGB')) / 255.
    mask = np.asarray(Image.open(trial / 'camera/mask.png')) > 0
    v, u = np.indices(depth.shape)
    pts = np.stack([(u-k['ppx'])*depth/k['fx'], (v-k['ppy'])*depth/k['fy'], depth], axis=-1)
    valid = np.isfinite(pts).all(-1) & (depth > .12) & (depth < 2.)
    world = pts @ camera[:3, :3].T + camera[:3, 3]
    result = []
    for keep in [valid, valid & mask]:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(world[keep])
        cloud.colors = o3d.utility.Vector3dVector(rgb[keep])
        cloud = cloud.voxel_down_sample(.002)
        if len(cloud.points) < 40:
            raise RuntimeError('Insufficient observed RGB-D points')
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.015, max_nn=30))
        cloud.orient_normals_towards_camera_location(camera[:3, 3])
        result.append(cloud)
    return *result, camera


def contact_pose(eef):
    pose = np.array(eef, dtype=float, copy=True)
    pose[:3, 3] += TCP_OFFSET * pose[:3, 2]
    return pose


def commanded_grasp(trial, entry):
    plan = read(trial / 'plan_result.json')['selected']
    pose = contact_pose(plan['grasp_world'])
    state = read(trial / 'state.json')
    last, last_close = None, None
    with (trial / 'execution.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['stage'] == 'approach':
                last = row
            elif row['stage'] == 'close':
                last_close = row
            elif last_close is not None:
                break
    if last is None:
        raise RuntimeError('No measured approach trace for feedback action')
    import sys
    sys.path.insert(0, str(ROOT / 'maniskill/arx_pipeline'))
    from scene_tools import ArmFK
    names = state['joint_names']
    q = [last['qpos'][names.index(f'right_joint{i}')] for i in range(21, 27)]
    # body4 can move slightly during grasp; its translation follows lift joint4.
    body = np.array(state['links']['body4'])
    body[2, 3] += last['qpos'][names.index('joint4')] - state['qpos'][names.index('joint4')]
    measured = body @ ArmFK().fk(q)
    command = np.array(plan['grasp_world'])
    error = {'position_m': float(np.linalg.norm(measured[:3, 3]-command[:3, 3])),
             'rotation_rad': float(Rotation.from_matrix(measured[:3, :3] @ command[:3, :3].T).magnitude())}
    if error['position_m'] > .005 or error['rotation_rad'] > .03:
        raise RuntimeError('Commanded grasp not accurately reached: ' + str(error))
    width = .088
    return {'pose': pose, 'point1': pose[:3, 3] - width/2 * pose[:3, 1],
            'point2': pose[:3, 3] + width/2 * pose[:3, 1], 'width': width,
            'success': entry['label'], 'metadata': {
                'source_trial_id': entry['id'], 'source_trial': str(trial),
                'label': entry['label'], 'source_network_row': plan['network_row'],
                'source_network_depth_m': plan['network_row'][3],
                'source_contact_pose_world': pose, 'commanded_eef_world': command,
                'measured_preclose_eef_world': measured, 'command_tracking_error': error,
                'last_approach_tick': last['tick'],
                'measured_finger_sum_at_approach_m': sum(last['qpos'][names.index(n)] for n in ['right_joint27', 'right_joint28']),
                'measured_finger_sum_at_close_m': (sum(last_close['qpos'][names.index(n)] for n in ['right_joint27', 'right_joint28']) if last_close else None),
                'original_result_reason': entry['original_reason'],
                'width_semantics': 'commanded preclose opening, nominal contact endpoints'}}


def load_model(out):
    from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule
    module = ContinualLearningModule.load_inplace(out / 'model', device='cpu')
    if module.encoder_pointcloud != 'with_table':
        raise RuntimeError('Unexpected encoder pointcloud configuration')
    return module


def build(out):
    reg = verify(out)
    with (out / 'bank_started.json').open('x') as f:
        json.dump({'time': datetime.now().astimezone().isoformat()}, f)
    seed()
    module = load_model(out)
    if len(module.grasp_scoring_module_cl.X_on):
        raise RuntimeError('Expected an empty new ARX memory')
    records = []
    try:
        for entry in reg['training_trials']:
            if entry['label'] is None:
                continue
            trial = Path(entry['trial'])
            scene, target, _ = clouds(trial)
            grasp = commanded_grasp(trial, entry)
            encoded = module.encoder.predict(scene, [grasp])
            module.grasp_scoring_module_cl.add_datapoint(encoded)
            if entry['label'] == 1:
                args = (target, scene, deepcopy(encoded), 'unknown', 'arx_verified_success')
                module.grasp_recall_module_object.add_demonstration(*args)
                args = (target, scene, deepcopy(encoded), 'unknown', 'arx_verified_success')
                module.grasp_recall_module_patch.add_demonstration(*args, debug=False)
            record = {'id': entry['id'], 'label': entry['label'], 'grasp': grasp,
                      'scene_points': len(scene.points), 'target_points': len(target.points)}
            records.append(record)
            save(out / 'demonstrations' / (entry['id'] + '.json'), record)
            print('bank', entry['id'], 'label', entry['label'], flush=True)
        module.feedback_counts = {'arx_verified': {'labels': len(records),
                                  'demonstrations': sum(r['label'] == 1 for r in records)}}
        module.save(out / 'model')
        save(out / 'bank_complete.json', {
            'positive_demonstrations': sum(r['label'] == 1 for r in records),
            'negative_feedback': sum(r['label'] == 0 for r in records),
            'object_templates': len(module.grasp_recall_module_object._db),
            'patch_templates': len(module.grasp_recall_module_patch._db),
            'online_points': len(module.grasp_scoring_module_cl.X_on),
            'model_sha256': {str(p): sha(p) for p in sorted((out / 'model').rglob('*')) if p.is_file()},
            'source_ids': [r['id'] for r in records]})
    except Exception:
        save(out / 'bank_error.json', {'traceback': traceback.format_exc(), 'completed_ids': [r['id'] for r in records]})
        raise


def zero_grasps(rows, camera, limit):
    grasps = []
    for idx in np.argsort(-rows[:, 0], kind='stable')[:limit]:
        row = rows[idx]
        rotation = camera[:3, :3] @ row[4:13].reshape(3, 3)
        center = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3] * rotation[:, 0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation @ AXES, center
        for flip in [False, True]:
            chosen = pose.copy()
            if flip:
                chosen[:3, :3] = chosen[:3, :3] @ np.diag([-1., -1., 1.])
            grasps.append({'pose': chosen, 'width': float(row[1]),
                           'point1': center - row[1]/2 * chosen[:3, 1],
                           'point2': center + row[1]/2 * chosen[:3, 1],
                           'metadata': {'origin': 'zerograsp', 'raw_row_index': int(idx),
                                        'original_network_score': float(row[0]),
                                        'source_network_depth_m': float(row[3]), 'finger_flip': flip}})
    return grasps


def display_row(grasp, camera):
    pose = np.asarray(grasp['pose'])
    rotation = pose[:3, :3] @ AXES.T
    depth = grasp['metadata']['source_network_depth_m']
    surface = pose[:3, 3] - depth * rotation[:, 0]
    inv = np.linalg.inv(camera)
    row = np.zeros(17)
    row[:4] = [grasp['score'], grasp['width'], .02, depth]
    row[4:13] = (inv[:3, :3] @ rotation).reshape(-1)
    row[13:16] = inv[:3, :3] @ surface + inv[:3, 3]
    row[16] = -1
    return row


def annotate_novelty(grasps, all_zero):
    centers = np.array([g['pose'][:3, 3] for g in all_zero])
    rotations = np.array([g['pose'][:3, :3] for g in all_zero])
    tree = cKDTree(centers)
    for grasp in grasps:
        pose = np.asarray(grasp['pose'])
        distance, index = tree.query(pose[:3, 3])
        near = tree.query_ball_point(pose[:3, 3], .005)
        min_angle = None
        if near:
            relative = rotations[near] @ pose[:3, :3].T
            angles = Rotation.from_matrix(relative).magnitude()
            min_angle = float(np.degrees(angles.min()))
        grasp['metadata']['novelty_against_all_zero'] = {
            'nearest_position_distance_m': float(distance),
            'min_rotation_deg_within_5mm': min_angle,
            'novel': not near or min_angle > 10.,
            'definition': 'No original ZeroGrasp row within 5mm and 10deg, including finger symmetry'}


def infer(module, trial, policy):
    """Public adapter used for saved and fresh frames; no outcomes enter inference."""
    import torch
    scene, target, camera = clouds(trial)
    rows = np.load(trial / 'raw_grasps.npy', allow_pickle=False)
    zero = zero_grasps(rows, camera, policy['zero_top_k'])
    recalled, recall_counts = [], {}
    for name, recall in [('object', module.grasp_recall_module_object), ('patch', module.grasp_recall_module_patch)]:
        matches = recall.predict(target, scene, debug=False,
            max_matches_per_template=policy['recall_max_matches_per_template'],
            max_total_time_ms=policy['patch_time_ms'])
        recall_counts[name] = len(matches)
        for grasp in matches:
            grasp['metadata']['origin'] = 'demonstration_' + name
        recalled.extend(matches)
    annotate_novelty(recalled, zero_grasps(rows, camera, len(rows)))
    support = cKDTree(np.asarray(target.points))
    accepted, rejected = [], []
    for grasp in zero + recalled:
        pose = np.asarray(grasp['pose'])
        distance = float(support.query(pose[:3, 3])[0])
        reason = None
        if not np.isfinite(pose).all() or not np.isfinite(grasp['width']):
            reason = 'non_finite'
        elif not 0 < grasp['width'] <= policy['max_width_m'] + 1e-6:
            reason = 'opening_exceeds_original_arx_limit'
        elif distance > policy['target_support_distance_m']:
            reason = 'no_observed_target_support_within_55mm'
        grasp['metadata']['observed_target_distance_m'] = distance
        if reason:
            rejected.append({'reason': reason, 'grasp': grasp})
        else:
            accepted.append(grasp)
    encoded = []
    for start in range(0, len(accepted), 64):
        with torch.no_grad():
            encoded.extend(module.encoder.predict(scene, accepted[start:start+64]))
    module.grasp_scoring_module_cl.score_grasps(encoded, estimation_mode='posterior')
    offline = deepcopy(encoded)
    module.grasp_scoring_module_cl.score_grasps(offline, estimation_mode='offline')
    for grasp, prior in zip(encoded, offline):
        grasp['offline_score'] = prior['score']
        grasp.pop('score_thompson', None)
    ordered = sorted(encoded, key=lambda g: -g['score'])
    return {'candidates': ordered, 'recalled_before_filter': recalled, 'rejected': rejected,
            'counts': {'raw_zero': len(rows), 'zero_scored_input': len(zero),
                       'recalled': recall_counts, 'scored': len(ordered), 'rejected': len(rejected)},
            'scene_points': len(scene.points), 'target_points': len(target.points), 'camera': camera}


def write_inference(result, trial, destination):
    import open3d as o3d
    save(destination / 'candidate_pool.json', result)
    for name in ['state.json', 'camera/params.json', 'camera/color.png', 'camera/depth.png', 'camera/mask.png']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(trial / name, target)
    if result['candidates']:
        rows = np.stack([display_row(g, result['camera']) for g in result['candidates']])
        np.save(destination / 'raw_grasps.npy', rows, allow_pickle=False)
        save(destination / 'network_candidates.json', {
            'model_name': 'ZeroGrasp + GraspCL', 'count': len(rows),
            'candidate_stage': 'GraspCL posterior mean over merged ZeroGrasp + demonstration recall; before ARX planning',
            'score_label': 'GraspCL score', 'physical_attempt': False})
    scene, target, _ = clouds(trial)
    o3d.io.write_point_cloud(str(destination / 'zg_outputs/observed_scene.ply'), scene)
    o3d.io.write_point_cloud(str(destination / 'zg_outputs/observed_target.ply'), target)


def evaluate(out):
    reg = verify(out)
    bank = read(out / 'bank_complete.json')
    for path, digest in bank['model_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen ARX bank changed: ' + path)
    seed()
    module = load_model(out)
    results = []
    for entry in reg['query_trials']:
        destination = out / 'queries' / entry['id']
        destination.mkdir(parents=True, exist_ok=False)
        (destination / 'zg_outputs').mkdir()
        started = time.monotonic()
        record = {'id': entry['id'], 'layout': entry['layout'],
                  'original_success': entry['original_success'], 'physical_attempt': False}
        try:
            seed()
            pool = infer(module, Path(entry['trial']), reg['candidate_policy'])
            write_inference(pool, Path(entry['trial']), destination)
            ranked = pool['candidates']
            recalled = [g for g in ranked if g['metadata']['origin'].startswith('demonstration_')]
            record.update(status='inference_completed', counts=pool['counts'],
                novel_recalled_count=sum(g['metadata']['novelty_against_all_zero']['novel'] for g in recalled),
                top10_recalled=sum(g['metadata']['origin'].startswith('demonstration_') for g in ranked[:10]),
                top1_origin=ranked[0]['metadata']['origin'] if ranked else None,
                top1_score=ranked[0]['score'] if ranked else None,
                maximum_score_change_from_feedback=max((abs(g['score']-g['offline_score']) for g in ranked), default=0.))
        except Exception:
            record.update(status='error', traceback=traceback.format_exc())
        record['elapsed_seconds'] = time.monotonic() - started
        save(destination / 'offline_result.json', record)
        results.append(record)
        save(out / 'offline_summary.json', {'scope': 'Saved ARX RGB-D inference only, no new physical grasp success measurement',
             'registered_queries': len(reg['query_trials']), 'completed': len(results), 'results': results})
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'build', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())
