"""VFAS-inspired RGB-D refinement for ARX, using an explicit geometric scorer.

This is not the published VFAS learned evaluator or GNP amortized inference.
The sampler, pose-continuity penalties and noise sensitivity follow VFAS ideas;
bilateral ARX fingertip support estimated from observed points replaces its
unavailable network. No dynamics parameters or success thresholds are changed.
"""
import json
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def observed_cloud(folder, snapshot, target, asset_size):
    """Use the same target isolation privilege as the existing ARX benchmark."""
    params = json.loads((folder / 'params.json').read_text())
    camera_id = params['camera_id']
    camera = np.asarray(snapshot['camera_world'][camera_id]) @ np.diag([1, -1, -1, 1])
    depth = np.asarray(Image.open(folder / 'depth.png')) * params['depth_scale']
    rgb = np.asarray(Image.open(folder / 'color.png').convert('RGB'), dtype=float)
    yy, xx = np.indices(depth.shape)
    k = params['intrinsics']
    points = np.stack([(xx - k['ppx']) / k['fx'] * depth,
                       (yy - k['ppy']) / k['fy'] * depth, depth], axis=-1)
    world = points @ camera[:3, :3].T + camera[:3, 3]
    valid = np.isfinite(world).all(axis=-1) & (depth > 0)
    target_world = np.asarray(snapshot['objects'][target]['world'])
    if target == 'test_object':
        local = (world - target_world[:3, 3]) @ target_world[:3, :3]
        size = np.asarray(asset_size)
        mask = ((np.abs(local) <= size / 2 + .004).all(axis=-1)
                & (local[:, :, 2] > -size[2] / 2 + .004))
    elif target == 'green_sphere':
        mask = ((rgb[:, :, 1] > 1.3 * rgb[:, :, 0])
                & (rgb[:, :, 1] > 1.05 * rgb[:, :, 2]) & (rgb[:, :, 1] > 50))
    elif target == 'blue_cylinder':
        mask = ((rgb[:, :, 2] > 1.5 * rgb[:, :, 0])
                & (rgb[:, :, 2] > .95 * rgb[:, :, 1]) & (rgb[:, :, 2] > 50))
    else:
        mask = ((rgb[:, :, 0] > 1.6 * rgb[:, :, 1])
                & (rgb[:, :, 0] > 1.6 * rgb[:, :, 2]) & (rgb[:, :, 0] > 50))
    # Reject distant scenery, matching the baseline's known-target isolation.
    mask &= valid & (np.linalg.norm(world - target_world[:3, 3], axis=-1) < .11)
    Image.fromarray(mask.astype(np.uint8)).save(folder / 'mask.png')
    cloud = world[mask]
    save(folder / 'observation.json', {'camera_world_optical': camera.tolist(),
         'snapshot_sim_time': snapshot['sim_time'], 'frame_sim_time': params['sim_time'],
         'mask_points': len(cloud), 'segmentation': 'Existing benchmark target pose/bounds or color; no mass/friction/mesh used in scoring'})
    return cloud, camera[:3, 3]


def surface_normals(points, camera_origins):
    _, ids = cKDTree(points).query(points, k=min(24, len(points)))
    neighbors = points[ids]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    facing = camera_origins - points
    normals *= np.where((normals * facing).sum(axis=1) < 0, -1., 1.)[:, None]
    confidence = 1. - eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12)
    return normals, confidence


def contact_quality(poses, points, normals):
    """Score observed antipodal support within the actual distal ARX pad span.

    USD collision mesh bounds in EEF coordinates give distal pad X approx
    [-.0165, .0045], Z [.030, .045] m. We use a smooth support kernel rather
    than assert unobserved surface contacts or force closure.
    """
    rot = poses[:, :3, :3]
    local = np.einsum('bni,bij->bnj', points[None] - poses[:, None, :3, 3], rot)
    align = np.einsum('ni,bi->bn', normals, rot[:, :, 1])
    support = np.exp(-.5 * (((local[:, :, 0] + .006) / .011) ** 2
                           + ((local[:, :, 2] - .0375) / .010) ** 2))
    within_opening = np.exp(-.5 * (np.maximum(np.abs(local[:, :, 1]) - .042, 0) / .004) ** 2)
    left = support * within_opening * np.maximum(align, 0) ** 2 * (local[:, :, 1] > .001)
    right = support * within_opening * np.maximum(-align, 0) ** 2 * (local[:, :, 1] < -.001)
    li, ri = left.argmax(axis=1), right.argmax(axis=1)
    ix = np.arange(len(poses))
    lq, rq = left[ix, li], right[ix, ri]
    yl, yr = local[ix, li, 1], local[ix, ri, 1]
    balance = np.exp(-.5 * ((yl + yr) / .008) ** 2)
    width = yl - yr
    feasible_width = (width > .003) & (width < .088)
    quality = np.sqrt(lq * rq) * balance * feasible_width
    return quality, {'left_support': lq, 'right_support': rq,
                     'left_y_m': yl, 'right_y_m': yr, 'width_m': width,
                     'balance': balance}


class VisualFeedbackRefiner:
    def __init__(self, client, out, target):
        self.client, self.out, self.target = client, Path(out), target
        self.folder = self.out / 'feedback'
        self.folder.mkdir(exist_ok=False)
        self.rng = np.random.default_rng(0)
        self.iteration = 0
        self.scale = 1.
        self.asset_size = (json.loads((self.out / 'active_asset.json').read_text())['size_m']
                           if target == 'test_object' else None)
        self.report = {'policy': 'vfas_inspired_geometric_feedback',
            'original_vfas_network': False, 'amortized_inference_trained': False,
            'observations': [], 'refinements': [],
            'limits': 'Stage-boundary updates, not 20 Hz visual servoing; partial RGB-D geometric contact proxy, not a learned success probability'}
        self.persist()

    def persist(self):
        save(self.out / 'visual_feedback_report.json', self.report)

    def observe(self):
        folder = self.folder / f'observation_{self.iteration:02d}'
        folder.mkdir()
        # Camera refresh is 5 Hz. No arm command runs concurrently with capture.
        time.sleep(.25)
        points, origins, views = [], [], []
        wrist = self.client.request('grasp_feedback_camera', enabled=True)
        try:
            for _ in range(30):
                time.sleep(.2)
                state = self.client.request('snapshot')
                stamp = state['camera_times'].get('right_wrist')
                if stamp is not None and state['sim_time'] - stamp <= .5:
                    break
            else:
                raise RuntimeError('Feedback wrist camera did not produce a fresh frame')
            with httpx.Client(base_url='http://127.0.0.1:8101', timeout=120, trust_env=False) as http:
                for camera_id in ('head', 'right_wrist'):
                    dest = folder / camera_id
                    container = '/workspace/COOLAS-ARX/' + str(dest.relative_to(self.out.parents[1]))
                    payload = {'camera_id': camera_id, 'output_dir': container}
                    if camera_id == 'right_wrist':
                        payload['serial'] = wrist['serial']
                    response = http.post('/camera/capture-rgbd', json=payload)
                    if response.is_error:
                        raise RuntimeError(f'Feedback capture {camera_id}: {response.status_code} {response.text}')
                    snapshot = self.client.request('snapshot')
                    save(dest / 'snapshot.json', snapshot)
                    cloud, origin = observed_cloud(dest, snapshot, self.target, self.asset_size)
                    views.append({'camera_id': camera_id, 'points': len(cloud),
                                  'frame_sim_time': json.loads((dest / 'params.json').read_text())['sim_time']})
                    if len(cloud) < 60:
                        continue
                    if len(cloud) > 1400:
                        cloud = cloud[self.rng.choice(len(cloud), 1400, replace=False)]
                    points.append(cloud)
                    origins.append(np.broadcast_to(origin, cloud.shape))
        finally:
            if wrist['owned_by_feedback']:
                self.client.request('grasp_feedback_camera', enabled=False)
        assert points, 'No visible target cloud at feedback observation'
        points, origins = np.concatenate(points), np.concatenate(origins)
        normals, confidence = surface_normals(points, origins)
        np.savez_compressed(folder / 'surface.npz', points=points, normals=normals,
                            origins=origins, normal_confidence=confidence)
        self.report['observations'].append({'folder': str(folder), 'views': views,
                                           'scoring_points': len(points)})
        self.persist()
        return points, normals

    def refine(self, selected, start, ik, route, open_checker, closed_checker):
        """Rank local samples, then admit only goals with checked ARX IK/paths."""
        started = time.monotonic()
        points, normals = self.observe()
        seed = np.asarray(selected['grasp_world']).copy()
        center = seed.copy()
        pool, round_reports = [], []
        for round_index in range(3):
            count = int(16 * self.scale)
            poses = np.repeat(center[None], count + 1, axis=0)
            perturb = np.repeat(np.eye(4)[None], count, axis=0)
            perturb[:, :3, 3] = self.rng.uniform(-.02 * self.scale, .02 * self.scale, (count, 3))
            angles = self.rng.uniform(-5 * self.scale, 5 * self.scale, (count, 3))
            angles[:, 1] = 0  # ARX +Y is the finger closing direction.
            perturb[:, :3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
            poses[1:] = center @ perturb
            noisy = [points, points + self.rng.normal(0, .001, points.shape),
                     points + self.rng.normal(0, .001, points.shape)]
            qualities = np.stack([contact_quality(poses, cloud, normals)[0] for cloud in noisy])
            mean, spread = qualities.mean(axis=0), np.ptp(qualities, axis=0)
            translation = np.linalg.norm(poses[:, :3, 3] - seed[:3, 3], axis=1)
            angle = Rotation.from_matrix(poses[:, :3, :3] @ seed[:3, :3].T).magnitude()
            score = mean - 2. * translation - .1 * angle - .6 * spread
            details = contact_quality(poses, points, normals)[1]
            samples = []
            for i, pose in enumerate(poses):
                value = {'pose_world': pose.tolist(), 'score': float(score[i]),
                         'quality_mean': float(mean[i]), 'quality_noise_range': float(spread[i]),
                         'seed_distance_m': float(translation[i]), 'seed_angle_rad': float(angle[i]),
                         **{k: float(v[i]) for k, v in details.items()}}
                samples.append(value)
                pool.append(value)
            best = int(score.argmax())
            center = poses[best].copy()
            round_reports.append({'round': round_index, 'scale': self.scale, 'samples': samples})
            self.scale = min(3., self.scale * 1.3) if score[best] < .5 else 1.
        # Always include the original target as the fallback candidate.
        original_quality = float(contact_quality(seed[None], points, normals)[0][0])
        pool.sort(key=lambda p: p['score'], reverse=True)
        trials, chosen = [], None
        for candidate in pool[:10]:
            grasp = np.asarray(candidate['pose_world'])
            lift = grasp.copy(); lift[2, 3] += .12
            qg = ik(grasp, start)
            entry = {'score': candidate['score'], 'reason': None}
            if qg is None:
                entry['reason'] = 'grasp_ik'
            elif not closed_checker.validate(qg[None, None]).all():
                entry['reason'] = 'nominal_closed_collision'
            else:
                ql = ik(lift, qg)
                if ql is None or not closed_checker.validate(ql[None, None]).all():
                    entry['reason'] = 'lift_ik_or_collision'
                elif route(start, qg) is None:
                    entry['reason'] = 'approach_route'
                else:
                    chosen = {**selected, 'grasp_world': grasp.tolist(), 'lift_world': lift.tolist(),
                              'q_grasp': qg.tolist(), 'q_lift': ql.tolist(),
                              'target_contact_world': (grasp[:3, 3] + grasp[:3, :3] @ [-.006, 0, .0375]).tolist(),
                              'visual_feedback': candidate}
            trials.append(entry)
            if chosen is not None:
                break
        if chosen is None:
            chosen = selected
        update = {'iteration': self.iteration, 'initial_pose': seed.tolist(),
                  'seed_geometric_quality': original_quality, 'rounds': round_reports,
                  'checked_candidates': trials, 'applied': chosen is not selected,
                  'pose_changed': not np.allclose(chosen['grasp_world'], seed, atol=1e-7),
                  'chosen_pose': chosen['grasp_world'], 'wall_seconds': time.monotonic() - started}
        self.report['refinements'].append(update)
        self.iteration += 1
        self.persist()
        print('VISUAL_FEEDBACK', json.dumps({k: update[k] for k in
              ('iteration', 'seed_geometric_quality', 'applied', 'wall_seconds')}), flush=True)
        return chosen
