"""Continue remaining scenes using settled-height occlusion measurements."""
from copy import deepcopy
from pathlib import Path
import argparse
import subprocess
import numpy as np

import workspace_layout as frozen
from asset_catalog import read, save, sha
import workspace_geometry as geometry


def settled_blockers(env, scene, baseline, state, pivot, radius, rng):
    import sapien
    import torch
    scenes = frozen.original.scenes
    camera, k = np.asarray(state['camera_world_optical']), np.asarray(state['intrinsics'])
    for item in scene['actors']:
        item['pose'] = env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()
    target = scene['actors'][0]
    lo, hi = baseline.bounds(target)
    center = (lo + hi) / 2
    direction = camera[:2, 3] - center[:2]
    forward = direction / np.linalg.norm(direction)
    side = np.array([-forward[1], forward[0]])
    target_projection = scenes.project(target, baseline, camera, k)
    diagnostics = []
    for blocker in scene['actors'][1:]:
        others = [a for a in scene['actors'] if a['name'] != blocker['name']]
        base_yaw = float(np.arctan2(forward[1], forward[0]))
        rest_z = blocker['pose'][2]
        positions = []
        for yaw in [base_yaw, base_yaw + np.pi / 2, base_yaw + np.pi / 4, base_yaw - np.pi / 4]:
            for along in np.linspace(.025, .36, 18):
                for lateral in np.linspace(-.30, .30, 81):
                    spec = baseline.actor_spec(blocker['name'], blocker['model_id'],
                        center[:2] + along * forward + lateral * side, yaw)
                    spec['pose'][2] = rest_z
                    if not geometry.placement_check(spec, baseline.DB, state, pivot, radius)['passed']:
                        continue
                    if not baseline.separate(spec, others, gap=.008):
                        continue
                    projected = scenes.project(spec, baseline, camera, k)
                    if projected is None:
                        continue
                    overlap = np.maximum(0., np.minimum(projected['hi'], target_projection['hi']) -
                                         np.maximum(projected['lo'], target_projection['lo']))
                    fraction = float(np.prod(overlap) / target_projection['area'])
                    if .015 <= fraction <= .98:
                        positions.append((abs(fraction - .30), spec))
        positions.sort(key=lambda pair: pair[0])
        # Sample across the ranked list as well as its head; large flat targets
        # have a projected bounding rectangle much larger than the true outline.
        indices = list(range(min(100, len(positions))))
        if len(positions) > 100:
            indices += np.linspace(100, len(positions) - 1, min(180, len(positions) - 100), dtype=int).tolist()
        best, maximum = None, 0.
        for index in dict.fromkeys(indices):
            _, spec = positions[index]
            p = spec['pose']
            actor = env.objects[spec['name']]
            actor.set_pose(sapien.Pose(p[:3], p[3:]))
            actor._bodies[0].linear_velocity = np.zeros(3)
            actor._bodies[0].angular_velocity = np.zeros(3)
            quality, _, _ = scenes.head_quality(env, baseline)
            fraction = quality['occlusion_fraction']
            maximum = max(maximum, fraction)
            if scenes.accepted(quality, 'S5_occluded'):
                # Validate at the final settled pose, using the registered
                # 20-40% criterion rather than a stricter provisional cutoff.
                trial = deepcopy(scene)
                trial['actors'][scene['actors'].index(blocker)] = deepcopy(spec)
                scenes.set_scene(env, trial, baseline, sapien, torch)
                final, _, _ = scenes.head_quality(env, baseline)
                if scenes.accepted(final, 'S5_occluded'):
                    for item in scene['actors']:
                        item['pose'] = env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()
                    return True
                fraction = final['occlusion_fraction']
            if fraction < .20 and (best is None or fraction > best[0]):
                best = fraction, deepcopy(spec)
        if best is not None:
            blocker['pose'] = best[1]['pose']
        p = blocker['pose']
        actor = env.objects[blocker['name']]
        actor.set_pose(sapien.Pose(p[:3], p[3:]))
        actor._bodies[0].linear_velocity = np.zeros(3)
        actor._bodies[0].angular_velocity = np.zeros(3)
        diagnostics.append({'blocker': blocker['name'], 'positions': len(positions), 'tested': len(indices),
                            'maximum_occlusion': maximum, 'kept_below_threshold': best[0] if best else None})
    print('OCCLUSION_SEARCH', diagnostics, flush=True)
    return False


def main(root):
    revision = root / 'construction_revisions/settled_occlusion_01'
    revision.mkdir(parents=True, exist_ok=True)
    receipt = revision / 'registration.json'
    if not receipt.exists():
        existing = {str(p): sha(p) for folder in (root / 'cases').iterdir()
                    if (folder / 'admission.json').exists() for name in
                    ['entry.json', 'admission.json', 'state.json', 'workspace_check.json', 'construction_attempts.json']
                    for p in [folder / name]}
        save(receipt, {'at': frozen.now(), 'source': str(Path(__file__)), 'source_sha256': sha(Path(__file__)),
                       'original_registration_sha256': sha(root / 'registration.json'),
                       'existing_case_file_hashes': existing,
                       'change': 'Use observed settled occluder heights and actual settled 20-40% admission; sample all ranks and four yaw directions. All object IDs, work-area, camera, physics, quotas and IK gates unchanged.',
                       'physical_grasps': 0})
        if (root / 'halt.json').exists():
            (root / 'halt.json').rename(revision / 'previous_halt.json')
    registered = read(receipt)
    assert registered['source_sha256'] == sha(Path(__file__))
    frozen.adjust_blockers = settled_blockers
    frozen.build(root, resume=True)
    for p, digest in registered['existing_case_file_hashes'].items():
        assert sha(p) == digest, p
    subprocess.run([str(frozen.PROJECT / 'envs/reconstruction/bin/python'),
                    str(frozen.HERE / 'verify_workspace_layout.py'), str(root)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
