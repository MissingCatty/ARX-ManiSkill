"""Generate registered scenes from geometry and images before any IK evaluation."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

import numpy as np

from asset_catalog import HERE, PROJECT, MS, read, save, sha
import pipeline
import geometric_sampling as sampling

SOURCE = HERE / 'datasets/diverse200_20261122_01'
DEFAULT = HERE / 'datasets/diverse200_geometry_20261122_01'
URDF = PROJECT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
MASTER_SEED = 2026112223
SOURCES = [Path(__file__), HERE / 'geometric_sampling.py', HERE / 'workspace_geometry.py',
           HERE / 'verify_geometric_dataset.py', HERE / 'test_geometric_sampling.py',
           HERE / 'pipeline.py', HERE / 'asset_environment.py', HERE / 'asset_catalog.py',
           pipeline.LEGACY / 'scene_builder.py', PROJECT / 'configs/head_camera_mount.json',
           MS / 'arx_pipeline/environment.py', MS / 'arx_pipeline/baseline.py',
           MS / 'arx_pipeline/scene_tools.py', URDF]


def now():
    return datetime.now().astimezone().isoformat()


def prepare(root):
    if root.exists():
        raise RuntimeError('New dataset directory required; completed data must be preserved')
    old = read(SOURCE / 'manifest.json')
    old_reg = read(SOURCE / 'registration.json')
    state = read(SOURCE / 'preflight/head_camera.json')
    region = sampling.define_region(state, URDF)
    schedule = []
    for entry in old['schedule']:
        row = {name: deepcopy(entry[name]) for name in
               ['id', 'model_id', 'semantic_class', 'family', 'target_text', 'layout']}
        row['seed'] = int.from_bytes(hashlib.sha256(f'{MASTER_SEED}:{entry["id"]}'.encode()).digest()[:4], 'little')
        # Transfer identities and quotas only, never prior poses or IK results.
        row['actor_identities'] = [{key: actor[key] for key in ['name', 'model_id']}
                                   for actor in entry['scene']['actors']]
        schedule.append(row)
    root.mkdir(parents=True)
    for name in ['cases', 'preflight', 'gallery', 'source']:
        (root / name).mkdir()
    shutil.copy2(SOURCE / 'object_catalog.json', root / 'object_catalog.json')
    save(root / 'sampling_region.json', region)
    reg = {'id': root.name, 'at': now(), 'schedule': schedule,
           'source_dataset': str(SOURCE), 'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
           'source_identity_fields_only': True, 'source_positions_used': False,
           'master_seed': MASTER_SEED, 'seed_rule': 'SHA256(master_seed:scene_id), first four bytes little-endian',
           'target_models': old_reg['target_models'], 'semantic_classes': old_reg['semantic_classes'],
           'layouts': dict.fromkeys(pipeline.LAYOUTS, 40), 'distractor_counts': old_reg['distractor_counts'],
           'object_catalog_sha256': sha(root / 'object_catalog.json'), 'sampling_region': region,
           'sampling_region_sha256': sha(root / 'sampling_region.json'),
           'lift_joint_m': .452, 'camera': 'unchanged downward head D405', 'physical_grasp_budget': 0,
           'grasp_tests_authorized': False, 'construction_independent_of_ik': True,
           'admission': {'maximum_variants': 80, 'maximum_target_proposals': 10000,
                         'target_yaw': 'continuous uniform [-pi, pi), same rule for all five layouts',
                         'position_sampling': 'uniform within the predeclared tabletop region; S2 uses near edge',
                         'minimum_target_pixels': 64, 'valid_depth_fraction': .98,
                         'occlusion_S5': [.20, .40], 'occlusion_others_max': .15,
                         'all_actors_fully_in_head_frame': True, 'initial_robot_contacts_allowed': False,
                         'IK_selection': False, 'planning_selection': False,
                         'SAM3_or_grasp_model_selection': False, 'physical_grasp_success_selection': False},
           'source_hashes': {str(p): sha(p) for p in SOURCES},
           'post_freeze_diagnosis': 'Only after complete geometry verification and file-hash freezing; every slot retained regardless of IK result.',
           'authorization': 'User approved fixed-region independent sampling after rejecting IK-conditioned placements.'}
    save(root / 'registration.json', reg)
    for p in SOURCES:
        shutil.copy2(p, root / 'source' / (p.parent.name + '__' + p.name))
    save(root / 'progress.json', {'phase': 'registered', 'registered': 200, 'completed': 0,
                                'physical_grasps': 0, 'construction_independent_of_ik': True, 'at': now()})
    print('REGISTERED_INDEPENDENT_SAMPLING', root, flush=True)


class GeometryBuilder:
    def __init__(self, root):
        self.root = root
        self.reg = read(root / 'registration.json')
        self.region = self.reg['sampling_region']
        self.db = load_object_catalog(root / 'object_catalog.json')['models']
        self.reference = read(SOURCE / 'preflight/head_camera.json')
        self.original_capture = pipeline.scenes.capture

    def state_for_projection(self, camera, k):
        return {**self.reference, 'camera_world_optical': camera, 'intrinsics': k}

    def targets(self, entry, baseline, camera, k, rng):
        state = self.state_for_projection(camera, k)
        targets = []
        counts = Counter()
        for index in range(self.reg['admission']['maximum_target_proposals']):
            target = sampling.sample_target(entry, baseline, self.region, rng)
            checked = sampling.check_placement(target, self.db, state, self.region)
            if checked['passed']:
                targets.append(target)
                if len(targets) == 80:
                    break
            else:
                for key in ['inside_registered_roi', 'inside_table', 'fully_in_head_frame', 'not_clearly_beyond_link_length']:
                    if not checked[key]:
                        counts[key] += 1
        save(self.root / 'preflight' / (entry['id'] + '_sampling.json'),
             {'proposals': index + 1, 'geometrically_admitted': len(targets),
              'rejection_counts': dict(counts), 'IK_queries': 0})
        if not targets:
            raise RuntimeError('No geometric head-view placement for ' + entry['id'])
        return targets

    def distractors(self, entry, reg, baseline, rng):
        return [a['model_id'] for a in entry['actor_identities'] if a['name'] != 'target']

    def layout(self, entry, target, models, baseline, camera, k, rng):
        state = self.state_for_projection(camera, k)
        actors = [deepcopy(target)]
        identities = [a for a in entry['actor_identities'] if a['name'] != 'target']
        assert models == [a['model_id'] for a in identities]
        for identity in identities:
            for _ in range(2500):
                center = rng.uniform(self.region['center_roi_min_xy_m'], self.region['center_roi_max_xy_m'])
                spec = baseline.actor_spec(identity['name'], identity['model_id'], center,
                                           float(rng.uniform(-np.pi, np.pi)))
                if not sampling.check_placement(spec, self.db, state, self.region)['passed']:
                    continue
                if baseline.separate(spec, actors, gap=.05 if entry['layout'] == 'S3_sparse' else .012):
                    actors.append(spec)
                    break
            else:
                raise RuntimeError('No stable nonoverlapping geometric distractor placement')
        return {'layout': entry['layout'], 'actors': actors, 'perception_camera': 'head'}

    def blockers(self, env, scene, baseline, camera, k, rng):
        import sapien
        import torch
        scenes = pipeline.scenes
        state = self.state_for_projection(camera, k)
        for actor in scene['actors']:
            actor['pose'] = env.objects[actor['name']].pose.raw_pose[0].cpu().numpy().tolist()
        target = scene['actors'][0]
        lo, hi = baseline.bounds(target)
        center = (lo + hi) / 2
        forward = camera[:2, 3] - center[:2]
        forward /= np.linalg.norm(forward)
        side = np.array([-forward[1], forward[0]])
        projection = scenes.project(target, baseline, camera, k)
        if projection is None:
            return False
        angle = float(np.arctan2(forward[1], forward[0]))
        for blocker in scene['actors'][1:]:
            others = [a for a in scene['actors'] if a['name'] != blocker['name']]
            rest_z = blocker['pose'][2]
            positions = []
            for yaw in [0., np.pi/2, angle, angle+np.pi/2, angle+np.pi/4, angle-np.pi/4]:
                for along in np.linspace(.025, .40, 18):
                    for lateral in np.linspace(-.32, .32, 65):
                        spec = baseline.actor_spec(blocker['name'], blocker['model_id'],
                                                   center[:2] + along*forward + lateral*side, yaw)
                        spec['pose'][2] = rest_z
                        if not sampling.check_placement(spec, self.db, state, self.region)['passed']:
                            continue
                        if not baseline.separate(spec, others, gap=.008):
                            continue
                        pr = scenes.project(spec, baseline, camera, k)
                        if pr is None:
                            continue
                        overlap = np.maximum(0., np.minimum(pr['hi'], projection['hi']) - np.maximum(pr['lo'], projection['lo']))
                        fraction = float(np.prod(overlap) / projection['area'])
                        if .015 <= fraction <= .98:
                            positions.append((abs(fraction-.30), spec))
            positions.sort(key=lambda item: item[0])
            indices = list(range(min(80, len(positions))))
            if len(positions) > 80:
                indices += np.linspace(80, len(positions)-1, min(120, len(positions)-80), dtype=int).tolist()
            best = None
            for index in dict.fromkeys(indices):
                spec = positions[index][1]
                actor = env.objects[spec['name']]
                p = spec['pose']
                actor.set_pose(sapien.Pose(p[:3], p[3:]))
                actor._bodies[0].linear_velocity = np.zeros(3)
                actor._bodies[0].angular_velocity = np.zeros(3)
                quality, _, _ = scenes.head_quality(env, baseline)
                if scenes.accepted(quality, 'S5_occluded'):
                    trial = deepcopy(scene)
                    trial['actors'][scene['actors'].index(blocker)] = deepcopy(spec)
                    scenes.set_scene(env, trial, baseline, sapien, torch)
                    final, _, _ = scenes.head_quality(env, baseline)
                    if scenes.accepted(final, 'S5_occluded'):
                        for item in scene['actors']:
                            item['pose'] = env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()
                        return True
                    fraction = final['occlusion_fraction']
                else:
                    fraction = quality['occlusion_fraction']
                if fraction < .20 and (best is None or fraction > best[0]):
                    best = fraction, deepcopy(spec)
            if best:
                blocker['pose'] = best[1]['pose']
            p = blocker['pose']
            actor = env.objects[blocker['name']]
            actor.set_pose(sapien.Pose(p[:3], p[3:]))
            actor._bodies[0].linear_velocity = np.zeros(3)
            actor._bodies[0].angular_velocity = np.zeros(3)
        return False

    def capture(self, root, entry, env, scene, quality, obs, mask, runner, profile, convert, maps):
        actual = deepcopy(scene)
        for actor in actual['actors']:
            actor['pose'] = env.objects[actor['name']].pose.raw_pose[0].cpu().numpy().tolist()
        state = runner.snapshot(env, obs)
        checks = {a['name']: sampling.check_placement(a, self.db, state, self.region) for a in actual['actors']}
        if not all(c['passed'] for c in checks.values()):
            raise RuntimeError('Settled geometric workspace or head-view admission failed')
        if entry['layout'] == 'S2_edge' and not sampling.edge_passed(actual['actors'][0], self.db, self.region):
            raise RuntimeError('Settled near-side edge admission failed')
        assert [(a['name'], a['model_id']) for a in actual['actors']] == [(a['name'], a['model_id']) for a in entry['actor_identities']]
        bound = self.original_capture(root, entry, env, actual, quality, obs, mask, runner, profile, convert, maps)
        save(root / 'cases' / entry['id'] / 'construction_check.json',
             {'passed': True, 'construction_independent_of_ik': True, 'IK_queries': 0,
              'actors': checks, 'physical_grasps': 0, 'at': now()})
        return bound

    def build(self, resume=False):
        pipeline.scenes.target_candidates = self.targets
        pipeline.scenes.layout_scene = self.layout
        pipeline.scenes.adjust_blocker = self.blockers
        pipeline.scenes.capture = self.capture
        pipeline.distractors = self.distractors
        pipeline.build(self.root, resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'build'])
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root.resolve())
    else:
        GeometryBuilder(args.root.resolve()).build(args.resume)
