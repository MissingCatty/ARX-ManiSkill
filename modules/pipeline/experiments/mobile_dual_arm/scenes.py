"""Freeze 100 newly sampled near-edge scenes before model or IK evaluation."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from arx_project.assets import load_object_catalog
from arx_project.layout import ROOT, now, read_json, sha256, write_json
from arx_project.scenes import current_dataset, current_schedule

HERE = Path(__file__).resolve().parent
DEFAULT = ROOT/'simulation/scenes/mobile100'


def imports():
    sys.path[:0] = [str(HERE.parent/'diverse_object_scenes'),
                   str(HERE.parent/'head_perception_scenes'), str(ROOT/'modules/maniskill'),
                   str(ROOT/'modules/maniskill/arx_pipeline')]


def arrange_occluders(env, scene, baseline, geometry, camera, k, sapien):
    """Combine existing small blockers for broad flat targets, using visibility."""
    target = scene['actors'][0]
    lo, hi = baseline.bounds(target)
    center = (lo+hi)/2
    forward = camera[:2, 3]-center[:2]
    forward /= np.linalg.norm(forward)
    side = np.array([-forward[1], forward[0]])
    projection = geometry.project(target, baseline, camera, k)
    for actor in scene['actors'][1:]:
        others = [a for a in scene['actors'] if a['name'] != actor['name']]
        sizes = baseline.DB[actor['model_id']]['size_m']
        yaw = float(np.arctan2(forward[1], forward[0])+(0 if sizes[0] < sizes[1] else np.pi/2))
        candidates = []
        for along in np.arange(.06, .43, .025):
            for lateral in np.arange(-.25, .26, .015):
                xy = center[:2]+along*forward+lateral*side
                spec = baseline.actor_spec(actor['name'], actor['model_id'], xy, yaw)
                if not geometry.nonoverlap(spec, others, baseline, .008):
                    continue
                view = geometry.project(spec, baseline, camera, k)
                if view is None:
                    continue
                overlap = np.maximum(0., np.minimum(view['hi'], projection['hi'])-np.maximum(view['lo'], projection['lo']))
                fraction = float(np.prod(overlap)/projection['area'])
                if fraction > .03:
                    candidates.append((abs(fraction-.25), spec))
        candidates.sort(key=lambda item: item[0])
        best, best_fraction = deepcopy(actor), -1.
        for _, spec in candidates[:100]:
            p = spec['pose']
            body = env.objects[actor['name']]
            body.set_pose(sapien.Pose(p[:3], p[3:]))
            body._bodies[0].linear_velocity = np.zeros(3)
            body._bodies[0].angular_velocity = np.zeros(3)
            q, _, _ = geometry.head_quality(env, baseline)
            fraction = q['occlusion_fraction']
            if geometry.accepted(q, 'S5_occluded'):
                actor.update(spec)
                return True
            if fraction <= .40 and fraction > best_fraction:
                best, best_fraction = deepcopy(spec), fraction
        actor.update(best)
        p = actor['pose']
        env.objects[actor['name']].set_pose(sapien.Pose(p[:3], p[3:]))
    return False


def prepare(root):
    if root.exists():
        raise ValueError('Scene collection already exists')
    source = current_schedule()
    categories = sorted({r['semantic_class'] for r in source})
    layouts = list(dict.fromkeys(r['layout'] for r in source))
    selected = []
    for i, layout in enumerate(layouts):
        # Alternating halves cover every category twice, with 20 a third time.
        subset = categories[(i % 2)*20:(i % 2+1)*20]
        for category in subset:
            row = deepcopy(next(r for r in source if r['layout'] == layout and r['semantic_class'] == category))
            selected.append({k: row[k] for k in ['model_id', 'semantic_class', 'family', 'target_text', 'layout']})
            selected[-1].update(id=f'M{len(selected):04d}_{layout}_{category}',
                seed=2026112504+len(selected)*100003,
                identities=[{'name': a['name'], 'model_id': a['model_id']} for a in row['scene']['actors']])
    root.mkdir(parents=True)
    shutil.copy2(current_dataset()/'object_catalog.json', root/'object_catalog.json')
    for name in ['cases', 'gallery']:
        (root/name).mkdir()
    write_json(root/'registration.json', {'version': 'V4', 'at': now(), 'schedule': selected,
        'layouts': dict(Counter(r['layout'] for r in selected)), 'scenes': 100,
        'sampling': {'target_footprint_near_edge_gap_m': [.10, .26], 'S2_edge_gap_m': [.08, .10],
                     'wide_flat_S5_gap_m': [.20, .40],
                     'target_center_y_m': [-.15, .50], 'target_yaw_radians': [-float(np.pi), float(np.pi)],
                     'maximum_geometry_variants': 100, 'IK_grasp_or_model_selection': False},
        'source_identifies_objects_only': str(current_dataset()/'manifest.json'),
        'source_sha256': sha256(__file__), 'catalog_sha256': sha256(root/'object_catalog.json')})


def build(root):
    reg = read_json(root/'registration.json')
    if (root/'frozen_dataset.json').exists():
        raise ValueError('Already frozen')
    if reg['source_sha256'] != sha256(__file__):
        raise ValueError('Scene generator changed after registration')
    imports()
    import pipeline as assets
    import scene_builder as geometry
    from PIL import Image
    environment, baseline, run, sapien, torch, factory, profile, convert, maps = assets.runtime(
        load_object_catalog(root/'object_catalog.json')['models'])
    completed = []
    for index, entry in enumerate(reg['schedule']):
        folder = root/'cases'/entry['id']
        if (folder/'entry.json').exists():
            completed.append(read_json(folder/'entry.json'))
            continue
        folder.mkdir(exist_ok=True)
        rng = np.random.default_rng(entry['seed'])
        target = baseline.actor_spec('target', entry['model_id'], [-.47, .20], 0.)
        scene = {'layout': entry['layout'], 'actors': [target]}
        # All actors must exist before variants can be repositioned.
        for i, identity in enumerate(entry['identities'][1:]):
            scene['actors'].append(baseline.actor_spec(identity['name'], identity['model_id'],
                                                      [-.1, -.7+i*.18], 0.))
        env = factory(entry['model_id'], scene)
        rejected = []
        try:
            # Dummy actor poses exist only to create the articulation/camera;
            # settle the actual sampled scene inside the variant loop.
            env.reset(seed=entry['seed'])
            state = run.snapshot(env, env.get_obs())
            camera, k = np.asarray(state['camera_world_optical']), np.asarray(state['intrinsics'])
            for variant in range(reg['sampling']['maximum_geometry_variants']):
                try:
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    temp = baseline.actor_spec('target', entry['model_id'], [0, 0], yaw)
                    lo, hi = baseline.bounds(temp)
                    half_x = (hi[0]-lo[0])/2
                    gaps = reg['sampling']['S2_edge_gap_m' if entry['layout'] == 'S2_edge' else 'target_footprint_near_edge_gap_m']
                    size = baseline.DB[entry['model_id']]['size_m']
                    if entry['layout'] == 'S5_occluded' and max(size[:2])/max(size[2], .001) > 10:
                        gaps = reg['sampling']['wide_flat_S5_gap_m']
                    x = geometry.TABLE_MIN[0]+half_x+float(rng.uniform(*gaps))
                    target = baseline.actor_spec('target', entry['model_id'],
                        [x, float(rng.uniform(*reg['sampling']['target_center_y_m']))], yaw)
                    if geometry.project(target, baseline, camera, k) is None:
                        raise ValueError('target_outside_head_view')
                    scene = geometry.layout_scene(entry, target,
                        [a['model_id'] for a in entry['identities'][1:]], baseline, camera, k, rng)
                    geometry.set_scene(env, scene, baseline, sapien, torch)
                    if entry['layout'] == 'S5_occluded':
                        # Use the tallest of the already registered objects as
                        # blocker, keeping every object's identity unchanged.
                        scene['actors'][1:] = sorted(scene['actors'][1:],
                            key=lambda a: baseline.DB[a['model_id']]['size_m'][2], reverse=True)
                        if not geometry.adjust_blocker(env, scene, baseline, camera, k, rng) \
                                and not arrange_occluders(env, scene, baseline, geometry, camera, k, sapien):
                            raise ValueError('No geometric 20-40% occlusion')
                        geometry.set_scene(env, scene, baseline, sapien, torch)
                    quality, obs, mask = geometry.head_quality(env, baseline)
                    if not geometry.accepted(quality, entry['layout']):
                        raise ValueError('geometric_layout_or_visibility: '+str(quality))
                    # Settled rigid transforms, not commanded spawn values.
                    settled = deepcopy(scene)
                    for actor in settled['actors']:
                        actor['pose'] = env.objects[actor['name']].pose.raw_pose[0].cpu().numpy().tolist()
                    row = {k: deepcopy(v) for k, v in entry.items() if k != 'identities'}
                    row.update(scene=settled, admission=quality, camera='head',
                        robot_initial_q=env.initial_q.tolist(),
                        base_pose=env.agent.robot.links_map['base_link'].pose.raw_pose[0].cpu().numpy().tolist(),
                        construction_variant=variant, planner_arm=None)
                    write_json(folder/'entry.json', row)
                    write_json(folder/'geometry_rejections.json', rejected)
                    Image.fromarray(obs['sensor_data']['head']['rgb'][0].cpu().numpy()).save(folder/'head.jpg', quality=86)
                    Image.fromarray(mask.astype(np.uint8)*255).save(folder/'target_mask.png')
                    Image.fromarray(env.render().cpu().numpy()[0]).save(folder/'overview.jpg', quality=86)
                    completed.append(row)
                    print('SCENE', index+1, entry['id'], 'variant', variant, flush=True)
                    break
                except (RuntimeError, ValueError, AssertionError) as error:
                    rejected.append({'variant': variant, 'reason': str(error)})
            else:
                write_json(folder/'geometry_rejections.json', rejected)
                raise RuntimeError('Geometry construction exhausted: '+entry['id'])
        finally:
            env.close()
        write_json(root/'progress.json', {'phase': 'constructing', 'completed': len(completed), 'registered': 100,
                                        'IK_calls': 0, 'model_calls': 0, 'physical_grasps': 0})
    write_json(root/'manifest.json', {'schedule': completed, 'completed': 100, 'physical_grasps': 0})
    write_json(root/'verification.json', {'passed': len(completed) == 100, 'scenes': len(completed),
        'layouts': dict(Counter(r['layout'] for r in completed)),
        'categories': len({r['semantic_class'] for r in completed}),
        'target_models': len({r['model_id'] for r in completed}), 'IK_calls': 0, 'model_calls': 0,
        'source_sha256': reg['source_sha256']})
    files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()
             and p.name not in ['progress.json', 'frozen_dataset.json']}
    write_json(root/'frozen_dataset.json', {'at': now(), 'files': files, 'scenes': 100,
                                          'selection': 'geometry_only_before_any_model_or_IK'})
    write_json(root/'progress.json', {'phase': 'frozen', 'completed': 100, 'registered': 100})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'build'])
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else build)(args.dataset)
