"""Audit and relocate all scenes without running perception or grasp attempts."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from asset_catalog import HERE, PROJECT, MS, read, save, sha
import pipeline as original
import workspace_geometry as geometry

DEFAULT = HERE / 'datasets/diverse200_workspace_20261122_01'
SOURCE = HERE / 'datasets/diverse200_20261122_01'
URDF = PROJECT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
SOURCES = [Path(__file__), HERE / 'workspace_geometry.py', HERE / 'workspace_ik.py',
           HERE / 'verify_workspace_layout.py', HERE / 'test_workspace_geometry.py',
           HERE / 'pipeline.py', HERE / 'asset_environment.py', HERE / 'asset_catalog.py',
           original.LEGACY / 'scene_builder.py', PROJECT / 'configs/head_camera_mount.json',
           MS / 'arx_pipeline/environment.py', MS / 'arx_pipeline/baseline.py',
           MS / 'arx_pipeline/scene_tools.py', MS / 'arx_pipeline/planner.py', URDF,
           URDF.with_name('arx.yml')]


def now():
    return datetime.now().astimezone().isoformat()


class StaticIK:
    def __init__(self, root):
        self.queue = queue.Queue()
        self.log = (root / 'preflight/ik_stdout.log').open('a')
        self.err = (root / 'preflight/ik_stderr.log').open('a')
        command = [str(PROJECT.parent / 'curobo/.venv/bin/python'), str(HERE / 'workspace_ik.py'),
                   str(root / 'preflight/head_camera.json'), str(root / 'preflight/ik_model')]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.err, text=True, bufsize=1)
        def receive():
            try:
                for line in self.process.stdout:
                    self.log.write(line)
                    self.log.flush()
                    if line.startswith('WORKSPACE_IK '):
                        self.queue.put(json.loads(line[len('WORKSPACE_IK '):]))
            finally:
                self.queue.put({'error': 'Static IK subprocess exited'})
        threading.Thread(target=receive, daemon=True).start()
        ready = self.queue.get(timeout=240)
        if not ready.get('ready'):
            raise RuntimeError(str(ready))
        save(root / 'preflight/ik_ready.json', ready)

    def check(self, spec, db, request_id):
        pts = geometry.world_corners(spec, db)
        request = {'id': request_id, 'center_world_m': ((pts.min(0) + pts.max(0)) / 2).tolist(),
                   'object_top_m': float(pts[:, 2].max())}
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        reply = self.queue.get(timeout=120)
        if reply.get('id') != request_id or 'error' in reply:
            raise RuntimeError('Static IK reply integrity error: ' + str(reply))
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop": true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=20)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=20)
        self.log.close()
        self.err.close()


def check_sources(reg):
    for file, digest in reg['source_hashes'].items():
        if sha(file) != digest:
            raise RuntimeError('Registered construction source changed: ' + file)


def prepare(root, development=False):
    if root.exists():
        raise RuntimeError('Dataset directory already exists; do not overwrite it')
    old_manifest = read(SOURCE / 'manifest.json')
    old_reg = read(SOURCE / 'registration.json')
    old_state = read(SOURCE / 'preflight/head_camera.json')
    catalog = load_object_catalog(SOURCE / 'object_catalog.json')
    pivot, radius = geometry.reach_bound(old_state, URDF)
    source_rows = old_manifest['schedule']
    if development:
        source_rows = [next(e for e in source_rows if e['id'] == case) for case in
                       ['D0043_S2_edge_book', 'D0121_S4_clutter_milk', 'D0161_S5_occluded_corn']
                       if any(e['id'] == case for e in source_rows)]
        # Use the first member if a named category occurs at a different index.
        for layout in ['S2_edge', 'S4_clutter', 'S5_occluded']:
            if not any(e['layout'] == layout for e in source_rows):
                source_rows.append(next(e for e in old_manifest['schedule'] if e['layout'] == layout))
    root.mkdir(parents=True)
    for name in ['cases', 'preflight', 'source', 'gallery', 'search_logs']:
        (root / name).mkdir()
    shutil.copy2(SOURCE / 'object_catalog.json', root / 'object_catalog.json')
    schedule, audits = [], []
    for old in source_rows:
        row = {key: deepcopy(old[key]) for key in
               ['id', 'model_id', 'semantic_class', 'family', 'target_text', 'layout', 'seed']}
        row.update(source_scene=deepcopy(old['scene']), source_case=str(SOURCE / 'cases' / old['id']))
        schedule.append(row)
        checks = {a['name']: geometry.placement_check(a, catalog['models'], old_state, pivot, radius)
                  for a in old['scene']['actors']}
        audits.append({'id': old['id'], 'layout': old['layout'], 'actors': checks,
                       'target_geometry_passed': checks['target']['passed'],
                       'all_actor_geometry_passed': all(c['passed'] for c in checks.values()),
                       'near_edge_definition_passed': geometry.edge_check(old['scene']['actors'][0], catalog['models'])
                       if old['layout'] == 'S2_edge' else None})
    reg = {'id': root.name, 'at': now(), 'source_dataset': str(SOURCE),
           'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
           'source_registration_sha256': sha(SOURCE / 'registration.json'),
           'object_catalog_sha256': sha(root / 'object_catalog.json'),
           'schedule': schedule, 'development_only': development,
           'semantic_classes': old_reg['semantic_classes'], 'target_models': old_reg['target_models'],
           'layouts': dict(Counter(e['layout'] for e in schedule)),
           'distractor_counts': old_reg['distractor_counts'],
           'lift_joint_m': .452, 'camera': 'unchanged downward head D405',
           'physical_grasp_budget': 0, 'grasp_tests_authorized': False,
           'admission': {'maximum_center_distance_m': geometry.MAX_CENTER_DISTANCE_M,
                         'near_table_edge_gap_m': list(geometry.EDGE_GAP_M),
                         'near_edge': 'robot-side table edge only (minimum world X)',
                         'target_prepare_pose_static_IK': True,
                         'prepare_pose_definition': 'EEF 8 cm above target top, plus 8 cm preparation and 3 cm lift',
                         'IK_joint_limits_and_original_table_collision': True,
                         'maximum_variants': 100, 'minimum_target_pixels': 64,
                         'valid_depth_fraction': .98, 'occlusion_S5': [.20, .40],
                         'occlusion_others_max': .15, 'all_actors_fully_in_head_frame': True,
                         'SAM3_or_grasp_model_selection': False,
                         'physical_grasp_success_selection': False},
           'shoulder_pivot_world_m': pivot.tolist(), 'eef_radius_upper_bound_m': radius,
           'source_hashes': {str(p): sha(p) for p in SOURCES},
           'authorization': 'User requested all 200 scenes individually checked and repositioned after stopping the grasp run.'}
    save(root / 'registration.json', reg)
    save(root / 'original_placement_audit.json', {'at': now(), 'scenes': len(audits), 'records': audits,
         'target_geometry_passes': sum(e['target_geometry_passed'] for e in audits),
         'all_actor_geometry_passes': sum(e['all_actor_geometry_passed'] for e in audits)})
    for path in SOURCES:
        shutil.copy2(path, root / 'source' / (path.parent.name + '__' + path.name))
    save(root / 'progress.json', {'phase': 'registered', 'registered': len(schedule),
                                'completed': 0, 'physical_grasps': 0, 'at': now()})
    print('REGISTERED', root, len(schedule), flush=True)


def target_candidates(entry, baseline, state, pivot, radius, rng):
    db = baseline.DB
    candidates = []
    original_target = deepcopy(next(a for a in entry['source_scene']['actors'] if a['name'] == 'target'))
    if geometry.placement_check(original_target, db, state, pivot, radius)['passed']:
        if entry['layout'] != 'S2_edge' or geometry.edge_check(original_target, db):
            candidates.append(original_target)
    for _ in range(1400):
        yaw = float(rng.uniform(-np.pi, np.pi))
        if entry['layout'] == 'S2_edge':
            temp = baseline.actor_spec('target', entry['model_id'], [0, 0], yaw)
            lo, hi = baseline.bounds(temp)
            center = [-.7245 + (hi[0] - lo[0]) / 2 + rng.uniform(*geometry.EDGE_GAP_M),
                      rng.uniform(-.29, .27)]
        elif entry['layout'] == 'S5_occluded':
            center = rng.uniform([-.48, -.25], [-.36, .20])
        else:
            center = rng.uniform([-.56, -.29], [-.28, .27])
        spec = baseline.actor_spec('target', entry['model_id'], center, yaw)
        if geometry.placement_check(spec, db, state, pivot, radius)['passed']:
            candidates.append(spec)
            if len(candidates) >= 100:
                break
    if not candidates:
        raise RuntimeError('No visible work-area placement: ' + entry['id'])
    return candidates


def scene_candidate(entry, target, baseline, state, pivot, radius, rng):
    actors = [deepcopy(target)]
    old_target = entry['source_scene']['actors'][0]
    shift = np.asarray(target['pose'][:2]) - np.asarray(old_target['pose'][:2])
    for source in entry['source_scene']['actors'][1:]:
        placed = None
        for index in range(1600):
            if index == 0:
                spec = deepcopy(source)
                spec['pose'][0] += float(shift[0])
                spec['pose'][1] += float(shift[1])
            else:
                center = rng.uniform([-.62, -.43], [-.20, .47])
                spec = baseline.actor_spec(source['name'], source['model_id'], center,
                                           float(rng.uniform(-np.pi, np.pi)))
            if not geometry.placement_check(spec, baseline.DB, state, pivot, radius)['passed']:
                continue
            if not baseline.separate(spec, actors, gap=.05 if entry['layout'] == 'S3_sparse' else .012):
                continue
            placed = spec
            break
        if placed is None:
            raise RuntimeError('No nonoverlapping visible work-area position for ' + source['name'])
        actors.append(placed)
    return {'layout': entry['layout'], 'actors': actors, 'perception_camera': 'head'}


def adjust_blockers(env, scene, baseline, state, pivot, radius, rng):
    import sapien
    scenes = original.scenes
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
    for blocker in scene['actors'][1:]:
        others = [a for a in scene['actors'] if a['name'] != blocker['name']]
        base_yaw = float(np.arctan2(forward[1], forward[0]))
        positions = []
        for yaw in [base_yaw, base_yaw + np.pi / 2]:
            for along in np.linspace(.035, .34, 13):
                for lateral in np.linspace(-.25, .25, 51):
                    spec = baseline.actor_spec(blocker['name'], blocker['model_id'],
                        center[:2] + along * forward + lateral * side, yaw)
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
                    if .03 <= fraction <= .95:
                        positions.append((abs(fraction - .30), spec))
        positions.sort(key=lambda pair: pair[0])
        best = None
        for _, spec in positions[:120]:
            p = spec['pose']
            actor = env.objects[spec['name']]
            actor.set_pose(sapien.Pose(p[:3], p[3:]))
            actor._bodies[0].linear_velocity = np.zeros(3)
            actor._bodies[0].angular_velocity = np.zeros(3)
            quality, _, _ = scenes.head_quality(env, baseline)
            fraction = quality['occlusion_fraction']
            if scenes.accepted(quality, 'S5_occluded') and .24 <= fraction <= .36:
                blocker['pose'] = spec['pose']
                return True
            if fraction < .20 and (best is None or fraction > best[0]):
                best = fraction, deepcopy(spec)
        if best is not None:
            blocker['pose'] = best[1]['pose']
        p = blocker['pose']
        actor = env.objects[blocker['name']]
        actor.set_pose(sapien.Pose(p[:3], p[3:]))
        actor._bodies[0].linear_velocity = np.zeros(3)
        actor._bodies[0].angular_velocity = np.zeros(3)
    return False


def build(root, resume=False):
    reg = read(root / 'registration.json')
    check_sources(reg)
    db = load_object_catalog(root / 'object_catalog.json')['models']
    environment, baseline, runner, sapien, torch, make, profile, convert, get_maps = original.runtime(db)
    maps = get_maps(profile)
    model = reg['schedule'][0]['model_id']
    probe = make(model, {'layout': 'probe', 'actors': [baseline.actor_spec('target', model, [-.4, 0.], 0.)]})
    try:
        probe.reset(seed=0)
        baseline.settle(probe)
        state = runner.snapshot(probe, probe.get_obs())
        save(root / 'preflight/head_camera.json', state)
    finally:
        probe.close()
    pivot, radius = geometry.reach_bound(state, URDF)
    completed, started, ik = [], time.perf_counter(), None
    try:
        ik = StaticIK(root)
        for entry in reg['schedule']:
            if (root / 'STOP').exists():
                raise RuntimeError('User STOP present; refusing further scene work')
            existing = root / 'cases' / entry['id'] / 'admission.json'
            if existing.exists():
                if not resume:
                    raise RuntimeError('Explicit --resume required for existing cases')
                saved = read(existing)
                assert saved['passed'] and read(existing.parent / 'workspace_check.json')['passed']
                completed.append(saved['entry'])
                continue
            print('CHECK', len(completed) + 1, '/', len(reg['schedule']), entry['id'], flush=True)
            rng = np.random.default_rng(entry['seed'] + 20261122)
            candidates = target_candidates(entry, baseline, state, pivot, radius, rng)
            env, bound, failures = None, None, []
            try:
                for variant, target in enumerate(candidates):
                    if (root / 'STOP').exists():
                        raise RuntimeError('User STOP present')
                    try:
                        if not ik.check(target, db, entry['id'] + f':candidate:{variant}')['passed']:
                            raise RuntimeError('No table-clear preparation-pose IK at candidate position')
                        # Preserve a suitable original layout before searching a replacement.
                        if variant == 0 and target == entry['source_scene']['actors'][0] and all(
                            geometry.placement_check(a, db, state, pivot, radius)['passed']
                            for a in entry['source_scene']['actors']):
                            scene = deepcopy(entry['source_scene'])
                        else:
                            scene = scene_candidate(entry, target, baseline, state, pivot, radius, rng)
                        if env is None:
                            env = make(entry['model_id'], scene)
                            env.reset(seed=entry['seed'])
                        original.scenes.set_scene(env, scene, baseline, sapien, torch)
                        quality, obs, mask = original.scenes.head_quality(env, baseline)
                        if entry['layout'] == 'S5_occluded' and not original.scenes.accepted(quality, entry['layout']):
                            if not adjust_blockers(env, scene, baseline, state, pivot, radius, rng):
                                raise RuntimeError('No physical 20-40% occlusion within the work area')
                            original.scenes.set_scene(env, scene, baseline, sapien, torch)
                            quality, obs, mask = original.scenes.head_quality(env, baseline)
                        if not original.scenes.accepted(quality, entry['layout']):
                            raise RuntimeError('Head observation admission: ' + str(quality))
                        actual = deepcopy(scene)
                        for a in actual['actors']:
                            a['pose'] = env.objects[a['name']].pose.raw_pose[0].cpu().numpy().tolist()
                        actual_state = runner.snapshot(env, obs)
                        actual_pivot, actual_radius = geometry.reach_bound(actual_state, URDF)
                        checks = {a['name']: geometry.placement_check(a, db, actual_state, actual_pivot, actual_radius)
                                  for a in actual['actors']}
                        if not all(c['passed'] for c in checks.values()):
                            raise RuntimeError('Settled actor outside admitted work area or camera view')
                        if entry['layout'] == 'S2_edge' and not geometry.edge_check(actual['actors'][0], db):
                            raise RuntimeError('Target no longer satisfies near-side edge gap')
                        certificate = ik.check(actual['actors'][0], db, entry['id'] + f':{variant}')
                        if not certificate['passed']:
                            raise RuntimeError('No table-clear preparation-pose IK certificate')
                        clean_entry = {k: deepcopy(v) for k, v in entry.items() if k != 'source_scene'}
                        bound = original.scenes.capture(root, clean_entry, env, actual, quality, obs, mask,
                                                        runner, profile, convert, maps)
                        moved = any(np.linalg.norm(np.asarray(a['pose'][:3]) - np.asarray(b['pose'][:3])) > .003
                                    for a, b in zip(actual['actors'], entry['source_scene']['actors']))
                        save(root / 'cases' / entry['id'] / 'workspace_check.json',
                             {'passed': True, 'actors': checks, 'preparation_pose_ik': certificate,
                              'placement_changed': moved, 'source_case': entry['source_case'],
                              'physical_grasps': 0, 'at': now()})
                        break
                    except (RuntimeError, AssertionError) as error:
                        if any(t in str(error).lower() for t in ['cuda', 'memory', 'integrity', 'static ik reply', 'user stop']):
                            raise
                        failures.append({'variant': variant, 'reason': str(error)})
                        save(root / 'search_logs' / (entry['id'] + '.json'), {'rejected': failures})
                if bound is None:
                    raise RuntimeError('No admitted replacement for ' + entry['id'] + ': ' + str(failures[-3:]))
            finally:
                if env is not None:
                    env.close()
            save(root / 'cases' / entry['id'] / 'construction_attempts.json',
                 {'accepted_variant': variant, 'rejected': failures})
            completed.append(bound)
            save(root / 'progress.json', {'phase': 'relocating', 'completed': len(completed),
                 'registered': len(reg['schedule']), 'by_layout': dict(Counter(e['layout'] for e in completed)),
                 'last_case': entry['id'], 'physical_grasps': 0, 'at': now(),
                 'elapsed_seconds': time.perf_counter() - started})
            print('ACCEPTED', len(completed), entry['id'], 'distance',
                  round(checks['target']['center_distance_m'], 3), 'variant', variant, flush=True)
        original.scenes.galleries(root, completed)
        save(root / 'manifest.json', {'registration_sha256': sha(root / 'registration.json'),
                                     'schedule': completed, 'completed': len(completed), 'physical_grasps': 0})
        save(root / 'progress.json', {'phase': 'geometry_complete', 'completed': len(completed),
                                     'registered': len(reg['schedule']), 'physical_grasps': 0, 'at': now()})
    except BaseException:
        save(root / 'halt.json', {'at': now(), 'completed': len(completed), 'error': traceback.format_exc()})
        raise
    finally:
        if ik is not None:
            ik.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'build'])
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root.resolve(), args.development)
    else:
        build(args.root.resolve(), args.resume)


if __name__ == '__main__':
    main()
