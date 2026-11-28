"""Diagnose saved scene-update costs using receipts and CPU mesh loading."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import time
from types import SimpleNamespace

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.curobo.state_updates.scenes import UnionSceneUpdates
from .latency_analysis import verified_read, completed_registration


def summary(values):
    return {'calls': len(values), 'total_seconds': sum(values),
            'mean_seconds': statistics.mean(values), 'median_seconds': statistics.median(values)}


def qualify(scene):
    scene = deepcopy(scene)
    scene['mesh'] = {name + '_' + sha256(g['file_path'])[:20]: g
                     for name, g in scene.get('mesh', {}).items()}
    return scene


def replay_union(folder):
    """Exercise the actual cache policy with real geometry and no CUDA owner."""
    changes = []
    owner = SimpleNamespace(scene_model=SimpleNamespace(
        update_obstacle_pose=lambda *args: None, enable_obstacle=lambda *args: None),
        clear_scene_cache=lambda: None, update_world=lambda scene: None)
    policy = UnionSceneUpdates(lambda scene: scene, lambda pose: pose)
    for arm in ['right', 'left', 'right']:
        source = folder / 'arms' / arm / 'opening_batch_start/scene.json'
        scene = qualify(read_json(source))
        changes.append({'arm': arm, 'source': str(source.relative_to(ROOT)),
                        'source_sha256': sha256(source), **policy.update(owner, scene)})
    return changes


def mesh_profile(state):
    # Only the CPU trimesh part used by cuRobo Mesh.get_mesh_data is exercised.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import trimesh
    import numpy as np
    rows = []
    for name in ['parked_robot_arm', 'target']:
        source = Path(state['scene_meshes'][name]['file_path'])
        expected = state['scene_meshes'][name]['file_sha256']
        if sha256(source) != expected:
            raise ValueError('Saved mesh content changed: ' + str(source))
        elapsed = []
        for _ in range(5):
            began = time.perf_counter()
            mesh = trimesh.load(str(source), process=True, force='mesh')
            vertices, faces = mesh.vertices.view(np.ndarray), mesh.faces
            elapsed.append(time.perf_counter() - began)
        profiler = cProfile.Profile()
        profiler.enable()
        mesh = trimesh.load(str(source), process=True, force='mesh')
        vertices, faces = mesh.vertices.view(np.ndarray), mesh.faces
        profiler.disable()
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).sort_stats('cumulative').print_stats(20)
        with source.open('rb') as stream:
            header = stream.read(512).split(b'end_header')[0].decode('ascii')
        rows.append({'name': name, 'source': str(source.relative_to(ROOT)),
                     'sha256': expected, 'bytes': source.stat().st_size, 'ply_header': header,
                     'processed_vertices': len(vertices), 'faces': len(faces),
                     'cpu_seconds': elapsed, **summary(elapsed), 'cpu_profile': text.getvalue()})
    return rows


def run(root, output):
    completed_registration(root)
    accepted = read_json(root / 'acceptance.json')
    progress = read_json(root / 'progress.json')
    if not accepted['task_complete'] or not accepted['record_integrity']:
        raise ValueError('Completed, accepted records are required')
    excluded = set(accepted['interrupted_timing_cases_excluded'])
    cases = sorted({r['id'] for r in progress['results'] if not r['common_precondition_unexecuted']
                    and not r.get('interrupted') and r['id'] not in excluded})
    groups, per_case = defaultdict(list), []
    for cid in cases:
        claim = read_json(root / 'claims' / (cid + '_optimized.json'))
        if claim['state'] != 'complete':
            raise ValueError('Incomplete claim in paired cohort')
        metric = verified_read(root / 'trials' / cid / 'optimized/initial_metrics.json', claim)
        timed = [e for e in metric['stages']['events'] if e['stage'] == 'binding_scene_update']
        bindings = metric['memory_updates']['events']
        if len(timed) != len(bindings):
            raise ValueError('Scene timings and operation receipts must align')
        replacements = 0
        for event, binding in zip(timed, bindings):
            operations = [o['operation'] for o in binding['scene_operations']]
            if len(operations) != 2 or operations[0] != operations[1]:
                raise ValueError('The two native scene owners need separate accounting')
            groups[operations[0]].append(event['wall_seconds'])
            replacements += operations[0] == 'replace_geometry'
        per_case.append({'case': cid, 'binding_calls': len(bindings), 'replacement_pairs': replacements,
                         'seconds': sum(e['wall_seconds'] for e in timed)})
    result = {'at': now(), 'source_run': str(root.relative_to(ROOT)), 'paired_scenes': len(cases),
              'new_planning_requests': 0, 'new_GPU_calls': 0, 'new_physical_grasps': 0,
              'accounting': 'one timing covers planner scene + independent checker scene + seed reset',
              'operation_groups': {key: {**summary(values), 'mean_per_scene_seconds': sum(values)/len(cases)}
                                   for key, values in groups.items()}, 'per_case': per_case,
              'replacement_count_distribution': dict(Counter(r['replacement_pairs'] for r in per_case))}
    whole = sum(sum(v) for v in groups.values())
    reported = read_json(root / 'latency_breakdown.json')['groups']['optimized']['phases']['initial']
    if abs(whole/len(cases) - reported['mean_named_per_request_seconds']['binding_scene_update']) > 1e-8:
        raise ValueError('Operation accounting differs from the accepted stage total')
    crowded = [r['case'] for r in per_case if r['replacement_pairs'] > 2]
    result['CPU_only_union_replay'] = {cid: replay_union(root / 'trials' / cid / 'optimized') for cid in crowded}
    result['CPU_only_union_control'] = replay_union(root / 'trials' / cases[0] / 'optimized')
    for events in result['CPU_only_union_replay'].values():
        if [e['operation'] for e in events] != ['replace_geometry'] * 3:
            raise ValueError('The crowded-scene cache-capacity explanation was not reproduced')
    if result['CPU_only_union_control'][-1]['operation'] not in ['reuse_scene', 'update_poses', 'update_enabled']:
        raise ValueError('The small-scene cache control must reuse geometry across coordinate changes')
    result['CPU_mesh_profile'] = mesh_profile(read_json(root / 'trials' / cases[0] / 'optimized/arms/right/state.json'))
    result['source_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in [
        root/'registration.json', root/'acceptance.json', root/'progress.json', Path(__file__),
        ROOT/'modules/curobo/state_updates/binding.py', ROOT/'modules/curobo/state_updates/scenes.py',
        ROOT/'modules/curobo/dual_arm/model.py',
        ROOT/'modules/curobo/source/curobo/_src/geom/data/data_mesh.py']}
    write_json(output / 'diagnosis.json', result)
    print(json.dumps({'output': str(output.relative_to(ROOT)), 'paired_scenes': len(cases),
                     'stage_mean_seconds': whole/len(cases),
                     'replacement_share': sum(groups['replace_geometry'])/whole,
                     'crowded_cases': crowded, 'new_GPU_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(ensure_run_path(args.root), ensure_run_path(args.output))
