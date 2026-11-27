"""Independent count, coverage, planning and fixed-work verification."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import yaml

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.curobo.sphere_study.profiles import digest
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard


COUNTS = [871, 512, 256]


def cover_margin(original, merged, members):
    """Independently check every original ball against its enclosing ball."""
    seen = []
    margins = []
    if len(merged) != len(members):
        raise ValueError('Missing coverage memberships')
    for ball, indices in zip(merged, members):
        for index in indices:
            source = original[index]
            margins.append(ball['radius'] - source['radius'] - float(np.linalg.norm(
                np.asarray(source['center']) - ball['center'])))
            seen.append(index)
    if sorted(seen) != list(range(len(original))) or min(margins) < -1e-12:
        raise ValueError('Original sphere coverage lost')
    return min(margins)


def mask_difference(reference, other):
    a, b = np.asarray(reference, dtype=bool), np.asarray(other, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('Different fixed-work mask shapes')
    return {'states': len(a), 'reference_clear': int(a.sum()), 'other_clear': int(b.sum()),
            'new_blocks': int((a & ~b).sum()), 'new_allows': int((~a & b).sum())}


def cpu_preparation(root, repeats=15):
    """Interleave model operations, separate from the complete RPC samples."""
    configs = {}
    files = {}
    for count in COUNTS:
        path = root/'profiles'/str(count)/'arx_source/scripts/curobo_arx/static_pick/reference/model/arx.yml'
        files[count] = path.read_text()
        configs[count] = yaml.load(files[count], Loader=yaml.CSafeLoader)
    outputs = {}
    for operation in ['C_yaml_parse', 'C_yaml_dump', 'deepcopy', 'sphere_digest']:
        samples = {n: [] for n in COUNTS}
        for repeat in range(repeats + 2):
            order = COUNTS[repeat % 3:] + COUNTS[:repeat % 3]
            for count in order:
                began = time.perf_counter()
                if operation == 'C_yaml_parse':
                    result = yaml.load(files[count], Loader=yaml.CSafeLoader)
                elif operation == 'C_yaml_dump':
                    result = yaml.dump(configs[count], Dumper=yaml.CSafeDumper, sort_keys=False)
                elif operation == 'deepcopy':
                    result = deepcopy(configs[count])
                else:
                    result = digest(configs[count]['robot_cfg']['kinematics']['collision_spheres'])
                seconds = time.perf_counter() - began
                if repeat >= 2:
                    samples[count].append(seconds * 1000)
                if result is None:
                    raise RuntimeError('CPU preparation did not produce an output')
        outputs[operation] = {str(n): {'median_ms': float(np.median(v)),
            'p90_ms': float(np.percentile(v, 90)), 'repeats': len(v)} for n, v in samples.items()}
    result = {'at': now(), 'same_reference_model_operations': True, 'interleaved': True,
        'yaml_bytes': {str(n): len(files[n].encode()) for n in COUNTS}, 'operations': outputs}
    write_json(root/'cpu_preparation.json', result)
    return result


def review(root):
    reg = guard(root)
    if not read_json(root/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Wait for the registered children to exit')
    if not read_json(root/'independent_review.json')['passed']:
        raise RuntimeError('Independent physical-arm endpoint review incomplete')
    progress = read_json(root/'progress.json')
    if progress['phase'] != 'complete' or len(progress['results']) != 32:
        raise RuntimeError('Incomplete full-planning study')
    source = read_json(ROOT/'simulation/robots/arx/collision/arx_refined_871.json')
    profiles = {}
    for count in COUNTS:
        profile = read_json(root/'profiles'/str(count)/'profile.json')
        if digest(profile['collision_spheres']) != profile['sphere_parameter_sha256']:
            raise ValueError('Registered sphere parameters damaged')
        margins = [cover_margin(source['collision_spheres'][name], balls,
            profile['coverage'][name]['members']) for name, balls in profile['collision_spheres'].items()]
        startup = read_json(root/'workers'/str(count)/'startup.json')
        if set(startup['actual_native_counts'].values()) != {count}:
            raise ValueError('Resident native model has the wrong sphere count')
        profiles[str(count)] = {'sphere_count': count, 'counts': {n: len(v) for n, v in
            profile['collision_spheres'].items()}, 'minimum_cover_margin_m': min(margins),
            'maximum_radius_m': max(b['radius'] for v in profile['collision_spheres'].values() for b in v),
            'self_collision_enabled': startup['self_collision_enabled'],
            'startup_seconds': startup['seconds']}
    comparisons = []
    for case in reg['cases']:
        original_pool = read_json(Path(case['source_directory'])/'candidates/candidate_pool.json')
        for count in COUNTS:
            for repeat in [1, 2]:
                out = root/'plans'/case['id']/f's{count}_r{repeat}'
                reference = root/'plans'/case['id']/f's871_r{repeat}'
                if read_json(out/'candidates/candidate_pool.json') != original_pool:
                    raise ValueError('Different candidate pool across counts')
                a, b = [read_json(p/'plan_result.json') for p in [out, reference]]
                if not a['success'] or not b['success']:
                    comparisons.append({'case': case['id'], 'count': count, 'repeat': repeat,
                        'success': a['success'], 'reference_success': b['success']})
                    continue
                binding = a['selected']['width_binding']
                config = yaml.load(Path(binding['robot_model']).read_text(), Loader=yaml.CSafeLoader)
                spheres = config['robot_cfg']['kinematics']['collision_spheres']
                profile = read_json(root/'profiles'/str(count)/'profile.json')
                if digest(spheres) != profile['sphere_parameter_sha256'] or binding['actual_native_sphere_count'] != count:
                    raise ValueError('Selected candidate used a different sphere model')
                comparisons.append({'case': case['id'], 'count': count, 'repeat': repeat,
                    'success': True, 'reference_success': True,
                    'same_grasp_arm_and_width': all(a['selected'][k] == b['selected'][k] for k in
                        ['arm', 'source_generator_index', 'contact_pose_world', 'open_finger_m']),
                    'same_rejections': a['rejected'] == b['rejected'],
                    'max_endpoint_joint_delta_rad': max(float(np.max(np.abs(
                        np.asarray(a['selected'][k]) - b['selected'][k]))) for k in ['q_pre', 'q_grasp', 'q_lift']),
                    'paths_bitwise_equal': a['paths'] == b['paths']})
    profiled = []
    for item in reg['schedule']:
        if not item['diagnostic_profile']:
            continue
        out = root/'plans'/item['case_id']/item['variant']
        plan = read_json(out/'plan_result.json')
        reference = read_json(root/'plans'/item['case_id']/'s871_r1/plan_result.json')
        config = yaml.load(Path(plan['selected']['width_binding']['robot_model']).read_text(),
                           Loader=yaml.CSafeLoader)
        if digest(config['robot_cfg']['kinematics']['collision_spheres']) != source['sphere_parameter_sha256']:
            raise ValueError('Profiler used a different sphere model')
        diagnostic = read_json(out/'full_planning_profile.json')
        profiled.append({'case': item['case_id'], 'sphere_count': 871,
            'same_selected_grasp_arm': all(plan['selected'][k] == reference['selected'][k] for k in
                ['arm', 'source_generator_index', 'contact_pose_world']),
            'profile': diagnostic, 'excluded_from_ordinary_timing': True})
    micro = []
    for ordinal, case in enumerate(reg['cases']):
        directory = root/'micro/871'/case['id']
        result = read_json(directory/'result.json')
        if sha256(directory/'fixed_joints.npy') != result['fixed_joint_sha256']:
            raise ValueError('Fixed joint inputs damaged')
        values = np.load(directory/'fixed_joints.npy', allow_pickle=False)
        if values.shape != (256, 6) or not np.isfinite(values).all():
            raise ValueError('Invalid fixed-work inputs')
        source_dir = ROOT/reg['micro_sources'][case['id']]
        saved = read_json(source_dir/'plan_result.json')
        path = np.concatenate([saved['paths'][stage]['q'] for stage in ['pregrasp', 'approach']])
        expected = path[np.linspace(0, len(path) - 1, 96, dtype=int)].astype(np.float32)
        if not np.array_equal(values[:96], expected) or result['states'] != 256:
            raise ValueError('Fixed-work trajectory sample changed')
        from modules.arx_core.arms import get_arm
        state = read_json(source_dir/'state.json')
        limits = np.asarray([state['joint_limits'][state['joint_names'].index(n)]
                             for n in get_arm(saved['selected']['arm']).joints])
        rng = np.random.default_rng(20261127 + ordinal)
        expected_random = rng.uniform(limits[:, 0] + 1e-4, limits[:, 1] - 1e-4, (160, 6)).astype(np.float32)
        if not np.array_equal(values[96:], expected_random):
            raise ValueError('Fixed-work random samples or joint limits changed')
        differences = {str(count): mask_difference(result['native_world_clear']['871'],
            result['native_world_clear'][str(count)]) for count in COUNTS}
        micro.append({'case': case['id'], 'source_arm': result['arm'], 'states': result['states'],
            'differences': differences, 'measurements': result['measurements'],
            'fixed_joint_sha256': result['fixed_joint_sha256']})
    preserved = {p: sha256(p) == value for p, value in reg['preserve_source_bytes'].items()}
    if not all(preserved.values()):
        raise ValueError('Production configuration changed')
    result = {'at': now(), 'passed': True, 'record_integrity': True,
        'checked_requests': len(comparisons) + len(profiled), 'profiles': profiles,
        'comparisons': comparisons, 'profiled_requests': profiled,
        'micro': micro, 'fixed_work_joint_states_per_count': sum(v['states'] for v in micro),
        'production_bytes_preserved': preserved, 'production_sphere_count': 871,
        'default_activated': False, 'physical_grasps': 0,
        'analysis_source_sha256': sha256(Path(__file__))}
    write_json(root/'sphere_count_review.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    review(root)
    cpu_preparation(root)
