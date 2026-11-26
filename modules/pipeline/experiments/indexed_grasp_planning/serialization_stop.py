"""Read-only accounting for a resource-stopped serialization comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from arx_project.layout import ROOT, ensure_run_path, now, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.postprocessing.ordering import validate_order
from .validation_replay import compare_pair, summarize, verify
from .validation_review import check_validation_receipt


def raw(path): return json.loads(Path(path).read_text())


def review_stopped(run, output):
    if output.exists(): raise ValueError('Preserve earlier stop reviews')
    if not (run/'STOP').exists(): raise ValueError('Resource stop receipt is required')
    reg = raw(run/'registration.json');verify(reg)
    output.mkdir(parents=True)
    write_json(output/'registration.json', {'at': now(), 'source_run': str(run.relative_to(ROOT)),
        'source_sha256': sha256(__file__), 'original_STOP_sha256': sha256(run/'STOP'),
        'planning_requests': 0, 'physical_grasps': 0})
    model = ArmModel();endpoints = [];pairs = [];completed = []
    counts = {mode: {'complete': 0, 'planning_passes': 0, 'resource_gate_not_executed': 0,
                     'unclaimed': 0} for mode in reg['variant_modes'].values()}
    yaml_cache = {};yaml_pairs = 0;candidate_rows = 0
    def yaml_value(path):
        digest = sha256(path)
        if digest not in yaml_cache:
            yaml_cache[digest] = json.dumps(yaml.load(path.read_text(), Loader=yaml.CSafeLoader),
                sort_keys=True, allow_nan=False, separators=(',', ':'))
        return yaml_cache[digest]
    stop = raw(run/'STOP')
    for case in reg['cases']:
        values = {};original = raw(Path(case['source_directory'])/'candidates/candidate_pool.json')['candidates']
        for variant in reg['variants']:
            mode = reg['variant_modes'][variant];claim_path = run/'claims'/(case['id']+'_'+variant+'.json')
            directory = run/'plans'/case['id']/variant
            if not claim_path.exists(): counts[mode]['unclaimed'] += 1;continue
            claim = raw(claim_path)
            if claim['state'] != 'complete':
                trace = raw(directory/'stage_timings.json')
                if 'Insufficient original CUDA reserve' not in stop['error'] or trace['events'] or trace['stages'] \
                        or (directory/'arms').exists() or (directory/'plan_result.json').exists():
                    raise ValueError('Claimed request cannot be classified as a pre-planning resource gate')
                counts[mode]['resource_gate_not_executed'] += 1;continue
            check_validation_receipt(directory, claim)
            if sha256(directory/'state.json') != sha256(Path(case['source_directory'])/'state.json'):
                raise ValueError('Measured input state differs from its registered source')
            metrics = raw(directory/'metrics.json');plan = raw(directory/'plan_result.json')
            manifest = raw(directory/'candidate_order.json');queue = raw(directory/'candidates/candidate_pool.json')
            order = validate_order(len(original), manifest['representatives'], manifest['deferred'])
            if order != manifest['priority_to_original_source'] or len(queue['candidates']) != len(original):
                raise ValueError('Original candidate mapping changed')
            if any(queue['candidates'][rank] != original[index] for rank, index in enumerate(order)):
                raise ValueError('Original candidate geometry, score or metadata changed')
            candidate_rows += len(original)
            if metrics['success'] != plan['success']: raise ValueError('Planning success receipts disagree')
            counts[mode]['complete'] += 1;counts[mode]['planning_passes'] += int(plan['success'])
            values[mode] = metrics;completed.append({'case': case['id'], 'variant': variant})
            if not plan['success']: continue
            selected = plan['selected'];arm = get_arm(selected['arm']);state = raw(directory/'state.json')
            index = order[selected['index']]
            if selected['original_source_index'] != index or selected['contact_pose_world'] != original[index]['pose']:
                raise ValueError('Selected original grasp changed')
            for stage, key in [('pre', 'pregrasp_world'), ('grasp', 'grasp_world'), ('lift', 'lift_world')]:
                actual = deepcopy(state);actual['links'] = {'base_link': state['links']['base_link']}
                q = np.asarray(selected['q_'+stage]);clear = True
                for name, value in zip(arm.joints, q):
                    i = state['joint_names'].index(name);lo, hi = state['joint_limits'][i]
                    clear = clear and bool(lo <= value <= hi);actual['qpos'][i] = float(value)
                reached = model.frames(actual)[arm.eef];goal = np.asarray(selected[key])
                pe = float(np.linalg.norm(reached[:3, 3]-goal[:3, 3]))
                re = float(Rotation.from_matrix(reached[:3, :3]@goal[:3, :3].T).magnitude())
                if not clear or not np.isfinite([pe, re]).all() or pe > .002 or re > .03:
                    raise ValueError('Independent physical-arm FK or limits failed')
                endpoints.append({'case': case['id'], 'variant': variant, 'stage': stage,
                                  'position_error_m': pe, 'rotation_error_rad': re})
        if len(values) == 2:
            comparison = compare_pair(values, 'fast_binding')
            if not all(comparison.values()): raise ValueError('Paired original decisions changed')
            control = run/'plans'/case['id']/'control_1';fast = run/'plans'/case['id']/'fast_binding_1'
            paths = {str(p.relative_to(control)) for p in control.rglob('*.yml')}
            if paths != {str(p.relative_to(fast)) for p in fast.rglob('*.yml')}:
                raise ValueError('Generated YAML coverage differs')
            for name in paths:
                if yaml_value(control/name) != yaml_value(fast/name): raise ValueError('Generated YAML values differ')
                yaml_pairs += 1
            pairs.append({'id': case['id'], 'variants': {mode+'_1': value for mode, value in values.items()},
                          'comparison': comparison})
    if sum(sum(v[k] for k in ['complete', 'resource_gate_not_executed', 'unclaimed']) for v in counts.values()) != reg['planning_requests']:
        raise ValueError('Stopped request budget does not balance')
    verify(reg)
    summaries = summarize(pairs, reg['variant_modes'], reg['primary_time_metric'])
    result = {'at': now(), 'task_complete': False, 'completed_record_integrity': True,
        'source_run': str(run.relative_to(ROOT)), 'reason': 'CUDA_reserve_below_1536_MiB',
        'completed_requests': len(completed), 'complete_pairs': len(pairs), 'counts': counts,
        'paired_summaries': summaries,
        'paired_total_time_reduction': 1-summaries['fast_binding']['total_seconds']/summaries['control']['total_seconds'],
        'checked_endpoints': len(endpoints), 'checked_original_candidate_rows': candidate_rows,
        'generated_YAML_pairs': yaml_pairs, 'paired_decisions_identical': True,
        'resource_gate_before_planner_verified': True, 'original_STOP_preserved': True,
        'new_planning_requests': 0, 'physical_grasps': 0, 'automatic_retry': False}
    write_json(output/'endpoints.json', endpoints);write_json(output/'summary.json', result)
    write_json(run/'stop_summary.json', result)
    pointer = ROOT/'tmp/indexed_grasp_planning/funnel_active.json';active = raw(pointer)
    active.update(at=now(), phase='binding_serialization_stopped_resource', stop_review=str(output.relative_to(ROOT)))
    write_json(pointer, active)
    print({k: v for k, v in result.items() if k != 'paired_summaries'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();review_stopped(ensure_run_path(args.run), ensure_run_path(args.output))
