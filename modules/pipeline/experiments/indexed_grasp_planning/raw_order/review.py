"""Reconcile original-order identity, actual settings, FK, and latency gates."""
import numpy as np

from arx_project.layout import now,sha256,write_json
from modules.pipeline.experiments.graspgenx_postprocessing.planning_review import review as original_review
from ..physical.registration import guard,raw
from .proxy import check_original_order


def summarize(rows):
    groups = {}
    for variant,mode in [('control_1','baseline'),('optimized_1','optimized')]:
        entries = [r['variants'][variant] for r in rows]
        times = [r['initial_planning_with_postprocessing_seconds'] for r in entries]
        groups[mode] = {'requests':len(entries),'successes':sum(r['success'] for r in entries),
            'total_seconds':sum(times),'median_seconds':float(np.median(times)),
            'p90_seconds':float(np.percentile(times,90))}
    return groups


def planning_review(root):
    reg = guard(root);accepted = raw(root/'acceptance.json')
    for case in reg['cases']:
        for variant,mode in reg['variant_modes'].items():
            directory = root/'plans'/case['id']/variant
            claim = raw(root/'claims'/(case['id']+'_'+variant+'.json'))
            if claim['state'] != 'complete': raise ValueError('Planning slot incomplete')
            for name,digest in claim['evidence_sha256'].items():
                if sha256(directory/name) != digest: raise ValueError('Planning receipt changed: '+name)
            check_original_order(raw(directory/'candidate_order.json'))
            receipt = raw(directory/'execution_planner_initial.json');enabled = mode == 'optimized'
            if (receipt['variant'] != mode or receipt['batch_capacity'] != (16 if enabled else 0)
                    or receipt['C_YAML_enabled'] != enabled or receipt['obstacle_major_queries'] != enabled
                    or not receipt['scalar_branch_validation'] or not receipt['geometry_and_thresholds_unchanged']):
                raise ValueError('Actual planning variant differs from the registered comparison')
    result = original_review(root)
    geometry = raw(root/'resident/layout_geometry_preflight/acceptance.json')
    ready = raw(root/'resident/layout_readiness.json')
    if not (geometry['passed'] and geometry['full_collision_cost_and_gradient_equal']
            and geometry['checked_spheres']==8919040 and ready['passed']
            and ready['original_and_fast_full_IK_have_separate_CUDA_graphs']):
        raise ValueError('Original exact collision-layout preflight failed')
    groups = summarize(accepted['cases']);a,b=groups['baseline'],groups['optimized']
    saving = 1-b['total_seconds']/a['total_seconds']
    gates = {'total_reduced_5pct':saving>=.05,'median_reduced':b['median_seconds']<a['median_seconds'],
        'p90_not_more_than_5pct_higher':b['p90_seconds']<=a['p90_seconds']*1.05}
    result.update(at=now(),original_order_verified=True,
        original_grasps_and_arms_retained=accepted['original_grasps_and_arms_retained'],
        joint_endpoints_verified=True,
        joint_endpoints_exactly_retained=all(r['comparison']['same_joint_endpoints'] for r in accepted['cases']),
        candidate_decisions_exactly_retained=all(r['comparison']['same_candidate_decisions'] for r in accepted['cases']),
        groups=groups,planning_total_time_reduction=saving,timing_gates=gates,
        timing_gate_passed=all(gates.values()),NMS_enabled=False,automatic_default_activation=False)
    write_json(root/'independent_review.json',result)
    return result


def physical_review(root):
    from ..physical.review import review
    checked = review(root);reg = guard(root)
    for item in reg['schedule']:
        directory = root/'trials'/item['case_id']/item['variant']
        check_original_order(raw(directory/'candidate_order.json'))
        if raw(directory/'candidate_preparation.json')['NMS_enabled']:
            raise ValueError('NMS was enabled in the original-order physical comparison')
    compared = []
    for case in reg['cases']:
        selections = [raw(root/'trials'/case['id']/mode/'plan_result.json').get('selected',{})
                      for mode in ['baseline','optimized']]
        same = all(selections[0].get(k)==selections[1].get(k)
                   for k in ['original_source_index','arm','contact_pose_world'])
        compared.append({'id':case['id'],'same_grasp_and_arm':same})
    checked.update(original_order_verified=True,original_grasps_and_arms_retained=all(r['same_grasp_and_arm'] for r in compared),
        selection_comparisons=compared,NMS_enabled=False)
    checked['optimization_accepted'] &= checked['original_grasps_and_arms_retained']
    write_json(root/'acceptance.json',checked)
    return checked
