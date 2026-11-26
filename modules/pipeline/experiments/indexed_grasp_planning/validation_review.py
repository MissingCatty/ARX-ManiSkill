"""Independent candidate/FK review plus measured branch and trace receipts."""
import argparse
from pathlib import Path

from arx_project.layout import ensure_run_path, now, sha256, write_json
from modules.pipeline.experiments.graspgenx_postprocessing.planning import raw, verify
from modules.pipeline.experiments.graspgenx_postprocessing.planning_review import review, check_receipt


def check_validation_receipt(directory, claim):
    check_receipt(directory, claim)
    for key, name in [('timing_sha256', 'stage_timings.json'),
                      ('validation_metrics_sha256', 'validation_metrics.json')]:
        if key not in claim or sha256(directory/name) != claim[key]:
            raise ValueError('Missing or changed V8 timing receipt: '+name)
    if set(claim.get('arm_results_sha256', {})) != {'right', 'left'}:
        raise ValueError('Incomplete two-arm validation evidence')
    for side, digest in claim['arm_results_sha256'].items():
        if sha256(directory/'arms'/side/'plan_result.json') != digest:
            raise ValueError('Arm validation result changed: '+side)


def validate(root):
    reg = raw(root/'registration.json');verify(reg)
    accepted = raw(root/'acceptance.json')
    if (root/'STOP').exists() or not accepted['task_complete']:
        raise ValueError('The stopped or incomplete pilot cannot pass validation')
    for case in reg['cases']:
        for variant in reg['variants']:
            directory = root/'plans'/case['id']/variant
            check_validation_receipt(directory, raw(root/'claims'/(case['id']+'_'+variant+'.json')))
            metrics = raw(directory/'metrics.json');trace = raw(directory/'stage_timings.json')
            if trace['stages'] != metrics['stage_summary']:
                raise ValueError('Timing summary differs from the actual trace')
            decisions = {}
            for side in ['right', 'left']:
                arm = raw(directory/'arms'/side/'plan_result.json')
                decisions[side] = [{k: e[k] for k in ['candidate_index', 'kind', 'kinematic_solutions', 'reasons'] if k in e}
                                   for e in arm.get('candidate_events', [])]
            if decisions != metrics['event_decisions']:
                raise ValueError('Recorded branch decisions differ from the actual arm evidence')
            selected = raw(directory/'plan_result.json').get('selected', {})
            if metrics['joint_endpoints'] != {k: selected.get(k) for k in ['q_pre', 'q_grasp', 'q_lift']}:
                raise ValueError('The measured endpoint comparison lost its actual joint values')
    reviewed = review(root)
    if not accepted['all_original_selections_and_branches_retained']:
        raise ValueError('Original choices or first passing branches changed')
    result = {**reviewed, 'at': now(), 'all_original_selections_and_branches_retained': True,
        'timing_and_arm_receipts_verified': True, 'stage_timing_inclusive_and_exclusive': True,
        'planning_total_time_reduction': accepted['planning_total_time_reduction'],
        'timing_gate_passed': accepted['timing_gate_passed'], 'automatic_full_comparison': False}
    write_json(root/'independent_review.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--root', type=Path, required=True)
    print(validate(ensure_run_path(parser.parse_args().root)))
