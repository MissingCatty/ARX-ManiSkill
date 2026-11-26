"""Fixed engineering cohort and one-shot execution/record requirements."""
from pathlib import Path

CASES = [
    'M0020_S1_open_donut', 'M0034_S2_edge_scissors', 'M0074_S4_clutter_scissors',
    'M0084_S5_occluded_banana', 'M0085_S5_occluded_bar_soap', 'M0099_S5_occluded_cucumber',
    'M0001_S1_open_apple', 'M0044_S3_sparse_banana', 'M0056_S3_sparse_cell_phone',
    'M0064_S4_clutter_mango',
]
VARIANTS = ['baseline', 'optimized']
EXECUTION_FILES = ['execution.jsonl', 'execution.mp4', 'head.mp4', 'stages.json',
    'contact_entities.json', 'execution_physics.json', 'execution_physics_metrics.json']


def changed_cases(accepted):
    return [row['id'] for row in accepted['cases'] if any(
        row['variants']['control'][key] != row['variants']['nms15'][key]
        for key in ['selected_arm', 'selected_original_source_index'])]


def schedule(cases):
    if len(cases) != 10 or len(set(cases)) != 10:
        raise ValueError('Exactly ten distinct fixed scenes are required')
    return [{'id': case+'_'+mode, 'case_id': case, 'variant': mode}
        for i, case in enumerate(cases) for mode in (VARIANTS if i%2 == 0 else VARIANTS[::-1])]


def check_trial(directory, result):
    from arx_project.json_io import read_json
    directory = Path(directory)
    for name in ['result.json', 'independent_evaluation.json', 'paired_restore_verification.json',
                 'candidate_order.json', 'latency.json', 'execution_planner_initial.json']:
        path = directory/name
        if not path.is_file() or not path.stat().st_size: raise ValueError('Missing trial evidence: '+name)
    if not read_json(directory/'independent_evaluation.json')['passed']:
        raise ValueError('Independent physical evaluation did not pass')
    if not read_json(directory/'paired_restore_verification.json')['passed']:
        raise ValueError('Frozen initial state was not restored')
    if (directory/'infrastructure_error.json').exists(): raise RuntimeError('Execution infrastructure fault')
    if result.get('planning_success'):
        if not read_json(directory/'prepared_plan_execution_gate.json')['passed']:
            raise ValueError('Prepared plan handoff did not pass')
    if result.get('physical_attempt'):
        for name in EXECUTION_FILES:
            path = directory/name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError('Missing executed action/video evidence: '+name)
    stages = check_executed_stages(directory)
    if stages and not result.get('physical_attempt'):
        raise ValueError('Actual execution was incorrectly reported as unexecuted')


def check_executed_stages(directory):
    import json
    directory = Path(directory);stages = set()
    trace = directory/'execution.jsonl'
    if trace.exists():
        with trace.open() as stream:
            for line in stream:
                if line.strip(): stages.add(json.loads(line)['stage'])
    for stage, before, receipt, plan in [
        ('approach', 'pregrasp_state.json', 'execution_planner_approach.json', 'approach_plan_result.json'),
        ('lift', 'closed_state.json', 'execution_planner_lift.json', 'lift_plan_result.json')]:
        happened = (stage in stages or stage == 'lift' and 'hold' in stages
            or (directory/before).exists() or (directory/('planner_rpc_'+stage+'.json')).exists())
        if happened:
            for name in [before, receipt, plan]:
                if not (directory/name).is_file():
                    raise ValueError('Executed/requested '+stage+' stage lost required record: '+name)
    return stages
