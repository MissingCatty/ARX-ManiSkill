"""Retest the 26 fixed-orientation GraspCL endpoint failures in ManiSkill."""
import argparse
from collections import Counter
from copy import deepcopy
import inspect
from pathlib import Path
import shutil
import sys
import textwrap
from unittest.mock import patch

import graspcl_reliability_20261118_01 as runtime

CODE = runtime.CODE
SELF = Path(__file__).resolve()
PRIMARY = runtime.MAIN
PRIMARY_ARM = PRIMARY / 'fixed_graspcl'
INPUTS = runtime.INPUTS
POLICY = CODE / 'tmp/physical_table_pair_20261118_01.py'
ARM = 'free_graspcl'
GROUPS = {
    '抓取 IK／场景碰撞未通过': 'grasp_endpoint',
    '预抓取或抬升 IK／碰撞未通过': 'pregrasp_or_lift_endpoint',
}
EXPECTED = {'grasp_endpoint': 22, 'pregrasp_or_lift_endpoint': 4}
SERVICE_STARTUP_TIMEOUT_SECONDS = 1200
read, save, sha = runtime.read, runtime.save, runtime.sha


def prepare(out, gpu):
    original = read(PRIMARY / 'graspcl/registration.json')
    assert read(PRIMARY / 'final_verification.json')['passed']
    cases = [row for row in read(PRIMARY / 'independent_case_metrics.json')
             if row['arm'] == 'fixed_graspcl' and row['failure_category'] in GROUPS]
    assert Counter(GROUPS[row['failure_category']] for row in cases) == EXPECTED
    assert len({row['id'] for row in cases}) == 26
    cases.sort(key=lambda row: (GROUPS[row['failure_category']] == 'grasp_endpoint', row['id']))

    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    shutil.copy2(SELF, source / SELF.name)
    inputs = out / 'inputs'
    inputs.mkdir()
    common = inputs / 'common_initial_conditions.json'
    storage = inputs / 'storage_policy.json'
    for path in (common, storage):
        shutil.copy2(INPUTS / path.name, path)

    sources = dict(original['source_hashes'])
    sources.update({str(p): sha(p) for p in (
        SELF, Path(runtime.__file__), runtime.RUNTIME, POLICY, common, storage,
        PRIMARY / 'independent_case_metrics.json', PRIMARY / 'final_verification.json',
    )})
    schedule, selection = [], []
    for row in cases:
        canonical = row['id']
        group = GROUPS[row['failure_category']]
        original_trial = PRIMARY_ARM / 'trials' / canonical
        plan_file = original_trial / 'plan_result.json'
        result_file = original_trial / 'result.json'
        plan = read(plan_file)
        assert sha(result_file) == row['result_sha256']
        assert not plan['success'] and plan['count'] == sum(plan['rejected'].values())
        assert plan['rejected']['closed_collision'] == plan['rejected']['path'] == plan['rejected']['optimizer'] == 0
        assert (plan['rejected']['pre_or_lift_ik'] == 0) == (group == 'grasp_endpoint')

        slot = canonical + '_FREE01'
        admitted = read(INPUTS / 'admission' / canonical / 'admission.json')
        assert admitted['passed'] and admitted['entry']['id'] == canonical
        admitted['entry'].update(id=slot, canonical_case_id=canonical, original_failure_group=group)
        admission = inputs / 'admission' / slot / 'admission.json'
        save(admission, admitted)
        reset = inputs / 'declared_resets' / slot / 'declared_reset.json'
        reset.parent.mkdir(parents=True)
        shutil.copy2(INPUTS / 'declared_resets' / canonical / reset.name, reset)
        sources.update({str(p): sha(p) for p in (plan_file, admission, reset)})
        schedule.append(admitted['entry'])
        selection.append({'id': slot, 'canonical_case_id': canonical, 'original_group': group,
                          'original_trial': str(original_trial), 'original_plan_counts': plan['rejected'],
                          'original_result_sha256': row['result_sha256']})

    selection_file = out / 'selection.json'
    save(selection_file, {'groups': EXPECTED, 'cases': selection, 'outcome_selected': True})
    sources[str(selection_file)] = sha(selection_file)
    arm = out / ARM
    for name in ('trials', 'claims', 'entries', 'workers', 'backend_requests', 'reference', 'resource_wait'):
        (arm / name).mkdir(parents=True)
    shutil.copy2(storage, arm / storage.name)
    arm_common = arm / 'reference' / common.name
    shutil.copy2(common, arm_common)
    (arm / 'resident').symlink_to(out / 'resident', target_is_directory=True)

    from native_pair_bench import now
    reg = deepcopy(original)
    reg.update(created_at=now(), source_hashes=sources, inputs=str(inputs), schedule=schedule,
               arms=[[ARM, 'graspcl_200', 'free_orientation']], total_slots=26,
               attempt_budget_by_arm={ARM: 26}, runtime_backend_gpu=gpu, runtime_scene_gpus=[gpu],
               scene_workers=1, backend_lock=str(out / 'backend.lock'),
               common_reset_source=str(common), admitted=26, admission_failed=0,
               inherited_completed=0, source_evaluation_results_carried=0, prior_test_results_carried=0,
               source_campaign=str(PRIMARY), parallel_suites={},
               algorithm_changed=True, test_design_scope='Targeted retest, selected by original endpoint failure.',
               frozen_algorithm='Use the existing 32-seed actual TCP43 free-orientation proposals. Retain original '
               'orientations, then apply the same bounded insertion truncation, frozen GraspCL reranking, full '
               'pose and scene-checked endpoint/path/trajectory validation, measured closed lift and 1cm/3s success. '
               'Each case resets and captures fresh wrist RGB-D. Read-only endpoint tracing records solver outcomes.')
    save(out / 'registration.json', reg)
    child = deepcopy(reg)
    child.update(suite=str(out), experiment_arm=ARM, pipeline_profile='graspcl_200',
                 orientation_mode='free_orientation', attempt_budget=26, common_reset_source=str(arm_common))
    child['source_hashes'][str(arm_common)] = sha(arm_common)
    save(arm / 'registration.json', child)
    save(arm / 'progress.json', {'phase': 'prepared', 'completed': 0, 'successes': 0, 'total': 26})
    import short_lift_bench
    short_lift_bench.update_suite(out, 'prepared')


def endpoint_rows(calls, plan):
    """Decode the original selector's grasp then (pregrasp, lift) IK calls."""
    rows, i = [], 0
    while i < len(calls):
        reach = calls[i]
        row = {'candidate_index': len(rows), 'grasp': reach, 'pregrasp': None, 'lift': None}
        i += 1
        if reach:
            row['pregrasp'], row['lift'] = calls[i:i + 2]
            i += 2
        rows.append(row)
    rejected = plan['rejected']
    assert sum(not r['grasp'] for r in rows) == rejected['grasp_ik']
    assert sum(r['grasp'] and not (r['pregrasp'] and r['lift']) for r in rows) == rejected['pre_or_lift_ik']
    assert len(rows) == sum(rejected.values()) + bool(plan['success'])
    if plan['success']:
        assert rows[-1]['candidate_index'] == plan['selected']['index']
        assert all(rows[-1][key] for key in ('grasp', 'pregrasp', 'lift'))
    return rows


def install_trace():
    from short_lift_planner import ShortLiftPlanner
    select = ShortLiftPlanner.select_tcp43

    def traced(self, state, rows, camera):
        calls = []
        ik = self.ik

        def record(world, start):
            solution = ik(world, start)
            calls.append(solution is not None)
            return solution

        with patch.object(self, 'ik', record):
            plan = select(self, state, rows, camera)
        save(self.out.parent / 'endpoint_checks.json', {
            'passed_plan': plan['success'], 'candidates': endpoint_rows(calls, plan),
            'scope': 'Initial open-model full-pose IK and scene checks, through first complete plan. '
                     'Null means not tested. Actual closed-model lift is recorded separately.',
        })
        return plan

    ShortLiftPlanner.select_tcp43 = traced


def summarize(out):
    arm = out / ARM
    rows = []
    groups = {group: Counter(registered=count) for group, count in EXPECTED.items()}
    for case in read(out / 'selection.json')['cases']:
        claim = arm / 'claims' / (case['id'] + '.json')
        if not claim.exists():
            continue
        try:
            claim_data = read(claim)
        except ValueError:  # Controller may be writing a live claim.
            continue
        if claim_data['state'] != 'complete':
            continue
        trial = arm / 'trials' / case['id']
        result = read(trial / 'result.json')
        endpoints = trial / 'endpoint_checks.json'
        checks = read(endpoints)['candidates'] if endpoints.exists() else []
        if (trial / 'plan_result.json').exists() and result['reason'] != 'measured_start_state_in_collision':
            assert endpoints.exists()
        stages = result.get('stages', {})
        stage_counts = {
            'grasp_endpoint_pass': any(r['grasp'] for r in checks),
            'grasp_and_pregrasp_pass': any(r['grasp'] and r['pregrasp'] for r in checks),
            'grasp_and_lift_pass': any(r['grasp'] and r['lift'] for r in checks),
            'all_three_endpoints_pass': any(r['grasp'] and r['pregrasp'] and r['lift'] for r in checks),
            'full_initial_plan_pass': result['planning_success'],
            'closure_completed': stages.get('approach', {}).get('settled', False)
                              and stages.get('close', {}).get('settled', False),
            'grasp_success': result['success'],
        }
        group = groups[case['original_group']]
        group['completed'] += 1
        group.update({k: int(v) for k, v in stage_counts.items()})
        rows.append({**case, **stage_counts, 'reason': result['reason'], 'trial': str(trial)})
    save(out / 'diagnostic_summary.json', {'completed': len(rows), 'registered': 26,
         'groups': groups, 'cases': rows, 'verification_complete': (out / 'final_verification.json').exists()})


def configure(reg):
    base = runtime.configure(reg)
    base.__file__ = str(SELF)
    base.report = summarize
    original_backends = base.Backends
    original_start = base.core.Backends.start
    startup_source = textwrap.dedent(inspect.getsource(original_start))
    old_deadline = 'deadline = time.monotonic()+240'
    assert startup_source.count(old_deadline) == 1
    startup_source = startup_source.replace(
        old_deadline, f'deadline = time.monotonic()+{SERVICE_STARTUP_TIMEOUT_SECONDS}')
    startup_namespace = dict(original_start.__globals__)
    exec(compile(startup_source, str(SELF) + ':service_startup_timeout', 'exec'), startup_namespace)
    base.core.Backends.start = startup_namespace['start']

    class TracedBackends(original_backends):
        def start(self):
            popen = base.core.subprocess.Popen

            def start_traced_service(command, *args, **kwargs):
                command = [str(SELF) if str(value) == str(POLICY) else value for value in command]
                return popen(command, *args, **kwargs)

            with patch.object(base.core.subprocess, 'Popen', start_traced_service):
                super().start()

    base.Backends = TracedBackends
    validate = base.existing.validate

    def validate_runtime(out, backends, registration):
        validate(out, backends, registration)
        for mode in ('fixed_orientation', 'free_orientation'):
            warmup = out / 'startup_validation' / mode / Path(registration['warmup_trial']).name
            assert (warmup / 'endpoint_checks.json').exists()
        receipt = read(out / 'startup_validation.json')
        receipt.update(endpoint_tracing_validated=True, nominal_closed_precheck=False,
                       extra_closed_table_margin_m=0.0, collision_models_and_table_margin_unchanged=None)
        save(out / 'startup_validation.json', receipt)

    base.existing.validate = validate_runtime
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'serve', 'status'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--gpu')
    args, remaining = parser.parse_known_args()
    if args.stage == 'serve':
        if remaining[remaining.index('--kind') + 1] == 'curobo':
            runtime.load_runtime().policy().install_collision_policy()
            install_trace()
        from short_lift_services import main as serve
        sys.argv = [str(SELF), *remaining]
        serve()
        return
    assert not remaining and args.output is not None
    out = args.output.resolve()
    if args.stage == 'prepare':
        assert args.gpu
        prepare(out, args.gpu)
    elif args.stage == 'status':
        summarize(out)
    elif args.stage == 'run':
        configure(read(out / 'registration.json')).run(out)
    else:
        configure(read(read(args.entry)['campaign_registration'])).execute(out, args.entry.resolve())


if __name__ == '__main__':
    main()
