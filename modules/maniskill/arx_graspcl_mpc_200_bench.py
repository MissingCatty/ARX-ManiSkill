"""Run the validated wrist MPC chain on all 200 frozen ManiSkill ARX cases."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import arx_route_mpc_wrist_bench as wrist

SELF = Path(__file__).resolve()
CODE = SELF.parent
PRIMARY = CODE / 'outputs/arx_physical_table_parallel_20261118_01'
PROVEN = CODE / 'outputs/arx_graspcl_wrist_mpc_retest26_20261119_01'
RECORD = CODE / '抓取基线批次记录.md'
ARM = 'free_graspcl_mpc'
read, save, sha = wrist.benchmark.read, wrist.benchmark.save, wrist.benchmark.sha


def replace_record(content):
    mode = RECORD.stat().st_mode
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=RECORD.parent, delete=False) as temporary:
        temporary.write(content)
        patch_path = Path(temporary.name)
    try:
        patch_path.chmod(mode)
        patch_path.replace(RECORD)
    finally:
        patch_path.unlink(missing_ok=True)


def marker(out):
    return f'<!-- {out.name} result -->'


def register_record(out):
    content = RECORD.read_text()
    heading = f'## {out.name}'
    assert heading not in content.splitlines(), 'Batch record already exists'
    location = out.relative_to(CODE)
    replace_record(content.rstrip() + f'''

{heading}

- 场景为原独立评测的同一组 200 例，五类各 40 例。193 例进入仿真，7 例保留原初始化失败。
- 腕部 RGB-D → ZeroGrasp 原生后处理 → 冻结 GraspCL 召回和重评分 → 非严格朝向 cuRobo 规划 → 平滑 MPC 跟踪和腕部精调 → ARX 闭爪与抬升。首例前预热，整批使用一组常驻组件。
- 产物见[场景登记]({location}/comparison_cases.json)和[批次进度]({location}/progress.json)。每例的抓取位姿图在 `zg_outputs/`。

{marker(out)}
已登记，结果将在整批通过核验后更新。
{marker(out)}
''')


def prepare(out, backend_gpu, scene_gpu):
    assert read(PRIMARY / 'final_verification.json')['passed']
    assert read(PROVEN / 'final_verification.json')['passed']
    assert read(PROVEN / 'suite_timing.json')['all_owned_services_exited']
    template = read(PROVEN / 'registration.json')
    baseline = read(PRIMARY / 'graspcl/registration.json')
    for key in ('bank', 'source_feedback_ids_in_memory_order', 'test_object_ids',
                'action_tcp_profile', 'collision_policy', 'asset_sha256', 'planned_lift_m'):
        assert template[key] == baseline[key], key
    evidence_path = PRIMARY / 'independent_case_metrics.json'
    old_rows = {row['id']: row for row in read(evidence_path) if row['arm'] == 'fixed_graspcl'}
    initial_inputs = Path(baseline['inputs'])
    assert len(old_rows) == len(baseline['schedule']) == 200
    assert set(Counter(row['layout'] for row in baseline['schedule']).values()) == {40}

    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PROVEN / 'source', out / 'source')
    shutil.copy2(SELF, out / 'source' / SELF.name)
    inputs = out / 'inputs'
    inputs.mkdir()
    arm = out / ARM
    for name in ('trials', 'claims', 'entries', 'workers', 'backend_requests', 'reference', 'resource_wait'):
        (arm / name).mkdir(parents=True)
    (arm / 'resident').symlink_to(out / 'resident', target_is_directory=True)
    common, storage = inputs / 'common_initial_conditions.json', inputs / 'storage_policy.json'
    for path in (common, storage):
        shutil.copy2(initial_inputs / path.name, path)
    arm_common = arm / 'reference' / common.name
    shutil.copy2(common, arm_common)
    shutil.copy2(storage, arm / storage.name)

    sources = dict(template['source_hashes'])
    sources.update({str(path): sha(path) for path in (
        SELF, common, storage, evidence_path, PRIMARY / 'final_verification.json',
        PROVEN / 'registration.json', PROVEN / 'final_verification.json',
    )})
    schedule, pairs = [], []
    admitted = 0
    for canonical in baseline['schedule']:
        case = canonical['id']
        slot = case + '_MPC200_01'
        reference = PRIMARY / 'fixed_graspcl/trials' / case
        before = old_rows[case]
        assert sha(reference / 'result.json') == before['result_sha256']
        admission = read(initial_inputs / 'admission' / case / 'admission.json')
        assert all(admission['entry'][k] == canonical[k] for k in ('id', 'model_id', 'layout', 'seed'))
        admission['entry'].update(id=slot, canonical_case_id=case)
        saved_admission = inputs / 'admission' / slot / 'admission.json'
        save(saved_admission, admission)
        sources[str(saved_admission)] = sha(saved_admission)
        if admission['passed']:
            reset = inputs / 'declared_resets' / slot / 'declared_reset.json'
            reset.parent.mkdir(parents=True)
            shutil.copy2(initial_inputs / 'declared_resets' / case / reset.name, reset)
            sources[str(reset)] = sha(reset)
            admitted += 1
        schedule.append(admission['entry'])
        pairs.append({'id': slot, 'canonical_case_id': case, 'layout': canonical['layout'],
                      'admitted': admission['passed'], 'reference_trial': str(reference),
                      'reference_planning_pass': before['planning_success'],
                      'reference_success': before['success'], 'reference_reason': before['reason']})
    assert admitted == 193
    selection = out / 'comparison_cases.json'
    save(selection, {'cases': pairs, 'scope': 'All 200 original reset recipes. Historical comparison uses fixed orientation, no MPC.'})
    sources[str(selection)] = sha(selection)

    from native_pair_bench import now
    n = len(schedule)
    registration = deepcopy(template)
    registration.update(created_at=now(), source_hashes=sources, inputs=str(inputs), schedule=schedule,
        arms=[[ARM, 'graspcl_200', 'free_orientation']], total_slots=n, attempt_budget_by_arm={ARM: n},
        runtime_backend_gpu=backend_gpu, runtime_scene_gpus=[scene_gpu], scene_workers=1,
        common_reset_source=str(common), backend_lock=str(out / 'backend.lock'),
        admitted=admitted, admission_failed=n-admitted, source_campaign=str(PRIMARY),
        validated_mpc_campaign=str(PROVEN), inherited_completed=0,
        test_design_scope='All 200 frozen scenes, one fresh attempt each, including original admission failures.',
        frozen_algorithm='Validated smooth global-route tracking and wrist-refined MPC from the 26-case batch.',
        result_record=str(RECORD))
    save(out / 'registration.json', registration)
    child = deepcopy(registration)
    child.update(suite=str(out), common_reset_source=str(arm_common), attempt_budget=n,
                 experiment_arm=ARM, pipeline_profile='graspcl_200', orientation_mode='free_orientation')
    child['source_hashes'][str(arm_common)] = sha(arm_common)
    save(arm / 'registration.json', child)
    progress = {'phase': 'prepared', 'completed': 0, 'successes': 0, 'total': n}
    save(arm / 'progress.json', progress)
    save(out / 'progress.json', {'phase': 'prepared', 'completed': 0, 'registered': n, 'rounds': {ARM: progress}})
    wrist.benchmark.summarize(out)
    register_record(out)


REASONS = {
    'independent_scene_admission_failed': '初始化未通过',
    'no_feasible_native_post_or_recalled_candidate': '初始规划未找到可行候选',
    'measured_start_state_in_collision': '实测起点碰撞',
    'mpc_approach_not_converged': '腕部精调未收敛',
    'no_bilateral_target_contact': '未形成双指接触',
    'close_not_converged': '闭爪未收敛',
    'forbidden_robot_contact': '执行中触发禁止接触',
}


def update_record(out, lines):
    content = RECORD.read_text()
    tag = marker(out)
    before, section, after = content.split(tag)
    replace_record(before + tag + '\n' + '\n'.join(lines) + '\n' + tag + after)


def finalize(out):
    verify = read(out / 'final_verification.json')
    clock = read(out / 'suite_timing.json')
    assert verify['passed'] and verify['terminal_results'] == 200 and clock['all_owned_services_exited']
    summary = read(out / 'route_mpc_results.json')
    cases = summary['cases']
    assert summary['verified'] and summary['completed'] == len(cases) == 200
    failures = Counter(row['reason'] for row in cases if not row['success'])
    assert failures['independent_scene_admission_failed'] == 7
    layout_counts = {}
    for layout in dict.fromkeys(row['layout'] for row in cases):
        group = [row for row in cases if row['layout'] == layout]
        layout_counts[layout] = {'completed': len(group),
            'planned': sum(row['planning_pass'] for row in group),
            'wrist_reached': sum(bool(row['mpc'].get('settled')) for row in group),
            'success': sum(row['success'] for row in group)}
    previous_plans = sum(row['reference_planning_pass'] for row in cases)
    previous_success = sum(row['reference_success'] for row in cases)
    paired = Counter(('success' if row['reference_success'] else 'failure') + '_to_' +
                     ('success' if row['success'] else 'failure') for row in cases)
    save(out / 'batch_summary.json', {'verified': True, 'completed': 200, 'admission_failed': 7,
         'planned': summary['planning_pass'], 'wrist_reached': summary['mpc_settled'],
         'successes': summary['success'], 'failure_reasons': dict(failures),
         'by_layout': layout_counts, 'paired_with_original_fixed_orientation': dict(paired),
         'shared_startup_seconds': clock['shared_startup_seconds'],
         'wall_seconds': clock['total_suite_wall_seconds']})
    why = '，'.join(f'{REASONS.get(key, key)} {count} 例' for key, count in failures.most_common())
    total = clock['total_suite_wall_seconds']
    location = out.relative_to(CODE)
    update_record(out, [
        f"- 200/200 完成并通过核验。初始规划 {summary['planning_pass']} 例，腕部到位 {summary['mpc_settled']} 例，抓取成功 {summary['success']} 例（{summary['success'] / 200:.1%}）。",
        f'- 同组原固定朝向基线通过规划 {previous_plans} 例，抓取成功 {previous_success} 例。两轮同时存在朝向搜索和执行控制差异。',
        f'- 失败分布为{why}。',
        f"- 整批用时 {total / 60:.1f} 分钟，预热 {clock['shared_startup_seconds']:.1f} 秒。[逐例结果]({location}/route_mpc_results.json) · [分场景汇总]({location}/batch_summary.json) · [耗时]({location}/suite_timing.json)",
    ])


def main():
    stage = sys.argv[1]
    if stage == 'prepare':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--backend-gpu', required=True)
        parser.add_argument('--scene-gpu', required=True)
        args = parser.parse_args(sys.argv[2:])
        prepare(args.output.resolve(), args.backend_gpu, args.scene_gpu)
        return
    if stage not in ('run', 'report'):
        with patch.object(wrist, 'SELF', SELF):
            wrist.main()
        return
    out = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    if stage == 'run':
        try:
            with patch.object(wrist, 'SELF', SELF):
                wrist.main()
        except Exception:
            progress = read(out / 'progress.json')
            location = out.relative_to(CODE)
            update_record(out, [f"批次中止，已完成 {progress['completed']}/200。 [进度]({location}/progress.json)"])
            raise
    finalize(out)


if __name__ == '__main__':
    main()
