"""Registered ManiSkill trial of cuRobo MPC for the last 3 cm of ARX approach."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import arx_mpc_approach as mpc

CODE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(CODE / 'tmp'))
import graspcl_relaxed_pose_20261118_01 as relaxed

SOURCE = CODE / 'outputs/arx_graspcl_relaxed_pose_20261118_01'
ARM = 'free_graspcl_mpc'
PILOT = ('I0117', 'I0031', 'I0125')
read, save, sha = relaxed.read, relaxed.save, relaxed.sha


def prepare(out, backend_gpu, scene_gpu, case_ids):
    previous = read(SOURCE / 'registration.json')
    assert read(SOURCE / 'final_verification.json')['passed']
    prior_cases = read(SOURCE / 'diagnostic_summary.json')['cases']
    originals = {row['canonical_case_id'].split('_')[0]: row for row in prior_cases}
    if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= originals.keys():
        raise ValueError('Pass unique I#### case IDs from the completed 26-scene test')
    cases = [originals[case] for case in case_ids]

    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    module = Path(mpc.__file__).resolve()
    for path in (SELF, module):
        shutil.copy2(path, source / path.name)
    inputs = out / 'inputs'
    inputs.mkdir()
    common = inputs / 'common_initial_conditions.json'
    storage = inputs / 'storage_policy.json'
    for path in (common, storage):
        shutil.copy2(relaxed.INPUTS / path.name, path)

    sources = dict(previous['source_hashes'])
    native = CODE.parent / 'curobo/source/curobo'
    mpc_sources = (native / 'model_predictive_control.py', native / '_src/solver/solver_mpc.py',
                   native / '_src/solver/solver_mpc_cfg.py', native / 'content/configs/task/mpc/lbfgs_mpc.yml',
                   native / 'content/configs/task/mpc/transition_bspline_mpc.yml')
    sources.update({str(p): sha(p) for p in (SELF, module, common, storage,
                    SOURCE / 'diagnostic_summary.json', SOURCE / 'final_verification.json', *mpc_sources)})
    schedule, pairs = [], []
    for case in cases:
        canonical = case['canonical_case_id']
        slot = canonical + '_MPC01'
        original = read(relaxed.INPUTS / 'admission' / canonical / 'admission.json')
        assert original['passed'] and original['entry']['id'] == canonical
        original['entry'].update(id=slot, canonical_case_id=canonical)
        admission = inputs / 'admission' / slot / 'admission.json'
        save(admission, original)
        reset = inputs / 'declared_resets' / slot / 'declared_reset.json'
        reset.parent.mkdir(parents=True)
        shutil.copy2(relaxed.INPUTS / 'declared_resets' / canonical / reset.name, reset)
        sources.update({str(p): sha(p) for p in (admission, reset)})
        schedule.append(original['entry'])
        pairs.append({'id': slot, 'canonical_case_id': canonical, 'reference_trial': case['trial'],
                      'reference_planning_pass': case['full_initial_plan_pass'],
                      'reference_success': case['grasp_success'], 'reference_reason': case['reason']})

    selection = out / 'comparison_cases.json'
    save(selection, {'cases': pairs, 'scope': 'Outcome-selected engineering pilot, not efficacy estimate.'})
    sources[str(selection)] = sha(selection)
    arm = out / ARM
    for directory in ('trials', 'claims', 'entries', 'workers', 'backend_requests', 'reference', 'resource_wait'):
        (arm / directory).mkdir(parents=True)
    shutil.copy2(storage, arm / storage.name)
    arm_common = arm / 'reference' / common.name
    shutil.copy2(common, arm_common)
    (arm / 'resident').symlink_to(out / 'resident', target_is_directory=True)

    from native_pair_bench import now

    n = len(schedule)
    registration = deepcopy(previous)
    registration.update(created_at=now(), source_hashes=sources, inputs=str(inputs), schedule=schedule,
        arms=[[ARM, 'graspcl_200', 'free_orientation']], total_slots=n, attempt_budget_by_arm={ARM: n},
        runtime_backend_gpu=backend_gpu, runtime_scene_gpus=[scene_gpu], scene_workers=1,
        common_reset_source=str(common), backend_lock=str(out / 'backend.lock'), admitted=n, admission_failed=0,
        source_campaign=str(SOURCE), algorithm_changed=True, service_startup_timeout_seconds=1200,
        test_design_scope='Local closed-loop approach integration pilot on the same registered scene recipes.',
        frozen_algorithm='Keep the original free-orientation candidate, global pregrasp and measured approach '
        'planning. Track the checked approach prefix and hand off at 3 cm. Roll cuRobo MPC from current '
        'joint state, update the bounded goal from wrist RGB-D, and check commands against the original '
        'scene. Close at the measured endpoint, replan +3 cm lift, and keep the same physical evaluator.',
        mpc={'solver': 'cuRobo ModelPredictiveControl', 'starts_at_tcp_distance_m': mpc.HANDOFF_M,
             'control_hz': mpc.CONTROL_HZ, 'solve_interval_sim_s': mpc.SOLVE_DT, 'wrist_feedback_hz': 5,
             'translation_correction_limit_m': mpc.MAX_CORRECTION_M,
             'position_tolerance_m': mpc.POSITION_TOLERANCE_M,
             'orientation_tolerance_rad': mpc.ORIENTATION_TOLERANCE_RAD,
             'solver_model_loads_per_batch': 1, 'collision_validation_before_commands': True})
    save(out / 'registration.json', registration)
    child = deepcopy(registration)
    child.update(suite=str(out), common_reset_source=str(arm_common), attempt_budget=n,
                 experiment_arm=ARM, pipeline_profile='graspcl_200', orientation_mode='free_orientation')
    child['source_hashes'][str(arm_common)] = sha(arm_common)
    save(arm / 'registration.json', child)
    save(arm / 'progress.json', {'phase': 'prepared', 'completed': 0, 'successes': 0, 'total': n})
    import short_lift_bench

    short_lift_bench.update_suite(out, 'prepared')
    summarize(out)


def summarize(out):
    counts = Counter()
    rows = []
    pairs = read(out / 'comparison_cases.json')['cases']
    for case in pairs:
        slot = case['id']
        claim = out / ARM / 'claims' / (slot + '.json')
        if not claim.exists():
            continue
        try:
            state = read(claim)['state']
        except ValueError:
            continue
        if state != 'complete':
            continue
        trial = out / ARM / 'trials' / slot
        result = read(trial / 'result.json')
        servo = trial / 'mpc_approach.json'
        servo = read(servo) if servo.exists() else {}
        counts.update(completed=1, planning_pass=int(result['planning_success']),
                      mpc_attempted=int(bool(servo.get('mpc_solves'))),
                      mpc_settled=int(servo.get('settled', False)), success=int(result['success']))
        rows.append({**case, 'trial': str(trial), 'planning_pass': result['planning_success'],
                     'success': result['success'], 'reason': result['reason'], 'mpc': servo})
    save(out / 'mpc_results.json', {'registered': len(pairs), **dict(counts), 'cases': rows,
         'verified': (out / 'final_verification.json').exists()})


def configure(reg):
    base = relaxed.configure(reg)
    base.__file__ = str(SELF)
    base.report = summarize
    original_backends = base.Backends

    class Backends(original_backends):
        def start(self):
            popen = base.core.subprocess.Popen

            def launch(command, *args, **kwargs):
                command = [str(SELF) if str(part) == str(relaxed.SELF) else part for part in command]
                return popen(command, *args, **kwargs)

            with patch.object(base.core.subprocess, 'Popen', launch):
                super().start()
            proof = read(self.campaign / 'resident/planner_warmup/mpc_ready.json')
            assert proof['passed'] and proof['model_load_count'] == 1

    base.Backends = Backends
    verify = base.verify

    def verify_mpc(out):
        import numpy as np

        passed = 0
        for trial in (out / ARM / 'trials').iterdir():
            result = read(trial / 'result.json')
            if (trial / 'mpc_contact.json').exists():
                contact = read(trial / 'mpc_contact.json')['selected']
                proof = read(trial / 'mpc_approach.json')
                assert proof['settled'] and proof['mpc_solves'] >= 2 and proof['fresh_wrist_frames'] > 0
                assert proof['tool_position_error_m'] <= mpc.POSITION_TOLERANCE_M
                assert proof['tool_rotation_error_rad'] <= mpc.ORIENTATION_TOLERANCE_RAD
                assert proof['commands'] == proof['mpc_solves'] * mpc.STEPS_PER_SOLVE
                base.lift_policy.validate_selected(contact)
                if (trial / 'lift_plan_result.json').exists():
                    grasp = np.asarray(contact['grasp_world'])
                    assert np.allclose(np.asarray(contact['lift_world'])[:3, 3] - grasp[:3, 3], [0, 0, .03])
                passed += 1
            if result['success']:
                assert (trial / 'mpc_contact.json').exists()
        verify(out)
        proof = read(out / 'final_verification.json')
        proof.update(mpc_approach_endpoints_verified=passed, curobo_mpc_preloaded=True)
        save(out / 'final_verification.json', proof)

    base.verify = verify_mpc
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'serve', 'status'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--backend-gpu')
    parser.add_argument('--scene-gpu')
    parser.add_argument('--case', action='append')
    args, remaining = parser.parse_known_args()
    if args.stage == 'serve':
        if remaining[remaining.index('--kind') + 1] == 'curobo':
            relaxed.runtime.load_runtime().policy().install_collision_policy()
            relaxed.install_trace()
            mpc.install_resident()
        from short_lift_services import main as serve

        sys.argv = [str(SELF), *remaining]
        serve()
        return
    assert not remaining and args.output is not None
    out = args.output.resolve()
    if args.stage == 'prepare':
        assert args.backend_gpu and args.scene_gpu
        prepare(out, args.backend_gpu, args.scene_gpu, args.case or PILOT)
    elif args.stage == 'status':
        summarize(out)
    elif args.stage == 'run':
        configure(read(out / 'registration.json')).run(out)
    else:
        base = configure(read(read(args.entry)['campaign_registration']))
        run, baseline = base.core.batch.original_modules()
        base.lift_policy.install_execution_policy(run, baseline)
        with mpc.execution_hook(run):
            base.execute(out, args.entry)


if __name__ == '__main__':
    main()
