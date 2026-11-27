"""Fresh simulator resets and unchanged measured-state execution for both groups."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

from arx_project.layout import ROOT, now, read_json, sha256
from arx_project.artifacts import atomic_json as write_json
from arx_project.client import Client
from .contract import VARIANTS, check_requests
from .registration import guard


def progress(root, reg, rows, phase, current=None):
    groups = {}
    for mode in VARIANTS:
        values = [r for r in rows if r['variant'] == mode]
        groups[mode] = {'registered': 74, 'processed': len(values),
            'completed': sum(not r.get('interrupted') for r in values),
            'interrupted': sum(bool(r.get('interrupted')) for r in values),
            'planning_passes': sum(r['planning_success'] for r in values),
            'physical_attempts': sum(r['physical_attempt'] for r in values),
            'successes': sum(r['success'] for r in values),
            'common_precondition_unexecuted': sum(r.get('common_precondition_unexecuted', False) for r in values)}
    write_json(root / 'progress.json', {'at': now(), 'phase': phase, 'completed_requests': len(rows),
        'total_requests': len(reg['schedule']), 'current': current, 'groups': groups, 'results': rows})


def common_failure(out, case):
    out.mkdir()
    result = {'id': case['id'], 'success': False, 'planning_success': False, 'physical_attempt': False,
        'reason': 'unable_to_reach_high_workspace', 'common_precondition_unexecuted': True,
        'inherited_from_frozen_stationing': True, 'source_result_sha256': case['baseline_result_sha256']}
    write_json(out / 'result.json', result)
    write_json(out / 'independent_evaluation.json', {'passed': True, 'physical_attempt': False,
        'independently_successful': False, 'event_applicability': 'unchanged common stationing precondition'})
    return result


def run(root, socket):
    reg = guard(root, full=False)
    from modules.pipeline.experiments.dual_arm_grasp.run import imports, gpu_inventory
    imports()
    from modules.pipeline.experiments.repaired_grasp_benchmark.simulation import install
    from modules.pipeline.experiments.indexed_grasp_planning.physical.simulation import Preparation
    from modules.pipeline.experiments.indexed_grasp_planning.physical.contract import check_trial
    from modules.pipeline.experiments.sam3_zerograsp_comparison.run import NORMAL
    executor, evaluator, _ = install(root)
    from .opening import install as install_width_execution
    install_width_execution(executor)
    client = Client(socket, timeout=1800, candidate_transport=False)
    cases = {c['id']: c for c in reg['cases']}; rows = read_json(root / 'progress.json')['results']
    continuation = reg.get('continuation') or {}
    class HandoffClient:
        def request(self, directory, **flags):
            slot = flags.get('case_id', '') + '_' + flags.get('execution_variant', '')
            if (slot == continuation.get('zero_motion_handoff_slot')
                    and Path(directory).is_relative_to(root / 'trials')
                    and not flags.get('approach') and not flags.get('lift')):
                from .continuation import CARRY_FILES, no_motion
                source = ROOT / continuation['zero_motion_handoff_directory']; no_motion(source)
                for name in CARRY_FILES:
                    target = Path(directory) / name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, target)
                shutil.copytree(source / 'arms', Path(directory) / 'arms', dirs_exist_ok=True)
                write_json(Path(directory) / 'initial_plan_carry.json', {'passed': True,
                    'source': str(source.relative_to(ROOT)), 'new_initial_solver_calls': 0,
                    'prior_motion_samples': 0, 'physical_grasps_repeated': 0})
                return {'status': 'ok', 'returncode': 0, 'result': read_json(Path(directory) / 'plan_result.json')}
            return client.request(directory, **flags)
    initial_client = HandoffClient()
    import torch
    case = cases['M0001_S1_open_apple']
    for mode in VARIANTS:
        out = root / 'preflight' / ('stationary_' + mode); out.mkdir()
        env = executor.make(case['entry']['model_id'], scene_spec=case['entry']['scene'])
        try:
            env.reset(seed=case['entry']['seed'])
            for _ in range(80): env._step_action(torch.tensor(env.initial_q[None], device=env.device))
            prep = Preparation(root, case, mode, client, executor)
            prepared = prep.prepare(env, out, case['entry'])
            if not prepared['success']: raise RuntimeError('Stationary initial planning did not pass')
            from modules.mobile_manipulation.workspace_priority.handoff import validate_prepared
            validate_prepared(out, prepared, executor.snapshot(env, env.get_obs()))
            check_requests(out)
        finally: env.close()
    write_json(root / 'preflight/all_ready.json', {'at': now(), 'passed': True,
        'both_modes_warmed': True, 'frozen_restore_and_handoff_passed': True,
        'physical_grasps': 0, 'prewarm_initial_requests': 2})
    for index, item in enumerate(reg['schedule']):
        if item['id'] in continuation.get('terminal_slot_ids', continuation.get('completed_slot_ids', [])): continue
        guard(root, full=False)
        if gpu_inventory()[reg['gpu']] < 1536: raise RuntimeError('Insufficient CUDA reserve before claiming slot')
        if shutil.disk_usage(root).free < 10 * 1024**3: raise RuntimeError('Insufficient recording disk space')
        case = cases[item['case_id']]; mode = item['variant']
        claim = root / 'claims' / (item['id'] + '.json'); out = root / 'trials' / case['id'] / mode
        out.parent.mkdir(exist_ok=True)
        if claim.exists() or out.exists(): raise RuntimeError('Claimed slots cannot be repeated')
        with claim.open('x') as stream: json.dump({'state': 'claimed', 'at': now(), **item}, stream)
        progress(root, reg, rows, 'executing', item)
        print('MEMORY_EXECUTION_BEGIN', index + 1, case['id'], mode, flush=True)
        began = time.perf_counter(); timing = {}
        if case.get('common_precondition_failure'):
            result = common_failure(out, case)
        else:
            def worker(script, directory, gpu, log, **flags):
                if script != 'planner.py' or not (flags.get('approach') or flags.get('lift')):
                    raise RuntimeError('Unexpected executor fallback')
                response = client.request(directory, execution_variant=mode, case_id=case['id'], **flags)
                write_json(log, response)
                if response['status'] != 'ok':
                    write_json(Path(directory) / 'infrastructure_error.json', {'at': now(), 'response': response})
                    raise RuntimeError('Measured-state planning RPC failed: ' + str(response))
                return response['returncode']
            executor.worker = worker
            prep = Preparation(root, case, mode, initial_client, executor)
            executor.trial(case['entry'], out, reg['gpu'], reg['gpu'], 'head', prepare_observation=prep.prepare)
            evaluator.verify_trial(out); result = read_json(out / 'result.json')
            if result['reason'] not in NORMAL | {'measured_start_state_in_collision'}:
                raise RuntimeError('Program/record fault: ' + result['reason'])
            timing = {**prep.timings, 'trial_wall_seconds': time.perf_counter() - began,
                'motion_and_replanning_seconds': time.perf_counter() - prep.returned
                    if result.get('physical_attempt') and prep.returned else 0.}
            write_json(out / 'latency.json', {'variant': mode, 'timings': timing,
                'perception_and_candidate_generation_rerun': False})
            check_trial(out, result); check_requests(out)
        row = {'id': case['id'], 'variant': mode, 'reason': result['reason'],
            'planning_success': bool(result.get('planning_success')), 'physical_attempt': bool(result.get('physical_attempt')),
            'success': bool(result['success']), 'selected_arm': result.get('selected_arm'),
            'minimum_hold_lift_m': result.get('minimum_hold_lift_m'),
            'common_precondition_unexecuted': bool(case.get('common_precondition_failure')),
            'directory': str(out.relative_to(ROOT)), 'timings': timing}
        evidence = [p for p in out.iterdir() if p.is_file() and p.suffix in ['.json', '.jsonl', '.mp4']]
        write_json(claim, {'at': now(), 'state': 'complete', **item,
            'files_sha256': {p.name: sha256(p) for p in evidence}})
        rows.append(row); progress(root, reg, rows, 'between_trials')
        print('MEMORY_EXECUTION_END', index + 1, case['id'], mode, result['reason'], flush=True)
    write_json(root / 'execution_complete.json', {'at': now(), 'completed_slots': len(rows),
        'physical_attempts': sum(r['physical_attempt'] for r in rows), 'automatic_retries': False})
    progress(root, reg, rows, 'execution_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True); args = p.parse_args()
    try: run(args.root, args.socket)
    except BaseException:
        write_json(args.root / 'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
