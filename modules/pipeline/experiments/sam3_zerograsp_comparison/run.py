"""One paired ZeroGrasp attempt per non-SAM-failed scene, frozen downstream."""
import argparse
import fcntl
from itertools import permutations
import os
from pathlib import Path
import shutil
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from arx_project.registry import module_records
from modules.pipeline.experiments.dual_arm_grasp.run import allocation_probes, gpu_inventory, imports
from .cohort import check_pair, freeze
from .metrics import baseline_roots, measured_times, request_index, summarize

HERE = Path(__file__).resolve().parent
CONFIG = ROOT/'configs/experiments/sam3_zerograsp_comparison.json'


def prepare(root):
    if (root/'registration.json').exists():
        raise RuntimeError('Comparison registration already exists')
    config = read_json(CONFIG)
    source = ROOT/config['baseline_run']
    previous = read_json(source/'registration.json')
    root.mkdir(parents=True, exist_ok=True)
    for name in ['preflight', 'resident', 'claims', 'trials', 'inputs', 'source_registered']:
        (root/name).mkdir(exist_ok=False)
    cohort = freeze(source, root/'inputs')
    for name in ['gripper_descriptor.json', 'object_catalog.json', 'common_initial_conditions.json']:
        shutil.copy2(source/name, root/name)
    sources = dict(previous['source_hashes'])
    added = [CONFIG, ROOT/'modules/zerograsp/arx_grasp_adapter.py']
    added += list(HERE.glob('*.py'))
    added += list((ROOT/'modules/zerograsp/server/vendor/zerograsp_mainline_minimal').rglob('*.py'))
    added.append(ROOT/'modules/zerograsp/server/vendor/zerograsp_mainline_minimal/configs/demo_640x480.yaml')
    added.append(ROOT/'website/tools/grasp_metrics.py')
    for path in added:
        sources[str(path)] = sha256(path)
    for filename, expected in sources.items():
        path = Path(filename)
        if sha256(path) != expected:
            raise RuntimeError('Baseline/common execution source changed: '+filename)
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    weight = ROOT/'weights/zerograsp/mainline/zerograsp_synthetic.ckpt'
    write_json(root/'registration.json', {'at': now(), 'version': 'V6', 'parent': 'V4',
        'user_authorization': config['user_request'], 'configuration': config,
        'baseline': config['baseline_run'], 'cohort': cohort, 'source_hashes': sources,
        'schedule': [pair['entry'] for pair in cohort['pairs']],
        'physical_slot_budget': sum(not p.get('common_precondition_failure', False) for p in cohort['pairs']),
        'automatic_retries': False, 'maximum_attempts_per_case': 1,
        'required_components': ['zerograsp', 'curobo_dual_arm'],
        'component_modules': {'zerograsp': 'zerograsp',
            'curobo_dual_arm': previous.get('component_modules', {}).get('curobo_dual_arm', 'curobo_dual_arm')},
        'weights': {str(weight): {'size': weight.stat().st_size, 'sha256': sha256(weight)}},
        'same_success_criteria': previous['success'],
        'source_dataset_freeze_sha256': previous['dataset_freeze_sha256']})
    write_json(root/'cohort.json', cohort)
    summarize(root, 'registered_waiting_for_resources')
    print('REGISTERED_V6', len(cohort['pairs']), 'paired scenes', flush=True)


def guard(root):
    if (root/'STOP').exists():
        raise RuntimeError('STOP exists; this comparison cannot restart')
    reg = read_json(root/'registration.json')
    for filename, expected in reg['source_hashes'].items():
        if sha256(filename) != expected:
            raise RuntimeError('Registered comparison source changed: '+filename)
    baseline = ROOT/reg['baseline']
    if sha256(baseline/'registration.json') != reg['cohort']['registration_sha256']:
        raise RuntimeError('Baseline registration changed')
    if sha256(baseline/'acceptance.json') != reg['cohort']['acceptance_sha256']:
        raise RuntimeError('Baseline acceptance changed')
    return reg


def await_resources(root, reg):
    config = reg['configuration']
    while True:
        guard(root)
        free = gpu_inventory()
        choices = [(min(free[a]-config['generator_startup_free_mib'],
                        free[b]-config['planner_startup_free_mib']), a, b)
                   for a, b in permutations(free, 2)
                   if free[a] >= config['generator_startup_free_mib']
                   and free[b] >= config['planner_startup_free_mib']]
        write_json(root/'preflight/resource_wait.json', {'at': now(), 'free_mib': free,
            'state': 'resource_available' if choices else 'waiting', 'floors_mib': {
                'zerograsp': config['generator_startup_free_mib'], 'curobo_simulation': config['planner_startup_free_mib']}})
        if choices:
            for _, generator, planner in sorted(choices, reverse=True):
                probes = allocation_probes([generator, planner])
                if all(p['passed'] for p in probes.values()):
                    roles = {'zerograsp': generator, 'curobo_dual_arm': planner, 'simulation': planner}
                    evidence = {'at': now(), 'gpu_roles': roles, 'free_mib': gpu_inventory(), 'probes': probes}
                    if (evidence['free_mib'][generator] >= config['generator_startup_free_mib'] and
                            evidence['free_mib'][planner] >= config['planner_startup_free_mib']):
                        write_json(root/'preflight/resources.json', evidence)
                        return roles
        time.sleep(1)


NORMAL = {'success', 'unable_to_reach_high_workspace', 'zerograsp_no_postprocessed_candidate',
    'no_arm_has_executable_candidate', 'no_bilateral_target_contact', 'insufficient_hold_height',
    'forbidden_robot_contact', 'measured_closure_lift_unreachable', 'measured_pregrasp_approach_unreachable',
    'gripper_open_not_converged', 'open_not_converged'} | {
    stage+'_not_converged' for stage in ['pregrasp', 'approach', 'close', 'lift', 'hold']}


def run(root):
    reg = guard(root)
    if (root/'started.json').exists() or list((root/'claims').glob('*.json')):
        raise RuntimeError('Comparison has already started; never replay a claimed scene')
    roles = await_resources(root, reg)
    for filename, weight in reg['weights'].items():
        if Path(filename).stat().st_size != weight['size'] or sha256(filename) != weight['sha256']:
            raise RuntimeError('ZeroGrasp weight changed')
    os.environ['CUDA_VISIBLE_DEVICES'] = roles['simulation']
    imports()
    from modules.pipeline.experiments.repaired_grasp_benchmark.simulation import install
    from .replay import PairedPreparation, restore
    from .services import Services
    executor, evaluator, _ = install(root)
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu_roles': roles})
    services = None
    try:
        # Preflight uses a paired measured state and cached SAM input, no grasp execution.
        pair = next(p for p in reg['cohort']['pairs'] if not p.get('common_precondition_failure'))
        frozen = check_pair(pair)
        warm = root/'preflight/head_warmup'
        for name in pair['input_hashes']:
            target = warm/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frozen/name, target)
        write_json(warm/'closed_state.json', read_json(warm/'state.json'))
        env = executor.make(pair['entry']['model_id'], scene_spec=pair['entry']['scene'])
        try:
            import torch
            env.reset(seed=pair['entry']['seed'])
            for _ in range(80):
                env._step_action(torch.tensor(env.initial_q[None], device=env.device))
            restore(env, executor, warm, read_json(warm/'state.json'))
        finally:
            env.close()
        services = Services(root, {**reg, 'gpu_roles': roles})
        services.start()
        response = services.request('curobo_dual_arm', warm,
            **({} if pair['workspace']['preferred_arm'] is None else {'arm': pair['workspace']['preferred_arm']}))
        write_json(root/'preflight/native_full_chain.json', response)
        write_json(root/'preflight/all_ready.json', {'at': now(), 'passed': True,
            'physical_grasps': 0, 'paired_restore_verified': True,
            'warm_candidate_count': read_json(warm/'zerograsp.json')['candidate_count'],
            'warm_planning_success': response['result']['success']})
        baseline_index = request_index(baseline_roots(ROOT/reg['baseline']))
        for number, pair in enumerate(reg['cohort']['pairs']):
            guard(root)
            check_pair(pair)
            if shutil.disk_usage(root).free < 10*1024**3:
                raise RuntimeError('Insufficient disk space')
            claim = root/'claims'/(pair['id']+'.json')
            out = root/'trials'/pair['id']
            if claim.exists() or out.exists():
                raise RuntimeError('Paired scene already claimed')
            write_json(claim, {'state': 'claimed', 'at': now(), 'scene': pair['id'], 'attempt': 1})
            write_json(root/'current_trial.json', {'at': now(), 'number': number+1,
                'total': len(reg['cohort']['pairs']), 'scene': pair['id']})

            def worker(script, directory, gpu, log, **flags):
                if script != 'planner.py' or not (flags.get('approach') or flags.get('lift')):
                    raise RuntimeError('Unregistered neural/observation fallback in paired comparison')
                try:
                    response = services.request('curobo_dual_arm', Path(directory), **flags)
                    write_json(log, response)
                    return response['returncode']
                except BaseException:
                    write_json(Path(directory)/'infrastructure_error.json', {'at': now(), 'error': traceback.format_exc()})
                    raise

            executor.worker = worker
            preparation = PairedPreparation(pair, services, executor)
            print('BEGIN_V6', number+1, pair['id'], flush=True)
            result = executor.trial(pair['entry'], out, roles['zerograsp'], roles['curobo_dual_arm'],
                'head', prepare_observation=preparation.prepare)
            evaluator.verify_trial(out)
            result = read_json(out/'result.json')
            if result['reason'] not in NORMAL or (out/'infrastructure_error.json').exists():
                raise RuntimeError('Execution/configuration/recording fault: '+result['reason'])
            evaluation = read_json(out/'independent_evaluation.json')
            if not evaluation['passed']:
                raise RuntimeError('Independent physical evaluation failed')
            if result.get('planning_success'):
                gate = read_json(out/'prepared_plan_execution_gate.json')
                if not gate['passed']:
                    raise RuntimeError('Paired execution handoff verification failed')
            if result.get('physical_attempt'):
                for filename in ['execution.jsonl', 'execution.mp4', 'head.mp4', 'stages.json',
                                 'contact_entities.json', 'paired_restore_verification.json']:
                    path = out/filename
                    if not path.is_file() or not path.stat().st_size:
                        raise RuntimeError('Missing paired physical evidence: '+filename)
            times = {'baseline': measured_times(pair, ROOT/pair['baseline_trial'], baseline_index, baseline=True),
                'zerograsp': measured_times(pair, out, request_index([root]))}
            write_json(out/'paired_latency.json', times)
            write_json(claim, {'state': 'complete', 'at': now(), 'scene': pair['id'],
                'result_sha256': sha256(out/'result.json'), 'paired_latency_sha256': sha256(out/'paired_latency.json')})
            summarize(root, 'running')
            print('RESULT_V6', number+1, pair['id'], result['reason'], flush=True)
        guard(root)
        summarize(root, 'complete')
    finally:
        if services is not None:
            services.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root)
        return
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc()})
            summarize(root, 'stopped')
            raise


if __name__ == '__main__':
    main()
