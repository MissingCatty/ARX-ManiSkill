"""Restore each frozen scene, plan once, then use the unchanged ARX executor."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from arx_project.layout import ROOT, now, sha256
from arx_project.artifacts import atomic_json as write_json
from arx_project.client import Client
from .contract import check_trial
from .registration import guard, raw


class Preparation:
    def __init__(self, root, case, variant, client, executor):
        self.root, self.case, self.variant = root, case, variant
        self.client, self.executor = client, executor
        self.returned = None
        self.timings = {}

    def prepare(self, env, out, entry):
        from PIL import Image
        from modules.pipeline.experiments.sam3_zerograsp_comparison.replay import restore
        begun = time.perf_counter()
        try:
            frozen = ROOT/self.case['input_directory']
            for name in self.case['input_hashes']:
                target = out/name;target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frozen/name, target)
            state = raw(out/'state.json')
            restore(env, self.executor, out, state)
            self.timings['restore_seconds'] = time.perf_counter()-begun
            Image.fromarray(env.render().cpu().numpy()[0]).save(out/'initial_overview.png')
            write_json(out/'initial_conditions.json', {'spawn': env.spawn,
                'base_pose': entry['base_pose'], 'initial_q': env.initial_q.tolist(),
                'object_bbox': env.object_bbox, 'paired_station_world': state['links']['base_link']})
            write_json(out/'capture.json', {'camera': 'head', 'mask_source': 'frozen successful SAM3',
                'paired_source': self.case['baseline_trial'], 'new_perception_calls': 0,
                'input_hashes': self.case['input_hashes']})
            flags = {} if self.case['workspace']['preferred_arm'] is None else {
                'arm': self.case['workspace']['preferred_arm']}
            begin = time.perf_counter()
            response = self.client.request(out, execution_variant=self.variant, case_id=self.case['id'], **flags)
            self.timings['initial_planning_with_postprocessing_seconds'] = time.perf_counter()-begin
            if response['status'] != 'ok': raise RuntimeError('Ordering/planning RPC failed: '+str(response))
            plan = raw(out/'plan_result.json')
            self.returned = time.perf_counter()
            if not plan['success']:
                return {'success': False, 'reason': plan['reason'], 'motion_started': False}
            names = ['state.json', 'plan_result.json', 'camera/color.png', 'camera/depth_m.npy',
                     'camera/mask.png', 'candidates/candidate_pool.json']
            return {'success': True, 'prepared_plan': True, 'motion_started': False,
                **self.case['workspace'], 'prepared_artifacts': {n: sha256(out/n) for n in names},
                'paired_source': self.case['baseline_trial']}
        except BaseException:
            write_json(out/'infrastructure_error.json', {'at': now(), 'error': traceback.format_exc()})
            raise


def progress(root, reg, results, phase, current=None):
    groups = {}
    for mode in ['baseline', 'optimized']:
        rows = [r for r in results if r['variant'] == mode]
        groups[mode] = {'completed': len(rows), 'registered': 10,
            'planning_passes': sum(r['planning_success'] for r in rows),
            'physical_attempts': sum(r['physical_attempt'] for r in rows),
            'successes': sum(r['success'] for r in rows)}
    write_json(root/'progress.json', {'at': now(), 'phase': phase,
        'completed_requests': len(results), 'total_requests': len(reg['schedule']),
        'current': current, 'groups': groups, 'results': results})


def run(root, socket):
    reg = guard(root, full=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu']
    from modules.pipeline.experiments.dual_arm_grasp.run import imports, gpu_inventory
    imports()
    from modules.pipeline.experiments.repaired_grasp_benchmark.simulation import install
    from modules.pipeline.experiments.sam3_zerograsp_comparison.run import NORMAL
    executor, evaluator, _ = install(root)
    client = Client(socket, timeout=1800, candidate_transport=False)
    cases = {c['id']: c for c in reg['cases']};results = []
    warm_case = cases['M0001_S1_open_apple']
    # Exercise the exact frozen restore and both planning modes without moving an arm.
    import torch
    for mode in ['baseline', 'optimized']:
        out = root/'preflight'/('stationary_'+mode);out.mkdir()
        env = executor.make(warm_case['entry']['model_id'], scene_spec=warm_case['entry']['scene'])
        try:
            env.reset(seed=warm_case['entry']['seed'])
            for _ in range(80): env._step_action(torch.tensor(env.initial_q[None], device=env.device))
            prep = Preparation(root, warm_case, mode, client, executor)
            prepared = prep.prepare(env, out, warm_case['entry'])
            if not prepared['success']: raise RuntimeError('Stationary execution preflight planning failed')
            from modules.mobile_manipulation.workspace_priority.handoff import validate_prepared
            validate_prepared(out, prepared, executor.snapshot(env, env.get_obs()))
        finally: env.close()
    write_json(root/'preflight/all_ready.json', {'at': now(), 'passed': True,
        'both_planning_modes_warmed': True, 'frozen_restore_and_handoff_passed': True,
        'physical_grasps': 0, 'prewarm_planning_requests': 2})
    for ordinal, item in enumerate(reg['schedule']):
        guard(root, full=False)
        if gpu_inventory()[reg['gpu']] < 1536: raise RuntimeError('Insufficient headroom before claiming physical slot')
        if shutil.disk_usage(root).free < 10*1024**3: raise RuntimeError('Insufficient recording disk space')
        case = cases[item['case_id']];mode = item['variant']
        claim = root/'claims'/(item['id']+'.json');out = root/'trials'/item['case_id']/mode
        out.parent.mkdir(exist_ok=True)
        if claim.exists() or out.exists(): raise RuntimeError('Physical slot already claimed; no retry')
        with claim.open('x') as stream: json.dump({'state': 'claimed', 'at': now(), **item}, stream)
        progress(root, reg, results, 'executing', item)

        def worker(script, directory, gpu, log, **flags):
            if script != 'planner.py' or not (flags.get('approach') or flags.get('lift')):
                raise RuntimeError('Unexpected physical fallback')
            try:
                response = client.request(directory, execution_variant=mode, case_id=case['id'], **flags)
                write_json(log, response)
                if response['status'] != 'ok': raise RuntimeError(str(response))
                return response['returncode']
            except BaseException:
                write_json(Path(directory)/'infrastructure_error.json', {'at': now(), 'error': traceback.format_exc()})
                raise

        executor.worker = worker
        prep = Preparation(root, case, mode, client, executor);began = time.perf_counter()
        print('V8_PHYSICAL_BEGIN', ordinal+1, item['case_id'], mode, flush=True)
        executor.trial(case['entry'], out, reg['gpu'], reg['gpu'], 'head', prepare_observation=prep.prepare)
        end = time.perf_counter();evaluator.verify_trial(out);result = raw(out/'result.json')
        if result['reason'] not in NORMAL: raise RuntimeError('Execution/program fault: '+result['reason'])
        timing = {**prep.timings, 'trial_wall_seconds': end-began,
            'motion_and_replanning_seconds': end-prep.returned if result.get('physical_attempt') and prep.returned else 0.}
        write_json(out/'latency.json', {'variant': mode, 'timings': timing,
            'perception_and_candidate_generation_rerun': False})
        check_trial(out, result)
        row = {'id': item['case_id'], 'variant': mode, 'reason': result['reason'],
            'planning_success': bool(result.get('planning_success')), 'success': bool(result['success']),
            'physical_attempt': bool(result.get('physical_attempt')), 'selected_arm': result.get('selected_arm'),
            'minimum_hold_lift_m': result.get('minimum_hold_lift_m'),
            'directory': str(out.relative_to(ROOT)), 'timings': timing}
        evidence = [p for p in out.iterdir() if p.is_file() and p.suffix in ['.json', '.jsonl', '.mp4']]
        write_json(claim, {'at': now(), 'state': 'complete', **item,
            'files_sha256': {p.name: sha256(p) for p in evidence}})
        results.append(row);progress(root, reg, results, 'between_trials')
        print('V8_PHYSICAL_END', ordinal+1, item['case_id'], mode, result['reason'], flush=True)
    write_json(root/'execution_complete.json', {'at': now(), 'completed_slots': len(results),
        'physical_attempts': sum(r['physical_attempt'] for r in results), 'automatic_retries': False})
    progress(root, reg, results, 'execution_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True);args = p.parse_args()
    try: run(args.root, args.socket)
    except BaseException:
        write_json(args.root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
