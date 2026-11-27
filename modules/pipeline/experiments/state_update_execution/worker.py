"""Apply the accepted state updates to initial, approach and lift requests."""
import argparse
import json
from pathlib import Path
import time
import traceback

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from .contract import VARIANTS, stage_name


class Worker:
    def __init__(self, planner, campaign):
        self.planner = planner; self.root = Path(campaign)
        self.reg = read_json(self.root / 'registration.json')
        self.cases = {case['id']: case for case in self.reg['cases']}

    def request(self, directory, *, execution_variant, case_id, **flags):
        import torch
        from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport
        from modules.curobo.scene_cache import mesh_stores, prune_inactive_meshes
        directory = Path(directory); stage = stage_name(flags)
        if execution_variant not in VARIANTS or case_id not in self.cases:
            raise ValueError('Request is outside the registered comparison')
        if not directory.resolve().is_relative_to(self.root.resolve()): raise ValueError('Wrong output root')
        ledger = directory / 'request_ledger.jsonl'
        events = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        if any(e['stage'] == stage for e in events): raise ValueError('No repeated request or automatic retry')
        free = torch.cuda.mem_get_info()[0] / 1024**2
        if free < 1536: raise RuntimeError('Insufficient original request CUDA reserve')
        def event(kind):
            with ledger.open('a') as stream:
                stream.write(json.dumps({'at': now(), 'stage': stage, 'event': kind,
                    'variant': execution_variant, 'case_id': case_id}) + '\n')
        event('started'); began = time.perf_counter(); pool = self.planner.pool
        enabled = execution_variant == 'optimized'; summary = None
        pool.memory_updates_enabled = enabled; pool.stage_timing.reset(); pool._pending_ik = []
        pool.stage_timing.active = True; pool.binding_codec.reset()
        try:
            if stage == 'initial':
                source = ROOT / self.cases[case_id]['input_directory'] / 'candidates/candidate_pool.json'
                if sha256(source) != self.reg['input_sha256'][str(source)]: raise ValueError('Frozen candidate pool changed')
                payload = read_json(source)
                write_json(directory / 'candidate_order.json', {'source_count': len(payload['candidates']),
                    'identity_order': True, 'shared_fine_NMS_m': .015, 'shared_fine_NMS_deg': 15,
                    'source': str(source.relative_to(ROOT)), 'new_postprocessing': False})
                flags['candidate_transport'] = publish_candidate_pool(directory, payload)
            torch.manual_seed(20261126)
            for context in pool.contexts.values(): context.planner.reset_seed()
            pool.pure_solver.reset_seed(); torch.cuda.synchronize()
            if enabled: pool.memory_bindings.begin()
            try:
                try: result = self.planner.request(directory, **flags)
                except AssertionError as error:
                    if str(error) != 'Measured state collides in unchanged ARX model': raise
                    result = {'success': False, 'reason': 'measured_start_state_in_collision',
                        'rejection_type': 'planning_constraint', 'original_collision_gate_preserved': True}
                    write_json(directory / ('plan_result.json' if stage == 'initial' else stage + '_plan_result.json'), result)
            finally:
                if enabled: summary = pool.memory_bindings.finish()
                if stage == 'initial': finish_candidate_transport(directory)
            torch.cuda.synchronize()
            timings = pool.stage_timing.report()
            receipt = {'at': now(), 'variant': execution_variant, 'stage': stage,
                'memory_state_updates_enabled': enabled, 'original_scalar_IK_and_candidate_order': True,
                'measured_execution_state_used': stage != 'initial', 'collision_spheres_per_arm': 871,
                'tcp_m': .043, 'V2_fingertip_table_policy': True, 'receipt_drain_completed': True,
                'memory_updates': summary, 'geometry_and_thresholds_unchanged': True}
            write_json(directory / ('execution_planner_' + stage + '.json'), receipt)
            write_json(directory / (stage + '_metrics.json'), {'at': now(), 'variant': execution_variant,
                'stage': stage, 'success': bool(result['success']), 'seconds': time.perf_counter() - began,
                'request_free_mib': free, 'stages': timings, 'memory_updates': summary})
            prune_inactive_meshes(mesh_stores(self.planner), torch.cuda.synchronize)
            event('complete')
            return result
        except BaseException:
            write_json(directory / 'infrastructure_error.json', {'at': now(), 'stage': stage,
                'error': traceback.format_exc()})
            raise
        finally:
            pool.memory_updates_enabled = False; pool.stage_timing.active = False


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor', 'campaign']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--kind', default='curobo_memory_state'); args = p.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.state_updates.runtime import create
    from modules.pipeline.experiments.state_update_planning.preflight import run
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0] / 1024**2 < 10000: raise RuntimeError('Original startup reserve unavailable')
    probe = torch.arange(1024, device='cuda', dtype=torch.float32)
    if float((probe * 2).sum()) != 1047552.: raise RuntimeError('CUDA arithmetic failed')
    began = time.perf_counter(); planner = create(args)
    proof = run(planner, args.warmup_trial, args.root / 'state_preflight')
    write_json(args.root / 'startup.json', {'at': now(), 'passed': proof['passed'],
        'seconds': time.perf_counter() - began, 'shared_original_solver': True})
    serve(args, worker=Worker(planner, args.campaign))


if __name__ == '__main__': main()
