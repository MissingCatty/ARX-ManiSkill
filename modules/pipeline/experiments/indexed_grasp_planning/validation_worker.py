"""Resident branch-validation comparison with original batch16 IK and gates."""
import argparse
from pathlib import Path
import time

from arx_project.layout import now, read_json, write_json
from modules.pipeline.experiments.graspgenx_postprocessing.planning_worker import MeasuredPlanner


class ValidationPlanner(MeasuredPlanner):
    def request(self, directory, *, validation_variant, **options):
        if validation_variant not in ['control', 'vectorized']:
            raise ValueError('Unknown branch validation variant')
        pool = self.planner.pool
        if pool.stage_timing.active:
            raise RuntimeError('Overlapping validation comparison requests')
        pool.vectorized_fallback = validation_variant == 'vectorized'
        pool._following_ik_stages = []
        pool.stage_timing.reset();pool.stage_timing.active = True
        try:
            result = super().request(directory, **options)
        finally:
            pool.stage_timing.active = False
            report = pool.stage_timing.report()
            report['variant'] = validation_variant
            write_json(Path(directory)/'stage_timings.json', report)
        result['validation_variant'] = validation_variant
        # Engine calls reset on every bind. Read all windows, not the last one.
        calls = []
        for side in ['right', 'left']:
            for path in sorted((Path(directory)/'arms'/side/'search_windows').glob('*.json')):
                calls.extend(read_json(path)['result'].get('batch_ik', {}).get('solver_calls', []))
        result['all_windows_grasp_IK_GPU_seconds'] = sum(call['gpu_seconds'] for call in calls)
        result['all_windows_grasp_IK_calls'] = len(calls)
        result['stage_summary'] = report['stages']
        write_json(Path(directory)/'validation_metrics.json', result)
        # Send only the small summary; the full trace stays in the receipt.
        return {key: value for key, value in result.items() if key != 'stage_summary'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--kind', default='curobo_indexed_planning')
    args = parser.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.indexed_planning.validation_service import create
    from modules.curobo.indexed_planning.validation_warmup import verify_row_validation
    from modules.curobo.indexed_planning.warmup import warm_solver_branches
    torch.set_num_threads(4)
    free, _ = torch.cuda.mem_get_info()
    if free/1024**2 < 10000:
        raise RuntimeError('Insufficient original startup CUDA memory')
    torch.zeros(1, device='cuda');torch.cuda.synchronize();began = time.perf_counter()
    planner = create(args)
    branches = warm_solver_branches(planner, args.warmup_trial/'state.json', args.root/'solver_prewarm')
    rows = verify_row_validation(planner, args.warmup_trial/'state.json', args.root/'validation_prewarm')
    write_json(args.root/'startup.json', {'at': now(), 'passed': True,
        'seconds': time.perf_counter()-began, 'shared_model_pool': True,
        'model_loads': planner.pool.loads, 'batch_capacity': 16,
        'negative_and_partial_batch_rows_verified': branches['passed'],
        'actual_world_row_validation_verified': rows['passed'],
        'actual_world_validation_rows': sum(r['rows'] for r in rows['reports']),
        'simulation_instances': 0, 'physical_grasps': 0})
    serve(args, worker=ValidationPlanner(planner))


if __name__ == '__main__': main()
