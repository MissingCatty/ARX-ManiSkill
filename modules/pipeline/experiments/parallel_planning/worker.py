"""One resident numerical planner with optional native trajectory batches."""
import argparse
from pathlib import Path
import time

from arx_project.layout import now, read_json, write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker


def preflight(planner, warmup, root):
    import torch
    from modules.graspgenx.native_postprocessing.width_planning import target_state
    rows = []
    for side in ['right', 'left']:
        state = planner.model.bind(read_json(warmup / 'state.json'), side, root / side / 'physical_arm')
        for width in [.088, .060, .040]:
            measured = target_state(state, width)
            context, validator = planner.pool.width_context(root / side / str(width), measured)
            start = context.measured(measured)
            for capacity in [4, 8]:
                session = planner.pool.prepare_parallel(context, capacity)
                jobs = [{'index': i, 'width': width, 'start': start.clone(), 'q_pre': start.clone()}
                        for i in range(2)]
                jobs[1]['q_pre'][0] = torch.clamp(start[0] + .003,
                    context.limits[0, 0] + 1e-4, context.limits[0, 1] - 1e-4)
                paths, report = session.plan(context, jobs)
                if len(paths) != 2:
                    raise RuntimeError('Parallel prewarm lost a legal stationary path: ' + str(report))
                rows.append({'arm': side, 'width_m': width, **report})
                write_json(root / 'progress.json', {'at': now(), 'completed_batch_queries': len(rows),
                    'maximum_batch_queries': 12, 'last': rows[-1]})
    torch.cuda.synchronize()
    write_json(root / 'acceptance.json', {'at': now(), 'passed': True, 'batch_queries': len(rows),
        'reports': rows, 'complete_planning_requests': 0, 'physical_grasps': 0})


class Worker(OriginalWorker):
    def request(self, directory, *, variant, **flags):
        capacities = {'control': 0, 'parallel4': 4, 'parallel8': 8}
        if variant not in capacities:
            raise ValueError('Unknown parallel planning variant')
        pool = self.planner.pool
        pool.parallel_capacity = capacities[variant]
        pool.parallel_statistics = []
        pool.stage_timing.reset(); pool._pending_ik = []; pool.stage_timing.active = True
        pool.binding_codec.reset()
        try:
            result = super().request(directory, **flags)
        finally:
            pool.stage_timing.active = False
            timings = pool.stage_timing.report()
            write_json(Path(directory) / 'stage_timings.json', timings)
        result.update(mode=variant, stages=timings['stages'], parallel_windows=pool.parallel_statistics,
            original_V3_scalar_interleaved_planner=variant == 'control')
        write_json(Path(directory) / 'planning_metrics.json', result)
        return result


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--kind', default='curobo_dual_arm')
    p.add_argument('--development-only', action='store_true')
    args = p.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.parallel_planning.runtime import create
    torch.set_num_threads(4)
    free = torch.cuda.mem_get_info()[0] / 1024**2
    if free < 10000:
        raise RuntimeError('Original startup GPU reserve unavailable')
    probe = torch.arange(1024, device='cuda', dtype=torch.float32)
    if float((probe * 2).sum()) != 1047552.:
        raise RuntimeError('CUDA arithmetic probe failed')
    began = time.perf_counter()
    planner = create(args)
    preflight(planner, args.warmup_trial, args.root / 'parallel_preflight')
    write_json(args.root / 'startup.json', {'at': now(), 'passed': True,
        'seconds': time.perf_counter() - began, 'GPU_free_mib_before': free,
        'capacities_warmed': sorted(planner.pool.parallel_sessions), 'physical_grasps': 0})
    if not args.development_only:
        serve(args, worker=Worker(planner))
    for session in planner.pool.parallel_sessions.values():
        session.destroy()


if __name__ == '__main__':
    main()
