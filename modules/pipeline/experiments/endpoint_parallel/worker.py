"""One warmed planner for original, endpoint-only and endpoint/path batches."""
import argparse
import os
from pathlib import Path
import time

import numpy as np

from arx_project.layout import now, read_json, write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker


def preflight(planner, warmup, root):
    import torch
    from modules.graspgenx.native_postprocessing.width_planning import target_state
    reports = []
    for side in ['right', 'left']:
        state = planner.model.bind(read_json(warmup / 'state.json'), side, root / side / 'physical_arm')
        for width in [.088, .060, .040]:
            measured = target_state(state, width)
            context, validator = planner.pool.width_context(root / side / str(width), measured)
            engine = planner.pool.prepare_endpoint_engine(context); engine.bind(context)
            start = context.measured(measured); second = start.clone()
            second[0] = torch.clamp(second[0] + .003, context.limits[0, 0] + 1e-4, context.limits[0, 1] - 1e-4)
            source_q = torch.stack([start, second])
            worlds = validator.geometry.frames(source_q.cpu().numpy())['right_eef_link']
            bad = np.eye(4); bad[:3, 3] = [10., 10., 10.]
            mixed = [worlds[0], worlds[1], bad]
            native = engine.native(mixed, start)
            pure = engine.pure(mixed, start)
            if (any(q is None for q in native[:2]) or native[2] is not None
                    or not all(len(q) for q in pure[:2]) or len(pure[2])):
                raise RuntimeError('Batched IK lost mixed positive/negative row semantics')
            seeded = torch.cat([source_q, source_q[1:2]])
            starts = torch.cat([seeded, seeded[:1].expand(13, -1)])
            for stage in ['pregrasp', 'lift']:
                q, valid = engine._solve(engine.full_solver, [worlds[1], worlds[0], bad], starts, 4, stage + '_preflight')
                if not bool(valid[:2].any(-1).all()) or bool(valid[2].any()):
                    raise RuntimeError('Per-target start rows failed a legal FK endpoint')
                if 2 not in engine.calls[-1]['refined_source_rows']:
                    raise RuntimeError('Negative target did not exercise seeded-row refinement')
                chosen = torch.stack([row[mask][0] for row, mask in zip(q[:2], valid[:2])])
                actual = validator.geometry.frames(chosen.cpu().numpy())['right_eef_link']
                if np.max(np.linalg.norm(actual[:, :3, 3] - worlds[::-1, :3, 3], axis=1)) > .002:
                    raise RuntimeError('Seeded endpoint output changed its target-row order')
            jobs = [{'index': i, 'width': width, 'start': start.clone(), 'q_pre': source_q[i].clone()} for i in range(2)]
            paths, receipt = planner.pool.prepare_parallel(context, 4).plan(context, jobs)
            if len(paths) != 2: raise RuntimeError('Parallel path component prewarm failed')
            reports.append({'arm': side, 'width_m': width, 'passed': True,
                'mixed_IK_rows_preserved': True, 'per_target_start_rows_verified': True,
                'IK_solver_calls': list(engine.calls), 'path': receipt})
            write_json(root / 'progress.json', {'at': now(), 'groups': len(reports), 'maximum_groups': 6})
    write_json(root / 'acceptance.json', {'at': now(), 'passed': True, 'groups': len(reports), 'reports': reports,
        'IK_batch_queries': 24, 'path_batch_queries': 6, 'complete_planning_requests': 0, 'physical_grasps': 0})


class Worker(OriginalWorker):
    def request(self, directory, *, variant, **flags):
        if getattr(self, 'strict_idle', False):
            from .resources import compute_processes, foreign_processes
            foreign = foreign_processes(os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(), compute_processes())
            if foreign: raise RuntimeError('Timing device has foreign compute processes: ' + str(foreign))
        if variant not in ['control', 'batched_endpoints', 'batched_paths4']:
            raise ValueError('Unknown endpoint/path comparison variant')
        pool = self.planner.pool
        pool.parallel_capacity = 0
        pool.windowed_enabled = variant != 'control'
        pool.windowed_path_capacity = 4 if variant == 'batched_paths4' else 0
        pool.windowed_statistics = []
        pool.endpoint_engine.reset_seed()
        pool.stage_timing.reset(); pool._pending_ik = []; pool.stage_timing.active = True
        pool.binding_codec.reset()
        try:
            result = super().request(directory, **flags)
        finally:
            pool.stage_timing.active = False
            timings = pool.stage_timing.report()
            write_json(Path(directory) / 'stage_timings.json', timings)
        result.update(mode=variant, stages=timings['stages'], parallel_windows=pool.windowed_statistics,
            original_V3_scalar_interleaved_planner=variant == 'control')
        if getattr(self, 'strict_idle', False):
            foreign = foreign_processes(os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(), compute_processes())
            result.update(foreign_GPU_processes_after_request=foreign, timing_eligible=not foreign,
                stop_after_request=bool(foreign))
        write_json(Path(directory) / 'planning_metrics.json', result)
        return result


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--kind', default='curobo_dual_arm')
    p.add_argument('--development-only', action='store_true')
    p.add_argument('--strict-idle', action='store_true')
    args = p.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.parallel_planning.windowed import create
    torch.set_num_threads(4)
    free = torch.cuda.mem_get_info()[0] / 1024**2
    if free < 10000: raise RuntimeError('Original startup reserve unavailable')
    probe = torch.arange(1024, device='cuda', dtype=torch.float32)
    if float((probe * 2).sum()) != 1047552.: raise RuntimeError('CUDA arithmetic probe failed')
    began = time.perf_counter(); planner = create(args)
    preflight(planner, args.warmup_trial, args.root / 'endpoint_preflight')
    write_json(args.root / 'startup.json', {'at': now(), 'passed': True, 'seconds': time.perf_counter() - began,
        'GPU_free_mib_before': free, 'endpoint_capacity': 16, 'path_capacity': 4, 'physical_grasps': 0})
    if not args.development_only:
        worker = Worker(planner); worker.strict_idle = args.strict_idle
        serve(args, worker=worker)
    planner.pool.endpoint_engine.full_solver.destroy(); planner.pool.endpoint_engine.pure_solver.destroy()
    for session in planner.pool.parallel_sessions.values(): session.destroy()


if __name__ == '__main__': main()
