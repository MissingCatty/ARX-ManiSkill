"""Use one warmed original planner for both state-update variants."""
import argparse
from pathlib import Path
import time

from arx_project.layout import now, write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker


class Worker(OriginalWorker):
    def request(self, directory, *, variant, **flags):
        if variant not in ['control', 'memory']:
            raise ValueError('Unknown state-update variant')
        pool = self.planner.pool
        pool.memory_updates_enabled = variant == 'memory'
        pool.stage_timing.reset()
        pool._pending_ik = []
        pool.stage_timing.active = True
        pool.binding_codec.reset()
        summary = None
        try:
            if pool.memory_updates_enabled:
                with pool.stage_timing.stage('memory_request_admission'):
                    pool.memory_bindings.begin()
            try:
                result = super().request(directory, **flags)
            finally:
                if pool.memory_updates_enabled:
                    with pool.stage_timing.stage('memory_receipt_drain_and_validation'):
                        summary = pool.memory_bindings.finish()
        finally:
            pool.memory_updates_enabled = False
            pool.stage_timing.active = False
            timings = pool.stage_timing.report()
            write_json(Path(directory) / 'stage_timings.json', timings)
        result.update(mode=variant, stages=timings['stages'], memory_updates=summary,
                      parallel_windows=[], original_V3_scalar_interleaved_planner=True)
        write_json(Path(directory) / 'memory_state_updates.json', {'enabled': variant == 'memory', 'details': summary})
        write_json(Path(directory) / 'planning_metrics.json', result)
        return result


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--kind', default='curobo_dual_arm')
    args = p.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.state_updates.runtime import create
    from .preflight import run
    torch.set_num_threads(4)
    free = torch.cuda.mem_get_info()[0] / 1024**2
    if free < 10000:
        raise RuntimeError('Original startup reserve unavailable')
    probe = torch.arange(1024, device='cuda', dtype=torch.float32)
    if float((probe * 2).sum()) != 1047552.:
        raise RuntimeError('CUDA arithmetic probe failed')
    began = time.perf_counter()
    planner = create(args)
    checked = run(planner, args.warmup_trial, args.root / 'state_preflight')
    write_json(args.root / 'startup.json', {'at': now(), 'passed': True,
        'seconds': time.perf_counter() - began, 'GPU_free_mib_before': free,
        'width_pairs': len(checked['width_pairs']), 'sweep_pairs': len(checked['sweep_pairs']),
        'shared_original_solver': True, 'batch_candidate_IK_enabled': False, 'physical_grasps': 0})
    serve(args, worker=Worker(planner))


if __name__ == '__main__':
    main()
