"""Compare original memory updates with CPU-only geometry reuse."""
import argparse
from pathlib import Path
import time

from arx_project.layout import ROOT, now, read_json, write_json
from modules.curobo.scene_reuse.prepared_meshes import PreparedMeshCache
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker
from .preflight import cold_meshes, run as preflight


class Worker(OriginalWorker):
    def __init__(self, planner, campaign):
        super().__init__(planner)
        self.campaign = Path(campaign)

    def request(self, directory, *, variant, **flags):
        if variant not in ['control', 'cached']:
            raise ValueError('Unknown CPU geometry variant')
        directory = Path(directory)
        pool = self.planner.pool; manager = self.planner.cpu_geometry_reuse
        pool.stage_timing.reset(); pool.stage_timing.active = True
        pool._pending_ik = []; pool.binding_codec.reset()
        cold = cold_meshes(self.planner)
        cache = None
        if variant == 'cached':
            relative = directory.relative_to(self.campaign)
            cache = PreparedMeshCache(ROOT/'tmp/curobo/mesh_cache'/self.campaign.name/relative)
            if cache.memory or list(cache.root.glob('*/metadata.json')):
                raise RuntimeError('Each formal CPU cache must start empty')
        manager.begin(cache)
        pool.memory_updates_enabled = True
        summary = None
        try:
            with pool.stage_timing.stage('memory_request_admission'):
                pool.memory_bindings.begin()
            try:
                result = super().request(directory, **flags)
            finally:
                with pool.stage_timing.stage('memory_receipt_drain_and_validation'):
                    summary = pool.memory_bindings.finish()
        finally:
            pool.memory_updates_enabled = False; pool.stage_timing.active = False
            geometry = manager.finish(); timing = pool.stage_timing.report()
            write_json(directory/'cpu_geometry.json', geometry)
            write_json(directory/'stage_timings.json', timing)
        result.update(mode=variant, stages=timing['stages'], memory_updates=summary,
                      parallel_windows=[], cold_geometry=cold, cpu_geometry=geometry)
        write_json(directory/'planning_metrics.json', result)
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--kind', default='curobo_dual_arm')
    args = parser.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.scene_reuse.runtime import create
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]/1024**2 < 10000:
        raise RuntimeError('Original startup reserve unavailable')
    probe = torch.arange(1024, device='cuda', dtype=torch.float32)
    if float((probe*2).sum()) != 1047552.:
        raise RuntimeError('CUDA arithmetic failed')
    campaign = args.root.parent; reg = read_json(campaign/'registration.json')
    began = time.perf_counter(); planner = create(args)
    checked = preflight(planner, args.root/'cpu_geometry_preflight', args.warmup_trial, reg['cases'])
    write_json(args.root/'startup.json', {'at': now(), 'passed': checked['passed'],
        'seconds': time.perf_counter()-began, 'component_pairs': checked['pairs'],
        'model_loads': planner.pool.loads, 'CPU_reuse_only': True})
    serve(args, worker=Worker(planner, campaign))


if __name__ == '__main__': main()
