"""GPU-only geometry sharing with the original memory-update baseline."""
import argparse
from pathlib import Path
import time

from arx_project.layout import ROOT, now, read_json, write_json
from modules.curobo.scene_cache import mesh_stores
from modules.curobo.scene_reuse.gpu_runtime import GPUGeometryReuse
from .worker import Worker as OriginalWorker
from .preflight import cold_meshes, run as component_preflight


class ComponentModes:
    """Reuse the existing exact native component comparisons for GPU sharing."""
    def __init__(self, cpu, gpu): self.cpu, self.gpu = cpu, gpu
    def begin(self, cache):
        self.cpu.begin(None); self.gpu.begin(cache is not None)
    def finish(self):
        result = self.cpu.finish(); result['GPU_geometry'] = self.gpu.finish()
        return result


def ownership_preflight(planner, warmup, output):
    import numpy as np
    import torch
    from curobo.types import Pose
    from modules.graspgenx.native_postprocessing.width_planning import target_state
    pool = planner.pool; cpu = planner.cpu_geometry_reuse; gpu = planner.gpu_geometry_reuse
    state = planner.model.bind(read_json(warmup/'state.json'), 'right', output/'physical_arm')
    state = target_state(state, .088)
    cold_meshes(planner); cpu.begin(None); gpu.begin(True)
    pool.memory_updates_enabled = True; pool.memory_bindings.begin()
    done = False
    try:
        context = pool.bind(output/'state', state, include_target=True)
        pool.contact_view(context, state)
        first = context.planner.scene_collision_checker
        second = context.checker.scene_model
        a, b = first.data.meshes, second.data.meshes
        if a is b: raise RuntimeError('Consumers unexpectedly share mutable scene data')
        common = set(a.wp_cache) & set(b.wp_cache)
        if not common: raise RuntimeError('Ownership fixture did not load shared meshes')
        if any(a.wp_cache[n].mesh_id != b.wp_cache[n].mesh_id for n in common):
            raise RuntimeError('GPU sharing did not reuse native handles')
        q = context.measured(state)
        spheres = context.checker.get_kinematics(q[None, None]).robot_spheres.detach().clone()
        def query():
            cost, gradient = context.checker.get_collision_vector(spheres.clone())
            return cost.cpu().numpy().copy(), gradient.cpu().numpy().copy()
        before = query(); poses = b.inv_pose.clone(); flags = b.enable.clone(); handles = b.mesh_ids.clone()
        name = sorted(common)[0]
        first.update_obstacle_pose(name, Pose.from_list([10., 10., 10., 1., 0., 0., 0.]))
        first.enable_obstacle(name, False)
        if not all(torch.equal(x, y) for x, y in [(poses,b.inv_pose),(flags,b.enable),(handles,b.mesh_ids)]):
            raise RuntimeError('One consumer changed another consumer scene buffers')
        torch.cuda.synchronize(); a.clear(clear_warp_cache=True)
        gpu.finish(); done = True
        after = query()
        if not all(np.array_equal(x, y) for x, y in zip(before, after)):
            raise RuntimeError('Releasing another consumer changed live collision results')
        result = {'at': now(), 'passed': True, 'shared_native_meshes': len(common),
            'other_consumer_pose_and_flags_independent': True,
            'second_consumer_valid_after_first_and_pool_release': True}
        write_json(output/'acceptance.json', result)
        return result
    finally:
        pool.memory_bindings.finish(); pool.memory_updates_enabled = False; cpu.finish()
        if not done: gpu.finish()


class Worker(OriginalWorker):
    def request(self, directory, *, variant, **flags):
        if variant not in ['control', 'cached']: raise ValueError('Unknown GPU geometry variant')
        gpu = self.planner.gpu_geometry_reuse
        gpu.begin(variant == 'cached')
        try:
            # Stage one was not adopted: both groups use the original CPU loader.
            result = super().request(directory, variant='control', **flags)
        finally:
            sharing = gpu.finish(); write_json(Path(directory)/'GPU_geometry.json', sharing)
        result.update(mode=variant, GPU_geometry=sharing)
        write_json(Path(directory)/'planning_metrics.json', result)
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--kind', default='curobo_dual_arm'); args = parser.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.scene_reuse.runtime import create
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]/1024**2 < 10000: raise RuntimeError('Original startup reserve unavailable')
    probe = torch.arange(1024, device='cuda', dtype=torch.float32)
    if float((probe*2).sum()) != 1047552.: raise RuntimeError('CUDA arithmetic failed')
    campaign = args.root.parent; reg = read_json(campaign/'registration.json')
    began = time.perf_counter(); planner = create(args)
    cpu = planner.cpu_geometry_reuse
    gpu = GPUGeometryReuse(mesh_stores(planner)); planner.gpu_geometry_reuse = gpu
    planner.cpu_geometry_reuse = ComponentModes(cpu, gpu)
    try:
        checked = component_preflight(planner, args.root/'GPU_geometry_preflight', args.warmup_trial, reg['cases'])
    finally: planner.cpu_geometry_reuse = cpu
    owned = ownership_preflight(planner, args.warmup_trial, args.root/'GPU_ownership_preflight')
    write_json(args.root/'startup.json', {'at': now(), 'passed': checked['passed'] and owned['passed'],
        'seconds': time.perf_counter()-began, 'component_pairs': checked['pairs'],
        'model_loads': planner.pool.loads, 'immutable_GPU_geometry_only': True})
    serve(args, worker=Worker(planner, campaign))


if __name__ == '__main__': main()
