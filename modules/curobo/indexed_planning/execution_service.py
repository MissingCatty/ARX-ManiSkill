"""Use the validated V8 combination through the ordinary grasp execution API."""
from pathlib import Path
import time

from arx_project.layout import write_json
from .layout_service import create


def configure(pool, variant):
    if variant not in ['baseline', 'optimized']:
        raise ValueError('Unknown physical comparison variant')
    if pool.endpoint_sequence is not None or pool.stage_timing.active:
        raise RuntimeError('Cannot change configuration during planning')
    enabled = variant == 'optimized'
    pool.ik_batch_size = 16 if enabled else 0
    pool.fast_binding_enabled = enabled
    pool.vectorized_fallback = False
    pool.mesh_vote_mode(enabled)


class ExecutionPlanner:
    def __init__(self, planner):
        self.planner = planner

    def request(self, directory, *, execution_variant, **flags):
        import torch
        from arx_project.json_io import read_json
        from modules.graspgenx.postprocessing.ordering import selection_source
        from modules.curobo.scene_cache import mesh_stores, prune_inactive_meshes
        directory = Path(directory);pool = self.planner.pool
        configure(pool, execution_variant)
        free, _ = torch.cuda.mem_get_info()
        if free/1024**2 < 1536: raise RuntimeError('Insufficient original request CUDA reserve')
        stage = 'lift' if flags.get('lift') else 'approach' if flags.get('approach') else 'initial'
        torch.manual_seed(20261126)
        for context in pool.contexts.values(): context.planner.reset_seed()
        pool.pure_solver.reset_seed()
        for engine in pool.batch_engines.values(): engine.reset_seed();engine.calls.clear()
        pool.binding_codec.reset();pool.stage_timing.reset();pool._following_ik_stages = []
        pool.stage_timing.active = True
        torch.cuda.synchronize();began = time.perf_counter()
        try:
            result = self.planner.request(directory, **flags)
            torch.cuda.synchronize();seconds = time.perf_counter()-began
        finally:
            pool.stage_timing.active = False;pool.mesh_vote.assert_original_untouched()
        if stage == 'initial':
            manifest = read_json(directory/'candidate_order.json')
            if result['success']:
                result['selected'].update(selection_source(manifest, result['selected']))
            write_json(directory/'plan_result.json', result)
        receipt = {'variant': execution_variant, 'stage': stage, 'passed': True,
            'planning_seconds': seconds, 'success': result['success'],
            'batch_capacity': pool.ik_batch_size, 'C_YAML_enabled': pool.fast_binding_enabled,
            'obstacle_major_queries': pool.mesh_vote.enabled,
            'scalar_branch_validation': not pool.vectorized_fallback,
            'request_free_mib': free/1024**2, 'geometry_and_thresholds_unchanged': True,
            'binding_codec': pool.binding_codec.report(), 'timings': pool.stage_timing.report()}
        write_json(directory/('execution_planner_'+stage+'.json'), receipt)
        prune_inactive_meshes(mesh_stores(self.planner), torch.cuda.synchronize)
        return result


if __name__ == '__main__':
    from modules.pipeline.experiments.indexed_grasp_planning.validation_worker import main
    main(factory=create, planner_type=ExecutionPlanner)
