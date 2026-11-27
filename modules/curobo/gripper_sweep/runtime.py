"""Explicit optional V7 entry; the global planner registry is unchanged."""
from arx_project.layout import ROOT, read_json


def create(args,*,pool_extension=None):
    config = read_json(ROOT/'configs/experiments/graspgenx_native_postprocessing.json')['optional_runtime']
    if pool_extension is None and config.get('GPU_analytic_endpoints',False):
        from modules.curobo.analytic_ik.planning import analytic_pool
        pool_extension=analytic_pool
    if config.get('C_YAML_binding', False):
        from .codec_adapter import create as factory
    else:
        from .adapter import create as factory
    planner = factory(args,**({} if pool_extension is None else {'pool_extension':pool_extension}))
    planner.pool.batch_sweep_enabled = bool(config.get('gripper_sweep_batching', False))
    if hasattr(planner.pool, 'fast_binding_enabled'):
        planner.pool.fast_binding_enabled = bool(config.get('C_YAML_binding', False))
    if config.get('gripper_native_sweep_fast',False):
        from functools import partial
        from .direct_sweep import native_opening_spheres
        planner.pool.fast_sweep_enabled=True
        planner.pool.make_opening_spheres=partial(native_opening_spheres,planner.pool)
    return planner
