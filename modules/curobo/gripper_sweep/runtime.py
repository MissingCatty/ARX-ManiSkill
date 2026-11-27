"""Explicit optional V7 entry; the global planner registry is unchanged."""
from arx_project.layout import ROOT, read_json


def create(args):
    config = read_json(ROOT/'configs/experiments/graspgenx_native_postprocessing.json')['optional_runtime']
    if config.get('C_YAML_binding', False):
        from .codec_adapter import create as factory
    else:
        from .adapter import create as factory
    planner = factory(args)
    planner.pool.batch_sweep_enabled = bool(config.get('gripper_sweep_batching', False))
    if hasattr(planner.pool, 'fast_binding_enabled'):
        planner.pool.fast_binding_enabled = bool(config.get('C_YAML_binding', False))
    return planner
