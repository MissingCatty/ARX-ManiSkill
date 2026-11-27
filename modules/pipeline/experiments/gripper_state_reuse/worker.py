"""Compare one state reuse change at a time on the accepted original pool."""
from pathlib import Path
from unittest.mock import patch

from arx_project.layout import read_json,write_json
from modules.curobo.gripper_sweep.state_reuse_adapter import create as base_create
from modules.pipeline.experiments.gripper_sweep_optimization import worker as base
from modules.pipeline.experiments.gripper_sweep_optimization import preflight as checked

original_preflight=checked.run
original_serial=checked.serial_opening


def create(args):
    planner=base_create(args)
    reg=read_json(Path(args.root).parent/'registration.json')
    planner.pool.reuse_phase=reg['reuse_phase']
    planner.pool.accepted_width_refresh=reg['accepted_width_refresh']
    planner.pool.sweep_strategy=reg.get('sweep_strategy','memo')
    planner.pool.fast_binding_enabled=True
    return planner


def preflight(planner,warmup,root):
    pool=planner.pool
    pool.width_refresh_enabled=(pool.reuse_phase=='binding' or pool.accepted_width_refresh)
    pool.sphere_reuse_enabled=pool.reuse_phase=='geometry' and pool.sweep_strategy=='memo'
    pool.fast_sweep_enabled=pool.reuse_phase=='geometry' and pool.sweep_strategy=='direct_fk'
    def serial(*args,**kwargs):
        flags=pool.width_refresh_enabled,pool.sphere_reuse_enabled,pool.fast_sweep_enabled
        pool.width_refresh_enabled=False;pool.sphere_reuse_enabled=False;pool.fast_sweep_enabled=False
        try:return original_serial(*args,**kwargs)
        finally:pool.width_refresh_enabled,pool.sphere_reuse_enabled,pool.fast_sweep_enabled=flags
    with patch.object(checked,'serial_opening',serial):reports=original_preflight(planner,warmup,root)
    if pool.reuse_phase=='binding' or pool.accepted_width_refresh:
        from .parameter_preflight import run
        run(planner,warmup,root.parent/'parameter_refresh_equivalence')
    return reports


class Worker(base.Worker):
    def request(self,directory,*,variant,**flags):
        if variant not in ['legacy','batched']:raise ValueError('Unknown reuse variant')
        pool=self.planner.pool;pool.width_only_binder.clear();pool.sphere_memo.reset()
        pool.width_refresh_enabled=(variant=='batched' if pool.reuse_phase=='binding' else pool.accepted_width_refresh)
        pool.sphere_reuse_enabled=pool.reuse_phase=='geometry' and pool.sweep_strategy=='memo' and variant=='batched'
        pool.fast_sweep_enabled=pool.reuse_phase=='geometry' and pool.sweep_strategy=='direct_fk' and variant=='batched'
        pool.fast_binding_enabled=True;pool.binding_codec.reset()
        result=super().request(directory,variant='batched',**flags)
        result.update(variant=variant,C_YAML_enabled=True,reuse_phase=pool.reuse_phase,
            width_refresh_enabled=pool.width_refresh_enabled,sphere_reuse_enabled=pool.sphere_reuse_enabled,
            fast_sweep_enabled=pool.fast_sweep_enabled,
            width_reuse={'hits':pool.width_only_binder.hits,'full_binds':pool.width_only_binder.full_binds},
            sphere_reuse=pool.sphere_memo.report())
        write_json(Path(directory)/'planning_metrics.json',result)
        return result


if __name__=='__main__':
    with patch('modules.curobo.gripper_sweep.adapter.create',create), \
         patch.object(base,'Worker',Worker),patch.object(checked,'run',preflight):base.main()
