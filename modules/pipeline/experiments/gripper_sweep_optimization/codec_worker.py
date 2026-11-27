"""Second comparison: batched grippers in both arms, original versus C YAML."""
from unittest.mock import patch
from pathlib import Path
from arx_project.layout import write_json
from modules.curobo.gripper_sweep.codec_adapter import create
from . import worker as base


class Worker(base.Worker):
    def request(self,directory,*,variant,**flags):
        pool=self.planner.pool;pool.fast_binding_enabled=variant=='batched';pool.binding_codec.reset()
        result=super().request(directory,variant='batched',**flags)
        result.update(variant=variant,C_YAML_enabled=pool.fast_binding_enabled,binding_codec=pool.binding_codec.report())
        write_json(Path(directory)/'planning_metrics.json',result)
        return result


if __name__=='__main__':
    with patch('modules.curobo.gripper_sweep.adapter.create',create),patch.object(base,'Worker',Worker):base.main()
