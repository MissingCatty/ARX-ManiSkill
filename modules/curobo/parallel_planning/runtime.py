"""Optional batch-path entry; normal registry and endpoint solver unchanged."""
from modules.curobo.gripper_sweep.runtime import create as create_original
from .planning import parallel_pool


def create(args):
    return create_original(args, pool_extension=parallel_pool)
