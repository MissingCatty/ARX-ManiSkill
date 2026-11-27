"""Create the accepted width-aware planner with batched analytic endpoints."""
from modules.curobo.gripper_sweep.runtime import create as create_original
from .planning import analytic_pool


def create(args):
    return create_original(args,pool_extension=analytic_pool)
