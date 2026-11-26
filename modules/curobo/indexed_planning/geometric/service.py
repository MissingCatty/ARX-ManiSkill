"""Optional D factory, without B batching or C context reuse."""
from modules.curobo.dual_arm.adapter import create as original_create
from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner, configure
from modules.curobo.dual_arm.witness_pool import witness_pool
from .pool import geometric_pool


def create(args):
    original = original_create(args, pool_extension=lambda base: geometric_pool(witness_pool(base)))
    return configure(InterleavedPlacementPlanner(original.pool, original.model, original.root))
