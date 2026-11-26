"""Optional C variant, independent of native endpoint batching."""
from modules.curobo.dual_arm.adapter import create as original_create
from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner, configure
from modules.curobo.dual_arm.witness_pool import witness_pool
from .context_reuse import routed_pool


def create(args):
    original = original_create(args, pool_extension=lambda base: routed_pool(witness_pool(base)))
    return configure(InterleavedPlacementPlanner(original.pool, original.model, original.root))
