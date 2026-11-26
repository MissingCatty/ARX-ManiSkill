"""Optional V8 factory; the active V3 scheduler remains the control."""
from modules.curobo.dual_arm.adapter import create as original_create
from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner, configure
from modules.curobo.dual_arm.witness_pool import witness_pool
from .pool import batched_pool


def create(args):
    selected = getattr(args, 'ik_batch_size', 16)
    capacities = tuple(getattr(args, 'warm_batch_sizes', (selected,)))
    def extend(base):
        return batched_pool(witness_pool(base), capacities, selected)
    original = original_create(args, pool_extension=extend)
    return configure(InterleavedPlacementPlanner(original.pool, original.model, original.root))
