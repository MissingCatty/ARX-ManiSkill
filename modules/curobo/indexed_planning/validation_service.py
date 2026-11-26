"""Optional B16 branch batching and stage instrumentation; default unchanged."""
from modules.curobo.dual_arm.adapter import create as original_create
from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner, configure
from modules.curobo.dual_arm.witness_pool import witness_pool
from .branch_validation import branch_pool
from .pool import batched_pool
from .stage_timing import measured_pool


def create(args):
    def extend(base):
        return measured_pool(batched_pool(witness_pool(branch_pool(base)), (16,), 16))
    original = original_create(args, pool_extension=extend)
    return configure(InterleavedPlacementPlanner(original.pool, original.model, original.root))
