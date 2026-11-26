"""Original NMS15+B16+C binding with optional actual-query measurement."""
from modules.curobo.dual_arm.adapter import create as original_create
from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner, configure
from modules.curobo.dual_arm.witness_pool import witness_pool
from .binding_serialization import serialization_pool
from .branch_validation import branch_pool
from .pool import batched_pool
from .stage_timing import measured_pool
from .query_probe.capture import capture_pool


def create(args):
    def extend(base):
        return capture_pool(measured_pool(batched_pool(
            witness_pool(branch_pool(serialization_pool(base))), (16,), 16)))
    original = original_create(args, pool_extension=extend)
    return configure(InterleavedPlacementPlanner(original.pool, original.model, original.root))
