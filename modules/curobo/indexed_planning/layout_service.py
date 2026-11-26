"""Identical planning models with separate original and reordered collision graphs."""
from arx_project.layout import ROOT, read_json, write_json
from modules.curobo.dual_arm.adapter import create as original_create
from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner, configure
from modules.curobo.dual_arm.witness_pool import witness_pool
from .binding_serialization import serialization_pool
from .branch_validation import branch_pool
from .pool import batched_pool
from .stage_timing import measured_pool
from .mesh_query.backend import mesh_vote_pool, MeshVoteSwitch


def create(args):
    from .kernel_layout.preflight import verify
    from .kernel_layout.kernels import obstacle_major_collision, CONTRACT
    from .warmup import warm_solver_branches
    registered = read_json(args.root.parent/'registration.json')
    verify(args.root/'layout_geometry_preflight', ROOT/registered['diagnostic_source'])

    def switch(scene):
        return MeshVoteSwitch(scene, mesh_kernel=obstacle_major_collision, contract=CONTRACT)

    def extend(base):
        return mesh_vote_pool(measured_pool(batched_pool(
            witness_pool(branch_pool(serialization_pool(base))), (16,), 16)), switch_factory=switch)
    original = original_create(args, pool_extension=extend)
    planner = configure(InterleavedPlacementPlanner(original.pool, original.model, original.root))
    planner.pool.mesh_vote_mode(True)
    proof = warm_solver_branches(planner, args.warmup_trial/'state.json', args.root/'layout_solver_prewarm')
    planner.pool.mesh_vote_mode(False)
    write_json(args.root/'layout_readiness.json', {'passed': proof['passed'],
        'original_and_fast_full_IK_have_separate_CUDA_graphs': True,
        'shared_scene_and_robot_parameters': True, 'contract': CONTRACT})
    return planner
