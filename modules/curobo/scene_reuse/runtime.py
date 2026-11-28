"""Optional CPU prepared-mesh reuse on top of the accepted memory planner."""
from modules.curobo.scene_cache import mesh_stores
from .cpu_runtime import CPUGeometryReuse


def create(args):
    from modules.curobo.state_updates.runtime import create as original_create
    planner = original_create(args)
    planner.cpu_geometry_reuse = CPUGeometryReuse(mesh_stores(planner))
    return planner
