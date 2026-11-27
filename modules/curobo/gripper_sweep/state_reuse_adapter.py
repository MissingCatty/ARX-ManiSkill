"""Optional state/geometry reuse around the accepted width-aware planner."""
from .codec_adapter import create as original_create
from .width_refresh import attach as attach_width
from .sphere_cache import attach as attach_spheres


def create(args):
    planner = original_create(args)
    attach_width(planner.pool)
    attach_spheres(planner.pool)
    return planner
