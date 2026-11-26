"""Derive an original-SDF kernel with obstacle-major indexing only."""
import inspect
import linecache

import warp as wp

from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
from .ordering import reorder_source


SOURCE = reorder_source(inspect.getsource(sphere_obstacle_collision_kernel.func))
FILENAME = '<arx_obstacle_major_collision>'
linecache.cache[FILENAME] = (len(SOURCE), None, SOURCE.splitlines(True), FILENAME)
namespace = dict(sphere_obstacle_collision_kernel.func.__globals__, __name__=__name__)
exec(compile(SOURCE, FILENAME, 'exec'), namespace)
obstacle_major_collision = wp.kernel(namespace['obstacle_major_collision'])
CONTRACT = {'only_task_index_order_changed': True, 'all_original_sphere_obstacle_pairs_preserved': True,
            'original_SDF_ray_votes_activation_and_accumulation': True,
            'upstream_collision_kernel_unchanged': True}
