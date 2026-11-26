"""Component probes on the actual original narrow-phase query workload."""
import warp as wp

from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.collision.wp_collision_kernel import is_obs_enabled, load_obstacle_transform
from curobo._src.geom.collision.wp_collision_common import load_sphere_query, swept_bounds_overlap


@wp.func_native('return wp::mesh_query_inside_ray_tracing(mesh_id, point);')
def original_sign(mesh_id: wp.uint64, point: wp.vec3) -> wp.float32:
    ...


@wp.kernel
def build_queries(scene: MeshDataWarp, spheres: wp.array(dtype=wp.vec4), eta: wp.array(dtype=float),
                  total: int, capacity: int, active: wp.array(dtype=int),
                  queries: wp.array(dtype=wp.vec4), slots: wp.array(dtype=int)):
    pair = wp.tid()
    sphere = pair/capacity
    slot = pair-sphere*capacity
    if sphere >= total or not is_obs_enabled(scene, 0, slot):
        return
    query = load_sphere_query(spheres, sphere, eta[0])
    if query.radius < 0.0:
        return
    point = wp.transform_point(load_obstacle_transform(scene, 0, slot), query.center)
    if not swept_bounds_overlap(scene, 0, slot, point, point, point, query.radius_adjusted):
        return
    dims = wp.vec3(scene.dims[slot, 0], scene.dims[slot, 1], scene.dims[slot, 2])
    maximum = wp.max(wp.length(dims)*0.5, query.radius_adjusted)
    active[pair] = 1
    slots[pair] = slot
    queries[pair] = wp.vec4(point[0], point[1], point[2], maximum)


@wp.kernel
def nearest_points(meshes: wp.array(dtype=wp.uint64), queries: wp.array(dtype=wp.vec4),
                   found: wp.array(dtype=int), faces: wp.array(dtype=int), uv: wp.array(dtype=wp.vec2)):
    i = wp.tid();p = queries[i]
    result = wp.mesh_query_point_no_sign(meshes[i], wp.vec3(p[0], p[1], p[2]), p[3])
    found[i] = 0;faces[i] = -1;uv[i] = wp.vec2(0.0)
    if result.result:
        found[i] = 1;faces[i] = result.face;uv[i] = wp.vec2(result.u, result.v)


@wp.kernel
def ray_votes(meshes: wp.array(dtype=wp.uint64), queries: wp.array(dtype=wp.vec4),
               found: wp.array(dtype=int), signs: wp.array(dtype=float)):
    i = wp.tid();p = queries[i];signs[i] = 0.0
    if found[i] != 0:
        signs[i] = original_sign(meshes[i], wp.vec3(p[0], p[1], p[2]))


@wp.kernel
def combined(meshes: wp.array(dtype=wp.uint64), queries: wp.array(dtype=wp.vec4),
              found: wp.array(dtype=int), faces: wp.array(dtype=int), uv: wp.array(dtype=wp.vec2),
              signs: wp.array(dtype=float)):
    i = wp.tid();p = queries[i]
    result = wp.mesh_query_point(meshes[i], wp.vec3(p[0], p[1], p[2]), p[3])
    found[i] = 0;faces[i] = -1;uv[i] = wp.vec2(0.0);signs[i] = 0.0
    if result.result:
        found[i] = 1;faces[i] = result.face;uv[i] = wp.vec2(result.u, result.v);signs[i] = result.sign
