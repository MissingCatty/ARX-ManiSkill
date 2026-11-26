"""Retain the original nearest point and exact three-axis majority result."""
from pathlib import Path
import types

import warp as wp

from curobo._src.geom.data.data_mesh import MeshDataWarp, get_obs_idx, _SDF_EPS
from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
from .native_contract import verify_native_contract


CONTRACT = verify_native_contract((Path(wp.__file__).parent/'native/mesh.h').read_text())


@wp.func_native('''
wp::Mesh mesh = wp::mesh_get(mesh_id);
int vote = 0;
float sign;
for (int i = 0; i < 3; ++i) {
    if (wp::mesh_query_ray_closest_sign(mesh, point,
            wp::vec3(i == 0 ? 1.0f : 0.0f, i == 1 ? 1.0f : 0.0f,
                     i == 2 ? 1.0f : 0.0f), sign) && sign < 0) {
        vote++;
    }
    if (vote >= 2) return -1.0f;
    if ((i + 1 - vote) >= 2) return 1.0f;
}
return vote >= 2 ? -1.0f : 1.0f;
''')
def exact_majority_sign(mesh_id: wp.uint64, point: wp.vec3) -> wp.float32:
    ...


@wp.func
def exact_mesh_sdf(obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32,
                   local_pt: wp.vec3, query_distance: wp.float32) -> wp.vec4:
    idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    mesh_id = obs_set.mesh_ids[idx]
    dims = wp.vec3(obs_set.dims[idx, 0], obs_set.dims[idx, 1], obs_set.dims[idx, 2])
    max_distance = wp.max(wp.length(dims)*0.5, query_distance)
    result = wp.mesh_query_point_no_sign(mesh_id, local_pt, max_distance)
    if not result.result:
        return wp.vec4(max_distance, 0.0, 0.0, 0.0)
    sign = exact_majority_sign(mesh_id, local_pt)
    closest = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    delta = closest-local_pt
    length = wp.length(delta)
    grad = wp.vec3(0.0, 0.0, 0.0)
    if length > _SDF_EPS:
        grad = sign*delta/length
    return wp.vec4(length*sign, grad[0], grad[1], grad[2])


def private_kernel():
    original = sphere_obstacle_collision_kernel.func
    namespace = dict(original.__globals__, compute_local_sdf_with_grad=exact_mesh_sdf)
    function = types.FunctionType(original.__code__, namespace, 'exact_vote_collision',
                                  original.__defaults__, original.__closure__)
    function.__annotations__ = original.__annotations__
    function.__module__ = __name__
    return wp.kernel(function)


exact_vote_collision = private_kernel()
