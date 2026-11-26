"""Fixed-input CUDA graph timings; never substitute probe outputs into IK."""
from pathlib import Path
import statistics

import numpy as np
import torch
import warp as wp

from arx_project.layout import sha256, write_json
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import get_warp_device_stream
from .kernels import build_queries, nearest_points, ray_votes, combined


def measure(snapshot, scene, directory):
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=False)
    spheres = snapshot['spheres'];data = scene.data.meshes
    if data is None or scene.data.num_envs != 1:
        raise ValueError('Component probe requires the frozen single mesh world')
    count = spheres.numel()//4;capacity = data.max_n;pairs = count*capacity
    active = torch.zeros(pairs, dtype=torch.int32, device=spheres.device)
    slots = torch.empty_like(active);queries = torch.empty((pairs, 4), device=spheres.device)
    device, stream = get_warp_device_stream(spheres)
    wp.launch(build_queries, dim=pairs, inputs=[data.to_warp(),
        wp.from_torch(spheres.detach().reshape(-1, 4), dtype=wp.vec4), wp.from_torch(snapshot['eta']),
        count, capacity, wp.from_torch(active), wp.from_torch(queries, dtype=wp.vec4), wp.from_torch(slots)],
        device=device, stream=stream)
    selected = active.nonzero().reshape(-1)
    if not len(selected): raise ValueError('No actual narrow-phase queries were captured')
    queries = queries.index_select(0, selected).contiguous()
    slots = slots.index_select(0, selected).long().contiguous()
    ids = data.mesh_ids.reshape(-1).index_select(0, slots).contiguous()
    n = len(queries)
    found = torch.empty(n, device=spheres.device, dtype=torch.int32);faces = torch.empty_like(found)
    uv = torch.empty((n, 2), device=spheres.device);signs = torch.empty(n, device=spheres.device)
    full_found = torch.empty_like(found);full_faces = torch.empty_like(faces)
    full_uv = torch.empty_like(uv);full_signs = torch.empty_like(signs)
    wids = wp.from_torch(ids, dtype=wp.uint64);wqueries = wp.from_torch(queries, dtype=wp.vec4)

    def nearest():
        wp.launch(nearest_points, dim=n, inputs=[wids, wqueries, wp.from_torch(found),
            wp.from_torch(faces), wp.from_torch(uv, dtype=wp.vec2)], device=device,
            stream=get_warp_device_stream(spheres)[1])

    def rays():
        wp.launch(ray_votes, dim=n, inputs=[wids, wqueries, wp.from_torch(found),
            wp.from_torch(signs)], device=device, stream=get_warp_device_stream(spheres)[1])

    def fused():
        wp.launch(combined, dim=n, inputs=[wids, wqueries, wp.from_torch(full_found),
            wp.from_torch(full_faces), wp.from_torch(full_uv, dtype=wp.vec2), wp.from_torch(full_signs)],
            device=device, stream=get_warp_device_stream(spheres)[1])

    # Give this diagnostic its own output/gradient buffers. The solver's
    # buffers and its cached metrics must remain untouched.
    buffer = CollisionBuffer.from_shape(spheres.shape, DeviceCfg())

    def dispatched():
        scene.checker.get_sphere_distance(scene=scene.data, query_sphere=spheres,
            collision_buffer=buffer, weight=snapshot['weight'], activation_distance=snapshot['eta'],
            env_query_idx=snapshot['env_index'], return_loss=snapshot['return_loss'])

    nearest();rays();fused();dispatched();torch.cuda.synchronize()
    if not all(torch.equal(a, b) for a, b in [(found, full_found), (faces, full_faces),
                                             (uv, full_uv), (signs, full_signs)]):
        raise ValueError('Separated nearest-point/ray results differ from the original combined query')
    functions = {'nearest_point': nearest, 'three_ray_votes': rays, 'combined_compact': fused,
                 'original_collision_dispatch': dispatched}
    graphs = {}
    repeats = 8
    for name, function in functions.items():
        for _ in range(2): function()
        torch.cuda.synchronize();graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats): function()
        graphs[name] = graph
    samples = {name: [] for name in functions};names = list(functions)
    # Rotate mode order; all modes use identical, immutable points and mesh IDs.
    for repetition in range(8):
        order = names[repetition%4:]+names[:repetition%4]
        for name in order:
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record();graphs[name].replay();end.record();end.synchronize()
            samples[name].append(begin.elapsed_time(end)/1000./repeats)
    torch.cuda.synchronize()
    np.savez(directory/'actual_queries.npz', spheres=spheres.detach().cpu().numpy(),
             queries=queries.detach().cpu().numpy(), mesh_slots=slots.detach().cpu().numpy(),
             nearest_found=found.detach().cpu().numpy(), eta=snapshot['eta'].detach().cpu().numpy(),
             weight=snapshot['weight'].detach().cpu().numpy())
    # Export the actual GPU meshes, not a later interpretation of a source
    # file that could have been replaced by another binding window.
    active_meshes = int(data.count[0].item())
    lookup = {item.mesh_id: item for item in data.wp_cache.values()}
    exported = {name: getattr(data, name).detach().cpu().numpy() for name in
                ['inv_pose', 'dims', 'bounds_lower', 'bounds_upper', 'enable', 'count']}
    for slot, mesh_id in enumerate(data.mesh_ids[0, :active_meshes].detach().cpu().tolist()):
        item = lookup[mesh_id]
        exported['vertices_'+str(slot)] = item.vertices.numpy()
        exported['faces_'+str(slot)] = item.faces.numpy()
    np.savez(directory/'actual_meshes.npz', **exported)
    (directory/'scene.json').write_bytes(snapshot['scene_bytes'])
    result = {'actual_sphere_shape': list(spheres.shape), 'sphere_count': count, 'mesh_cache_capacity': capacity,
        'active_meshes': active_meshes, 'launched_pairs': pairs,
        'actual_narrow_phase_queries': n, 'nearest_found': int(found.sum().item()),
        'timings': {name: {'samples_seconds': values, 'median_seconds': statistics.median(values)}
                    for name, values in samples.items()},
        'nearest_face_UV_and_ray_sign_bitwise_equal': True, 'modes_use_identical_queries': True,
        'timing_is_standalone_component_not_planning_RPC': True,
        'query_sha256': sha256(directory/'actual_queries.npz'), 'mesh_sha256': sha256(directory/'actual_meshes.npz'),
        'scene_sha256': sha256(directory/'scene.json')}
    write_json(directory/'components.json', result)
    return result
