"""Compare actual scene SDFs and complete collision costs before planning."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from arx_project.layout import ROOT, now, sha256, write_json
from curobo.scene import Scene
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.collision.wp_collision_kernel import compute_local_sdf_with_grad
from curobo._src.util.warp import get_warp_device_stream
from .kernels import exact_mesh_sdf
from .backend import MeshVoteSwitch


@wp.kernel
def compare_sdf(scene: MeshDataWarp, points: wp.array(dtype=wp.vec3), index: int,
                radius: float, original: wp.array(dtype=wp.vec4), fast: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    original[i] = compute_local_sdf_with_grad(scene, 0, index, points[i], radius)
    fast[i] = exact_mesh_sdf(scene, 0, index, points[i], radius)


def verify(directory, source):
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=False)
    reg = json.loads((source/'registration.json').read_text());rng = np.random.default_rng(20261126)
    reports = [];checked_points = 0
    for case in reg['cases']:
        path = source/'plans'/case['id']/'profiled_1/arms/right/plan/scene.json'
        scene = Scene.create(json.loads(path.read_text()))
        device = DeviceCfg()
        world = SceneCollision.from_config(SceneCollisionCfg(device_cfg=device, scene_model=scene,
            cache={'mesh': 8, 'primitive': 8}))
        switch = MeshVoteSwitch(world)
        mesh_data = world.data.meshes
        if mesh_data is None: raise ValueError('Actual scene mesh data is missing')
        mesh_reports = [];near_world = []
        for index, mesh in enumerate(scene.mesh):
            vertices = np.asarray(mesh.get_mesh_data()[0], dtype=np.float32)
            pose = np.asarray(mesh.pose)
            rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            local = vertices[rng.integers(len(vertices), size=64)]
            near_world.extend(local@rotation.T+pose[:3])
            lo, hi = vertices.min(0), vertices.max(0)
            points = rng.uniform(lo-.03, hi+.03, (1024, 3)).astype(np.float32)
            near = vertices[rng.integers(len(vertices), size=1024)]
            points = np.concatenate([points, near,
                near+rng.normal(size=near.shape).astype(np.float32)*.001,
                np.asarray([(lo+hi)/2, lo, hi], dtype=np.float32)])
            points = torch.as_tensor(points, device='cuda').contiguous()
            original = torch.empty((len(points), 4), device='cuda');fast = torch.empty_like(original)
            wp_device, stream = get_warp_device_stream(points)
            wp.launch(compare_sdf, dim=len(points), inputs=[mesh_data.to_warp(),
                wp.from_torch(points, dtype=wp.vec3), index, .02,
                wp.from_torch(original, dtype=wp.vec4), wp.from_torch(fast, dtype=wp.vec4)],
                device=wp_device, stream=stream)
            torch.cuda.synchronize()
            delta = float((original-fast).abs().max().item())
            if not torch.isfinite(original).all() or not torch.equal(original, fast):
                raise ValueError(f'Exact mesh SDF/gradient mismatch for {mesh.name}: {delta}')
            checked_points += len(points)
            mesh_reports.append({'name': mesh.name, 'points': len(points), 'SDF_and_gradient_bitwise_equal': True,
                                 'maximum_absolute_difference': delta})
        # Include the real transforms and original atomic accumulation, not
        # just mesh-local point queries. Different obstacle threads may add
        # in different orders; allow float32 roundoff only in this sum.
        lo = np.array([-.8, -.8, -.1]);hi = np.array([.8, .8, 1.8])
        xyz = rng.uniform(lo, hi, (4, 1, 256, 3)).astype(np.float32)
        xyz.reshape(-1, 3)[:len(near_world)] = np.asarray(near_world, dtype=np.float32)
        radii = rng.uniform(.005, .04, (4, 1, 256, 1)).astype(np.float32)
        spheres = torch.as_tensor(np.concatenate([xyz, radii], -1), device='cuda')
        weight = torch.tensor([1.0], device='cuda');eta = torch.tensor([.02], device='cuda')
        values = []
        for enabled in [False, True]:
            switch.enabled = enabled
            query = spheres.detach().clone().requires_grad_(True)
            buffer = CollisionBuffer.from_shape(query.shape, device)
            cost = world.checker.get_sphere_distance(scene=world.data, query_sphere=query,
                collision_buffer=buffer, weight=weight, activation_distance=eta, return_loss=True)
            cost.sum().backward()
            values.append((cost.detach().clone(), query.grad.detach().clone()))
        differences = []
        if not (values[0][0] > 0).any() or not (values[0][1].abs() > 0).any():
            raise ValueError('Collision preflight did not exercise positive cost and gradients')
        for first, second in zip(values[0], values[1]):
            if not torch.isfinite(first).all() or not torch.allclose(first, second, rtol=2e-6, atol=2e-7):
                raise ValueError('Complete collision cost or gradient changed')
            differences.append(float((first-second).abs().max().item()))
        reports.append({'case': case['id'], 'source_scene_sha256': sha256(path), 'meshes': mesh_reports,
                        'complete_cost_gradient_differences': differences, 'complete_spheres': spheres.numel()//4,
                        'nonzero_collision_rows': int((values[0][0] > 0).sum().item())})
        del world, switch, mesh_data
    result = {'at': now(), 'passed': True, 'checked_sdf_points': checked_points,
              'cases': reports, 'geometry_and_thresholds_unchanged': True,
              'physical_grasps': 0, 'planning_requests': 0}
    write_json(directory/'acceptance.json', result)
    return result
