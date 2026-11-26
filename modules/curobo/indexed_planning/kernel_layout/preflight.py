"""Replay captured full-IK collision inputs before measuring planner latency."""
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from arx_project.layout import now, sha256, write_json
from curobo.scene import Scene
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from ..mesh_query.backend import MeshVoteSwitch
from .kernels import obstacle_major_collision, CONTRACT


def captured_world(directory):
    """Use the GPU-exported triangles and packed transforms, without re-meshing."""
    meshes = np.load(directory/'actual_meshes.npz', allow_pickle=False)
    metadata = json.loads((directory/'components.json').read_text())
    source = json.loads((directory/'scene.json').read_text())
    source['mesh'] = {f'slot_{i}': {'pose': [0, 0, 0, 1, 0, 0, 0],
        'vertices': meshes[f'vertices_{i}'].tolist(),
        'faces': meshes[f'faces_{i}'].reshape(-1, 3).tolist()}
        for i in range(metadata['active_meshes'])}
    device = DeviceCfg()
    world = SceneCollision.from_config(SceneCollisionCfg(device_cfg=device, scene_model=Scene.create(source),
        cache={'mesh': metadata['mesh_cache_capacity'], 'primitive': 8}))
    data = world.data.meshes
    for name in ['inv_pose', 'dims', 'bounds_lower', 'bounds_upper', 'enable', 'count']:
        target = getattr(data, name)
        value = torch.as_tensor(meshes[name], device=target.device, dtype=target.dtype)
        if value.shape != target.shape:
            raise ValueError('Captured mesh layout changed: '+name)
        target.copy_(value)
        if not torch.equal(value, target): raise ValueError('Captured mesh field changed: '+name)
    meshes.close()
    return world, device


def compare_outputs(original, reordered):
    differences = []
    for first, second in zip(original, reordered):
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError('Nonfinite collision cost or gradient')
        if not torch.allclose(first, second, rtol=2e-6, atol=2e-7):
            raise ValueError('Collision cost or gradient changed beyond float32 accumulation roundoff')
        differences.append(float((first-second).abs().max().item()))
    return differences


def verify(directory, source):
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=False)
    registration = json.loads((source/'registration.json').read_text())
    reports = []
    for case in registration['cases']:
        captured = source/'plans'/case['id']/'captured_1/query_components'
        world, device = captured_world(captured)
        switch = MeshVoteSwitch(world, mesh_kernel=obstacle_major_collision, contract=CONTRACT)
        with np.load(captured/'actual_queries.npz', allow_pickle=False) as values:
            spheres = torch.as_tensor(values['spheres'], device=device.device).contiguous()
            eta = torch.as_tensor(values['eta'], device=device.device).contiguous()
            weight = torch.as_tensor(values['weight'], device=device.device).contiguous()
        outputs = [];buffers = []
        for enabled in [False, True]:
            switch.enabled = enabled
            query = spheres.detach().clone().requires_grad_(True)
            buffer = CollisionBuffer.from_shape(query.shape, device);buffers.append(buffer)
            cost = world.checker.get_sphere_distance(scene=world.data, query_sphere=query,
                collision_buffer=buffer, weight=weight, activation_distance=eta, return_loss=True)
            cost.sum().backward()
            outputs.append((cost.detach().clone(), query.grad.detach().clone()))
        differences = compare_outputs(*outputs)
        # Separate fixed graphs; these timings do not enter the planning RPC table.
        graphs = []
        for enabled, buffer in zip([False, True], buffers):
            switch.enabled = enabled
            def launch():
                world.checker.get_sphere_distance(scene=world.data, query_sphere=spheres,
                    collision_buffer=buffer, weight=weight, activation_distance=eta, return_loss=True)
            for _ in range(2): launch()
            torch.cuda.synchronize();graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8): launch()
            graphs.append(graph)
        times = [[], []]
        for repeat in range(8):
            for index in ([0, 1] if repeat%2 == 0 else [1, 0]):
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record();graphs[index].replay();end.record();end.synchronize()
                times[index].append(begin.elapsed_time(end)/8000.)
        switch.assert_original_untouched()
        reports.append({'case': case['id'], 'sphere_shape': list(spheres.shape),
            'spheres': spheres.numel()//4, 'active_meshes': int(world.data.meshes.count[0].item()),
            'maximum_cost_gradient_difference': differences,
            'positive_cost_rows': int((outputs[0][0] > 0).sum().item()),
            'nonzero_gradient_values': int((outputs[0][1] != 0).sum().item()),
            'collision_seconds': {name: {'samples': samples, 'median': statistics.median(samples)}
                for name, samples in zip(['control', 'obstacle_major'], times)},
            'input_sha256': {p: sha256(captured/p) for p in
                ['actual_queries.npz', 'actual_meshes.npz', 'scene.json', 'components.json']}})
        del graphs, buffers, outputs, world, switch
    if not sum(r['positive_cost_rows'] for r in reports) or not sum(r['nonzero_gradient_values'] for r in reports):
        raise ValueError('Captured preflight never exercised positive collision cost and gradient')
    result = {'at': now(), 'passed': True, 'captured_cases': len(reports),
        'checked_spheres': sum(r['spheres'] for r in reports), 'cases': reports,
        'full_collision_cost_and_gradient_equal': True, 'contract': CONTRACT,
        'microtimings_separate_from_planning_RPC': True, 'physical_grasps': 0, 'planning_requests': 0}
    write_json(directory/'acceptance.json', result)
    return result
