"""Probe-only kernels must reproduce the original nearest point and sign."""
import unittest

import numpy as np
import trimesh
import warp as wp

from curobo._src.geom.data.data_mesh import MeshDataWarp
from .kernels import nearest_points, ray_votes, combined, build_queries


class ComponentTests(unittest.TestCase):
    def test_separated_queries_match_original_inside_outside_surface_and_miss(self):
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        points = wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device='cpu')
        faces = wp.array(mesh.faces.reshape(-1).astype(np.int32), dtype=wp.int32, device='cpu')
        model = wp.Mesh(points=points, indices=faces, bvh_constructor='sah')
        queries = wp.array(np.array([[0, 0, 0, 3], [1, 0, 0, 3], [1.01, .2, .1, 3],
                                    [-.8, -.7, -.9, 3], [4, 0, 0, .2], [1, 1, 1, 3]], dtype=np.float32),
                           dtype=wp.vec4, device='cpu')
        ids = wp.array(np.full(6, model.id, dtype=np.uint64), dtype=wp.uint64, device='cpu')
        first = [wp.empty(6, dtype=t, device='cpu') for t in [wp.int32, wp.int32, wp.vec2, wp.float32]]
        second = [wp.empty(6, dtype=t, device='cpu') for t in [wp.int32, wp.int32, wp.vec2, wp.float32]]
        wp.launch(nearest_points, dim=6, inputs=[ids, queries, *first[:3]], device='cpu')
        wp.launch(ray_votes, dim=6, inputs=[ids, queries, first[0], first[3]], device='cpu')
        wp.launch(combined, dim=6, inputs=[ids, queries, *second], device='cpu')
        for a, b in zip(first, second): np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertEqual(first[3].numpy()[0], -1.)
        self.assertEqual(first[0].numpy()[4], 0)

    def test_disabled_slots_do_not_become_worklist_entries(self):
        scene = MeshDataWarp();scene.max_n = 1
        scene.n_per_env = wp.zeros(1, dtype=wp.int32, device='cpu')
        spheres = wp.zeros(2, dtype=wp.vec4, device='cpu');eta = wp.ones(1, device='cpu')
        active = wp.zeros(2, dtype=wp.int32, device='cpu')
        queries = wp.empty(2, dtype=wp.vec4, device='cpu');slots = wp.empty(2, dtype=wp.int32, device='cpu')
        wp.launch(build_queries, dim=2, inputs=[scene, spheres, eta, 2, 1, active, queries, slots], device='cpu')
        np.testing.assert_array_equal(active.numpy(), [0, 0])


if __name__ == '__main__': unittest.main()
