"""CPU geometry metrics for a union of balls and real convex colliders."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import trimesh


def directions(count):
    index = np.arange(count, dtype=float) + .5
    z = 1 - 2 * index / count
    angle = index * (np.pi * (3 - np.sqrt(5)))
    radial = np.sqrt(1 - z * z)
    return np.c_[radial * np.cos(angle), radial * np.sin(angle), z]


def ball_array(spheres):
    return np.asarray([[*s['center'], s['radius']] for s in spheres], dtype=float)


def sphere_gap(points, balls, chunk=2048):
    """Positive values are exact distances from outside the sphere union."""
    result = np.empty(len(points))
    for start in range(0, len(points), chunk):
        distance = cdist(points[start:start + chunk], balls[:, :3])
        result[start:start + chunk] = (distance - balls[:, 3]).min(axis=1)
    return result


def sphere_boundary(balls, per_sphere=1024):
    balls = np.unique(balls, axis=0)
    unit = np.concatenate([directions(per_sphere), np.eye(3), -np.eye(3)])
    points = (balls[:, None, :3] + balls[:, None, 3:] * unit).reshape(-1, 3)
    weights = np.repeat(4 * np.pi * balls[:, 3] ** 2 / len(unit), len(unit))
    visible = sphere_gap(points, balls) >= -1e-9
    return points[visible], weights[visible]


def weighted_stats(values, weights=None):
    values = np.asarray(values, dtype=float)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != len(weights) or not len(values):
        raise ValueError('Missing geometry samples')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid geometry samples or weights')
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return {'samples': len(values), 'area_m2': float(weights.sum()),
        'mean_mm': float(np.average(values, weights=weights) * 1000),
        'p50_mm': float(values[order[min(np.searchsorted(cdf, .5), len(values)-1)]] * 1000),
        'p95_mm': float(values[order[min(np.searchsorted(cdf, .95), len(values)-1)]] * 1000),
        'p99_mm': float(values[order[min(np.searchsorted(cdf, .99), len(values)-1)]] * 1000),
        'sampled_max_mm': float(values.max() * 1000),
        'fraction_over': {str(mm): float(weights[values > mm / 1000].sum() / weights.sum())
            for mm in [.05, .1, .5, 1, 2, 3, 5]}}


@dataclass
class Part:
    vertices: np.ndarray
    faces: np.ndarray
    equations: object
    normals: np.ndarray


class ConvexUnion:
    """Preserve convex pieces and zero-thickness polygons without filling gaps."""
    def __init__(self, pieces):
        self.parts = []
        for values in pieces:
            points = np.unique(np.asarray(values, dtype=float), axis=0)
            centered = points - points.mean(0)
            rank = np.linalg.matrix_rank(centered)
            if rank == 3:
                hull = ConvexHull(points)
                faces = hull.simplices
                equations = hull.equations
                normals = equations[:, :3]
            elif rank == 2:
                _, _, basis = np.linalg.svd(centered, full_matrices=False)
                polygon = ConvexHull(centered @ basis[:2].T).vertices
                faces = np.array([[polygon[0], polygon[i], polygon[i+1]] for i in range(1, len(polygon)-1)])
                equations = None
                normals = np.broadcast_to(basis[2], (len(faces), 3)).copy()
            else:
                raise ValueError('Expected a convex solid or a planar collider')
            self.parts.append(Part(points, faces, equations, normals))
        self.mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=p.vertices, faces=p.faces,
            process=False) for p in self.parts])
        self.triangles = self.mesh.triangles
        self.normals = np.concatenate([p.normals for p in self.parts])
        self.area = trimesh.triangles.area(self.triangles)

    def inside(self, points, tolerance=1e-10):
        inside = np.zeros(len(points), dtype=bool)
        for part in self.parts:
            if part.equations is None:
                continue
            lo, hi = part.vertices.min(0), part.vertices.max(0)
            indices = np.flatnonzero(~inside & (points >= lo-tolerance).all(axis=1) &
                                     (points <= hi+tolerance).all(axis=1))
            for start in range(0, len(indices), 1024):
                idx = indices[start:start+1024]
                distances = points[idx] @ part.equations[:, :3].T + part.equations[:, 3]
                inside[idx] |= (distances <= tolerance).all(axis=1)
        return inside

    def outside_distance(self, points, chunk=1024):
        result = np.zeros(len(points))
        indices = np.flatnonzero(~self.inside(points))
        for start in range(0, len(indices), chunk):
            idx = indices[start:start+chunk]
            _, distances, _ = trimesh.proximity.closest_point(self.mesh, points[idx])
            result[idx] = distances
        return result

    def sample_surface(self, count, seed):
        rng = np.random.default_rng(seed)
        faces = rng.choice(len(self.triangles), size=count, p=self.area/self.area.sum())
        u = np.sqrt(rng.uniform(size=count));v = rng.uniform(size=count)
        triangle = self.triangles[faces]
        points = (1-u[:, None])*triangle[:, 0] + (u*(1-v))[:, None]*triangle[:, 1] + (u*v)[:, None]*triangle[:, 2]
        # A sample hidden behind another solid's outward side is internal.
        exposed = ~self.inside(points + self.normals[faces] * 1e-7)
        weights = np.full(count, self.area.sum()/count)
        return points[exposed], weights[exposed]

    def critical_vertices(self):
        return np.unique(np.concatenate([p.vertices[np.unique(p.faces)] for p in self.parts]), axis=0)


def support_differences(balls, reference, normals):
    """Exact support-plane displacement for every supplied direction."""
    a = (normals @ balls[:, :3].T + balls[:, 3]).max(axis=1)
    b = (normals @ reference[:, :3].T + reference[:, 3]).max(axis=1)
    return a - b
