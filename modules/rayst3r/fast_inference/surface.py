"""Deterministic area-uniform sampling for independent mesh evaluation."""
import numpy as np


def uniform_surface(vertices, faces, count=50000, seed=20261122):
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int64)
    triangles=vertices[faces]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('Invalid ground-truth mesh surface')
    rng=np.random.default_rng(seed)
    chosen=rng.choice(len(triangles),size=count,p=area/area.sum())
    u=np.sqrt(rng.random(count));v=rng.random(count)
    return ((1-u)[:,None]*triangles[chosen,0]+(u*(1-v))[:,None]*triangles[chosen,1]
            +(u*v)[:,None]*triangles[chosen,2])
