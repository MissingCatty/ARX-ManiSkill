"""Cull provably irrelevant scene points, preserving every detector output."""
import numpy as np
from .kernels import native_detect


def relevant_points(poses,points):
    poses=np.asarray(poses,dtype=float);points=np.asarray(points,dtype=float)
    if not len(poses) or not len(points):return points,np.arange(len(points),dtype=int)
    if not np.isfinite(poses).all() or not np.isfinite(points).all():raise ValueError('Nonfinite spatial-crop input')
    # The detector uses dot products with the three pose axes. Bound the
    # inverse map, including slightly nonorthonormal saved FP32 rotations.
    singular=np.linalg.svd(poses[:,:3,:3],compute_uv=False)[:,-1]
    if (singular<.5).any():raise ValueError('Singular or invalid grasp frame')
    # Relative to contact: x in [-70,0]mm, y in ±56.5mm, z in ±10mm.
    # Includes original/refined fingers, palm, and the complete inner cavity.
    radius=float(np.sqrt(.070**2+.0565**2+.010**2)/singular.min()+1e-7)
    lower=poses[:,:3,3].min(0)-radius;upper=poses[:,:3,3].max(0)+radius
    keep=((points>=lower)&(points<=upper)).all(1)
    outside=np.flatnonzero(~keep)
    if len(outside):
        # Original ZeroGrasp sets outside-cavity contact Y to zero. Its first
        # such point can win an argmin tie for a one-sided object and affect
        # depth. Keep the earliest removed point, at its original list index.
        keep[outside[0]]=True
    ids=np.flatnonzero(keep)
    return points[ids],ids


def detect(poses,points):
    cropped,_=relevant_points(poses,points)
    return native_detect(poses,cropped)
