"""Small CPU batch kernel; build products belong under tmp, never modules."""
import ctypes
import fcntl
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import cKDTree

from arx_project.layout import ROOT


@lru_cache(maxsize=1)
def backend():
    source=Path(__file__).with_suffix('.cpp')
    flags=['-std=c++17','-O3','-shared','-fPIC','-ffp-contract=off']
    compiler=subprocess.run(['c++','--version'],check=True,capture_output=True,text=True).stdout
    key=hashlib.sha256(source.read_bytes()+compiler.encode()+repr(flags).encode()).hexdigest()
    directory=ROOT/'tmp/graspgenx_postprocessing/native_kernel'/key
    directory.mkdir(parents=True,exist_ok=True);binary=directory/'libgeometry.so'
    with (directory/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not binary.exists():
            output=directory/'libgeometry.build.so'
            subprocess.run(['c++',*flags,str(source),'-o',str(output)],check=True,capture_output=True)
            output.replace(binary)
    library=ctypes.CDLL(str(binary));function=library.arx_point_rejections
    pointer=ctypes.c_void_p;integer=ctypes.c_int64
    function.argtypes=[integer,pointer,pointer,pointer,pointer,integer,pointer,pointer,
        pointer,ctypes.c_double,ctypes.c_double,pointer,pointer]
    function.restype=ctypes.c_int
    cavity=library.arx_cavity_counts
    cavity.argtypes=[integer,pointer,integer,pointer,pointer,pointer,pointer]
    cavity.restype=ctypes.c_int
    return function,binary,library


def point_rejections(model,eef,points,margin,pre_distance,skip):
    poses=np.ascontiguousarray(eef,dtype=np.float64)
    points=np.ascontiguousarray(points,dtype=np.float64)
    if poses.shape!=(len(poses),4,4) or points.shape!=(len(points),3):
        raise ValueError('Invalid batch geometry shape')
    if not np.isfinite(poses).all() or not np.isfinite(points).all():
        raise ValueError('Nonfinite batch geometry')
    if skip is None:skip=np.zeros(len(poses),dtype=bool)
    skip=np.asarray(skip,dtype=bool)
    if skip.shape!=(len(poses),):raise ValueError('Invalid geometry skip mask')
    centers=np.einsum('kij,j->ki',poses[:,:3,:3],model.center)+poses[:,:3,3]-pre_distance/2*poses[:,:3,2]
    active=np.flatnonzero(~skip);neighbors=[[] for _ in poses]
    groups=cKDTree(points).query_ball_point(centers[active],model.radius+pre_distance/2,workers=1)
    for i,group in zip(active,groups):neighbors[i]=group
    offsets=np.r_[0,np.cumsum([len(ids) for ids in neighbors])].astype(np.int64)
    ids=np.ascontiguousarray([item for group in neighbors for item in group],dtype=np.int64)
    bounds=np.ascontiguousarray([np.r_[p.lower,p.upper] for p in model.parts],dtype=np.float64)
    equations=[p.equations for p in model.parts if p.equations is not None]
    planes=np.ascontiguousarray(np.concatenate(equations) if equations else np.empty((0,4)),dtype=np.float64)
    plane_offsets=np.r_[0,np.cumsum([0 if p.equations is None else len(p.equations) for p in model.parts])].astype(np.int64)
    rejected=np.zeros(len(poses),dtype=np.uint8);counts=np.zeros(len(poses),dtype=np.int64)
    function,_,_=backend()
    status=function(len(poses),poses.ctypes.data,points.ctypes.data,offsets.ctypes.data,ids.ctypes.data,
        len(model.parts),bounds.ctypes.data,planes.ctypes.data,plane_offsets.ctypes.data,
        margin,pre_distance,rejected.ctypes.data,counts.ctypes.data)
    if status:raise RuntimeError('CPU convex geometry kernel failed')
    return rejected.astype(bool),counts


def cavity_counts(model,eef,points):
    poses=np.ascontiguousarray(eef,dtype=np.float64);points=np.ascontiguousarray(points,dtype=np.float64)
    if poses.shape!=(len(poses),4,4) or points.shape!=(len(points),3):
        raise ValueError('Invalid cavity geometry shape')
    if not np.isfinite(poses).all() or not np.isfinite(points).all():
        raise ValueError('Nonfinite cavity geometry')
    lower,upper=[np.ascontiguousarray(a,dtype=np.float64) for a in model.cavity]
    counts=np.zeros(len(poses),dtype=np.int64);_,_,library=backend()
    status=library.arx_cavity_counts(len(poses),poses.ctypes.data,len(points),points.ctypes.data,
        lower.ctypes.data,upper.ctypes.data,counts.ctypes.data)
    if status:raise RuntimeError('CPU cavity geometry kernel failed')
    return counts
