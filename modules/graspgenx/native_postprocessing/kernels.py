"""Build a small deterministic CPU kernel in the project's runtime directory."""
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
    source=Path(__file__).with_suffix('.cpp').with_name('kernel.cpp')
    flags=['-std=c++17','-O3','-shared','-fPIC','-ffp-contract=off']
    compiler=subprocess.run(['c++','--version'],check=True,capture_output=True).stdout
    digest=hashlib.sha256(source.read_bytes()+compiler+repr(flags).encode()).hexdigest()
    directory=ROOT/'tmp/graspgenx_native_postprocessing/kernel'/digest
    directory.mkdir(parents=True,exist_ok=True);binary=directory/'kernel.so'
    with (directory/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not binary.exists():
            out=binary.with_suffix('.building.so')
            subprocess.run(['c++',*flags,str(source),'-o',str(out)],check=True,capture_output=True)
            out.replace(binary)
    lib=ctypes.CDLL(str(binary));p=ctypes.c_void_p;i=ctypes.c_int64
    lib.zg_detect.argtypes=[i,p,i,p,p,p,p,p,p];lib.zg_detect.restype=ctypes.c_int
    lib.arx_variable_collision.argtypes=[i,p,p,p,p,p,i,p,p,p,p,ctypes.c_double,p]
    lib.arx_variable_collision.restype=ctypes.c_int
    return lib,binary


def array(values):return np.ascontiguousarray(values,dtype=np.float64)


def native_detect(poses,points):
    poses,points=array(poses),array(points);n=len(poses)
    if poses.shape!=(n,4,4) or points.shape!=(len(points),3) or not np.isfinite(points).all() or not np.isfinite(poses).all():
        raise ValueError('Invalid native postprocessing geometry')
    bad=np.zeros(n,np.uint8);final=np.zeros(n,np.uint8);counts=np.zeros(n,np.int64)
    widths=np.zeros(n);depths=np.zeros(n)
    if not len(points):return {'bad':np.ones(n,bool),'final_bad':np.ones(n,bool),'counts':counts,'widths':widths,'depths':depths}
    status=backend()[0].zg_detect(n,poses.ctypes.data,len(points),points.ctypes.data,
        bad.ctypes.data,final.ctypes.data,counts.ctypes.data,widths.ctypes.data,depths.ctypes.data)
    if status:raise RuntimeError('Native collision kernel failed')
    return {'bad':bad.astype(bool),'final_bad':final.astype(bool),'counts':counts,'widths':widths,'depths':depths}


def variable_collision(model,eef,widths,points):
    eef,widths,points=array(eef),array(widths),array(points)
    if widths.shape!=(len(eef),) or (widths<0).any() or (widths>.088+1e-12).any():
        raise ValueError('The refined opening exceeds the frozen ARX stroke')
    centers=np.einsum('kij,j->ki',eef[:,:3,:3],model.center)+eef[:,:3,3]-.04*eef[:,:3,2]
    groups=cKDTree(points).query_ball_point(centers,model.radius+.084,workers=1)
    offsets=np.r_[0,np.cumsum([len(g) for g in groups])].astype(np.int64)
    ids=np.asarray([j for g in groups for j in g],dtype=np.int64)
    bounds=array([np.r_[p.lower,p.upper] for p in model.parts])
    direction=array([0 if not p.finger else np.sign(p.vertices[:,1].mean()) for p in model.parts])
    poff=np.r_[0,np.cumsum([0 if p.equations is None else len(p.equations) for p in model.parts])].astype(np.int64)
    planes=array(np.concatenate([p.equations for p in model.parts if p.equations is not None]))
    bad=np.zeros(len(eef),np.uint8)
    status=backend()[0].arx_variable_collision(len(eef),eef.ctypes.data,widths.ctypes.data,
        points.ctypes.data,offsets.ctypes.data,ids.ctypes.data,len(model.parts),bounds.ctypes.data,
        planes.ctypes.data,poff.ctypes.data,direction.ctypes.data,.001,bad.ctypes.data)
    if status:raise RuntimeError('Variable-width authored geometry kernel failed')
    return bad.astype(bool)
