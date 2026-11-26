"""Native GraspNet NMS with original IDs, suppression groups and deferred rows."""
import time
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from arx_project.layout import ROOT


@lru_cache(maxsize=1)
def native_backend():
    spec=importlib.util.find_spec('grasp_nms')
    if spec is None:
        version=f'{sys.version_info.major}.{sys.version_info.minor}'
        directory=ROOT/f'environments/runtime_bundle/env/lib/python{version}/site-packages'
        paths=list(directory.glob('grasp_nms*.so'))
        if len(paths)!=1:raise ImportError('The frozen native GraspNet NMS extension is unavailable')
        spec=importlib.util.spec_from_file_location('grasp_nms',paths[0])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.modules['grasp_nms']=module
    return module.nms_grasp,Path(module.__file__)


def native_nms(poses,scores,widths,translation,rotation,source_indices=None):
    nms_grasp,_=native_backend()
    poses=np.asarray(poses);scores=np.asarray(scores);widths=np.asarray(widths)
    count=len(poses)
    if (poses.shape!=(count,4,4) or scores.shape!=(count,) or widths.shape!=(count,)
            or not np.isfinite(poses).all() or not np.isfinite(scores).all()
            or not np.isfinite(widths).all() or not 0<translation<=.03 or not 0<rotation<=np.pi/6+1e-10):
        raise ValueError('Invalid native NMS input or registered threshold')
    source=np.arange(count,dtype=int) if source_indices is None else np.asarray(source_indices,dtype=int)
    if source.shape!=(count,) or len(set(source.tolist()))!=count:raise ValueError('Duplicate or missing candidate IDs')
    if not count:return {'representatives':[],'deferred':[],'representative_for_input':[],
        'native_seconds':0.,'provenance_seconds':0.}
    rows=np.zeros((count,17),dtype=np.float64)
    rows[:,0]=scores;rows[:,1]=widths;rows[:,2]=.015
    rows[:,4:13]=poses[:,:3,:3].reshape(-1,9);rows[:,13:16]=poses[:,:3,3]
    # object_id is carried by the native implementation unchanged.
    rows[:,16]=np.arange(count)
    began=time.perf_counter();kept=nms_grasp(rows,translation,rotation)
    elapsed=time.perf_counter()-began
    local=kept[:,16].astype(int)
    if len(set(local.tolist()))!=len(local) or (local<0).any() or (local>=count).any():
        raise RuntimeError('Native NMS lost candidate source identity')
    local=np.sort(local);mapping=np.full(count,-1,dtype=int);mapping[local]=local
    began=time.perf_counter();tree=cKDTree(poses[local,:3,3])
    missing=np.flatnonzero(mapping<0)
    neighbors=tree.query_ball_point(poses[missing,:3,3],translation+1e-7,workers=1)
    for item,choices in zip(missing,neighbors):
        choices=np.sort(local[np.asarray(choices,dtype=int)])
        if len(choices):
            relative=poses[choices,:3,:3]@poses[item,:3,:3].T
            angle=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
            choices=choices[angle<=rotation+1e-6]
        if not len(choices):raise RuntimeError('Cannot trace a suppressed native NMS candidate')
        mapping[item]=choices[0]
    return {'representatives':source[local].tolist(),'deferred':source[missing].tolist(),
        'representative_for_input':source[mapping].tolist(),'native_seconds':elapsed,
        'provenance_seconds':time.perf_counter()-began}
