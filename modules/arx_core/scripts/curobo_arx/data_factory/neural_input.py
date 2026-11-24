"""Calibrated ARX point-cloud preparation shared by neural grasp backends."""
import numpy as np
from .geometry import rigid


def cloud_input(depth, row, asset):
    depth=np.asarray(depth,dtype=float);k=np.asarray(row['intrinsics'],dtype=float)
    if depth.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('Invalid ARX head RGB-D calibration')
    yy,xx=np.indices(depth.shape)
    points=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],axis=-1)
    valid=np.isfinite(points).all(axis=2)&(depth>0)
    camera=rigid(row['camera_world_optical']);obj=rigid(row['snapshot']['objects']['test_object']['world'])
    local=points@(np.linalg.inv(obj)@camera)[:3,:3].T+(np.linalg.inv(obj)@camera)[:3,3]
    size=np.asarray(asset['size_m'],dtype=float)
    # Keep surrounding context for neural point-cloud collision filtering. The
    # complete original scene is separately retained by the ARX planner.
    context=valid&(np.abs(local)<=size/2+.25).all(axis=2)
    target=(np.abs(local)<=size/2+.004).all(axis=2)&(local[:,:,2]>-size[2]/2+.004)
    points=points[context].astype(np.float32);region=target[context].astype(bool)
    if region.sum()<50:raise ValueError('Insufficient fresh rendered target depth; do not disable region steering')
    return points,region
