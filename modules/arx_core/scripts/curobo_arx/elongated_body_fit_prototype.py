"""UNREGISTERED CPU-only broad elongated-body proposal fit. No model/sim/motion queries."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares


def fit(folder):
    folder=Path(folder);cam=folder/'camera';params=json.loads((cam/'params.json').read_text());k=params['intrinsics']
    depth=np.asarray(Image.open(cam/'depth.png'),dtype=float)*params['depth_scale']
    mask=binary_erosion(np.asarray(Image.open(cam/'mask.png'))>0,iterations=2)&(depth>0)
    y,x=np.where(mask);z=depth[y,x]
    if len(z)<150:return {'accepted':False,'reason':'Insufficient RGB-D'}
    pts=np.column_stack([(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z])
    world=np.asarray(json.loads((folder/'camera_world_optical.json').read_text()));pts=pts@world[:3,:3].T+world[:3,3]
    pts=pts[::max(1,len(pts)//4000)];mean=pts.mean(0)
    _,vectors=np.linalg.eigh(np.cov((pts-mean).T));major=vectors[:,-1]
    if abs(major[2])>.35:return {'accepted':False,'reason':'Major axis not horizontal'}
    axis=np.r_[major[:2]/np.linalg.norm(major[:2]),0.];radial=np.cross(axis,[0,0,1.])
    along=(pts-mean)@axis;lo,hi=np.quantile(along,[.01,.99]);length=float(hi-lo)
    if not .07<=length<=.25:return {'accepted':False,'reason':'Outside fixed length family','length_m':length}
    mid=(lo+hi)/2;selection=np.abs(along-mid)<=length*.25
    if selection.sum()<100:return {'accepted':False,'reason':'Insufficient central-body points'}
    cross=np.column_stack([(pts[selection]-mean)@radial,pts[selection,2]-mean[2]])
    design=np.c_[2*cross,np.ones(len(cross))];sol=np.linalg.lstsq(design,(cross*cross).sum(1),rcond=None)[0]
    radius=np.sqrt(max(1e-8,sol[2]+(sol[:2]**2).sum()))
    result=least_squares(lambda v:np.linalg.norm(cross-v[:2],axis=1)-v[2],np.r_[sol[:2],radius],loss='soft_l1',f_scale=.0005,max_nfev=100)
    cx,cz,r=result.x;residual=np.abs(np.linalg.norm(cross-result.x[:2],axis=1)-r)
    angles=np.sort(np.mod(np.arctan2(cross[:,1]-cz,cross[:,0]-cx),2*np.pi));coverage=1-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()/(2*np.pi)
    p95=float(np.quantile(residual,.95));center=mean+axis*mid+radial*cx+[0,0,cz]
    accepted=bool(result.success and .008<=r<=.045 and length/(2*r)>=1.8 and p95<.006 and coverage>=.20)
    return {'accepted':accepted,'method':'Robust central-body circle fit about observed horizontal major axis',
        'center_world':center.tolist(),'axis_world':axis.tolist(),'closing_axis_world':radial.tolist(),
        'radius_m':float(r),'length_m':length,'p95_residual_m':p95,'observed_cross_section_coverage':float(coverage),
        'central_points':int(selection.sum()),'scope':'Only existing masked RGB-D and fixed extrinsics. No label, object dimensions/pose, outcome or simulator queries. Not grasp/IK/contact validation.'}
