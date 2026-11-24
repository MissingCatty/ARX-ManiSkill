"""UNREGISTERED offline RGB-D torus fit. Estimates shape; never authors a scene."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares


def fit(folder):
    folder=Path(folder);cam=folder/'camera';params=json.loads((cam/'params.json').read_text());k=params['intrinsics']
    depth=np.asarray(Image.open(cam/'depth.png'),dtype=float)*params['depth_scale'];mask=binary_erosion(np.asarray(Image.open(cam/'mask.png'))>0,iterations=2)&(depth>0)
    y,x=np.where(mask);z=depth[y,x]
    if len(z)<100:return {'accepted':False,'reason':'Insufficient RGB-D points'}
    pts=np.column_stack([(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z]);world=np.asarray(json.loads((folder/'camera_world_optical.json').read_text()));pts=pts@world[:3,:3].T+world[:3,3];pts=pts[::max(1,len(pts)//1500)]
    span=np.quantile(pts,.99,axis=0)-np.quantile(pts,.01,axis=0)
    if span[2]>.055 or max(span[:2])/max(1e-6,min(span[:2]))>1.8:return {'accepted':False,'reason':'Visible shape outside fixed prototype horizontal near-round family','span_m':span.tolist()}
    origin=pts.mean(0);local=pts-origin;radial=np.linalg.norm(local[:,:2],axis=1);best=None
    def residual(v):
        rho=np.sqrt((local[:,0]-v[0])**2+(local[:,1]-v[1])**2)
        return np.sqrt((rho-v[3])**2+(local[:,2]-v[2])**2)-v[4]
    for minor in (.008,.012,.018):
        for zoff in (-minor/2,-minor):
            init=np.array([0.,0.,zoff,np.clip(np.quantile(radial,.95)-minor,.013,.054),minor])
            result=least_squares(residual,init,bounds=([-.06,-.06,-.04,.012,.004],[.06,.06,.02,.055,.025]),loss='soft_l1',f_scale=.0008,max_nfev=80)
            error=np.abs(residual(result.x));key=float(np.median(error))
            if best is None or key<best[0]:best=(key,result,error)
    _,result,error=best;cx,cy,cz,major,minor=result.x;center=origin+result.x[:3];rho=np.linalg.norm(pts[:,:2]-center[:2],axis=1);angles=np.sort(np.mod(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]),2*np.pi));coverage=1-float(np.diff(np.r_[angles,angles[0]+2*np.pi]).max())/(2*np.pi);inside=float(np.mean(rho<major-minor-.0025));p95=float(np.quantile(error,.95))
    accepted=bool(result.success and major>1.2*minor and p95<.0025 and coverage>.8 and inside<.01)
    return {'accepted':accepted,'method':'Robust horizontal torus surface fit, offline prototype','center_world':center.tolist(),'major_radius_m':float(major),'minor_radius_m':float(minor),'p95_residual_m':p95,'median_residual_m':float(np.median(error)),'angular_coverage':coverage,'points_in_predicted_hole_fraction':inside,'points':len(pts),'scope':'Only existing maskedRGB-D/extrinsics; no simulator object pose/dimensions or outcome used. Not execution/collision/contact-quality validation.'}
