"""Unregistered CPU-only RGB-D shape study. No robot or simulator commands."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares


def fit(folder):
    folder=Path(folder);camera=folder/'camera'
    params=json.loads((camera/'params.json').read_text());k=params['intrinsics']
    depth=np.asarray(Image.open(camera/'depth.png'),dtype=float)*params['depth_scale']
    mask=binary_erosion(np.asarray(Image.open(camera/'mask.png'))>0,iterations=2)&(depth>0)
    y,x=np.where(mask);z=depth[y,x]
    if len(z)<100:return {'accepted':False,'reason':'Insufficient RGB-D'}
    pts=np.column_stack([(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z])
    world=np.asarray(json.loads((folder/'camera_world_optical.json').read_text()))
    pts=pts@world[:3,:3].T+world[:3,3];pts=pts[::max(1,len(pts)//1500)]
    span=np.quantile(pts,.99,axis=0)-np.quantile(pts,.01,axis=0)
    if span[2]>.055 or max(span[:2])/max(1e-6,min(span[:2]))>1.8:
        return {'accepted':False,'reason':'Outside same fixed horizontal near-round family','span_m':span.tolist()}
    origin=pts.mean(0);local=pts-origin;radial=np.linalg.norm(local[:,:2],axis=1)
    def details(v):
        cx,cy,cz,a,b,r,yaw=v;c,s=np.cos(yaw),np.sin(yaw)
        xy=(local[:,:2]-[cx,cy])@np.array([[c,-s],[s,c]])
        # Closest point on the ellipse (Newton iterations, no outcome input).
        phi=np.arctan2(a*xy[:,1],b*xy[:,0])
        for _ in range(10):
            cp,sp=np.cos(phi),np.sin(phi)
            f=(b*b-a*a)*sp*cp+a*xy[:,0]*sp-b*xy[:,1]*cp
            df=(b*b-a*a)*(cp*cp-sp*sp)+a*xy[:,0]*cp+b*xy[:,1]*sp
            phi-=np.clip(f/np.where(np.abs(df)<1e-10,1e-10,df),-.3,.3)
        distance=np.linalg.norm(xy-np.column_stack([a*np.cos(phi),b*np.sin(phi)]),axis=1)
        return np.sqrt(distance**2+(local[:,2]-cz)**2)-r,xy,phi
    best=None
    for minor in (.008,.012,.018):
        for yaw in (0.,np.pi/4,np.pi/2):
            major=np.clip(np.quantile(radial,.95)-minor,.013,.054)
            init=[0.,0.,-minor/2,major,major,minor,yaw]
            result=least_squares(lambda v:details(v)[0],init,
                bounds=([-.06,-.06,-.04,.012,.012,.004,-np.pi],[.06,.06,.02,.055,.055,.025,np.pi]),
                loss='soft_l1',f_scale=.0008,max_nfev=100)
            error=np.abs(details(result.x)[0]);key=float(np.median(error))
            if best is None or key<best[0]:best=(key,result,error)
    _,result,error=best;cx,cy,cz,a,b,r,yaw=result.x
    _,xy,phi=details(result.x);angles=np.sort(np.mod(phi,2*np.pi))
    coverage=1-float(np.diff(np.r_[angles,angles[0]+2*np.pi]).max())/(2*np.pi)
    # A conservative elliptical inner hole; retain the original 1% criterion.
    inner=float(np.mean((xy[:,0]/max(a-r-.0025,1e-6))**2+(xy[:,1]/max(b-r-.0025,1e-6))**2<1))
    p95=float(np.quantile(error,.95))
    accepted=bool(result.success and min(a,b)>1.2*r and max(a,b)/min(a,b)<1.8 and p95<.0025 and coverage>.8 and inner<.01)
    return {'accepted':accepted,'method':'Elliptical horizontal torus surface, unregistered offline study',
        'center_world':(origin+result.x[:3]).tolist(),'ellipse_radii_m':[float(a),float(b)],'minor_radius_m':float(r),
        'yaw_rad':float(yaw),'p95_residual_m':p95,'median_residual_m':float(np.median(error)),
        'angular_coverage':coverage,'points_in_predicted_hole_fraction':inner,'points':len(pts),
        'scope':'Only archived masked RGB-D/extrinsics. No target pose, label, dimensions, outcome or simulator queries. No execution evidence.'}
