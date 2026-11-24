"""RGB-D sphere fit and an ARX pad-centre refinement, without a quality proxy.

This is a shape-conditioned engineering policy. It is neither VFAS's learned
evaluator nor GNP. The fitting function never reads the simulator object pose.
"""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PAD_CENTER_EEF = np.array([-.006, 0., .0375])


def fit_rgbd_sphere(folder):
    folder=Path(folder);cam=folder/'camera'
    params=json.loads((cam/'params.json').read_text());k=params['intrinsics']
    depth=np.asarray(Image.open(cam/'depth.png'),dtype=float)*params['depth_scale']
    mask=binary_erosion(np.asarray(Image.open(cam/'mask.png'))>0,iterations=2)&(depth>0)
    y,x=np.where(mask);z=depth[y,x]
    if len(z)<100:raise ValueError('Insufficient ARX RGB-D points for sphere fit')
    points=np.column_stack([(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z])
    world=np.asarray(json.loads((folder/'camera_world_optical.json').read_text()))
    points=points@world[:3,:3].T+world[:3,3]
    points=points[::max(1,len(points)//6000)]
    mean=points.mean(0);local=points-mean
    design=np.column_stack([2*local,np.ones(len(local))]);target=(local*local).sum(1)
    algebraic=np.linalg.lstsq(design,target,rcond=None)[0]
    center=algebraic[:3]+mean;radius=float(np.sqrt(max(1e-8,algebraic[3]+(algebraic[:3]**2).sum())))
    fitted=least_squares(lambda v:np.linalg.norm(points-v[:3],axis=1)-v[3],
                         np.r_[center,radius],loss='soft_l1',f_scale=.0005,max_nfev=100)
    center=fitted.x[:3];radius=float(fitted.x[3]);errors=np.abs(np.linalg.norm(points-center,axis=1)-radius)
    accepted=bool(fitted.success and .02<=radius<=.06 and np.quantile(errors,.95)<.0015)
    report={'method':'Robust sphere fit to eroded target RGB-D surface',
            'input_scope':'Existing segmentation and camera extrinsic; no simulator object pose or dimensions used by sphere fit',
            'center_world':center.tolist(),'radius_m':radius,'point_count':len(points),
            'median_abs_residual_m':float(np.median(errors)),'p95_abs_residual_m':float(np.quantile(errors,.95)),
            'accepted':accepted,'limits':{'radius_m':[.02,.06],'p95_abs_residual_m':.0015}}
    return report


def refine(selected, fit, height_m=.005):
    if not fit['accepted']:raise ValueError('RGB-D target does not support the preregistered sphere model')
    selected=copy.deepcopy(selected);old=np.asarray(selected['grasp_world'])
    theta=float(np.arctan2(old[2,1],old[2,0]));theta=(theta+np.pi/2)%np.pi-np.pi/2
    rot=old[:3,:3]@Rotation.from_rotvec([0,0,theta]).as_matrix()
    assert abs(rot[2,1])<1e-8, 'Closing axis must be horizontal'
    center=np.asarray(fit['center_world'])+np.array([0.,0.,height_m])
    grasp=np.eye(4);grasp[:3,:3]=rot;grasp[:3,3]=center-rot@PAD_CENTER_EEF
    pre=grasp.copy();pre[:3,3]-=.08*rot[:,2]
    lift=grasp.copy();lift[2,3]+=.12
    selected.update(grasp_world=grasp.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),target_contact_world=center.tolist())
    selected['research_change']={'hypothesis':'H007','method_scope':'RGB-D sphere-fit geometry refinement; no learned VFAS/GNP evaluator',
        'original_proposal_score_scope':'Network score refers only to the original unrefined proposal',
        'world_z_pad_offset_m':height_m,'closing_axis_z_before':float(old[2,1]),
        'closing_axis_z_after':float(rot[2,1]),'rotation_about_approach_rad':theta,
        'pad_center_eef_m':PAD_CENTER_EEF.tolist(),'pad_center_target_world':center.tolist(),
        'computed_pad_center_world':(grasp[:3,3]+rot@PAD_CENTER_EEF).tolist(),
        'sphere_fit':fit}
    return selected
