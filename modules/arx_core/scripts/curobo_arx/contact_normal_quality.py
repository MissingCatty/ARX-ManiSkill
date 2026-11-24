"""UNREGISTERED observed visible-surface quality, not force-closure certification.

Geometric inspiration: GPG antipodal normal alignment. This is a distinct ARX
single-visible-side heuristic; upstream geometric/friction parameters NOT used.
Robot models, physics, calibration and actual success/safety checks unchanged.
"""
import json
import numpy as np
from scipy.spatial import cKDTree
from palm_target_quality import PalmTargetQuality,PROXY


class ObservedContactQuality:
    def __init__(self,folder):
        self.palm=PalmTargetQuality(folder)
        points=self.palm.points
        _,index=np.unique(np.round(points/.001).astype(int),axis=0,return_index=True)
        self.points=points[np.sort(index)]
        distance,near=cKDTree(self.points).query(self.points,k=24,workers=1)
        neighbours=self.points[near];centred=neighbours-neighbours.mean(1)[:,None,:]
        covariance=np.einsum('nki,nkj->nij',centred,centred)/24
        values,vectors=np.linalg.eigh(covariance);self.normals=vectors[:,:,0]
        # Absolute alignment is invariant to the unobservable normal sign.
        self.reliable=distance[:,-1]<=.010
        with np.load(PROXY) as data:
            tip=data['right_link27'][data['right_link27'][:,2]>=.025]
            self.tip_min=tip.min(0);self.tip_max=tip.max(0)

    def evaluate_normals(self,world):
        world=np.asarray(world);local=(self.points-world[:3,3])@world[:3,:3]
        normals=self.normals@world[:3,:3]
        keep=self.reliable&(local[:,0]>=self.tip_min[0])&(local[:,0]<=self.tip_max[0])&(local[:,2]>=self.tip_min[2])&(local[:,2]<=self.tip_max[2])&(np.abs(local[:,1])<=.044)
        number=int(keep.sum());dot=np.abs(normals[keep,1])
        median=float(np.median(dot)) if number else 0.;q25=float(np.quantile(dot,.25)) if number else 0.
        return {'pass':number>=100 and median>=.9 and q25>=.85,'visible_tip_ROI_points':number,'median_abs_surface_normal_dot_closing':median,'q25_abs_surface_normal_dot_closing':q25,'required_points':100,'required_median':.9,'required_q25':.85,
                'scope':'Additional visible-side pose quality only; single RGBD cannot certify hidden opposing contacts or force closure. Fixed actual-CAD tip ROI/1mm point voxel/24nearest normal PCA; no physical friction/geometry/model edits.'}
