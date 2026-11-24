"""Observed RGB-D/frozen ARX CAD features for an additional learned selector only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
PROXY = ROOT / 'runs/research_contract_20261019/fixed_gripper_quality_proxy_2mm.npz'
NAMES = ['palm_min_distance', 'finger_min_distance', 'goal_near_fraction', 'sweep_max_near_fraction',
         'near_fraction_proximal', 'near_fraction_distal', 'pad_roi_fraction', 'pad_roi_width',
         'pad_roi_center_y', 'pad_roi_center_z', 'pad_roi_span_x', 'pad_roi_span_z',
         'cloud_median_x', 'cloud_median_y', 'cloud_median_z', 'cloud_span_x', 'cloud_span_y', 'cloud_span_z',
         'normal_alignment_median', 'normal_alignment_q25', 'left_support_fraction', 'right_support_fraction',
         'neural_score_prior', 'neural_width', 'neural_depth', 'pretrained_quality_logit']


def observed(folder, max_points=1500):
    folder = Path(folder)
    params = json.loads((folder / 'camera/params.json').read_text()); k=params['intrinsics']
    depth=np.asarray(Image.open(folder/'camera/depth.png'),dtype=float)*params['depth_scale']
    mask=binary_erosion(np.asarray(Image.open(folder/'camera/mask.png'))>0,iterations=2)&(depth>0)
    y,x=np.where(mask);z=depth[y,x]
    cloud=np.c_[(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z]
    camera=np.asarray(json.loads((folder/'camera_world_optical.json').read_text()))
    cloud=cloud@camera[:3,:3].T+camera[:3,3]
    assert len(cloud)>100
    if max_points and len(cloud)>max_points:
        ids=np.random.default_rng(0).choice(len(cloud),max_points,replace=False);cloud=cloud[ids]
    return cloud


class QualityFeatures:
    def __init__(self, folder):
        self.points=observed(folder)
        self.meta=json.loads(PROXY.with_suffix('.json').read_text())
        assert hashlib.sha256(PROXY.read_bytes()).hexdigest()==self.meta['data_sha256']
        data=np.load(PROXY)
        self.palm=cKDTree(data['right_link26'])
        self.fingers={link:cKDTree(data[link]) for link in ('right_link27','right_link28')}
        distance,near=cKDTree(self.points).query(self.points,k=16,workers=1)
        neighbours=self.points[near];c=neighbours-neighbours.mean(1)[:,None]
        _,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',c,c)/16)
        self.normals=vectors[:,:,0];self.reliable=distance[:,-1]<=.015

    def evaluate(self, pose, selected, pretrained_score):
        pose=np.asarray(pose);p=(self.points-pose[:3,3])@pose[:3,:3]
        nominal_open=.044
        goal=np.zeros(len(p),dtype=bool);sweep=0.;minimum=1.
        for link,tree in self.fingers.items():
            shift=np.asarray(self.meta['motion_axis_by_link'][link])*nominal_open
            for step in np.linspace(.08,0,9):
                d=tree.query(p+np.array([0,0,step])-shift,workers=1)[0]
                mask=d<=self.meta['sample_cover_radius_m']
                sweep=max(sweep,float(mask.mean()));minimum=min(minimum,float(d.min()))
                if step==0:goal|=mask
        palm=float(self.palm.query(p,workers=1)[0].min())
        # Frozen pad-centre ROI; an input feature, not an altered collision margin.
        roi=(p[:,0]>=-.0165)&(p[:,0]<=.0045)&(p[:,2]>=.02997)&(p[:,2]<=.04497)&(np.abs(p[:,1])<=.046)
        r=p[roi]
        if len(r)>=4:
            lo,hi=np.quantile(r,[.1,.9],axis=0);median=np.median(r,axis=0)
        else:lo=hi=median=np.zeros(3)
        normal=(self.normals@pose[:3,:3])[roi&self.reliable,1]
        align=np.abs(normal)
        qlo,qhi=np.quantile(p,[.1,.9],axis=0);qm=np.median(p,axis=0)
        v=[palm,minimum,float(goal.mean()),sweep,
           float((goal&(p[:,2]<.025)).mean()),float((goal&(p[:,2]>=.025)).mean()),
           float(roi.mean()),float(hi[1]-lo[1]),float(median[1]),float(median[2]-.0375),float(hi[0]-lo[0]),float(hi[2]-lo[2]),
           *qm.tolist(),*(qhi-qlo).tolist(),float(np.median(align)) if len(align) else 0.,
           float(np.quantile(align,.25)) if len(align) else 0.,
           float((roi&(p[:,1]<0)).mean()),float((roi&(p[:,1]>=0)).mean()),
           float(selected['score']),float(selected['width_m']),float(selected['depth_m']),
           float(np.log(np.clip(pretrained_score,1e-7,1-1e-7)/np.clip(1-pretrained_score,1e-7,1-1e-7)))]
        assert len(v)==len(NAMES) and np.isfinite(v).all()
        return np.asarray(v,dtype=float)
