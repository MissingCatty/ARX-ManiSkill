"""UNREGISTERED extra RGB-D/frozen-CAD proposal quality, not a robot model.

The original cuRobo and PhysX geometry, filtering and margins are untouched.
Observed points cannot certify clearance from hidden object surfaces.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

PROXY=Path(__file__).resolve().parents[2]/'runs/research_contract_20261019/fixed_gripper_quality_proxy_2mm.npz'
REQUIRED_OBSERVED_PALM_GAP_M=.010


class PalmTargetQuality:
    def __init__(self,folder):
        folder=Path(folder);read=lambda n:json.loads((folder/n).read_text())
        self.meta=json.loads(PROXY.with_suffix('.json').read_text())
        assert hashlib.sha256(PROXY.read_bytes()).hexdigest()==self.meta['data_sha256']
        with np.load(PROXY) as data:self.tree=cKDTree(data['right_link26'].copy())
        params=read('camera/params.json');k=params['intrinsics']
        depth=np.asarray(Image.open(folder/'camera/depth.png'),dtype=float)*params['depth_scale']
        mask=binary_erosion(np.asarray(Image.open(folder/'camera/mask.png'))>0,iterations=2)&(depth>0)
        y,x=np.where(mask);z=depth[y,x]
        if len(z)<150:raise ValueError('Insufficient original RGB-D for additional palm quality')
        points=np.c_[(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z]
        camera=np.asarray(read('camera_world_optical.json'))
        self.points=points@camera[:3,:3].T+camera[:3,3]

    def evaluate(self,world):
        world=np.asarray(world)
        local=(self.points-world[:3,3])@world[:3,:3]
        distance=float(self.tree.query(local,workers=1)[0].min())
        estimate=distance-self.meta['sample_cover_radius_m']
        return {'pass':estimate>=REQUIRED_OBSERVED_PALM_GAP_M,
                'observed_palm_gap_lower_estimate_m':estimate,
                'required_observed_palm_gap_m':REQUIRED_OBSERVED_PALM_GAP_M,
                'proxy_sample_cover_radius_m':self.meta['sample_cover_radius_m'],
                'proxy_data_sha256':self.meta['data_sha256'],'observed_points':len(self.points),
                'scope':'Additional pose-quality heuristic from original RGB-D/fixed palm CAD volume only. Not a cuRobo/PhysX margin/filter/geometry edit, not hidden-surface clearance proof or an executed contact guarantee.'}
