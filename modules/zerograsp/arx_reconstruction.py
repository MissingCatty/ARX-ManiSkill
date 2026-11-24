"""ZeroGrasp reconstruction head, separate from its grasp decoding."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from arx_project.layout import module_dir,weights_dir
class ZeroGraspAdapter:
    metric = True

    def __init__(self):
        sys.path.insert(0,str(module_dir('zerograsp')/'server/vendor/zerograsp_mainline_minimal'))
        import demo
        from zerograsp.utils.config import parse_config
        package=module_dir('zerograsp')/'server/vendor/zerograsp_mainline_minimal'
        self.config=parse_config(str(package/'configs/demo_640x480.yaml'))
        self.config.update_octree=True
        self.checkpoint=weights_dir('zerograsp')/'mainline/zerograsp_synthetic.ckpt'
        self.model=demo.BaseTrainer.load_from_checkpoint(str(self.checkpoint),config=self.config,strict=False,map_location='cuda')
        self.model.to('cuda').eval()
        self.configuration={'input_depth_internal_unit':'millimeters, float32','input_depth_conversion':'depth_m * 1000',
                            'scope':'shared network forward and reconstruction decode only',
                            'grasp_decode':False,'grasp_postprocessing':False,'checkpoint':str(self.checkpoint)}

    def predict(self,rgb,depth,mask,k,work):
        import torch
        import torch.nn.functional as F
        from zerograsp.utils.dataset import make_batch_from_arrays
        from zerograsp.utils.math import unnormalize_pts
        from zerograsp.nets.utils import get_xyz_from_octree
        batch=make_batch_from_arrays(rgb,depth*1000.,mask.astype(np.int32),k,self.config,depth_scale=1.,device='cuda')
        with torch.inference_mode():
            output=self.model.model(batch)
            octree=output['octrees_out']
            points,_=get_xyz_from_octree(octree,self.config.max_lod,nempty=True,return_batch=True)
            points=unnormalize_pts(points,batch[-2][0],self.config.grid_size,1<<self.config.min_lod)
            normals=F.normalize(octree.normals[self.config.max_lod],dim=-1)
            sdf=octree.features[self.config.max_lod][:,:1]
            points=(points.reshape(-1,3)-normals.reshape(-1,3)*sdf.reshape(-1,1)).float().cpu().numpy()/1000.
        return {'points':points,'coordinate_frame':'camera_optical_m','metric':True}


