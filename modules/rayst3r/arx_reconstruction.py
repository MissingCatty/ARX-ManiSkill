"""Official RaySt3R backend shared by benchmarks and grasping."""
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from arx_project.layout import ROOT, module_dir, weights_dir, read_json as read, write_json as save, sha256 as sha
class RaySt3RAdapter:
    metric = True

    def __init__(self):
        import torch
        sys.path.insert(0,str(module_dir('rayst3r')))
        from eval_wrapper.eval import EvalWrapper,eval_scene
        self.scene_function=eval_scene
        self.model=EvalWrapper(str(weights_dir('rayst3r')/'rayst3r.pth'),distributed=False,device='cuda')
        self.dino=torch.hub.load(str(module_dir('dinov2')),'dinov2_vitl14_reg',source='local',pretrained=False)
        weights=torch.load(weights_dir('dinov2')/'dinov2_vitl14_reg4_pretrain.pth',map_location='cpu',weights_only=True)
        self.dino.load_state_dict(weights,strict=True);self.dino.eval().to('cuda')
        self.configuration={'n_pred_views':5,'set_conf':5,'filter_all_masks':False,'tsdf':False,
                            'dino_resident':True,'dino_model':'dinov2_vitl14_reg',
                            'native_depth_encoding':'uint16 / 65535 * 10 meters','precision':'official float32 wrapper'}

    def predict(self,rgb,depth,mask,k,work):
        import torch
        work.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rgb).save(work/'rgb.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(work/'mask.png')
        Image.fromarray(np.clip(np.rint(depth/10.*65535),0,65535).astype(np.uint16)).save(work/'depth.png')
        torch.save(torch.as_tensor(k,dtype=torch.float32),work/'intrinsics.pt')
        torch.save(torch.eye(4),work/'cam2world.pt')
        with torch.inference_mode():
            points=self.scene_function(self.model,str(work),visualize=False,n_pred_views=5,set_conf=5,
                                       dino_model=self.dino,do_filter_all_masks=False,tsdf=False).float().cpu().numpy()
        return {'points':points,'coordinate_frame':'camera_optical_m','metric':True}

