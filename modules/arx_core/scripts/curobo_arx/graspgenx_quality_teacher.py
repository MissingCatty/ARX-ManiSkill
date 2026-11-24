"""CPU pretrained discriminator adapter; never imports simulator/control APIs."""
from pathlib import Path
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent / 'research_dependencies/GraspGenX'
PREP = ROOT / 'runs/graspgenx_feasibility_v2_20261021'
CHECKPOINT = ROOT.parent / 'research_dependencies/graspgenx_checkpoints/release/dis'


class QualityTeacher:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES']=''
        os.environ['GRASPGENX_CHECKPOINT_DIR']=str(CHECKPOINT.parents[1])
        os.environ['GRASPGENX_GRIPPER_CFG_DIR']=str(PREP)
        sys.path.insert(0,str(REPO))
        import torch
        from omegaconf import OmegaConf
        from graspgenx.models.discriminator import GraspGenDiscriminator
        assert not torch.cuda.is_available()
        torch.set_num_threads(4);torch.manual_seed(0)
        cfg=OmegaConf.load(CHECKPOINT/'config.yaml')
        cfg.discriminator.ptv3vanilla.enable_flash=False
        self.model=GraspGenDiscriminator.from_config(cfg.discriminator).cpu()
        self.model.load_state_dict(torch.load(CHECKPOINT/'epoch_1056.pth',map_location='cpu',weights_only=True)['model'],strict=True)
        self.model.eval();self.torch=torch;self.calls=0
        d=json.loads((PREP/'frozen_arx_sweep_descriptor.json').read_text())
        self.basis=np.asarray(d['canonical_to_arx_eef']);desc=d['model_input']
        self.sweep=torch.tensor([desc[k] for k in ('extents_open','offset_open','extents_mid','offset_mid')],dtype=torch.float32).reshape(1,12)

    def score(self, points, worlds):
        assert self.calls<100,'Registered per-process CPU query cap'
        torch=self.torch
        rng=np.random.default_rng(0)
        pc=points[rng.integers(0,len(points),size=3500)]
        center=pc.mean(0);pc=pc-center
        poses=np.stack([np.asarray(w)@self.basis for w in worlds]);poses[:,:3,3]-=center
        data={'points':torch.tensor(pc,dtype=torch.float32)[None],
              'grasps':torch.tensor(poses,dtype=torch.float32)[None],
              'sweep_volume_open_and_mid':self.sweep}
        with torch.inference_mode():output,_,_=self.model(data)
        self.calls+=1
        value=output['grasp_confidence'].reshape(-1).numpy()
        assert np.isfinite(value).all()
        return value


def predict_adapter(features, model):
    features=np.asarray(features,dtype=float)
    x=(features-np.asarray(model['feature_mean']))/np.asarray(model['feature_scale'])
    values=[]
    for m in model['ensemble']:
        pa=1/(1+np.exp(-np.clip(x@np.asarray(m['approach_coef'])+m['approach_intercept'],-40,40)))
        ph=1/(1+np.exp(-np.clip(x@np.asarray(m['hold_coef'])+m['hold_intercept'],-40,40)))
        values.append(pa*ph)
    values=np.asarray(values)
    return {'mean':values.mean(axis=0),'std':values.std(axis=0),'ranking':values.mean(axis=0)-values.std(axis=0)}
