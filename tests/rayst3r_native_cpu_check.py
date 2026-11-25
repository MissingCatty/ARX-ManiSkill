"""Exercise native encoders, heads and scene geometry without allocating CUDA."""
from copy import deepcopy
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'modules/rayst3r'))
from models.rayquery import RayQuery, RayEncoder, PointmapEncoder
from utils.geometry import compute_rays
from engine import eval_model
from modules.rayst3r.optimized.inference import prediction_only, SceneInference

torch.set_num_threads(1)
torch.manual_seed(7)


class Criterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, prediction, truth):
        self.calls += 1
        return {'loss': torch.tensor(0.)}


def model_with_dino(enabled):
    return RayQuery(
        ray_enc=RayEncoder(dim=16, depth=1, num_heads=2, img_size=(32, 32)),
        pointmap_enc=PointmapEncoder(dim=16, depth=1, num_heads=2, img_size=(32, 32)),
        decoder_dim=16, decoder_depth=1, decoder_num_heads=2, imshape=(32, 32),
        pts_head_type='linear', classifier_head_type='linear_classifier',
        criterion=Criterion(), dino_layers=[23] if enabled else []).eval()


for enabled in [False, True]:
    network = model_with_dino(enabled)
    batch = {'input_cams': {
        'pointmaps': torch.randn(1, 32, 32, 3),
        'valid_masks': torch.ones(1, 32, 32, dtype=torch.bool),
        'dino_features': torch.randn(1, 4, 1024)},
        'new_cams': {'pointmaps': torch.zeros(1, 32, 32, 3),
            'Ks': torch.tensor([[[20., 0., 16.], [0., 20., 16.], [0., 0., 1.]]])}}
    with torch.inference_mode():
        original, _, _ = network(deepcopy(batch), mode='viz')
        optimized, _ = prediction_only(network, deepcopy(batch), compute_rays)
    for key in original:
        torch.testing.assert_close(original[key], optimized[key], rtol=0, atol=0)
    assert network.criterion.calls == 1


class Dino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_intermediate_layers(self, rgb, layers):
        self.calls += 1
        # Deterministic image-dependent features, retaining the native shape.
        return [rgb.mean((1, 2, 3))[:, None, None].expand(-1, 4, 1024) for _ in layers]


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, dino_model):
        return eval_model(self.model, batch, mode='viz', device='cpu',
                          dino_model=dino_model, return_scale=True)


rgb = torch.randint(0, 255, (1, 1, 32, 32, 3), dtype=torch.uint8)
mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
mask[..., 0, 0] = False
k = torch.tensor([[[[20., 0., 16.], [0., 20., 16.], [0., 0., 1.]]]])


def scene_batch(views):
    return {
        'input_cams': {'imgs': rgb.clone(), 'valid_masks': mask.clone(),
            'depths': torch.full((1, 1, 32, 32), 2000.),
            'Ks': k.clone(), 'c2ws': torch.eye(4)[None, None].clone()},
        'new_cams': {'depths': torch.zeros(1, views, 32, 32),
            'valid_masks': torch.ones(1, views, 32, 32, dtype=torch.bool),
            'Ks': k.repeat(1, views, 1, 1),
            'c2ws': torch.eye(4)[None, None].repeat(1, views, 1, 1)}}


native = Wrapper(model_with_dino(True))
dino = Dino().eval()
with torch.inference_mode():
    reference = [native(scene_batch(views), dino) for views in [1, 3]]
assert dino.calls == 2
for reuse, skip in [(True, False), (False, True), (True, True)]:
    wrapper = SceneInference(native, reuse_dino=reuse, skip_metrics=skip)
    dino.calls = 0
    native.model.criterion.calls = 0
    with torch.inference_mode(), wrapper.scene() as stats:
        actual = [wrapper(scene_batch(views), dino) for views in [1, 3]]
    for old, new in zip(reference, actual):
        for key in old[0]:
            torch.testing.assert_close(old[0][key], new[0][key], rtol=0, atol=0)
        assert old[3] == new[3]
    assert dino.calls == (1 if reuse else 2)
    assert native.model.criterion.calls == (0 if skip else 2)
    assert stats['metric_calls'] == (0 if skip else 2)
print(json.dumps({'native_network_parity': 2, 'native_two_stage_parity': 3,
                  'coordinate_bitwise_equal': True, 'cuda_allocations': 0}))
