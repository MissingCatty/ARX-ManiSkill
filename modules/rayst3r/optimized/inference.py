"""Prediction-only execution and DINO reuse within one native two-stage scene.

The native camera loader, normalization, network weights, heads, postprocessing,
view sampling and precision are retained. No native module is monkey-patched.
"""
from contextlib import contextmanager, nullcontext
from threading import Lock
from types import SimpleNamespace

import torch


class SceneFeatureCache:
    """One scene only; compare image/mask contents before reusing features."""

    def __init__(self):
        self.active = False
        self.entry = None
        self.stats = {}
        self._lock = Lock()

    @contextmanager
    def scene(self, stats):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Overlapping scene inference is not supported')
        self.active, self.entry, self.stats = True, None, stats
        try:
            yield
        finally:
            self.entry, self.active = None, False
            self._lock.release()

    def get(self, rgb, mask, model, layers, compute):
        if not self.active:
            raise RuntimeError('DINO reuse requires an active scene')
        if getattr(model, 'training', False):
            raise ValueError('DINO must remain in evaluation mode')
        key = (id(model), tuple(layers), rgb.dtype, mask.dtype,
               rgb.device, mask.device, tuple(rgb.shape), tuple(mask.shape))
        old = self.entry
        if old is not None and old[0] == key and torch.equal(old[1], rgb) and torch.equal(old[2], mask):
            self.stats['feature_cache_hits'] += 1
            return old[3]
        self.stats['dino_forwards'] += 1
        features = compute()
        # Copies protect the key from subsequent in-place preprocessing.
        self.entry = (key, rgb.detach().clone(), mask.detach().clone(), features, model)
        return features


def prediction_only(network, batch, compute_rays):
    """The native RayQuery prediction path, ending before criterion/evaluation."""
    rays = compute_rays(batch)
    rays, rays_pos, points, points_pos = network.forward_encoders(
        rays, batch['input_cams']['pointmaps'], masks=batch['input_cams']['valid_masks'])
    if not network.skip_dino:
        features = network.dino_proj(batch['input_cams']['dino_features'])
        if features.ndim == 4:
            features = features.squeeze(1)
        feature_positions = network.get_dino_pos(features)
        points = torch.cat([points, features], dim=1)
        points_pos = torch.cat([points_pos, feature_positions], dim=1)
    rays = network.forward_decoder(rays, rays_pos, points, points_pos)
    prediction = {**network.pts_head(rays, network.imshape),
                  **network.classifier_head(rays, network.imshape)}
    return prediction, batch['new_cams']


def native_operations():
    # These imports resolve in the existing RaySt3R source environment.
    from engine import batch_to_device, eval_pred
    from utils.batch_prep import (compute_pointmaps, depth_to_metric,
                                 compute_dino_and_store_features, normalize_batch)
    from utils.geometry import (center_pointmaps, uncenter_pointmaps,
                                compute_rays, depth2pts)
    from utils.utils import scenes_to_batch, batch_to_scenes
    return SimpleNamespace(**locals())


class SceneInference(torch.nn.Module):
    """Native EvalWrapper interface with independently selectable optimizations."""

    def __init__(self, wrapper, *, reuse_dino=True, skip_metrics=True, operations=None):
        super().__init__()
        self.wrapper = wrapper
        self.reuse_dino = reuse_dino
        self.skip_metrics = skip_metrics
        self.ops = operations or native_operations()
        self.cache = SceneFeatureCache()
        self.stats = {}

    @contextmanager
    def scene(self):
        stats = {'dino_forwards': 0, 'feature_cache_hits': 0,
                 'network_forwards': 0, 'loss_calls': 0, 'metric_calls': 0}
        with self.cache.scene(stats):
            self.stats = stats
            yield stats

    def forward(self, batch, dino_model=None):
        if not self.cache.active:
            raise RuntimeError('Use a fresh scene context for each reconstruction')
        network = self.wrapper.model
        if network.training:
            raise ValueError('RaySt3R must remain in evaluation mode')
        layers = network.dino_layers
        needs_preparation = 'pointmaps' not in batch['input_cams']
        # Keep CPU references for a content-checked cache key while preserving
        # the native device-transfer / geometry / feature-extraction order.
        source_rgb = batch['input_cams'].get('imgs')
        source_mask = batch['input_cams']['valid_masks']
        device = next(network.parameters()).device
        batch = self.ops.batch_to_device(batch, device)
        if needs_preparation:
            for name in ['new_cams', 'input_cams']:
                data = batch[name]
                data['depths'] = self.ops.depth_to_metric(data['depths'])
            for name in ['new_cams', 'input_cams']:
                data = batch[name]
                data['pointmaps'] = self.ops.compute_pointmaps(data['depths'], data['Ks'], data['c2ws'])
            if dino_model is not None and layers:
                def extract():
                    source = batch['input_cams']
                    return self.ops.compute_dino_and_store_features(
                        dino_model, source['imgs'], source['valid_masks'], layers)
                if self.reuse_dino:
                    features = self.cache.get(source_rgb, source_mask, dino_model, layers, extract)
                else:
                    self.stats['dino_forwards'] += 1
                    features = extract()
                batch['input_cams']['dino_features'] = features

        batch, scales = self.ops.normalize_batch(batch, 'median')
        batch, count = self.ops.scenes_to_batch(batch)
        batch = self.ops.center_pointmaps(batch)
        amp = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        self.stats['network_forwards'] += 1
        with amp:
            if self.skip_metrics:
                pred, truth = prediction_only(network, batch, self.ops.compute_rays)
                losses = {}
            else:
                pred, truth, losses = network(batch, mode='viz')
                self.stats['loss_calls'] += 1
        if 'pointmaps' not in pred:
            pred['pointmaps'] = self.ops.depth2pts(pred['depths'].squeeze(-1), batch['new_cams']['Ks'])
        elif 'depths' not in pred:
            pred['depths'] = pred['pointmaps'][..., -1]
        if not self.skip_metrics:
            losses = {**losses, **self.ops.eval_pred(pred, truth)}
            self.stats['metric_calls'] += 1
        pred, truth, batch = self.ops.uncenter_pointmaps(pred, truth, batch)
        pred, truth, batch = self.ops.batch_to_scenes(pred, truth, batch, count)
        return pred, truth, losses, scales[0].item()
