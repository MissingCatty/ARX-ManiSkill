"""CPU checks for stale-feature failures and prediction-only inference."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from modules.rayst3r.optimized.inference import (
    SceneFeatureCache, SceneInference, prediction_only)


def counters():
    return {'dino_forwards': 0, 'feature_cache_hits': 0,
            'network_forwards': 0, 'loss_calls': 0, 'metric_calls': 0}


class FeatureCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache = SceneFeatureCache()
        self.rgb = torch.arange(12).reshape(1, 1, 2, 2, 3)
        self.mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        self.dino = SimpleNamespace(training=False)
        self.calls = 0

    def compute(self):
        self.calls += 1
        return self.rgb.float().sum()+self.calls

    def get(self, **kw):
        args = dict(rgb=self.rgb, mask=self.mask, model=self.dino,
                    layers=[23], compute=self.compute)
        return self.cache.get(**{**args, **kw})

    def test_identical_new_loader_tensors_reuse_exact_feature(self):
        stats = counters()
        with self.cache.scene(stats):
            first = self.get()
            second = self.get(rgb=self.rgb.clone(), mask=self.mask.clone())
            self.assertIs(first, second)
            self.assertEqual((self.calls, stats['feature_cache_hits']), (1, 1))
        self.assertIsNone(self.cache.entry)

    def test_changed_pixels_mask_layers_and_model_invalidate_cache(self):
        with self.cache.scene(counters()):
            self.get()
            image = self.rgb.clone(); image[..., 0] += 1
            self.get(rgb=image)
            mask = self.mask.clone(); mask[..., 0] = False
            self.get(mask=mask)
            self.get(layers=[22])
            self.get(model=SimpleNamespace(training=False))
        self.assertEqual(self.calls, 5)

    def test_in_place_source_edit_cannot_modify_stored_cache_key(self):
        with self.cache.scene(counters()):
            self.get()
            self.rgb.add_(1)
            self.get()
        self.assertEqual(self.calls, 2)

    def test_dtype_change_does_not_reuse_feature(self):
        with self.cache.scene(counters()):
            self.get()
            self.get(rgb=self.rgb.float())
        self.assertEqual(self.calls, 2)

    def test_new_scene_never_reuses_previous_observation(self):
        for _ in range(2):
            with self.cache.scene(counters()):
                self.get()
        self.assertEqual(self.calls, 2)

    def test_failed_extraction_releases_scene_without_stale_feature(self):
        def fault():
            raise RuntimeError('injected DINO failure')
        with self.assertRaisesRegex(RuntimeError, 'injected DINO'):
            with self.cache.scene(counters()):
                self.get(compute=fault)
        self.assertFalse(self.cache.active)
        self.assertIsNone(self.cache.entry)
        with self.cache.scene(counters()):
            self.get()
        self.assertEqual(self.calls, 1)

    def test_outside_scene_training_and_concurrent_reuse_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'active scene'):
            self.get()
        with self.cache.scene(counters()):
            with self.assertRaisesRegex(ValueError, 'evaluation mode'):
                self.get(model=SimpleNamespace(training=True))
            def overlap():
                with self.cache.scene(counters()):
                    pass
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaisesRegex(RuntimeError, 'Overlapping'):
                    pool.submit(overlap).result()
            self.get()


class TinyNetwork(torch.nn.Module):
    def __init__(self, skip_dino=True):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(2.))
        self.skip_dino = skip_dino
        self.dino_layers = [] if skip_dino else [23]
        self.dino_proj = torch.nn.Identity()
        self.imshape = (1, 2)
        self.eval()

    def forward_encoders(self, rays, points, masks=None):
        return rays+self.offset, torch.zeros_like(rays), points, torch.zeros_like(points)

    def get_dino_pos(self, features):
        return torch.zeros_like(features)

    def forward_decoder(self, rays, ray_pos, points, point_pos):
        return rays+points.mean(dim=1, keepdim=True)

    def pts_head(self, rays, shape):
        return {'pointmaps': rays*3}

    def classifier_head(self, rays, shape):
        return {'classifier': rays.sum(dim=-1)}

    def forward(self, *args, **kw):
        raise AssertionError('The loss-bearing forward must not be called')


class PredictionOnlyTest(unittest.TestCase):
    def batch(self):
        return {'input_cams': {
            'pointmaps': torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            'valid_masks': torch.ones(1, 2, dtype=torch.bool),
            'dino_features': torch.full((1, 1, 2, 3), 4.)},
            'new_cams': {'rays': torch.ones(1, 2, 3)}}

    def test_heads_match_known_prediction_without_loss_for_both_dino_paths(self):
        for use_dino in [False, True]:
            network, batch = TinyNetwork(skip_dino=not use_dino), self.batch()
            before = deepcopy(network.state_dict())
            pred, truth = prediction_only(network, batch, lambda b: b['new_cams']['rays'])
            points = batch['input_cams']['pointmaps']
            if use_dino:
                points = torch.cat([points, torch.full((1, 2, 3), 4.)], dim=1)
            expected = 3+points.mean(dim=1, keepdim=True)
            torch.testing.assert_close(pred['pointmaps'], (expected*3).expand(1, 2, 3), rtol=0, atol=0)
            torch.testing.assert_close(pred['classifier'], expected.sum(-1).expand(1, 2), rtol=0, atol=0)
            self.assertIs(truth, batch['new_cams'])
            for key, value in before.items():
                torch.testing.assert_close(network.state_dict()[key], value, rtol=0, atol=0)

    def test_scene_wrapper_never_calls_loss_or_metrics_and_clears_state(self):
        network = TinyNetwork()
        def forbidden(*args, **kw):
            raise AssertionError('Metric computation must not run')
        ops = SimpleNamespace(
            batch_to_device=lambda b, d: b,
            normalize_batch=lambda b, m: (b, [torch.tensor(1.)]),
            scenes_to_batch=lambda b: (b, 1), center_pointmaps=lambda b: b,
            compute_rays=lambda b: b['new_cams']['rays'], eval_pred=forbidden,
            uncenter_pointmaps=lambda p, g, b: (p, g, b),
            batch_to_scenes=lambda p, g, b, n: (p, g, b))
        wrapper = SceneInference(SimpleNamespace(model=network), operations=ops)
        with self.assertRaisesRegex(RuntimeError, 'fresh scene'):
            wrapper(self.batch())
        with wrapper.scene() as stats:
            pred, truth, losses, scale = wrapper(self.batch())
            self.assertEqual((losses, scale), ({}, 1.))
            self.assertEqual((stats['loss_calls'], stats['metric_calls']), (0, 0))
            self.assertEqual(stats['network_forwards'], 1)
        self.assertIsNone(wrapper.cache.entry)
        network.train()
        with wrapper.scene():
            with self.assertRaisesRegex(ValueError, 'evaluation mode'):
                wrapper(self.batch())


class NativeNetworkTest(unittest.TestCase):
    def test_real_native_encoders_heads_and_two_stage_geometry_on_cpu(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'tests/rayst3r_native_cpu_check.py')],
            cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                              OMP_NUM_THREADS='1', PYTHONPATH=str(root)),
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('"coordinate_bitwise_equal": true', result.stdout)


if __name__ == '__main__':
    unittest.main()
