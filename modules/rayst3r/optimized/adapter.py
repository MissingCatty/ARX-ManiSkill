"""Optional optimized reconstruction using the exact native input/output path."""
from copy import deepcopy

from arx_project.layout import write_json
from modules.rayst3r.arx_reconstruction import RaySt3RAdapter as NativeAdapter
from modules.rayst3r.arx_adapter import RaySt3R as NativeWorker


VARIANTS = {'cache_only': (True, False), 'prediction_only': (False, True),
            'combined': (True, True)}


class OptimizedRaySt3RAdapter(NativeAdapter):
    def __init__(self, *, baseline=None, variant='combined'):
        if variant not in VARIANTS:
            raise ValueError('Unknown compute optimization variant: '+variant)
        if baseline is None:
            super().__init__()
        else:
            # Benchmark variants share resident weights, not feature caches.
            self.model, self.dino = baseline.model, baseline.dino
            self.scene_function = baseline.scene_function
            self.configuration = deepcopy(baseline.configuration)
        from modules.rayst3r.optimized.inference import SceneInference
        reuse, skip = VARIANTS[variant]
        self.model = SceneInference(self.model, reuse_dino=reuse, skip_metrics=skip)
        self.variant = variant
        self.last_trace = {}
        self.configuration = {**self.configuration, 'compute_optimization': variant,
            'feature_cache_scope': 'one_reconstruction_identical_RGB_mask_and_DINO_layers',
            'native_loader_postprocessing_precision_views_unchanged': True,
            'ray_network_precision': 'native_BF16_autocast', 'dino_precision': 'native_FP32'}

    def predict(self, rgb, depth, mask, k, work):
        with self.model.scene() as stats:
            try:
                output = super().predict(rgb, depth, mask, k, work)
            finally:
                self.last_trace = dict(stats)
        return {**output, 'optimization_trace': self.last_trace}


class OptimizedWorker(NativeWorker):
    def __init__(self, args):
        super().__init__(args)
        self.model = OptimizedRaySt3RAdapter(baseline=self.model)

    def request(self, directory, **options):
        result = super().request(directory, **options)
        result['optimization_trace'] = self.model.last_trace
        write_json(directory/'rayst3r.json', result)
        return result


# Both reconstruction and RPC contracts are available for a future registration.
RaySt3RAdapter = OptimizedRaySt3RAdapter


def create(args):
    return OptimizedWorker(args)
