"""Four inference variants sharing native weights and full-view geometry."""
from contextlib import nullcontext
from threading import Lock

from modules.rayst3r.arx_reconstruction import RaySt3RAdapter as NativeAdapter
from modules.rayst3r.memory_efficient.attention import reuse_attention
from modules.rayst3r.optimized.inference import SceneInference
from .attention import fused_attention

VARIANTS = ('baseline', 'combined', 'sdpa', 'sdpa_combined')


class FastAdapter(NativeAdapter):
    def __init__(self, *, baseline=None, variant='sdpa_combined'):
        if variant not in VARIANTS:
            raise ValueError('Unknown inference variant')
        if baseline is None:
            super().__init__()
        else:
            self.model, self.dino, self.scene_function = baseline.model, baseline.dino, baseline.scene_function
            self.configuration = dict(baseline.configuration)
        self.native_model = self.model
        if variant in ('combined', 'sdpa_combined'):
            self.model = SceneInference(self.native_model, reuse_dino=True, skip_metrics=True)
        self.variant, self.lock, self.last_trace = variant, Lock(), {}
        self.configuration = {**self.configuration, 'inference_acceleration':variant,
            'actual_novel_views':21, 'input_resolution_unchanged':True,
            'dino_precision':'native_FP32', 'ray_precision':'native_BF16_autocast',
            'attention_backend':'flash_only' if variant.startswith('sdpa') else 'native_inplace',
            'automatic_activation':False}

    def predict(self, *args, **kwargs):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Overlapping model inference is unsupported')
        trace = {}
        try:
            attention = fused_attention(self.native_model, trace) if self.variant.startswith('sdpa') else reuse_attention(self.native_model)
            scene = self.model.scene() if isinstance(self.model, SceneInference) else nullcontext({})
            with scene as stats, attention:
                result = super().predict(*args, **kwargs)
            trace.update(stats)
            self.last_trace = trace
            return {**result, 'optimization_trace':trace}
        finally:
            self.lock.release()
