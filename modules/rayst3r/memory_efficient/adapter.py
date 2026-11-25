"""The native RPC/model contracts with inference-only logits-buffer reuse."""
from threading import Lock

from modules.rayst3r.arx_adapter import RaySt3R as NativeWorker
from modules.rayst3r.arx_reconstruction import RaySt3RAdapter as NativeAdapter
from modules.rayst3r.memory_efficient.attention import reuse_attention


class MemoryEfficientAdapter(NativeAdapter):
    def __init__(self, *, baseline=None):
        if baseline is None:
            super().__init__()
        else:
            self.model, self.dino, self.scene_function = baseline.model, baseline.dino, baseline.scene_function
            self.configuration = dict(baseline.configuration)
        self.configuration = {**self.configuration, 'attention_buffer_reuse': 'inplace_scale_softmax',
            'native_precision_views_points_weights_and_math_unchanged': True}
        self._inference_lock = Lock()

    def predict(self, *args, **kwargs):
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError('Overlapping reconstruction is unsupported')
        try:
            with reuse_attention(self.model):
                return super().predict(*args, **kwargs)
        finally:
            self._inference_lock.release()


class MemoryEfficientWorker(NativeWorker):
    def __init__(self, args):
        super().__init__(args)
        self.model = MemoryEfficientAdapter(baseline=self.model)
        # Keep the inference pool resident while waiting for robot/planner
        # stages, so the selected device remains available to this service.
        # Releasing these tensors to this process's allocator (not to CUDA)
        # lets inference reuse the storage without a gap in GPU ownership.
        self._workspace = []
        self._hold_workspace()

    def _hold_workspace(self):
        # One large block serves attention matrices; smaller blocks can be
        # reclaimed even when CUDA libraries retain small allocations.
        self._workspace = [self.torch.empty(mib*1024**2, dtype=self.torch.uint8,
            device='cuda') for mib in [7168, *([512]*14)]]

    def request(self, directory, **options):
        self._workspace.clear()
        result = super().request(directory, **options)
        self._hold_workspace()
        # Keep the live workspace and weights; discard other unused blocks.
        self.torch.cuda.empty_cache()
        return result


RaySt3RAdapter = MemoryEfficientAdapter


def create(args):
    return MemoryEfficientWorker(args)
