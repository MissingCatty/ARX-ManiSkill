"""Native attention values, precision, restoration and inference-only guards."""
import sys
import unittest

from arx_project.layout import ROOT
sys.path.insert(0, str(ROOT/'modules/rayst3r'))
import torch
from models.blocks import Attention, CrossAttention
from modules.rayst3r.memory_efficient.attention import reuse_attention


class AttentionMemoryTest(unittest.TestCase):
    def test_full_self_and_cross_attention_match_in_fp32_and_bf16(self):
        for kind in [Attention, CrossAttention]:
            for dtype in [torch.float32, torch.bfloat16]:
                with self.subTest(kind=kind.__name__, dtype=dtype):
                    torch.manual_seed(35)
                    model = kind(64, num_heads=8).to(dtype).eval()
                    x, y = [torch.randn(2, n, 64).to(dtype) for n in [97, 51]]
                    px, py = torch.zeros(2, 97, 2), torch.zeros(2, 51, 2)
                    args = (x, px) if kind is Attention else (x, y, y, px, py)
                    original = model.forward
                    with torch.inference_mode():
                        baseline = model(*args)
                        with reuse_attention(model):
                            output = model(*args)
                        self.assertTrue(torch.equal(baseline, output))
                        self.assertEqual(baseline.dtype, output.dtype)
                        self.assertEqual(model.forward, original)

    def test_fault_restores_native_method(self):
        model = Attention(64).eval()
        previous = model.forward
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            with reuse_attention(model):
                raise RuntimeError('injected failure')
        self.assertEqual(model.forward, previous)

    def test_training_or_gradient_execution_is_rejected(self):
        model = Attention(64)
        with self.assertRaisesRegex(RuntimeError, 'evaluation mode'):
            with reuse_attention(model):
                pass
        model.eval()
        with reuse_attention(model):
            with self.assertRaisesRegex(RuntimeError, 'inference-only'):
                model(torch.randn(1, 2, 64), torch.zeros(1, 2, 2))


if __name__ == '__main__':
    unittest.main()
