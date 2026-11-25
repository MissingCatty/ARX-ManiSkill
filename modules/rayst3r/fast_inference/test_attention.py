"""Check native self/cross attention semantics and scoped restoration on CPU."""
from pathlib import Path
import sys
import unittest

import torch
from arx_project.layout import ROOT
sys.path.insert(0,str(ROOT/'modules/rayst3r'))
from models.blocks import Attention,CrossAttention
from .attention import fused_attention


class FusedAttention(unittest.TestCase):
    def test_self_and_cross_attention_keep_projection_and_custom_scale(self):
        torch.manual_seed(1)
        for kind in [Attention,CrossAttention]:
            layer = kind(32,num_heads=4,qkv_bias=True).eval()
            layer.scale = .25
            query,key = torch.randn(2,7,32),torch.randn(2,5,32)
            args=(query,None) if kind is Attention else (query,key,key,None,None)
            original=layer.forward
            with torch.inference_mode():
                expected=layer(*args)
                trace={}
                with fused_attention(layer,trace):actual=layer(*args)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(trace['attention_calls'],1)
            self.assertEqual(layer.forward,original)
            self.assertFalse(hasattr(layer,'_ray_fast_trace'))

    def test_exception_restores_layers_and_nested_scope_is_rejected(self):
        layer=Attention(32,num_heads=4).eval();original=layer.forward
        with self.assertRaisesRegex(ValueError,'injected'):
            with fused_attention(layer,{}):
                with self.assertRaises(RuntimeError):
                    with fused_attention(layer,{}):pass
                raise ValueError('injected')
        self.assertEqual(layer.forward,original)
        self.assertFalse(hasattr(layer,'_ray_fast_trace'))

    def test_training_and_grad_enabled_calls_are_rejected(self):
        layer=Attention(32,num_heads=4)
        with self.assertRaises(RuntimeError):
            with fused_attention(layer,{}):pass
        layer.eval()
        with fused_attention(layer,{}):
            with self.assertRaises(RuntimeError):layer(torch.randn(1,3,32),None)


if __name__=='__main__':unittest.main()
