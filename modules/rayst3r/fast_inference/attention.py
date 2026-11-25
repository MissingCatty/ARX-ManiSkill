"""Scoped SDPA for native RaySt3R attention; weights and RoPE are unchanged.

Projection/layout follows models/blocks.py, originally Naver CroCo,
Copyright 2022-present Naver Corporation, CC BY-NC-SA 4.0.
"""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def attend(layer, q, k, v):
    if layer.training or torch.is_grad_enabled():
        raise RuntimeError('Fused RaySt3R attention is inference-only')
    trace = layer._ray_fast_trace
    trace['attention_calls'] += 1
    shape = (tuple(q.shape), tuple(k.shape), str(q.dtype))
    if shape not in trace['attention_shapes']:
        trace['attention_shapes'].append(shape)
    backend = SDPBackend.FLASH_ATTENTION if q.is_cuda else SDPBackend.MATH
    with sdpa_kernel(backend):
        return scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False, scale=layer.scale)


def self_attention(layer, x, xpos):
    b, n, c = x.shape
    qkv = layer.qkv(x).reshape(b, n, 3, layer.num_heads, c//layer.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    if layer.rope is not None:
        q, k = layer.rope(q, xpos), layer.rope(k, xpos)
    x = attend(layer, q, k, v).transpose(1, 2).reshape(b, n, c)
    return layer.proj_drop(layer.proj(x))


def cross_attention(layer, query, key, value, qpos, kpos):
    b, n, c = query.shape
    q = layer.projq(query).reshape(b, n, layer.num_heads, c//layer.num_heads).permute(0, 2, 1, 3)
    k = layer.projk(key).reshape(b, key.shape[1], layer.num_heads, c//layer.num_heads).permute(0, 2, 1, 3)
    v = layer.projv(value).reshape(b, value.shape[1], layer.num_heads, c//layer.num_heads).permute(0, 2, 1, 3)
    if layer.rope is not None:
        q, k = layer.rope(q, qpos), layer.rope(k, kpos)
    x = attend(layer, q, k, v).transpose(1, 2).reshape(b, n, c)
    return layer.proj_drop(layer.proj(x))


@contextmanager
def fused_attention(model, trace):
    from models.blocks import Attention, CrossAttention
    changed = []
    trace.update(attention_calls=0, attention_shapes=[])
    try:
        for layer in model.modules():
            if type(layer) not in (Attention, CrossAttention):
                continue
            if layer.training or hasattr(layer, '_ray_fast_trace'):
                raise RuntimeError('Attention is training or already managed by another inference')
            changed.append((layer, layer.forward))
            layer._ray_fast_trace = trace
            layer.forward = MethodType(self_attention if type(layer) is Attention else cross_attention, layer)
        if not changed:
            raise RuntimeError('No native attention layers found')
        trace['patched_layers'] = len(changed)
        yield
    finally:
        for layer, forward in reversed(changed):
            layer.forward = forward
            del layer._ray_fast_trace
