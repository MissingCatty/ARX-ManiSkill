"""Reuse attention logits for scale and softmax during inference.

Projection, RoPE, matrix products and output projection follow models/blocks.py
(Naver CroCo, CC BY-NC-SA 4.0). No query, key, view or point is removed.
Only each resident instance is changed inside a scoped context.
"""
from contextlib import contextmanager
from types import MethodType

import torch


def probabilities(layer, q, k):
    if layer.training or torch.is_grad_enabled():
        raise RuntimeError('Attention buffer reuse is inference-only')
    weights = q @ k.transpose(-2, -1)
    weights.mul_(layer.scale)
    # CUDA autocast's native softmax first promotes low-precision logits to
    # FP32. Keep that promotion, then reuse its FP32 storage for the output.
    if (weights.device.type == 'cuda' and torch.is_autocast_enabled('cuda')
            and weights.dtype in (torch.float16, torch.bfloat16)):
        weights = weights.float()
    torch.softmax(weights, dim=-1, out=weights)
    return layer.attn_drop(weights)


def self_attention(layer, x, xpos):
    b, n, c = x.shape
    qkv = layer.qkv(x).reshape(b, n, 3, layer.num_heads, c//layer.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    if layer.rope is not None:
        q, k = layer.rope(q, xpos), layer.rope(k, xpos)
    x = (probabilities(layer, q, k) @ v).transpose(1, 2).reshape(b, n, c)
    return layer.proj_drop(layer.proj(x))


def cross_attention(layer, query, key, value, qpos, kpos):
    b, nq, c = query.shape
    q = layer.projq(query).reshape(b, nq, layer.num_heads, c//layer.num_heads).permute(0, 2, 1, 3)
    k = layer.projk(key).reshape(b, key.shape[1], layer.num_heads, c//layer.num_heads).permute(0, 2, 1, 3)
    v = layer.projv(value).reshape(b, value.shape[1], layer.num_heads, c//layer.num_heads).permute(0, 2, 1, 3)
    if layer.rope is not None:
        q, k = layer.rope(q, qpos), layer.rope(k, kpos)
    x = (probabilities(layer, q, k) @ v).transpose(1, 2).reshape(b, nq, c)
    return layer.proj_drop(layer.proj(x))


@contextmanager
def reuse_attention(model):
    from models.blocks import Attention, CrossAttention
    changed = []
    try:
        for layer in model.modules():
            if type(layer) not in (Attention, CrossAttention):
                continue
            if layer.training:
                raise RuntimeError('RaySt3R attention must remain in evaluation mode')
            changed.append((layer, layer.forward))
            layer.forward = MethodType(self_attention if type(layer) is Attention else cross_attention, layer)
        if not changed:
            raise RuntimeError('No native RaySt3R attention layers found')
        yield len(changed)
    finally:
        for layer, previous in reversed(changed):
            layer.forward = previous
