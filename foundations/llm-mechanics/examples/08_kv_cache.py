#!/usr/bin/env python3
"""Append to a KV cache and verify cached attention matches full recomputation."""

import torch


class KVCache:
    def __init__(self, layers: int, batch: int, capacity: int, heads: int, head_dim: int) -> None:
        shape = (layers, batch, capacity, heads, head_dim)
        self.keys = torch.zeros(shape)
        self.values = torch.zeros(shape)
        self.position = 0

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        count = keys.size(2)
        end = self.position + count
        if end > self.keys.size(2):
            raise ValueError("KV cache capacity exceeded")
        self.keys[:, :, self.position:end] = keys
        self.values[:, :, self.position:end] = values
        self.position = end


torch.manual_seed(11)
layers, batch, heads, head_dim = 2, 1, 2, 3
prompt_k = torch.randn(layers, batch, 4, heads, head_dim)
prompt_v = torch.randn_like(prompt_k)
new_k = torch.randn(layers, batch, 1, heads, head_dim)
new_v = torch.randn_like(new_k)

cache = KVCache(layers, batch, capacity=8, heads=heads, head_dim=head_dim)
cache.append(prompt_k, prompt_v)
cache.append(new_k, new_v)

query = torch.randn(batch, heads, 1, head_dim)
cached_k = cache.keys[0, :, : cache.position].transpose(1, 2)  # [B,H,T,D]
cached_v = cache.values[0, :, : cache.position].transpose(1, 2)
full_k = torch.cat([prompt_k[0], new_k[0]], dim=1).transpose(1, 2)
full_v = torch.cat([prompt_v[0], new_v[0]], dim=1).transpose(1, 2)

cached_attention = (query @ cached_k.transpose(-1, -2)).softmax(dim=-1) @ cached_v
full_attention = (query @ full_k.transpose(-1, -2)).softmax(dim=-1) @ full_v

bytes_used = cache.position * layers * batch * heads * head_dim * 2 * 4
print("cache shape:", tuple(cache.keys.shape), "position:", cache.position)
print("bytes used (fp32 K+V):", bytes_used)
print("cached output matches full recomputation:", torch.allclose(cached_attention, full_attention))

assert cache.position == 5
assert torch.allclose(cached_attention, full_attention, atol=1e-6)
assert bytes_used == 5 * layers * batch * heads * head_dim * 2 * 4

