#!/usr/bin/env python3
"""Apply RoPE and verify norm preservation plus relative-position invariance."""

import torch


def rope(vector: torch.Tensor, position: int, base: float = 10_000.0) -> torch.Tensor:
    width = vector.numel()
    frequencies = base ** (-torch.arange(0, width, 2, dtype=vector.dtype) / width)
    angles = position * frequencies
    even, odd = vector[0::2], vector[1::2]
    rotated = torch.empty_like(vector)
    rotated[0::2] = even * angles.cos() - odd * angles.sin()
    rotated[1::2] = even * angles.sin() + odd * angles.cos()
    return rotated


q = torch.tensor([1.0, 2.0, -1.0, 0.5])
k = torch.tensor([0.5, -1.0, 2.0, 1.0])

dot_2_5 = torch.dot(rope(q, 2), rope(k, 5))
dot_9_12 = torch.dot(rope(q, 9), rope(k, 12))

print("norm before/after:", q.norm().item(), rope(q, 7).norm().item())
print("same relative offset dot products:", dot_2_5.item(), dot_9_12.item())

assert torch.allclose(q.norm(), rope(q, 7).norm(), atol=1e-6)
assert torch.allclose(dot_2_5, dot_9_12, atol=1e-5)

