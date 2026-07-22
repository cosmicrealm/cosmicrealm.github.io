#!/usr/bin/env python3
"""Compute a SwiGLU feed-forward layer and contrast it with ReLU-squared."""

import torch
from torch.nn import functional as F


def swiglu(x: torch.Tensor, gate: torch.Tensor, value: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ gate) * (x @ value)) @ out


torch.manual_seed(3)
batch, length, width, hidden = 2, 4, 6, 10
x = torch.randn(batch, length, width)
gate = torch.randn(width, hidden)
value = torch.randn(width, hidden)
out = torch.randn(hidden, width)

y = swiglu(x, gate, value, out)
relu_squared = F.relu(x @ gate).square() @ out

print("input/output shape:", tuple(x.shape), tuple(y.shape))
print("SwiGLU differs from ReLU²:", not torch.allclose(y, relu_squared))

assert y.shape == x.shape
assert torch.isfinite(y).all()
assert not torch.allclose(y, relu_squared)
