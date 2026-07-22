#!/usr/bin/env python3
"""Run one pre-norm decoder-only Transformer block."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.ffn_norm = RMSNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)
        self.ffn = SwiGLU(width, hidden=2 * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        normalized = self.attn_norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=future)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


torch.manual_seed(7)
tokens = torch.randn(2, 5, 8)  # [batch, sequence, model width]
block = DecoderBlock(width=8, heads=2)
output = block(tokens)

print("input shape:", tuple(tokens.shape))
print("output shape:", tuple(output.shape))
print("residual changed:", not torch.allclose(tokens, output))

assert output.shape == tokens.shape
assert torch.isfinite(output).all()
assert not torch.allclose(tokens, output)

