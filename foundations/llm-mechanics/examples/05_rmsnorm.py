#!/usr/bin/env python3
"""Implement RMSNorm and compare it with torch.nn.functional.rms_norm."""

import torch
from torch.nn import functional as F


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms_inverse = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (x.float() * rms_inverse * scale.float()).to(x.dtype)


x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 0.0, 2.0, 1.0]])
scale = torch.ones(4)
manual = rms_norm(x, scale)
reference = F.rms_norm(x, (x.size(-1),), weight=scale, eps=1e-6)
output_rms = manual.square().mean(dim=-1).sqrt()

print("output RMS:", output_rms.tolist())
print("matches PyTorch:", torch.allclose(manual, reference, atol=1e-6))

assert torch.allclose(manual, reference, atol=1e-6)
assert torch.allclose(output_rms, torch.ones_like(output_rms), atol=1e-5)
assert torch.isfinite(rms_norm(torch.zeros_like(x), scale)).all()  # eps prevents division by zero

