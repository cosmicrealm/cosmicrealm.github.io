#!/usr/bin/env python3
"""Apply a causal mask before softmax and verify zero future attention."""

import torch


length = 4
scores = torch.zeros(length, length)
future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
masked_scores = scores.masked_fill(future, float("-inf"))
attention = masked_scores.softmax(dim=-1)

without_mask = scores.softmax(dim=-1)
future_mass_without_mask = without_mask[future].sum().item()

print(attention)
print("shape:", tuple(attention.shape))
print("future mass with mask:", attention[future].sum().item())
print("future mass without mask:", future_mass_without_mask)

assert attention.shape == (length, length)
assert torch.all(attention[future] == 0)
assert torch.allclose(attention.sum(dim=-1), torch.ones(length))
assert future_mass_without_mask > 0

