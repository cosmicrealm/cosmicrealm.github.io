#!/usr/bin/env python3
"""Show that prefill processes a sequence while decode processes one new query."""

import torch


def project(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return x, 0.7 * x, 1.3 * x


def causal_attention(x: torch.Tensor) -> torch.Tensor:
    q, k, v = project(x)
    scores = q @ k.transpose(-1, -2) / x.size(-1) ** 0.5
    future = torch.triu(torch.ones(x.size(0), x.size(0), dtype=torch.bool), diagonal=1)
    return scores.masked_fill(future, float("-inf")).softmax(dim=-1) @ v


torch.manual_seed(5)
prompt = torch.randn(4, 6)
new_token = torch.randn(1, 6)

prefill_output = causal_attention(prompt)
full_output = causal_attention(torch.cat([prompt, new_token], dim=0))

q_new, _, _ = project(new_token)
_, k_all, v_all = project(torch.cat([prompt, new_token], dim=0))
decode_scores = q_new @ k_all.transpose(-1, -2) / prompt.size(-1) ** 0.5
decode_output = decode_scores.softmax(dim=-1) @ v_all

print("prefill queries:", prefill_output.size(0))
print("decode queries:", decode_output.size(0))
print("decode matches full final position:", torch.allclose(decode_output, full_output[-1:]))

assert prefill_output.shape == prompt.shape
assert decode_output.shape == new_token.shape
assert torch.allclose(decode_output, full_output[-1:], atol=1e-6)

