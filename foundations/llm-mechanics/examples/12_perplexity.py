#!/usr/bin/env python3
"""Compute perplexity and expose its tokenization-dependent denominator."""

import math


def perplexity(token_nlls: list[float]) -> float:
    if not token_nlls:
        raise ValueError("perplexity needs at least one evaluated token")
    return math.exp(sum(token_nlls) / len(token_nlls))


def bits_per_byte(token_nlls: list[float], byte_count: int) -> float:
    if byte_count <= 0:
        raise ValueError("byte_count must be positive")
    return sum(token_nlls) / (math.log(2) * byte_count)


# The same four bytes can be segmented into two or four tokens.
two_token_nlls = [0.30, 0.50]
four_token_nlls = [0.18, 0.22, 0.20, 0.20]
byte_count = 4

ppl_two = perplexity(two_token_nlls)
ppl_four = perplexity(four_token_nlls)
bpb_two = bits_per_byte(two_token_nlls, byte_count)
bpb_four = bits_per_byte(four_token_nlls, byte_count)

print("perplexity (2 tokens / 4 tokens):", ppl_two, ppl_four)
print("bits per byte:", bpb_two, bpb_four)
print("limitation: token-level averages change when tokenization changes")

assert ppl_two > 1 and ppl_four > 1
assert not math.isclose(ppl_two, ppl_four)
assert math.isclose(bpb_two, bpb_four)

