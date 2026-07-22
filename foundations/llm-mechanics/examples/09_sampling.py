#!/usr/bin/env python3
"""Compare greedy, temperature sampling, and top-p filtering."""

import math
import random


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax sampling")
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exps = [math.exp(value - maximum) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def nucleus(probs: list[float], p: float) -> list[tuple[int, float]]:
    if not 0 < p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    kept: list[tuple[int, float]] = []
    cumulative = 0.0
    for item in ranked:
        kept.append(item)
        cumulative += item[1]
        if cumulative >= p:
            break
    return kept


def sample(items: list[tuple[int, float]], seed: int) -> int:
    rng = random.Random(seed)
    total = sum(probability for _, probability in items)
    draw = rng.random() * total
    cumulative = 0.0
    for token_id, probability in items:
        cumulative += probability
        if draw <= cumulative:
            return token_id
    return items[-1][0]


logits = [2.5, 1.7, 0.4, -0.8]
greedy = max(range(len(logits)), key=logits.__getitem__)
cool = softmax(logits, temperature=0.5)
warm = softmax(logits, temperature=1.5)
candidates = nucleus(warm, p=0.8)
sampled = sample(candidates, seed=9)

print("greedy token:", greedy)
print("max probability cool/warm:", max(cool), max(warm))
print("top-p candidates:", candidates, "sampled:", sampled)

assert greedy == 0
assert max(cool) > max(warm)
assert candidates and sum(prob for _, prob in candidates) >= 0.8
assert sampled in {token_id for token_id, _ in candidates}

