#!/usr/bin/env python3
"""Apply repetition penalty and stop generation when EOS is selected."""


def penalize(logits: list[float], seen: set[int], penalty: float) -> list[float]:
    if penalty < 1:
        raise ValueError("repetition penalty must be at least 1")
    adjusted = logits.copy()
    for token_id in seen:
        adjusted[token_id] = adjusted[token_id] / penalty if adjusted[token_id] >= 0 else adjusted[token_id] * penalty
    return adjusted


EOS = 2
step_logits = [
    [4.0, 3.0, -2.0],
    [4.0, 3.7, -1.0],
    [0.2, 0.1, 5.0],
    [9.0, 0.0, 0.0],  # never reached because EOS stops the loop
]

generated: list[int] = []
for logits in step_logits:
    adjusted = penalize(logits, set(generated), penalty=1.5)
    next_token = max(range(len(adjusted)), key=adjusted.__getitem__)
    if next_token == EOS:
        print("EOS reached; stop before appending it")
        break
    generated.append(next_token)

print("generated token ids:", generated)

assert generated == [0, 1]
assert len(generated) < len(step_logits)
assert penalize([-1.0, 1.0], {0, 1}, 2.0) == [-2.0, 0.5]

