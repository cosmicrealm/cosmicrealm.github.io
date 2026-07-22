#!/usr/bin/env python3
"""Align teacher-forcing inputs and one-token-shifted targets."""

import torch
from torch.nn import functional as F


tokens = torch.tensor([[1, 4, 2, 5, 3]])
inputs = tokens[:, :-1]
targets = tokens[:, 1:]
vocab_size = 6

# Construct logits that strongly predict the correctly shifted target.
logits = F.one_hot(targets, num_classes=vocab_size).float() * 7.0
correct_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
unshifted_loss = F.cross_entropy(logits.reshape(-1, vocab_size), inputs.reshape(-1))

print("inputs: ", inputs.tolist())
print("targets:", targets.tolist())
print("correct/unshifted loss:", correct_loss.item(), unshifted_loss.item())

assert inputs.shape == targets.shape == (1, 4)
assert torch.equal(inputs[:, 1:], targets[:, :-1])
assert correct_loss < unshifted_loss

