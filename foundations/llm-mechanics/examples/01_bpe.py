#!/usr/bin/env python3
"""Train a tiny byte-level BPE tokenizer and verify UTF-8 round-trip."""

from collections import Counter


def pair_counts(ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(ids, ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            output.append(new_id)
            index += 2
        else:
            output.append(ids[index])
            index += 1
    return output


def train(text: str, num_merges: int) -> tuple[list[tuple[tuple[int, int], int]], dict[int, bytes]]:
    ids = list(text.encode("utf-8"))
    vocab = {index: bytes([index]) for index in range(256)}
    rules: list[tuple[tuple[int, int], int]] = []
    for step in range(num_merges):
        counts = pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=lambda item: (counts[item], item))
        new_id = 256 + step
        ids = merge(ids, pair, new_id)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        rules.append((pair, new_id))
    return rules, vocab


def encode(text: str, rules: list[tuple[tuple[int, int], int]]) -> list[int]:
    ids = list(text.encode("utf-8"))
    for pair, new_id in rules:
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[token_id] for token_id in ids).decode("utf-8")


corpus = "low lower lowest low 你好"
rules, vocab = train(corpus, num_merges=8)
encoded = encode(corpus, rules)
decoded = decode(encoded, vocab)

print("bytes:", len(corpus.encode("utf-8")), "tokens:", len(encoded))
print("first merges:", rules[:3])
print("round-trip:", decoded == corpus)

assert decoded == corpus
assert len(encoded) < len(corpus.encode("utf-8"))
assert all(token_id in vocab for token_id in encoded)

