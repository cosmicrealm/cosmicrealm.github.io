#!/usr/bin/env python3
"""Static contract checks for the LLM Mechanics Foundations lecture."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "foundations" / "llm-mechanics" / "index.html"
CSS = ROOT / "foundations" / "llm-mechanics" / "static" / "css" / "index.css"
JS = ROOT / "foundations" / "llm-mechanics" / "static" / "js" / "index.js"
EXAMPLES_DIR = ROOT / "foundations" / "llm-mechanics" / "examples"

CHAPTERS = ["tokens", "decoder", "training", "inference", "decoding"]
LABS = ["bpe", "decoder", "teacher", "prefill", "kv", "sampling"]
CONCEPTS = [
    "BPE",
    "decoder-only",
    "causal mask",
    "RoPE",
    "RMSNorm",
    "SwiGLU",
    "prefill",
    "decode",
    "KV cache",
    "greedy",
    "temperature",
    "top-p",
    "repetition",
    "EOS",
    "teacher forcing",
    "perplexity",
]
FORBIDDEN_LABELS = ["小白", "专家", "Expert Note", "Beginner Note"]


class DocumentIndex(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.downloads: list[str] = []
        self.labels = 0
        self.aria_labels = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if identifier := values.get("id"):
            self.ids.add(identifier)
        if href := values.get("href"):
            self.hrefs.append(href)
        if "download" in values:
            self.downloads.append(values.get("href", ""))
        if tag == "label":
            self.labels += 1
        if values.get("aria-label") or values.get("aria-labelledby"):
            self.aria_labels += 1


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []
    if not PAGE.exists():
        print(f"FAIL: missing page: {PAGE.relative_to(ROOT)}", file=sys.stderr)
        return 1

    html = PAGE.read_text(encoding="utf-8")
    parser = DocumentIndex()
    parser.feed(html)

    for chapter in CHAPTERS:
        require(chapter in parser.ids, f"missing chapter id: {chapter}", failures)
        require(f'href="#{chapter}"' in html, f"missing TOC link: {chapter}", failures)
    for lab in LABS:
        require(f"lab-{lab}" in parser.ids, f"missing lab id: lab-{lab}", failures)

    for concept in CONCEPTS:
        require(concept.lower() in html.lower(), f"missing concept text: {concept}", failures)

    examples = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py")) if EXAMPLES_DIR.exists() else []
    require(len(examples) == 12, f"expected 12 example files, found {len(examples)}", failures)
    for index in range(1, 13):
        prefix = f"{index:02d}_"
        matching = [path for path in examples if path.name.startswith(prefix)]
        require(len(matching) == 1, f"expected one example with prefix {prefix}", failures)
        if matching:
            href = f"./examples/{matching[0].name}"
            require(href in parser.hrefs, f"page does not link example: {href}", failures)
            require(href in parser.downloads, f"example link lacks download: {href}", failures)

    require("https://cs336.stanford.edu/" in parser.hrefs, "missing CS336 2026 course link", failures)
    require(
        any("stanford-cs336/assignment1-basics" in href for href in parser.hrefs),
        "missing CS336 Assignment 1 source",
        failures,
    )
    require(
        any("92d63d4e8bb4df75c3b71618f31ddde2378b2bcd" in href for href in parser.hrefs),
        "missing commit-pinned nanochat source",
        failures,
    )
    require("bits per byte" in html.lower(), "missing nanochat BPB comparison", failures)
    require("ReLU²" in html, "missing nanochat ReLU-squared boundary", failures)
    require("没有直接" in html or "未直接" in html, "missing source-absence boundary", failures)

    require(CSS.exists(), f"missing stylesheet: {CSS.relative_to(ROOT)}", failures)
    require(JS.exists(), f"missing script: {JS.relative_to(ROOT)}", failures)
    require('./static/css/index.css' in html, "page does not load local stylesheet", failures)
    require('./static/js/index.js' in html, "page does not load local script", failures)
    require('/assets/vendor/mathjax/tex-mml-chtml.js' in html, "page does not load shared MathJax", failures)
    require(parser.labels >= 12, f"expected at least 12 labels, found {parser.labels}", failures)
    require(parser.aria_labels >= 12, f"expected at least 12 ARIA labels, found {parser.aria_labels}", failures)
    require("lab-error" in html, "missing lab-local error fallback", failures)
    require("data-copy-target" in html, "missing code copy controls", failures)
    require("readingProgressBar" in parser.ids, "missing reading progress bar", failures)
    require("tocToggle" in parser.ids and "lectureToc" in parser.ids, "missing responsive TOC controls", failures)

    for label in FORBIDDEN_LABELS:
        require(label.lower() not in html.lower(), f"forbidden audience label: {label}", failures)

    duplicate_ids = [
        identifier
        for identifier in set(re.findall(r'\bid="([^"]+)"', html))
        if len(re.findall(rf'\bid="{re.escape(identifier)}"', html)) > 1
    ]
    require(not duplicate_ids, f"duplicate ids: {', '.join(sorted(duplicate_ids))}", failures)

    if failures:
        print("LLM mechanics verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("LLM mechanics static contract passed")
    print(f"- chapters: {len(CHAPTERS)}")
    print(f"- labs: {len(LABS)}")
    print(f"- examples: {len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
