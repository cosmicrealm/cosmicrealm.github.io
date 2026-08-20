#!/usr/bin/env python3
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATHJAX_SRC = 'src="/assets/vendor/mathjax/tex-mml-chtml.js"'
MATHJAX_ENTRYPOINTS = (
    "foundations/aigc-llm-math/index.html",
    "foundations/generation-acceleration/index.html",
    "foundations/generation-distillation/index.html",
    "foundations/generation-math/index.html",
    "foundations/image-generation-data-training/index.html",
    "foundations/leetcode-hot100/index.html",
    "foundations/llm-interview-qa/index.html",
    "foundations/llm-mechanics/index.html",
    "foundations/video-generation/index.html",
)


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def assert_contains(path, needle):
    haystack = read(path)
    assert needle in haystack, f"{path} should contain {needle!r}"


def assert_not_contains(path, needle):
    haystack = read(path)
    assert needle not in haystack, f"{path} should not contain {needle!r}"


def assert_exists(path):
    assert (ROOT / path).exists(), f"{path} should exist"


def assert_not_exists(path):
    assert not (ROOT / path).exists(), f"{path} should not exist"


def main():
    assert_contains("_data/navigation.yml", 'title: "Foundations"')
    assert_contains("_data/navigation.yml", "url: /foundations/")

    assert_exists("_data/foundations.yml")
    assert_contains("_data/foundations.yml", "生成模型加速技术")
    assert_contains("_data/foundations.yml", "/foundations/generation-acceleration/")
    assert_contains("_data/foundations.yml", "生成模型原理介绍")
    assert_contains("_data/foundations.yml", "/foundations/generation-math/")
    assert_contains("_data/foundations.yml", "生成模型蒸馏技术介绍")
    assert_contains("_data/foundations.yml", "/foundations/generation-distillation/")

    assert_contains("_data/foundations.yml", "LeetCode 热题 100")
    assert_contains("_data/foundations.yml", "/foundations/leetcode-hot100/")
    assert_contains("_data/foundations.yml", "视频生成技术")
    assert_contains("_data/foundations.yml", "/foundations/video-generation/")
    assert_contains("_data/foundations.yml", "图像生成技术")
    assert_contains("_data/foundations.yml", "/foundations/image-generation-data-training/")

    assert_not_contains("_data/projects.yml", "生成模型加速技术")
    assert_not_contains("_data/projects.yml", "生成模型原理介绍")
    assert_not_contains("_data/projects.yml", "生成模型蒸馏技术介绍")

    assert_exists("_pages/foundations.md")
    assert_contains("_pages/foundations.md", "permalink: /foundations/")
    assert_contains("_pages/foundations.md", "site.data.foundations")

    assert_exists("foundations/generation-math/index.html")
    assert_exists("foundations/generation-distillation/index.html")
    assert_exists("foundations/generation-acceleration/index.html")
    assert_contains(
        "foundations/generation-acceleration/index.html",
        'href="https://cosmicrealm.github.io/foundations/generation-acceleration/"',
    )
    assert_contains(
        "foundations/generation-math/index.html",
        'href="/foundations/generation-math/"',
    )
    assert_contains(
        "foundations/generation-math/index.html",
        MATHJAX_SRC,
    )
    assert_contains(
        "foundations/generation-distillation/index.html",
        'href="/foundations/generation-distillation/"',
    )

    assert_contains("_pages/home.md", "site.data.foundations")
    assert_contains("_pages/home.md", "/foundations/")

    assert_exists("assets/vendor/mathjax/tex-mml-chtml.js")
    for path in MATHJAX_ENTRYPOINTS:
        assert read(path).count(MATHJAX_SRC) == 1, f"{path} should load shared MathJax exactly once"

    assert_not_exists("foundations/generation-math/static/vendor/mathjax")
    assert_not_exists("foundations/llm-mechanics/static/vendor/mathjax")

    bundle = read("assets/vendor/mathjax/tex-mml-chtml.js")
    referenced_fonts = set(re.findall(r"MathJax_[A-Za-z0-9-]+\.woff", bundle))
    font_root = ROOT / "assets/vendor/mathjax/output/chtml/fonts/woff-v2"
    missing_fonts = sorted(name for name in referenced_fonts if not (font_root / name).is_file())
    assert not missing_fonts, f"MathJax bundle references missing fonts: {missing_fonts}"


if __name__ == "__main__":
    main()
