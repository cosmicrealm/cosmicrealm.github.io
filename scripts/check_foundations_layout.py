#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        'href="/foundations/generation-acceleration/"',
    )
    assert_contains(
        "foundations/generation-math/index.html",
        'href="/foundations/generation-math/"',
    )
    assert_contains(
        "foundations/generation-distillation/index.html",
        'href="/foundations/generation-distillation/"',
    )

    assert_contains("_pages/about.md", "site.data.foundations")
    assert_contains("_pages/about.md", "/foundations/")


if __name__ == "__main__":
    main()
