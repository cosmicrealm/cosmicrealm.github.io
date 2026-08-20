#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_CHOICES = ("routes", "config", "legacy", "projects", "build", "dirty")
TEXT_SCAN_DIRS = (
    "_pages",
    "_data",
    "_publications",
    "_includes",
    "_layouts",
    "_posts",
    "projects",
    "foundations",
)
TEXT_SUFFIXES = {
    "",
    ".md",
    ".markdown",
    ".html",
    ".xml",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".css",
    ".js",
    ".svg",
    ".py",
    ".liquid",
}
ZERO_REF_SCAN_FILES = ("README.md", "_config.yml")
EXPECTED_PUBLICATION_PERMALINKS = {
    "dcpn.md": "/publications/dcpn/",
    "iconface.md": "/publications/iconface/",
    "ldcr.md": "/publications/ldcr/",
    "ntire-2026-face-restoration.md": "/publications/ntire-2026-face-restoration/",
    "ukl.md": "/publications/ukl/",
}
LEGACY_FULL_URLS = (
    "/IConFace/",
    "https://cosmicrealm.github.io/style-talking/",
    "/year-archive/",
    "/publication/",
)
LEGACY_FORBIDDEN_PATHS = [
    "IConFace",
    "style-talking",
    "Projects",
    "generation-math",
    "generation-distillation",
    "TestPage",
    "foundations/diffusion-distillation-math-lab",
    "_portfolio",
    "_talks",
    "_teaching",
    "_drafts/post-draft.md",
    "_data/comments",
    "_data/authors.yml",
    "_data/cv.json",
    "markdown_generator",
    "talkmap",
    "talkmap.py",
    "talkmap.ipynb",
    "talkmap_out.ipynb",
    "CONTRIBUTING.md",
    "_pages/talkmap.html",
    "_pages/talks.html",
    "_pages/teaching.html",
    "_pages/portfolio.html",
    "_pages/markdown.md",
    "_layouts/talk.html",
    "_layouts/cv-layout.html",
    "_layouts/splash.html",
    "_includes/comment.html",
    "_includes/comments.html",
    "_includes/comments-providers",
    "_includes/cv-template.html",
    "_includes/feature_row",
    "_includes/gallery",
    "_includes/paginator.html",
    "_includes/archive-single-talk.html",
    "_includes/archive-single-talk-cv.html",
    "_includes/archive-single-cv.html",
    "scripts/cv_markdown_to_json.py",
    "scripts/update_cv_json.sh",
]
ZERO_REF_TARGETS = [
    "_includes/archive-single-talk-cv.html",
    "_includes/archive-single-cv.html",
    "images/editing-talk.png",
    "images/themes/homepage-air-dark.png",
    "images/themes/homepage-air-light.png",
    "images/themes/homepage-dark.png",
    "images/themes/homepage-light.png",
    "images/500x300.png",
    "images/bio-photo.jpg",
    "images/bio-photo-2.jpg",
]
FORBIDDEN_BUILD_OUTPUTS = [
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docker-compose.yaml",
    "docs",
    "scripts",
    "tests",
    "markdown_generator",
    "IConFace",
    "style-talking",
    "Projects",
    "generation-math",
    "generation-distillation",
    "TestPage",
    "publication",
    "year-archive",
    "talkmap",
    "talks",
    "teaching",
    "portfolio",
    "about",
    "foundations/diffusion-distillation-math-lab",
    "markdown",
    "cv-json",
    "resume",
]


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def exact_case_path(path: Path) -> Path | None:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return path if path.exists() else None

    current = REPO_ROOT
    if relative == Path("."):
        return current

    for part in relative.parts:
        try:
            next_path = next(candidate for candidate in current.iterdir() if candidate.name == part)
        except (FileNotFoundError, NotADirectoryError, StopIteration):
            return None
        current = next_path

    return current


def exact_case_exists(path: Path) -> bool:
    return exact_case_path(path) is not None


def front_matter(path: Path) -> dict[str, str]:
    text = read_text(path)
    if not text.startswith("---\n"):
        return {}
    lines = text.splitlines()
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line or line.startswith(" ") or line.startswith("\t") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def has_front_matter_key(path: Path, key: str) -> bool:
    text = read_text(path)
    if not text.startswith("---\n"):
        return False
    lines = text.splitlines()
    for line in lines[1:]:
        if line.strip() == "---":
            return False
        if line.startswith(f"{key}:"):
            return True
    return False


def tracked_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    result = run_git("ls-files", "--", str(path.relative_to(REPO_ROOT)))
    return len([line for line in result.stdout.splitlines() if line.strip()])


def total_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def iter_text_files() -> Iterable[Path]:
    for rel_dir in TEXT_SCAN_DIRS:
        base = REPO_ROOT / rel_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.is_relative_to(REPO_ROOT / ".git") or path.is_relative_to(REPO_ROOT / "_site"):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def find_refs(needle: str) -> list[str]:
    matches: set[str] = set()
    for path in iter_text_files():
        text = read_text(path)
        if needle in text:
            rel = path.relative_to(REPO_ROOT)
            matches.add(str(rel))
    return sorted(matches)


def git_check_ignore_verbose(path: str) -> subprocess.CompletedProcess[str]:
    return run_git("check-ignore", "-v", "--", path)


def tracked_files(prefix: Path) -> list[Path]:
    rel_prefix = str(prefix.relative_to(REPO_ROOT))
    result = run_git("ls-files", "--", rel_prefix)
    files: list[Path] = []
    for line in result.stdout.splitlines():
        if line.strip():
            files.append(REPO_ROOT / line.strip())
    return files


def check_dirty(site_dir: Path) -> list[str]:
    del site_dir
    expected = [
        " M foundations/image-generation-data-training/index.html",
        " M foundations/image-generation-data-training/static/css/index.css",
        "?? foundations/image-generation-data-training/static/img/",
    ]
    result = run_git("status", "--short", "--", "foundations/image-generation-data-training")
    actual = result.stdout.splitlines()
    failures: list[str] = []
    if actual != expected:
        failures.append(
            "git status --short mismatch: expected "
            + repr(expected)
            + ", got "
            + repr(actual)
        )
    return failures


def check_routes(site_dir: Path) -> list[str]:
    del site_dir
    failures: list[str] = []

    expected_present = [
        REPO_ROOT / "_pages/home.md",
        REPO_ROOT / "_pages/writing.html",
    ]
    expected_absent = [
        REPO_ROOT / "_pages/about.md",
        REPO_ROOT / "_pages/year-archive.html",
    ]
    for path in expected_present:
        if not path.exists():
            failures.append(f"missing route page: {path.relative_to(REPO_ROOT)}")
    for path in expected_absent:
        if path.exists():
            failures.append(f"legacy route page still exists: {path.relative_to(REPO_ROOT)}")

    for rel_path in ("_pages/home.md", "_pages/writing.html", "_pages/cv.md"):
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"missing page required for redirect_from audit: {rel_path}")
            continue
        if has_front_matter_key(path, "redirect_from"):
            failures.append(f"{rel_path} still declares redirect_from")

    writing_path = REPO_ROOT / "_pages/writing.html"
    if writing_path.exists():
        writing_meta = front_matter(writing_path)
        if writing_meta.get("title", "").strip("\"'") != "Writing":
            failures.append("_pages/writing.html title must be Writing")
        if writing_meta.get("permalink", "").strip("\"'") != "/writing/":
            failures.append("_pages/writing.html permalink must be /writing/")

    navigation_path = REPO_ROOT / "_data/navigation.yml"
    if navigation_path.exists():
        navigation_text = read_text(navigation_path)
        if re.search(r'title:\s*"Writing"\s*\n\s*url:\s*/writing/', navigation_text) is None:
            failures.append("_data/navigation.yml must expose Writing at /writing/")
    else:
        failures.append("missing _data/navigation.yml")

    tag_chip_path = REPO_ROOT / "_includes/tag-chip.html"
    if tag_chip_path.exists():
        tag_chip_text = read_text(tag_chip_path)
        if '"/writing/?tags="' not in tag_chip_text and "'/writing/?tags='" not in tag_chip_text:
            failures.append("_includes/tag-chip.html must route chips to /writing/?tags=")
    else:
        failures.append("missing _includes/tag-chip.html")

    projects_data_path = REPO_ROOT / "_data/projects.yml"
    if projects_data_path.exists():
        projects_text = read_text(projects_data_path)
        if "/projects/iconface/" not in projects_text:
            failures.append("_data/projects.yml must reference /projects/iconface/")
        if "/projects/style-talking/" not in projects_text:
            failures.append("_data/projects.yml must reference /projects/style-talking/")
    else:
        failures.append("missing _data/projects.yml")

    publication_dir = REPO_ROOT / "_publications"
    publication_files = sorted(path for path in publication_dir.glob("*.md"))
    expected_names = set(EXPECTED_PUBLICATION_PERMALINKS)
    actual_names = {path.name for path in publication_files}
    if actual_names != expected_names:
        failures.append(f"_publications filenames must be lowercase set {sorted(expected_names)}")
    for filename, expected_permalink in EXPECTED_PUBLICATION_PERMALINKS.items():
        path = publication_dir / filename
        if not path.exists():
            failures.append(f"missing publication file: _publications/{filename}")
            continue
        meta = front_matter(path)
        if meta.get("permalink", "").strip("\"'") != expected_permalink:
            failures.append(f"_publications/{filename} permalink must be {expected_permalink}")

    iconface_publication = REPO_ROOT / "_publications/iconface.md"
    if iconface_publication.exists():
        iconface_text = read_text(iconface_publication)
        if "/projects/iconface/" not in iconface_text:
            failures.append("_publications/iconface.md must link to /projects/iconface/")
    else:
        failures.append("missing _publications/iconface.md")

    banned_text = LEGACY_FULL_URLS
    for needle in banned_text:
        refs = find_refs(needle)
        if refs:
            failures.append(f"legacy source reference {needle} still present in {', '.join(sorted(refs)[:8])}")

    return failures


def check_config(site_dir: Path) -> list[str]:
    del site_dir
    failures: list[str] = []
    config_path = REPO_ROOT / "_config.yml"
    config_text = read_text(config_path)

    if re.search(r'^url:\s*"https://cosmicrealm\.github\.io"\s*$', config_text, re.MULTILINE) is None:
        failures.append('_config.yml must contain single-line `url: "https://cosmicrealm.github.io"`')

    exclude_block = re.search(r"^exclude:\s*\n(?P<body>(?:^[ \t]+-.*\n?)*)", config_text, re.MULTILINE)
    exclude_items: set[str] = set()
    if exclude_block:
        for line in exclude_block.group("body").splitlines():
            item = line.strip()
            if item.startswith("-"):
                exclude_items.add(item[1:].strip().strip("\"'"))
    required_excludes = {
        "AGENTS.md",
        "docker-compose.yaml",
        "docs/",
        "scripts/",
        "tests/",
        "*.ipynb",
        "README.md",
        "package-lock.json",
    }
    missing_excludes = sorted(item for item in required_excludes if item not in exclude_items)
    if missing_excludes:
        failures.append(f"_config.yml exclude missing {missing_excludes}")

    forbidden_config_tokens = (
        "talkmap_link:",
        "comments:",
        "staticman:",
    )
    forbidden_plugin_tokens = (
        "jekyll-gist",
        "jekyll-paginate",
        "jekyll-redirect-from",
        "jemoji",
    )
    for token in forbidden_config_tokens:
        if token in config_text:
            failures.append(f"_config.yml still contains forbidden token {token}")

    gemfile_text = read_text(REPO_ROOT / "Gemfile")
    for required_gem in ("jekyll-feed", "jekyll-sitemap"):
        if required_gem not in gemfile_text:
            failures.append(f"Gemfile missing {required_gem}")
    for token in forbidden_plugin_tokens:
        if token in config_text:
            failures.append(f"_config.yml still contains forbidden plugin token {token}")
        if token in gemfile_text:
            failures.append(f"Gemfile still contains forbidden plugin token {token}")

    collections_block = re.search(r"^collections:\s*\n(?P<body>(?:^[ \t]+.*\n?)*)", config_text, re.MULTILINE)
    collections_body = collections_block.group("body") if collections_block else ""
    for forbidden_collection in ("talks:", "teaching:", "portfolio:"):
        if forbidden_collection in collections_body:
            failures.append(f"_config.yml collections still contains {forbidden_collection[:-1]}")

    return failures


def check_legacy(site_dir: Path) -> list[str]:
    del site_dir
    failures: list[str] = []

    for rel_path in LEGACY_FORBIDDEN_PATHS:
        if exact_case_exists(REPO_ROOT / rel_path):
            failures.append(f"legacy path still exists: {rel_path}")

    single_layout_path = REPO_ROOT / "_layouts/single.html"
    if single_layout_path.exists() and "comments.html" in read_text(single_layout_path):
        failures.append("_layouts/single.html still references comments")

    for rel_path in ZERO_REF_TARGETS:
        if exact_case_exists(REPO_ROOT / rel_path):
            failures.append(f"legacy asset/include still exists: {rel_path}")

    scan_candidates = list(iter_text_files()) + [REPO_ROOT / rel_path for rel_path in ZERO_REF_SCAN_FILES]
    for rel_path in ZERO_REF_TARGETS:
        name = Path(rel_path).name
        refs: set[str] = set()
        for path in scan_candidates:
            if not path.exists() or not path.is_file():
                continue
            if path.relative_to(REPO_ROOT).as_posix() == rel_path:
                continue
            if name in read_text(path):
                refs.add(str(path.relative_to(REPO_ROOT)))
        if refs:
            failures.append(f"legacy asset/include reference {name} still present in {', '.join(sorted(refs)[:8])}")

    return failures


def check_projects(site_dir: Path) -> list[str]:
    del site_dir
    failures: list[str] = []

    iconface_dir = REPO_ROOT / "projects/iconface"
    style_talking_dir = REPO_ROOT / "projects/style-talking"

    iconface_files = tracked_files(iconface_dir)
    iconface_tracked = len(iconface_files)
    if iconface_tracked != 1836:
        failures.append(f"projects/iconface tracked file count must be 1836, got {iconface_tracked}")
    if sum(path.stat().st_size for path in iconface_files if path.exists()) < 80_000_000:
        failures.append("projects/iconface must be at least 80MB tracked content")

    style_talking_files = tracked_files(style_talking_dir)
    style_talking_tracked = len(style_talking_files)
    if style_talking_tracked != 3:
        failures.append(f"projects/style-talking tracked file count must be 3, got {style_talking_tracked}")
    if sum(path.stat().st_size for path in style_talking_files if path.exists()) < 500_000:
        failures.append("projects/style-talking must be at least 500KB tracked content")

    ignore_probe = "projects/iconface/static/gallery/reference/ffhq-ref-severe/00078/lq.png"
    ignore_result = git_check_ignore_verbose(ignore_probe)
    if ignore_result.returncode != 0:
        failures.append(f"{ignore_probe} must stay ignored")
    elif "static/gallery/**/*.png" not in ignore_result.stdout:
        failures.append(f"{ignore_probe} ignore rule must come from static/gallery/**/*.png")

    expected_canonicals = {
        "projects/iconface/index.html": "https://cosmicrealm.github.io/projects/iconface/",
        "projects/style-talking/index.html": "https://cosmicrealm.github.io/projects/style-talking/",
    }
    old_urls = {
        "projects/iconface/index.html": "https://cosmicrealm.github.io/IConFace/",
        "projects/style-talking/index.html": "https://cosmicrealm.github.io/style-talking/",
    }
    for rel_path, canonical in expected_canonicals.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"missing project page: {rel_path}")
            continue
        text = read_text(path)
        if canonical not in text:
            failures.append(f"{rel_path} missing canonical {canonical}")
        if old_urls[rel_path] in text:
            failures.append(f"{rel_path} still references legacy production url {old_urls[rel_path]}")

    return failures


def check_build(site_dir: Path) -> list[str]:
    failures: list[str] = []

    required_paths = [
        site_dir / "index.html",
        site_dir / "writing/index.html",
        site_dir / "publications/iconface/index.html",
        site_dir / "projects/iconface/index.html",
        site_dir / "projects/style-talking/index.html",
    ]
    for path in required_paths:
        if not path.exists():
            failures.append(f"missing built artifact: {display_path(path)}")

    for rel_path in FORBIDDEN_BUILD_OUTPUTS:
        if (site_dir / rel_path).exists():
            failures.append(f"forbidden build output still exists: {display_path(site_dir / rel_path)}")

    return failures


CHECK_HANDLERS = {
    "routes": check_routes,
    "config": check_config,
    "legacy": check_legacy,
    "projects": check_projects,
    "build": check_build,
    "dirty": check_dirty,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify homepage architecture migration contracts.")
    parser.add_argument(
        "--check",
        dest="checks",
        action="append",
        required=True,
        choices=CHECK_CHOICES,
        help="Contract checks to run. Repeatable.",
    )
    parser.add_argument(
        "--site-dir",
        default="_site",
        help="Built site directory to validate for build checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    site_dir = (REPO_ROOT / args.site_dir).resolve()
    requested_checks = list(dict.fromkeys(args.checks))

    failures: list[str] = []
    for check_name in requested_checks:
        failures.extend(f"{check_name}: {msg}" for msg in CHECK_HANDLERS[check_name](site_dir))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1

    print(f"homepage architecture checks passed: {','.join(requested_checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
