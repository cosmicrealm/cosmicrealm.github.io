# Homepage Architecture Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize the personal-site source tree and public routes around `projects/`, `foundations/`, `publications/`, `writing/`, and `cv/`, while safely migrating `IConFace` and `style-talking`, tightening publish boundaries, and preserving the user's existing dirty Foundations work.

**Architecture:** Add one executable architecture contract first, then migrate canonical routes, prune template residue, move standalone project pages into `projects/<slug>/`, and centralize the shared MathJax vendor into `assets/vendor/mathjax/`. Validation is split into source-tree checks, git-status boundary checks, and a Docker-backed Jekyll build check so the reorganization can be proven without relying on visual-theme work.

**Tech Stack:** Jekyll, Docker Compose, Python 3 standard library, Node.js, YAML/Liquid/Markdown, static HTML/CSS/JS.

---

## File Structure And Responsibilities

- `scripts/verify_homepage_architecture.py`: the source-of-truth contract for route normalization, legacy cleanup, publish excludes, no-legacy-URL enforcement, dirty-worktree preservation, project migration safety counts, and built-site assertions.
- `_pages/home.md`, `_pages/writing.html`, `_data/navigation.yml`, `_includes/tag-chip.html`: canonical homepage and Writing route definitions.
- `_publications/*.md`, `_data/projects.yml`: canonical `publications` and `projects` URL surfaces.
- `_config.yml`, `Gemfile`, `README.md`, `package.json`, `.devcontainer/devcontainer.json`: site identity, publish boundary, and template-detachment metadata.
- `projects/iconface/`, `projects/style-talking/`, `scripts/projects/iconface/`, `docs/projects/iconface.md`: normalized standalone-project layout.
- `assets/vendor/mathjax/`, `foundations/generation-math/index.html`, `foundations/llm-mechanics/index.html`: shared local MathJax vendor path.
- Legacy deletions: `Projects/`, `generation-math/`, `generation-distillation/`, `TestPage/`, `foundations/diffusion-distillation-math-lab/`, `_portfolio/`, `_talks/`, `_teaching/`, template `_pages/*`, `markdown_generator/`, `talkmap/`, `talkmap.py`, `talkmap.ipynb`, `talkmap_out.ipynb`, `_drafts/post-draft.md`, `.github/workflows/scrape_talks.yml`, `.github/ISSUE_TEMPLATE/*`, `_data/authors.yml`, `_includes/cv-template.html`, the unused comment provider stack, unused `feature_row` / `gallery` / paginator includes, retired CV scripts/layouts, `_layouts/splash.html`, `CONTRIBUTING.md`, and confirmed-zero-reference template images.

## Non-Goals For This Plan

- No B-style visual redesign implementation.
- No shared theme-controller rewrite.
- No homepage typography/color overhaul.
- No repair of external paper, GitHub, or search-engine legacy links.

### Task 1: Add the failing architecture contract

**Files:**
- Create: `scripts/verify_homepage_architecture.py`
- Modify: `.gitignore`
- Test: `_config.yml`
- Test: `_pages/about.md`
- Test: `_pages/year-archive.html`
- Test: `_pages/cv.md`
- Test: `_publications/IConFace.md`
- Test: `_data/projects.yml`
- Test: `foundations/image-generation-data-training/`

- [ ] **Step 1: Write the verifier before touching routes or directories**

- [ ] **Step 2: Implement the full verifier, not partial snippets**

Use only the Python standard library plus `subprocess`. The script must be executable as written below and fail if any forbidden path still exists, if canonical pages/routes are missing, if `_config.yml` still exposes template plugins or talk/comment settings, if any `redirect_from:` remains in first-party pages, if the three user dirty paths change status, or if project tracked-file counts and ignore rules do not match the migrated state.

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = {"routes", "config", "legacy", "projects", "build", "dirty"}

EXPECTED_DIRTY = [
    "M foundations/image-generation-data-training/index.html",
    "M foundations/image-generation-data-training/static/css/index.css",
    "?? foundations/image-generation-data-training/static/img/",
]

EXPECTED_PROJECT_COUNTS = {
    "projects/iconface": {"tracked_files": 1836, "min_tracked_bytes": 80_000_000},
    "projects/style-talking": {"tracked_files": 3, "min_tracked_bytes": 500_000},
}

EXPECTED_PUBLICATION_PERMALINKS = {
    "_publications/dcpn.md": "/publications/dcpn/",
    "_publications/iconface.md": "/publications/iconface/",
    "_publications/ldcr.md": "/publications/ldcr/",
    "_publications/ntire-2026-face-restoration.md": "/publications/ntire-2026-face-restoration/",
    "_publications/ukl.md": "/publications/ukl/",
}

LEGACY_SOURCE_PATHS = [
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
    "_data/authors.yml",
    "_data/comments",
    "_data/cv.json",
    "_includes/cv-template.html",
    "_includes/comment.html",
    "_includes/comments.html",
    "_includes/comments-providers",
    "_includes/feature_row",
    "_includes/gallery",
    "_includes/paginator.html",
    "_includes/archive-single-talk.html",
    "_layouts/talk.html",
    "_layouts/cv-layout.html",
    "_layouts/splash.html",
    "markdown_generator",
    "talkmap",
    "talkmap.py",
    "talkmap.ipynb",
    "talkmap_out.ipynb",
    "scripts/cv_markdown_to_json.py",
    "scripts/update_cv_json.sh",
    "CONTRIBUTING.md",
]

REMOVE_AFTER_ZERO_REFS = [
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

REQUIRED_EXCLUDES = [
    "AGENTS.md",
    "docker-compose.yaml",
    "docs/",
    "scripts/",
    "tests/",
    "*.ipynb",
    "README.md",
    "package-lock.json",
]

FORBIDDEN_CONFIG_SNIPPETS = [
    "talkmap_link:",
    "comments:",
    "staticman:",
    "jekyll-gist",
    "jekyll-paginate",
    "jekyll-redirect-from",
    "jemoji",
]

FORBIDDEN_SOURCE_URLS = [
    "/IConFace/",
    "https://cosmicrealm.github.io/style-talking/",
    "/year-archive/",
    "/publication/",
]
SOURCE_SCAN_GLOBS = [
    "_pages",
    "_data",
    "_publications",
    "_includes",
    "_layouts",
    "_posts",
    "projects",
    "foundations",
]
FORBIDDEN_BUILT_PATHS = [
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
    "TestPage",
    "generation-math",
    "generation-distillation",
    "foundations/diffusion-distillation-math-lab",
    "talkmap",
    "talks",
    "teaching",
    "portfolio",
    "markdown",
    "cv-json",
    "publication",
    "year-archive",
    "about",
    "resume",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def git_lines(*args: str) -> list[str]:
    return subprocess.check_output(args, cwd=ROOT, text=True).splitlines()


def tracked_bytes(prefix: str) -> int:
    files = git_lines("git", "ls-files", prefix)
    return sum((ROOT / path).stat().st_size for path in files)


def front_matter_value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def zero_refs(paths: list[str]) -> None:
    haystacks = ["_layouts", "_includes", "_pages", "_data", "_posts", "_publications", "README.md", "_config.yml"]
    for target in paths:
        require(not (ROOT / target).exists(), f"template-only file still exists: {target}")
        result = subprocess.run(
            ["rg", "-n", re.escape(target.split("/")[-1]), *haystacks],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        require(result.returncode == 1, f"still referenced template asset: {target}")


def check_routes(_: Path) -> None:
    require((ROOT / "_pages/home.md").exists(), "missing canonical page: _pages/home.md")
    require((ROOT / "_pages/writing.html").exists(), "missing canonical page: _pages/writing.html")
    require(not (ROOT / "_pages/about.md").exists(), "legacy page still exists: _pages/about.md")
    require(not (ROOT / "_pages/year-archive.html").exists(), "legacy page still exists: _pages/year-archive.html")

    home = read_text("_pages/home.md")
    writing = read_text("_pages/writing.html")
    cv = read_text("_pages/cv.md")
    nav = read_text("_data/navigation.yml")
    tags = read_text("_includes/tag-chip.html")
    projects = read_text("_data/projects.yml")

    require("redirect_from:" not in home, "legacy redirect remains in _pages/home.md")
    require("redirect_from:" not in writing, "legacy redirect remains in _pages/writing.html")
    require("redirect_from:" not in cv, "legacy redirect remains in _pages/cv.md")
    require('title: "Writing"' in writing, "writing title not normalized")
    require("permalink: /writing/" in writing, "writing permalink not normalized")
    require('/writing/?tags=' in tags, "tag chip still points to year-archive")
    require('url: /writing/' in nav, "navigation still points to old writing route")
    require('/projects/iconface/' in projects, "IConFace project URL not normalized")
    require('/projects/style-talking/' in projects, "Style-Talking project URL not normalized")

    for rel_path, permalink in EXPECTED_PUBLICATION_PERMALINKS.items():
        text = read_text(rel_path)
        require(front_matter_value(text, "permalink") == permalink, f"publication permalink mismatch: {rel_path}")
    require("[Project](/projects/iconface/)" in read_text("_publications/iconface.md"), "IConFace publication still links to legacy project URL")
    for root in SOURCE_SCAN_GLOBS:
        for path in sorted((ROOT / root).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".html", ".yml", ".yaml", ".js", ".json", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in FORBIDDEN_SOURCE_URLS:
                require(needle not in text, f"forbidden legacy URL remains in source: {path.relative_to(ROOT)} -> {needle}")


def check_config(_: Path) -> None:
    config = read_text("_config.yml")
    gemfile = read_text("Gemfile")
    require('url: "https://cosmicrealm.github.io"' in config, "_config.yml url is not a single-line string")
    require("https://cosmicrealm.github.io #" not in config, "_config.yml still contains legacy url mapping form")
    for item in REQUIRED_EXCLUDES:
        require(item in config, f"missing required exclude: {item}")
    for snippet in FORBIDDEN_CONFIG_SNIPPETS:
        require(snippet not in config, f"forbidden config snippet remains: {snippet}")
        require(snippet not in gemfile, f"forbidden gem remains: {snippet}")
    require("gem 'jekyll-feed'" in gemfile, "jekyll-feed missing from Gemfile")
    require("gem 'jekyll-sitemap'" in gemfile, "jekyll-sitemap missing from Gemfile")
    require("gem 'jemoji'" not in gemfile, "jemoji still present in Gemfile")
    require("talks:" not in config and "teaching:" not in config and "portfolio:" not in config, "unused collections remain in _config.yml")


def check_legacy(_: Path) -> None:
    for rel in LEGACY_SOURCE_PATHS:
        require(not (ROOT / rel).exists(), f"legacy path still exists: {rel}")
    require("comments.html" not in read_text("_layouts/single.html"), "single layout still references retired comments UI")
    zero_refs(REMOVE_AFTER_ZERO_REFS)


def check_projects(_: Path) -> None:
    for rel, meta in EXPECTED_PROJECT_COUNTS.items():
        tracked = git_lines("git", "ls-files", rel)
        require(len(tracked) == meta["tracked_files"], f"tracked file count mismatch for {rel}: {len(tracked)}")
        require(tracked_bytes(rel) >= meta["min_tracked_bytes"], f"tracked bytes below floor for {rel}")
    ignore = git_lines("git", "check-ignore", "-v", "projects/iconface/static/gallery/reference/ffhq-ref-severe/00078/lq.png")
    require(ignore and "static/gallery/**/*.png" in ignore[0], "iconface gallery ignore rule missing after move")

    iconface = read_text("projects/iconface/index.html")
    style = read_text("projects/style-talking/index.html")
    require("https://cosmicrealm.github.io/projects/iconface/" in iconface, "iconface canonical URL not updated")
    require("https://cosmicrealm.github.io/projects/style-talking/" in style, "style-talking canonical URL not updated")
    require("https://cosmicrealm.github.io/IConFace/" not in iconface, "iconface legacy URL still present")
    require("https://cosmicrealm.github.io/style-talking/" not in style, "style-talking legacy root URL still present")


def check_build(site_dir: Path) -> None:
    require((site_dir / "index.html").exists(), "built site missing index.html")
    require((site_dir / "writing" / "index.html").exists(), "built site missing /writing/")
    require((site_dir / "publications" / "iconface" / "index.html").exists(), "built site missing /publications/iconface/")
    require((site_dir / "projects" / "iconface" / "index.html").exists(), "built site missing /projects/iconface/")
    require((site_dir / "projects" / "style-talking" / "index.html").exists(), "built site missing /projects/style-talking/")
    for rel in FORBIDDEN_BUILT_PATHS:
        require(not (site_dir / rel).exists(), f"forbidden built path still exists: {rel}")


def check_dirty(_: Path) -> None:
    status = git_lines("git", "status", "--short", "--", "foundations/image-generation-data-training")
    require(status == EXPECTED_DIRTY, f"dirty Foundations paths changed: {status}")


CHECK_DISPATCH = {
    "routes": check_routes,
    "config": check_config,
    "legacy": check_legacy,
    "projects": check_projects,
    "build": check_build,
    "dirty": check_dirty,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="append", choices=sorted(CHECKS), required=True)
    parser.add_argument("--site-dir", default="_site")
    args = parser.parse_args()
    for name in args.check:
        CHECK_DISPATCH[name](ROOT / args.site_dir)
    print("homepage architecture checks passed:", ",".join(args.check))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the contract and verify the expected failure**

Run: `python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check dirty`

Expected: non-zero exit. The first failure should be one of the current structural violations such as `missing canonical page: _pages/home.md`, `legacy path still exists: IConFace`, `legacy redirect remains in _pages/cv.md`, or `publication permalink mismatch: _publications/iconface.md`.

- [ ] **Step 4: Add an ignored baseline area for dirty Foundations preservation**

Append the ignore entry if it is missing:

```gitignore
.superpowers/
```

Then create the baseline snapshot:

```bash
mkdir -p .superpowers/homepage-migration-baseline
cp foundations/image-generation-data-training/index.html \
  .superpowers/homepage-migration-baseline/image-generation-index.html
shasum -a 256 foundations/image-generation-data-training/static/css/index.css \
  > .superpowers/homepage-migration-baseline/image-generation-css.sha256
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path

root = Path('foundations/image-generation-data-training/static/img')
lines = []
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    lines.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
Path('.superpowers/homepage-migration-baseline/image-generation-img.sha256').write_text(
    '\n'.join(lines) + ('\n' if lines else ''),
    encoding='utf-8',
)
PY
```

Expected: the three baseline files exist under `.superpowers/homepage-migration-baseline/` and remain ignored.

- [ ] **Step 5: Run the verifier file itself through syntax and smoke checks**

Run:

```bash
python3 -m py_compile scripts/verify_homepage_architecture.py
python3 scripts/verify_homepage_architecture.py --check dirty
```

Expected: the compile step exits 0; the dirty-only step prints `homepage architecture checks passed: dirty`.

- [ ] **Step 6: Commit the contract**

```bash
git add scripts/verify_homepage_architecture.py .gitignore
git commit -m "test: define homepage architecture contract"
```

### Task 2: Normalize Home, Writing, Publications, and project data routes

**Files:**
- Move: `_pages/about.md` → `_pages/home.md`
- Move: `_pages/year-archive.html` → `_pages/writing.html`
- Modify: `_pages/cv.md`
- Modify: `_data/navigation.yml`
- Modify: `_includes/tag-chip.html`
- Move: `_publications/DCPN.md` → `_publications/dcpn.md`
- Move: `_publications/IConFace.md` → `_publications/iconface.md`
- Move: `_publications/LDCR.md` → `_publications/ldcr.md`
- Move: `_publications/NTIRE2026-Face-Restoration.md` → `_publications/ntire-2026-face-restoration.md`
- Move: `_publications/UKL.md` → `_publications/ukl.md`
- Modify: `_data/projects.yml`

- [ ] **Step 1: Rename the two canonical page sources**

Use `git mv` so history follows the canonical filenames:

```bash
git mv _pages/about.md _pages/home.md
git mv _pages/year-archive.html _pages/writing.html
```

Then edit front matter and route text:

```yaml
# _pages/home.md
permalink: /
title: "Jinyang Zhang"
seo_title: "Jinyang Zhang - Projects, Publications, Foundations, Writing, and CV"
excerpt: "Personal website for projects, publications, foundations, writing, and CV."
author_profile: true
```

```yaml
# _pages/writing.html
layout: archive
permalink: /writing/
title: "Writing"
author_profile: true
blog_tag_filter: true
```

Delete all `redirect_from:` blocks from `_pages/home.md`, `_pages/writing.html`, and `_pages/cv.md`. No legacy URL aliases remain in the final architecture.

- [ ] **Step 2: Rewrite homepage and navigation references from Blogs/year-archive to Writing/writing**

Update every first-party Writing entrypoint to use `/writing/`:

```liquid
<h2>Writing</h2>
<a class="section-heading__link" href="{{ '/writing/' | relative_url }}">All writing</a>
```

```yaml
- title: "Writing"
  url: /writing/
```

```liquid
{% assign tag_chip_url = "/writing/?tags=" | append: tag_chip_slug %}
```

- [ ] **Step 3: Normalize all publication permalinks and the IConFace project link**

Rename the five publication sources to lowercase and edit them to these exact permalinks:

```bash
git mv _publications/DCPN.md _publications/dcpn.md
git mv _publications/IConFace.md _publications/iconface.md
git mv _publications/LDCR.md _publications/ldcr.md
git mv _publications/NTIRE2026-Face-Restoration.md _publications/ntire-2026-face-restoration.md
git mv _publications/UKL.md _publications/ukl.md
```

```yaml
# _publications/dcpn.md
permalink: /publications/dcpn/

# _publications/iconface.md
permalink: /publications/iconface/
projecturl: '/projects/iconface/'

# _publications/ldcr.md
permalink: /publications/ldcr/

# _publications/ntire-2026-face-restoration.md
permalink: /publications/ntire-2026-face-restoration/

# _publications/ukl.md
permalink: /publications/ukl/
```

Also replace the IConFace body link:

```markdown
[Project](/projects/iconface/) / [arXiv](https://arxiv.org/abs/2605.02814)
```

- [ ] **Step 4: Normalize project-data links to future canonical project routes**

Edit `_data/projects.yml` so the two standalone project entries already point to their destination routes:

```yaml
- name: Style-Talking
  teaser: /projects/style-talking/static/images/teaser.jpg
  links:
    - label: Project
      url: /projects/style-talking/

- name: IConFace
  links:
    - label: Project
      url: /projects/iconface/
```

- [ ] **Step 5: Run focused route checks and commit**

Run: `python3 scripts/verify_homepage_architecture.py --check routes`

Expected: `homepage architecture checks passed: routes`

```bash
git add \
  _pages/home.md \
  _pages/writing.html \
  _pages/cv.md \
  _data/navigation.yml \
  _includes/tag-chip.html \
  _publications/dcpn.md \
  _publications/iconface.md \
  _publications/ldcr.md \
  _publications/ntire-2026-face-restoration.md \
  _publications/ukl.md \
  _data/projects.yml
git commit -m "feat: normalize homepage, writing, and publication routes"
```

### Task 3: Tighten publish boundaries and remove template residue

**Files:**
- Modify: `_config.yml`
- Modify: `_layouts/single.html`
- Modify: `Gemfile`
- Modify: `README.md`
- Modify: `package.json`
- Modify: `.gitignore`
- Modify: `.devcontainer/devcontainer.json`
- Delete: `CONTRIBUTING.md`
- Delete: `.github/workflows/scrape_talks.yml`
- Delete: `.github/ISSUE_TEMPLATE/feature_request.md`
- Delete: `.github/ISSUE_TEMPLATE/bug_report.md`
- Delete: `Projects/`
- Delete: `generation-math/`
- Delete: `generation-distillation/`
- Delete: `TestPage/`
- Delete: `foundations/diffusion-distillation-math-lab/`
- Delete: `_portfolio/`
- Delete: `_talks/`
- Delete: `_teaching/`
- Delete: `_drafts/post-draft.md`
- Delete: `_pages/archive-layout-with-content.md`
- Delete: `_pages/collection-archive.html`
- Delete: `_pages/cv-json.md`
- Delete: `_pages/markdown.md`
- Delete: `_pages/non-menu-page.md`
- Delete: `_pages/page-archive.html`
- Delete: `_pages/portfolio.html`
- Delete: `_pages/talkmap.html`
- Delete: `_pages/talks.html`
- Delete: `_pages/teaching.html`
- Delete: `_pages/terms.md`
- Delete: `markdown_generator/`
- Delete: `talkmap/`
- Delete: `talkmap.py`
- Delete: `talkmap.ipynb`
- Delete: `talkmap_out.ipynb`
- Delete: `_data/authors.yml`
- Delete: `_data/comments/`
- Delete: `_data/cv.json`
- Delete: `_includes/cv-template.html`
- Delete: `_includes/comment.html`
- Delete: `_includes/comments.html`
- Delete: `_includes/comments-providers/`
- Delete: `_includes/feature_row`
- Delete: `_includes/gallery`
- Delete: `_includes/paginator.html`
- Delete: `scripts/cv_markdown_to_json.py`
- Delete: `scripts/update_cv_json.sh`
- Delete: `_layouts/talk.html`
- Delete: `_layouts/cv-layout.html`
- Delete: `_layouts/splash.html`
- Delete: `_includes/archive-single-talk.html`

- [ ] **Step 1: Rewrite `_config.yml` around the real site, not the Academic Pages template**

Use a single-line `url`, keep `baseurl: ""`, remove unused collections, remove `comments`, `staticman`, `talkmap_link`, all first-party redirect/plugin residues, and expand `exclude` for non-site artifacts:

```yaml
url: "https://cosmicrealm.github.io"
repository: "cosmicrealm/cosmicrealm.github.io"

exclude:
  - AGENTS.md
  - CONTRIBUTING.md
  - docker-compose.yaml
  - docs/
  - scripts/
  - tests/
  - "*.ipynb"
  - .github
  - knowledge
  - node_modules
  - package.json
  - package-lock.json
  - README.md
```

```yaml
collections:
  publications:
    output: true
    permalink: /:collection/:path/
```

```yaml
defaults:
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: false
      related: true
  - scope:
      path: ""
      type: pages
    values:
      layout: single
      author_profile: true
  - scope:
      path: ""
      type: publications
    values:
      layout: single
      author_profile: true
```

```yaml
plugins:
  - jekyll-feed
  - jekyll-sitemap

whitelist:
  - jekyll-feed
  - jekyll-sitemap
```

- [ ] **Step 2: Remove redirect-based and template-only Ruby metadata**

After Task 2 there should be no remaining first-party `redirect_from` use. Remove every unused template plugin from `Gemfile` and keep only the two still-used site plugins:

```ruby
group :jekyll_plugins do
  gem 'jekyll'
  gem 'jekyll-feed'
  gem 'jekyll-sitemap'
  gem 'webrick', '~> 1.8'
end
```

- [ ] **Step 3: Replace the upstream repository identity metadata**

Change repository identity fields, but in `.devcontainer/devcontainer.json` edit only the `name` key and leave every other field untouched.

```json
{
  "name": "cosmicrealm-homepage",
  "description": "Jinyang Zhang personal homepage build scripts",
  "repository": {
    "type": "git",
    "url": "https://github.com/cosmicrealm/cosmicrealm.github.io"
  },
  "bugs": {
    "url": "https://github.com/cosmicrealm/cosmicrealm.github.io/issues"
  },
  "homepage": "https://cosmicrealm.github.io"
}
```

```json
{
  "name": "COSMICREALM HOMEPAGE"
}
```

Rewrite `README.md` into a project-local site README with complete content, not a placeholder outline:

````markdown
# cosmicrealm.github.io

Jinyang Zhang 的个人主页仓库，基于 Jekyll 维护 `Projects`、`Foundations`、`Publications`、`Writing` 和 `CV` 五类公开内容。

## Local Preview

站点默认通过仓库内的 Docker Compose 环境运行：

```bash
docker compose up
```

服务启动后访问 `http://localhost:4000/`。修改 `_config.yml`、Gemfile 或布局文件后需要重启容器。

## Content Namespaces

- `projects/`: 独立项目页与自包含静态资源，canonical URL 为 `/projects/<slug>/`
- `foundations/`: 原理讲义、交互式学习页与数学推导，canonical URL 为 `/foundations/<slug>/`
- `_publications/`: 论文元数据与详情页，canonical URL 为 `/publications/<slug>/`
- `_posts/`: 写作归档，主入口为 `/writing/`
- `_pages/`: 首页、列表页、CV、404 等主站入口

## Validation

结构重组后先运行：

```bash
python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check dirty
docker compose up -d --build
python3 scripts/verify_homepage_architecture.py --check build --site-dir _site
docker compose down
```

## Editing Rules

- 不在根目录新增单项目页面或 redirect 壳。
- 新项目页进入 `projects/<slug>/`，新讲义页进入 `foundations/<slug>/`。
- 工程脚本、文档、测试和 notebook 不能发布到生成站点。
````

- [ ] **Step 4: Delete every confirmed template-only source path**

Before deleting residue, stop ignoring `package-lock.json` so a later theme/UI pass can commit a lockfile after adding browser-QA dependencies such as `playwright`. Do not generate a new lockfile in this architecture pass unless `package.json` actually changes dependencies.

```gitignore
-package-lock.json
```

Remove the comments block from `_layouts/single.html`; with comments disabled globally and no page-level opt-ins, no comments include remains reachable. Then remove the confirmed template-only paths and explicitly prove the conditional template files and images are unreferenced before deleting them:

```bash
git rm -r \
  .github/ISSUE_TEMPLATE \
  .github/workflows/scrape_talks.yml \
  CONTRIBUTING.md \
  Projects \
  TestPage \
  foundations/diffusion-distillation-math-lab \
  _portfolio \
  _talks \
  _teaching \
  _drafts/post-draft.md \
  _data/authors.yml \
  _data/comments \
  _includes/comments-providers \
  generation-distillation \
  generation-math \
  markdown_generator \
  talkmap

git rm \
  _data/cv.json \
  _includes/comment.html \
  _includes/comments.html \
  _includes/cv-template.html \
  _includes/feature_row \
  _includes/gallery \
  _includes/paginator.html \
  _pages/archive-layout-with-content.md \
  _pages/collection-archive.html \
  _pages/cv-json.md \
  _pages/markdown.md \
  _pages/non-menu-page.md \
  _pages/page-archive.html \
  _pages/portfolio.html \
  _pages/talkmap.html \
  _pages/talks.html \
  _pages/teaching.html \
  _pages/terms.md \
  _layouts/talk.html \
  _layouts/cv-layout.html \
  _layouts/splash.html \
  _includes/archive-single-talk.html \
  scripts/cv_markdown_to_json.py \
  scripts/update_cv_json.sh \
  talkmap.ipynb \
  talkmap.py \
  talkmap_out.ipynb

rg -n 'archive-single-talk-cv|archive-single-cv|editing-talk.png|homepage-air-dark|homepage-air-light|homepage-dark|homepage-light|500x300.png|bio-photo.jpg|bio-photo-2.jpg' \
  _layouts _includes _pages _data _posts _publications README.md _config.yml
```

Expected: the final `rg` exits 1 with no matches.

Then delete the now-unreferenced template-only files:

```bash
git rm \
  _includes/archive-single-talk-cv.html \
  _includes/archive-single-cv.html \
  images/editing-talk.png \
  images/themes/homepage-air-dark.png \
  images/themes/homepage-air-light.png \
  images/themes/homepage-dark.png \
  images/themes/homepage-light.png \
  images/500x300.png \
  images/bio-photo.jpg \
  images/bio-photo-2.jpg
```

- [ ] **Step 5: Run config and legacy checks, then commit**

Run: `python3 scripts/verify_homepage_architecture.py --check config --check legacy --check dirty`

Expected: `homepage architecture checks passed: config,legacy,dirty`

```bash
git add _config.yml _layouts/single.html Gemfile README.md package.json .gitignore .devcontainer/devcontainer.json
git commit -m "refactor: remove template residue and tighten publish boundaries"
```

### Task 4: Safely migrate `IConFace` and `style-talking` into `projects/`

**Files:**
- Modify: `_data/projects.yml`
- Modify: `_publications/iconface.md`
- Create: `projects/iconface/`
- Create: `projects/style-talking/`
- Create: `scripts/projects/iconface/`
- Create: `docs/projects/iconface.md`
- Modify: `.gitignore`
- Modify: `projects/iconface/index.html`
- Modify: `projects/style-talking/index.html`
- Delete: `IConFace/.nojekyll`
- Move: `IConFace/scripts/sync_paper_assets.py` → `scripts/projects/iconface/sync_paper_assets.py`
- Move: `IConFace/README.md` → `docs/projects/iconface.md`
- Move: `IConFace/` → `projects/iconface/`
- Move: `style-talking/` → `projects/style-talking/`

- [ ] **Step 1: Record the pre-migration safety baseline for tracked and ignored files**

Run these commands before any move and keep the numbers in the commit message notes or work log:

```bash
git ls-files IConFace | wc -l
python3 - <<'PY'
import pathlib, subprocess
paths = subprocess.check_output(['git', 'ls-files', 'IConFace'], text=True).splitlines()
print(sum(pathlib.Path(p).stat().st_size for p in paths))
PY
git check-ignore -v IConFace/static/gallery/reference/ffhq-ref-severe/00078/lq.png

git ls-files style-talking | wc -l
python3 - <<'PY'
import pathlib, subprocess
paths = subprocess.check_output(['git', 'ls-files', 'style-talking'], text=True).splitlines()
print(sum(pathlib.Path(p).stat().st_size for p in paths))
PY
```

Expected:

```text
1839
82164146
IConFace/.gitignore:2:static/gallery/**/*.png IConFace/static/gallery/reference/ffhq-ref-severe/00078/lq.png
3
549073
```

- [ ] **Step 2: Move the project directories on the same filesystem and extract non-runtime files**

Use `git mv` only; do not copy directories:

```bash
mkdir -p projects scripts/projects/iconface docs/projects
git mv IConFace projects/iconface
git mv style-talking projects/style-talking
git mv projects/iconface/scripts/sync_paper_assets.py scripts/projects/iconface/sync_paper_assets.py
git mv projects/iconface/README.md docs/projects/iconface.md
git rm projects/iconface/.nojekyll
rmdir projects/iconface/scripts
```

- [ ] **Step 3: Rewrite the two standalone project pages to their canonical URLs**

First repair the relocated helper so it still writes to the project runtime and the repository-level teaser images:

```python
# scripts/projects/iconface/sync_paper_assets.py
repo_root = Path(__file__).resolve().parents[3]
site_root = repo_root / "projects/iconface"
output_root = site_root / "static/gallery/v2"
image_root = site_root / "static/images/v2"

# near the final teaser export
make_square_preview(square_source, repo_root / "images/projects/iconface.jpg")
make_square_preview(square_source, repo_root / "images/publications/iconface.jpg")
```

Update `docs/projects/iconface.md` to use the canonical URL and helper command:

````markdown
- Project: https://cosmicrealm.github.io/projects/iconface/

```bash
python3 scripts/projects/iconface/sync_paper_assets.py \
  --paper-root /Users/zhangjinyang/code/local/flux-restoration/paper_submission_iconface
```
````

Move general Python cache ignores to the root and update the Style-Talking video path:

```gitignore
__pycache__/
*.py[cod]
projects/style-talking/static/videos/
```

Remove `scripts/__pycache__/` from `projects/iconface/.gitignore`; keep every gallery/image ignore rule unchanged.

Then update the two standalone pages.

Update `projects/iconface/index.html`:

```html
<meta property="og:url" content="https://cosmicrealm.github.io/projects/iconface/">
<meta property="og:image" content="https://cosmicrealm.github.io/projects/iconface/static/images/social_preview.jpg">
<meta name="twitter:image" content="https://cosmicrealm.github.io/projects/iconface/static/images/social_preview.jpg">
<link rel="canonical" href="https://cosmicrealm.github.io/projects/iconface/">
<a class="project-home-link" href="https://cosmicrealm.github.io/" aria-label="Go to Jinyang Zhang's homepage">← Home</a>
```

```json
"url": "https://cosmicrealm.github.io/projects/iconface/",
"image": "https://cosmicrealm.github.io/projects/iconface/static/images/social_preview.jpg"
```

Update `projects/style-talking/index.html`:

```html
<meta property="og:url" content="https://cosmicrealm.github.io/projects/style-talking/">
<meta property="og:image" content="https://cosmicrealm.github.io/projects/style-talking/static/images/teaser.jpg">
<meta name="twitter:image" content="https://cosmicrealm.github.io/projects/style-talking/static/images/teaser.jpg">
<link rel="canonical" href="https://cosmicrealm.github.io/projects/style-talking/">
<link rel="icon" href="/images/favicon.ico">
<a class="project-home-link" href="https://cosmicrealm.github.io/" aria-label="Go to Jinyang Zhang homepage">
```

The verifier should also reject any remaining source occurrence of:

```text
https://cosmicrealm.github.io/IConFace/
https://cosmicrealm.github.io/style-talking/
/publication/
/year-archive/
```

- [ ] **Step 4: Verify that tracked counts and ignore rules survived the move**

Run: `python3 scripts/verify_homepage_architecture.py --check projects --check dirty`

Expected: `homepage architecture checks passed: projects,dirty`

Also re-run the raw safety commands on the migrated paths:

```bash
git ls-files projects/iconface | wc -l
python3 - <<'PY'
import pathlib, subprocess
paths = subprocess.check_output(['git', 'ls-files', 'projects/iconface'], text=True).splitlines()
total = sum(pathlib.Path(p).stat().st_size for p in paths)
assert total >= 80_000_000, total
print("iconface tracked bytes ok", total)
PY
git check-ignore -v projects/iconface/static/gallery/reference/ffhq-ref-severe/00078/lq.png

git ls-files projects/style-talking | wc -l
python3 - <<'PY'
import pathlib, subprocess
paths = subprocess.check_output(['git', 'ls-files', 'projects/style-talking'], text=True).splitlines()
total = sum(pathlib.Path(p).stat().st_size for p in paths)
assert total >= 500_000, total
print("style-talking tracked bytes ok", total)
PY
```

Expected after the canonical URL and helper edits:

```text
1836
iconface tracked bytes ok <value at least 80000000>
projects/iconface/.gitignore:2:static/gallery/**/*.png projects/iconface/static/gallery/reference/ffhq-ref-severe/00078/lq.png
3
style-talking tracked bytes ok <value at least 500000>
```

After editing, prove the two relocated tracked files still exist and that the only intentionally removed tracked file is `.nojekyll`:

```bash
test -s docs/projects/iconface.md
test -s scripts/projects/iconface/sync_paper_assets.py
test "$(git ls-files projects/iconface docs/projects/iconface.md scripts/projects/iconface | wc -l | tr -d ' ')" -eq 1838
```

- [ ] **Step 5: Commit the safe migration**

```bash
git add .gitignore _data/projects.yml _publications/iconface.md projects scripts/projects/iconface docs/projects/iconface.md
git commit -m "refactor: move standalone project pages under projects"
```

### Task 5: Centralize the shared MathJax vendor under `assets/vendor/`

**Files:**
- Create: `assets/vendor/mathjax/`
- Modify: `foundations/aigc-llm-math/index.html`
- Modify: `foundations/generation-acceleration/index.html`
- Modify: `foundations/generation-distillation/index.html`
- Modify: `foundations/generation-math/index.html`
- Modify: `foundations/image-generation-data-training/index.html`
- Modify: `foundations/leetcode-hot100/index.html`
- Modify: `foundations/llm-interview-qa/index.html`
- Modify: `foundations/llm-mechanics/index.html`
- Modify: `foundations/video-generation/index.html`
- Modify: `scripts/check_foundations_layout.py`
- Modify: `scripts/verify_llm_mechanics.py`
- Delete: `foundations/generation-math/static/vendor/mathjax/`
- Delete: `foundations/llm-mechanics/static/vendor/mathjax/`

- [ ] **Step 1: Move one canonical MathJax copy into shared assets**

Use the `generation-math` copy as the canonical source and remove the duplicate later:

```bash
mkdir -p assets/vendor
git mv foundations/generation-math/static/vendor/mathjax assets/vendor/mathjax
```

- [ ] **Step 2: Repoint the eight clean Foundations pages to the shared vendor path**

Edit the MathJax script tag only; do not change theme, prose, labs, or layout:

```html
<script defer src="/assets/vendor/mathjax/tex-mml-chtml.js" id="MathJax-script"></script>
```

Apply it to these exact clean entrypoints. Do not edit `foundations/image-generation-data-training/index.html` here; Step 4 alone owns that dirty working file.

```text
foundations/aigc-llm-math/index.html
foundations/generation-acceleration/index.html
foundations/generation-distillation/index.html
foundations/generation-math/index.html
foundations/leetcode-hot100/index.html
foundations/llm-interview-qa/index.html
foundations/llm-mechanics/index.html
foundations/video-generation/index.html
```

The surrounding `window.MathJax = { ... }` config stays local to each page. Update `scripts/check_foundations_layout.py` and `scripts/verify_llm_mechanics.py` so their path assertions expect `/assets/vendor/mathjax/tex-mml-chtml.js`.

- [ ] **Step 3: Remove the duplicate vendor tree from `llm-mechanics`**

After the script tag points to `/assets/vendor/mathjax/`, delete the duplicate copy:

```bash
git rm -r foundations/llm-mechanics/static/vendor/mathjax
```

- [ ] **Step 4: Replace the dirty Foundations working-tree path and verify the baseline invariants**

```bash
python3 - <<'PY'
from pathlib import Path

path = Path('foundations/image-generation-data-training/index.html')
text = path.read_text(encoding='utf-8')
old = '../generation-math/static/vendor/mathjax/tex-mml-chtml.js'
new = '/assets/vendor/mathjax/tex-mml-chtml.js'
count = text.count(old)
assert count == 1, f"expected exactly one old MathJax path, got {count}"
path.write_text(text.replace(old, new), encoding='utf-8')
PY
shasum -a 256 -c .superpowers/homepage-migration-baseline/image-generation-css.sha256
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path

root = Path('foundations/image-generation-data-training/static/img')
current = []
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    current.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
expected = Path('.superpowers/homepage-migration-baseline/image-generation-img.sha256').read_text().splitlines()
assert current == expected, 'dirty static/img manifest changed'
PY
python3 - <<'PY'
from pathlib import Path

baseline = Path('.superpowers/homepage-migration-baseline/image-generation-index.html').read_text(encoding='utf-8')
current = Path('foundations/image-generation-data-training/index.html').read_text(encoding='utf-8')
normalized = current.replace('/assets/vendor/mathjax/tex-mml-chtml.js', '../generation-math/static/vendor/mathjax/tex-mml-chtml.js')
assert normalized == baseline, 'working-tree dirty index changed beyond the MathJax path'
PY
```

Expected: CSS hash is `OK`, the image-manifest assertion passes, and the normalized dirty HTML matches the baseline copy exactly.

- [ ] **Step 5: Stage the dirty Foundations HTML by synthetic HEAD-derived blob only**

```bash
tempdir="$(mktemp -d)"
git show HEAD:foundations/image-generation-data-training/index.html \
  > "$tempdir/image-generation-data-training.head.html"
TEMP_HEAD_HTML="$tempdir/image-generation-data-training.head.html" python3 - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["TEMP_HEAD_HTML"])
text = path.read_text(encoding="utf-8")
old = '../generation-math/static/vendor/mathjax/tex-mml-chtml.js'
new = '/assets/vendor/mathjax/tex-mml-chtml.js'
count = text.count(old)
assert count == 1, f"expected exactly one MathJax path, got {count}"
path.write_text(text.replace(old, new), encoding="utf-8")
PY
mode="$(git ls-files -s foundations/image-generation-data-training/index.html | awk '{print $1}')"
blob="$(git hash-object -w "$tempdir/image-generation-data-training.head.html")"
git update-index --cacheinfo "$mode,$blob,foundations/image-generation-data-training/index.html"
git diff --cached -- foundations/image-generation-data-training/index.html
git status --short -- foundations/image-generation-data-training/index.html
```

Do not run `git add foundations/image-generation-data-training/index.html`. The staged content comes only from `git update-index`.

Expected:

```text
git diff --cached -- foundations/image-generation-data-training/index.html
# shows exactly one replaced script line

git status --short -- foundations/image-generation-data-training/index.html
MM foundations/image-generation-data-training/index.html
```

- [ ] **Step 6: Run focused checks that do not inspect dirty status**

Run: `python3 scripts/verify_homepage_architecture.py --check routes --check projects`

Expected: `homepage architecture checks passed: routes,projects`

Run: `python3 scripts/check_foundations_layout.py && python3 scripts/verify_llm_mechanics.py`

Expected: both existing verifiers pass.

Run: `rg -n 'generation-math/static/vendor/mathjax|llm-mechanics/static/vendor/mathjax' foundations/*/index.html`

Expected: exit 1 with no matches.

- [ ] **Step 7: Commit the vendor centralization without staging user CSS/img**

Before committing, prove the user CSS and image paths are still unstaged. Exit 1 with no output is the expected result:

```bash
git diff --cached --name-only | rg 'foundations/image-generation-data-training/static/css|foundations/image-generation-data-training/static/img'
```

```bash
git add \
  assets/vendor/mathjax \
  foundations/aigc-llm-math/index.html \
  foundations/generation-acceleration/index.html \
  foundations/generation-distillation/index.html \
  foundations/generation-math/index.html \
  foundations/leetcode-hot100/index.html \
  foundations/llm-interview-qa/index.html \
  foundations/llm-mechanics/index.html \
  foundations/video-generation/index.html \
  scripts/check_foundations_layout.py \
  scripts/verify_llm_mechanics.py
git commit -m "refactor: centralize shared MathJax vendor assets"
```

After committing, the worktree should return to the normal dirty shape for this Foundations area:

```bash
git status --short -- foundations/image-generation-data-training
python3 scripts/verify_homepage_architecture.py --check dirty
shasum -a 256 -c .superpowers/homepage-migration-baseline/image-generation-css.sha256
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path

root = Path('foundations/image-generation-data-training/static/img')
current = []
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    current.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
expected = Path('.superpowers/homepage-migration-baseline/image-generation-img.sha256').read_text().splitlines()
assert current == expected, 'post-commit static/img manifest changed'
PY
python3 - <<'PY'
from pathlib import Path

baseline = Path('.superpowers/homepage-migration-baseline/image-generation-index.html').read_text(encoding='utf-8')
current = Path('foundations/image-generation-data-training/index.html').read_text(encoding='utf-8')
normalized = current.replace('/assets/vendor/mathjax/tex-mml-chtml.js', '../generation-math/static/vendor/mathjax/tex-mml-chtml.js')
assert normalized == baseline, 'post-commit dirty working tree drifted from baseline'
PY
```

Expected:

```text
M foundations/image-generation-data-training/index.html
M foundations/image-generation-data-training/static/css/index.css
?? foundations/image-generation-data-training/static/img/
homepage architecture checks passed: dirty
foundations/image-generation-data-training/static/css/index.css: OK
```

### Task 6: Build the site and validate generated artifacts

**Files:**
- Test: `scripts/verify_homepage_architecture.py`
- Test: `_site/`

- [ ] **Step 1: Run the full source-tree contract before the Docker build**

Run: `python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check dirty`

Expected: `homepage architecture checks passed: routes,config,legacy,projects,dirty`

- [ ] **Step 2: Build and serve the site through the repository Docker setup**

Run:

```bash
docker compose up -d --build
docker compose logs --tail=40 jekyll-site
```

Expected log evidence includes both of these lines:

```text
Server address: http://0.0.0.0:4000
Server running... press ctrl-c to stop.
```

- [ ] **Step 3: Verify canonical routes and removed routes over HTTP**

Run:

```bash
curl -I http://127.0.0.1:4000/
curl -I http://127.0.0.1:4000/writing/
curl -I http://127.0.0.1:4000/publications/iconface/
curl -I http://127.0.0.1:4000/projects/iconface/
curl -I http://127.0.0.1:4000/projects/style-talking/
curl -I http://127.0.0.1:4000/IConFace/
curl -I http://127.0.0.1:4000/style-talking/
curl -I http://127.0.0.1:4000/year-archive/
curl -I http://127.0.0.1:4000/publication/iconface/
curl -I http://127.0.0.1:4000/talks/
curl -I http://127.0.0.1:4000/teaching/
curl -I http://127.0.0.1:4000/portfolio/
curl -I http://127.0.0.1:4000/markdown/
curl -I http://127.0.0.1:4000/cv-json/
curl -I http://127.0.0.1:4000/about/
curl -I http://127.0.0.1:4000/resume
```

Expected:

```text
HTTP/1.1 200 OK   # for /, /writing/, /publications/iconface/, /projects/iconface/, /projects/style-talking/
HTTP/1.1 404 Not Found   # for /IConFace/, /style-talking/, /year-archive/, /publication/iconface/, /talks/, /teaching/, /portfolio/, /markdown/, /cv-json/, /about/, /resume
```

- [ ] **Step 4: Validate the generated `_site` tree**

Run: `python3 scripts/verify_homepage_architecture.py --check build --site-dir _site`

Expected: `homepage architecture checks passed: build`

This build-mode check must assert all of the following:

```python
require((site_dir / "projects" / "iconface" / "index.html").exists(), "missing built iconface page")
for rel in FORBIDDEN_BUILT_PATHS:
    require(not (site_dir / rel).exists(), f"forbidden built path still exists: {rel}")
```

- [ ] **Step 5: Tear down the container and assert there is no unexpected verification diff**

Run:

```bash
docker compose down
git diff -- scripts/verify_homepage_architecture.py
git diff --cached -- scripts/verify_homepage_architecture.py
```

Expected: both diff commands print nothing. If and only if you fixed `scripts/verify_homepage_architecture.py` during build debugging, then stage and commit that file in a separate follow-up commit.

## Self-Review

### Spec coverage

- `projects/<slug>/`, `publications/<slug>/`, `/writing/`, `_pages/home.md`: covered by Task 2 and Task 4.
- Publish excludes, template cleanup, repo identity cleanup: covered by Task 3.
- Safe `IConFace` / `style-talking` migration plus ignored-asset protection: covered by Task 4.
- Shared MathJax vendor under `assets/vendor/mathjax/`: covered by Task 5.
- Docker/Jekyll structural validation and built-site route checks: covered by Task 6.
- Explicit preservation of `foundations/image-generation-data-training/*` dirty state: enforced in Task 1 and reused in Tasks 3-6.
- Visual B theme implementation: intentionally excluded from this plan.

### Placeholder scan

- No `TBD`, `TODO`, or “similar to Task N” placeholders remain.
- Every task names exact files and exact commands.
- Every code-edit step includes concrete snippets rather than abstract instructions.

### Type and naming consistency

- Canonical routes are used consistently as `/projects/iconface/`, `/projects/style-talking/`, `/publications/<slug>/`, and `/writing/`.
- The verifier uses the same check names everywhere: `routes`, `config`, `legacy`, `projects`, `build`, `dirty`.
- The tracked-count safety baseline is consistent across pre/post migration checks: pre-migration `1839 / 82164146` for `IConFace`, post-migration `1836 / >= 80000000` for `projects/iconface`, and `3 / 549073` for `style-talking`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-19-homepage-architecture-migration.md`.

- Default execution mode: inline implementation against this plan, because the design/spec approval is already settled.
- Optional execution mode: subagent-per-task orchestration if the parent workflow explicitly chooses parallel task dispatch later.
