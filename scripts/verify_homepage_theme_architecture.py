import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REAL_FOUNDATIONS = [
    "foundations/aigc-llm-math/index.html",
    "foundations/generation-acceleration/index.html",
    "foundations/generation-distillation/index.html",
    "foundations/generation-math/index.html",
    "foundations/image-generation-data-training/index.html",
    "foundations/leetcode-hot100/index.html",
    "foundations/llm-interview-qa/index.html",
    "foundations/llm-mechanics/index.html",
    "foundations/video-generation/index.html",
]

REAL_PROJECTS = [
    "projects/iconface/index.html",
    "projects/style-talking/index.html",
]

LEGACY_FILES = [
    "assets/js/main.min.js",
    "assets/js/_main.js",
    "assets/js/theme.js",
    "assets/js/plugins/jquery.greedy-navigation.js",
    "assets/js/collapse.js",
]

SCAN_DIRS = [
    "_data",
    "_includes",
    "_layouts",
    "_pages",
    "_posts",
    "_sass",
    "assets",
    "foundations",
    "projects",
]

SCAN_FILES = [
    "_config.yml",
    "_config_docker.yml",
    "index.html",
]

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def read(rel):
    return (ROOT / rel).read_text()

head = read("_includes/head.html")
scripts = read("_includes/scripts.html")
seo = read("_includes/seo.html")
masthead = read("_includes/masthead.html")
base_path = read("_includes/base_path")
home = read("_pages/home.md")
writing = read("_pages/writing.html")
home_layout = read("_layouts/home.html")
single = read("_layouts/single.html")
archive = read("_layouts/archive.html")
tag_chip = read("_includes/tag-chip.html")
theme_tokens = read("assets/css/theme-tokens.css")
settings = read("_sass/theme/_settings.scss")
main_scss = read("assets/css/main.scss")
cosmicrealm = read("_sass/layout/_cosmicrealm.scss")
buttons = read("_sass/layout/_buttons.scss")
navigation = read("_sass/layout/_navigation.scss")
page_scss = read("_sass/layout/_page.scss")
projects_yml = read("_data/projects.yml")
config = read("_config.yml")
default_layout = read("_layouts/default.html")
author_profile = read("_includes/author-profile.html")
package = json.loads(read("package.json"))

require("assets/js/theme-init.js" in head, "head.html missing shared theme-init.js")
require("theme-controller.js" not in head, "head.html must not load theme-controller.js")
require("site.js" not in head, "head.html must not load site.js")
require(head.index("theme-init.js") < head.index("theme-tokens.css") < head.index("main.css"), "head asset order must be init -> tokens -> main CSS")
require('theme-controller.js' in scripts and 'defer' in scripts, "scripts.html missing deferred theme-controller.js")
require('site.js' in scripts and 'defer' in scripts, "scripts.html missing deferred site.js")
require("social-share" not in single, "single layout still renders share include")
require(not (ROOT / "_layouts/talk.html").exists(), "_layouts/talk.html should have been removed by the architecture migration")
require(not (ROOT / "_includes/footer/custom.html").exists(), "global footer renderer include still exists")
require("{% if page.mathjax %}" in scripts and "{% if page.mermaid %}" in scripts, "scripts.html must gate heavy renderers by page flags")
require("cdn.jsdelivr.net/npm/mathjax" not in scripts.lower(), "scripts.html still points to remote MathJax CDN")
require('data-theme="light"' in default_layout, "default layout lacks explicit light no-JS fallback")
require("/writing/?tags=" in tag_chip, "tag chip still points at old writing route")
require("URLSearchParams" not in writing and "applyFilter" not in writing, "writing page still owns duplicate inline filter runtime")
require("page.blog_tag_filter" in archive and "include sidebar.html" in archive, "archive layout no longer renders Writing filter sidebar")
require("layout: home" in home and "author_profile: false" in home, "home.md not on home layout")
require("Selected Work" in home and "Representative Publications" in home and "Foundations" in home and "Recent Writing" in home, "home.md missing homepage sections")
require("homepage:" in projects_yml and "homepage_order:" in projects_yml, "projects.yml missing homepage curation fields")
require("social:" in config and "type: Person" in config and "github.com/cosmicrealm" in config, "_config.yml missing non-empty Person social config")
require("package-lock.json" in config, "_config.yml must exclude package-lock.json from publish output")
require("site_theme" not in config, "_config.yml must not keep site_theme")
require('assign base_path = site.baseurl' in base_path and "site.url" not in base_path and "site.github" not in base_path, "base_path include not reduced to baseurl-only semantics")
require("twitter:" not in config.lower(), "_config.yml still contains X/Twitter configuration")
require("twitter" not in author_profile.lower() and "x (formerly" not in author_profile.lower(), "author profile still contains X/Twitter display copy")
require("--global-base-color" in theme_tokens, "theme-tokens.css missing global-base-color")
require("--global-fig-caption-color" in theme_tokens, "theme-tokens.css missing global-fig-caption-color")
for token in ["--cr-accent-strong", "--cr-warm", "--cr-page-bg", "--cr-card-bg", "--cr-panel-bg", "--cr-surface", "--cr-ink-soft", "--cr-border", "--cr-border-strong", "--cr-shadow", "--cr-shadow-hover", "--cr-font-serif", "--cr-font-sans", "--cr-font-mono"]:
    require(token in theme_tokens, f"theme-tokens.css missing {token}")
require(":root {" not in cosmicrealm, "_cosmicrealm.scss still owns token blocks")
for variable in ["$doc-font-size", "$type-size-1", "$global-font-family", "$small", "$x-large", "$susy", "$gray", "$danger-color", "$border-radius", "$masthead-height"]:
    require(variable in settings, f"_settings.scss missing compile-time variable {variable}")
require('"theme/settings"' in main_scss, "main.scss not importing shared theme settings")
require('"layout/tables"' in main_scss and '"layout/notices"' in main_scss and '"layout/sidebar"' in main_scss and '"syntax"' in main_scss, "main.scss dropped required layout imports")
require('"layout/json_cv"' not in main_scss and not (ROOT / "_sass/layout/_json_cv.scss").exists(), "retired JSON CV Sass remains")
require(not (ROOT / "_sass/_themes.scss").exists(), "legacy Sass settings owner still exists")
for text in [buttons, navigation, page_scss]:
    require("page__share" not in text.lower() and "social buttons" not in text.lower(), "share-specific CSS still present in Sass")
require("twitter:" not in seo and "facebook:" not in seo.lower(), "seo.html still contains network-specific meta")
require(re.search(r'"@type"\s*:\s*"Person"', seo), "seo.html must keep Person JSON-LD")
for rel in REAL_FOUNDATIONS + REAL_PROJECTS:
    html = read(rel)
    require("theme-init.js" in html, f"{rel} missing shared theme-init.js")
    require("theme-controller.js" in html and "site.js" in html, f"{rel} missing deferred runtime scripts")
    require("theme-tokens.css" in html, f"{rel} missing shared token stylesheet")
    require("content-theme-bridge.css" in html, f"{rel} missing bridge stylesheet")
    require('class="has-shared-theme"' in html, f"{rel} missing shared-theme html class")
    require("has-shared-theme-shell" in html, f"{rel} missing shared-theme shell class")
    require(html.count("data-theme-toggle") == 1, f"{rel} must contain exactly one theme toggle")
    require('name="twitter:' not in html.lower(), f"{rel} still contains Twitter-specific metadata")
for rel in LEGACY_FILES:
    require(not (ROOT / rel).exists(), f"{rel} still exists")

for dependency in ["jquery", "fitvids", "jquery-smooth-scroll", "plotly.js-dist-min", "onchange", "uglify-js"]:
    require(dependency not in package.get("dependencies", {}) and dependency not in package.get("devDependencies", {}), f"package.json still owns {dependency}")

def iter_source_files():
    for rel in SCAN_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        for ref in base.rglob("*"):
            if ref.is_file() and ref.suffix in {".html", ".js", ".scss", ".md", ".yml"}:
                yield ref
    for rel in SCAN_FILES:
        ref = ROOT / rel
        if ref.exists() and ref.is_file():
            yield ref

for ref in iter_source_files():
    if "assets/vendor" in ref.as_posix() or "/static/vendor/" in ref.as_posix():
        continue
    text = ref.read_text(encoding="utf-8", errors="ignore")
    for legacy in ["main.min.js", "jquery.greedy-navigation", "plotly.js-dist-min", "fitvids", "jquery-smooth-scroll"]:
        require(legacy not in text, f"{ref.relative_to(ROOT)} still references {legacy}")

for ref in list((ROOT / "assets/css").glob("*.css")) + list((ROOT / "_sass").rglob("*.scss")):
    if ref == ROOT / "assets/css/theme-tokens.css":
        continue
    text = ref.read_text(encoding="utf-8", errors="ignore")
    require(not re.search(r"--(?:global|cr)-[a-z0-9-]+\s*:", text), f"runtime token declaration escaped canonical owner: {ref.relative_to(ROOT)}")
