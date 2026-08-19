# Homepage B Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the personal homepage into the approved B-style editorial academic system with shared light/dark theming, a small native runtime, full homepage composition, no visible share UI, and one coherent theme protocol across the main site, `projects/iconface`, `projects/style-talking`, and the nine real Foundations pages.

**Architecture:** The Jekyll shell owns routing, homepage composition, section-page layout, navigation, footer, SEO, and the only theme token source. Every page, including standalone project/foundation HTML, loads one tiny synchronous `assets/js/theme-init.js` before CSS to avoid flash, then loads `assets/js/theme-controller.js` and `assets/js/site.js` with `defer`; only `site.js` mounts behavior after DOM parse. Light/dark CSS variables live only in `assets/css/theme-tokens.css`; Sass keeps compile-time variables in one shared `_sass/theme/_settings.scss` and stops redefining runtime tokens.

**Tech Stack:** Jekyll, Liquid, Sass, vanilla JavaScript, Python 3, Node.js built-in `assert`, Playwright as a tracked `devDependency`, system Chrome via `CHROME_PATH`, Docker Compose Jekyll preview.

**Prerequisite:** Complete `docs/superpowers/plans/2026-08-19-homepage-architecture-migration.md` first and start this plan from its clean index. This plan intentionally assumes the canonical `projects/iconface/`, `projects/style-talking/`, `_pages/home.md`, `_pages/writing.html`, lowercase publication files, centralized `assets/vendor/mathjax/`, and deleted template/CV-json/talk residue already exist.

**Upstream posture:** This plan does **not** migrate the whole site onto Academic Pages, Minimal Mistakes, al-folio, or minimal-light as a new base theme. The current repository has already diverged heavily with custom Foundations pages and standalone project microsites, so whole-theme migration would cost more than it saves. Instead, it selectively borrows: Minimal Mistakes' token/layout layering (`https://github.com/mmistakes/minimal-mistakes`), al-folio's page-level opt-in runtime idea (`https://github.com/alshedivat/al-folio`), minimal-light's restrained homepage / dual-theme sensibility (`https://github.com/yaoyao-liu/minimal-light`), and Academic Pages' collection-oriented academic content framing (`https://github.com/academicpages/academicpages.github.io`). The implementation target remains this repository's existing Jekyll site, not an upstream theme replacement.

---

## File Structure Map

**Global config, data, and runtime**

- Create: `assets/js/theme-init.js`
- Create: `assets/js/theme-controller.js`
- Create: `assets/js/site.js`
- Create: `assets/js/mermaid-init.js`
- Create: `assets/css/theme-tokens.css`
- Create: `assets/css/content-theme-bridge.css`
- Move: `_sass/_themes.scss` -> `_sass/theme/_settings.scss`
- Modify: `_config.yml`
- Modify: `package.json`
- Modify: `package-lock.json`
- Modify: `_data/projects.yml`
- Delete: `assets/js/main.min.js`
- Delete: `assets/js/_main.js`
- Delete: `assets/js/theme.js`
- Delete: `assets/js/plugins/jquery.greedy-navigation.js`
- Delete: `assets/js/collapse.js`
- Delete: `_includes/social-share.html`
- Delete: `_sass/theme/_default_light.scss`
- Delete: `_sass/theme/_default_dark.scss`
- Delete: `_sass/theme/_air_light.scss`
- Delete: `_sass/theme/_air_dark.scss`
- Delete: `_sass/layout/_json_cv.scss`

**Templates, includes, and layouts**

- Create: `_layouts/home.html`
- Create: `_includes/home-project-card.html`
- Create: `_includes/home-foundation-card.html`
- Modify: `_includes/base_path`
- Modify: `_includes/head.html`
- Modify: `_includes/masthead.html`
- Modify: `_includes/footer.html`
- Delete: `_includes/footer/custom.html`
- Modify: `_includes/seo.html`
- Modify: `_includes/tag-chip.html`
- Modify: `_includes/head/custom.html`
- Modify: `_includes/author-profile.html`
- Modify: `_layouts/default.html`
- Modify: `_layouts/archive.html`
- Modify: `_layouts/single.html`

**Main-site Sass and pages**

- Modify: `assets/css/main.scss`
- Modify: `_sass/layout/_page.scss`
- Modify: `_sass/layout/_buttons.scss`
- Modify: `_sass/layout/_navigation.scss`
- Modify: `_sass/layout/_cosmicrealm.scss`
- Create: `_sass/layout/_home.scss`
- Modify: `_pages/home.md`
- Modify: `_pages/writing.html`
- Modify: `_pages/projects.md`
- Modify: `_pages/publications.html`
- Modify: `_pages/foundations.md`
- Modify: `_data/navigation.yml`

**Standalone pages**

- Modify: `projects/iconface/index.html`
- Modify: `projects/style-talking/index.html`
- Modify: `projects/iconface/static/css/index.css`
- Modify: `projects/style-talking/static/css/index.css`
- Modify: `foundations/aigc-llm-math/index.html`
- Modify: `foundations/generation-acceleration/index.html`
- Modify: `foundations/generation-distillation/index.html`
- Modify: `foundations/generation-math/index.html`
- Modify: `foundations/image-generation-data-training/index.html`
- Modify: `foundations/leetcode-hot100/index.html`
- Modify: `foundations/llm-interview-qa/index.html`
- Modify: `foundations/llm-mechanics/index.html`
- Modify: `foundations/video-generation/index.html`

**Posts and verification**

- Modify: `_posts/2025-12-25-blog-nonochat-1-base.md`
- Create: `scripts/verify_homepage_theme_architecture.py`
- Create: `tests/theme_controller.test.cjs`
- Create: `tests/site_browser_qa.cjs`

### Task 1: Add the failing contract for structure, runtime, and routes

**Files:**

- Create: `scripts/verify_homepage_theme_architecture.py`
- Create: `tests/theme_controller.test.cjs`
- Create: `tests/site_browser_qa.cjs`

- [ ] **Step 1: Write the Python verifier before any implementation**

This verifier must fail until the new architecture exists. It checks:

- shared `theme-init.js`, `theme-controller.js`, `site.js`, and `theme-tokens.css`
- no `_layouts/talk.html` references
- no legacy runtime files or references
- complete homepage sections and curated homepage metadata
- all 9 Foundations + 2 projects patched
- `theme-tokens.css` owns all runtime token declarations
- internal links use `relative_url`, not production-origin `site.url`
- `base_path` resolves only `site.baseurl`

```python
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
```

- [ ] **Step 2: Write the failing theme-controller unit test**

The controller must resolve themes, update `aria-label`, and never auto-mount itself.

```javascript
"use strict";

const assert = require("node:assert/strict");
const { createThemeController, resolveTheme } = require("../assets/js/theme-controller.js");

assert.equal(resolveTheme("dark", false), "dark");
assert.equal(resolveTheme("light", true), "light");
assert.equal(resolveTheme("system", true), "dark");
assert.equal(resolveTheme("system", false), "light");

const storage = new Map();
const toggles = [
  {
    attrs: {},
    setAttribute(key, value) { this.attrs[key] = value; },
  },
];
const root = {
  attrs: {},
  setAttribute(key, value) { this.attrs[key] = value; },
  removeAttribute(key) { delete this.attrs[key]; },
};

const controller = createThemeController({
  storage: {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, value); },
  },
  root,
  toggles,
  systemDark: () => true,
});

controller.applyInitialTheme();
assert.equal(root.attrs["data-theme"], "dark");
assert.equal(toggles[0].attrs["aria-pressed"], "true");
assert.match(toggles[0].attrs["aria-label"], /Switch to light mode/);
controller.toggleExplicitTheme();
assert.equal(storage.get("theme"), "light");
assert.equal(root.attrs["data-theme"], "light");
assert.equal(toggles[0].attrs["aria-pressed"], "false");
assert.match(toggles[0].attrs["aria-label"], /Switch to dark mode/);
assert.equal(typeof globalThis.mountAll, "undefined");
```

- [ ] **Step 3: Write the browser QA as a failing complete script**

The script must already contain the full final checks, even though they fail initially:

- all 11 standalone pages reachable
- `360`, `390`, `768`, `1280`, `1440`
- same-origin `requestfailed`, same-origin HTTP `>=400`, console errors, page errors all empty
- optional external font / CDN / R2 media requests do not create false failures
- no homepage heavy resources
- no production-origin internal resources locally
- system light/dark, toggle refresh, cross-page persistence, keyboard, ARIA, reduced-motion
- no-JS nav fallback
- light/dark screenshots saved to `tmp/homepage-theme-qa/`

```javascript
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const SITE_URL = process.env.SITE_URL || "http://127.0.0.1:4000";
const CHROME_PATH = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const ARTIFACT_DIR = path.join(process.cwd(), "tmp", "homepage-theme-qa");
const INTERNAL_ORIGIN = new URL(SITE_URL).origin;
const WIDTHS = [360, 390, 768, 1280, 1440];
const STANDALONE_PATHS = [
  "/projects/iconface/",
  "/projects/style-talking/",
  "/foundations/aigc-llm-math/",
  "/foundations/generation-acceleration/",
  "/foundations/generation-distillation/",
  "/foundations/generation-math/",
  "/foundations/image-generation-data-training/",
  "/foundations/leetcode-hot100/",
  "/foundations/llm-interview-qa/",
  "/foundations/llm-mechanics/",
  "/foundations/video-generation/",
];

function ensureArtifacts() {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
}

function isInternalUrl(rawUrl) {
  try {
    return new URL(rawUrl).origin === INTERNAL_ORIGIN;
  } catch (error) {
    return false;
  }
}

function isOptionalExternalUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    if (url.origin === INTERNAL_ORIGIN) return false;
    return (
      url.hostname === "fonts.googleapis.com" ||
      url.hostname === "fonts.gstatic.com" ||
      url.hostname === "cdnjs.cloudflare.com" ||
      url.hostname === "cdn.jsdelivr.net" ||
      url.hostname.endsWith(".r2.dev")
    );
  } catch (error) {
    return false;
  }
}

function bindCollectors(page) {
  const requestfailed = [];
  const badResponses = [];
  const consoleErrors = [];
  const pageErrors = [];
  page.on("requestfailed", (req) => {
    const url = req.url();
    if (isOptionalExternalUrl(url)) return;
    if (!isInternalUrl(url) && req.resourceType() !== "document") return;
    requestfailed.push(`${req.method()} ${url}`);
  });
  page.on("response", (res) => {
    const url = res.url();
    if (res.status() >= 400 && isInternalUrl(url)) {
      badResponses.push(`${res.status()} ${url}`);
    }
  });
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const sourceUrl = msg.location().url || "";
    if (sourceUrl && isOptionalExternalUrl(sourceUrl)) return;
    consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => pageErrors.push(err.message));
  return { requestfailed, badResponses, consoleErrors, pageErrors };
}

async function gotoReady(page, url, readySelector) {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.locator(readySelector).first().waitFor({ state: "visible" });
}

async function assertNoOverflow(page, width, height) {
  await page.setViewportSize({ width, height });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => resolve())));
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  assert.equal(metrics.scrollWidth, metrics.clientWidth, `${width}px overflow detected`);
}

async function assertThemeToggle(page) {
  const button = page.locator("[data-theme-toggle]").first();
  const before = await button.getAttribute("aria-pressed");
  await button.focus();
  await page.keyboard.press("Enter");
  const afterEnter = await button.getAttribute("aria-pressed");
  const labelAfterEnter = await button.getAttribute("aria-label");
  assert.notEqual(afterEnter, before, "Enter key did not toggle theme");
  assert.ok(labelAfterEnter.includes("Switch to"), "theme toggle aria-label missing action text after Enter");
  await page.keyboard.press("Space");
  const afterSpace = await button.getAttribute("aria-pressed");
  const labelAfterSpace = await button.getAttribute("aria-label");
  assert.equal(afterSpace, before, "Space key did not toggle theme back");
  assert.ok(labelAfterSpace.includes("Switch to"), "theme toggle aria-label missing action text after Space");
}

async function assertReducedMotion(page) {
  const reduced = await page.evaluate(() => ({
    matches: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    scrollBehavior: getComputedStyle(document.documentElement).scrollBehavior,
  }));
  assert.equal(reduced.matches, true, "reduced-motion media query not active in QA context");
  assert.equal(reduced.scrollBehavior, "auto", "reduced-motion CSS did not disable smooth scrolling");
}

async function assertHomepageResources(page) {
  const resources = await page.evaluate(() =>
    performance.getEntriesByType("resource").map((entry) => entry.name)
  );
  for (const forbidden of ["main.min.js", "plotly", "mathjax", "mermaid"]) {
    assert.equal(resources.some((name) => name.toLowerCase().includes(forbidden)), false, `${forbidden} loaded on homepage`);
  }
  assert.equal(resources.some((name) => name.startsWith("https://cosmicrealm.github.io/")), false, "local run loaded production-origin internal resource");
}

async function assertThemePersistence(browser, colorScheme) {
  const context = await browser.newContext({ colorScheme, reducedMotion: "reduce", viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  await gotoReady(page, `${SITE_URL}/`, "[data-theme-toggle]");
  await assertReducedMotion(page);
  const initial = await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light");
  if (colorScheme === "dark") assert.equal(initial, "dark");
  if (colorScheme === "light") assert.equal(initial, "light");
  await page.locator("[data-theme-toggle]").first().click();
  const explicit = await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light");
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.locator("[data-theme-toggle]").first().waitFor({ state: "visible" });
  assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light"), explicit);
  await gotoReady(page, `${SITE_URL}/writing/`, "main");
  assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light"), explicit);
  await gotoReady(page, `${SITE_URL}/projects/iconface/`, "[data-theme-toggle]");
  assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light"), explicit);
  await context.close();
}

async function main() {
  ensureArtifacts();
  const browser = await chromium.launch({ headless: true, executablePath: CHROME_PATH });
  try {
    await assertThemePersistence(browser, "light");
    await assertThemePersistence(browser, "dark");

    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, reducedMotion: "reduce" });
    const page = await context.newPage();
    const collectors = bindCollectors(page);
    await gotoReady(page, `${SITE_URL}/`, "[data-theme-toggle]");
    await assertHomepageResources(page);
    await assertThemeToggle(page);
    await assertReducedMotion(page);
    for (const width of WIDTHS) {
      await assertNoOverflow(page, width, 900);
    }
    await page.screenshot({ path: path.join(ARTIFACT_DIR, "home-light.png"), fullPage: true });
    await page.locator("[data-theme-toggle]").first().click();
    await page.screenshot({ path: path.join(ARTIFACT_DIR, "home-dark.png"), fullPage: true });
    for (const url of STANDALONE_PATHS) {
      await gotoReady(page, `${SITE_URL}${url}`, "[data-theme-toggle]");
      assert.ok(await page.locator("[data-theme-toggle]").count(), `missing toggle on ${url}`);
      await assertReducedMotion(page);
      for (const width of WIDTHS) {
        await assertNoOverflow(page, width, 900);
      }
    }
    assert.deepEqual(collectors.requestfailed, []);
    assert.deepEqual(collectors.badResponses, []);
    assert.deepEqual(collectors.consoleErrors, []);
    assert.deepEqual(collectors.pageErrors, []);
    await context.close();

    const noJs = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 1280, height: 900 } });
    const noJsPage = await noJs.newPage();
    await noJsPage.goto(`${SITE_URL}/`, { waitUntil: "domcontentloaded" });
    const navState = await noJsPage.evaluate(() => {
      const menu = document.querySelector("[data-nav-menu]");
      const toggle = document.querySelector("[data-nav-toggle]");
      const projectLink = document.querySelector('a[href="/projects/"], a[href$="/projects/"]');
      const writingLink = document.querySelector('a[href="/writing/"], a[href$="/writing/"]');
      if (!menu || !toggle || !projectLink || !writingLink) return null;
      const menuStyle = window.getComputedStyle(menu);
      const toggleStyle = window.getComputedStyle(toggle);
      return {
        menuVisible: menuStyle.display !== "none" && menuStyle.visibility !== "hidden",
        toggleHidden: toggleStyle.display === "none",
      };
    });
    assert.ok(navState && navState.menuVisible && navState.toggleHidden, "no-JS nav fallback is not visibly expanded");
    await noJs.close();
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

- [ ] **Step 4: Run the failing checks**

Run: `python3 scripts/verify_homepage_theme_architecture.py`

Expected: non-zero exit because shared runtime, curated homepage data, and standalone patches do not exist yet.

Run: `node tests/theme_controller.test.cjs`

Expected: non-zero exit with `Cannot find module '../assets/js/theme-controller.js'`.

- [ ] **Step 5: Commit the failing contract**

```bash
git add scripts/verify_homepage_theme_architecture.py tests/theme_controller.test.cjs tests/site_browser_qa.cjs
git commit -m "test: define homepage B-theme architecture contract"
```

### Task 2: Introduce the shared synchronous init and deferred controller chain

**Files:**

- Create: `assets/js/theme-init.js`
- Create: `assets/js/theme-controller.js`
- Create: `assets/js/site.js`
- Modify: `_includes/head.html`
- Modify: `_includes/scripts.html`
- Modify: `_layouts/default.html`

- [ ] **Step 1: Create the tiny shared synchronous init**

This file runs before CSS on every page and does only first-paint theme resolution.

```javascript
(function () {
  var root = document.documentElement;
  var stored = null;
  var setting = "system";
  try {
    stored = localStorage.getItem("theme");
  } catch (error) {
    stored = null;
  }
  if (stored === "light" || stored === "dark") setting = stored;
  var systemDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var resolved = setting === "dark" || (setting === "system" && systemDark) ? "dark" : "light";
  root.setAttribute("data-theme", resolved);
  root.setAttribute("data-theme-setting", setting);
}());
```

- [ ] **Step 2: Load `theme-init.js` before CSS in `_includes/head.html`**

The head must load `theme-init.js` first, then CSS assets. Internal asset links should use `relative_url`, not production-origin concatenation.

```liquid
<script src="{{ '/assets/js/theme-init.js' | relative_url }}"></script>
<link rel="stylesheet" href="{{ '/assets/css/theme-tokens.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/css/main.css' | relative_url }}?v={{ site.time | date: '%Y%m%d%H%M%S' }}">
```

Also give no-JS browsers an explicit light fallback and remove the retired `site_theme` branch in `_layouts/default.html`:

```liquid
<html lang="{{ site.locale | slice: 0,2 }}" class="no-js" data-theme="light" data-theme-setting="system">
```

- [ ] **Step 3: Implement `theme-controller.js` without auto-mount**

The controller must update theme state, `aria-pressed`, and `aria-label`, but never bind itself until `site.js` calls it.

```javascript
"use strict";

function resolveTheme(setting, systemDark) {
  if (setting === "dark") return "dark";
  if (setting === "light") return "light";
  return systemDark ? "dark" : "light";
}

function updateToggle(toggle, resolved) {
  toggle.setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
  toggle.setAttribute("aria-label", resolved === "dark" ? "Switch to light mode" : "Switch to dark mode");
}

function createThemeController(options) {
  const storage = options.storage;
  const root = options.root;
  const toggles = options.toggles;
  const systemDark = options.systemDark;

  function readSetting() {
    let value = null;
    try {
      value = storage.getItem("theme");
    } catch (error) {
      value = null;
    }
    return value === "light" || value === "dark" ? value : "system";
  }

  function applyResolved(resolved, setting) {
    root.setAttribute("data-theme", resolved);
    root.setAttribute("data-theme-setting", setting);
    toggles.forEach((toggle) => updateToggle(toggle, resolved));
    if (typeof document !== "undefined") {
      document.querySelectorAll("[data-theme-color]").forEach((meta) => {
        meta.setAttribute("content", resolved === "dark" ? "#121b28" : "#f6f1e8");
      });
    }
    return resolved;
  }

  function apply(setting) {
    return applyResolved(resolveTheme(setting, systemDark()), setting);
  }

  return {
    applyInitialTheme() { return apply(readSetting()); },
    toggleExplicitTheme() {
      const current = resolveTheme(readSetting(), systemDark());
      const next = current === "dark" ? "light" : "dark";
      try {
        storage.setItem("theme", next);
      } catch (error) {
        // The resolved theme still applies when storage is unavailable.
      }
      return apply(next);
    },
    applySystemChange() {
      if (readSetting() === "system") return apply("system");
      return apply(readSetting());
    },
  };
}

if (typeof window !== "undefined") {
  window.CosmicThemeController = { createThemeController, resolveTheme };
}

if (typeof module !== "undefined") {
  module.exports = { createThemeController, resolveTheme };
}
```

- [ ] **Step 4: Implement `site.js` with full boot, navigation, and writing filter**

`site.js` must own all DOM-ready behavior:

- `mountTheme()`
- `mountNavigation()`
- `mountWritingFilter()`

```javascript
"use strict";

(function () {
  function mountTheme() {
    const toggles = Array.from(document.querySelectorAll("[data-theme-toggle]"));
    if (!window.CosmicThemeController || toggles.length === 0) return;
    const controller = window.CosmicThemeController.createThemeController({
      storage: window.localStorage,
      root: document.documentElement,
      toggles,
      systemDark: () => window.matchMedia("(prefers-color-scheme: dark)").matches,
    });
    controller.applyInitialTheme();
    toggles.forEach((toggle) => {
      toggle.addEventListener("click", () => controller.toggleExplicitTheme());
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => controller.applySystemChange());
  }

  function mountNavigation() {
    const button = document.querySelector("[data-nav-toggle]");
    const menu = document.querySelector("[data-nav-menu]");
    if (!button || !menu) return;

    function closeMenu() {
      button.setAttribute("aria-expanded", "false");
      menu.hidden = true;
    }

    function openMenu() {
      button.setAttribute("aria-expanded", "true");
      menu.hidden = false;
    }

    function syncDesktop() {
      if (window.innerWidth >= 1024) {
        menu.hidden = false;
        button.setAttribute("aria-expanded", "false");
      } else {
        menu.hidden = button.getAttribute("aria-expanded") !== "true";
      }
    }

    button.addEventListener("click", () => {
      const expanded = button.getAttribute("aria-expanded") === "true";
      if (expanded) closeMenu();
      else openMenu();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeMenu();
    });

    window.addEventListener("resize", syncDesktop);
    syncDesktop();
  }

  function mountWritingFilter() {
    const root = document.querySelector("[data-blog-filter-root]");
    const list = document.querySelector("[data-blog-list]");
    if (!root || !list) return;
    const buttons = Array.from(document.querySelectorAll("[data-blog-tag-filter]"));
    const clearButton = root.querySelector("[data-blog-clear-filter]");
    const count = root.querySelector("[data-blog-filter-count]");
    const empty = document.querySelector("[data-blog-empty]");
    const items = Array.from(list.querySelectorAll("[data-blog-item]"));
    const selected = new Set((new URLSearchParams(window.location.search).get("tags") || "").split(",").filter(Boolean));

    function render() {
      let visible = 0;
      items.forEach((item) => {
        const tags = (item.dataset.blogTags || "").split(" ").filter(Boolean);
        const matches = selected.size === 0 || tags.some((tag) => selected.has(tag));
        item.hidden = !matches;
        if (matches) visible += 1;
      });
      buttons.forEach((button) => {
        const active = selected.has(button.dataset.blogTagFilter);
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
      if (clearButton) clearButton.setAttribute("aria-pressed", selected.size === 0 ? "true" : "false");
      if (count) count.textContent = `${visible} / ${items.length}`;
      if (empty) empty.hidden = visible !== 0;
    }

    function syncUrl() {
      const url = new URL(window.location.href);
      if (selected.size > 0) url.searchParams.set("tags", Array.from(selected).join(","));
      else url.searchParams.delete("tags");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    }

    document.addEventListener("click", (event) => {
      const tagButton = event.target.closest("[data-blog-tag-filter]");
      const resetButton = event.target.closest("[data-blog-clear-filter]");
      if (tagButton) {
        const tag = tagButton.dataset.blogTagFilter;
        if (selected.has(tag)) selected.delete(tag);
        else selected.add(tag);
        render();
        syncUrl();
      } else if (resetButton) {
        selected.clear();
        render();
        syncUrl();
      }
    });

    render();
  }

  function boot() {
    mountTheme();
    mountNavigation();
    mountWritingFilter();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
}());
```

- [ ] **Step 5: Load only deferred controller and site runtime**

```liquid
<script src="{{ '/assets/js/theme-controller.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/site.js' | relative_url }}" defer></script>
{% include analytics.html %}
```

- [ ] **Step 6: Run the unit test and commit**

Run: `node tests/theme_controller.test.cjs`

Expected: pass.

```bash
git add assets/js/theme-init.js assets/js/theme-controller.js assets/js/site.js _includes/head.html _includes/scripts.html _layouts/default.html
git commit -m "feat: add shared theme init and deferred runtime chain"
```

### Task 3: Unify tokens and Sass ownership, and remove old runtime files

**Files:**

- Create: `assets/css/theme-tokens.css`
- Move: `_sass/_themes.scss` -> `_sass/theme/_settings.scss`
- Modify: `assets/css/main.scss`
- Modify: `_sass/layout/_page.scss`
- Modify: `_sass/layout/_buttons.scss`
- Modify: `_sass/layout/_navigation.scss`
- Modify: `_sass/layout/_cosmicrealm.scss`
- Create: `_sass/layout/_home.scss`
- Delete: `_sass/theme/_default_light.scss`
- Delete: `_sass/theme/_default_dark.scss`
- Delete: `_sass/theme/_air_light.scss`
- Delete: `_sass/theme/_air_dark.scss`
- Delete: `_sass/layout/_json_cv.scss`
- Delete: `assets/js/main.min.js`
- Delete: `assets/js/_main.js`
- Delete: `assets/js/theme.js`
- Delete: `assets/js/plugins/jquery.greedy-navigation.js`
- Delete: `assets/js/collapse.js`

- [ ] **Step 1: Create the shared runtime token file**

This file must contain every token the current Sass expects, but express B-style as flat paper/card surfaces, thin borders, and no gradient/glass/blur.

```css
:root,
html[data-theme="light"] {
  color-scheme: light;
  --global-base-color: #666d78;
  --global-bg-color: #f6f1e8;
  --global-footer-bg-color: #f0eadf;
  --global-border-color: #d7d1c6;
  --global-dark-border-color: #c7bfb2;
  --global-code-background-color: #efe8de;
  --global-code-text-color: #172235;
  --global-fig-caption-color: #6a7280;
  --global-link-color: #9a4736;
  --global-link-color-hover: #7c382b;
  --global-link-color-visited: #7c382b;
  --global-masthead-link-color: #172235;
  --global-masthead-link-color-hover: #9a4736;
  --global-text-color: #172235;
  --global-text-color-light: #5c6778;
  --global-thead-color: #e9e2d7;
  --cr-accent: #9a4736;
  --cr-accent-strong: #7c382b;
  --cr-accent-soft: rgba(154, 71, 54, 0.1);
  --cr-warm: #8c6f54;
  --cr-page-bg: #f6f1e8;
  --cr-card-bg: #fbf8f1;
  --cr-panel-bg: #f1ebe2;
  --cr-surface: #fbf8f1;
  --cr-ink-soft: #5c6778;
  --cr-border: #d7d1c6;
  --cr-border-strong: #c7bfb2;
  --cr-shadow: 0 0 0 0 transparent;
  --cr-shadow-hover: 0 0 0 0 transparent;
  --cr-font-serif: Georgia, "Times New Roman", "Noto Serif SC", serif;
  --cr-font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
  --cr-font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

html[data-theme="dark"] {
  color-scheme: dark;
  --global-base-color: #9aa4b2;
  --global-bg-color: #121b28;
  --global-footer-bg-color: #101724;
  --global-border-color: #283549;
  --global-dark-border-color: #34445b;
  --global-code-background-color: #182331;
  --global-code-text-color: #eef1f5;
  --global-fig-caption-color: #9ea8b6;
  --global-link-color: #d57a62;
  --global-link-color-hover: #ebb19e;
  --global-link-color-visited: #d57a62;
  --global-masthead-link-color: #eef1f5;
  --global-masthead-link-color-hover: #ebb19e;
  --global-text-color: #eef1f5;
  --global-text-color-light: #9ea8b6;
  --global-thead-color: #182331;
  --cr-accent: #d57a62;
  --cr-accent-strong: #ebb19e;
  --cr-accent-soft: rgba(213, 122, 98, 0.12);
  --cr-warm: #b6946d;
  --cr-page-bg: #121b28;
  --cr-card-bg: #162131;
  --cr-panel-bg: #182331;
  --cr-surface: #162131;
  --cr-ink-soft: #9ea8b6;
  --cr-border: #283549;
  --cr-border-strong: #34445b;
  --cr-shadow: 0 0 0 0 transparent;
  --cr-shadow-hover: 0 0 0 0 transparent;
}

html {
  background: var(--cr-page-bg);
}

:focus-visible {
  outline: 2px solid var(--cr-accent);
  outline-offset: 3px;
}

@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto !important; }
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

- [ ] **Step 2: Move the complete compile-time settings owner**

Do not replace the current 103-line `_themes.scss` with a shortened variable list: typography, type scale, breakpoints, Susy grid, and utility brand colors are still compiled by retained layout partials. Move it intact, then append the compile-time values currently supplied by `_default_light.scss`:

```bash
git mv _sass/_themes.scss _sass/theme/_settings.scss
```

```scss
$gray: #7a8288;
$dark-gray: mix(#000, $gray, 40%);
$darker-gray: mix(#000, $gray, 60%);
$light-gray: mix(#fff, $gray, 50%);
$lighter-gray: mix(#fff, $gray, 90%);
$danger-color: #ee5f5b;
$info-color: #2f7f93;
$notice-color: #7a8288;
$success-color: #62c462;
$warning-color: #f89406;
$border-radius: 4px;
$box-shadow: none;
$global-transition: all 0.2s ease-in-out;
$masthead-height: 70px;
$navicon-width: 28px;
$navicon-height: 4px;
$sidebar-link-max-width: 250px;
$sidebar-screen-min-width: 1024px;
```

The `@include breakpoint-set(...)` call inside the moved file is why `vendor/breakpoint/breakpoint` must remain before `theme/settings` in `main.scss`.

- [ ] **Step 3: Rewire `main.scss` and split homepage styles**

`main.scss` must import `_settings.scss` and `_home.scss`, and stop importing deleted theme partials.

```scss
@import
  "vendor/breakpoint/breakpoint",
  "theme/settings",
  "include/mixins",
  "vendor/susy/susy",
  "layout/reset",
  "layout/base",
  "include/utilities",
  "layout/tables",
  "layout/buttons",
  "layout/notices",
  "layout/masthead",
  "layout/navigation",
  "layout/footer",
  "syntax",
  "layout/forms",
  "layout/page",
  "layout/archive",
  "layout/sidebar",
  "layout/cosmicrealm",
  "layout/home",
  "vendor/font-awesome/fontawesome",
  "vendor/font-awesome/solid",
  "vendor/font-awesome/brands";
```

Delete only the retired `layout/json_cv` import; the architecture prerequisite already removed `/cv-json/`. Keep every other import above.

Create `_sass/layout/_home.scss` as the final override layer for the B-style shell and homepage. This is the visual implementation, not a placeholder partial:

```scss
body {
  background: var(--cr-page-bg);
  color: var(--global-text-color);
  font-family: var(--cr-font-sans);
}

.masthead {
  background: var(--cr-page-bg);
  border-bottom: 1px solid var(--cr-border);
  box-shadow: none;
}

.masthead__inner-wrap,
.site-nav,
.home-shell,
.site-footer__inner {
  width: min(100% - 2rem, 74rem);
  margin-inline: auto;
}

.site-nav {
  min-height: 4.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.site-nav__brand {
  color: var(--global-text-color);
  font-family: var(--cr-font-serif);
  font-size: 1.1rem;
  font-weight: 700;
}

.site-nav__menu {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.site-nav__menu a {
  color: var(--global-text-color-light);
  font-size: 0.9rem;
  text-decoration: none;
}

.site-nav__menu a:hover,
.site-nav__menu a:focus-visible {
  color: var(--cr-accent);
}

.site-nav__toggle,
.theme-toggle,
.hero-actions a,
.section-heading__link {
  border: 1px solid var(--cr-border-strong);
  border-radius: 0;
  background: transparent;
  color: var(--global-text-color);
  font: inherit;
}

.site-nav__toggle,
.theme-toggle {
  min-height: 2.4rem;
  padding: 0.45rem 0.7rem;
  cursor: pointer;
}

.site-nav__toggle { display: none; }

.home-shell {
  padding-block: clamp(3rem, 7vw, 6.5rem);
}

.home-document {
  display: grid;
  gap: clamp(3.5rem, 7vw, 6.5rem);
}

.home-hero {
  max-width: 62rem;
  padding-top: clamp(1rem, 4vw, 3rem);
}

.home-hero__eyebrow,
.section-heading__link,
.focus-strip span {
  color: var(--cr-accent);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.home-hero h1 {
  max-width: 15ch;
  margin: 0.45rem 0 1.25rem;
  color: var(--global-text-color);
  font-family: var(--cr-font-serif);
  font-size: clamp(2.8rem, 7vw, 6.4rem);
  font-weight: 500;
  line-height: 0.98;
  letter-spacing: -0.045em;
}

.home-hero__lead {
  max-width: 48rem;
  color: var(--cr-ink-soft);
  font-size: clamp(1rem, 2vw, 1.25rem);
  line-height: 1.75;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  margin-top: 1.75rem;
}

.hero-actions a,
.section-heading__link {
  padding: 0.55rem 0.8rem;
  text-decoration: none;
}

.hero-actions a:hover,
.section-heading__link:hover {
  border-color: var(--cr-accent);
  color: var(--cr-accent);
}

.focus-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border-block: 1px solid var(--cr-border);
}

.focus-strip__item {
  display: grid;
  gap: 0.25rem;
  padding: 1.35rem 1rem;
}

.focus-strip__item + .focus-strip__item { border-left: 1px solid var(--cr-border); }
.focus-strip strong { font-family: var(--cr-font-serif); font-size: 2rem; font-weight: 500; }

.home-section { display: grid; gap: 1.5rem; }

.section-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid var(--cr-border-strong);
}

.section-heading h2 {
  margin: 0;
  font-family: var(--cr-font-serif);
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 500;
}

.home-card-grid,
.publication-grid {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 1rem;
}

.home-card,
.publication-card {
  grid-column: span 4;
  min-width: 0;
  background: var(--cr-card-bg);
  border: 1px solid var(--cr-border);
  box-shadow: none;
}

.publication-card { padding: 1.25rem; }
.home-card__cover { display: block; border-bottom: 1px solid var(--cr-border); }
.home-card__cover img { display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: cover; }
.home-card__body { padding: 1.15rem; }
.home-card h3,
.publication-card h3 { margin: 0 0 0.65rem; font-family: var(--cr-font-serif); font-size: 1.25rem; }
.home-card p,
.publication-card p { margin: 0; color: var(--cr-ink-soft); line-height: 1.65; }
.home-card a,
.publication-card a { color: var(--global-text-color); }
.home-card a:hover,
.publication-card a:hover { color: var(--cr-accent); }

.compact-list {
  margin: 0;
  padding: 0;
  list-style: none;
  border-top: 1px solid var(--cr-border);
}

.compact-list li {
  display: grid;
  grid-template-columns: 7rem minmax(0, 1fr);
  gap: 1rem;
  padding: 0.9rem 0;
  border-bottom: 1px solid var(--cr-border);
}

.compact-list time { color: var(--cr-ink-soft); font-family: var(--cr-font-mono); font-size: 0.82rem; }
.compact-list a { color: var(--global-text-color); text-decoration: none; }
.compact-list a:hover { color: var(--cr-accent); }

.archive-layout--full .archive,
.archive-layout--full .archive-shell {
  float: none;
  width: min(100% - 2rem, 74rem);
  margin-inline: auto;
  padding-inline: 0;
}

.page__footer {
  position: static;
  margin-top: 5rem;
  background: var(--global-footer-bg-color);
  border-top: 1px solid var(--cr-border);
  box-shadow: none;
}

.site-footer__inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-block: 1.4rem;
  color: var(--cr-ink-soft);
  font-size: 0.82rem;
}

.site-footer__links { display: flex; flex-wrap: wrap; gap: 1rem; }
.site-footer__links a { color: inherit; }

@media (max-width: 63.99rem) {
  .site-nav { flex-wrap: wrap; padding-block: 0.8rem; }
  .site-nav__toggle { display: inline-flex; }
  .site-nav__menu { flex-basis: 100%; flex-direction: column; align-items: stretch; gap: 0; }
  .site-nav__menu[hidden] { display: none; }
  .site-nav__menu li { padding-block: 0.55rem; border-top: 1px solid var(--cr-border); }
}

@media (min-width: 64rem) {
  .site-nav__menu[hidden] { display: flex; }
}

@media (max-width: 48rem) {
  .home-card,
  .publication-card { grid-column: 1 / -1; }
  .focus-strip { grid-template-columns: 1fr; }
  .focus-strip__item + .focus-strip__item { border-left: 0; border-top: 1px solid var(--cr-border); }
  .section-heading,
  .site-footer__inner { align-items: flex-start; flex-direction: column; }
  .compact-list li { grid-template-columns: 1fr; gap: 0.3rem; }
}
```

- [ ] **Step 4: Remove duplicate tokens and the complete share-style surface**

Delete every `:root` / `html[data-theme="dark"]` runtime-token block from `_cosmicrealm.scss`. Delete the `/* social buttons */` map from `_buttons.scss`, the `.page__share` block and title rules from `_page.scss`, and the `.page__share + .pagination` coupling from `_navigation.scss`. Delete `_sass/layout/_json_cv.scss`; do not delete `layout/sidebar`, which still serves Writing filters and any article/sidebar opt-in.

- [ ] **Step 5: Delete the legacy runtime files and prove references are gone**

```bash
git rm assets/js/main.min.js assets/js/_main.js assets/js/theme.js assets/js/plugins/jquery.greedy-navigation.js assets/js/collapse.js _sass/layout/_json_cv.scss _sass/theme/_default_light.scss _sass/theme/_default_dark.scss _sass/theme/_air_light.scss _sass/theme/_air_dark.scss
```

Run: `rg -n "main.min.js|_main.js|theme.js|jquery.greedy-navigation|plotly.js-dist-min|fitvids|jquery-smooth-scroll|collapse.js" _data _includes _layouts _pages _posts _sass assets foundations projects _config.yml _config_docker.yml index.html`

Expected: exit 1 with zero references after deletion and template rewiring.

- [ ] **Step 6: Commit the shared token system and runtime deletion**

```bash
git add assets/css/theme-tokens.css assets/css/main.scss _sass/theme/_settings.scss _sass/layout/_page.scss _sass/layout/_buttons.scss _sass/layout/_navigation.scss _sass/layout/_cosmicrealm.scss _sass/layout/_home.scss
git commit -m "refactor: unify theme ownership and remove legacy runtime"
```

### Task 4: Fix internal path semantics, navigation, and SEO ownership

**Files:**

- Modify: `_includes/base_path`
- Modify: `_includes/masthead.html`
- Modify: `_includes/footer.html`
- Delete: `_includes/footer/custom.html`
- Modify: `_includes/seo.html`
- Modify: `_includes/head/custom.html`
- Modify: `_includes/author-profile.html`
- Modify: `_layouts/default.html`
- Modify: `_config.yml`

- [ ] **Step 1: Reduce `_includes/base_path` to baseurl-only semantics**

The include must no longer switch to production-origin `site.url` for internal assets or nav. Absolute URLs belong only in SEO.

```liquid
{% assign base_path = site.baseurl | default: "" %}
```

- [ ] **Step 2: Rebuild masthead markup with full nav hooks and no-JS fallback**

Use `relative_url` for internal links, and provide `data-nav-toggle` / `data-nav-menu`.

```liquid
<div class="masthead">
  <div class="masthead__inner-wrap">
    <nav class="site-nav" aria-label="Primary navigation">
      <a class="site-nav__brand" href="{{ '/' | relative_url }}">{{ site.title }}</a>
      <button type="button" class="site-nav__toggle" data-nav-toggle aria-expanded="false" aria-controls="site-nav-menu">Menu</button>
      <ul id="site-nav-menu" class="site-nav__menu" data-nav-menu hidden>
        {% for link in site.data.navigation.main %}
          <li><a href="{% if link.url contains 'http' %}{{ link.url }}{% else %}{{ link.url | relative_url }}{% endif %}">{{ link.title }}</a></li>
        {% endfor %}
        <li><button type="button" class="theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Switch to dark mode">Theme</button></li>
      </ul>
    </nav>
    <noscript><style>#site-nav-menu{display:flex!important}.site-nav__toggle{display:none!important}</style></noscript>
  </div>
</div>
```

- [ ] **Step 3: Replace the upstream/template footer with a minimal content footer**

Delete `_includes/footer/custom.html` and remove its include line from `_layouts/default.html`; optional renderers move to `_includes/scripts.html` in Task 6. Replace `_includes/footer.html` with:

```liquid
<div class="site-footer__inner">
  <p>&copy; {{ site.time | date: '%Y' }} Jinyang Zhang.</p>
  <nav class="site-footer__links" aria-label="Footer navigation">
    <a href="mailto:{{ site.author.email }}">Email</a>
    <a href="https://github.com/{{ site.author.github }}" rel="me">GitHub</a>
    <a href="{{ '/cv/' | relative_url }}">CV</a>
    <a href="{{ '/sitemap/' | relative_url }}">Sitemap</a>
  </nav>
</div>
```

In `_includes/head/custom.html`, keep the existing favicon/manifest links but change the fixed color meta to a controller-addressable light fallback:

```html
<meta name="theme-color" content="#f6f1e8" data-theme-color>
```

- [ ] **Step 4: Keep generic SEO and Person JSON-LD, but delete network-specific blocks**

Set non-empty `social` in `_config.yml`:

```yaml
social:
  type: Person
  name: Jinyang Zhang
  links:
    - https://github.com/cosmicrealm
```

Keep in `_includes/seo.html`:

```liquid
{% assign seo_origin = site.url | append: site.baseurl %}
<link rel="canonical" href="{{ page.url | prepend: seo_origin | replace: '/index.html', '/' }}">
<meta property="og:url" content="{{ page.url | prepend: seo_origin | replace: '/index.html', '/' }}">
<meta property="og:site_name" content="{{ site.title }}">
<meta property="og:title" content="{{ page.title | default: site.title | markdownify | strip_html | strip_newlines | escape_once }}">
<meta property="og:description" content="{{ seo_description }}">
<script type="application/ld+json">
{
  "@context": "http://schema.org",
  "@type": "Person",
  "name": "{{ site.social.name }}",
  "url": {{ seo_origin | jsonify }},
  "sameAs": {{ site.social.links | jsonify }}
}
</script>
```

Delete all `twitter:*` and Facebook publisher/app-id blocks.

Also remove the top-level `twitter:` block, `author.twitter`, and any empty network-specific keys from `_config.yml`. Delete the Twitter/X conditional and its “X (formerly Twitter)” text from `_includes/author-profile.html`; GitHub, Email, CV, publication/code links, canonical metadata, and generic Open Graph remain.

- [ ] **Step 5: Remove `site_theme`, keep publish excludes explicit**

`_config.yml` must:

- delete `site_theme`
- exclude `package-lock.json`
- keep `node_modules`
- stop reintroducing `share: false` defaults

```yaml
exclude:
  - node_modules
  - package-lock.json
  - docs/
  - scripts/
  - tests/
```

- [ ] **Step 6: Commit path semantics and SEO ownership**

```bash
git add _includes/base_path _includes/masthead.html _includes/footer.html _includes/seo.html _includes/head/custom.html _includes/author-profile.html _layouts/default.html _config.yml
git add -u _includes/footer/custom.html
git commit -m "refactor: fix internal path semantics and retain generic seo"
```

### Task 5: Build the home layout, curated homepage data, and schema-correct cards

**Files:**

- Create: `_layouts/home.html`
- Create: `_includes/home-project-card.html`
- Create: `_includes/home-foundation-card.html`
- Modify: `_pages/home.md`
- Modify: `_pages/writing.html`
- Modify: `_pages/projects.md`
- Modify: `_pages/publications.html`
- Modify: `_pages/foundations.md`
- Modify: `_data/projects.yml`
- Modify: `_data/navigation.yml`
- Modify: `_includes/tag-chip.html`
- Modify: `_layouts/archive.html`

- [ ] **Step 1: Add curated homepage metadata to `_data/projects.yml`**

Mark IConFace, Style-Talking, and Voice Studio as homepage-selected rather than relying on latest-date limits. Add an explicit `homepage_url` so the homepage card never depends on `links` ordering.

```yaml
- name: IConFace
  homepage: true
  homepage_order: 1
  homepage_url: /projects/iconface/
  ...

- name: Style-Talking
  homepage: true
  homepage_order: 2
  homepage_url: /projects/style-talking/
  ...

- name: Voice Studio
  homepage: true
  homepage_order: 3
  homepage_url: https://github.com/cosmicrealm/VoiceStudio
  ...
```

- [ ] **Step 2: Create `_layouts/home.html` and remove sidebar from home/section pages**

```liquid
---
layout: default
---

<main id="main" class="home-shell" role="main">
  <article class="home-document">
    {{ content }}
  </article>
</main>
```

Set:

```yaml
---
layout: home
author_profile: false
---
```

for `_pages/home.md`, and `author_profile: false` for `_pages/writing.html`, `_pages/projects.md`, `_pages/publications.html`, `_pages/foundations.md`.

- [ ] **Step 3: Make archive layout full-width when no author profile**

Keep the existing hero, breadcrumbs, title, and `.archive` wrapper. Replace only the `<div id="main">` and unconditional sidebar lines with the following; `page.blog_tag_filter` must still render `_includes/sidebar.html`, otherwise Writing loses its filter root:

```liquid
<div id="main" role="main" class="{% if page.blog_tag_filter %}blog-filter-layout{% elsif page.author_profile or layout.author_profile or page.sidebar %}{% else %}archive-layout--full{% endif %}">
  {% if page.author_profile or layout.author_profile or page.sidebar or page.blog_tag_filter %}
    {% include sidebar.html %}
  {% endif %}
  <div class="archive">
    {% unless page.header.overlay_color or page.header.overlay_image %}
      <h1 class="page__title">{{ page.title }}</h1>
    {% endunless %}
    {{ content }}
  </div>
</div>
```

- [ ] **Step 4: Create explicit home cards for projects and foundations**

Do not reuse the existing `project-card` for foundations because foundations only expose `url`, not `links`.

`_includes/home-project-card.html`

```liquid
{% assign project_url = include.project.homepage_url | default: include.project.links.first.url %}
<article class="home-card home-card--project">
  <a class="home-card__cover" href="{% if project_url contains 'http' %}{{ project_url }}{% else %}{{ project_url | relative_url }}{% endif %}">
    <img src="{{ include.project.teaser | relative_url }}" alt="{{ include.project.teaser_alt | default: include.project.name }}">
  </a>
  <div class="home-card__body">
    <h3><a href="{% if project_url contains 'http' %}{{ project_url }}{% else %}{{ project_url | relative_url }}{% endif %}">{{ include.project.name }}</a></h3>
    <p>{{ include.project.summary }}</p>
  </div>
</article>
```

`_includes/home-foundation-card.html`

```liquid
<article class="home-card home-card--foundation">
  <a class="home-card__cover" href="{{ include.foundation.url | relative_url }}">
    <img src="{{ include.foundation.teaser | relative_url }}" alt="{{ include.foundation.teaser_alt | default: include.foundation.name }}">
  </a>
  <div class="home-card__body">
    <h3><a href="{{ include.foundation.url | relative_url }}">{{ include.foundation.name }}</a></h3>
    <p>{{ include.foundation.highlight }}</p>
  </div>
</article>
```

- [ ] **Step 5: Rewrite `home.md` with the complete homepage content**

```liquid
<section class="home-hero">
  <p class="home-hero__eyebrow">Jinyang Zhang</p>
  <h1>Generative AI systems from research mechanism to deployable workflow.</h1>
  <p class="home-hero__lead">AIGC systems, talking avatars, image restoration, inference pipelines, and source-first technical foundations.</p>
  <div class="hero-actions">
    <a href="mailto:{{ site.author.email }}">Email</a>
    <a href="https://github.com/{{ site.author.github }}" target="_blank" rel="noopener">GitHub</a>
    <a href="{{ '/cv/' | relative_url }}">CV</a>
  </div>
</section>

<section class="focus-strip" aria-label="Site overview">
  <article class="focus-strip__item"><strong>{{ site.data.projects | size }}</strong><span>Projects</span></article>
  <article class="focus-strip__item"><strong>{{ site.publications | size }}</strong><span>Publications</span></article>
  <article class="focus-strip__item"><strong>{{ site.posts | size }}</strong><span>Writing</span></article>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Selected Work</h2>
    <a class="section-heading__link" href="{{ '/projects/' | relative_url }}">All projects</a>
  </div>
  {% assign featured_home_projects = site.data.projects | where: 'homepage', true | sort: 'homepage_order' %}
  <div class="home-card-grid">
    {% for project in featured_home_projects %}
      {% include home-project-card.html project=project %}
    {% endfor %}
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Representative Publications</h2>
    <a class="section-heading__link" href="{{ '/publications/' | relative_url }}">All publications</a>
  </div>
  {% assign selected_publications = site.publications | sort: 'date' | reverse %}
  <div class="publication-grid">
    {% for publication in selected_publications limit:4 %}
      {% include publication-card.html publication=publication %}
    {% endfor %}
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Foundations</h2>
    <a class="section-heading__link" href="{{ '/foundations/' | relative_url }}">All foundations</a>
  </div>
  {% assign featured_foundations = site.data.foundations | where: 'featured', true %}
  <div class="home-card-grid">
    {% for foundation in featured_foundations limit:4 %}
      {% include home-foundation-card.html foundation=foundation %}
    {% endfor %}
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Recent Writing</h2>
    <a class="section-heading__link" href="{{ '/writing/' | relative_url }}">All writing</a>
  </div>
  <ul class="compact-list compact-list--dated">
    {% for post in site.posts limit:4 %}
      <li><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>
```

- [ ] **Step 6: Keep Writing filter explicit and route-correct**

`_pages/writing.html` must keep `blog-filter` on page body and use `/writing/?tags=`.

```yaml
---
layout: archive
permalink: /writing/
title: "Writing"
author_profile: false
blog_tag_filter: true
---
```

```liquid
{% assign tag_chip_url = "/writing/?tags=" | append: tag_chip_slug %}
```

Delete the entire legacy inline `<script>` from `_pages/writing.html`; `assets/js/site.js` is now the only owner of `URLSearchParams`, filter clicks, counts, empty state, and URL synchronization. Keep the `data-blog-list`, `data-blog-item`, `data-blog-tags`, and `data-blog-empty` markup unchanged.

- [ ] **Step 7: Commit homepage data and layout changes**

```bash
git add _layouts/home.html _layouts/archive.html _includes/home-project-card.html _includes/home-foundation-card.html _pages/home.md _pages/writing.html _pages/projects.md _pages/publications.html _pages/foundations.md _data/projects.yml _data/navigation.yml _includes/tag-chip.html
git commit -m "feat: build curated homepage and schema-correct home cards"
```

### Task 6: Remove visible share UI, keep MathJax defaults, and make Mermaid opt-in

**Files:**

- Modify: `_layouts/single.html`
- Delete: `_includes/social-share.html`
- Modify: `_config.yml`
- Modify: `_includes/scripts.html`
- Create: `assets/js/mermaid-init.js`
- Modify: `_posts/2025-12-25-blog-nonochat-1-base.md`

- [ ] **Step 1: Remove the visible share include completely**

Delete this line from `_layouts/single.html`:

```liquid
{% if page.share %}{% include social-share.html %}{% endif %}
```

- [ ] **Step 2: Keep posts `mathjax: true` by default as compatibility tradeoff**

This preserves formulas across posts without re-auditing every article immediately. Homepage and section pages never set `mathjax`, so they still avoid the cost.

```yaml
defaults:
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: false
      mathjax: true
      mermaid: false
```

Add the only Mermaid opt-in directly to `_posts/2025-12-25-blog-nonochat-1-base.md` front matter:

```yaml
mermaid: true
```

- [ ] **Step 3: Load MathJax and Mermaid conditionally**

```liquid
{% if page.mathjax %}
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['\\(', '\\)'], ['$', '$']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };
  </script>
  <script defer src="{{ '/assets/vendor/mathjax/tex-mml-chtml.js' | relative_url }}" id="MathJax-script"></script>
{% endif %}
{% if page.mermaid %}
  <script type="module" src="{{ '/assets/js/mermaid-init.js' | relative_url }}"></script>
{% endif %}
```

```javascript
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const isDark = document.documentElement.getAttribute("data-theme") === "dark";
mermaid.initialize({ startOnLoad: false, theme: isDark ? "dark" : "default" });
if (document.querySelector("code.language-mermaid")) {
  await mermaid.run({ querySelector: "code.language-mermaid" });
}
```

- [ ] **Step 4: Commit share cleanup and renderer policy**

```bash
git rm _includes/social-share.html
git add _layouts/single.html _config.yml _includes/scripts.html assets/js/mermaid-init.js _posts/2025-12-25-blog-nonochat-1-base.md
git commit -m "refactor: remove visible share ui and gate heavy renderers"
```

### Task 7: Patch standalone pages without adding second headers or nested mains

**Files:**

- Create: `assets/css/content-theme-bridge.css`
- Modify: `projects/iconface/index.html`
- Modify: `projects/style-talking/index.html`
- Modify: `projects/iconface/static/css/index.css`
- Modify: `projects/style-talking/static/css/index.css`
- Modify: `foundations/aigc-llm-math/index.html`
- Modify: `foundations/generation-acceleration/index.html`
- Modify: `foundations/generation-distillation/index.html`
- Modify: `foundations/generation-math/index.html`
- Modify: `foundations/image-generation-data-training/index.html`
- Modify: `foundations/leetcode-hot100/index.html`
- Modify: `foundations/llm-interview-qa/index.html`
- Modify: `foundations/llm-mechanics/index.html`
- Modify: `foundations/video-generation/index.html`

- [ ] **Step 1: Create the shared-content bridge stylesheet**

Load this file after each microsite's local stylesheet. It maps the common local variable names used by the existing project/Foundation layouts onto the canonical B tokens, provides the shared toggle/focus treatment, and neutralizes decorative shell gradients/glass without rewriting content-specific grids or labs.

```css
html.has-shared-theme {
  --paper: var(--cr-page-bg);
  --bg: var(--cr-page-bg);
  --surface: var(--cr-card-bg);
  --panel: var(--cr-card-bg);
  --panel-strong: var(--cr-panel-bg);
  --soft: var(--cr-panel-bg);
  --ink: var(--global-text-color);
  --muted: var(--cr-ink-soft);
  --line: var(--cr-border);
  --accent: var(--cr-accent);
  --accent-deep: var(--cr-accent-strong);
  --accent-ink: var(--cr-accent-strong);
  --blue: var(--cr-accent);
  --blue-dark: var(--cr-accent-strong);
  --blue-soft: var(--cr-accent-soft);
  --rose: var(--cr-warm);
  --shadow: none;
  --shadow-small: none;
  --serif: var(--cr-font-serif);
  --sans: var(--cr-font-sans);
  --mono: var(--cr-font-mono);
}

html.has-shared-theme body,
html.has-shared-theme .site-header,
html.has-shared-theme .lecture-toc,
html.has-shared-theme .publication-header {
  background: var(--cr-page-bg);
  color: var(--global-text-color);
}

html.has-shared-theme .site-header,
html.has-shared-theme .lecture-toc,
html.has-shared-theme .project-home-link {
  -webkit-backdrop-filter: none;
  backdrop-filter: none;
  box-shadow: none;
}

.theme-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  min-height: 2.2rem;
  padding: 0.4rem 0.7rem;
  color: var(--global-text-color);
  border: 1px solid var(--cr-border);
  background: var(--cr-card-bg);
  cursor: pointer;
  font: 600 0.82rem/1 var(--cr-font-sans);
}

.has-shared-theme-shell {
  background: var(--cr-page-bg);
  color: var(--global-text-color);
}

.project-theme-toggle {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 60;
}

html.has-shared-theme .publication-header,
html.has-shared-theme .teaser-cell,
html.has-shared-theme .detail-comparison article,
html.has-shared-theme .contribution-grid article,
html.has-shared-theme .result-grid article,
html.has-shared-theme .method-pillars article,
html.has-shared-theme .framework-card,
html.has-shared-theme .dataset-block,
html.has-shared-theme .evidence-figure,
html.has-shared-theme .large-figure-disclosure,
html.has-shared-theme .image-tile-label,
html.has-shared-theme .image-tile-score {
  background: var(--cr-card-bg);
  border-color: var(--cr-border);
  box-shadow: none;
}

html.has-shared-theme .section-title::after {
  background: var(--cr-accent);
}

html.has-shared-theme .lead,
html.has-shared-theme .pill-row span,
html.has-shared-theme .pipeline span {
  color: var(--global-text-color);
}

html.has-shared-theme .gallery-error {
  background: var(--cr-accent-soft);
  border-color: var(--cr-accent);
  color: var(--cr-accent-strong);
}
```

- [ ] **Step 2: Patch both project pages with exact, page-specific anchors**

Do not add a second header or nested `<main>`. On both pages, change `<html lang="en">` (or the existing language value) to include `class="has-shared-theme"`. Remove every contiguous `<meta name="twitter:*">` line while retaining canonical, description, Open Graph, citation metadata, and JSON-LD.

Load `theme-init.js` before any stylesheet, load `theme-tokens.css` before the local stylesheet, and load `content-theme-bridge.css` after the local stylesheet so its mappings win:

```html
<script src="/assets/js/theme-init.js"></script>
<link rel="stylesheet" href="/assets/css/theme-tokens.css">
<!-- existing project-local stylesheet remains here -->
<link rel="stylesheet" href="/assets/css/content-theme-bridge.css">
<script src="/assets/js/theme-controller.js" defer></script>
<script src="/assets/js/site.js" defer></script>
```

For `projects/iconface/index.html`, add the toggle immediately after the existing fixed `.project-home-link`, and change its existing main only:

```html
<button type="button" class="theme-toggle project-theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Switch to dark mode">Theme</button>
<main id="main-content" class="has-shared-theme-shell">
```

For `projects/style-talking/index.html`, insert the standard toggle before the first `</nav>` inside `.site-header` and change its existing main only:

```html
<button type="button" class="theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Switch to dark mode">Theme</button>
<main class="has-shared-theme-shell">
```

Use these exact fallback roots in the project-local CSS; the later bridge maps them to the current shared light/dark tokens:

```css
/* projects/iconface/static/css/index.css */
:root {
  --ink: #172235;
  --muted: #5c6778;
  --blue: #9a4736;
  --blue-dark: #7c382b;
  --blue-soft: #f1e6df;
  --rose: #8c6f54;
  --surface: #fbf8f1;
  --soft: #f1ebe2;
  --line: #d7d1c6;
  --shadow: none;
  --shadow-small: none;
  --radius: 8px;
}

/* projects/style-talking/static/css/index.css */
:root {
  --bg: #f6f1e8;
  --panel: #fbf8f1;
  --ink: #172235;
  --muted: #5c6778;
  --line: #d7d1c6;
  --soft: #f1ebe2;
  --accent: #9a4736;
  --accent-ink: #7c382b;
  --shadow: none;
}
```

In both project CSS files, replace hard-coded white card/label backgrounds with the corresponding surface variable, replace IConFace's `publication-header` radial background with `var(--surface)`, replace its `.section-title::after` linear gradient with `var(--blue)`, replace remaining hard-coded shadows with `none`, and remove `backdrop-filter`. Run:

```bash
rg -n -i 'gradient|backdrop-filter|blur\(' projects/iconface/static/css/index.css projects/style-talking/static/css/index.css
```

Expected: exit 1. Do not alter scientific images, gallery behavior, video URLs, or content layout.

- [ ] **Step 3: Patch the 8 clean Foundations with the same protocol**

Patch every listed Foundation except `foundations/image-generation-data-training/index.html`; Step 4 alone owns that dirty file. Each clean Foundation already has `.site-header`, `.module-nav`, and `main.lecture`. Add `class="has-shared-theme"` to its existing `<html>` element, remove its `twitter:*` meta lines, use the same head asset order from Step 2, insert exactly one standard toggle immediately before the first `</nav>` in `.module-nav`, and change:

```html
<main id="main" class="lecture has-shared-theme-shell">
```

Do not create another header. Do not wrap `main.lecture` in another `<main>`.

- [ ] **Step 4: Stage `image-generation-data-training/index.html` safely with a full Python replacement**

The working file is user-dirty. Build the staged version from HEAD, apply only the shared-theme patch to both the HEAD baseline and the working file, and verify CSS/img hashes never change.

Run the exact script below from repo root:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import re
import subprocess
import tempfile

repo = Path(".").resolve()
target = repo / "foundations/image-generation-data-training/index.html"
css_path = repo / "foundations/image-generation-data-training/static/css/index.css"
img_dir = repo / "foundations/image-generation-data-training/static/img"

def sha256(path):
    if path.is_file():
      return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()

css_before = sha256(css_path)
img_before = sha256(img_dir)
working_before = target.read_text()
head_text = subprocess.check_output(
    ["git", "show", f"HEAD:{target.relative_to(repo).as_posix()}"],
    text=True,
)

OLD_THEME_INIT = '''  <script>
    (function () {
      try {
        var theme = localStorage.getItem("theme");
        var systemDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (theme === "dark" || (theme === "system" && systemDark)) {
          document.documentElement.setAttribute("data-theme", "dark");
        } else {
          document.documentElement.removeAttribute("data-theme");
        }
      } catch (error) {
        document.documentElement.removeAttribute("data-theme");
      }
    })();
  </script>'''

SHARED_INIT = '\n'.join([
    '  <script src="/assets/js/theme-init.js"></script>',
    '  <link rel="stylesheet" href="/assets/css/theme-tokens.css">',
])

LOCAL_CSS = '  <link rel="stylesheet" href="./static/css/index.css">'
SHARED_AFTER_LOCAL = '\n'.join([
    LOCAL_CSS,
    '  <link rel="stylesheet" href="/assets/css/content-theme-bridge.css">',
    '  <script src="/assets/js/theme-controller.js" defer></script>',
    '  <script src="/assets/js/site.js" defer></script>',
])

TOGGLE = '<button type="button" class="theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Switch to dark mode">Theme</button>'

def patch_html(text):
    assert text.count('<html lang="zh-CN">') == 1
    assert text.count(OLD_THEME_INIT) == 1
    assert text.count(LOCAL_CSS) == 1
    assert text.count('<main id="main" class="lecture">') == 1
    text, twitter_count = re.subn(r'^  <meta name="twitter:[^\n]+\n', '', text, flags=re.MULTILINE)
    assert twitter_count == 4, f"expected four Twitter meta lines, got {twitter_count}"
    text = text.replace('<html lang="zh-CN">', '<html lang="zh-CN" class="has-shared-theme">', 1)
    text = text.replace(OLD_THEME_INIT, SHARED_INIT, 1)
    text = text.replace(LOCAL_CSS, SHARED_AFTER_LOCAL, 1)
    text = text.replace('</nav>', f'      {TOGGLE}\n    </nav>', 1)
    text = text.replace('<main id="main" class="lecture">', '<main id="main" class="lecture has-shared-theme-shell">', 1)
    assert text.count('data-theme-toggle') == 1
    assert 'name="twitter:' not in text.lower()
    return text

patched_head = patch_html(head_text)
patched_working = patch_html(working_before)

target.write_text(patched_working)
with tempfile.NamedTemporaryFile("w", delete=False) as handle:
    handle.write(patched_head)
    tmp_path = Path(handle.name)

blob = subprocess.check_output(["git", "hash-object", "-w", str(tmp_path)], text=True).strip()
tmp_path.unlink()
subprocess.check_call(["git", "update-index", "--cacheinfo", "100644", blob, target.relative_to(repo).as_posix()])

assert sha256(css_path) == css_before, "css hash changed"
assert sha256(img_dir) == img_before, "img hash changed"

cached = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
assert "foundations/image-generation-data-training/static/css/index.css" not in cached
assert not any(path.startswith("foundations/image-generation-data-training/static/img/") for path in cached)
PY
```

Expected:

- staged diff for `foundations/image-generation-data-training/index.html` contains only the shared-theme HTML class/assets/toggle/main-class patch plus removal of the four Twitter-specific meta lines
- CSS hash unchanged
- `static/img/` hash unchanged
- working-tree diff in that directory still belongs only to the user’s existing content plus the same minimal `index.html` patch

- [ ] **Step 5: Commit standalone integration with explicit file adds only**

```bash
git add assets/css/content-theme-bridge.css
git add projects/iconface/index.html projects/style-talking/index.html
git add projects/iconface/static/css/index.css projects/style-talking/static/css/index.css
git add foundations/aigc-llm-math/index.html foundations/generation-acceleration/index.html foundations/generation-distillation/index.html foundations/generation-math/index.html foundations/leetcode-hot100/index.html foundations/llm-interview-qa/index.html foundations/llm-mechanics/index.html foundations/video-generation/index.html
git diff --cached --name-only
git commit -m "feat: extend shared theme protocol to standalone pages"
```

Expected:

- staged paths include `foundations/image-generation-data-training/index.html`
- staged paths do not include `foundations/image-generation-data-training/static/css/index.css`
- staged paths do not include anything under `foundations/image-generation-data-training/static/img/`

### Task 8: Track Playwright properly, run final QA, and finish with a clean index

**Files:**

- Modify: `package.json`
- Modify: `package-lock.json`

- [ ] **Step 1: Replace old npm ownership with tracked Playwright QA**

Use `npm install --save-dev playwright` instead of hard-coding a version. Commit `package-lock.json`. Remove legacy dependencies and build scripts.

First replace `package.json` with the exact dependency-free manifest below; this preserves the repository identity established by the architecture plan while removing the retired bundler scripts. Then let npm write the Playwright version instead of inserting a fake or stale version by hand:

```json
{
  "name": "cosmicrealm-homepage",
  "version": "1.0.0",
  "description": "Build and QA helpers for cosmicrealm.github.io",
  "private": true,
  "repository": {
    "type": "git",
    "url": "https://github.com/cosmicrealm/cosmicrealm.github.io"
  },
  "bugs": {
    "url": "https://github.com/cosmicrealm/cosmicrealm.github.io/issues"
  },
  "homepage": "https://cosmicrealm.github.io",
  "scripts": {
    "test:structure": "python3 scripts/verify_homepage_theme_architecture.py",
    "test:theme": "node tests/theme_controller.test.cjs",
    "test:browser": "node tests/site_browser_qa.cjs"
  }
}
```

After `npm install --save-dev playwright`, npm adds exactly one `devDependencies.playwright` entry to that manifest and writes `package-lock.json`; do not hand-edit the resolved version.

The post-install reality should be:

- `package.json` records `playwright`
- `package-lock.json` is updated and committed
- `jquery`, `fitvids`, `jquery-smooth-scroll`, `plotly.js-dist-min`, `onchange`, and `uglify-js` are gone

- [ ] **Step 2: Install Playwright and commit the lockfile**

Run: `npm install --save-dev playwright`

Expected: `package.json` and `package-lock.json` update; `node_modules/playwright` exists locally.

Run:

```bash
node - <<'NODE'
const assert = require("node:assert/strict");
const pkg = require("./package.json");
assert.deepEqual(Object.keys(pkg.devDependencies || {}), ["playwright"]);
for (const name of ["jquery", "fitvids", "jquery-smooth-scroll", "plotly.js-dist-min", "onchange", "uglify-js"]) {
  assert.equal(Boolean((pkg.dependencies || {})[name] || (pkg.devDependencies || {})[name]), false, name);
}
NODE
```

Expected: exit 0.

```bash
git add package.json package-lock.json
git commit -m "test: track browser QA runtime"
```

- [ ] **Step 3: Run the full verification matrix**

Run: `python3 scripts/verify_homepage_theme_architecture.py`

Expected: pass.

Run: `node tests/theme_controller.test.cjs`

Expected: pass.

Run: `docker compose up -d`

Expected: site available at `http://127.0.0.1:4000/`.

Run: `CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" SITE_URL="http://127.0.0.1:4000" node tests/site_browser_qa.cjs`

Expected:

- empty same-origin `requestfailed`
- empty same-origin HTTP `>=400`
- empty console/page errors
- all 11 standalone pages reachable
- all widths pass overflow checks
- system light/dark and explicit persistence pass
- ARIA label and `aria-pressed` pass
- reduced-motion context passes
- no-JS nav retains internal links
- homepage loads no heavy renderers and no `https://cosmicrealm.github.io/` internal assets during local run
- screenshots written to `tmp/homepage-theme-qa/home-light.png` and `tmp/homepage-theme-qa/home-dark.png`

- [ ] **Step 4: If QA finds defects, fix and commit each repair separately**

Do not do a mass final `git add`. Any post-QA repair must be its own scoped commit with only the files needed for that fix.

- [ ] **Step 5: End with a clean index and preserved user dirties**

Run: `git status --short`

Expected:

- index clean
- no staged files remain
- only the known user dirties under:
  - `foundations/image-generation-data-training/index.html`
  - `foundations/image-generation-data-training/static/css/index.css`
  - `foundations/image-generation-data-training/static/img/`

If `git status --short` shows anything else, stop and resolve it before handoff.

## Self-Review

**Spec coverage:** This plan now covers the shared `theme-init.js`, deferred `theme-controller.js` and `site.js`, full `mountNavigation()` behavior, full `mountWritingFilter()` behavior, `aria-label` updates, base-path semantics, generic SEO plus Person JSON-LD, curated homepage data, schema-correct foundation cards, split homepage Sass, deletion of legacy runtime files, tracked `playwright` plus committed `package-lock.json`, all 9 Foundations + 2 projects, safe staging of the dirty `image-generation-data-training/index.html`, and final browser QA with screenshots.

**Placeholder scan:** No `TODO`, `TBD`, “apply edits”, “handle appropriately”, or “similar to Task N” placeholders remain. Every task includes exact files, code, commands, and expected outputs.

**Conflict check:** The plan contains no `_layouts/talk.html` work, assumes architecture migration already removed that layout, scopes legacy-string scans to source files instead of scanning plan docs or vendor output, treats optional external font / CDN / R2 media failures as non-blocking in browser QA, does not add a second header or nested `<main>` to standalone pages, does not reintroduce `share:` flags, does not keep `site_theme`, and does not touch the user’s dirty CSS or image files under `foundations/image-generation-data-training/static/`.
