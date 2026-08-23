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
    requestfailed.push(`${req.method()} ${url}`);
  });
  page.on("response", (res) => {
    const url = res.url();
    if (res.status() >= 400 && !isOptionalExternalUrl(url)) {
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

function assertCollectorsEmpty(collectors, label) {
  assert.deepEqual(collectors.requestfailed, [], `${label} had failed requests`);
  assert.deepEqual(collectors.badResponses, [], `${label} had HTTP error responses`);
  assert.deepEqual(collectors.consoleErrors, [], `${label} had console errors`);
  assert.deepEqual(collectors.pageErrors, [], `${label} had page errors`);
}

async function waitForDocumentStable(page) {
  await page.waitForLoadState("load");
  await page.evaluate(async () => {
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
    }
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
}

async function gotoReady(page, url, readySelector) {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.locator(readySelector).first().waitFor({ state: "visible" });
  await waitForDocumentStable(page);
}

async function assertNoOverflow(page, width, height, label) {
  await page.setViewportSize({ width, height });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  assert.equal(metrics.scrollWidth, metrics.clientWidth, `${label} overflow detected at ${width}px`);
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
  assert.equal(resources.some((name) => name.startsWith("https://cosmicrealm.github.io/")), false, "homepage loaded production-origin internal resource locally");
}

async function assertHomepageEditorialRows(page) {
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const sections = await page.evaluate(() => {
    const referenceFontSize = getComputedStyle(document.querySelector(".compact-list a")).fontSize;
    const selectors = {
      "Recent Project": [".home-card--project", "h3"],
      "Recent Publications": [".publication-card", ".publication-card__title"],
      "Foundations": [".home-card--foundation", "h3"],
    };
    return Object.entries(selectors).map(([heading, [selector, titleSelector]]) => {
      const title = [...document.querySelectorAll(".section-heading h2")]
        .find((node) => node.textContent.trim() === heading);
      const section = title?.closest(".home-section");
      const items = [...(section?.querySelectorAll(selector) || [])];
      return {
        heading,
        count: items.length,
        rows: new Set(items.map((item) => Math.round(item.getBoundingClientRect().top))).size,
        widths: items.map((item) => item.getBoundingClientRect().width),
        titleFontSizes: items.map((item) => getComputedStyle(item.querySelector(titleSelector)).fontSize),
        referenceFontSize,
        containerWidth: section?.getBoundingClientRect().width || 0,
      };
    });
  });
  for (const section of sections) {
    assert.equal(section.count, 3, `${section.heading} should keep its three recent items`);
    assert.equal(section.rows, 3, `${section.heading} should render one item per row`);
    assert.equal(
      section.widths.every((width) => width >= section.containerWidth * 0.95),
      true,
      `${section.heading} rows should span the section width`
    );
    assert.equal(
      section.titleFontSizes.every((fontSize) => fontSize === section.referenceFontSize),
      true,
      `${section.heading} titles should match Recent Writing font size`
    );
  }

  const voiceStudioAudio = await page.evaluate(() => {
    const card = [...document.querySelectorAll(".home-card--project")]
      .find((item) => item.querySelector("h3")?.textContent.trim() === "Voice Studio");
    const audio = card?.querySelector("audio");
    return audio
      ? {
          controls: audio.controls,
          preload: audio.preload,
          src: audio.currentSrc || audio.querySelector("source")?.src || "",
        }
      : null;
  });
  assert.ok(voiceStudioAudio, "Voice Studio should include an inline audio preview");
  assert.equal(voiceStudioAudio.controls, true, "Voice Studio audio preview should expose playback controls");
  assert.equal(voiceStudioAudio.preload, "none", "Voice Studio audio should not preload on the homepage");
  assert.match(voiceStudioAudio.src, /Voice-Studio-0\.01-dialogue-demo\.mp3$/, "Voice Studio should use the published v0.01 dialogue sample");
}

async function assertNoProductionOriginResources(page, label) {
  const resources = await page.evaluate(() =>
    performance.getEntriesByType("resource").map((entry) => entry.name)
  );
  assert.equal(
    resources.some((name) => name.startsWith("https://cosmicrealm.github.io/")),
    false,
    `${label} loaded production-origin internal resource locally`
  );
}

async function assertThemePersistence(browser, colorScheme) {
  const context = await browser.newContext({ colorScheme, reducedMotion: "reduce", viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const collectors = bindCollectors(page);
  try {
    await gotoReady(page, `${SITE_URL}/`, "[data-theme-toggle]");
    await assertReducedMotion(page);
    const initial = await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light");
    if (colorScheme === "dark") assert.equal(initial, "dark");
    if (colorScheme === "light") assert.equal(initial, "light");
    await page.locator("[data-theme-toggle]").first().click();
    const explicit = await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light");
    assert.notEqual(explicit, initial, `${colorScheme} theme toggle did not change the resolved theme`);
    await assertNoProductionOriginResources(page, `${colorScheme} persistence homepage`);

    await page.reload({ waitUntil: "domcontentloaded" });
    await page.locator("[data-theme-toggle]").first().waitFor({ state: "visible" });
    await waitForDocumentStable(page);
    assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light"), explicit);
    await assertNoProductionOriginResources(page, `${colorScheme} persistence homepage reload`);

    await gotoReady(page, `${SITE_URL}/writing/`, "main");
    assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light"), explicit);
    await assertNoProductionOriginResources(page, `${colorScheme} persistence writing`);

    await gotoReady(page, `${SITE_URL}/projects/iconface/`, "[data-theme-toggle]");
    assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-theme") || "light"), explicit);
    await assertNoProductionOriginResources(page, `${colorScheme} persistence iconface`);
    assertCollectorsEmpty(collectors, `${colorScheme} persistence flow`);
  } finally {
    await context.close();
  }
}

async function assertWritingAccentPalette(browser, colorScheme) {
  const context = await createQaContext(browser, colorScheme);
  const page = await context.newPage();
  const collectors = bindCollectors(page);
  try {
    await gotoReady(page, `${SITE_URL}/writing/`, "#main.blog-filter-layout");
    const palette = await page.evaluate(() => {
      const rootStyle = getComputedStyle(document.documentElement);
      const mainStyle = getComputedStyle(document.querySelector("#main.blog-filter-layout"));
      const activeFilterStyle = getComputedStyle(document.querySelector(".blog-filter__clear.is-active"));
      const dateStyle = getComputedStyle(document.querySelector(".blog-list__date time"));
      return {
        rootAccent: rootStyle.getPropertyValue("--cr-accent").trim(),
        accent: mainStyle.getPropertyValue("--cr-accent").trim(),
        accentStrong: mainStyle.getPropertyValue("--cr-accent-strong").trim(),
        accentSoft: mainStyle.getPropertyValue("--cr-accent-soft").trim(),
        pageBackground: getComputedStyle(document.body).backgroundColor,
        activeBackground: activeFilterStyle.backgroundColor,
        activeColor: activeFilterStyle.color,
        dateBackground: dateStyle.backgroundColor,
      };
    });
    const expected = colorScheme === "dark"
      ? {
          rootAccent: "#f5f5f5",
          accent: "#a8d5b8",
          accentStrong: "#c8ead3",
          accentSoft: "rgba(168, 213, 184, 0.14)",
          pageBackground: "rgb(0, 0, 0)",
          activeBackground: "rgb(200, 234, 211)",
          activeColor: "rgb(0, 0, 0)",
          dateBackground: "rgba(168, 213, 184, 0.14)",
        }
      : {
          rootAccent: "#111111",
          accent: "#5f8f72",
          accentStrong: "#315d46",
          accentSoft: "#eaf4ed",
          pageBackground: "rgb(255, 255, 255)",
          activeBackground: "rgb(49, 93, 70)",
          activeColor: "rgb(255, 255, 255)",
          dateBackground: "rgb(234, 244, 237)",
        };
    assert.deepEqual(palette, expected, `${colorScheme} Writing page should use its scoped green accent palette`);
    assertCollectorsEmpty(collectors, `${colorScheme} Writing accent palette`);
  } finally {
    await context.close();
  }
}

async function captureHomepagePreview(browser, colorScheme, filename) {
  const context = await browser.newContext({
    colorScheme,
    reducedMotion: "reduce",
    viewport: { width: 1440, height: 960 },
  });
  const page = await context.newPage();
  const collectors = bindCollectors(page);
  try {
    await gotoReady(page, `${SITE_URL}/`, ".home-hero h1");
    const palette = await page.evaluate(() => {
      const rootStyle = getComputedStyle(document.documentElement);
      const primaryAction = document.querySelector(".home-hero .hero-actions a:first-child");
      const eyebrow = document.querySelector(".home-hero__eyebrow");
      return {
        theme: document.documentElement.getAttribute("data-theme") || "light",
        background: getComputedStyle(document.body).backgroundColor,
        accent: rootStyle.getPropertyValue("--cr-accent").trim(),
        accentStrong: rootStyle.getPropertyValue("--cr-accent-strong").trim(),
        warm: rootStyle.getPropertyValue("--cr-warm").trim(),
        primaryBackground: getComputedStyle(primaryAction).backgroundColor,
        primaryColor: getComputedStyle(primaryAction).color,
        eyebrowColor: getComputedStyle(eyebrow).color,
      };
    });
    assert.equal(palette.theme, colorScheme, `${colorScheme} preview resolved the wrong theme`);
    assert.equal(
      palette.background,
      colorScheme === "dark" ? "rgb(0, 0, 0)" : "rgb(255, 255, 255)",
      `${colorScheme} homepage canvas is not pure ${colorScheme === "dark" ? "black" : "white"}`
    );
    const expectedPalette = colorScheme === "dark"
      ? {
          accent: "#f5f5f5",
          accentStrong: "#ffffff",
          warm: "#b5b5b5",
          primaryBackground: "rgb(255, 255, 255)",
          primaryColor: "rgb(0, 0, 0)",
          eyebrowColor: "rgb(245, 245, 245)",
        }
      : {
          accent: "#111111",
          accentStrong: "#000000",
          warm: "#555555",
          primaryBackground: "rgb(0, 0, 0)",
          primaryColor: "rgb(255, 255, 255)",
          eyebrowColor: "rgb(17, 17, 17)",
        };
    assert.deepEqual(
      {
        accent: palette.accent,
        accentStrong: palette.accentStrong,
        warm: palette.warm,
        primaryBackground: palette.primaryBackground,
        primaryColor: palette.primaryColor,
        eyebrowColor: palette.eyebrowColor,
      },
      expectedPalette,
      `${colorScheme} homepage should use the monochrome accent treatment`
    );
    await page.screenshot({
      path: path.join(ARTIFACT_DIR, filename),
      fullPage: false,
      animations: "disabled",
    });
    assertCollectorsEmpty(collectors, `${colorScheme} homepage preview`);
  } finally {
    await context.close();
  }
}

function createQaContext(browser, colorScheme = "light", viewport = { width: 1440, height: 960 }) {
  return browser.newContext({ colorScheme, reducedMotion: "reduce", viewport });
}

async function main() {
  ensureArtifacts();
  const browser = await chromium.launch({ headless: true, executablePath: CHROME_PATH });
  try {
    await assertThemePersistence(browser, "light");
    await assertThemePersistence(browser, "dark");
    await assertWritingAccentPalette(browser, "light");
    await assertWritingAccentPalette(browser, "dark");

    const context = await createQaContext(browser);
    const homepage = await context.newPage();
    const homepageCollectors = bindCollectors(homepage);
    try {
      await gotoReady(homepage, `${SITE_URL}/`, "[data-theme-toggle]");
      await assertThemeToggle(homepage);
      await assertReducedMotion(homepage);
      for (const width of WIDTHS) {
        await assertNoOverflow(homepage, width, 900, "homepage");
      }
      await assertHomepageEditorialRows(homepage);
      await waitForDocumentStable(homepage);
      await assertHomepageResources(homepage);
      assertCollectorsEmpty(homepageCollectors, "homepage");
    } finally {
      await homepage.close();
    }

    for (const url of STANDALONE_PATHS) {
      const routeContext = await createQaContext(browser);
      const standalone = await routeContext.newPage();
      const routeCollectors = bindCollectors(standalone);
      try {
        await gotoReady(standalone, `${SITE_URL}${url}`, "[data-theme-toggle]");
        assert.ok(await standalone.locator("[data-theme-toggle]").count(), `missing toggle on ${url}`);
        await assertThemeToggle(standalone);
        await assertReducedMotion(standalone);
        for (const width of WIDTHS) {
          await assertNoOverflow(standalone, width, 900, url);
        }
        await waitForDocumentStable(standalone);
        await assertNoProductionOriginResources(standalone, url);
        assertCollectorsEmpty(routeCollectors, url);
      } finally {
        await routeContext.close();
      }
    }
    await context.close();

    await captureHomepagePreview(browser, "light", "home-light.png");
    await captureHomepagePreview(browser, "dark", "home-dark.png");

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
        theme: document.documentElement.getAttribute("data-theme"),
      };
    });
    assert.ok(navState && navState.menuVisible && navState.toggleHidden, "no-JS nav fallback is not visibly expanded");
    assert.equal(navState.theme, "light", "no-JS theme fallback must remain light");
    await noJs.close();
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
