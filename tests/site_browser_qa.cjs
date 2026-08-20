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
  assert.equal(resources.some((name) => name.startsWith("https://cosmicrealm.github.io/")), false, "homepage loaded production-origin internal resource locally");
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

async function main() {
  ensureArtifacts();
  const browser = await chromium.launch({ headless: true, executablePath: CHROME_PATH });
  try {
    await assertThemePersistence(browser, "light");
    await assertThemePersistence(browser, "dark");

    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, reducedMotion: "reduce" });
    const homepage = await context.newPage();
    const homepageCollectors = bindCollectors(homepage);
    try {
      await gotoReady(homepage, `${SITE_URL}/`, "[data-theme-toggle]");
      await assertThemeToggle(homepage);
      await assertReducedMotion(homepage);
      for (const width of WIDTHS) {
        await assertNoOverflow(homepage, width, 900);
      }
      await homepage.screenshot({ path: path.join(ARTIFACT_DIR, "home-light.png"), fullPage: true });
      await homepage.locator("[data-theme-toggle]").first().click();
      await homepage.screenshot({ path: path.join(ARTIFACT_DIR, "home-dark.png"), fullPage: true });
      await waitForDocumentStable(homepage);
      await assertHomepageResources(homepage);
      assertCollectorsEmpty(homepageCollectors, "homepage");
    } finally {
      await homepage.close();
    }

    for (const url of STANDALONE_PATHS) {
      const standalone = await context.newPage();
      const routeCollectors = bindCollectors(standalone);
      try {
        await gotoReady(standalone, `${SITE_URL}${url}`, "[data-theme-toggle]");
        assert.ok(await standalone.locator("[data-theme-toggle]").count(), `missing toggle on ${url}`);
        await assertThemeToggle(standalone);
        await assertReducedMotion(standalone);
        for (const width of WIDTHS) {
          await assertNoOverflow(standalone, width, 900);
        }
        await waitForDocumentStable(standalone);
        await assertNoProductionOriginResources(standalone, url);
        assertCollectorsEmpty(routeCollectors, url);
      } finally {
        await standalone.close();
      }
    }
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
