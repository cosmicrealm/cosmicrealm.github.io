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
