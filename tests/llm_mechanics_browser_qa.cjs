const assert = require("node:assert/strict");
const { chromium } = require("playwright");

const PAGE_URL = process.env.LLM_MECHANICS_URL
  || "http://127.0.0.1:4173/foundations/llm-mechanics/";
const CHROME_PATH = process.env.CHROME_PATH
  || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

async function collectPageErrors(page) {
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  return errors;
}

async function checkDesktop(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = await collectPageErrors(page);
  const response = await page.goto(PAGE_URL, { waitUntil: "networkidle" });
  assert.equal(response.status(), 200, "desktop page should return HTTP 200");

  const layout = await page.evaluate(() => {
    const toc = document.querySelector(".lecture-toc").getBoundingClientRect();
    const lecture = document.querySelector(".lecture").getBoundingClientRect();
    return {
      tocRight: toc.right,
      lectureLeft: lecture.left,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      labs: document.querySelectorAll(".interactive-lab").length,
      examples: document.querySelectorAll(".example-card").length,
    };
  });

  assert.ok(layout.tocRight <= layout.lectureLeft, "desktop TOC must not overlap lecture content");
  assert.equal(layout.scrollWidth, layout.clientWidth, "desktop page must not overflow horizontally");
  assert.equal(layout.labs, 6, "desktop page should expose six interactive labs");
  assert.equal(layout.examples, 12, "desktop page should expose twelve example cards");

  await page.locator("#bpeStep").click();
  assert.equal(await page.locator("#bpeStepCount").textContent(), "1", "BPE lab should execute one merge");
  assert.notEqual(await page.locator("#bpePair").textContent(), "—", "BPE lab should report the merged pair");

  await page.locator('[data-decoder-mode="rope"]').click();
  assert.match(await page.locator("#decoderReadout").textContent(), /旋转角.*范数/, "RoPE lab should expose its invariant");

  await page.locator("#teacherFault").check();
  assert.match(await page.locator("#teacherReadout").textContent(), /错误模式/, "teacher-forcing lab should explain the injected shift fault");

  await page.locator("#promptLength").evaluate((element) => {
    element.value = "16";
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator("#generatedLength").evaluate((element) => {
    element.value = "8";
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  assert.equal(await page.locator("#naiveWork").textContent(), "156", "prefill lab should calculate naive work");
  assert.equal(await page.locator("#cachedWork").textContent(), "23", "prefill lab should calculate cached work");

  await page.locator("#kvContext").evaluate((element) => {
    element.value = "32768";
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  assert.equal(await page.locator("#kvPerToken").textContent(), "128.00 KiB", "KV lab should calculate per-token cache bytes");
  assert.equal(await page.locator("#kvTotal").textContent(), "4.00 GiB", "KV lab should calculate total cache bytes");

  for (let step = 0; step < 8; step += 1) {
    if (await page.locator("#sampleNext").isDisabled()) break;
    await page.locator("#sampleNext").click();
  }
  assert.equal(await page.locator("#stopState").textContent(), "stopped", "sampling lab should reach an EOS or max-token stop state");
  assert.ok(await page.locator("#generatedTokens span").count() > 0, "sampling lab should emit visible tokens");
  assert.deepEqual(errors, [], "desktop page should have no console or runtime errors");
  await page.close();
}

async function checkResponsiveViewport(browser, width, height) {
  const page = await browser.newPage({
    viewport: { width, height },
    isMobile: width <= 440,
    hasTouch: width <= 440,
  });
  const errors = await collectPageErrors(page);
  const response = await page.goto(PAGE_URL, { waitUntil: "networkidle" });
  assert.equal(response.status(), 200, "mobile page should return HTTP 200");

  const viewport = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  assert.equal(viewport.scrollWidth, viewport.clientWidth, `${width}px page must not overflow horizontally`);

  await page.locator(".toc-toggle").click();
  await page.waitForTimeout(250);
  const toc = await page.locator(".lecture-toc").boundingBox();
  assert.ok(toc, "mobile TOC should be visible after opening");
  assert.ok(toc.x >= 0, `${width}px TOC should start inside the viewport`);
  assert.ok(
    toc.x + toc.width <= width,
    `${width}px TOC should end inside the viewport: ${JSON.stringify(toc)}`,
  );
  assert.deepEqual(errors, [], `${width}px page should have no console or runtime errors`);
  await page.close();
}

async function main() {
  const browser = await chromium.launch({ headless: true, executablePath: CHROME_PATH });
  try {
    await checkDesktop(browser);
    await checkResponsiveViewport(browser, 1024, 768);
    await checkResponsiveViewport(browser, 768, 1024);
    await checkResponsiveViewport(browser, 390, 844);
    console.log("LLM mechanics browser QA passed");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
