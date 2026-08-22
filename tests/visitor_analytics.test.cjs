"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  CACHE_KEY,
  createVisitorAnalytics,
  formatCount,
} = require("../assets/js/visitor-analytics.js");

function createElement(dataset = {}) {
  const attrs = {};
  const classes = new Set();
  return {
    attrs,
    dataset,
    hidden: false,
    style: {
      values: {},
      setProperty(key, value) { this.values[key] = value; },
    },
    classList: {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    setAttribute(key, value) { attrs[key] = value; },
    getAttribute(key) { return attrs[key] ?? null; },
    textContent: "",
  };
}

function createComponent() {
  const total = createElement();
  const today = createElement();
  const countries = createElement();
  const status = createElement();
  const accessible = createElement();
  const cn = createElement({ countryCode: "CN" });
  const us = createElement({ countryCode: "US" });
  const fr = createElement({ countryCode: "FR" });
  const selectors = {
    "[data-visitor-total]": total,
    "[data-visitor-today]": today,
    "[data-visitor-countries]": countries,
    "[data-visitor-status]": status,
    "[data-visitor-accessible-summary]": accessible,
  };
  return {
    component: {
      classList: createElement().classList,
      querySelector(selector) { return selectors[selector] || null; },
      querySelectorAll(selector) {
        return selector === "[data-country-code]" ? [cn, us, fr] : [];
      },
    },
    countries: { cn, fr, us },
    nodes: { accessible, countries, status, today, total },
  };
}

function createStorage(seed = {}) {
  const values = new Map(Object.entries(seed));
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, value); },
  };
}

function createWindow(overrides = {}) {
  return {
    location: { hostname: "cosmicrealm.github.io" },
    navigator: { doNotTrack: "0", globalPrivacyControl: false },
    setTimeout,
    clearTimeout,
    ...overrides,
  };
}

const summary = {
  totalViews: 12846,
  visitorsToday: 37,
  countriesReached: 2,
  countries: [
    { code: "CN", views: 5210 },
    { code: "US", views: 1880 },
  ],
  updatedAt: "2026-08-22T10:00:00.000Z",
};

test("formats public counters without compacting their meaning", () => {
  assert.equal(formatCount(12846), "12,846");
  assert.equal(formatCount(0), "0");
  assert.equal(formatCount(undefined), "—");
});

test("collects production visits and renders a country heat map", async () => {
  const calls = [];
  const { component, countries, nodes } = createComponent();
  const fetchImpl = async (url, options = {}) => {
    calls.push({ options, url });
    if (options.method === "POST") return new Response(null, { status: 202 });
    return new Response(JSON.stringify(summary), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  const analytics = createVisitorAnalytics({
    component,
    endpoint: "https://visitor.example",
    fetchImpl,
    storage: createStorage(),
    window: createWindow(),
  });

  await analytics.mount();

  assert.equal(calls.length, 2);
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.keepalive, true);
  assert.equal(calls[1].url, "https://visitor.example/v1/summary");
  assert.equal(nodes.total.textContent, "12,846");
  assert.equal(nodes.today.textContent, "37");
  assert.equal(nodes.countries.textContent, "2");
  assert.match(nodes.accessible.textContent, /China/i);
  assert.equal(countries.cn.classList.contains("is-visited"), true);
  assert.equal(countries.us.classList.contains("is-visited"), true);
  assert.equal(countries.fr.classList.contains("is-visited"), false);
  assert.equal(component.classList.contains("is-ready"), true);
});

test("respects browser privacy signals", async () => {
  const calls = [];
  const analytics = createVisitorAnalytics({
    component: null,
    endpoint: "https://visitor.example",
    fetchImpl: async (...args) => { calls.push(args); },
    storage: createStorage(),
    window: createWindow({
      navigator: { doNotTrack: "1", globalPrivacyControl: true },
    }),
  });

  await analytics.mount();
  assert.equal(calls.length, 0);
});

test("does not pollute production statistics from localhost", async () => {
  const calls = [];
  const analytics = createVisitorAnalytics({
    component: null,
    endpoint: "https://visitor.example",
    fetchImpl: async (...args) => { calls.push(args); },
    storage: createStorage(),
    window: createWindow({ location: { hostname: "localhost" } }),
  });

  await analytics.mount();
  assert.equal(calls.length, 0);
});

test("falls back to a recent cached public summary without showing zero", async () => {
  const { component, nodes } = createComponent();
  const cachedAt = Date.parse("2026-08-22T10:00:00.000Z");
  const storage = createStorage({
    [CACHE_KEY]: JSON.stringify({ cachedAt, summary }),
  });
  const analytics = createVisitorAnalytics({
    component,
    endpoint: "https://visitor.example",
    fetchImpl: async () => { throw new Error("offline"); },
    now: () => new Date("2026-08-22T11:00:00.000Z"),
    storage,
    window: createWindow(),
  });

  const result = await analytics.loadSummary();

  assert.equal(result.source, "cache");
  assert.equal(nodes.total.textContent, "12,846");
  assert.match(nodes.status.textContent, /cached/i);
  assert.notEqual(nodes.total.textContent, "0");
});

test("shows an unavailable state when no live or cached summary exists", async () => {
  const { component, nodes } = createComponent();
  const analytics = createVisitorAnalytics({
    component,
    endpoint: "https://visitor.example",
    fetchImpl: async () => { throw new Error("offline"); },
    storage: createStorage(),
    window: createWindow(),
  });

  const result = await analytics.loadSummary();

  assert.equal(result.source, "unavailable");
  assert.equal(nodes.total.textContent, "—");
  assert.match(nodes.status.textContent, /temporarily unavailable/i);
  assert.equal(component.classList.contains("is-unavailable"), true);
});
