import assert from "node:assert/strict";
import test from "node:test";

import {
  createWorker,
  hashVisitor,
  normalizeCountry,
  utcDay,
} from "../src/index.mjs";

class MemoryRepository {
  constructor() {
    this.visits = [];
    this.cleanupBefore = null;
    this.summary = {
      totalViews: 12846,
      visitorsToday: 37,
      countriesReached: 2,
      countries: [
        { code: "CN", views: 5210 },
        { code: "US", views: 1880 },
      ],
      updatedAt: "2026-08-22T10:00:00.000Z",
    };
  }

  async recordVisit(visit) {
    this.visits.push(visit);
    return { counted: true };
  }

  async getSummary() {
    return this.summary;
  }

  async cleanup(beforeDay) {
    this.cleanupBefore = beforeDay;
  }
}

const fixedNow = new Date("2026-08-22T10:00:00.000Z");

function request(path, options = {}) {
  return new Request(`https://visitor-api.example${path}`, options);
}

function createHarness() {
  const repository = new MemoryRepository();
  const worker = createWorker({
    createRepository: () => repository,
    now: () => fixedNow,
  });
  const env = {
    ALLOWED_ORIGINS: "https://cosmicrealm.github.io,http://localhost:4000",
    MAX_VIEWS_PER_VISITOR: "50",
    MIN_COUNTRY_VIEWS: "3",
    VISITOR_HASH_SECRET: "test-secret-with-enough-entropy",
  };
  return { env, repository, worker };
}

test("normalizes supported Cloudflare country codes", () => {
  assert.equal(normalizeCountry("cn"), "CN");
  assert.equal(normalizeCountry("US"), "US");
  assert.equal(normalizeCountry("XX"), null);
  assert.equal(normalizeCountry("T1"), null);
  assert.equal(normalizeCountry("not-a-country"), null);
});

test("creates a stable visitor hash for one UTC day and rotates it the next day", async () => {
  const first = await hashVisitor("203.0.113.8", "2026-08-22", "secret");
  const repeat = await hashVisitor("203.0.113.8", "2026-08-22", "secret");
  const nextDay = await hashVisitor("203.0.113.8", "2026-08-23", "secret");

  assert.equal(first, repeat);
  assert.notEqual(first, nextDay);
  assert.doesNotMatch(first, /203\.0\.113\.8/);
  assert.match(first, /^[a-f0-9]{64}$/);
});

test("rejects collection from an unapproved origin", async () => {
  const { env, repository, worker } = createHarness();
  const response = await worker.fetch(request("/v1/collect", {
    method: "POST",
    headers: {
      Origin: "https://counter-spam.example",
      "CF-Connecting-IP": "203.0.113.8",
    },
  }), env, {});

  assert.equal(response.status, 403);
  assert.equal(repository.visits.length, 0);
});

test("collects an allowed page view without persisting the raw IP", async () => {
  const { env, repository, worker } = createHarness();
  const incoming = request("/v1/collect", {
    method: "POST",
    headers: {
      Origin: "https://cosmicrealm.github.io",
      "CF-Connecting-IP": "203.0.113.8",
      "CF-IPCountry": "CN",
    },
  });
  Object.defineProperty(incoming, "cf", {
    configurable: true,
    value: { country: "CN" },
  });

  const response = await worker.fetch(incoming, env, {});

  assert.equal(response.status, 202);
  assert.equal(response.headers.get("Access-Control-Allow-Origin"), "https://cosmicrealm.github.io");
  assert.equal(repository.visits.length, 1);
  assert.equal(repository.visits[0].day, "2026-08-22");
  assert.equal(repository.visits[0].country, "CN");
  assert.equal(repository.visits[0].maxViewsPerVisitor, 50);
  assert.notEqual(repository.visits[0].visitorHash, "203.0.113.8");
  assert.equal(JSON.stringify(repository.visits).includes("203.0.113.8"), false);
});

test("returns only the public aggregate summary", async () => {
  const { env, worker } = createHarness();
  const response = await worker.fetch(request("/v1/summary", {
    headers: { Origin: "https://cosmicrealm.github.io" },
  }), env, {});
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "public, max-age=300");
  assert.deepEqual(payload, {
    totalViews: 12846,
    visitorsToday: 37,
    countriesReached: 2,
    countries: [
      { code: "CN", views: 5210 },
      { code: "US", views: 1880 },
    ],
    updatedAt: "2026-08-22T10:00:00.000Z",
  });
  assert.equal(JSON.stringify(payload).includes("visitorHash"), false);
});

test("scheduled cleanup retains only today and yesterday visitor hashes", async () => {
  const { env, repository, worker } = createHarness();
  await worker.scheduled({}, env, {});
  assert.equal(repository.cleanupBefore, "2026-08-21");
});

test("uses UTC calendar days", () => {
  assert.equal(utcDay(new Date("2026-08-22T23:59:59.999Z")), "2026-08-22");
  assert.equal(utcDay(new Date("2026-08-23T00:00:00.000Z")), "2026-08-23");
});
