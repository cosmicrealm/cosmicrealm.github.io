const DEFAULT_ALLOWED_ORIGINS = [
  "https://cosmicrealm.github.io",
  "http://localhost:4000",
  "http://127.0.0.1:4000",
];

const PUBLIC_CACHE_SECONDS = 300;

export function utcDay(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

export function normalizeCountry(country) {
  const normalized = String(country || "").trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(normalized) || normalized === "XX" || normalized === "T1") return null;
  return normalized;
}

export async function hashVisitor(ipAddress, day, secret) {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { hash: "SHA-256", name: "HMAC" },
    false,
    ["sign"],
  );
  const digest = await crypto.subtle.sign("HMAC", key, encoder.encode(`${day}:${ipAddress}`));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function parseInteger(value, fallback, minimum, maximum) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, parsed));
}

function allowedOrigins(env) {
  const configured = String(env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
  return new Set(configured.length > 0 ? configured : DEFAULT_ALLOWED_ORIGINS);
}

function corsHeaders(origin, allowed) {
  const headers = new Headers({ Vary: "Origin" });
  if (origin && allowed.has(origin)) headers.set("Access-Control-Allow-Origin", origin);
  return headers;
}

function emptyResponse(status, origin, allowed) {
  return new Response(null, { headers: corsHeaders(origin, allowed), status });
}

function jsonResponse(payload, status, origin, allowed) {
  const headers = corsHeaders(origin, allowed);
  headers.set("Cache-Control", `public, max-age=${PUBLIC_CACHE_SECONDS}`);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("X-Content-Type-Options", "nosniff");
  return new Response(JSON.stringify(payload), { headers, status });
}

function safePublicSummary(summary) {
  return {
    totalViews: Math.max(0, Number(summary.totalViews) || 0),
    visitorsToday: Math.max(0, Number(summary.visitorsToday) || 0),
    countriesReached: Math.max(0, Number(summary.countriesReached) || 0),
    countries: Array.isArray(summary.countries)
      ? summary.countries
        .map((entry) => ({
          code: normalizeCountry(entry.code),
          views: Math.max(0, Number(entry.views) || 0),
        }))
        .filter((entry) => entry.code && entry.views > 0)
      : [],
    updatedAt: summary.updatedAt || null,
  };
}

export class D1AnalyticsRepository {
  constructor(database) {
    this.database = database;
  }

  async recordVisit({ country, day, maxViewsPerVisitor, seenAt, visitorHash }) {
    await this.database.prepare(`
      INSERT OR IGNORE INTO visitor_days (day, visitor_hash, country, counted_views, last_seen_at)
      VALUES (?1, ?2, ?3, 0, ?4)
    `).bind(day, visitorHash, country, seenAt).run();

    const visitorUpdate = await this.database.prepare(`
      UPDATE visitor_days
      SET counted_views = counted_views + 1, last_seen_at = ?3
      WHERE day = ?1 AND visitor_hash = ?2 AND counted_views < ?4
    `).bind(day, visitorHash, seenAt, maxViewsPerVisitor).run();
    const counted = Number(visitorUpdate.meta?.changes || 0) > 0;
    if (!counted) return { counted: false };

    const statements = [
      this.database.prepare(`
        INSERT INTO site_totals (id, total_views, updated_at)
        VALUES (1, 1, ?1)
        ON CONFLICT(id) DO UPDATE SET
          total_views = total_views + 1,
          updated_at = excluded.updated_at
      `).bind(seenAt),
      this.database.prepare(`
        INSERT INTO daily_totals (day, pageviews)
        VALUES (?1, 1)
        ON CONFLICT(day) DO UPDATE SET pageviews = pageviews + 1
      `).bind(day),
    ];
    if (country) {
      statements.push(this.database.prepare(`
        INSERT INTO country_totals (country, views)
        VALUES (?1, 1)
        ON CONFLICT(country) DO UPDATE SET views = views + 1
      `).bind(country));
    }
    await this.database.batch(statements);
    return { counted: true };
  }

  async getSummary({ day, minCountryViews }) {
    const site = await this.database.prepare(`
      SELECT total_views AS totalViews, updated_at AS updatedAt
      FROM site_totals WHERE id = 1
    `).first();
    const today = await this.database.prepare(`
      SELECT COUNT(*) AS visitorsToday
      FROM visitor_days WHERE day = ?1
    `).bind(day).first();
    const countryResult = await this.database.prepare(`
      SELECT country AS code, views
      FROM country_totals
      WHERE views >= ?1
      ORDER BY views DESC, country ASC
    `).bind(minCountryViews).all();
    const countries = countryResult.results || [];
    return {
      totalViews: site?.totalViews || 0,
      visitorsToday: today?.visitorsToday || 0,
      countriesReached: countries.length,
      countries,
      updatedAt: site?.updatedAt || null,
    };
  }

  async cleanup(beforeDay) {
    await this.database.prepare("DELETE FROM visitor_days WHERE day < ?1").bind(beforeDay).run();
  }
}

function createDefaultRepository(env) {
  if (!env.DB) throw new Error("D1 binding DB is required");
  return new D1AnalyticsRepository(env.DB);
}

export function createWorker(options = {}) {
  const createRepository = options.createRepository || createDefaultRepository;
  const now = options.now || (() => new Date());

  return {
    async fetch(request, env) {
      const url = new URL(request.url);
      const origin = request.headers.get("Origin");
      const allowed = allowedOrigins(env);
      const originAllowed = Boolean(origin && allowed.has(origin));

      if (request.method === "OPTIONS") {
        if (!originAllowed) return emptyResponse(403, origin, allowed);
        const headers = corsHeaders(origin, allowed);
        headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        headers.set("Access-Control-Max-Age", "86400");
        return new Response(null, { headers, status: 204 });
      }

      if (origin && !originAllowed) return emptyResponse(403, origin, allowed);
      const repository = createRepository(env);

      if (url.pathname === "/v1/collect" && request.method === "POST") {
        const ipAddress = request.headers.get("CF-Connecting-IP");
        if (!ipAddress) return emptyResponse(202, origin, allowed);
        if (!env.VISITOR_HASH_SECRET) return emptyResponse(503, origin, allowed);

        const current = now();
        const day = utcDay(current);
        const visitorHash = await hashVisitor(ipAddress, day, env.VISITOR_HASH_SECRET);
        const country = normalizeCountry(request.cf?.country || request.headers.get("CF-IPCountry"));
        await repository.recordVisit({
          country,
          day,
          maxViewsPerVisitor: parseInteger(env.MAX_VIEWS_PER_VISITOR, 50, 1, 500),
          seenAt: current.toISOString(),
          visitorHash,
        });
        return emptyResponse(202, origin, allowed);
      }

      if (url.pathname === "/v1/summary" && request.method === "GET") {
        const summary = await repository.getSummary({
          day: utcDay(now()),
          minCountryViews: parseInteger(env.MIN_COUNTRY_VIEWS, 3, 1, 1000),
        });
        return jsonResponse(safePublicSummary(summary), 200, origin, allowed);
      }

      return emptyResponse(404, origin, allowed);
    },

    async scheduled(_event, env) {
      const cutoff = new Date(now());
      cutoff.setUTCDate(cutoff.getUTCDate() - 1);
      await createRepository(env).cleanup(utcDay(cutoff));
    },
  };
}

export default createWorker();
