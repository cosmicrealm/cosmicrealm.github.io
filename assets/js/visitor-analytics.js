"use strict";

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CosmicVisitorAnalytics = api;
}(typeof window !== "undefined" ? window : null, function () {
  const CACHE_KEY = "cosmic:visitor-summary:v1";
  const CACHE_MAX_AGE = 24 * 60 * 60 * 1000;
  const DEFAULT_ENDPOINT = "https://cosmicrealm-visitor-stats.pages.dev";

  function formatCount(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0 || value === null || value === undefined) return "—";
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(number);
  }

  function getStorage(windowObject, storageOverride) {
    if (storageOverride) return storageOverride;
    try {
      return windowObject.localStorage;
    } catch (error) {
      return null;
    }
  }

  function countryName(code) {
    try {
      if (typeof Intl.DisplayNames === "function") {
        return new Intl.DisplayNames(["en"], { type: "region" }).of(code) || code;
      }
    } catch (error) {
      // Fall back to the stable ISO code.
    }
    return code;
  }

  function validSummary(summary) {
    return summary
      && Number.isFinite(Number(summary.totalViews))
      && Number.isFinite(Number(summary.visitorsToday))
      && Number.isFinite(Number(summary.countriesReached))
      && Array.isArray(summary.countries);
  }

  function createVisitorAnalytics(options = {}) {
    const windowObject = options.window || (typeof window !== "undefined" ? window : null);
    const documentObject = options.document || (typeof document !== "undefined" ? document : null);
    const component = options.component || documentObject?.querySelector?.("[data-visitor-stats]") || null;
    const endpoint = String(options.endpoint || DEFAULT_ENDPOINT).replace(/\/$/, "");
    const fetchImpl = options.fetchImpl || windowObject?.fetch?.bind(windowObject);
    const now = options.now || (() => new Date());
    const storage = getStorage(windowObject || {}, options.storage);

    function privacyEnabled() {
      const navigatorObject = windowObject?.navigator || {};
      return navigatorObject.globalPrivacyControl === true
        || navigatorObject.doNotTrack === "1"
        || windowObject?.doNotTrack === "1";
    }

    function isProductionSite() {
      return windowObject?.location?.hostname === "cosmicrealm.github.io";
    }

    function writeCache(summary) {
      if (!storage) return;
      try {
        storage.setItem(CACHE_KEY, JSON.stringify({
          cachedAt: now().getTime(),
          summary,
        }));
      } catch (error) {
        // Browsers may block storage; live data still renders.
      }
    }

    function readCache() {
      if (!storage) return null;
      try {
        const cached = JSON.parse(storage.getItem(CACHE_KEY) || "null");
        if (!cached || !validSummary(cached.summary)) return null;
        if (now().getTime() - Number(cached.cachedAt) > CACHE_MAX_AGE) return null;
        return cached.summary;
      } catch (error) {
        return null;
      }
    }

    function setText(selector, value) {
      const node = component?.querySelector?.(selector);
      if (node) node.textContent = value;
    }

    function renderMap(countries) {
      if (!component?.querySelectorAll) return;
      const byCode = new Map(countries.map((entry) => [String(entry.code).toUpperCase(), Number(entry.views) || 0]));
      const maximum = Math.max(1, ...byCode.values());
      component.querySelectorAll("[data-country-code]").forEach((region) => {
        const views = byCode.get(String(region.dataset.countryCode || "").toUpperCase()) || 0;
        region.classList.remove("is-visited");
        if (views <= 0) return;
        region.classList.add("is-visited");
        const intensity = 0.28 + 0.72 * Math.sqrt(views / maximum);
        region.style.setProperty("--visitor-intensity", intensity.toFixed(3));
      });
    }

    function renderSummary(summary, source) {
      if (!component) return;
      setText("[data-visitor-total]", formatCount(summary.totalViews));
      setText("[data-visitor-today]", formatCount(summary.visitorsToday));
      setText("[data-visitor-countries]", formatCount(summary.countriesReached));
      const topCountries = summary.countries
        .slice(0, 5)
        .map((entry) => `${countryName(String(entry.code).toUpperCase())} ${formatCount(entry.views)}`)
        .join(", ");
      setText(
        "[data-visitor-accessible-summary]",
        `${formatCount(summary.countriesReached)} countries reached.${topCountries ? ` Top regions: ${topCountries}.` : ""}`,
      );
      setText(
        "[data-visitor-status]",
        source === "cache" ? "Showing cached statistics" : "Updated a few moments ago",
      );
      renderMap(summary.countries);
      component.classList.remove("is-unavailable");
      component.classList.add("is-ready");
    }

    function renderUnavailable() {
      if (!component) return;
      setText("[data-visitor-total]", "—");
      setText("[data-visitor-today]", "—");
      setText("[data-visitor-countries]", "—");
      setText("[data-visitor-status]", "Live statistics temporarily unavailable");
      component.classList.add("is-unavailable");
    }

    async function collect() {
      if (!fetchImpl || !isProductionSite() || privacyEnabled()) return { skipped: true };
      try {
        await fetchImpl(`${endpoint}/v1/collect`, {
          body: "",
          cache: "no-store",
          credentials: "omit",
          keepalive: true,
          method: "POST",
          mode: "cors",
        });
        return { skipped: false };
      } catch (error) {
        return { skipped: false, unavailable: true };
      }
    }

    async function loadSummary() {
      if (!component || !fetchImpl) return { source: "none" };
      try {
        const response = await fetchImpl(`${endpoint}/v1/summary`, {
          cache: "no-store",
          credentials: "omit",
          method: "GET",
          mode: "cors",
        });
        if (!response.ok) throw new Error(`summary request failed: ${response.status}`);
        const summary = await response.json();
        if (!validSummary(summary)) throw new Error("invalid summary payload");
        writeCache(summary);
        renderSummary(summary, "live");
        return { source: "live", summary };
      } catch (error) {
        const cached = readCache();
        if (cached) {
          renderSummary(cached, "cache");
          return { source: "cache", summary: cached };
        }
        renderUnavailable();
        return { source: "unavailable" };
      }
    }

    async function mount() {
      const results = await Promise.all([collect(), loadSummary()]);
      return { collection: results[0], summary: results[1] };
    }

    return { collect, loadSummary, mount, renderSummary };
  }

  return {
    CACHE_KEY,
    DEFAULT_ENDPOINT,
    createVisitorAnalytics,
    formatCount,
  };
}));
