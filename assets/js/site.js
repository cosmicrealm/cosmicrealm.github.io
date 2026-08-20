"use strict";

(function () {
  function createVolatileStorage() {
    const values = new Map();
    return {
      getItem(key) { return values.has(key) ? values.get(key) : null; },
      setItem(key, value) { values.set(key, value); },
    };
  }

  function getThemeStorage() {
    try {
      const storage = window.localStorage;
      if (storage) return storage;
    } catch (error) {
      // Access can be blocked even before getItem is called.
    }
    return createVolatileStorage();
  }

  function getMediaQuery(query) {
    if (typeof window.matchMedia !== "function") return null;
    try {
      return window.matchMedia(query);
    } catch (error) {
      return null;
    }
  }

  function addMediaChangeListener(mediaQuery, listener) {
    if (!mediaQuery) return;
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", listener);
    } else if (typeof mediaQuery.addListener === "function") {
      mediaQuery.addListener(listener);
    }
  }

  function mountTheme() {
    const toggles = Array.from(document.querySelectorAll("[data-theme-toggle]"));
    if (!window.CosmicThemeController || toggles.length === 0) return;
    const colorScheme = getMediaQuery("(prefers-color-scheme: dark)");
    const controller = window.CosmicThemeController.createThemeController({
      storage: getThemeStorage(),
      root: document.documentElement,
      toggles,
      systemDark: () => Boolean(colorScheme && colorScheme.matches),
    });
    controller.applyInitialTheme();
    toggles.forEach((toggle) => {
      toggle.addEventListener("click", () => controller.toggleExplicitTheme());
    });
    addMediaChangeListener(colorScheme, () => controller.applySystemChange());
  }

  function mountNavigation() {
    const button = document.querySelector("[data-nav-toggle]");
    const menu = document.querySelector("[data-nav-menu]");
    if (!button || !menu) return;
    const desktopMedia = getMediaQuery("(min-width: 64rem)");

    function isDesktop() {
      return desktopMedia ? desktopMedia.matches : window.innerWidth >= 1024;
    }

    function closeMenu() {
      button.setAttribute("aria-expanded", "false");
      menu.hidden = true;
    }

    function openMenu() {
      button.setAttribute("aria-expanded", "true");
      menu.hidden = false;
    }

    function syncDesktop() {
      if (isDesktop()) {
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
      if (event.key !== "Escape" || isDesktop() || button.getAttribute("aria-expanded") !== "true") return;
      closeMenu();
      if (typeof button.focus === "function") button.focus();
    });

    window.addEventListener("resize", syncDesktop);
    addMediaChangeListener(desktopMedia, syncDesktop);
    syncDesktop();
  }

  function mountWritingFilter() {
    const layout = document.querySelector(".blog-filter-layout");
    if (!layout) return;
    const root = layout.querySelector("[data-blog-filter-root]");
    const list = layout.querySelector("[data-blog-list]");
    if (!root || !list) return;
    const buttons = Array.from(layout.querySelectorAll("[data-blog-tag-filter]"));
    const clearButton = root.querySelector("[data-blog-clear-filter]");
    const count = root.querySelector("[data-blog-filter-count]");
    const empty = layout.querySelector("[data-blog-empty]");
    const items = Array.from(list.querySelectorAll("[data-blog-item]"));
    const selected = new Set(
      new URLSearchParams(window.location.search)
        .getAll("tags")
        .flatMap((value) => value.split(","))
        .map((tag) => tag.trim())
        .filter(Boolean)
    );

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
      if (clearButton) {
        const clearActive = selected.size === 0;
        clearButton.classList.toggle("is-active", clearActive);
        clearButton.setAttribute("aria-pressed", clearActive ? "true" : "false");
      }
      if (count) count.textContent = `${visible} / ${items.length}`;
      if (empty) empty.hidden = visible !== 0;
    }

    function syncUrl() {
      const url = new URL(window.location.href);
      if (selected.size > 0) url.searchParams.set("tags", Array.from(selected).join(","));
      else url.searchParams.delete("tags");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    }

    layout.addEventListener("click", (event) => {
      const tagButton = event.target.closest("[data-blog-tag-filter]");
      const resetButton = event.target.closest("[data-blog-clear-filter]");
      if (tagButton) {
        const tag = (tagButton.dataset.blogTagFilter || "").trim();
        if (!tag) return;
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
