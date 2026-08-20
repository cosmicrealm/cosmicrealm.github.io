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
