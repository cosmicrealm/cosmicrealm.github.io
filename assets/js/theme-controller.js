"use strict";

function resolveTheme(setting, systemDark) {
  if (setting === "dark") return "dark";
  if (setting === "light") return "light";
  return systemDark ? "dark" : "light";
}

function updateToggle(toggle, resolved) {
  toggle.setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
  toggle.setAttribute("aria-label", resolved === "dark" ? "Switch to light mode" : "Switch to dark mode");
}

function createThemeController(options) {
  const storage = options.storage;
  const root = options.root;
  const toggles = options.toggles;
  const systemDark = options.systemDark;
  let inMemorySetting = null;

  function readSetting() {
    if (inMemorySetting !== null) return inMemorySetting;
    let value = null;
    try {
      value = storage.getItem("theme");
    } catch (error) {
      value = null;
    }
    inMemorySetting = value === "light" || value === "dark" ? value : "system";
    return inMemorySetting;
  }

  function applyResolved(resolved, setting) {
    root.setAttribute("data-theme", resolved);
    root.setAttribute("data-theme-setting", setting);
    toggles.forEach((toggle) => updateToggle(toggle, resolved));
    if (typeof document !== "undefined") {
      document.querySelectorAll("[data-theme-color]").forEach((meta) => {
        meta.setAttribute("content", resolved === "dark" ? "#000000" : "#ffffff");
      });
    }
    return resolved;
  }

  function apply(setting) {
    return applyResolved(resolveTheme(setting, systemDark()), setting);
  }

  return {
    applyInitialTheme() { return apply(readSetting()); },
    toggleExplicitTheme() {
      const current = resolveTheme(readSetting(), systemDark());
      const next = current === "dark" ? "light" : "dark";
      inMemorySetting = next;
      try {
        storage.setItem("theme", next);
      } catch (error) {
        // The resolved theme still applies when storage is unavailable.
      }
      return apply(next);
    },
    applySystemChange() {
      return apply(readSetting());
    },
  };
}

if (typeof window !== "undefined") {
  window.CosmicThemeController = { createThemeController, resolveTheme };
}

if (typeof module !== "undefined") {
  module.exports = { createThemeController, resolveTheme };
}
