"use strict";

const assert = require("node:assert/strict");
const { createThemeController, resolveTheme } = require("../assets/js/theme-controller.js");

assert.equal(resolveTheme("dark", false), "dark");
assert.equal(resolveTheme("light", true), "light");
assert.equal(resolveTheme("system", true), "dark");
assert.equal(resolveTheme("system", false), "light");

const storage = new Map();
const toggles = [
  {
    attrs: {},
    setAttribute(key, value) { this.attrs[key] = value; },
  },
];
const root = {
  attrs: {},
  setAttribute(key, value) { this.attrs[key] = value; },
  removeAttribute(key) { delete this.attrs[key]; },
};

const controller = createThemeController({
  storage: {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, value); },
  },
  root,
  toggles,
  systemDark: () => true,
});

controller.applyInitialTheme();
assert.equal(root.attrs["data-theme"], "dark");
assert.equal(toggles[0].attrs["aria-pressed"], "true");
assert.match(toggles[0].attrs["aria-label"], /Switch to light mode/);
controller.toggleExplicitTheme();
assert.equal(storage.get("theme"), "light");
assert.equal(root.attrs["data-theme"], "light");
assert.equal(toggles[0].attrs["aria-pressed"], "false");
assert.match(toggles[0].attrs["aria-label"], /Switch to dark mode/);
assert.equal(typeof globalThis.mountAll, "undefined");
