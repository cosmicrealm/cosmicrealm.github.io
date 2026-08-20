"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const controllerPath = path.resolve(__dirname, "../assets/js/theme-controller.js");
const previousDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
let controllerModule;
try {
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    get() {
      throw new Error("theme-controller.js accessed document while loading");
    },
  });
  controllerModule = require(controllerPath);
} finally {
  if (previousDocument) {
    Object.defineProperty(globalThis, "document", previousDocument);
  } else {
    delete globalThis.document;
  }
}

const source = fs.readFileSync(controllerPath, "utf8");
assert.doesNotMatch(source, /\bDOMContentLoaded\b/, "theme-controller.js must not register a DOMContentLoaded auto-mount");
assert.doesNotMatch(source, /\bdata-theme-toggle\b/, "theme-controller.js must not discover or bind DOM theme toggles");

const { createThemeController, resolveTheme } = controllerModule;

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
