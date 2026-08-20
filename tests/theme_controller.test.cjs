"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

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
assert.equal(toggles[1].attrs["aria-pressed"], "true");
assert.match(toggles[1].attrs["aria-label"], /Switch to light mode/);
controller.toggleExplicitTheme();
assert.equal(storage.get("theme"), "light");
assert.equal(root.attrs["data-theme"], "light");
assert.equal(toggles[0].attrs["aria-pressed"], "false");
assert.match(toggles[0].attrs["aria-label"], /Switch to dark mode/);
assert.equal(toggles[1].attrs["aria-pressed"], "false");
assert.match(toggles[1].attrs["aria-label"], /Switch to dark mode/);

const unavailableRoot = {
  attrs: {},
  setAttribute(key, value) { this.attrs[key] = value; },
};
const unavailableToggle = {
  attrs: {},
  setAttribute(key, value) { this.attrs[key] = value; },
};
const unavailableController = createThemeController({
  storage: {
    getItem() { throw new Error("storage read blocked"); },
    setItem() { throw new Error("storage write blocked"); },
  },
  root: unavailableRoot,
  toggles: [unavailableToggle],
  systemDark: () => true,
});

unavailableController.applyInitialTheme();
assert.equal(unavailableRoot.attrs["data-theme"], "dark");
assert.equal(unavailableRoot.attrs["data-theme-setting"], "system");
unavailableController.toggleExplicitTheme();
assert.equal(unavailableRoot.attrs["data-theme"], "light");
assert.equal(unavailableRoot.attrs["data-theme-setting"], "light");
unavailableController.toggleExplicitTheme();
assert.equal(unavailableRoot.attrs["data-theme"], "dark");
assert.equal(unavailableRoot.attrs["data-theme-setting"], "dark");
unavailableController.toggleExplicitTheme();
unavailableController.applySystemChange();
assert.equal(unavailableRoot.attrs["data-theme"], "light");
assert.equal(unavailableRoot.attrs["data-theme-setting"], "light");

let storageAvailable = false;
const recoveredStorage = new Map();
const recoveredRoot = {
  attrs: {},
  setAttribute(key, value) { this.attrs[key] = value; },
};
const recoveredController = createThemeController({
  storage: {
    getItem(key) {
      if (!storageAvailable) throw new Error("storage read blocked");
      return recoveredStorage.has(key) ? recoveredStorage.get(key) : null;
    },
    setItem(key, value) {
      if (!storageAvailable) throw new Error("storage write blocked");
      recoveredStorage.set(key, value);
    },
  },
  root: recoveredRoot,
  toggles: [],
  systemDark: () => true,
});

recoveredController.applyInitialTheme();
recoveredController.toggleExplicitTheme();
storageAvailable = true;
recoveredController.applySystemChange();
assert.equal(recoveredRoot.attrs["data-theme"], "light");
assert.equal(recoveredRoot.attrs["data-theme-setting"], "light");
recoveredController.toggleExplicitTheme();
assert.equal(recoveredStorage.get("theme"), "dark");
assert.equal(recoveredRoot.attrs["data-theme"], "dark");

const sitePath = path.resolve(__dirname, "../assets/js/site.js");
const siteSource = fs.readFileSync(sitePath, "utf8");

function executeSite(document, window) {
  vm.runInNewContext(siteSource, {
    document,
    window,
    URL,
    URLSearchParams,
  }, { filename: sitePath });
}

function createElement(dataset = {}, initialClasses = []) {
  const attrs = {};
  const classes = new Set(initialClasses);
  const listeners = {};
  return {
    attrs,
    dataset,
    hidden: false,
    focusCount: 0,
    listeners,
    classList: {
      contains(name) { return classes.has(name); },
      toggle(name, force) {
        const active = force === undefined ? !classes.has(name) : Boolean(force);
        if (active) classes.add(name);
        else classes.delete(name);
        return active;
      },
    },
    addEventListener(type, listener) { listeners[type] = listener; },
    focus() { this.focusCount += 1; },
    getAttribute(key) { return attrs[key] ?? null; },
    setAttribute(key, value) { attrs[key] = value; },
    closest() { return null; },
  };
}

{
  const toggle = createElement();
  let controllerOptions = null;
  const document = {
    readyState: "complete",
    documentElement: createElement(),
    querySelector(selector) { return null; },
    querySelectorAll(selector) {
      return selector === "[data-theme-toggle]" ? [toggle] : [];
    },
    addEventListener() {},
  };
  const window = {
    CosmicThemeController: {
      createThemeController(options) {
        controllerOptions = options;
        return {
          applyInitialTheme() {},
          applySystemChange() {},
          toggleExplicitTheme() {},
        };
      },
    },
    addEventListener() {},
  };
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    get() { throw new Error("localStorage getter blocked"); },
  });

  assert.doesNotThrow(() => executeSite(document, window));
  assert.ok(controllerOptions, "site.js did not mount the theme controller");
  assert.equal(controllerOptions.systemDark(), false);
  assert.equal(controllerOptions.storage.getItem("theme"), null);
  controllerOptions.storage.setItem("theme", "dark");
  assert.equal(controllerOptions.storage.getItem("theme"), "dark");
}

{
  const toggle = createElement();
  const mediaQueries = [];
  let mediaListener = null;
  let systemChanges = 0;
  let controllerOptions = null;
  const themeMedia = {
    matches: true,
    addListener(listener) { mediaListener = listener; },
  };
  const document = {
    readyState: "complete",
    documentElement: createElement(),
    querySelector() { return null; },
    querySelectorAll(selector) {
      return selector === "[data-theme-toggle]" ? [toggle] : [];
    },
    addEventListener() {},
  };
  const window = {
    localStorage: {
      getItem() { return null; },
      setItem() {},
    },
    CosmicThemeController: {
      createThemeController(options) {
        controllerOptions = options;
        return {
          applyInitialTheme() {},
          applySystemChange() { systemChanges += 1; },
          toggleExplicitTheme() {},
        };
      },
    },
    matchMedia(query) {
      mediaQueries.push(query);
      return themeMedia;
    },
    addEventListener() {},
  };

  assert.doesNotThrow(() => executeSite(document, window));
  assert.deepEqual(mediaQueries, ["(prefers-color-scheme: dark)"]);
  assert.equal(controllerOptions.systemDark(), true);
  assert.equal(typeof mediaListener, "function");
  mediaListener({ matches: false });
  assert.equal(systemChanges, 1);
}

{
  const button = createElement();
  const menu = createElement();
  const documentListeners = {};
  const windowListeners = {};
  const mediaQueries = [];
  const desktopMedia = { matches: true };
  const document = {
    readyState: "complete",
    documentElement: createElement(),
    querySelector(selector) {
      if (selector === "[data-nav-toggle]") return button;
      if (selector === "[data-nav-menu]") return menu;
      return null;
    },
    querySelectorAll() { return []; },
    addEventListener(type, listener) { documentListeners[type] = listener; },
  };
  const window = {
    innerWidth: 500,
    matchMedia(query) {
      mediaQueries.push(query);
      return desktopMedia;
    },
    addEventListener(type, listener) { windowListeners[type] = listener; },
  };

  executeSite(document, window);
  assert.deepEqual(mediaQueries, ["(min-width: 64rem)"]);
  assert.equal(menu.hidden, false);
  documentListeners.keydown({ key: "Escape" });
  assert.equal(menu.hidden, false);
  assert.equal(button.focusCount, 0);

  desktopMedia.matches = false;
  windowListeners.resize();
  assert.equal(menu.hidden, true);
  button.listeners.click();
  assert.equal(menu.hidden, false);
  assert.equal(button.getAttribute("aria-expanded"), "true");
  documentListeners.keydown({ key: "Escape" });
  assert.equal(menu.hidden, true);
  assert.equal(button.getAttribute("aria-expanded"), "false");
  assert.equal(button.focusCount, 1);
}

{
  const button = createElement();
  const menu = createElement();
  const document = {
    readyState: "complete",
    documentElement: createElement(),
    querySelector(selector) {
      if (selector === "[data-nav-toggle]") return button;
      if (selector === "[data-nav-menu]") return menu;
      return null;
    },
    querySelectorAll() { return []; },
    addEventListener() {},
  };
  const window = {
    innerWidth: 1280,
    addEventListener() {},
  };

  assert.doesNotThrow(() => executeSite(document, window));
  assert.equal(menu.hidden, false);
}

{
  const sidebarTag = createElement({ blogTagFilter: "diffusion" });
  const listTag = createElement({ blogTagFilter: "llm" });
  const visionTag = createElement({ blogTagFilter: "vision" });
  const clearButton = createElement({}, ["is-active"]);
  const count = { textContent: "" };
  const empty = { hidden: true };
  const items = [
    createElement({ blogTags: "diffusion" }),
    createElement({ blogTags: "llm vision" }),
    createElement({ blogTags: "other" }),
  ];
  const layoutListeners = {};
  const root = {
    querySelector(selector) {
      if (selector === "[data-blog-clear-filter]") return clearButton;
      if (selector === "[data-blog-filter-count]") return count;
      return null;
    },
  };
  const list = {
    querySelectorAll(selector) {
      return selector === "[data-blog-item]" ? items : [];
    },
  };
  const layout = {
    querySelector(selector) {
      if (selector === "[data-blog-filter-root]") return root;
      if (selector === "[data-blog-list]") return list;
      if (selector === "[data-blog-empty]") return empty;
      return null;
    },
    querySelectorAll(selector) {
      return selector === "[data-blog-tag-filter]" ? [sidebarTag, listTag, visionTag] : [];
    },
    addEventListener(type, listener) { layoutListeners[type] = listener; },
  };
  sidebarTag.closest = (selector) => selector === "[data-blog-tag-filter]" ? sidebarTag : null;
  listTag.closest = (selector) => selector === "[data-blog-tag-filter]" ? listTag : null;
  visionTag.closest = (selector) => selector === "[data-blog-tag-filter]" ? visionTag : null;
  clearButton.closest = (selector) => selector === "[data-blog-clear-filter]" ? clearButton : null;

  const search = "?tags=diffusion&tags=llm,%20vision%20&tags=,%20";
  let replacedUrl = null;
  const document = {
    readyState: "complete",
    documentElement: createElement(),
    querySelector(selector) {
      if (selector === "[data-nav-toggle]" || selector === "[data-nav-menu]") return null;
      if (selector === ".blog-filter-layout") return layout;
      throw new Error(`site.js queried ${selector} outside .blog-filter-layout`);
    },
    querySelectorAll(selector) {
      if (selector === "[data-theme-toggle]") return [];
      throw new Error(`site.js queried ${selector} globally`);
    },
    addEventListener(type) {
      throw new Error(`site.js registered global ${type} listener for writing filters`);
    },
  };
  const window = {
    location: {
      href: `https://example.test/writing/${search}`,
      search,
    },
    history: {
      replaceState(state, title, url) { replacedUrl = url; },
    },
    addEventListener() {},
  };

  assert.doesNotThrow(() => executeSite(document, window));
  assert.equal(sidebarTag.classList.contains("is-active"), true);
  assert.equal(listTag.classList.contains("is-active"), true);
  assert.equal(visionTag.classList.contains("is-active"), true);
  assert.equal(clearButton.classList.contains("is-active"), false);
  assert.equal(clearButton.getAttribute("aria-pressed"), "false");
  assert.equal(items[0].hidden, false);
  assert.equal(items[1].hidden, false);
  assert.equal(items[2].hidden, true);
  assert.equal(count.textContent, "2 / 3");
  assert.equal(typeof layoutListeners.click, "function");

  layoutListeners.click({ target: clearButton });
  assert.equal(clearButton.classList.contains("is-active"), true);
  assert.equal(clearButton.getAttribute("aria-pressed"), "true");
  assert.equal(items.every((item) => item.hidden === false), true);
  assert.equal(count.textContent, "3 / 3");
  assert.equal(replacedUrl, "/writing/");

  layoutListeners.click({ target: listTag });
  assert.equal(sidebarTag.classList.contains("is-active"), false);
  assert.equal(listTag.classList.contains("is-active"), true);
  assert.equal(visionTag.classList.contains("is-active"), false);
  assert.equal(items[0].hidden, true);
  assert.equal(items[1].hidden, false);
  assert.equal(items[2].hidden, true);
  assert.equal(replacedUrl, "/writing/?tags=llm");
}
