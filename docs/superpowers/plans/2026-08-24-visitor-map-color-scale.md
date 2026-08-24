# Visitor Map Color Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monochrome visited-country fill with a blue-to-green-to-orange visit-volume scale while preserving the site's pure black and pure white themes.

**Architecture:** Keep visit data and normalization in `assets/js/visitor-analytics.js`, expose only a per-country hue through a CSS custom property, and let theme tokens control saturation and lightness. Add a static legend inside the existing map component; do not change the Worker API or D1 schema.

**Tech Stack:** Jekyll/Liquid, vanilla JavaScript, SCSS/CSS custom properties, Node.js built-in test runner, Playwright browser QA.

---

## File map

- Modify `tests/visitor_analytics.test.cjs`: specify deterministic hue calculation and stale-style cleanup.
- Modify `assets/js/visitor-analytics.js`: calculate normalized country hue and write/remove `--visitor-hue`.
- Modify `scripts/check_visitor_analytics.py`: require the legend and color-scale CSS contract.
- Modify `_includes/home-visitor-stats.html`: render the accessible color-scale legend below the SVG.
- Modify `assets/css/theme-tokens.css`: define light and dark map saturation/lightness tokens.
- Modify `_sass/layout/_home.scss`: render country fills and the legend from the new tokens.

### Task 1: Country hue calculation

**Files:**
- Modify: `tests/visitor_analytics.test.cjs`
- Modify: `assets/js/visitor-analytics.js`

- [ ] **Step 1: Write the failing hue tests**

Extend the fake style object so cleanup is observable:

```js
style: {
  values: {},
  setProperty(key, value) { this.values[key] = value; },
  removeProperty(key) { delete this.values[key]; },
},
```

Import `visitorHue`, then add:

```js
const {
  CACHE_KEY,
  createVisitorAnalytics,
  formatCount,
  visitorHue,
} = require("../assets/js/visitor-analytics.js");

test("maps relative visits from blue through green to orange", () => {
  assert.equal(visitorHue(100, 100), 18);
  assert.equal(visitorHue(25, 100), 114);
  assert.equal(visitorHue(1, 100), 191);
  assert.equal(visitorHue(0, 100), null);
});
```

Append these assertions to the existing heat-map test:

```js
assert.equal(countries.cn.style.values["--visitor-hue"], "18");
assert.equal(countries.us.style.values["--visitor-hue"], "95");
assert.equal(countries.fr.style.values["--visitor-hue"], undefined);

analytics.renderSummary({
  ...summary,
  countriesReached: 1,
  countries: [{ code: "CN", views: 5210 }],
}, "live");

assert.equal(countries.us.classList.contains("is-visited"), false);
assert.equal(countries.us.style.values["--visitor-hue"], undefined);
```

- [ ] **Step 2: Run the unit test and verify RED**

Run:

```bash
node --test tests/visitor_analytics.test.cjs
```

Expected: FAIL because `visitorHue` is not exported and the map does not set `--visitor-hue`.

- [ ] **Step 3: Implement the minimal hue mapping**

Add this pure helper to `assets/js/visitor-analytics.js`:

```js
function visitorHue(views, maximum) {
  const count = Number(views);
  const ceiling = Number(maximum);
  if (!Number.isFinite(count) || !Number.isFinite(ceiling) || count <= 0 || ceiling <= 0) return null;
  const position = Math.sqrt(Math.min(1, count / ceiling));
  return Math.round(210 - (192 * position));
}
```

Update `renderMap` so every region first removes `is-visited`, `--visitor-intensity`, and `--visitor-hue`. For positive views, add `is-visited` and set `--visitor-hue` to `String(visitorHue(views, maximum))`. Export `visitorHue` from the module factory.

- [ ] **Step 4: Run the unit test and verify GREEN**

Run:

```bash
node --test tests/visitor_analytics.test.cjs
```

Expected: all visitor analytics unit tests PASS.

- [ ] **Step 5: Commit the behavior change**

```bash
git add tests/visitor_analytics.test.cjs assets/js/visitor-analytics.js
git commit -m "feat: color visitor map by visit volume"
```

### Task 2: Legend and theme-aware color rendering

**Files:**
- Modify: `scripts/check_visitor_analytics.py`
- Modify: `_includes/home-visitor-stats.html`
- Modify: `assets/css/theme-tokens.css`
- Modify: `_sass/layout/_home.scss`

- [ ] **Step 1: Write the failing structure contract**

In `scripts/check_visitor_analytics.py`, load `assets/css/theme-tokens.css` and require:

```python
require("data-visitor-map-legend" in visitor_component, "visitor map is missing its color scale legend")
require("--visitor-map-saturation" in theme_tokens, "theme tokens are missing visitor map saturation")
require("--visitor-map-lightness" in theme_tokens, "theme tokens are missing visitor map lightness")
require("hsl(var(--visitor-hue" in styles, "visited countries must use the volume hue scale")
```

- [ ] **Step 2: Run the structure check and verify RED**

Run:

```bash
python3 scripts/check_visitor_analytics.py
```

Expected: FAIL with `visitor map is missing its color scale legend`.

- [ ] **Step 3: Add the legend markup**

Immediately after `{% include world-map.svg %}` in `_includes/home-visitor-stats.html`, add:

```html
<div class="visitor-reach__legend" data-visitor-map-legend aria-label="Map color scale from fewer to more visits">
  <span>Fewer visits</span>
  <span class="visitor-reach__legend-scale" aria-hidden="true"></span>
  <span>More visits</span>
</div>
```

- [ ] **Step 4: Add light and dark visualization tokens**

Add the following variables to the existing light and dark blocks in `assets/css/theme-tokens.css`:

```css
--visitor-map-saturation: 78%;
--visitor-map-lightness: 46%;
```

```css
--visitor-map-saturation: 82%;
--visitor-map-lightness: 62%;
```

- [ ] **Step 5: Render the color scale and legend**

Replace the monochrome `.visitor-map path.is-visited` declarations in `_sass/layout/_home.scss` with:

```scss
.visitor-map path.is-visited {
  fill: hsl(var(--visitor-hue, 210) var(--visitor-map-saturation) var(--visitor-map-lightness));
  fill-opacity: 1;
}
```

Add a compact three-column legend. Its middle scale must use:

```scss
.visitor-reach__legend {
  display: grid;
  grid-template-columns: max-content minmax(8rem, 1fr) max-content;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.8rem;
  color: var(--cr-ink-soft);
  font-family: var(--cr-font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  line-height: 1;
  text-transform: uppercase;
}

.visitor-reach__legend-scale {
  display: block;
  height: 0.38rem;
  border: 1px solid var(--cr-border-strong);
  background: linear-gradient(
    90deg,
    hsl(210 var(--visitor-map-saturation) var(--visitor-map-lightness)),
    hsl(145 var(--visitor-map-saturation) var(--visitor-map-lightness)),
    hsl(75 var(--visitor-map-saturation) var(--visitor-map-lightness)),
    hsl(18 var(--visitor-map-saturation) var(--visitor-map-lightness))
  );
}
```

The `minmax(8rem, 1fr)` middle column keeps the scale legible on desktop and lets it shrink to the available width on mobile.

- [ ] **Step 6: Run structure and visitor suites and verify GREEN**

Run:

```bash
python3 scripts/check_visitor_analytics.py
npm run test:visitors
```

Expected: both commands PASS.

- [ ] **Step 7: Commit the presentation change**

```bash
git add scripts/check_visitor_analytics.py _includes/home-visitor-stats.html assets/css/theme-tokens.css _sass/layout/_home.scss
git commit -m "style: add visitor map color scale legend"
```

### Task 3: Rendered verification

**Files:**
- Verify: `http://localhost:4000/`
- Verify: `tests/site_browser_qa.cjs`

- [ ] **Step 1: Run the full scoped checks**

```bash
npm run test:visitors
npm run test:structure
node tests/site_browser_qa.cjs
python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check build --site-dir _site
git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 2: Inspect the light theme in the browser**

Open `http://localhost:4000/` at approximately `1440x900`, scroll to `Global Reach`, and verify:

- visited countries use at least two distinct non-gray colors;
- highest-volume country is on the orange end;
- legend is visible and matches the map scale;
- page background remains pure white;
- no console errors or horizontal overflow appear.

- [ ] **Step 3: Inspect the dark theme and mobile layout**

Toggle dark mode and verify the same map colors remain distinct on pure black. Then inspect at approximately `390x844` and verify the map and legend fit the viewport without clipping or overflow.

- [ ] **Step 4: Report without pushing**

Leave all implementation commits on the current branch and report the local preview result. Do not push to `main` until the user explicitly confirms the preview.
