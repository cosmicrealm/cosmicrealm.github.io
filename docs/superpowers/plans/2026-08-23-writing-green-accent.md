# Writing Green Accent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `/writing/` 增加同时适配浅色和深色主题的局部浅绿色强调色，不改变站点其他页面的黑白主题。

**Architecture:** 使用现有 `#main.blog-filter-layout` 作为作用域，在 `_sass/layout/_cosmicrealm.scss` 中覆盖 Writing 页局部 CSS variables。现有筛选器、日期卡片、标签和链接已经消费这些 token，因此无需修改页面数据或 JavaScript。

**Tech Stack:** Jekyll、SCSS、Playwright、Node.js assertions

---

### Task 1: Writing 页局部主题

**Files:**
- Modify: `tests/site_browser_qa.cjs`
- Modify: `_sass/layout/_cosmicrealm.scss`

- [ ] **Step 1: 写入失败的浏览器断言**

在 `tests/site_browser_qa.cjs` 新增 `assertWritingAccentPalette`，分别断言浅色与深色页面中：

```js
assert.deepEqual(actual, {
  accent: colorScheme === "dark" ? "#a8d5b8" : "#5f8f72",
  accentStrong: colorScheme === "dark" ? "#c8ead3" : "#315d46",
  pageBackground: colorScheme === "dark" ? "rgb(0, 0, 0)" : "rgb(255, 255, 255)",
});
```

- [ ] **Step 2: 验证断言因缺少 Writing 局部主题而失败**

运行：

```bash
node tests/site_browser_qa.cjs
```

预期：Writing 页仍继承全站黑白 token，断言在 `accent` 或 `accentStrong` 上失败。

- [ ] **Step 3: 实现局部浅绿色 token 与可读选中态**

在 `_sass/layout/_cosmicrealm.scss` 增加：

```scss
#main.blog-filter-layout {
  --cr-accent: #5f8f72;
  --cr-accent-strong: #315d46;
  --cr-accent-soft: #eaf4ed;
  --cr-warm: #6f8c79;
}

html[data-theme="dark"] #main.blog-filter-layout {
  --cr-accent: #a8d5b8;
  --cr-accent-strong: #c8ead3;
  --cr-accent-soft: rgba(168, 213, 184, 0.14);
  --cr-warm: #9ab8a3;
}
```

并把 Writing 筛选器和文章标签的选中态文字色从硬编码白色改为 `var(--global-on-base-color)`。

- [ ] **Step 4: 运行完整验证**

运行：

```bash
node tests/site_browser_qa.cjs
python3 scripts/verify_homepage_theme_architecture.py
python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check build --site-dir _site
git diff --check
```

预期：全部退出码为 `0`，Writing 页浅色与深色断言通过，其他页面无回归。

- [ ] **Step 5: 浏览器检查本地页面**

打开 `http://0.0.0.0:4000/writing/`，检查浅色与深色模式中的筛选标签、日期卡片和文章标签，确认无溢出、无控制台错误，并保留本地预览供用户确认。
