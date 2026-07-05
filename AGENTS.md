# cosmicrealm.github.io Codex Notes

本仓库是 `cosmicrealm.github.io` 的 Jekyll 个人站点。默认用中文沟通和写说明，但代码标识、配置键、命令和路径保持英文。

## Local Run And Verification

- 本项目默认通过系统 Docker 环境运行；先确认 Docker Desktop / 系统 Docker 已经打开并可用。
- 本地预览和验证优先使用仓库内已经建好的 Compose 环境：

```bash
docker-compose up
```

- 如果当前机器没有 legacy `docker-compose` 可执行文件，但 Docker Desktop 提供 Compose v2，则使用等价命令：

```bash
docker compose up
```

- 服务启动后默认访问 `http://localhost:4000/`。
- 不要默认改用本机 Ruby / Bundler / `bundle exec jekyll serve`，除非 Docker 环境不可用或用户明确要求。
- 需要验证页面时，保持 `docker-compose up` 运行，再用浏览器检查对应页面。

## Editing Scope

- 项目列表由 `_data/projects.yml` 驱动，`_pages/projects.md` 渲染 `/projects/`。
- 原理性解释、数学推导、教学实验、可交互学习笔记、foundation-style 长文页面默认归入 Foundations，而不是 Projects。此类页面放在 `foundations/<slug>/`，入口由 `_data/foundations.yml` 驱动，`_pages/foundations.md` 渲染 `/foundations/`。
- Foundations 页面应优先复用 `foundations/generation-math/` 的讲义风格和结构：`main.lecture`、`paper-header`、`chapter`、MathJax 公式、algorithm panel、可视化 lab、理解检查和方法比较。新增页面不要破坏已有 `generation-math`。
- Foundations 页面若使用固定左侧目录（例如 `.lecture-toc`），必须给正文留出真实横向 gutter，不能让 `main.lecture` 仍按页面居中后被 fixed TOC 覆盖。修复或新增此类目录时，至少用浏览器验证 `toc.right <= lecture.left`、桌面无覆盖、移动端目录隐藏/按钮可用且无横向溢出；不要只凭单一截图或单一 1440px 视口判断。
- 已出现过的错误：`foundations/generation-acceleration/` 的 fixed `.lecture-toc` 在中等桌面宽度覆盖正文标题和章节内容。遇到类似问题时，不要用提高目录 `z-index`、缩小正文截图、或只看一个 viewport 来“修复”；应提高 fixed TOC 显示断点或重新计算 `main.lecture` 的 `margin-left`/可用宽度，并用浏览器断言 `toc.right <= lecture.left` 后再完成。
- 尽量只改和当前内容相关的数据、页面模板或样式，避免顺手整理旧模板文件。
- 不要覆盖未跟当前任务相关的草稿、博客图片或用户未跟踪文件。
