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
- 尽量只改和当前内容相关的数据、页面模板或样式，避免顺手整理旧模板文件。
- 不要覆盖未跟当前任务相关的草稿、博客图片或用户未跟踪文件。
