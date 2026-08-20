# cosmicrealm.github.io

本仓库是 `Jinyang Zhang` 的个人 Jekyll 主页源码，站点主域名为 `https://cosmicrealm.github.io`。仓库只保留当前主页实际使用的内容命名空间与构建脚本，不再承载 Academic Pages 模板残留页面、模板说明或根级实验壳。

## Local Preview

本地预览默认走仓库自带的 Docker Compose 环境，不依赖宿主机 Ruby/Bundler。

```bash
docker compose up
```

服务启动后访问 `http://localhost:4000/`。如果你的本机只提供 legacy 命令，也可使用：

```bash
docker-compose up
```

停止预览时执行：

```bash
docker compose down
```

## Content Namespaces

当前站点只维护以下内容空间与对应 canonical URL：

- `projects/`：项目静态页面与资源，canonical URL 形如 `/projects/<slug>/`
- `foundations/`：原理讲义、交互式技术笔记与实验页，canonical URL 形如 `/foundations/<slug>/`
- `_publications/`：论文条目，canonical URL 形如 `/publications/<slug>/`
- `_posts/`：博客与写作内容，主入口为 `/writing/`
- `_pages/`：站点级页面，例如 `/`、`/projects/`、`/foundations/`、`/publications/`、`/writing/`、`/cv/`

可下载附件继续放在 `files/`，发布后路径为 `/files/<filename>`。

## Validation

架构与发布边界变更先跑架构验证器：

```bash
python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check dirty
```

需要验证 Jekyll 构建与输出时，使用 Docker 环境：

```bash
docker compose up --build -d
python3 scripts/verify_homepage_architecture.py --check build --site-dir _site
docker compose down
```

如果只需要完整回归，也可以串行执行：

```bash
python3 scripts/verify_homepage_architecture.py --check routes --check config --check legacy --check projects --check dirty
docker compose up --build -d
python3 scripts/verify_homepage_architecture.py --check build --site-dir _site
docker compose down
```

## Editing Rules

- 不要新增根级项目壳、模板 redirect 页面或 Academic Pages 遗留 shell。
- `scripts/`、`docs/`、`tests/`、notebooks、开发容器配置与本地说明文档不属于发布内容，修改后也不应暴露到最终站点。
- `projects/`、`foundations/`、`_publications/`、`_posts/`、`_pages/` 之外的新内容目录，只有在确认需要成为长期站点命名空间时才应加入。
- 修改固定目录、模板删除或路由迁移前，先更新并运行 `scripts/verify_homepage_architecture.py`，避免把历史模板路径重新发布出去。
