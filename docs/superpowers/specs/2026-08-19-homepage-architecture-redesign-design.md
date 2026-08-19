# 个人主页架构重组与 B 风格双主题设计

## 1. 目标与决策

本次改造同时解决两个问题：

1. 将个人主页改造成克制、可信、信息密度高的“极简学术档案”界面；
2. 让源码目录、公开 URL 与内容分类保持一致，去掉 AcademicPages 模板遗留和历史兼容层。

用户已确认选择 B 风格，并明确表示暂不考虑论文、GitHub 或搜索引擎中的旧链接。因此本次采用一次性规范化迁移，不为旧 URL 建立 redirect，也不保留仅用于兼容的根目录壳。

核心规范：

```text
项目展示     → /projects/<slug>/
原理讲义     → /foundations/<slug>/
论文详情     → /publications/<slug>/
写作归档     → /writing/
个人简历     → /cv/
```

本次不迁移到 React、Astro、Next.js 或另一套 Jekyll 主题。现有站点以静态内容为主，Jekyll 与 GitHub Pages 已能满足需求；架构优化重点是清理内容边界、路由、共享主题和运行时依赖，而不是为了“现代化”引入新的部署与依赖复杂度。本次也不重写论文、文章或讲义正文。

## 2. 当前站点判断

当前内容基础完整：`_data/projects.yml` 管理 8 个项目，`_data/foundations.yml` 管理 9 个讲义入口，`_publications/` 有 5 篇论文，`_posts/` 有 49 篇文章，并包含 CV 与个人介绍。

现有优点：

- Projects 与 Foundations 列表已经数据驱动；
- Jekyll 主站已有 light / dark 状态和主题切换入口；
- Foundations 与两个独立项目页都采用自包含页面，迁移边界清晰；
- IConFace 与 Style-Talking 已具备完整项目内容，而非占位页。

主要问题：

- 根目录同时存在 `IConFace/`、`style-talking/`、`Projects/`、`generation-math/`、`generation-distillation/` 和 `TestPage/`，内容源与 redirect 壳混杂；
- 主页由 `_pages/about.md` 承担，文件名不能准确表达职责；
- 主站、Foundations、IConFace、Style-Talking 使用彼此割裂的主题逻辑；
- `_portfolio/`、`_talks/`、`_teaching/`、模板演示页、talkmap 与 markdown generator 仍在仓库中；
- `_config.yml` 没有阻止 `AGENTS.md`、`docs/`、`scripts/`、`tests/`、notebook 和 Docker 配置进入生成站点；
- 全站 JavaScript 将 Plotly 与通用交互打进约 4.6 MB 的单一 bundle，普通页面承担了不需要的成本；
- 页面仍显示 X、Facebook、LinkedIn、Mastodon 等社交分享区。

## 3. 内容分类边界

### Projects

放置可展示的研究系统、工程实现和独立项目页。项目必须使用小写 kebab-case slug：

```text
projects/iconface/
projects/style-talking/
```

每个项目可保持自包含结构：

```text
projects/<slug>/
  index.html
  static/
    css/
    js/
    images/
```

项目索引仍由 `_data/projects.yml` 驱动；数据文件只描述项目，不复制项目页正文。

项目目录只保存浏览器运行时需要的文件。IConFace 的素材同步脚本迁到 `scripts/projects/iconface/`，项目维护说明迁到 `docs/projects/`；这些工程文件不应随项目页面发布。项目局部 `.gitignore` 可保留，但 `.nojekyll` 不再需要。

### Publications

只保存论文元数据、摘要与论文链接，不承担项目展示。所有详情页统一为小写 slug：

```text
/publications/dcpn/
/publications/iconface/
/publications/ldcr/
/publications/ntire-2026-face-restoration/
/publications/ukl/
```

IConFace 的 publication 条目链接到 `/projects/iconface/`，但两者保持不同职责。

### Foundations

保存原理解释、数学推导、教学实验和交互学习笔记。`foundations/<slug>/` 继续作为唯一 canonical 内容源。

`generation-math/`、`generation-distillation/`、`Projects/diffusion-distillation-math-lab/` 与 `foundations/diffusion-distillation-math-lab/` 仅为历史跳转壳，直接删除，不建立替代 redirect。

### Writing

保存时间线文章与博客。主入口从模板式 `/year-archive/` 规范为 `/writing/`，导航文案统一为 `Writing`。

迁移必须同步更新 `_data/navigation.yml`、主页的 Writing 入口、`_includes/tag-chip.html` 的标签查询地址，以及归档页的 `permalink`。现有 `?tags=<slug>` 过滤协议保持不变，只替换其基础路径，保证 `/writing/?tags=<slug>` 仍能直接打开对应筛选状态。

### Site pages

主页源码从 `_pages/about.md` 改名为 `_pages/home.md`，继续输出 `/`。保留真正使用的入口页：

```text
_pages/
  home.md
  projects.md
  publications.html
  foundations.md
  writing.html
  cv.md
  category-archive.html
  tag-archive.html
  sitemap.md
  404.md
```

文件扩展名沿用页面当前最自然的 Liquid / Markdown 形态，不为了统一后缀制造无意义 churn；这里约束的是职责和输出路由。

## 4. 目标目录

```text
cosmicrealm.github.io/
├── _pages/                  # 主站入口页
├── _data/                   # Projects、Foundations、导航与站点数据
├── _posts/                  # Writing 内容
├── _publications/           # 论文元数据与详情
├── projects/                # 独立项目页，URL 与目录一致
│   ├── iconface/
│   └── style-talking/
├── foundations/             # 原理讲义与交互学习页
├── assets/                  # 主站共享 CSS、JS、图标与图片
├── images/                  # 现有全局内容图片，首轮保持路径稳定
├── files/                   # CV 与文章附件
├── _includes/               # 主站组件
├── _layouts/                # Jekyll 布局
├── _sass/                   # 主站样式源
├── scripts/                 # 构建与检查脚本，不发布
├── tests/                   # 自动化测试，不发布
└── docs/                    # 设计与实施文档，不发布
```

根目录只保留 Jekyll、构建、仓库说明和一级内容命名空间，不再放置单个项目、实验页或 redirect 目录。

## 5. 清理范围

删除以下确定不属于当前站点的内容：

- `TestPage/`；
- `Projects/`；
- `generation-math/`；
- `generation-distillation/`；
- `foundations/diffusion-distillation-math-lab/` redirect 壳；
- `_portfolio/`、`_talks/`、`_teaching/` 中的模板样例；
- `_pages/archive-layout-with-content.md`；
- `_pages/collection-archive.html`；
- `_pages/cv-json.md`；
- `_pages/markdown.md`；
- `_pages/non-menu-page.md`；
- `_pages/page-archive.html`；
- `_pages/portfolio.html`；
- `_pages/talkmap.html`；
- `_pages/talks.html`；
- `_pages/teaching.html`；
- `_pages/terms.md`；
- `markdown_generator/`；
- `talkmap/` 与根目录 talkmap notebook / Python 文件；
- `.github/workflows/scrape_talks.yml` 与上游模板 Issue forms；
- `_drafts/post-draft.md` 模板草稿；
- `_data/authors.yml` 中的模板作者；
- `_data/comments/` 中的模板评论；
- 未使用的 `_data/cv.json` 与 `_includes/cv-template.html`；
- 已被 `cv_profile.yml` 工作流取代的 `scripts/cv_markdown_to_json.py` 与 `scripts/update_cv_json.sh`；
- 未被真实站点使用的 collection 配置与默认值。

`_drafts/`、`files/`、`images/` 和当前真实内容不因为结构整理而顺手删除。所有删除均由 Git 记录，可从历史恢复。

`_config.yml` 必须显式排除：

```text
AGENTS.md
CONTRIBUTING.md
docker-compose.yaml
docs/
scripts/
tests/
*.ipynb
```

并保留已有的构建依赖、缓存、`knowledge/` 与源文件排除规则。

仓库身份也同步收口：将上游 AcademicPages 说明改写为当前个人站的 README，删除不再适用的 `CONTRIBUTING.md`，更新 `.devcontainer` 名称以及 `package.json` 的 name、description 与 repository 信息。删除 talks 后同时移除 `talkmap_link`。若全仓检查确认没有真实内容使用 redirect、gist、pagination 或 jemoji，则同时从 `_config.yml` 与 Gemfile 中移除对应插件；不凭模板默认配置假定它们仍有用途。

## 6. B 风格视觉系统

B 风格定位为“极简学术档案”：依靠排版、数字、细线分隔和内容次序建立权威感，不使用大面积渐变、玻璃卡片、霓虹效果或装饰性动画。

### 浅色模式

- 页面底色：温和象牙白；
- 主文字：深海军蓝；
- 次级文字：灰蓝；
- 强调色：克制砖红；
- 分隔线：低对比冷灰；
- 图片使用固定比例和干净边界，不叠加强阴影。

### 深色模式

- 页面底色：墨蓝黑；
- 内容面：略浅的深蓝；
- 主文字：暖白；
- 次级文字：低饱和蓝灰；
- 强调色：提亮后的砖红；
- 保持与浅色模式相同的字号、间距、布局和组件语义。

深色模式不是 C 风格的霓虹实验室版本，只是 B 风格的暗色映射。

### 主页层次

主页按以下顺序组织：

1. 顶部导航；
2. 姓名、研究方向、简介与核心联系入口；
3. Projects / Publications / Writing 数量概览；
4. Selected Work；
5. Representative Publications；
6. Foundations；
7. Recent Writing；
8. 极简页脚。

桌面端不再使用长期占据横向空间的作者侧栏。头像、简介和 Email / GitHub / CV 合并进 hero；移动端按自然阅读顺序堆叠。

## 7. 双主题机制

建立一个不依赖 jQuery 的共享 theme controller：

- 在首屏样式绘制前读取 `localStorage.theme`；
- 支持 `light`、`dark` 和首次访问时的系统偏好；
- 手动切换后持久保存；
- 在 `<html data-theme="...">` 上表达最终主题；
- 监听系统主题变化，但不覆盖用户的显式选择；
- 通过统一按钮的 `aria-label`、`aria-pressed` 和图标状态提供可访问反馈。

主站、Projects 和 Foundations 都读取相同状态。共享品牌色、字体、间距、边界和 focus ring 通过 CSS custom properties 定义；各项目页可以保留自己的内容布局，但不得另建不兼容的主题协议。

## 8. 社交与 SEO

删除页面上的整块社交分享 UI，包括 X / Twitter、Facebook、LinkedIn、Mastodon 与 Bluesky。

保留：

- Email；
- GitHub；
- CV；
- 论文、代码、演示等与内容直接相关的链接；
- 通用 Open Graph、description、canonical 与结构化数据。

移除空的社交账号配置和 X / Twitter 专属展示文案。不可见的 SEO 元数据不应反向生成分享按钮。

## 9. 性能与前端边界

主站基础交互与可选重型库分离：

- theme controller 独立为小型原生 JavaScript；
- responsive navigation 不依赖 Plotly；
- 当前真实内容没有 Plotly 图表，现有 Plotly code block 只存在于待删除的模板演示页，因此移除全站 Plotly 集成与 npm 依赖；未来如有真实图表，再由对应页面局部引入；
- `_includes/footer/custom.html` 不再全局加载 CDN MathJax 和 Mermaid；真实 Markdown 文章通过 `mathjax: true` / `mermaid: true` front matter 按需加载；
- Foundations 共用的本地 MathJax 从 `foundations/generation-math/static/vendor/` 与 `foundations/llm-mechanics/static/vendor/` 抽到 `assets/vendor/mathjax/`，避免让某篇讲义承担共享资源宿主，也避免保存两份 vendor；
- 首页不因导航布局等待 4.6 MB bundle；
- 用原生 JavaScript 与 CSS 替代主站对 jQuery、FitVids 和 smooth-scroll 插件的依赖；视频响应式由 `aspect-ratio` 负责；
- 尽量复用系统字体或已有本地资源，避免新增不必要的第三方请求；
- 遵守 `prefers-reduced-motion`；
- JavaScript 关闭时核心内容和导航仍可访问。

## 10. 迁移顺序

1. 收紧 `_config.yml` 发布边界并移除未使用 collection；
2. 删除模板样例、测试页、生成器与历史 redirect；
3. 将 `IConFace/` 和 `style-talking/` 迁入小写 `projects/`，逐项复核 favicon、canonical、Open Graph image、JSON-LD、Home 链接、CSS / JS / image 相对路径和 `_data/projects.yml`；
4. 逐篇规范 Publications 的 `permalink`、正文项目链接和 `projecturl`，并同步迁移 Writing 页面、导航、主页入口与 tag query 基础路径；
5. 建立共享主题 controller 与 B 风格 token；
6. 重做主页、导航、列表组件与页脚；
7. 让独立 Projects 和 Foundations 接入统一主题状态；
8. 拆分全站 JavaScript，移除可见分享区；
9. 构建并执行路由、链接、资源和浏览器 QA。

不创建旧路径 redirect；发现旧路径引用时直接修正到 canonical URL。

## 11. 风险控制

### IConFace 大体量本地素材

`IConFace/` 工作目录约 720 MB，其中大量 gallery PNG 被 `.gitignore` 忽略，Git 实际跟踪约 82 MB。迁移时必须：

- 先记录 tracked 与 ignored 文件清单；
- 在同一文件系统内移动完整目录，避免复制出第二份大目录；
- 移动后确认忽略规则仍生效；
- 不把本地 gallery 生成物误加入 Git；
- 比较迁移前后的 tracked 文件数与体积。

### 用户现有未提交修改

以下修改属于用户且不在本次首轮重构范围：

```text
foundations/image-generation-data-training/index.html
foundations/image-generation-data-training/static/css/index.css
foundations/image-generation-data-training/static/img/
```

实施时不覆盖、回滚或批量格式化这些文件。只有在统一 theme controller 接入确实需要触碰页面入口时，才做最小编辑并在提交前单独核对 diff。

### 独立静态页

IConFace、Style-Talking 与 Foundations 页面不是 Jekyll layout 派生页。共享主题接入采用稳定的 CSS variables 和轻量脚本，不强行把所有页面一次性重写成 Liquid layout，避免内容回归。

## 12. 验收标准

### 架构

- 根目录不再出现单项目目录、测试页或 redirect 壳；
- 项目页只存在于 `projects/<slug>/`；
- Foundations 只存在于 `foundations/<slug>/`；
- Publications URL 全部为 `/publications/<slug>/`；
- Writing 主入口为 `/writing/`；
- 不存在指向旧路径的站内链接；
- 生成站点不包含工程文档、脚本、测试、notebook 或 Docker 文件。

### 视觉与交互

- 首页与列表页符合 B 风格；
- light / dark 在刷新和跨页面跳转后保持一致；
- 系统主题首次访问正确生效；
- 主站、两个项目页与代表性 Foundations 页面都能切换主题；
- 360、390、768、1280、1440 px 视口无横向溢出；
- 导航不依赖重型 JavaScript 完成首屏布局；
- 页面没有可见社交分享区；
- 键盘 focus、对比度、ARIA 与 reduced-motion 基础检查通过。

### 构建与回归

- Docker Compose Jekyll 构建通过；
- canonical 路由全部返回成功；
- 旧根目录路由不再被生成；
- 内部链接、图片、CSS、JS 与 favicon 无 404；
- 两个项目页的 canonical、Open Graph image、JSON-LD 与 Home 链接均指向新路径；
- `/writing/?tags=<slug>` 能恢复标签筛选状态；
- 普通主页不请求 MathJax、Mermaid 或 Plotly，带对应 front matter 的文章仍能渲染公式与图表；
- IConFace gallery 跟踪边界未变化；
- 当前用户未提交修改完整保留；
- 浏览器控制台无新增错误；
- 首页基础 bundle 不再包含全量 Plotly。
