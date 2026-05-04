# IConFace Project Page

这个目录是 `IConFace` 的静态项目页，部署目标路径为：

- `https://cosmicrealm.github.io/IConFace/`

## 当前页面结构

页面基于 `Academic-project-page-template` 精简改造，当前包含：

- 标题与论文入口
- Teaser 可视化
- Abstract
- Method Overview
- Reference-aware Restoration
- No-reference Generalization
- Fine-Grained Detail Recovery
- Paper PDF 内嵌预览
- BibTeX

## 主要文件

- `index.html`
  - 页面主体内容与各 section 布局
- `static/css/index.css`
  - 页面样式与 IConFace 定制布局
- `static/js/index.js`
  - BibTeX 复制、返回顶部等轻量交互
- `static/pdfs/iconface_paper.pdf`
  - 当前论文 PDF

## 主要素材来源

素材主要来自：

- `/Users/zhangjinyang/code/local/flux-restoration/paper_submission/figures`
- `/Users/zhangjinyang/code/local/flux-restoration/paper_submission/fingers`

其中包括：

- teaser 示例图
- IConFace 方法框架图
- reference-aware / no-reference 定性对比图
- 局部细节放大图

## 当前约定

- 作者信息当前仍为 `Anonymous Authors`
- 页面默认使用本地论文 PDF 作为 `Paper` 按钮目标
- 未接入外部 `Code`、`arXiv`、`Video` 链接

## 后续可继续补充

- 替换为正式作者与机构信息
- 增加外部代码仓库链接
- 增加 supplementary / arXiv 链接
- 增加更多 reference-aware / ablation 可视化页块
