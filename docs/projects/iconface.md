# IConFace Project Page

Static project page for:

> IConFace: Fine-Grained Identity Conditioning for Reference-Aware Face Restoration

- Project: https://cosmicrealm.github.io/projects/iconface/
- Paper: https://arxiv.org/pdf/2605.02814
- arXiv: https://arxiv.org/abs/2605.02814
- Supplementary: https://arxiv.org/pdf/2605.02814#page=10

The public code button is intentionally hidden until a repository is released.

## Source of truth

The page follows the final v1 activity files in `paper_submission_iconface`:

- `IConface_v1.tex`
- `IConface_supp_v1.tex`
- `figures/` and `fingers/` referenced by those files

The website does not modify the paper sources. Main-paper and supplementary visual assets are mirrored into `static/gallery/v2/`; ordinary face panels use high-quality WebP, while the framework and localized-detail composites remain lossless PNG.

## Refreshing paper assets

```bash
python3 scripts/projects/iconface/sync_paper_assets.py \
  --paper-root /Users/zhangjinyang/code/local/flux-restoration/paper_submission_iconface
```

The script uses an explicit source map rather than parsing TeX macros. Update that map whenever selected paper cases change.
It requires `cwebp` and `ffmpeg` on `PATH`. Each sync also regenerates the page teasers at `images/projects/iconface.jpg` and `images/publications/iconface.jpg`; treat those two files as generated outputs rather than hand-edited assets.

The legacy `static/gallery/paper/` tree is kept only as historical retained assets. The current page consumes `static/gallery/v2/` and the teaser/image outputs above.

## Main files

- `index.html`: metadata, paper narrative, result summaries, and gallery anchors.
- `static/css/index.css`: responsive project-page styling.
- `static/js/index.js`: on-demand gallery rendering, BibTeX copy, and scroll behavior.
- `static/gallery/v2/manifest.json`: generated gallery metadata.
- `scripts/projects/iconface/sync_paper_assets.py`: reproducible asset synchronization.
