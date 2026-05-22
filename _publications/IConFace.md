---
title: "IConFace: Identity-Structure Asymmetric Conditioning for Unified Reference-Aware Face Restoration"
collection: publications
category: preprints
permalink: /publication/IConFace
excerpt:
date: 2026-05-04
venue: 'arXiv preprint arXiv:2605.02814'
summary: 'A unified reference-aware and no-reference face restoration framework that uses reliability-weighted identity anchors and degraded-image spatial structure anchors in one checkpoint.'
paperurl: 'https://arxiv.org/abs/2605.02814'
projecturl: 'https://cosmicrealm.github.io/IConFace/'
codeurl: 'https://github.com/cosmicrealm/IConFace'
codelabel: 'Code'
---

[Project](https://cosmicrealm.github.io/IConFace/) / [arXiv](https://arxiv.org/abs/2605.02814) / [Code](https://github.com/cosmicrealm/IConFace)

Abstract:
Blind face restoration is challenging under severe degradation because identity-critical details may be missing from the degraded input. Same-identity references can reduce this ambiguity, but mismatched pose, expression, illumination, age, makeup, or local facial states may cause overuse of reference appearance. IConFace addresses this with a unified reference-aware and no-reference framework based on identity-structure asymmetric conditioning. It distills references into a norm-weighted global AdaFace identity anchor for image-only modulation, reinforces the degraded image as the spatial structure anchor through low-rank residuals and block-wise degraded cross-attention, and uses a two-route memory design. A single checkpoint can use references when available and fall back to no-reference restoration when absent.
