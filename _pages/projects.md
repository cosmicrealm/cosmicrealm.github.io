---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

A compact index of public projects around face restoration, FLUX inference, and related tooling.

## Face Restoration

### [IConFace](https://github.com/cosmicrealm/IConFace)

Unified reference-aware and no-reference face restoration with identity-structure asymmetric conditioning. The method uses same-identity references when available and falls back to degraded-only restoration when references are absent.

Links: [Project Page](https://cosmicrealm.github.io/IConFace/) / [arXiv](https://arxiv.org/abs/2605.02814) / [Code Coming Soon](https://github.com/cosmicrealm/IConFace)

### [flux-restoration](https://github.com/cosmicrealm/flux-restoration)

Inference-only release package for a unified blind and reference-based face restoration adapter built on FLUX.2-klein-base-4B. It includes example visual results and pretrained model instructions for restoration workflows.

Links: [Code](https://github.com/cosmicrealm/flux-restoration)

### [ComfyUI-Flux-FaceIR](https://github.com/cosmicrealm/ComfyUI-Flux-FaceIR)

ComfyUI extension for FLUX FaceIR face restoration. It supports aligned face restoration from cropped face patches and full-image restoration with face detection, alignment, parameter export, and paste-back.

Links: [Code](https://github.com/cosmicrealm/ComfyUI-Flux-FaceIR)

## FLUX Inference

### [flux-inference](https://github.com/cosmicrealm/flux-inference)

FLUX.1-dev inference optimization project with PyTorch baseline, ONNX Runtime, and TensorRT backends, covering model export and optimized image generation pipelines.

Links: [Code](https://github.com/cosmicrealm/flux-inference)
