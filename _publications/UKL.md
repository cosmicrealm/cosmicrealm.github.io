---
title: "Improving Hyperspectral Image Classification with Unsupervised Knowledge Learning"
collection: publications
category: conferences
permalink: /publication/UKL
excerpt: 
date: 2019-07-28
venue: '(IGARSS) IEEE International Geoscience and Remote Sensing Symposium'
teaser: /images/publications/ukl-framework.jpg
teaser_alt: UKL framework figure
summary: 'Unsupervised knowledge learning for hyperspectral image classification, injecting clustering structure into supervised learning to improve generalization.'
paperurl: 'https://ieeexplore.ieee.org/document/8898323'
codeurl: 'https://github.com/zhangjinyangnwpu/UKL'

---

[Code](https://github.com/zhangjinyangnwpu/UKL)

Abstract:
Deep CNN-based methods perform well on hyperspectral image classification, but their large parameter counts make them prone to overfitting when labeled samples are limited. This work introduces unsupervised knowledge from both labeled and unlabeled samples as a regularizer for supervised learning. The proposed two-branch network shares a feature extractor while separately performing clustering and classification, allowing clustering structure such as intra-cluster similarity and inter-cluster dissimilarity to guide the supervised branch. Experiments on two widely used HSI datasets show improved generalization and classification performance.
