---
title: "Improving Hyperspectral Image Classification with Unsupervised Knowledge Learning"
collection: publications
category: conferences
permalink: /publication/UKL
excerpt: 
date: 2019-07-28
venue: '(IGARSS) IEEE International Geoscience and Remote Sensing Symposium'
paperurl: 'https://ieeexplore.ieee.org/document/8898323'

---

[Code](https://github.com/zhangjinyangnwpu/UKL)

Abstract:
Recently, deep convolutional neural networks(DCNNs) based methods have shown pleasing performance in hyperspectral image(HSI) classification. However, due to extensive coefficients resulted by the deep structure, these methods are prone to be overfitting during training, especially when the labeled samples are limited. To address this problem, we propose to learn the unsupervised knowledge from both unlabeled and labeled samples to regularize the conventional supervised learning. Following this idea, we present a two-branch network, in which two branches are separately utilized to perform the clustering and classification based on a shared feature extraction module. Thanks to the shared structure, the crucial unsupervised information (e.g., intra-cluster similarity & inter-cluster dissimilarity, etc.) can be injected into the supervised learning procedure, and thus leads to improved generalization capacity. Experiments on two widely used HSI datasets show the superior performance of the proposed method.