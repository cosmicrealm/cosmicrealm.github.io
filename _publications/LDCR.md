---
title: "Learning Discriminative Compact Representation for Hyperspectral Imagery Classification"
collection: publications
category: manuscripts
permalink: /publication/LDCR
excerpt: 
date: 2019-10-01
venue: 'IEEE Transactions on Geoscience and Remote Sensing (TGRS) '
summary: 'A multi-task deep network for hyperspectral image classification that jointly learns compact spectral representation, reconstruction, and pixelwise classification.'
paperurl: 'https://ieeexplore.ieee.org/document/8741172'
codeurl: 'https://github.com/zhangjinyangnwpu/LDCR'
# citation: 'Your Name, You. (2024). &quot;Paper Title Number 3.&quot; <i>GitHub Journal of Bugs</i>. 1(3).'
---
[Code](https://github.com/zhangjinyangnwpu/LDCR)

Abstract:
Hyperspectral images provide rich spectral cues for remote-sensing classification, but their high dimensionality increases computation, transmission, and storage costs. This paper learns a discriminative compact representation that reduces redundancy while preserving information needed for pixelwise classification. The method uses a multi-task two-branch network: an encoder compresses the input HSI, an autoencoding branch reconstructs the original data, and a classification branch predicts labels from the compact representation. Joint training encourages the representation to be both reconstructive and discriminative, and experiments on four HSI datasets demonstrate the effectiveness of the framework.
