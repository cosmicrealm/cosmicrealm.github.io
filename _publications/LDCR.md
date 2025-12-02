---
title: "Learning Discriminative Compact Representation for Hyperspectral Imagery Classification"
collection: publications
category: manuscripts
permalink: /publication/LDCR
excerpt: 
date: 2019-10-01
venue: 'IEEE Transactions on Geoscience and Remote Sensing (TGRS) '
paperurl: 'https://ieeexplore.ieee.org/document/8741172'
# citation: 'Your Name, You. (2024). &quot;Paper Title Number 3.&quot; <i>GitHub Journal of Bugs</i>. 1(3).'
---
[Code](https://github.com/zhangjinyangnwpu/LDCR)

Abstract:
Abundant spectral information of hyperspectral images (HSIs) has shown an obvious advantage in improving the performance of classification in the remote sensing domain. However, this is paid by the expensive consumption on the computation, transmission, as well as storage of HSIs. To address this problem, we propose to learn the discriminative compact representation for HSIs classification, which not only greatly reduces the data redundancy in the image but also preserves the discriminative information required for pixelwise classification in HSIs. To this end, we present a multi-task deep learning framework, which integrates HSIs autoencoding and classification into a two-branch deep neural network for jointly learning. In the network, we employ an encoder block to learn the compact representation of the input HSI via compression in the spectral domain. Being fed with the compact representation, the autoencoding branch then employs a decoder block to reconstruct the input HSI, while the classification branch utilizes a classifier block to predict the label for each pixel. Through end-to-end joint learning, the compact representation is not only informative enough to accurately reconstruct the original HSI but also discriminative enough to appropriately label each pixel with the trained classier. Sufficient experimental results on four HSIs classification data sets demonstrate the effectiveness of the proposed framework.