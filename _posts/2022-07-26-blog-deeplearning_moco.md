---
title: 'moco 论文摘要'
date: 2022-07-26
permalink: /posts/2021/05/2022-07-26-blog-deeplearning_moco/
tags:
  - unsupervised learning
  - imagenet
  - moco
---


### MoCo: Momentum Contrast for Unsupervised Visual Representation Learning


- 学习一个又大又一致的字典
    - 学习的方式总结为 4 个方向：
        - 生成式，比如 autoencoder
        - 判别式，比如位置预测
        - 对比学习的方式，
        - 对抗训练，GAN
    - 大，表示模型可以表示的信息丰富
    - 一致，需要避免模型学习一个 shortcut solution
- 使用对比学习的方式，用无监督的方式训练模型
    - 使用 infoNCE，
        - 对于 instance classification 的任务，种类数过多是的模型最后的全连接层参数过多，无法训练
        - nce 采用另一种策略，只讲样本与噪声样本区分，因此可以表示为一个二分类任务，区分正样本和噪声样本，可以看成一个二分的交叉熵损失
        - infoNCE 在 nce 的基础上，将噪声样本再细分为 k 类，本质上也是一个交叉熵损失，一个 anchor(query) sample，对用 1 个 postive(query) sample 和 K 个 negative(query) sample
- 使用动量更新的方式，保证字典学习到特征的一致性
    - 作者介绍了三种网络更新的方式
        - end-to-end, 难以保证字典的大小，首先于 batch size 的大小，即显存的大小
        - memory bank，难以保证学习到的特征的一致性，将数据集的所有图片的特征离线保存，然后每次采样负样本后，动态更新特征，这样一个 opoch，整个字典才能更新一次
        - moment update，使用一个队列，动态维护队列的大小，同时使用动量更新的方式来更新 key 对应的编码器的参数
- 将任务理解成一个 query 和 key 查询的问题，（正样本对应的 query，负样本对应的 key），每次都在字典中对 query 进行一次查询
    - 使用队列保证维护一个大的字典
    - 每次只更新 query encoder 对应的参数，key encoder 的参数通过动量更新的方式，缓慢更新，保证所有 key 的一致性
    - 不需要很大的 batch size
- 需要注意的几点
    - 这里的学习率设置的很大：30，比较反常，但是也说明无监督学习的方式和有监督可能不一样
    - 使用了 sync-BN 操作
    - 在伪代码中，labels 使用的是 0，要注意意思为所有的正样本相当于恰好都是第 0 类，作者的实现超级优雅
    - 越大的 moment 参数效果越好，0.9999 better than 0.9，为了建立一个一致性的字典
    - 是否应该使用更好的代码任务，不仅使用 instance discrimination，作者已经想到，使用 mae
    - 越大的数据集，模型性能应该更好，最好不要存在上限