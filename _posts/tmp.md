---
layout: post
title: "Paper Reading: FLOAT — Flow Matching for Audio-driven Talking Portraits"
date: 2025-12-04
categories: [paper-reading, talking-head, flow-matching]
tags: [FLOAT, flow-matching, LIA, talking-head, audio-driven, motion-latent]
---

# FLOAT: Flow Matching 驱动的语音口型合成方法阅读笔记

本笔记对论文 **FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait** 进行系统的工程化解读。内容包括：

- Motion Latent Space 与 LIA 背景  
- Motion Auto-Encoder 的结构与训练  
- Flow Matching 如何用于序列生成  
- Frame-wise AdaLN 与 Temporal Masked Attention  
- L 与 L′ 的窗口机制  
- FLOAT 的消融实验（Ablation Study）——**新增部分**  
- 推理阶段的细节与指导信号 CFV  

---

# 1. Motivation：FLOAT 为何存在？

当前的 audio-driven talking head 主要利用 Stable Diffusion latent space 建模每帧图像，但存在：

- 跨帧不稳定  
- 嘴部同步不准确（尤其破音、爆破音）  
- 对表情迁移、非语言动作支持弱  
- 推理速度慢  

FLOAT 的核心创新是：

> **在 Motion Latent Space 中进行序列生成 + Flow Matching 替代 Diffusion**。

使得整个系统更加高效、更稳定、更易控制。

---

# 2. Motion Latent Space（来自 LIA）

FLOAT 采用了 LIA 中常见的 identity-motion 分解：

$$
w = w_{\text{id}} + w_{\text{motion}}
$$

并对 motion latent 建立一组正交 basis：

$$
w_{\text{motion}} = \sum_{m=1}^{M} \lambda_m v_m
$$

意义：

- 可控性强（编辑 λ 即可改变表情）  
- 稳定、结构化适合 Flow Matching  

---

# 3. Motion Auto-Encoder

FLOAT 使用一个 Auto-Encoder 将图像空间映射到 motion latent。  
对每帧：

$$
w_D = w_{D\to r} + w_{r\to D}
$$

训练目标：

- 来源图 S 提供身份 latent  
- 驱动图 D 提供运动 latent  
- 重建 D 以监督 motion encoder

损失包含：

- 全图 L1 + LPIPS  
- 局部嘴部、眼睛 LPIPS  
- 全局 & 局部 GAN  
- 局部 FSM（改善牙齿、眼睛结构）

---

# 4. Flow Matching 在视频序列中的使用

Flow Matching 直接学习一个向量场：

$$
x_t = (1 - t)x_0 + t x_1
$$

并学习：

$$
v_\theta(x_t, c_t) \approx x_1 - x_0
$$

相比 diffusion：

- 更少采样步数  
- 直接建模连续轨迹  
- 更适合序列建模  

---

# 5. Frame-wise AdaLN（逐帧条件调制）

FLOAT 使用逐帧 AdaLN，而不是 cross-attention：

$$
x^l = \gamma^l(c^l)\cdot\text{LN}(x^l) + \beta^l(c^l)
$$

优势：

- 每帧可独立调制，利于 lip-sync  
- 情绪控制更精确  
- 对应的音频帧 → 对应的图像帧  

---

# 6. Temporal Masked Attention（局部时间注意力）

FLOAT 只让 attention 关注局部邻域（如 t±2）：

- 限制长距离依赖  
- 减少抖动  
- 稳定表情与头部运动序列  

---

# 7. L′ 与 L：FLOAT 的滑动窗口机制（关键更新）

FLOAT 的序列建模不是一次性处理全序列，而是：

## ✔ L = 当前窗口要生成的帧数  
## ✔ L′ = 上一窗口提供的上下文帧数（history / overlap）

窗口长度 = L′ + L。

序列输入结构：

| 区域 | 长度 | 内容 | 是否预测 |
|------|------|--------|-----------|
| -L′ : 0 | L′ | 上一窗口 motion latent & audio | ❌ 不预测 |
| 1 : L | L | 当前窗口 latent | ✔ 预测 |

这样保证：

- 横跨窗口的嘴部运动连续  
- 头部姿态平滑  
- 长序列无 jump-cut  

---

## Flow Matching Loss (Eq. 11)

$$
\mathcal{L}_{OT}(\theta) =
\| v_t^{1:L} - u_t(x|w_{r→D^{1:L}}) \| +
\| v_t^{-L′:0} - w_{r→D^{-L′:0}} \|
$$

解释：

- 第一项：预测当前窗口的 L 帧  
- 第二项：保持历史 L′ 帧不变，维持跨窗口连续性  

Velocity loss：

$$
\mathcal{L}_{vel} = \|\Delta v_t - \Delta u_t\|
$$

保证时间梯度一致，减少 jitter。

---

# 8. 推理阶段（Inference）

推理时采用滑动窗口：

1. 初始化第一窗口的 L′ 帧为零  
2. 生成 L 帧  
3. 取最后 L′ 帧作为下一窗口 history  
4. 重复直到生成完所有帧  

推理效率高，因为 Flow Matching 通常只需 ~10 步。

---

# 9. 消融实验（Ablation Study）

这一部分是新增集成内容，对 FLOAT 的设计决策进行逐项分析。

---

## 9.1 Frame-wise AdaLN vs Cross-Attention

论文实验对比了两种方式：

> 逐帧 AdaLN 更好地生成表情、嘴部同步，并产生更丰富的头部动作。  
> （Tab. 3）

原因：

- AdaLN 逐帧调制 γ、β → 精确控制每帧  
- Cross-attention 会让条件与序列互相干扰  
- 在局部 Mask Attention 中，AdaLN 控制力更强  

### 综合评价

| 方法 | lip-sync | 表情质量 | 多样性 | 时序稳定性 |
|------|-----------|------------|------------|----------------|
| **Frame-wise AdaLN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Cross-Attention | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 9.2 Flow Matching vs Diffusion（ϵ / x₀ prediction）

对比项：

1. Diffusion (ϵ prediction)  
2. Diffusion (x₀ prediction)  
3. **Flow Matching（本方法）**

**结果：**

- FID、FVD 持平甚至更好  
- lip-sync（LSE-D）明显优于 Diffusion  
- 推理速度是 Diffusion 的 5×  

表格摘要：

| 方法 | LSE-D ↓ | FVD ↓ | NFE |
|------|---------|---------|------|
| Diffusion | 较差 | 中等 | 50 |
| **Flow Matching** | **最好** | **最好** | **10** |

说明 Flow Matching 更适合捕捉嘴部微运动的连续性。

---

## 9.3 Guidance 系数 γₐ（音频）与 γₑ（情绪）

论文发现：

- γₐ ↑ → lip-sync & FVD 提升  
- γₑ ↑ → 情绪表达更强（E-FID 提升）

表格（Tab. 6）显示，最佳组合通常为：