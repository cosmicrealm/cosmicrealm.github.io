---
title: 'Self-Forcing 与 Self-Forcing++ 演进之路'
date: 2026-01-08
permalink: /posts/2025/12/2026-01-08-blog-video-gen-self-forcing/
tags:
  - llm
---

[self-forcing](https://github.com/guandeh17/Self-Forcing)

[self-forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus)

# Self-Forcing & Self-Forcing++：从训练-推理鸿沟到分钟级实时流式视频生成（技术拆解）

> 核心关键词：Autoregressive Video Diffusion、Exposure Bias、KV Cache、Rolling KV、Few-step Diffusion、Distribution Matching Distillation、Minute-scale Generation、FPS / Latency

---

## 1. 背景：为什么“自回归视频扩散”会在长视频上崩？

自回归视频生成通常写成：

$$
p_\theta(x_{1:N})=\prod_{i=1}^{N} p_\theta(x_i \mid x_{<i})
$$

**训练-推理鸿沟（train-test gap / exposure bias）**来自于：

- 训练阶段常用 **Teacher Forcing (TF)**：条件上下文用真实历史帧 $$x_{<i}$$；
- 推理阶段必须用 **模型自己生成的历史帧** $$\hat x_{<i}$$ 当条件。

结果：误差会随时间累积，越往后越偏离“训练时见过的条件分布”，最终出现：
- 画面冻结、漂移、闪烁
- 语义/角色一致性断裂
- 长程结构坍塌（超过训练片段长度尤其明显）

Self-Forcing 系列的中心思想是：**训练时就让模型以“真实推理方式”rollout，并在 rollout 分布上做整体分布对齐**，从根上缩小 train-test gap。

---

## 2. 统一符号与张量维度（工程实现对照）

为了能落到代码层面，先把维度讲清楚（论文通常不写死具体值，用符号化更通用）。

### 2.1 基础维度
- batch：$$B$$
- 视频长度（训练/推理生成的 latent 帧数）：$$N$$
- 每帧 RGB 分辨率：$$H \times W$$
- VAE latent 分辨率：$$H_\ell \times W_\ell$$
- latent 通道数：$$C_\ell$$
- Transformer hidden dim：$$D$$
- attention head 数：$$n_h$$，每头维度：$$d_h$$，满足 $$D=n_h\cdot d_h$$

### 2.2 张量表示
- 像素空间视频：$$x^{pix}_{1:N} \in \mathbb{R}^{B\times N\times H\times W\times 3}$$
- latent 空间视频：$$x_{1:N} \in \mathbb{R}^{B\times N\times C_\ell\times H_\ell\times W_\ell}$$

把每帧 latent patchify / flatten 成 token：
- 每帧 token 数：$$P=H_\ell\cdot W_\ell$$
- 单帧 token：$$\text{tok}(x_i) \in \mathbb{R}^{B\times P\times D}$$
- 历史上下文 token 长度：$$S_{<i}=(i-1)\cdot P$$

### 2.3 KV Cache 形状（典型实现）
对第 $$l$$ 层：
- $$K^{(l)},V^{(l)} \in \mathbb{R}^{B\times n_h\times S\times d_h}$$  
其中 $$S$$ 是当前 cache 中累计 token 数（rolling KV 会限制 $$S\le L\cdot P$$）。

---

## 3. Base：Self-Forcing（核心机制 = 推理式 rollout + 视频级分布对齐）

Self-Forcing 的两个关键点：

1) **训练时真的按推理方式做自回归 rollout**：  
每一步生成当前帧时，条件上下文来自模型自己的历史输出。

2) **损失从帧级 denoising（MSE）升级为视频级分布匹配（holistic distribution matching）**：  
直接对齐“模型会生成的视频分布”与“真实数据分布”。

---

## 4. TF / DF：传统训练方式的数据流与损失（对比基线）

### 4.1 扩散前向加噪（每帧）
$$
x_t = \alpha_t x + \sigma_t \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

维度不变：$$x_t,\epsilon \in \mathbb{R}^{B\times C_\ell\times H_\ell\times W_\ell}$$。

### 4.2 Teacher Forcing (TF) 与 Diffusion Forcing (DF)
- TF：条件上下文 $$c = x_{<i}$$（干净 GT 历史帧）
- DF：条件上下文 $$c = \{x^j_{t_j}\}_{j<i}$$（带噪历史帧）

### 4.3 帧级 denoising 损失（典型）
模型预测噪声 $$\hat\epsilon_\theta$$，损失：

$$
\mathcal{L}^{DM}_\theta=\mathbb{E}_{x,t,\epsilon}\left[w_t\left\|\hat\epsilon_\theta-\epsilon\right\|_2^2\right]
$$

> 痛点：即便 DF 用带噪上下文，也仍然不等价于推理时“模型自生成历史帧”分布；训练-推理分布仍然错位。

---

## 5. Self-Forcing 训练：推理式 rollout + 可训练的资源控制

### 5.1 训练核心：从模型分布 rollout
训练时生成整段：

$$
\{x^\theta_{1:N}\}\sim p_\theta(x_{1:N})
$$

即每一帧（或 chunk）的条件来自前面模型输出，并用 KV cache 累积上下文。

### 5.2 few-step diffusion：把每帧扩散步数压到很少
对每帧生成不走长链路（几十/上百步），而走少步时间网格：

$$
0=t_0<t_1<\dots<t_T=1
$$

其中 $$T$$ 通常很小（工程上可做到 4 steps）。

每帧内部 denoise loop（概念）：
1. 初始化 $$x_{t_T}\sim\mathcal{N}(0,I)$$
2. for $$j=T..1$$：
   - $$\hat x_0 \leftarrow G_\theta(x_{t_j}, t_j, KV)$$
   - 若 $$j>1$$：重加噪得到 $$x_{t_{j-1}}$$

### 5.3 训练能跑起来：梯度截断 + 随机截断步
直接对“AR rollout × diffusion steps”全图反传会爆显存/爆算力。

常用做法（自回归 + few-step 场景很有效）：
- 随机采样一个截断步 $$s\in\{1..T\}$$；
- 只在 $$j=s$$ 的那一次前向开启梯度，其余 denoise steps 用 `no_grad`；
- 同时对历史帧的 KV cache 做 `stop_grad`，避免跨帧反传。

工程直觉：**让每个时间步都有机会被监督，但每次只训练一小段图**。

---

## 6. Self-Forcing 的“视频级损失”：三种可选目标（DMD / SiD / GAN）

Self-Forcing 的关键升级：不再只优化逐帧 MSE，而是直接对齐整体视频分布。

典型做法：对真实视频与生成视频都加同一层 forward diffusion 噪声，比较噪声域分布 $$p_{*,t}$$。

### 6.1 DMD：反向 KL（Reverse KL）
目标形式：

$$
\mathbb{E}_t\left[D_{KL}\left(p_{\theta,t}\,\|\,p_{data,t}\right)\right]
$$

实现上常通过 teacher / score 估计，把“分布对齐梯度”转为可训练的方向信号。

### 6.2 SiD：Fisher divergence（score matching 风格）
$$
\mathbb{E}_{t,\,x\sim p_{\theta,t}}\left[\left\|\nabla\log p_{\theta,t}(x)-\nabla\log p_{data,t}(x)\right\|_2^2\right]
$$

### 6.3 GAN：JS 近似的对抗分布对齐
训练判别器区分真实/生成视频，生成器（rollout 的 AR diffusion 模型）通过对抗学习逼近数据分布。

> 三者本质：都在“推理分布上的样本”层面做 **holistic distribution alignment**，而不是只做帧级 denoising。

---

## 7. 推理：如何做到“实时流式”？

Self-Forcing 的推理阶段就是：**自回归 + few-step + KV cache**。

### 7.1 推理数据流（frame-wise / chunk-wise）
- 输入：prompt conditioning（文本 embedding 等）
- 初始化 KV cache 为空
- 对 i=1..N：
  - 初始化当前帧/当前 chunk 的噪声 latent：$$x_{t_T}$$
  - 运行 $$T$$ 次 denoise（T 很小，如 4）
  - 得到 $$\hat x_0$$（当前输出）
  - 计算当前输出 token 的 K/V 并 append 到 cache
  - 立即解码输出（可流式边生成边播放）

**流式关键**：每生成一帧/一段，就能立刻输出，不需要等完整视频生成结束。

---

## 8. 为什么可以跑到 FPS≈17：三个增益叠加

把“视频扩散很慢”的主要开销拆成两类：
1) **扩散步数**（网络前向次数）
2) **注意力计算**（上下文长、滑窗重算等）

Self-Forcing 的 17 FPS 主要来自：

### 8.1 Few-step diffusion：把网络前向次数从“几十/上百”降到“4”
扩散推理时间大体随步数线性增长。把 $$T$$ 压到 4，吞吐直接提升一个数量级。

### 8.2 因果注意力 + KV cache：历史不重算
每次只需要为“新 token”计算 Q，并与 cache 中的 K/V 做 attention；历史帧的 K/V 不再重复计算。

### 8.3 Chunk-wise：一次生成多个 latent 帧，摊薄固定开销
如果每次生成的是一个 chunk（例如 3 个 latent frames），那么：
- 调度器开销、cache 管理开销、部分 embedding 开销被摊薄；
- 吞吐更高、首帧延迟仍可控。

---

## 9. Base 的长视频问题：Rolling KV 的训练/推理不一致

即使推理用 rolling KV 来保持长视频复杂度稳定，如果训练阶段使用的是不同的 cache 策略（例如固定窗口或不同的 cache 行为），模型依然会遇到 **cache-state mismatch**：推理时遇到的“滚动上下文状态”在训练时没见过，长视频仍可能闪烁或退化。

这正是升级版 Self-Forcing++ 要解决的核心之一。

---

## 10. 升级版：Self-Forcing++（分钟级高质量长视频的关键改动）

Self-Forcing++ 的目标：把“只会 5 秒”升级为“分钟级”（100s 甚至更长）仍保持质量与一致性。

核心改动可以概括为三件套：
1) **长 rollout（N ≫ K）**
2) **Extended-DMD：从长 rollout 中随机抽窗做分布蒸馏**
3) **训练阶段也使用 rolling KV cache**
并辅以一个可选增强：
4) **GRPO + optical-flow reward**（长期平滑性）

---

## 11. Self-Forcing++ 训练数据流（含维度）

### 11.1 关键长度标量
- $$N$$：student rollout 总长度（很长，分钟级）
- $$K$$：teacher 能覆盖的窗口长度（短，通常对应训练上限，比如 ~5s 的 latent 长度）
- $$L$$：rolling KV 的 cache 长度上限（以帧或 token 计）
- chunk size：一次生成 $$m$$ 个 latent frames（如 m=3）

### 11.2 Step-by-step（核心流程）
#### Step A：student 长 rollout（使用 rolling KV）
生成：

$$
V \in \mathbb{R}^{B\times N\times C_\ell\times H_\ell\times W_\ell}
$$

并且 KV cache 在整个 rollout 中滚动维护，保持推理一致的 cache-state 分布。

#### Step B：从长轨迹中均匀采样一个连续窗口
采样起点 $$i\sim\{1,\dots,N-K+1\}$$：

$$
W = V[i:i+K-1] \in \mathbb{R}^{B\times K\times C_\ell\times H_\ell\times W_\ell}
$$

#### Step C：Backward Noise Initialization（后向噪声初始化）
直觉：窗口蒸馏不能从“纯随机噪声”起步，否则和长上下文解耦。  
做法：对窗口 $$W$$ 重新注入噪声，构造与扩散时间步一致的输入 $$x_t(W)$$：

$$
x_t(W)\in \mathbb{R}^{B\times K\times C_\ell\times H_\ell\times W_\ell}
$$

这样 teacher / student 在同一“长上下文一致”的带噪窗口上对齐。

#### Step D：Teacher / Student 在窗口上输出 score / noise 预测
- teacher 输出：$$s^T(x_t(W),t)$$
- student 输出：$$s^S_\theta(x_t(W),t)$$  
二者维度同 $$x_t(W)$$。

#### Step E：Extended-DMD 更新
在随机窗口上做分布匹配蒸馏（本质是把 teacher 的“局部纠错能力”教给 student，使其能从长 rollout 的退化态恢复）。

---

## 12. Self-Forcing++ 损失函数：Extended-DMD +（可选）GRPO

### 12.1 Extended-DMD（核心监督）
在随机窗口上最小化 student 与 teacher 的分布差异（可写成 KL 形式的目标），并用 DMD 的梯度估计实现可训练更新。

关键点：监督覆盖的是“长 rollout 中任意位置的短窗口”，因此 student 会学到：
- **错误累积态下的恢复能力**
- **分钟级生成中局部片段仍像“真实短视频分布”**

### 12.2 GRPO + Optical-flow reward（可选增强）
分钟级生成仍可能出现长期一致性衰减（物体突然出现/消失、突兀切场）。  
引入 RL 微调，用光流相关的 reward 作为时间平滑 proxy，进一步优化长程稳定性。

---

## 13. Base vs ++：对比总结（关键差异点）

| 维度 | Self-Forcing (base) | Self-Forcing++ |
|---|---|---|
| 训练覆盖时间 | 短（teacher 覆盖长度内） | 长 rollout 到 $$N\gg K$$，从长序列抽窗蒸馏 |
| 蒸馏/对齐区域 | 生成段整体（但主要是短域） | 任意位置窗口（覆盖退化态） |
| 噪声初始化 | 常规扩散起点 | **Backward Noise Init** 保持窗口与长上下文一致 |
| KV 策略 | 推理 rolling，但训练可能不同 | **训练与推理都 rolling KV**（消 cache-state mismatch） |
| 长视频退化处理 | 依赖 base 对齐能力 | 显式学习“从退化态纠错恢复” + 可选 RL 平滑 |

一句话：  
**Self-Forcing++ = “先让学生在真实推理形态下长跑出错，再用 teacher 在短窗口里纠错式蒸馏，把恢复能力学回去”，并让训练/推理 cache 行为完全一致。**

---

## 14. 工程实现要点（能跑到流式实时的关键细节）

### 14.1 推理循环（伪代码）
```python
KV = init_empty_cache()

for i in range(num_steps_over_time):  # frame-wise or chunk-wise
    x_t = randn_latent()  # [B, Cℓ, Hℓ, Wℓ] or [B, m, Cℓ, Hℓ, Wℓ]
    for j in reversed(range(T)):      # T is small, e.g. 4
        x0_hat = model(x_t, t_j, KV)  # predict x0 (or eps)
        if j > 0:
            x_t = add_noise(x0_hat, t_{j-1})  # one-step scheduler
    out = x0_hat
    KV.append(compute_kv(out))
    KV = rolling_evict_if_needed(KV, L)
    stream_decode_and_send(out)  # optional: decode immediately

```
### 14.2 训练的资源控制（伪代码骨架）
```
    # rollout phase (no_grad for most)
    KV = init_empty_cache()
    V = []
    for i in range(N):
        out = autoregressive_generate_one_step(KV, T=few_steps, grad=False)
        V.append(out)
        KV.append(compute_kv(out))
        KV = rolling_evict_if_needed(KV, L)

    # sample window
    i0 = uniform(0, N-K)
    W = stack(V[i0:i0+K])

    # backward noise init
    t = sample_timestep()
    Wt = add_noise(W, t)

    # distillation update (enable grad only here)
    loss = extended_dmd_loss(student(Wt,t), teacher(Wt,t))
    loss.backward()
    opt.step()
```

### 14.3 为什么 rolling KV 能让长视频不降速？

不 rolling：上下文长度随时间增长，注意力开销越来越大；

rolling：上下文 token 长度上限固定为 $$L\cdot P$$，每步只做“固定上限”的 attention，长视频复杂度稳定。

## 15. 结语：一条清晰的路线图

要实时流式：few-step（如 4 steps） + causal attention KV cache +（可选）chunk-wise

要长视频不崩：rolling KV + 在训练中也使用 rolling KV（避免 cache-state mismatch）

要分钟级高质量：长 rollout + 随机抽窗 Extended-DMD + backward noise init（保持上下文一致）

要更平滑：在分钟级后训练阶段加上 GRPO / flow reward 等时间一致性奖励