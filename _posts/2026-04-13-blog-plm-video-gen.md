---
title: LPM 1.0：视频驱动角色 Performance 模型的全栈解析
date: 2026-04-13
permalink: /posts/2026-04-13-blog-plm-video-gen/
tags:
#   - FlowMatching, multimode generation
---


> **论文**：LPM 1.0: Video-based Character Performance Model  
> **项目主页**：[large-performance-model.github.io](https://large-performance-model.github.io)  
> **参数量**：17B (Base LPM) · 基座：Wan2.1-I2V 14B

---

## 开篇点评：这篇论文解决了一个真实存在的问题

在 talking-head 类工作井喷的当下，大多数论文都在追相同的几个指标：PSNR、FID、AV-Sync。LPM 1.0 是少数几篇真正从**应用场景倒推技术设计**的工作之一，值得认真读。

它的出发点是一个非常具体的观察：**现有视频模型无法同时满足表现力、实时推理、长时身份稳定这三个条件**，三者之间存在根本性的张力——专注表现力往往破坏实时性，追求速度则导致动作僵化，维持长时稳定又容易让身份和细节逐渐漂移。这个三角被命名为 **Performance Trilemma**，是全文分析框架的基石。

更聪明的选择是，他们没有试图在所有任务上同时碾压，而是把目标场景锁定在 **single-person full-duplex audio-visual conversation**——这个场景天然地把三难困境压缩到最极端的形式：说话时需要精确口型同步（表现力），倾听时需要实时非语言反应（实时性），整个对话过程要维持视觉身份稳定（长时稳定）。选好赛道比堆参数更重要。

从技术路线看，LPM 1.0 的核心主张是：**Performance Trilemma 不是一个架构问题，而是一个系统级问题**。它需要数据、控制信号、推理机制三个层面的协同设计，而不能靠单一模块的精巧来解决。这个判断在论文展开后得到了验证——数据层的三状态标注、Base Model 的交错音频注入、Online Model 的 Backbone-Refiner 分离，每一个设计决策都是在系统层面回应同一个问题的不同侧面。

当然，这篇论文也有一些值得审视的地方，放到最后讨论。

---

## 一、问题框架：Performance Trilemma

### 1.1 三个维度的定义

| 维度 | 描述 | 当前工作的典型失败模式 |
|---|---|---|
| **Expressive Quality** | 真实人类动作——丰富的对话动作、微表情、非重复性 | 口型精准但肢体僵化，缺乏非语言信号 |
| **Real-time Inference** | 因果式实时生成，支持流式直播 | 高质量生成通常需要多步去噪，无法实时 |
| **Long-horizon Stability** | 长时间跨度内身份、解剖结构、视觉保真度稳定 | 自回归漂移导致面部/体型随时间退化 |

### 1.2 Conversation 为何是最苛刻的场景

对话场景同时激活所有三个维度的压力：

- **说话阶段**：高频局部运动（口型），精确时序对齐
- **倾听阶段**：低频全局反应（点头/眼神/微表情），语义感知
- **身份维持**：跨说/听切换时面部外观不漂移
- **时间跨度**：一次对话可能持续数分钟乃至数小时

论文还归纳了现有范式的四个缺失：倾听行为缺失、多模态可控性差、单图身份欠规约、缺乏大规模人体中心预训练基座。这四点对应了 LPM 数据构建和架构设计的四条主线。

---

## 二、数据 Pipeline：从 Raw Video 到可训练数据

这是 LPM 1.0 技术含量最高、也最容易被低估的部分。整体留存率不足 10%，最终产出约 **31M 可训练 clips**，总计超过 **1.7 trillion multimodal tokens**。

### 2.1 四阶段过滤漏斗

```
Raw Videos [100%]
    │
    ▼ Stage 1: Single-Shot Extraction
Single-Shot Clips [56.82%]   →  短于 3s 的 clips 被移除 [43.18%]
    │
    ▼ Stage 2: Filtering + Cropping
After Filtering [26.11%]
    │  ├── 人体检测过滤（无人/人群过密）
    │  ├── 质量/美学过滤（模糊/偏色/闪烁）
    │  ├── 叠加层过滤（logo/字幕/特效/美颜滤镜）
    │  ├── 构图过滤（不完整人体/画中画布局）
    │  └── 音视频同步过滤（无音频口型/对不上口型）
    │
    ▼ Stage 3: Conversation Detection + Clipping
Speaker          [8.50%,  23M clips]
Conversation/Listen [2.02%,  5M clips]
Idle             [1.01%,  3M clips]
    │
    ▼ Stage 4: Captioning + Embedding Generation
Trainable        [11.52%, 31M clips]
```

**值得注意的工程细节**：

Stage 2 的五类缺陷过滤同时使用人工审核和模型方法的组合，最终人工质检中缺陷率低于 1%。这个 bar 非常高，意味着他们的自动过滤已经做得相当准确。

Stage 3 保留了 multi-person clips，因为多人镜头天然包含更丰富的说/听切换动态，比单人镜头更适合训练对话生成能力。所有 multi-person clips 会先完成 per-person 追踪和裁剪，转化为 single-person 单元，再统一处理。

### 2.2 三状态对话标注：框架设计

说话人检测（Active Speaker Detection, ASD）是现有工具的能力边界——已有方法只做 "谁在说话" 的二分类，对**倾听**和**空闲**没有建模。LPM 将这个问题重新定义为三状态帧级分类：

| 状态 | 判定条件 | 挑战来源 |
|---|---|---|
| **Speaking** | 检测到语音 + 该人口型与音频同步 | 区分前景说话与后景噪声 |
| **Listening** | 检测到语音 + 该人口型静止 | 区分在对话中倾听 vs 无关的口型静止 |
| **Idle** | 无语音 + 口型静止 | 区分自然停顿与真实空闲 |
| **丢弃** | 检测到语音 + 口型运动但不同步 | 表示数据不一致，直接过滤 |
| **丢弃** | 无语音 + 存在无法解释的口型运动 | 同上 |

这个分类逻辑的核心信号是两个：**语音检测**（谁的语音、何时有语音）+ **音视频同步**（该人的口型是否与音频对应）。两个信号的联合状态决定帧级标签。

### 2.3 LR-ASD 微调：三状态分类器的实现

直接在 LR-ASD（Lightweight and Robust ASD）backbone 上通过迁移学习实现三状态扩展：

**训练数据**：内部标注的 20K 单镜头 clips（约 95 小时），人工帧级标注。

**性能结果**：

| Domain | 准确率 | Speak Recall | Listen Recall |
|---|---|---|---|
| Domain 1 | 89.75% | 91.62% | 87.99% |
| Domain 2 | 87.63% | 94.05% | **81.05%** |

Domain 2 的 listen recall 显著低于 speak recall（81.05% vs 94.05%）——这正是论文预期的失败模式：在声学变化较大的域（环境噪声、画外旁白），部分真实的倾听帧被误分为说话。这个误差是系统性的，正是下一步语义验证要处理的。

**多人镜头的 Re-ranking 模块**：

对于多人场景，在 per-person 概率序列上做规则驱动的帧级 re-ranking：

1. 逐帧对比各人的说话/倾听置信度
2. 时序平滑 + 持续性检验，抑制短暂尖峰
3. 优先选取时序一致的说话者-倾听者模式（稳定的说话段 + 连贯的切换）

基于 re-ranking 结果，为每个多人 clip 构建两路音频变体：
- **Speaker-only audio**：抑制倾听者的语音分量
- **Listener-only audio**：抑制说话者的语音分量（空闲段两路都保留）

### 2.4 语义验证：Qwen3-Omni Fine-tuning

帧级标注基于低层音视频信号，存在两类系统性误差：

- **False Listening**：模型把低能量语音段（停顿/轻声）中的静止口型标为 listening，实际上对方还在说话
- **Not-in-Dialogue Listening**：有语音但可见人物与之无语义关联（画外旁白/背景对话）

解决方案：在 Qwen3-Omni 上用 15K 人工标注数据微调一个**音视频理解模型**，做六类语义验证分类：

| 类别 | 含义 |
|---|---|
| Conversation | 可见人物参与双向对话 |
| Listen_dialogue | 在对话中倾听 |
| Listen_nondialogue | 对非对话音频的听觉反应（如随音乐律动） |
| Silence | 无语音，无明显反应 |
| Speak | 在说话 |
| Unknown | 无法判断 |

**对比结果**（vs Gemini 2.5 Pro baseline）：

| 模型 | Conversation | Listen_dlg | Listen_nondlg | Silence | Speak | Overall F1 |
|---|---|---|---|---|---|---|
| Gemini 2.5 Pro | 76.80 | 67.58 | 64.76 | 63.96 | 69.88 | 70.47 |
| Qwen3-Omni (ft) | 76.22 | 76.48 | 56.77 | **83.36** | **80.55** | **78.37** |
| **Δ** | −0.58 | +8.90 | −7.99 | **+19.40** | **+10.67** | **+7.90** |

最大改进在 silence（+19.40）和 speak（+10.67），直接对应两类主要失败模式：silence-to-speak 误分率从 24.4% 降至 9.0%，conversation-to-listen_dialogue 泄漏从 12.7% 降至 4.8%。

Qwen3-Omni 在 Listen_nondialogue 上有所退步（−7.99），说明 fine-tuning 数据在这个类别上的覆盖不足——这也是论文诚实指出的 residual weakness。

### 2.5 Audio-Video Captioning

对每个 clip 生成两种互补的 caption：

- **固定长度 clip 的 caption**：描述动作、情绪、环境、镜头和电影化属性的密集自然语言
- **句子边界的变长段 caption**：先用 ASR + 标点拆分为语义完整的句子单元，再对每段标注

Caption 包含精细的分类标签，覆盖情绪类别（78 类）、表情类别（22 类基础表情）、运动描述符（5000+）、镜头类型等。这些标签在推理时被用作文本控制信号。

**倾听视频的 caption 特殊处理**：听者 caption 被限制为**时不变属性**（外貌/性格/关系），不做逐帧行为描述。理由是：在线生成时，说话者音频流到来时来不及生成帧级描述，必须让模型直接学习从音频信号到视觉反应的映射，而不依赖文本作为中间媒介。这个设计约束反向推动了模型能力——倾听生成能力必须来自对音频的真正理解，而非对文本指令的执行。

**倾听数据的分布再平衡**：

约 3.5M 倾听 clips 的分布严重偏斜：情绪和表情 70%+ 集中在中性/认知类，运动强度约 90% 低强度。对此做针对性筛选：

- 筛出 ~470K 高表现力/高参与度 clips 做 SFT
- 对中性类下采样，对罕见类（高能反应/强烈表情/多样身体动作）上采样
- 额外收录两类对训练动态倾听行为有特殊价值的 clip：
  - **情感对比**（emotion contrast）：单个 clip 内对立情感共存（如好笑与尴尬交替）
  - **表情振荡**（expression oscillation）：表情三阶段切换（如从微笑到惊讶再回微笑）

### 2.6 多粒度身份参考图 Pipeline

单张参考图的根本缺陷：模型被迫推断从未见过的角度、未见过的表情状态下角色的样貌，在长序列中不可避免地产生漂移。LPM 的解决方案是为每个训练对象构建**三类互补参考**：

#### (a) 全局外观参考（Global Reference）

从长视频源中随机采样若干帧，刻意包含**训练 clip 以外的帧**。

这个设计有明确的反过拟合动机：如果参考帧总是从同一短 clip 中取，模型可能学到 pixel-level copy-paste，在生成时出现"粘贴感"伪像，而不是理解和重建角色外观。通过跨时间采样，模型被强迫真正内化角色的视觉身份，而不是机械地复制像素。

#### (b) 多视角体参考（Multi-view Body References）

技术上最有挑战性的部分。需要自动判断每帧中摄像机与人体的相对视角，才能有系统地选出覆盖各方向的帧集合。

实现路径：

```
视频帧
  │
  ▼
GVHMR（World-grounded Human Motion Recovery）
  → 估计人体朝向（SMPL root joint Z轴方向）
  │
  ▼
SLAM（DROID-SLAM）
  → 估计摄像机位姿
  │
  ▼
计算摄像机与人体朝向的夹角 θ
  │
  ▼
按阈值分类为 4 个视角类别：
  - Frontal:       θ ≥ 155°
  - Rear:          θ ≤ 45°
  - Left-profile:  45° < θ < 155°，relative direction x分量为负
  - Right-profile: 45° < θ < 155°，relative direction x分量为正
  │
  ▼
从每类中选取代表性帧，组成 1–4 张多视角体参考集
```

这让模型具备**视角一致生成**的先验：知道角色从后面看服装 logo 是什么样，从侧面看头部轮廓是什么样，在头部转动时不需要凭空猜测。

#### (c) 表情参考（Facial Expression References）

长对话中，角色特定的表情细节（笑容形态、笑时的牙齿暴露方式、皱纹纹路）是最难从单张中性表情参考图推断的信息，也是最容易产生跨帧不一致的因素。

构建流程：

1. 仅使用原始分辨率 ≥ 1080P 的视频（保证面部细节清晰）
2. 用 EmotiEff Lib 扫描视频，定位包含 8 类预设表情的帧
3. 筛选包含至少 2 种不同表情的 clips，保证每个对象的参考集内部多样性
4. 用图像理解模型（Gemini 2.5 Pro）对提取帧做**二次语义验证**，修正由过渡表情或模糊表情导致的错误标签

最终每个对象的表情参考集包含 1–8 张，涵盖其实际的表情空间。

---

## 三、Base LPM：模型架构详解

### 3.1 总体架构

Base LPM 在 Wan2.1-I2V（14B）基础上新增约 3B 参数，总计 **17B**。架构核心是一个多模态条件化的 DiT，每个 block 三段式：

```
Self-Attention (AdaLN)
    ↓  [视频 token + 第一帧 token + 多参考图 token，联合计算]
Multi-modal Cross-Attention
    ↓  [文本全局注意力 + 音频交错注意力，分支输出加权合并]
FFN (AdaLN)
```

输入条件：
- 噪声视频 latent $x_t \in \mathbb{R}^{B \times C \times T \times H \times W}$
- 扩散时间步 $t$
- 文本描述 $c_\text{text}$
- 说话音频 $c_\text{speak}$
- 倾听音频 $c_\text{listen}$
- 多参考图集合 $\{I_k\}_{k=1}^{K}$

输出：噪声预测 $\epsilon_\theta(x_t, t, c_\text{text}, c_\text{speak}, c_\text{listen}, \{I_k\})$

### 3.2 交错双音频注入（Interleaved Dual-Audio Injection）

这是 Base LPM 最核心的架构创新，也是实现 full-duplex 生成的关键机制。

**问题**：说话音频（speak）和倾听音频（listen）是语义差异极大的两路信号：

- Speak：高频局部运动（口型），精确时序对应
- Listen：低频全局反应（点头/眼神），语义感知，响应时间尺度更长

如果把两路信号同时注入所有层：(1) cross-attention 参数量加倍；(2) 两种信号在同一表示子空间中互相干扰，梯度相互冲突。

**解决方案**：偶数层注入 speak，奇数层注入 listen。

```python
# 偶数层：speak audio cross-attention
K_s = W_k_spk @ c_speak
V_s = W_v_spk @ c_speak

# 奇数层：listen audio cross-attention  
K_l = W_k_lis @ c_listen
V_l = W_v_lis @ c_listen

# 文本与音频的最终输出合并（独立投影）
out = W_o_txt @ A_text + W_o_aud @ A_audio
```

**时域对齐窗口**的差异化设计：

- Speak branch：**局部窗口**（每个视频 token 只关注其时间邻域内的音频帧）→ 精确口型对齐
- Listen branch：**大窗口**（更长的音频上下文）→ 语义理解后的延迟反应

**优势总结**：

| 优势 | 说明 |
|---|---|
| **运动分布对齐** | 相邻层的 self-attn 和 FFN 参数自然形成隐式 "speak-tuned" 和 "listen-tuned" 子网络 |
| **梯度解耦** | 说话和倾听的 loss landscape 各自独立，无梯度干扰 |
| **参数效率** | 每路音频只注入一半的层，相比全层注入节省 50% audio cross-attn 参数和对应 FLOPs |

### 3.3 多参考图身份注入（Identity Multi-reference Conditioning）

**注入方式**：将参考图 token 直接**拼接进 self-attention 的 token 序列**，而不是通过额外的 cross-attention 分支。这是一个参数零增加的设计——身份注入完全复用现有 self-attn 权重。

**位置区分机制**：用 segment-wise 3D RoPE 赋予不同类型参考图不同的时间偏移：

$$\text{RoPE}_{ij} = \text{RoPE}(t + o_i + so_j, h, w)$$

- $i$：参考类型（expression 或 multi-view）
- $j$：子类型（具体的表情类别，如 happy/sad 等）
- $o_i$：类型级 base offset
- $so_j$：子类型级 sub offset

这让模型隐式学习 RoPE 位置作为一种弱条件，编码参考身份信息，无需引入额外的可学习 embedding 来区分参考类型。

**实际配置**：支持 1–8 张表情参考 + 1–4 张体视角参考，任意组合，支持缺失（即 inference 时不提供某类参考）。

**能力来源**：

- 多视角体参考 → 几何先验，头部转动/侧面镜头时不需要猜测
- 多表情参考 → 表情变形空间的约束，同一人的笑容始终一致
- 参考 token 在整个去噪过程中持续存在 → 跨滑动窗口的身份锚点，防止漂移

### 3.4 Base LPM 的训练策略

**初始化与多阶段课程**：

以 Wan2.1-I2V（16B）为起点，移除原始的 CLIP image cross-attention block 和 channel-mask-image in self-attention，进行简化。

```
阶段1：Speak 音频对齐
  - 仅在 speaking-only + silence 数据上训练 speak audio pathway
  - audio cross-attn 的 value projection 零初始化（稳定适配，不破坏视觉生成能力）
  - 预训练文本和 self-attn 权重大体保持

阶段2：Listen 音频对齐
  - 从零初始化 listen audio pathway
  - 使用 speak + listen 数据的 balanced mixture
  - 逐步混入 silence, listening, conversation 数据

阶段3：身份一致性训练
  - 所有 clips 提供全局外观参考
  - 30% 的 clips 加入表情参考和多视角体参考（稀缺数据，混合训练）

贯穿全程：对文本和音频条件均使用 CFG（Classifier-Free Guidance）
```

**长视频推理适配**：

训练时引入两个特殊措施以应对 chunk-wise 自回归推理：

1. 将全局参考图的 clean latent 拼接进 DiT block 计算，在训练和推理时都提供稳定的全局锚点
2. 以一定概率随机 drop 前 2–5 个 ground-truth video latent，迫使模型适应以自回归生成的 causal latent 开头的续生序列（消除 training-inference mismatch）

**DPO 后训练**：

Flow Matching 的重建损失对以下感知质量维度不敏感：手部/肢体变形、物理合理性、倾听行为的自然性（静止 vs 活跃）。用 DPO 进行后对齐：

- **说话任务目标**：减少手部/肢体伪像，提升大幅度运动时的物理合理性，维持精确口型
- **倾听任务目标**：增强非语言行为的自然性和多样性，避免输出近乎冻结的静止帧

**偏好对数据构建**：同一条件下用不同 noise seed 生成多个候选，Pareto 效率选择——候选 A 被选为 "preferred" 当且仅当它在至少一个维度上严格优于所有对手，且在任何维度上都不差于对手。同时加入 Flow Matching 正则化项防止 DPO 优化过程中的颜色偏移。

---

## 四、Online LPM：从离线 Bidirectional 到实时 Causal

### 4.1 核心挑战：自回归漂移

离线生成可以全局编码音频和文本，在完整上下文下去噪；在线生成只有截断的历史上下文，且每一步的生成结果成为下一步的条件。两个耦合问题：

- **Train-Inference Mismatch**：流式控制信号与离线训练的条件分布不同
- **Causal Rollout Exposure Bias**：每步微小误差积累，逐渐偏离 teacher 的有效轨迹

### 4.2 流式音频处理

音频是最敏感的流式信号（dense frame-level 特征，对截断最不鲁棒）：

采用**重叠感知 chunk-wise 音频编码**：

```
3s 窗口 = 2s 历史音频 + 1s 当前音频，步长 1s
```

重叠区域提供跨更新的时序连续性，抑制 boundary artifacts，同时保持每步延迟有界。在 600K 流式格式样本上微调进一步提升在线稳定性。

文本条件鲁棒性更高（更低密度），无需额外微调，直接增量更新。

### 4.3 Backbone-Refiner 架构：序列松弛

**核心方法论**：把 online 生成分解为两个子问题：

```
轨迹稳定（trajectory stabilization）
  → Backbone（2-step）
  → 基于 noisy-history KV cache
  → 容忍并适应自身历史的误差分布
  → 目标：维持在 teacher 轨迹的有效邻域内

              +

细节恢复（detail reconstruction）
  → Refiner（1-step）
  → 基于 clean-history KV cache
  → 在稳定轨迹上恢复高频细节
  → 目标：重建清晰的视觉质量
```

**Noisy vs Clean history 的设计意图**：

Backbone 用 noisy history KV cache（历史 latent 加噪后的 forward pass 结果），这让 Backbone 的训练分布更接近它在 inference 时实际看到的东西（自生成的有误差的历史），提升鲁棒性。

Refiner 用 clean history KV cache（clean 历史的 forward pass 结果），因为轨迹已被 Backbone 稳定后，提供更强的上下文有助于恢复精细细节。

两者都继承 Base LPM 的 DiT 架构，但将 bidirectional attention 替换为 **chunk-wise causal attention**：token 只关注当前 chunk、之前的视频 chunk、以及参考图 latent token。

### 4.4 四阶段 DMD 课程训练

在 DMD（Distribution Matching Distillation）框架下逐步收紧训练分布：

**Stage 1：ODE-based Initialization（监督热身）**

```python
# 从预训练 teacher 运行 ODE 轨迹，在 {T0, T1, T2, 0} 几个时间步采样
# 监督回归到 clean target
L_reg = E_{i,t} || G_backbone(x_t_i, t) - x_0_i ||_2^2
```

目的：让 Backbone 先学会 online chunk-wise 去噪调度，获得稳定的初始化，避免直接 DMD 时的不稳定。

**Stage 2：Off-policy DMD（Teacher 分布匹配）**

```python
# 从 ODE 轨迹数据集取 clean teacher latent，重加噪构成输入
# DMD 目标 + LPIPS 感知正则（防止 mode collapse）
L_off = E[L_DMD(G_backbone(x_hat_t_i, t)) + w * L_LPIPS(G_backbone(x_hat_t_i, t), x_0_i)]
```

训练状态来自 teacher 分布（off-policy），比自身 rollout 更稳定，作为 Stage 1 到 on-policy 的过渡桥接。

**Stage 3：On-policy DMD（自身 Rollout 分布匹配）**

```python
# 从 Backbone 自身的自回归 rollout 采样，重加噪构成输入
L_on = E[L_DMD(G_backbone(x_bar_t_i, t))]
```

这是消除 exposure bias 的关键步骤。训练分布 = 推理分布，Backbone 学会从自己的错误历史中恢复。

**Stage 4：Refinement DMD（Refiner 精修）**

```python
# Backbone rollout 的结果重加噪到 T2，作为 Refiner 输入
# DMD 相对 ODE 轨迹的 clean target 做监督
L_refiner = E[L_DMD(G_refiner(x_bar_T2_i, T2))]
```

Refiner 专门在 backbone-generated 输出（真实 online 误差分布）上训练，学习修正残余伪像并恢复高频细节。

**On-policy 的定义**：训练输入分布 $q$ = 当前 Backbone rollout 分布 $p_\theta$ 时为 on-policy；$q \neq p_\theta$（来自 teacher）时为 off-policy。这个区分贯穿了整个训练课程的设计逻辑。

### 4.5 Online 推理：长时稳定的实现

**滑动窗口解码**：

不在每步保留完整历史，而是维护一个固定大小的时间窗口：当前 chunk + 有限个历史 chunk + 参考图 latent token。每步延迟保持稳定，不随生成长度增加。

**Pre-RoPE KV Caching**：

在 RoPE 变换**之前**缓存历史 KV 状态。每个新滑动窗口到来时，动态对缓存的 KV 施加更新后的相对位置 RoPE，避免重计算全历史 transformer 激活，同时保持位置一致性。

**Attention Sink Tokens**（参考 StreamingLLM）：

在每个滑动窗口中保留少量 sink tokens 作为持久注意力锚点，跨窗口位移时提供稳定的注意力通路，减少长序列中的时序抖动和漂移伪像。

**上下文缓存结构**：

```
[固定 sink 3 chunks] + [滑动窗口 2 chunks] + [当前 chunk]
= 共 5 chunks 的有效上下文
```

这个固定大小的缓存确保了常数级内存占用，同时提供足够的身份锚定（sink）和时序连续性（sliding window）。

### 4.6 推理 Pipeline 的延迟分解

系统采用三阶段流水线并行：

```
Chunk n-1:  [Generator done] → [Refiner] → [VAE decode] → [Output]
Chunk n  :  [Generator     ]               [Refiner running in parallel]
```

实际延迟（单 GPU）：

| 组件 | 延迟 |
|---|---|
| Generator（Backbone, 2-step） | ~700 ms |
| Refiner（1-step） | ~700 ms |
| VAE decode | ~180 ms |
| Text/Audio encoders | 可忽略 |

生成单元为 1 秒 chunk（24 fps），Generator 处理当前 chunk 时 Refiner 精修上一 chunk，VAE 解码再上一 chunk，实现实时流式输出。

**系统级优化**：

- **State Splitting**：持久视觉状态与可刷新条件 cache 分离，对新输入响应快速且不打断视觉连续性
- **Boundary-Aligned Updates**：条件更新仅在 chunk 边界生效，每个 chunk 在固定条件下完成生成
- **Controlled Lookahead**：调度器控制提前量不过多，减少中断时的积压延迟

---

## 五、评估体系：LPM-Bench

### 5.1 基准设计

1000 个测试用例，每个提供：高分辨率初始帧、多参考图、结构化文本指令、配对说话/倾听音频。

| 场景 | 样本数 | 覆盖内容 |
|---|---|---|
| Speaking | ~400 | 78 类情绪，22 类表情基，合唱类型，双语发音，全身运动 |
| Listening | ~200 | 不同关系/人格/语音内容/情感背景/双语 |
| Conversation | ~200 | 多轮对话，从单轮到多轮，说/听自然切换 |
| Diverse Human Motion | ~100 | 对话之外的人体动作、物体交互、物理动作 |
| Character Generalization | ~100 | 写实/动画/3D渲染/艺术风格角色 |

音频测试集：5 秒–180 秒为主，另有 10% 的长形内容（3 分钟–1 小时）专门测试时序一致性。

### 5.2 评估维度

| 维度 | 说明 |
|---|---|
| **Motion Dynamics** | 时序连贯性与物理合理性，惩罚变形/闪烁/体部消失/近静止输出 |
| **Identity Consistency** | 面部和体部属性在全程生成中是否与参考图保持感知一致 |
| **Text Controllability** | 视频是否遵从文本指令（指定动作/视线/表情/情绪/时序） |
| **Audio-Video Synchronization** | 说话→口型精度；倾听→抑制假口型+视觉反应与音频语义对齐；对话→切换自然性 |

评估协议：GSB 框架（每对3独立评审员，多数投票）+ 1–5 Likert 绝对评分。

### 5.3 主要定量结果

**Base LPM (720P) 对比结果**：

| 对比方 | Overall 偏好 | 最大优势维度 |
|---|---|---|
| vs Kling-Avatar-2 | **64.3%** preferred | Identity Consistency（58.5%相对差距最大） |
| vs OmniHuman-1.5 | **42.5%** preferred | Identity Consistency (58.5%) + Text Ctrl (55.7%) |
| vs Wan2.1-I2V | Motion: 81.7%, ID: 88.3% | 全面碾压 |

**Base LPM 绝对分（Likert 1-5）**：

| 场景 | Motion Dyn. | ID Consist. | Text Ctrl. | A-V Sync. |
|---|---|---|---|---|
| Speak | 3.96 | 3.83 | 3.70 | **4.13** |
| Listen | 3.90 | **4.62** | **4.50** | **5.00** |
| Conversation | 3.24 | 3.90 | 4.32 | 3.34 |

Listen 任务的音视频同步达到满分 5.00，倾听身份保持 4.62，是表现最好的场景。Conversation 的 motion dynamics 3.24 是最大瓶颈（长序列中手部关节退化），对话切换打断了 AV sync（3.34）。

**Online LPM (480P) 对比结果**：

| 对比方 | Overall 偏好 | 弱势维度 |
|---|---|---|
| vs LiveAvatar | **82.5%** preferred | — |
| vs SoulX | **64.1%** preferred | Identity Consistency（SoulX 67.3% preferred） |

Online vs SoulX 的 Identity Consistency 逆转（SoulX 67.3% vs LPM 7.5%）揭示了一个权衡：SoulX 通过保守的近正面、低幅度生成来维持面部稳定，在身份一致性指标上占优，但失去了行为真实性。Overall 维度（"哪个看起来更像真人"）中评审员更倾向 Online LPM，表明行为真实性在整体感知中权重更高。

---

## 六、结语与批判性思考

### 这篇论文做得好的地方

**系统级视角**是最大的贡献。把 Performance Trilemma 明确化之后，整篇论文的每个技术选择都能被归因到三角的哪条边上，这让论文的逻辑链条异常清晰——从数据、到 Base Model、到 Online Model，每一层都在回答同一个系统级问题的某个子问题，而不是各自独立地"做了个有趣的 ablation"。

**数据工程的诚实**也值得尊重。三状态标注、多粒度参考图、倾听数据的再平衡，这些都是真实有用但不够"sexy"的工作。大多数论文会把数据部分压缩到一段，LPM 用了整整两章。事后看，这些数据工程上的决策直接决定了系统能不能完成任务——没有真实的 listening data，倾听生成能力就是无源之水。

**Online 蒸馏的问题拆解清晰**。把 online few-step generation 建模为 "sparse-target hitting over autoregressive latent trajectories" 这个视角很准确，Backbone-Refiner 的序列松弛分解也有充分的直觉动机。4阶段课程中 off-policy → on-policy 的渐进过渡设计得当，解决了直接 on-policy 训练的不稳定问题。

### 值得商榷的地方

**"Performance Trilemma" 这个名字的营销成分**不可忽视。说现有方法"无法同时满足三者"，很大程度上是在用自己的评估框架（LPM-Bench）定义问题，然后证明自己在这个框架下更优。Kling-Avatar-2 和 OmniHuman-1.5 在它们自己的评测框架下可能并不认为自己有 trilemma 问题。这种"先定义问题框架再解决"的策略是学术写作的常规操作，但读者要有意识地区分框架本身的合理性与结果的有效性。

**Conversation 的 AV-sync 3.34 是明显短板**。说/听切换的自然性是 full-duplex 对话的核心能力，但这恰恰是分数最低的维度之一，且 78% 的 text controllability 失败集中在多段落动作序列。论文对这些失败模式的分析比较浅——为什么多段落动作序列会失败？是 chunk-level text prompt 的时序对齐问题，还是 DiT 在长文本上的 attention 退化？这些问题没有得到深入回答。

**Listen 任务的 Online vs Base 质量差是一个结构性问题**（motion dynamics 40.0% Base preferred vs 12.0% Online preferred）。Online 模型的时序正则化（noisy-history conditioning + causal masking）倾向于抑制低幅度的缓慢运动，而倾听行为恰好以低幅度缓慢运动为主。这个 trade-off 可能不是超参调整能解决的，需要在架构层面对倾听行为做专门处理（比如独立的 temporal regularization 强度）。

**安全性章节有"checkbox"的嫌疑**。不可见水印、检测模型、分级访问控制——这些措施在面对真实的恶意使用场景时有多大实际效果，论文并没有给出量化论据。这不是 LPM 独有的问题，但对于一个生成如此高真实感人物视频的系统，安全性分析的深度与技术部分明显不匹配。

**数据成本和可复现性**是这类工作的隐性问题。95 小时的人工帧级标注、15K 对话理解标注、20K ASD 标注——这些加起来对大多数学术团队来说几乎不可能独立复现。论文描述了方法，但对于需要类似能力的研究者来说，"我知道他们怎么做的，但我做不了"的局面并没有改变。

### 最后

LPM 1.0 是一篇值得认真读的工程论文。它解决了一个真实存在的问题，有清晰的系统级视角，数据和训练设计都有充分的动机解释。其最大的贡献不是某个单一技术点，而是证明了：**通过系统层面的协同设计——数据覆盖、多模态条件化、在线稳定化——full-duplex 对话角色生成在可部署延迟下是可行的**。这个系统级的存在性证明比任何单一模块的改进都更有价值。
