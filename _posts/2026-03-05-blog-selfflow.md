---
title: 'Self-Flow：自监督 Flow Matching 实现可扩展的多模态生成'
date: 2026-03-05
permalink: /posts/2026-03-05-blog-selfflow/
tags:
#   - FlowMatching, multimode generation
---

# Self-Flow：自监督 Flow Matching 实现可扩展的多模态生成

> 论文：*Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*
> 作者：Hila Chefer, Patrick Esser 等（Black Forest Labs & MIT）
> 发布时间：2026 年 3 月 4 日
> 项目主页：https://bfl.ai/research/self-flow

---

## 一、问题背景：Flow Matching 的表征困境

现代生成模型（如 SiT、FLUX）基于 flow matching 训练，核心目标是预测从噪声到数据的速度场——本质上是一个去噪任务。这个目标虽然足以生成高质量样本，但存在一个根本性缺陷：**模型没有动力学习强语义表征**。去噪任务可以通过局部像素相关性完成，模型无需真正理解图像的全局语义结构。

这一缺陷的直接后果是，一个 86M 参数的 DINO 判别式编码器就能显著提升数十亿参数级别生成模型的性能——这暴露了生成模型内部表征的薄弱。

### 1.1 现有方案：外部表征对齐（REPA）

为弥补这一缺陷，目前主流方法是将生成模型的中间层特征与外部预训练编码器对齐。代表方法 REPA（Representation Alignment）的工作流程如下：

**训练阶段**：在标准 flow matching 损失之上，加一个辅助的特征对齐损失。取生成模型第 $$l$$ 层的中间特征，通过一个 MLP 投影头映射后，与冻结的外部编码器（如 DINOv2-B）的第 $$k$$ 层特征计算相似度。两个损失联合优化，只更新生成模型参数，外部编码器始终冻结。

**推理阶段**：外部编码器和投影头完全丢弃，推理过程与标准 flow matching 完全一致。外部编码器仅在训练阶段充当辅助监督信号。

可以将 REPA 理解为一种知识蒸馏：DINOv2 是"老师"，不教学生生成图片，而是教学生理解图片——迫使生成模型在去噪过程中构建出与语义编码器相似的内部表示。

### 1.2 外部对齐的三大根本缺陷

尽管 REPA 在 ImageNet 上效果显著，但 Self-Flow 的作者揭示了其三个根本性问题：

**缩放行为异常。** 使用更强的外部编码器反而导致更差的结果。实验中将 REPA 的编码器从 DINOv2-B 逐步升级到 DINOv2-L、DINOv3-B、DINOv3-H+，FID 指标呈现反直觉的逆相关——最弱的 DINOv2-B 反而效果最好。这说明外部对齐会产生瓶颈，生成模型被绑定在固定的外部表征上，无法充分利用更强编码器的优势。

**跨模态泛化差。** 对视频和音频生成，大多数外部编码器的对齐反而损害性能。视频专用的 V-JEPA 2 和 Depth Anything 3 对齐后的 FVD 甚至不如不做任何对齐的 vanilla flow matching。音频编码器 MERT 的对齐同样无益。

**编码器选择不可预测。** SigLIP 2 拥有文本监督和多宽高比支持，理论上更适合文本到图像任务，但实际表现却不如 DINOv2。选择哪个编码器进行对齐变成了一个充满不确定性的工程问题。

---

## 二、Self-Flow 方法详解

Self-Flow 的核心思想是：不借用外部表征，而是在 flow matching 框架内部构建自监督信号，让模型自己学出强语义表征。

### 2.1 核心机制：Dual-Timestep Scheduling（双时间步调度）

标准 flow matching 对所有 token 施加相同噪声水平，模型可以仅依赖局部关联完成去噪。要让模型学习全局语义关系，需要引入信息不对称——让一些 token 比其他 token 更"干净"，迫使模型利用干净 token 推断噪声 token。

论文对比了三种引入异质噪声的策略：

| 策略 | 做法 | 问题 |
|------|------|------|
| Full Masking | 随机将部分 token 设为纯噪声 ($$t=1$$) | 训练时常见"部分纯噪声+部分有信息"的组合，推理时却是均匀噪声，产生严重训推差距 |
| Diffusion Forcing | 每个 token 独立采样不同时间步 | 同样的训推差距问题 |
| **Dual-Timestep** | 只用两个时间步，均从同一噪声分布采样 | 保持每个 token 的边际噪声分布不变，避免训推差距 |

实验证实，Full Masking 和 Diffusion Forcing 严重损害生成质量（FID 大幅上升），而 Dual-Timestep Scheduling 即使不加自监督损失也能略微改善性能。

Dual-Timestep Scheduling 的具体操作：

1. 采样两个时间步 $$t, s \sim p(t)$$
2. 随机选取一个 token 子集 $$M$$（比例 $$\mathcal{R}_M \leq 0.5$$），$$M$$ 内的 token 使用时间步 $$s$$，其余使用 $$t$$
3. 构造异质噪声输入 $$\mathbf{x}_\tau$$：每个 token 按各自的时间步加噪

这样输入中同时存在不同噪声水平的 token，形成信息不对称。关键在于每个 token 的边际噪声分布与标准 flow matching 一致，因此推理时使用均匀噪声不会产生训推差距。

### 2.2 自监督表征学习框架

在信息不对称的基础上，Self-Flow 构建了 Student-Teacher 框架来鼓励模型学习强表征：

**Student 网络 $$f_\theta$$（主模型）**：接收异质噪声输入 $$\mathbf{x}_\tau$$，其中不同 token 有不同噪声水平。

**Teacher 网络 $$f_{\theta'}$$（EMA 副本）**：接收更干净的输入 $$\mathbf{x}_{\tau_{\min}}$$，所有 token 统一使用两个时间步中较小的那个 $$\tau_{\min} = \min(t, s)$$ 加噪。

Teacher 始终看到比 Student 更干净（或同样干净）的输入，形成信息优势。

### 2.3 训练过程

每一步训练执行以下操作：

**第一步：构造两个输入。** 给定干净数据 $$\mathbf{x}_0$$ 和噪声 $$\mathbf{x}_1$$，构造 Student 的异质噪声输入 $$\mathbf{x}_\tau$$ 和 Teacher 的均匀干净输入 $$\mathbf{x}_{\tau_{\min}}$$。

**第二步：两次前向传播。** 将 $$\mathbf{x}_\tau$$ 送入 Student，将 $$\mathbf{x}_{\tau_{\min}}$$ 送入 Teacher（不计算梯度）。

**第三步：计算生成损失。** 标准 flow matching 损失，注意每个 token 的 target 对应其自身的时间步：
$$\mathcal{L}_{\text{gen}} = \mathbb{E}\|f_\theta(\mathbf{x}_\tau, \boldsymbol{\tau}) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2$$

模型的时间步条件从标量扩展为向量，每个 token 被告知自己的噪声水平。

**第四步：计算表征对齐损失。** 取 Student 第 $$l$$ 层特征，通过 MLP 投影头映射后，与 Teacher 第 $$k$$ 层特征（$$k > l$$）计算余弦相似度：
$$\mathcal{L}_{\text{rep}} = -\cos\left(h_\theta^{(l)}(\mathbf{x}_\tau, \boldsymbol{\tau}),\ f_{\theta'}^{(k)}(\mathbf{x}_{\tau_{\min}}, \tau_{\min})\right)$$

Student 只看到部分被严重破坏的输入，却要预测出 Teacher 在更干净输入下产生的特征。这迫使 Student 利用较干净的 token 推断较噪 token 的语义信息，建立超越局部的全局关联。

**第五步：联合优化。** 总损失为：
$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \gamma \cdot \mathcal{L}_{\text{rep}}$$

只更新 Student 和投影头参数，Teacher 通过 EMA 缓慢跟踪 Student：$$\theta' \leftarrow 0.9999 \cdot \theta' + 0.0001 \cdot \theta$$。

### 2.4 推理过程

推理时与标准 flow matching **完全一致**：

1. 丢弃 Teacher 网络和投影头
2. 所有 token 使用统一时间步（无双时间步或 mask）
3. 从纯噪声 $$\mathbf{x}_1 \sim \mathcal{N}(0, \mathbf{I})$$ 出发
4. 用 Student $$f_\theta$$ 预测速度场，通过 ODE 求解器从 $$t=1$$ 积分到 $$t=0$$

零额外推理开销。训练阶段的所有辅助组件都不出现在推理中。

### 2.5 Self-Flow 与 REPA 的关键区别

| 维度 | REPA | Self-Flow |
|------|------|-----------|
| 表征来源 | 冻结的外部编码器（DINOv2） | 模型自身的 EMA 副本 |
| 信息不对称来源 | 外部编码器天然拥有更好的语义表征 | Dual-Timestep Scheduling 制造的噪声差异 |
| 跨模态泛化 | 每个模态需选择合适编码器，常常失效 | 天然适用于任何模态 |
| 缩放行为 | 更强编码器反而可能更差 | 遵循正常缩放规律 |
| 推理开销 | 无额外开销 | 同样无额外开销 |

更宏观地看，可以将三种方法理解为一条谱系上的三个位置：Vanilla Flow Matching（不管表征）→ REPA（借外部表征）→ Self-Flow（自己生长出表征）。

---

## 三、统一的多模态架构

Self-Flow 的一大亮点是**同一个 Transformer 可以同时处理图像、视频和音频三种模态**。

### 3.1 设计原理

核心思路是：不同模态的原始数据先通过各自的自编码器变成统一的"token 序列"形式，Transformer 处理的只是这些 token 序列。对 Transformer 来说，它看到的就是一串向量，不关心这些向量原本代表的是图像 patch、视频帧还是音频片段。

各模态的 token 化过程如下：

| 模态 | 自编码器 | 输入 | Token 数量 | Token 维度 |
|------|---------|------|-----------|-----------|
| 图像 (256×256) | FLUX.2 AE | 原始像素 | 256 | 128 |
| 视频 (45帧, 192p) | WAN2.2 AE | 原始视频帧 | ~3000 | 48 |
| 音频 (10秒) | Songbloom AE | 原始音频波形 | 250 | 64 |

不同模态编码出来的 token 维度不同、序列长度不同，通过**模态特定的投影层**统一到 Transformer 的隐藏维度。

### 3.2 模型结构

```
文本提示 ──────────────────────────────────────────┐
                                                   │
纯噪声 → 模态特定输入投影层 → 共享 Transformer → 模态特定输出投影层 → 模态特定解码器 → 输出
         (三套，按模态选一套)     (完全共享)      (三套，按模态选一套)    (三套，按模态选一套)
```

**共享部分**包括整个 Transformer 的 attention 层、FFN 层、modulation 层等（占绝大部分参数）。**模态独立部分**仅有每个模态的输入/输出投影层，负责维度转换。

Transformer 架构基于 FLUX，配置为隐藏维度 1152、MLP ratio 4、16 个注意力头、7 层双流 MMBlock + 14 层单流 Block，总参数量约 625M。使用 3D RoPE 位置编码，天然支持不同维度的序列。

### 3.3 混合模态训练

训练时每个 mini-batch 只包含单一模态的数据，不在一个 batch 中混合不同模态。模态之间通过采样概率交替训练（图像 57%、视频 30%、音频 13%），通过模态损失权重 $$(w_I, w_V, w_A)$$ 控制各模态优先级。

每步训练流程为：按概率选择一个模态 → 取该模态的一个 batch → 通过相应自编码器编码 → 通过对应输入投影层映射到 1152 维 → 送入共享 Transformer 执行 Self-Flow 训练 → 通过对应输出投影层映射回模态维度 → 计算损失并更新共享 Transformer 和对应投影层参数。

### 3.4 推理时的模态选择

推理时根据目标模态选择对应的投影层和解码器即可。生成图像时使用图像投影层 + FLUX.2 AE 解码；生成视频时使用视频投影层 + WAN2.2 AE 解码；生成音频时使用音频投影层 + Songbloom AE 解码。模型不需要显式的"模态选择"机制，选择投影层本身就决定了输出模态。

### 3.5 联合多模态生成

除了分别生成单一模态，模型还能同时生成多种模态。例如联合视频-音频生成任务中，序列中包含视频帧 token 和音频 token，共享 Transformer 一起处理，输出后分别通过各自的投影层和解码器还原。

---

## 四、实验全景

### 4.1 实验概览与数据

| 任务 | 数据集 | 数据量 | 训练步数 | 骨干网络 |
|------|--------|--------|---------|---------|
| ImageNet (Class→Image) | ImageNet-1K | 128万图像 | 4M步 | SiT-XL (~675M) |
| 文本→图像 | 内部数据集 | 2000万图像 | 1M步 | FLUX.2 (~625M) |
| 文本→视频 | 内部数据集 | 600万视频 | 600K步 | FLUX.2 (~625M) |
| 文本→音频 | FMA (CC协议) | 100万音频 | 350K步 | FLUX.2 (~625M) |
| 混合多模态 | 上述三者组合 | 全量 | 1M步 | FLUX.2 (~625M) |
| 联合视频-动作 | RT-1 机器人数据 | 73,500 episodes | 100K步（微调） | FLUX.2 (~625M) |

所有单模态实验和多模态训练均从零开始。联合视频-动作实验从多模态模型微调。

### 4.2 单模态结果

**ImageNet (256×256)：** Self-Flow 达到 FID 5.70，优于 REPA 的 5.89。值得注意的是，REPA 使用的 DINOv2 在 ImageNet 上大量训练过，Self-Flow 在"客场"依然胜出。据作者所知，这是首次自监督方法在 ImageNet 上超越外部对齐方法。

**文本→图像：** Self-Flow 的 FID 为 3.61，超越 REPA（3.92）、SigLIP 2（3.97）和 SRA（3.70）。在以 DINOv2 特征计算的 FD-DINOv2 指标上，Self-Flow（167.98）甚至优于直接与 DINOv2 对齐的 REPA（173.35），同时 CLIP score 也最高。

**文本→视频：** Self-Flow 的 FVD 为 47.81，大幅领先次优的 REPA（49.59）。视频专用编码器 V-JEPA 2 和 Depth Anything 3 的对齐反而损害性能。论文推测视频时序关系比空间关系更难学习，目标不匹配更难弥合，且视频时序冗余让模型可以走捷径——而 Self-Flow 的 masking 机制自然抑制了这种行为。

**文本→音频：** Self-Flow 在所有 CLAP 变体的 FAD 指标上均最优，音频编码器 MERT 的对齐完全无益。

### 4.3 缩放实验

在 290M → 420M → 625M → 1B 四个模型规模上对比 Self-Flow 与 REPA，关键发现有两个：

- 随模型规模增大，Self-Flow 与 REPA 的性能差距**持续扩大**
- Self-Flow 的 625M 模型甚至优于 REPA 的 1B 模型

这直接验证了论文的核心论点：外部对齐将模型绑定在固定编码器上，形成缩放瓶颈；Self-Flow 的统一框架则遵循正常的缩放规律。

### 4.4 多模态实验

**混合模态训练：** 测试了 5 种不同的模态损失权重配置（从图像偏重到音频偏重），Self-Flow 在**所有配置下对所有三个模态同时带来提升**，没有出现任何模态退化的情况。

**联合视频-动作预测：** 在 SIMPLER 机器人模拟器上评估 4 类任务。Self-Flow 在整个微调过程中持续优于 vanilla flow matching。特别是对复杂的多物体和多步骤任务（Move Near、Open and Place），Self-Flow 保持显著优势，而简单任务（Pick Coke Can、Open/Close Drawer）两者趋于收敛。这表明 Self-Flow 学到的表征能改善复杂视觉推理能力。

**联合视频-音频预测：** 多模态预训练初始化优于仅视频预训练。更有意思的是，仅视频初始化的 Self-Flow 甚至优于多模态初始化的 vanilla flow matching，进一步证明了 Self-Flow 学到的表征的通用性。

### 4.5 消融实验

在 ImageNet 上逐一移除或修改关键组件：

| 消融项 | FID 影响 | 结论 |
|--------|---------|------|
| 去掉 $$\mathcal{L}_{\text{rep}}$$ | 退化 4+ 分 | 自监督表征学习是最关键组件 |
| 去掉 masking 机制 | 退化 1+ 分 | 显式自监督范式不可或缺 |
| 限制第二时间步 $$s \in [t, t-0.2]$$ | 退化接近去掉 masking | 需要充分的信息不对称 |
| $$\ell_1$$ 替代余弦相似度 | 训练后期不稳定 | 余弦相似度更适合特征对齐 |

层选择方面，Student 层 $$l = 0.3D$$、Teacher 层 $$k = 0.7D$$ 附近性能稳定。过浅的 Teacher 层语义信号太弱，过深的 Student 层会干扰生成。

### 4.6 表征质量验证

线性探测实验直接验证了表征强度：在 ImageNet 上训练 2M 步后，逐层提取特征做线性分类。Self-Flow 在早期和中间层的分类准确率显著高于 vanilla flow matching，确认表征确实随生成能力一同增强。

### 4.7 自编码器泛化

Self-Flow 不依赖特定自编码器。在 SD-VAE、FLUX.2 AE、RAE（表征自编码器）、WAN2.2 AE、Songbloom AE 五种不同自编码器上均展现一致提升。特别是在 RAE 这种已有语义结构的潜空间上，FID 仍从 3.24 降至 2.95，说明 Self-Flow 与语义自编码器具有互补效应。

### 4.8 定性改善

Self-Flow 在以下方面展现显著的定性改善：

- **文字渲染**：4B 参数多模态模型在仅 100K 步高分辨率微调后，文字渲染准确度大幅提升
- **结构一致性**：人脸、手部等挑战性结构的生成更加准确
- **视频时序连贯性**：基线方法出现的肢体突然消失等时序伪影在 Self-Flow 中得到抑制

---

## 五、训练开销与局限

### 额外训练开销

Self-Flow 相比 vanilla flow matching 的额外成本来自 Teacher 网络的前向传播（每步需要两次前向）。但加速的收敛和更好的性能抵消了这一成本。从 FLOPs 对比来看，在相同计算量下 Self-Flow 仍优于 REPA。

### 噪声调度器敏感性

与相关工作一致，噪声调度器 $$p(t)$$ 的选择影响较大。在文本→图像任务中，均匀调度器优于 logit-normal 调度器。虽然更好的噪声调度对 REPA 和 Self-Flow 都有益，但 Self-Flow 从中获益更大，原因在于噪声调度同时决定了 masking 行为的优化。

---

## 六、总结与展望

Self-Flow 挑战了一个普遍假设：生成模型需要外部编码器才能获得强表征。通过 Dual-Timestep Scheduling 在 flow matching 内部构建自监督信号，Self-Flow 证明生成与表征学习可以在统一框架内互相增强，且遵循预期的缩放规律。

这项工作的意义超越了生成质量的提升。通过消除对外部编码器的依赖，Self-Flow 为构建真正统一的多模态世界模型开辟了道路——这些模型可以利用视觉生成模型的可扩展性和感知基础，同时不牺牲规划和理解所需的语义抽象能力。论文在机器人视频-动作预测中的初步验证已经展示了这一方向的潜力。