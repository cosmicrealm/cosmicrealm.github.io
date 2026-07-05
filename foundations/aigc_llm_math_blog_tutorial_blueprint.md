# AIGC 与 LLM 数学基础系统教程蓝图

> 用途：这份 Markdown 文档可以直接交给 Codex、Cursor、Claude Code 或其他代码/文档生成工具，让它们据此生成一套系统博客教程。  
> 目标读者：具备机器学习或算法背景，希望系统补齐 AIGC、LLM、Diffusion、RLHF、DPO 等方向所需数学基础的研究者或工程师。  
> 写作原则：每一个名词都要做到“先讲直觉，再给数学形式，再说明在模型中的作用，最后给一个最小例子”。

---

## 0. 如何使用这份文档

这份文档不是传统数学课的目录，而是从 AIGC/LLM 的实际模型机制倒推数学需求。推荐学习主线如下：

```text
张量线性代数
  → 矩阵微积分与自动微分
  → 概率统计与信息论
  → 优化理论与数值计算
  → 深度学习机制
  → Transformer 与 LLM
  → 生成模型：VAE / GAN / Diffusion / Flow Matching
  → RLHF / DPO / 偏好优化
  → 评测统计、泛化理论与研究方法
```

Codex 生成博客教程时，建议将本文档拆成一个系列：

```text
01_为什么学习_aigc_llm_数学.md
02_线性代数与张量计算.md
03_矩阵微积分与自动微分.md
04_概率统计与语言建模.md
05_信息论_cross_entropy_kl_perplexity.md
06_优化算法_sgd_adam_adamw.md
07_数值计算_混合精度_量化_flashattention.md
08_深度学习通用机制.md
09_transformer_attention_rope.md
10_llm_预训练与_next_token_prediction.md
11_lora_adapter_moe_高效微调.md
12_vae_elbo_变分推断.md
13_gan_对抗生成与分布匹配.md
14_diffusion_ddpm_score_sde.md
15_latent_diffusion_多模态生成.md
16_flow_matching_ode_optimal_transport.md
17_rlhf_ppo_偏好学习.md
18_dpo_及现代对齐方法.md
19_llm_评测统计与置信区间.md
20_研究路线_论文阅读与实践项目.md
```

每篇博客建议采用统一结构：

```text
1. 这个主题解决什么问题？
2. 为什么 AIGC/LLM 需要它？
3. 通俗解释
4. 数学定义
5. 关键公式推导
6. 在模型中的具体位置
7. 最小代码或伪代码
8. 常见误区
9. 练习题
10. 延伸阅读
```

---

## 1. 总体学习目标

学完这条路线后，应该能够：

1. 看懂 Transformer、Diffusion、RLHF、DPO 等主流论文中的核心公式。
2. 从零实现一个小型 decoder-only Transformer。
3. 理解 next-token prediction、cross-entropy、perplexity 与最大似然之间的关系。
4. 推导 attention、softmax、LayerNorm/RMSNorm、LoRA、DPO loss 等核心表达。
5. 理解 VAE、GAN、DDPM、Score-based Model、Latent Diffusion、Flow Matching 的概率建模思想。
6. 用统计方法评估模型结果，而不是只报告单一 benchmark 分数。
7. 根据研究方向选择进阶数学：随机过程、最优传输、随机矩阵、信息几何、强化学习、因果推断等。

---

## 2. 数学模块总览

| 数学模块 | 需要掌握到什么程度 | 在 AIGC/LLM 中的对应位置 |
|---|---|---|
| 线性代数与张量计算 | 向量、矩阵、张量、矩阵乘法、特征值、SVD、范数、投影、低秩分解、高维几何 | embedding、attention、LoRA、KV cache、MoE、表示空间分析 |
| 矩阵微积分与自动微分 | Jacobian、Hessian、链式法则、VJP、JVP、trace trick、softmax 梯度 | backprop、Transformer block、optimizer、adapter/LoRA 微调 |
| 概率论与统计推断 | 条件概率、期望、方差、MLE、MAP、Bayes、KL、采样、Monte Carlo、置信区间 | next-token modeling、sampling、perplexity、diffusion noising、模型评测 |
| 信息论 | entropy、cross-entropy、KL divergence、mutual information、bits/nats | 语言模型 loss、压缩视角、distillation、RLHF/DPO 中的 KL 正则 |
| 优化理论 | SGD、Momentum、Adam、AdamW、学习率调度、正则化、非凸优化、约束优化、对偶 | 预训练、SFT、LoRA、RLHF、DPO、scaling 实验 |
| 数值计算 | 浮点误差、conditioning、mixed precision、量化、矩阵乘法复杂度、memory bandwidth | FP16/BF16、INT8/INT4 quantization、FlashAttention、训练稳定性 |
| 统计学习理论 | bias-variance、泛化误差、overfitting、Rademacher、VC、PAC-Bayes、分布偏移 | scaling laws、benchmark 置信度、ablation、OOD/generalization |
| 随机过程/ODE/SDE | Markov chain、Gaussian transition、Brownian motion、reverse SDE、ODE solver | DDPM、score-based diffusion、flow matching、图像/视频生成 |
| 强化学习与偏好学习 | MDP、policy gradient、advantage、PPO、KL-constrained optimization、Bradley–Terry 模型 | RLHF、RLAIF、DPO、GRPO、agent 训练 |

---

# Part I：线性代数与张量计算

## 3. 为什么线性代数是 LLM 的第一语言

LLM 中几乎所有核心计算都可以看成矩阵或张量运算：

- token 被映射为 embedding 向量；
- attention 用矩阵乘法计算 token 之间的相似度；
- MLP 用线性变换加非线性激活提取特征；
- LoRA 用低秩矩阵更新减少微调参数量；
- KV cache 保存历史 key/value 张量；
- MoE 用路由矩阵选择专家网络；
- 量化、压缩、蒸馏也大量依赖矩阵近似。

如果说程序员看到的是代码，模型看到的是向量空间。

---

## 4. 核心概念解释：线性代数

### 4.1 Scalar：标量

**通俗解释**：标量就是一个普通数字，比如温度、概率、loss 值。

**数学定义**：标量通常记作 \(a \in \mathbb{R}\)，表示实数域中的一个元素。

**在 LLM 中的作用**：

- learning rate 是标量；
- loss 是标量；
- attention score 中每两个 token 的相似度是标量；
- softmax 输出的每个概率也是标量。

**例子**：

```python
loss = 2.37
learning_rate = 3e-4
```

---

### 4.2 Vector：向量

**通俗解释**：向量是一串数字，可以表示一个对象在多个维度上的特征。

**数学定义**：

\[
\mathbf{x} = [x_1, x_2, \dots, x_d]^\top \in \mathbb{R}^d
\]

**在 LLM 中的作用**：一个 token 经过 embedding layer 后会变成一个向量。例如“猫”这个 token 可能被表示成 4096 维向量。

**直觉**：向量之间的方向相近，通常代表语义上更相近。

---

### 4.3 Matrix：矩阵

**通俗解释**：矩阵是二维数字表。它可以表示一组向量，也可以表示一种线性变换。

**数学定义**：

\[
A \in \mathbb{R}^{m \times n}
\]

表示一个有 \(m\) 行、\(n\) 列的矩阵。

**在 LLM 中的作用**：

- embedding table 是矩阵；
- attention 中的 \(W_Q, W_K, W_V\) 是矩阵；
- MLP 中的上投影、下投影都是矩阵；
- LoRA 的 \(A, B\) 也是矩阵。

**例子**：

如果输入 hidden state 为：

\[
X \in \mathbb{R}^{T \times d_{model}}
\]

查询矩阵为：

\[
W_Q \in \mathbb{R}^{d_{model} \times d_k}
\]

那么：

\[
Q = XW_Q \in \mathbb{R}^{T \times d_k}
\]

---

### 4.4 Tensor：张量

**通俗解释**：张量是多维数组。标量是 0 维张量，向量是 1 维张量，矩阵是 2 维张量，更高维的数据就是高阶张量。

**数学定义**：

\[
X \in \mathbb{R}^{B \times T \times d}
\]

可以表示 batch size 为 \(B\)、序列长度为 \(T\)、hidden dimension 为 \(d\) 的一批 token 表示。

**在 LLM 中的作用**：深度学习框架中的大部分数据都是张量。

常见 shape：

```text
input_ids:      [B, T]
embedding:      [B, T, d_model]
Q, K, V:        [B, H, T, d_head]
attention map:  [B, H, T, T]
logits:         [B, T, vocab_size]
```

---

### 4.5 Shape：形状

**通俗解释**：shape 描述张量每个维度有多大。

**在 LLM 中的作用**：理解 shape 是调试 Transformer 的基础。很多模型 bug 不是算法错，而是 shape 对不上。

**例子**：

假设：

```text
B = 2          # batch size
T = 5          # sequence length
d_model = 768  # hidden size
H = 12         # number of heads
d_head = 64    # 768 / 12
```

则 embedding shape 是：

```text
[B, T, d_model] = [2, 5, 768]
```

拆成多头后：

```text
[B, H, T, d_head] = [2, 12, 5, 64]
```

---

### 4.6 Dot Product：点积 / 内积

**通俗解释**：点积衡量两个向量方向是否相近。方向越相近，点积越大。

**数学定义**：

\[
\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^d x_i y_i
\]

**在 attention 中的作用**：attention 通过 \(QK^\top\) 计算每个 token 对其他 token 的相关性。

**直觉**：如果 query 向量和 key 向量方向接近，说明当前 token 应该更多关注那个 token。

---

### 4.7 Matrix Multiplication：矩阵乘法

**通俗解释**：矩阵乘法可以看成“批量做点积”，也可以看成“对一批向量做线性变换”。

**数学定义**：

如果：

\[
A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times p}
\]

则：

\[
AB \in \mathbb{R}^{m \times p}
\]

其中：

\[
(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
\]

**在 LLM 中的作用**：

- embedding lookup 后的投影；
- attention score：\(QK^\top\)；
- attention 加权求和：\(AV\)；
- feed-forward network：\(XW_1\)、\(XW_2\)；
- logits：\(XW_{vocab}\)。

---

### 4.8 Vector Space：向量空间

**通俗解释**：向量空间是所有可能向量组成的空间。模型内部的 hidden state 就生活在这样的空间里。

**数学定义**：一个集合如果对向量加法和标量乘法封闭，并满足若干代数性质，就叫向量空间。

**在 LLM 中的作用**：LLM 的语义、语法、事实、风格等信息都被编码到高维向量空间中。

**例子**：词向量中常见的类比现象：

```text
king - man + woman ≈ queen
```

这不是严格规则，但体现了语义可能以方向和子空间形式存在。

---

### 4.9 Basis：基

**通俗解释**：基是一组“坐标轴”。有了基，就可以用坐标表示空间中的任意向量。

**数学定义**：如果一组向量线性无关，并且能张成整个空间，它们就是一组基。

**在 LLM 中的作用**：embedding 维度可以理解为某种隐式坐标系。不过神经网络中的基通常没有人类可解释的含义。

---

### 4.10 Projection：投影

**通俗解释**：投影就是把一个向量“照到”某个方向或子空间上，看看它在那个方向上有多少成分。

**数学定义**：向量 \(x\) 在单位向量 \(u\) 上的投影为：

\[
\mathrm{proj}_u(x)= (x^\top u)u
\]

**在 LLM 中的作用**：

- attention 中 \(W_Q,W_K,W_V\) 可视作把 hidden state 投影到不同子空间；
- interpretability 中常用投影分析某个语义方向；
- LoRA 的低秩更新也可以理解为限制参数变化在低维子空间内。

---

### 4.11 Norm：范数

**通俗解释**：范数衡量向量或矩阵的大小。

**常见形式**：

\[
\|x\|_2 = \sqrt{\sum_i x_i^2}
\]

\[
\|x\|_1 = \sum_i |x_i|
\]

**在 LLM 中的作用**：

- gradient norm 用于判断梯度是否爆炸；
- weight decay 控制权重范数；
- normalization 层与向量尺度有关；
- embedding norm 会影响 logits 与 softmax 分布。

---

### 4.12 Cosine Similarity：余弦相似度

**通俗解释**：余弦相似度只关心两个向量方向是否接近，不太关心长度。

**数学定义**：

\[
\cos(\theta)=\frac{x^\top y}{\|x\|\|y\|}
\]

**在 AIGC/LLM 中的作用**：

- 文本 embedding 检索；
- RAG 相似度搜索；
- 表示空间分析；
- 聚类和近邻查询。

---

### 4.13 Eigenvalue / Eigenvector：特征值与特征向量

**通俗解释**：如果一个向量经过矩阵变换后方向不变，只是长度被缩放了，那么它就是这个矩阵的特征向量，缩放倍数就是特征值。

**数学定义**：

\[
Av = \lambda v
\]

其中 \(v\) 是特征向量，\(\lambda\) 是特征值。

**在 LLM 中的作用**：

- 分析权重矩阵的谱性质；
- 研究训练稳定性；
- 理解 Hessian 曲率；
- 分析表示空间的主方向。

---

### 4.14 SVD：奇异值分解

**通俗解释**：SVD 把一个矩阵拆成“旋转—缩放—旋转”的形式，可以看出矩阵最重要的方向。

**数学定义**：

\[
A = U\Sigma V^\top
\]

其中 \(\Sigma\) 对角线上的值叫奇异值。

**在 LLM 中的作用**：

- 低秩近似；
- 模型压缩；
- 权重分析；
- LoRA 的数学直觉；
- 表示空间主成分分析。

---

### 4.15 Rank：秩

**通俗解释**：秩表示矩阵中真正独立的信息维度。

**数学定义**：矩阵的秩是其列空间或行空间的维数。

**在 LoRA 中的作用**：LoRA 假设模型微调所需的权重变化可以近似为低秩矩阵：

\[
\Delta W = BA, \quad \mathrm{rank}(\Delta W) \le r
\]

其中 \(r\) 远小于原矩阵维度，所以参数量显著减少。

---

### 4.16 Low-rank Approximation：低秩近似

**通俗解释**：低秩近似就是用更少的独立方向近似原始矩阵，保留主要信息，丢掉次要信息。

**数学形式**：

\[
A \approx A_r = U_r\Sigma_rV_r^\top
\]

**在 AIGC/LLM 中的作用**：

- LoRA；
- 模型压缩；
- adapter 参数高效微调；
- 权重矩阵分析；
- 表示空间降维。

---

### 4.17 High-dimensional Geometry：高维几何

**通俗解释**：高维空间和二维、三维空间的直觉很不一样。例如高维空间中随机向量往往近似正交。

**在 LLM 中的作用**：

- embedding 生活在高维空间；
- attention 用高维向量内积衡量相关性；
- 大模型表示可能分布在复杂的高维流形上；
- 高维稀疏性、集中现象会影响模型行为。

---

# Part II：矩阵微积分与自动微分

## 5. 为什么需要矩阵微积分

训练神经网络本质上是在问：

> 参数变一点，loss 会怎么变？

这个“怎么变”就是梯度。LLM 有数十亿到数万亿参数，不可能手工对每个参数求导，因此需要把矩阵微积分、链式法则和自动微分结合起来。

---

## 6. 核心概念解释：微积分与自动微分

### 6.1 Derivative：导数

**通俗解释**：导数表示一个量变化时，另一个量变化得有多快。

**数学定义**：

\[
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
\]

**在模型训练中的作用**：导数告诉我们应该如何调整参数来降低 loss。

---

### 6.2 Gradient：梯度

**通俗解释**：梯度是多变量函数中“上升最快的方向”。如果要最小化 loss，就沿着负梯度方向走。

**数学定义**：

\[
\nabla_x f(x)=\left[\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},\dots,\frac{\partial f}{\partial x_d}\right]^\top
\]

**在 LLM 中的作用**：每次训练都会计算 loss 对参数的梯度，然后 optimizer 根据梯度更新参数。

---

### 6.3 Jacobian：雅可比矩阵

**通俗解释**：Jacobian 描述一个向量函数的每个输出对每个输入的敏感程度。

**数学定义**：如果 \(f: \mathbb{R}^n \to \mathbb{R}^m\)，则：

\[
J_{ij}=\frac{\partial f_i}{\partial x_j}
\]

**在 LLM 中的作用**：

- softmax 的导数是 Jacobian；
- attention 输出对输入的敏感度可由 Jacobian 描述；
- 分析模型局部行为、对抗扰动和 interpretability 时会用到。

---

### 6.4 Hessian：海森矩阵

**通俗解释**：Hessian 描述 loss 曲面的弯曲程度。

**数学定义**：

\[
H_{ij}=\frac{\partial^2 f}{\partial x_i \partial x_j}
\]

**在 LLM 中的作用**：

- 分析 loss landscape；
- 判断优化难度；
- 二阶优化方法；
- pruning、量化、sharpness 分析。

---

### 6.5 Chain Rule：链式法则

**通俗解释**：复杂函数由很多简单函数嵌套组成，链式法则告诉我们如何把每一层的导数乘起来。

**数学定义**：

如果：

\[
y=f(g(x))
\]

则：

\[
\frac{dy}{dx}=\frac{df}{dg}\frac{dg}{dx}
\]

**在神经网络中的作用**：反向传播就是链式法则在计算图上的系统应用。

---

### 6.6 Computational Graph：计算图

**通俗解释**：计算图把一次模型前向计算拆成节点和边。节点是操作，边是数据流。

**在 LLM 中的作用**：深度学习框架会记录计算图，然后自动做反向传播。

**例子**：

```text
input → embedding → attention → MLP → logits → loss
```

反向传播方向相反：

```text
loss → logits → MLP → attention → embedding → parameters
```

---

### 6.7 Backpropagation：反向传播

**通俗解释**：反向传播把 loss 的责任从输出层一层层分配回所有参数。

**数学本质**：链式法则 + 动态规划。

**在 LLM 中的作用**：训练时计算所有参数的梯度。

---

### 6.8 Automatic Differentiation：自动微分

**通俗解释**：自动微分不是数值差分，也不是符号求导，而是把每个基本操作的精确导数按链式法则组合起来。

**两种常见模式**：

- forward-mode：适合输入维度少、输出维度多；
- reverse-mode：适合输入维度多、输出是标量 loss。神经网络训练主要用 reverse-mode。

---

### 6.9 VJP：Vector-Jacobian Product

**通俗解释**：VJP 不是显式构造巨大的 Jacobian，而是直接计算一个向量乘 Jacobian 的结果。

**数学形式**：

\[
v^\top J
\]

**在深度学习中的作用**：反向传播主要计算 VJP，因为完整 Jacobian 太大，显式存储不可行。

---

### 6.10 JVP：Jacobian-Vector Product

**通俗解释**：JVP 计算 Jacobian 乘一个向量。

**数学形式**：

\[
Jv
\]

**用途**：

- forward-mode autodiff；
- 二阶优化；
- influence function；
- 一些高效敏感性分析。

---

### 6.11 Trace Trick：迹技巧

**通俗解释**：trace trick 是矩阵求导里常用的整理技巧，可以把标量写成 trace 形式，让求导更容易。

**常见恒等式**：

\[
x^\top Ay = \mathrm{tr}(x^\top Ay)=\mathrm{tr}(yx^\top A)
\]

**在深度学习中的作用**：推导线性层、attention、矩阵范数、低秩分解等梯度时非常有用。

---

### 6.12 Softmax Gradient：Softmax 梯度

**softmax 定义**：

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

**Jacobian**：

\[
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j)
\]

其中 \(\delta_{ij}\) 是 Kronecker delta，\(i=j\) 时为 1，否则为 0。

**和交叉熵结合时的经典结果**：

如果：

\[
L=-\sum_i y_i\log p_i
\]

则：

\[
\frac{\partial L}{\partial z_i}=p_i-y_i
\]

这个公式是语言模型训练中最重要的梯度公式之一。

---

# Part III：概率统计与信息论

## 7. 为什么语言模型首先是概率模型

LLM 的核心任务不是“直接输出一个确定答案”，而是对下一个 token 的概率分布建模：

\[
p_\theta(x_t|x_{<t})
\]

也就是说，给定前文，模型预测下一个 token 是每个词的概率。

---

## 8. 核心概念解释：概率统计

### 8.1 Random Variable：随机变量

**通俗解释**：随机变量是一个结果不确定的量。

**数学定义**：随机变量是从样本空间到数值空间的函数。

**在 LLM 中的作用**：下一个 token 可以看成一个离散随机变量。

---

### 8.2 Probability Distribution：概率分布

**通俗解释**：概率分布告诉我们每个结果发生的可能性。

**离散分布例子**：

```text
P(token = “猫”) = 0.30
P(token = “狗”) = 0.20
P(token = “车”) = 0.01
```

**在 LLM 中的作用**：logits 经过 softmax 后得到 vocabulary 上的概率分布。

---

### 8.3 Categorical Distribution：类别分布

**通俗解释**：类别分布表示从多个类别中选一个。

**数学定义**：

\[
X \sim \mathrm{Categorical}(p_1,p_2,\dots,p_K)
\]

其中：

\[
\sum_{i=1}^K p_i=1
\]

**在 LLM 中的作用**：每一步生成 token，本质上就是从 vocabulary 的 categorical distribution 中采样。

---

### 8.4 Gaussian Distribution：高斯分布 / 正态分布

**通俗解释**：高斯分布就是常见的钟形曲线。

**数学定义**：

\[
x \sim \mathcal{N}(\mu,\sigma^2)
\]

密度函数为：

\[
p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\]

**在 AIGC 中的作用**：

- VAE latent variable 通常假设为高斯；
- Diffusion 的加噪过程是高斯 transition；
- 初始化、噪声采样、重参数化技巧都常用高斯分布。

---

### 8.5 Conditional Probability：条件概率

**通俗解释**：条件概率表示在已知某件事发生的情况下，另一件事发生的概率。

**数学定义**：

\[
P(A|B)=\frac{P(A,B)}{P(B)}
\]

**在 LLM 中的作用**：语言模型建模的是：

\[
p(x_t|x_1,x_2,\dots,x_{t-1})
\]

也就是“给定前文，下一个 token 的概率”。

---

### 8.6 Bayes Rule：贝叶斯公式

**通俗解释**：贝叶斯公式用于根据观察到的证据更新我们对假设的相信程度。

**数学定义**：

\[
P(H|D)=\frac{P(D|H)P(H)}{P(D)}
\]

其中：

- \(P(H)\)：先验；
- \(P(D|H)\)：似然；
- \(P(H|D)\)：后验；
- \(P(D)\)：证据。

**在 AIGC/LLM 中的作用**：

- MAP 估计；
- Bayesian inference；
- classifier guidance；
- 不确定性建模；
- latent variable model。

---

### 8.7 Expectation：期望

**通俗解释**：期望是随机变量的平均结果。

**数学定义**：

离散情形：

\[
\mathbb{E}[X]=\sum_x xP(X=x)
\]

连续情形：

\[
\mathbb{E}[X]=\int xp(x)dx
\]

**在深度学习中的作用**：训练目标通常是数据分布上的期望风险：

\[
\min_\theta \mathbb{E}_{(x,y)\sim p_{data}}[L(f_\theta(x),y)]
\]

现实中我们用 mini-batch 均值近似这个期望。

---

### 8.8 Variance：方差

**通俗解释**：方差衡量随机变量波动有多大。

**数学定义**：

\[
\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]
\]

**在模型训练中的作用**：

- 梯度估计有方差；
- 初始化要控制激活方差；
- attention 缩放因子 \(\sqrt{d_k}\) 与点积方差有关；
- diffusion 的 noise schedule 控制噪声方差。

---

### 8.9 Covariance：协方差

**通俗解释**：协方差衡量两个变量是否一起变化。

**数学定义**：

\[
\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]
\]

**在 AIGC/LLM 中的作用**：

- 表示空间分析；
- PCA；
- whitening；
- Gaussian latent variable；
- feature correlation 分析。

---

### 8.10 Maximum Likelihood Estimation：最大似然估计，MLE

**通俗解释**：MLE 的思想是：选择一组参数，让已经观察到的数据出现的概率最大。

**数学定义**：

\[
\theta^*=\arg\max_\theta \prod_{i=1}^n p_\theta(x_i)
\]

通常取 log：

\[
\theta^*=\arg\max_\theta \sum_{i=1}^n \log p_\theta(x_i)
\]

**在 LLM 中的作用**：next-token prediction 可以看成最大化训练文本的条件似然：

\[
\max_\theta \sum_t \log p_\theta(x_t|x_{<t})
\]

---

### 8.11 MAP：最大后验估计

**通俗解释**：MAP 在 MLE 的基础上加入先验偏好。

**数学定义**：

\[
\theta^*=\arg\max_\theta p(\theta|D)
=\arg\max_\theta p(D|\theta)p(\theta)
\]

**在深度学习中的作用**：正则化可以看成某种先验。例如 L2 regularization 对应高斯先验。

---

### 8.12 Monte Carlo：蒙特卡洛方法

**通俗解释**：当期望或积分算不出来时，就随机采样很多次，用平均值近似。

**数学形式**：

\[
\mathbb{E}_{x\sim p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i)
\]

**在 AIGC/LLM 中的作用**：

- mini-batch training；
- sampling；
- diffusion sampling；
- policy gradient；
- benchmark bootstrap。

---

### 8.13 Sampling：采样

**通俗解释**：采样就是从一个概率分布中抽取一个具体结果。

**在 LLM 生成中的作用**：模型输出 vocabulary 上的概率分布，然后通过采样策略选择下一个 token。

常见采样方法：

- greedy decoding：永远选概率最大的 token；
- temperature sampling：调节分布尖锐程度；
- top-k sampling：只从概率最高的 k 个 token 中采样；
- top-p sampling：只从累计概率达到 p 的 token 集合中采样。

---

## 9. 核心概念解释：信息论

### 9.1 Entropy：熵

**通俗解释**：熵衡量不确定性。分布越均匀，不确定性越高；分布越集中，不确定性越低。

**数学定义**：

\[
H(p)=-\sum_i p_i \log p_i
\]

**在 LLM 中的作用**：

- 生成分布的不确定性；
- 模型信心；
- decoding 策略；
- 数据压缩视角。

---

### 9.2 Cross-Entropy：交叉熵

**通俗解释**：交叉熵衡量用一个分布 \(q\) 去表示真实分布 \(p\) 时的代价。

**数学定义**：

\[
H(p,q)=-\sum_i p_i\log q_i
\]

**在 LLM 中的作用**：语言模型训练通常最小化 next-token cross-entropy。

如果真实标签是 one-hot，那么 loss 就是正确 token 的负 log 概率：

\[
L=-\log q_{y}
\]

---

### 9.3 KL Divergence：KL 散度

**通俗解释**：KL 散度衡量两个概率分布的差异，但它不是对称距离。

**数学定义**：

\[
D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}
\]

**在 AIGC/LLM 中的作用**：

- VAE 的 posterior regularization；
- RLHF 中限制新 policy 不要偏离 reference model；
- DPO 中 policy ratio 的理论基础；
- distillation 中对齐 teacher/student 分布。

---

### 9.4 Mutual Information：互信息

**通俗解释**：互信息衡量知道一个变量后，能减少另一个变量多少不确定性。

**数学定义**：

\[
I(X;Y)=D_{KL}(p(x,y)\|p(x)p(y))
\]

**在 AIGC/LLM 中的作用**：

- 表示学习；
- bottleneck 分析；
- 多模态对齐；
- prompt 和输出之间的信息依赖。

---

### 9.5 Perplexity：困惑度

**通俗解释**：perplexity 可以理解为模型在每一步平均“困惑于多少个选择”。越低越好。

**数学定义**：如果 cross-entropy 是 \(H\)，则：

\[
\mathrm{PPL}=\exp(H)
\]

**在 LLM 中的作用**：常用于衡量语言模型对测试文本的预测能力。

**注意**：perplexity 低不等于回答能力强，因为指令遵循、推理、工具使用、安全性、事实性等能力无法完全由 perplexity 捕捉。

---

### 9.6 Bits 与 Nats

**通俗解释**：信息量可以用不同单位度量。使用 \(\log_2\) 时单位是 bits，使用自然对数 \(\log_e\) 时单位是 nats。

**在 LLM 中的作用**：loss 通常用 nats 表示；压缩视角下也可以换算成 bits-per-token。

---

# Part IV：优化理论与数值计算

## 10. 为什么优化决定模型能不能训好

有了模型和 loss，还需要找到让 loss 尽可能低的参数。LLM 训练通常是大规模非凸优化问题：参数极多，数据极大，loss landscape 非常复杂。

---

## 11. 核心概念解释：优化

### 11.1 Objective Function：目标函数

**通俗解释**：目标函数定义了我们要优化什么。

**在 LLM 中**：

\[
\min_\theta \mathcal{L}(\theta)
= -\sum_t \log p_\theta(x_t|x_{<t})
\]

---

### 11.2 Loss Function：损失函数

**通俗解释**：loss 衡量模型当前做得有多差。

**在训练中**：loss 越小，说明模型对训练目标拟合得越好。

常见 loss：

- cross-entropy loss；
- mean squared error；
- contrastive loss；
- DPO loss；
- diffusion denoising loss。

---

### 11.3 SGD：随机梯度下降

**通俗解释**：SGD 每次用一小批数据估计梯度，然后沿负梯度方向更新参数。

**数学形式**：

\[
\theta_{t+1}=\theta_t-\eta \nabla_\theta L(\theta_t)
\]

**在深度学习中的作用**：SGD 是现代优化器的基础。

---

### 11.4 Mini-batch

**通俗解释**：一次不用全部数据，而是抽一小批数据来估计梯度。

**优点**：

- 降低计算成本；
- 利用 GPU 并行；
- 引入适度噪声，有时能帮助泛化。

---

### 11.5 Momentum：动量

**通俗解释**：动量让优化像带惯性的球，减少梯度方向来回震荡。

**数学形式**：

\[
v_t=\beta v_{t-1}+\nabla_\theta L(\theta_t)
\]

\[
\theta_{t+1}=\theta_t-\eta v_t
\]

---

### 11.6 Adam

**通俗解释**：Adam 同时估计梯度的一阶矩和二阶矩，为每个参数自适应调整步长。

**核心形式**：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

**在 LLM 中的作用**：Adam/AdamW 是训练 Transformer 的常见优化器。

---

### 11.7 AdamW

**通俗解释**：AdamW 把 weight decay 从梯度更新中解耦出来，通常比 Adam + L2 regularization 更适合大规模深度网络。

**核心区别**：

- Adam + L2：把正则项加入梯度；
- AdamW：单独对权重做衰减。

**在 LLM 中的作用**：预训练和微调中非常常见。

---

### 11.8 Learning Rate Schedule：学习率调度

**通俗解释**：学习率决定每一步走多远。训练早期通常需要 warmup，后期需要 decay。

常见策略：

- linear warmup；
- cosine decay；
- step decay；
- constant with warmup。

**在 LLM 中的作用**：学习率设置直接影响训练稳定性和最终性能。

---

### 11.9 Weight Decay：权重衰减

**通俗解释**：weight decay 会让权重不要无限变大，从而起到正则化作用。

**数学直觉**：惩罚大的权重范数。

\[
L'=L+\lambda\|\theta\|_2^2
\]

---

### 11.10 Gradient Clipping：梯度裁剪

**通俗解释**：当梯度过大时，把它缩小到合理范围，防止训练发散。

**数学形式**：

如果 \(\|g\| > c\)，则：

\[
g \leftarrow c\frac{g}{\|g\|}
\]

**在 LLM 中的作用**：长序列、大 batch、RLHF 训练中都常用。

---

### 11.11 Non-convex Optimization：非凸优化

**通俗解释**：非凸问题的 loss landscape 可能有很多山谷、鞍点和平坦区域。

**在深度学习中的作用**：神经网络训练几乎都是非凸优化。

**重要直觉**：大模型虽然非凸，但高维参数空间中存在很多可用的低 loss 区域，实际训练通常能找到表现不错的解。

---

### 11.12 Regularization：正则化

**通俗解释**：正则化是防止模型死记硬背训练集的方法。

常见正则化方法：

- weight decay；
- dropout；
- data augmentation；
- early stopping；
- label smoothing；
- KL regularization。

---

### 11.13 Constrained Optimization：约束优化

**通俗解释**：优化时不仅要让目标函数变好，还要满足某些约束。

**数学形式**：

\[
\min_x f(x) \quad \text{s.t.} \quad g(x)\le 0
\]

**在 RLHF 中的作用**：对齐训练常把“提高 reward”和“不要偏离原模型太远”一起考虑：

\[
\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi\|\pi_{ref})
\]

---

### 11.14 Duality：对偶

**通俗解释**：对偶把一个带约束的问题转换成另一个相关问题，有时更容易分析。

**在 LLM alignment 中的作用**：DPO 的推导涉及 KL-constrained RL 与 reward-policy duality。

---

## 12. 核心概念解释：数值计算

### 12.1 Floating Point：浮点数

**通俗解释**：计算机不能精确表示所有实数，只能用有限位数近似。

**在 LLM 中的作用**：训练大模型时，数值精度会影响稳定性、速度和显存占用。

---

### 12.2 FP32、FP16、BF16

**通俗解释**：这些是不同精度的浮点格式。

- FP32：精度高，但显存和计算成本高；
- FP16：更省显存、更快，但容易溢出或下溢；
- BF16：指数范围接近 FP32，训练大模型更稳定。

---

### 12.3 Mixed Precision：混合精度训练

**通俗解释**：部分计算用低精度提高速度，关键累积或参数更新用高精度保持稳定。

**在 LLM 中的作用**：几乎是现代大模型训练的标配。

---

### 12.4 Quantization：量化

**通俗解释**：量化是用更少 bit 表示权重或激活，例如从 FP16 降到 INT8 或 INT4。

**在 LLM 中的作用**：

- 降低显存占用；
- 提高推理速度；
- 支持本地部署；
- 可能带来精度损失。

---

### 12.5 Conditioning：条件数与病态问题

**通俗解释**：如果输入稍微变一点，输出就剧烈变化，这个问题就很病态。

**在训练中的作用**：差的 conditioning 会让优化更困难，可能导致梯度不稳定。

---

### 12.6 FLOPs

**通俗解释**：FLOPs 表示浮点运算次数，是衡量计算量的重要单位。

**在 LLM 中的作用**：训练 compute、推理成本、scaling law 分析都会用到 FLOPs。

---

### 12.7 Memory Bandwidth：显存带宽

**通俗解释**：显存带宽表示数据从显存读写的速度。很多 LLM 推理瓶颈不是算力，而是搬数据。

**在 LLM 中的作用**：KV cache、attention、large batch inference 都受显存带宽限制。

---

### 12.8 FlashAttention

**通俗解释**：FlashAttention 不是改变 attention 数学公式，而是更高效地组织计算和内存访问，减少显存读写。

**核心思想**：避免显式存储完整 \(T\times T\) attention matrix，使用分块计算提高效率。

**在 LLM 中的作用**：长上下文训练和推理的重要底层技术。

---

# Part V：深度学习通用机制

## 13. 核心概念解释：深度学习基础

### 13.1 Neuron：神经元

**通俗解释**：神经元接收输入，做加权求和，再经过非线性函数。

**数学形式**：

\[
y=\sigma(w^\top x+b)
\]

---

### 13.2 MLP：多层感知机

**通俗解释**：MLP 是由多层线性变换和非线性激活组成的网络。

**数学形式**：

\[
h_1=\sigma(W_1x+b_1)
\]

\[
y=W_2h_1+b_2
\]

**在 Transformer 中的作用**：Transformer block 中 attention 后面的 feed-forward network 本质上就是 MLP。

---

### 13.3 Activation Function：激活函数

**通俗解释**：激活函数提供非线性，否则多层线性网络仍然等价于一层线性变换。

常见激活函数：

- ReLU；
- GELU；
- SiLU；
- SwiGLU。

---

### 13.4 Initialization：初始化

**通俗解释**：初始化决定训练开始时参数的尺度。如果尺度不合适，信号可能爆炸或消失。

**在 LLM 中的作用**：大模型训练稳定性高度依赖初始化、归一化和残差结构。

---

### 13.5 Residual Connection：残差连接

**通俗解释**：残差连接让网络学习“在原输入基础上改一点”，而不是每层都完全重写表示。

**数学形式**：

\[
y=x+F(x)
\]

**在 Transformer 中的作用**：帮助梯度流动，使很深的网络可以训练。

---

### 13.6 Normalization：归一化

**通俗解释**：归一化控制激活的尺度，让训练更稳定。

常见方法：

- BatchNorm；
- LayerNorm；
- RMSNorm。

---

### 13.7 LayerNorm

**通俗解释**：LayerNorm 对一个 token 的 hidden dimension 做归一化。

**数学形式**：

\[
\mu=\frac{1}{d}\sum_i x_i
\]

\[
\sigma^2=\frac{1}{d}\sum_i(x_i-\mu)^2
\]

\[
\mathrm{LayerNorm}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

---

### 13.8 RMSNorm

**通俗解释**：RMSNorm 不减均值，只用 root mean square 控制向量尺度。

**数学形式**：

\[
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}
\]

\[
\mathrm{RMSNorm}(x)=\gamma\frac{x}{\mathrm{RMS}(x)}
\]

**在 LLM 中的作用**：许多现代 LLM 使用 RMSNorm，因为它简单、高效、稳定。

---

### 13.9 Dropout

**通俗解释**：训练时随机丢掉一部分神经元或激活，防止模型过度依赖某些路径。

**在大模型中的注意点**：预训练大模型中 dropout 的使用和小模型不同，有些大模型会少用或不用 dropout，更多依赖数据规模和其他正则机制。

---

# Part VI：Transformer 与 LLM

## 14. Transformer 的核心公式

Transformer 的核心计算是 scaled dot-product attention：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

---

## 15. 核心概念解释：Transformer / LLM

### 15.1 Token

**通俗解释**：token 是模型处理文本的基本单位。它可能是一个字、一个词、一个子词，甚至一个标点或空格片段。

**在 LLM 中的作用**：LLM 不是直接处理自然语言字符串，而是处理 token id 序列。

---

### 15.2 Tokenizer

**通俗解释**：tokenizer 把文本切成 token，并映射为整数 id。

**例子**：

```text
"I love AI" → [40, 3021, 15592]
```

**常见 tokenizer**：

- BPE；
- SentencePiece；
- WordPiece；
- unigram language model tokenizer。

---

### 15.3 Vocabulary：词表

**通俗解释**：词表是模型认识的所有 token 集合。

**在 LLM 中的作用**：模型最终输出一个长度为 vocabulary size 的 logits 向量，然后 softmax 成每个 token 的概率。

---

### 15.4 Embedding

**通俗解释**：embedding 把离散 token id 映射成连续向量。

**数学形式**：

如果词表大小为 \(V\)，hidden dimension 为 \(d\)，则 embedding table 是：

\[
E\in\mathbb{R}^{V\times d}
\]

每个 token id 对应矩阵中的一行。

---

### 15.5 Positional Encoding：位置编码

**通俗解释**：Transformer 本身不天然知道 token 的顺序，所以需要额外注入位置信息。

**在 LLM 中的作用**：帮助模型区分“我爱你”和“你爱我”。

常见位置方法：

- sinusoidal positional encoding；
- learned positional embedding；
- RoPE；
- ALiBi。

---

### 15.6 RoPE：Rotary Position Embedding

**通俗解释**：RoPE 用旋转的方式把位置信息编码到 query 和 key 中，使 attention score 能感知相对位置。

**数学直觉**：把向量的二维子空间看成复平面，对不同位置施加不同角度的旋转。

**在 LLM 中的作用**：现代 decoder-only LLM 中非常常见，尤其适合相对位置建模。

---

### 15.7 Attention

**通俗解释**：attention 让每个 token 根据相关性选择应该关注其他哪些 token。

**三类向量**：

- Query：我现在想找什么信息；
- Key：我能提供什么索引；
- Value：我真正携带的信息内容。

---

### 15.8 Scaled Dot-product Attention

**数学公式**：

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

**为什么要除以 \(\sqrt{d_k}\)**：

如果 \(Q\) 和 \(K\) 的元素方差约为 1，那么点积的方差会随 \(d_k\) 增大。除以 \(\sqrt{d_k}\) 可以让 attention logits 的尺度更稳定，避免 softmax 饱和。

---

### 15.9 Causal Mask

**通俗解释**：causal mask 防止模型在预测当前位置时偷看未来 token。

**在 decoder-only LLM 中的作用**：保证自回归生成成立。

**形式**：attention matrix 中未来位置被设为 \(-\infty\)，softmax 后概率接近 0。

---

### 15.10 Multi-head Attention

**通俗解释**：multi-head attention 让模型在多个子空间中并行关注不同关系。

**数学形式**：

\[
\mathrm{head}_i=\mathrm{Attention}(XW_Q^i,XW_K^i,XW_V^i)
\]

\[
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)W_O
\]

**直觉**：一个 head 可能关注语法依赖，另一个 head 可能关注实体关系，还有一个 head 可能关注局部上下文。

---

### 15.11 Residual Stream

**通俗解释**：residual stream 是 Transformer 中信息流动的主干，每一层 attention 和 MLP 都是在这个主干上写入增量信息。

**在 interpretability 中的作用**：很多机制解释会把 Transformer 看成多个模块不断向 residual stream 写入特征。

---

### 15.12 MLP / FFN

**通俗解释**：Transformer 中的 MLP 负责对每个 token 的表示做非线性变换。

**常见形式**：

\[
\mathrm{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2
\]

---

### 15.13 SwiGLU

**通俗解释**：SwiGLU 是一种带门控的激活结构，可以让模型选择性通过信息。

**简化形式**：

\[
\mathrm{SwiGLU}(x)=\mathrm{SiLU}(xW_1)\odot (xW_2)
\]

其中 \(\odot\) 表示逐元素乘法。

---

### 15.14 Next-token Prediction

**通俗解释**：给定前面的 token，预测下一个 token。

**数学形式**：

\[
\max_\theta \sum_t \log p_\theta(x_t|x_{<t})
\]

**训练 loss**：

\[
\mathcal{L}=-\sum_t \log p_\theta(x_t|x_{<t})
\]

---

### 15.15 Logits

**通俗解释**：logits 是 softmax 之前的原始分数。它们还不是概率。

**数学形式**：

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

其中 \(z_i\) 是第 \(i\) 个 token 的 logit。

---

### 15.16 Temperature

**通俗解释**：temperature 控制采样的随机性。

**数学形式**：

\[
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

- \(T<1\)：分布更尖锐，更保守；
- \(T>1\)：分布更平坦，更随机。

---

### 15.17 Top-k Sampling

**通俗解释**：只从概率最高的 k 个 token 里采样，避免采到极低概率 token。

---

### 15.18 Top-p Sampling / Nucleus Sampling

**通俗解释**：选择累计概率达到 \(p\) 的最小 token 集合，然后在其中采样。

**区别于 top-k**：top-k 固定数量，top-p 根据分布形状动态改变候选集合大小。

---

### 15.19 KV Cache

**通俗解释**：自回归生成时，每一步都会用到过去 token 的 key 和 value。KV cache 把它们缓存起来，避免重复计算。

**在推理中的作用**：显著加速长文本生成，但会占用大量显存。

---

### 15.20 LoRA

**通俗解释**：LoRA 不直接更新完整大矩阵，而是只学习一个低秩增量。

**数学形式**：

\[
W'=W+\Delta W
\]

\[
\Delta W=BA
\]

其中：

\[
B\in\mathbb{R}^{d_{out}\times r},\quad A\in\mathbb{R}^{r\times d_{in}},\quad r\ll \min(d_{in},d_{out})
\]

**好处**：参数少、显存低、适合任务微调。

---

### 15.21 Adapter

**通俗解释**：adapter 是插入模型中的小模块。原模型参数可以冻结，只训练 adapter。

**在 LLM 中的作用**：参数高效微调。

---

### 15.22 Prefix Tuning / Prompt Tuning

**通俗解释**：不改模型主体参数，而是学习一些虚拟 token 或 prefix，引导模型完成任务。

**在 LLM 中的作用**：轻量微调、任务适配。

---

### 15.23 MoE：Mixture of Experts

**通俗解释**：MoE 有多个专家网络，每个 token 只激活其中一部分专家。

**数学直觉**：通过稀疏激活扩大模型总参数量，同时控制每次推理的计算量。

**关键概念**：

- router；
- expert；
- top-k routing；
- load balancing loss。

---

### 15.24 Scaling Laws

**通俗解释**：scaling laws 描述模型性能如何随着参数量、数据量、计算量增长而变化。

**常见形式**：

\[
L(N,D,C) \approx aN^{-\alpha}+bD^{-\beta}+cC^{-\gamma}
\]

其中 \(N\) 表示模型规模，\(D\) 表示数据规模，\(C\) 表示计算量。

**在研究中的作用**：指导模型尺寸、数据量和训练 compute 的分配。

---

# Part VII：AIGC 生成模型数学

## 16. 生成模型的统一视角

生成模型的目标是学习数据分布 \(p_{data}(x)\)，然后从中生成新样本。

不同方法的建模方式不同：

| 方法 | 核心思想 | 数学关键词 |
|---|---|---|
| VAE | 学习 latent variable model，用 ELBO 近似似然 | 变分推断、KL、重参数化 |
| GAN | 生成器和判别器进行 minimax game | 博弈、JS divergence、Wasserstein |
| Diffusion | 从数据逐步加噪，再学习反向去噪 | Markov chain、Gaussian、score matching |
| Score-based Model | 学习 \(\nabla_x\log p_t(x)\) | score、SDE、reverse process |
| Latent Diffusion | 在 latent space 中做 diffusion | autoencoder、cross-attention、conditional generation |
| Flow Matching | 学习从噪声到数据的连续 vector field | ODE、probability path、optimal transport |

---

## 17. VAE：变分自编码器

### 17.1 Latent Variable：潜变量

**通俗解释**：潜变量是看不见但影响观测数据的隐藏因素。

**例子**：一张人脸图片背后可能有姿态、光照、表情、身份等潜变量。

**数学形式**：

\[
z \sim p(z), \quad x \sim p_\theta(x|z)
\]

---

### 17.2 Autoencoder：自编码器

**通俗解释**：自编码器先把输入压缩成 latent 表示，再从 latent 表示重构输入。

**结构**：

```text
x → encoder → z → decoder → reconstructed x
```

---

### 17.3 ELBO：Evidence Lower Bound

**通俗解释**：真实的 \(\log p_\theta(x)\) 通常难以直接最大化，所以 VAE 最大化它的一个下界。

**数学形式**：

\[
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
D_{KL}(q_\phi(z|x)\|p(z))
\]

**两项解释**：

- reconstruction term：希望 decoder 能重构输入；
- KL term：希望 encoder 得到的 latent distribution 不要偏离先验太远。

---

### 17.4 Variational Inference：变分推断

**通俗解释**：真实后验太难算，就用一个简单分布 \(q_\phi(z|x)\) 去近似它。

**在 VAE 中的作用**：encoder 输出近似后验。

---

### 17.5 Reparameterization Trick：重参数化技巧

**通俗解释**：为了让采样过程可微，把随机性从参数中分离出来。

**数学形式**：

\[
z=\mu+\sigma\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
\]

这样梯度可以通过 \(\mu\) 和 \(\sigma\) 传播。

---

## 18. GAN：生成对抗网络

### 18.1 Generator：生成器

**通俗解释**：生成器把随机噪声变成看起来像真实数据的样本。

**数学形式**：

\[
\hat{x}=G(z),\quad z\sim p(z)
\]

---

### 18.2 Discriminator：判别器

**通俗解释**：判别器判断一个样本是真实数据还是生成器伪造的。

**数学形式**：

\[
D(x)\in[0,1]
\]

---

### 18.3 Minimax Game：极小极大博弈

**通俗解释**：生成器想骗过判别器，判别器想识别真假，两者互相对抗。

**经典目标**：

\[
\min_G \max_D
\mathbb{E}_{x\sim p_{data}}[\log D(x)]
+
\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
\]

---

### 18.4 Jensen-Shannon Divergence

**通俗解释**：JS divergence 是一种衡量两个分布差异的方法。原始 GAN 在理想情况下与最小化 JS divergence 有关。

---

### 18.5 Wasserstein Distance

**通俗解释**：Wasserstein distance 可以理解为把一个分布搬运成另一个分布所需的最小成本。

**在 GAN 中的作用**：WGAN 使用 Wasserstein 距离缓解训练不稳定和梯度消失问题。

---

### 18.6 Mode Collapse：模式崩塌

**通俗解释**：生成器只会生成少数几种样本，忽略数据分布中的多样性。

**例子**：训练人脸生成模型，但它总生成非常相似的几张脸。

---

## 19. Diffusion：扩散模型

### 19.1 Diffusion Model

**通俗解释**：扩散模型先把真实图片一步步加噪成纯噪声，再学习如何从噪声一步步去噪回图片。

**两阶段**：

```text
forward process:  x0 → x1 → x2 → ... → xT ≈ noise
reverse process:  noise → ... → x2 → x1 → x0
```

---

### 19.2 Markov Chain：马尔可夫链

**通俗解释**：下一步只依赖当前状态，不依赖更早历史。

**数学形式**：

\[
p(x_t|x_{t-1},x_{t-2},\dots,x_0)=p(x_t|x_{t-1})
\]

**在 DDPM 中的作用**：forward noising process 是一个 Markov chain。

---

### 19.3 Gaussian Transition：高斯转移

**DDPM 前向过程**：

\[
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
\]

**通俗解释**：每一步都把图像稍微缩小一点，再加一点高斯噪声。

---

### 19.4 Closed Form of Noising

DDPM 中可以直接从 \(x_0\) 采样到任意时刻 \(x_t\)：

\[
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
\]

其中：

\[
\alpha_t=1-\beta_t,\quad \bar{\alpha}_t=\prod_{s=1}^t\alpha_s
\]

等价采样形式：

\[
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I)
\]

---

### 19.5 Denoising Objective：去噪目标

常见训练目标：

\[
\mathbb{E}_{t,x_0,\epsilon}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
\]

**通俗解释**：模型看到加噪后的 \(x_t\)，学习预测当初加入的噪声 \(\epsilon\)。

---

### 19.6 Score

**通俗解释**：score 是 log probability 对输入的梯度，指向概率密度增加最快的方向。

**数学定义**：

\[
s(x)=\nabla_x\log p(x)
\]

**在 diffusion 中的作用**：反向去噪过程可以由 score 指导。

---

### 19.7 Score Matching

**通俗解释**：score matching 不直接学习概率密度，而是学习概率密度的梯度。

**在生成模型中的作用**：score-based generative model 的核心训练思想。

---

### 19.8 SDE：随机微分方程

**通俗解释**：SDE 描述带随机噪声的连续时间动态过程。

**常见形式**：

\[
dx=f(x,t)dt+g(t)dw
\]

其中 \(dw\) 表示 Brownian motion 的随机增量。

**在 diffusion 中的作用**：连续时间 diffusion 可以用 SDE 描述。

---

### 19.9 Reverse SDE

**通俗解释**：如果正向 SDE 把数据变成噪声，反向 SDE 就把噪声变回数据。

**在生成中的作用**：从高斯噪声开始，沿反向 SDE 采样得到图片或视频。

---

### 19.10 ODE：常微分方程

**通俗解释**：ODE 描述确定性的连续变化。

**数学形式**：

\[
\frac{dx}{dt}=f(x,t)
\]

**在生成模型中的作用**：probability flow ODE、flow matching、rectified flow 都会用到 ODE。

---

### 19.11 Classifier-free Guidance

**通俗解释**：classifier-free guidance 同时使用有条件和无条件预测，增强生成结果对 prompt 的遵循程度。

**常见形式**：

\[
\epsilon_{guided}=\epsilon_{uncond}+w(\epsilon_{cond}-\epsilon_{uncond})
\]

其中 \(w\) 是 guidance scale。

**直觉**：如果 \(w\) 太小，图像可能不听 prompt；如果太大，可能过度锐化或产生伪影。

---

## 20. Latent Diffusion 与多模态生成

### 20.1 Latent Space：潜空间

**通俗解释**：潜空间是压缩后的表示空间。图片不直接在像素空间生成，而是在更小、更语义化的 latent space 中生成。

---

### 20.2 Latent Diffusion

**通俗解释**：先用 autoencoder 把图片压缩到 latent space，再在 latent space 中运行 diffusion。

**优点**：比直接在像素空间做 diffusion 更省计算。

---

### 20.3 Cross-attention

**通俗解释**：cross-attention 让一种模态的信息去关注另一种模态的信息。

**在 text-to-image 中的作用**：图像 latent query 关注文本 token 的 key/value，从而把文本条件注入图像生成过程。

---

### 20.4 Conditional Generation：条件生成

**通俗解释**：生成过程不是无条件随机生成，而是在某些条件下生成。

条件可以是：

- 文本 prompt；
- 类别标签；
- 草图；
- 深度图；
- 边缘图；
- 音频；
- 视频帧。

---

### 20.5 Information Bottleneck：信息瓶颈

**通俗解释**：压缩表示时，保留任务相关信息，丢掉无关细节。

**在 latent diffusion 中的作用**：autoencoder latent space 可视为一种信息瓶颈，减少生成建模负担。

---

## 21. Flow Matching

### 21.1 Continuous Normalizing Flow

**通俗解释**：把一个简单分布通过连续可逆变换变成复杂数据分布。

**数学关键词**：

- ODE；
- vector field；
- change of variables；
- log-density evolution。

---

### 21.2 Vector Field：向量场

**通俗解释**：向量场给空间中每个点分配一个方向和速度。

**在 flow matching 中的作用**：模型学习一个 vector field，把噪声样本逐渐推向数据样本。

---

### 21.3 Probability Path：概率路径

**通俗解释**：概率路径描述从噪声分布到数据分布之间的一系列中间分布。

**在生成模型中的作用**：flow matching 通过学习这条路径上的速度场来生成样本。

---

### 21.4 Continuity Equation：连续性方程

**通俗解释**：连续性方程描述概率密度如何随向量场流动而变化。

**直觉**：概率质量不会凭空产生或消失，只是在空间中移动。

---

### 21.5 Optimal Transport：最优传输

**通俗解释**：最优传输研究如何以最小成本把一个分布搬运成另一个分布。

**在生成模型中的作用**：某些 flow matching 路径与 optimal transport 有紧密关系，可以产生更直的生成路径和更高效的采样。

---

# Part VIII：RLHF、DPO 与偏好优化

## 22. 为什么 LLM 对齐需要强化学习和偏好学习

预训练模型学会了预测文本，但不一定会按照人类意图回答。后训练阶段需要让模型更有帮助、更诚实、更安全、更符合指令。常见路线包括：

1. SFT：supervised fine-tuning，用人工示范训练；
2. RM：reward model，用人类偏好排序训练奖励函数；
3. PPO/RLHF：用强化学习优化 reward，同时用 KL 限制偏离；
4. DPO：直接从偏好数据优化 policy，避免显式 reward model 和复杂 RL loop。

---

## 23. 核心概念解释：强化学习

### 23.1 MDP：Markov Decision Process

**通俗解释**：MDP 是强化学习中描述智能体与环境交互的数学框架。

**组成部分**：

- state：状态；
- action：动作；
- reward：奖励；
- transition：状态转移；
- policy：策略；
- discount factor：折扣因子。

---

### 23.2 Policy：策略

**通俗解释**：策略决定在某个状态下选择什么动作。

**数学定义**：

\[
\pi(a|s)
\]

表示状态 \(s\) 下选择动作 \(a\) 的概率。

**在 LLM 中的对应**：

- state：prompt 和已生成上下文；
- action：下一个 token；
- policy：LLM 的 token 分布。

---

### 23.3 Reward：奖励

**通俗解释**：奖励告诉模型某个行为好不好。

**在 RLHF 中**：reward model 会给模型输出一个分数，表示它有多符合人类偏好。

---

### 23.4 Value Function：价值函数

**通俗解释**：价值函数估计从当前状态开始，未来能获得多少奖励。

**数学形式**：

\[
V^\pi(s)=\mathbb{E}_\pi\left[\sum_t \gamma^t r_t|s_0=s\right]
\]

---

### 23.5 Advantage：优势函数

**通俗解释**：advantage 衡量某个动作比平均水平好多少。

**数学形式**：

\[
A(s,a)=Q(s,a)-V(s)
\]

**在 PPO 中的作用**：policy gradient 常用 advantage 降低方差并提升训练稳定性。

---

### 23.6 Policy Gradient

**通俗解释**：policy gradient 直接调整策略参数，让高奖励动作概率变大，低奖励动作概率变小。

**数学形式**：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}[\nabla_\theta\log\pi_\theta(a|s)A(s,a)]
\]

---

### 23.7 PPO

**通俗解释**：PPO 是一种稳定的 policy optimization 方法，通过限制新旧策略变化幅度来避免训练崩掉。

**核心直觉**：不要让 policy 一步更新太远。

**在 RLHF 中的作用**：经典 RLHF pipeline 使用 PPO 优化 reward model 给出的奖励，同时加入 KL penalty。

---

### 23.8 KL-constrained RL

**通俗解释**：希望模型输出更符合偏好，但不要偏离原模型太远，否则可能语言质量下降或 reward hacking。

**数学形式**：

\[
\max_\pi
\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-
\beta D_{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x))
\]

---

## 24. 偏好学习与 DPO

### 24.1 Preference Data：偏好数据

**通俗解释**：偏好数据不是告诉模型标准答案是什么，而是告诉模型两个回答中哪个更好。

**形式**：

```text
prompt: x
chosen response: y_w
rejected response: y_l
```

---

### 24.2 Bradley–Terry Model

**通俗解释**：Bradley–Terry 模型用两个候选对象的分数差来表示其中一个被偏好的概率。

**数学形式**：

\[
P(y_w \succ y_l)=\sigma(r(x,y_w)-r(x,y_l))
\]

其中 \(\sigma\) 是 sigmoid 函数。

---

### 24.3 Reward Model

**通俗解释**：reward model 学习给一个 prompt-response 对打分。

**训练目标**：让被人类偏好的回答分数高于不被偏好的回答。

---

### 24.4 DPO：Direct Preference Optimization

**通俗解释**：DPO 不显式训练 reward model，也不用 PPO，而是直接用偏好数据更新语言模型。

**核心 loss**：

\[
\mathcal{L}_{DPO}
=
-\log \sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
\]

**直觉解释**：

- 如果 chosen response 相对 reference model 的概率提升更多；
- rejected response 相对 reference model 的概率提升更少；
- 那么 loss 会下降。

**关键点**：DPO 不是简单地让 chosen 概率变大，还通过 reference model 控制偏离幅度。

---

### 24.5 RLAIF

**通俗解释**：RLAIF 用 AI feedback 替代或辅助 human feedback，降低标注成本。

**风险**：AI feedback 可能继承 judge model 的偏见和盲点。

---

### 24.6 Reward Hacking

**通俗解释**：模型找到提高 reward 的捷径，但这些行为并不真正符合人类目标。

**例子**：模型输出看似礼貌、很长、很自信的答案，从 reward model 获得高分，但事实错误。

---

# Part IX：统计学习、评测与泛化

## 25. 为什么评测统计很重要

LLM 评测经常受到随机性、样本量、prompt 格式、judge 偏差、数据污染、任务选择等影响。研究中不能只报告一个分数，还需要说明不确定性。

---

## 26. 核心概念解释：评测统计

### 26.1 Benchmark

**通俗解释**：benchmark 是用于比较模型能力的数据集或任务集合。

**注意**：benchmark 分数不等于真实能力。一个模型可能在某个 benchmark 上高分，但在真实用户场景中表现一般。

---

### 26.2 Win-rate

**通俗解释**：win-rate 表示模型 A 在成对比较中胜过模型 B 的比例。

**数学形式**：

\[
\mathrm{win\ rate}=\frac{\#\mathrm{wins}}{\#\mathrm{comparisons}}
\]

---

### 26.3 Confidence Interval：置信区间

**通俗解释**：置信区间表示估计值的不确定范围。

**例子**：模型 A 的 win-rate 是 54%，95% 置信区间是 [51%, 57%]，说明真实 win-rate 很可能在这个范围附近。

---

### 26.4 Bootstrap

**通俗解释**：bootstrap 通过对样本重复有放回抽样来估计统计量的不确定性。

**在模型评测中的作用**：给 accuracy、win-rate、BLEU、ROUGE、reward score 等指标加置信区间。

---

### 26.5 P-value

**通俗解释**：p-value 衡量在零假设成立时，观察到当前或更极端结果的概率。

**注意**：p-value 不是“模型 A 比模型 B 更好的概率”。

---

### 26.6 Multiple Comparison：多重比较

**通俗解释**：如果同时比较很多模型或很多指标，偶然显著的概率会升高，需要校正。

**在 LLM 评测中的作用**：leaderboard 中大量模型比较时尤其重要。

---

### 26.7 Calibration：校准

**通俗解释**：如果模型说自己 80% 确信，那么类似情况下它应该大约 80% 正确。

**在 LLM 中的作用**：事实性回答、不确定性表达、风险控制。

---

### 26.8 OOD：Out-of-Distribution

**通俗解释**：OOD 指测试数据和训练数据分布不同。

**在 LLM 中的作用**：真实用户问题往往和训练 benchmark 不完全同分布。

---

## 27. 核心概念解释：泛化理论

### 27.1 Bias-Variance Tradeoff

**通俗解释**：bias 是模型太简单导致系统性错误，variance 是模型太敏感导致不稳定。

**在深度学习中的特点**：大模型的泛化行为并不完全符合传统小模型直觉，但 bias-variance 仍是重要基础。

---

### 27.2 Overfitting

**通俗解释**：模型在训练集上表现很好，但在新数据上表现差。

**在 LLM 中的表现**：

- benchmark contamination；
- memorization；
- prompt 格式过拟合；
- reward model overfitting。

---

### 27.3 Rademacher Complexity

**通俗解释**：衡量函数类拟合随机噪声的能力，能力越强，过拟合风险越高。

**在大模型研究中的作用**：作为理论分析工具，帮助理解模型容量和泛化。

---

### 27.4 VC Dimension

**通俗解释**：VC 维衡量模型能够打散多少样本，是传统统计学习理论中的容量度量。

**注意**：VC 维对现代大模型的实际泛化解释有限，但它仍是理解学习理论的重要基础。

---

### 27.5 PAC-Bayes

**通俗解释**：PAC-Bayes 用概率分布描述模型参数，并给出泛化界。

**在大模型中的作用**：常用于理解随机化模型、posterior、flat minima、压缩与泛化之间的关系。

---

# Part X：学习路线与实践项目

## 28. 六个月学习路线

### 阶段 1：数学基础快速重建，4–6 周

目标：能看懂深度学习论文中的大部分公式。

重点：

1. 线性代数：向量空间、矩阵分解、特征值/SVD、范数、投影、低秩近似、张量乘法。
2. 概率统计：条件概率、Bayes、MLE/MAP、KL、cross-entropy、Monte Carlo、Gaussian 分布族。
3. 矩阵微积分：Jacobian、Hessian、trace trick、softmax 梯度、cross-entropy 梯度、LayerNorm/RMSNorm 梯度。
4. 优化：SGD、Momentum、AdamW、learning rate schedule、weight decay、gradient clipping、非凸优化直觉。

实践：

- 手写矩阵乘法、softmax、cross-entropy；
- 手推 softmax + cross-entropy 梯度；
- 用 NumPy 实现两层 MLP；
- 比较不同学习率下的 loss curve。

---

### 阶段 2：深度学习数学，4–6 周

目标：理解 MLP、normalization、residual、regularization、optimization 的机制。

重点：

- computational graph；
- backpropagation；
- initialization；
- activation function；
- normalization；
- residual connection；
- dropout / weight decay；
- loss landscape；
- generalization。

实践：

- 从零写一个小 autodiff 引擎；
- 实现 LayerNorm 与 RMSNorm；
- 手写 AdamW；
- 对同一模型比较 SGD、Momentum、Adam、AdamW。

---

### 阶段 3：LLM 数学，6–8 周

目标：能独立推导和实现一个小型 decoder-only Transformer。

重点：

- tokenization；
- embedding matrix；
- positional encoding / RoPE；
- scaled dot-product attention；
- causal mask；
- multi-head attention；
- MLP/SwiGLU；
- normalization；
- residual stream；
- next-token objective；
- perplexity；
- temperature/top-p sampling；
- scaling laws；
- LoRA / adapter / prefix tuning。

实践：

- 实现一个 nanoGPT 级别模型；
- 训练一个小语料 next-token LM；
- 写 LoRA 低秩更新；
- 实现 SFT loss；
- 做一次 mini scaling law 实验。

---

### 阶段 4：AIGC 生成模型数学，6–8 周

目标：从概率建模角度统一理解 VAE、GAN、Diffusion、Flow Matching。

重点顺序：

1. VAE：ELBO、KL、reparameterization；
2. GAN：minimax、divergence、Wasserstein distance；
3. DDPM：Gaussian noising、Markov chain、denoising objective；
4. Score-based model：score matching、reverse SDE；
5. Latent Diffusion：latent space、autoencoder、cross-attention、conditional generation；
6. Flow Matching：ODE、vector field、probability path、optimal transport。

实践：

- 手推 DDPM 中 \(q(x_t|x_0)\) 的 closed form；
- 实现一个 MNIST/CIFAR 小型 DDPM；
- 比较 \(\epsilon\)-prediction、\(x_0\)-prediction、\(v\)-prediction；
- 实现 classifier-free guidance；
- 写一个 2D toy flow matching demo。

---

### 阶段 5：Alignment、Agent、评测统计，4–6 周

目标：能研究 LLM 后训练、偏好学习、agent 评估。

重点：

- preference data modeling；
- Bradley–Terry model；
- pairwise ranking loss；
- KL-constrained policy optimization；
- PPO 基础；
- DPO / IPO / KTO / ORPO 类方法；
- reward hacking；
- off-policy correction；
- benchmark 统计显著性；
- bootstrap confidence interval；
- multiple comparison；
- human eval 设计。

实践：

- 用公开 preference dataset 训练一个 DPO 小模型；
- 对两个模型输出做 pairwise win-rate 评估；
- 用 bootstrap 给 win-rate 加置信区间；
- 做一次 reward model overfitting 分析。

---

## 29. 公式能力检查清单

学完后应该能独立完成以下推导或解释：

### 29.1 Softmax + Cross-Entropy 梯度

\[
\frac{\partial L}{\partial z_i}=p_i-y_i
\]

要求：能说明为什么 softmax 和 cross-entropy 组合后梯度如此简洁。

---

### 29.2 Attention 的 shape 与复杂度

\[
QK^\top: [B,H,T,d]\times[B,H,d,T]\to[B,H,T,T]
\]

时间复杂度：

\[
O(BHT^2d)
\]

要求：能解释为什么长上下文 attention 成本随 \(T^2\) 增长。

---

### 29.3 VAE ELBO

\[
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
D_{KL}(q_\phi(z|x)\|p(z))
\]

要求：能解释 reconstruction term 和 KL term 的作用。

---

### 29.4 DDPM 前向过程闭式形式

\[
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
\]

要求：能从递推加噪推到闭式采样。

---

### 29.5 Score

\[
s_\theta(x,t)\approx \nabla_x\log p_t(x)
\]

要求：能解释为什么 score 指向更高概率密度方向。

---

### 29.6 Policy Gradient

\[
\nabla_\theta J(\theta)=
\mathbb{E}[\nabla_\theta\log\pi_\theta(a|s)A(s,a)]
\]

要求：能解释 advantage 如何影响 action probability。

---

### 29.7 DPO Loss

\[
\mathcal{L}_{DPO}
=
-\log \sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
\]

要求：能解释 chosen、rejected、reference model、policy ratio 的作用。

---

### 29.8 LoRA 低秩参数化

\[
\Delta W=BA,\quad \mathrm{rank}(\Delta W)\le r
\]

要求：能解释为什么低秩分解可以节省参数。

---

### 29.9 AdamW 与 Adam + L2 的区别

要求：能说明 decoupled weight decay 为什么不同于把 L2 项直接加到 Adam 的梯度里。

---

### 29.10 Benchmark 置信区间

要求：给定模型 A/B 的比较结果，能用 bootstrap 计算 win-rate 的置信区间。

---

# Part XI：按研究方向选择进阶数学

## 30. 如果偏 LLM Pretraining / Scaling

优先补：

- high-dimensional probability；
- random matrix theory；
- optimization dynamics；
- scaling law fitting；
- data distribution modeling；
- information theory；
- numerical linear algebra。

典型问题：

- 参数量、数据量、compute 如何分配？
- loss 是否符合 power-law？
- 数据质量如何影响 scaling？
- optimizer 和 learning rate schedule 如何影响大规模训练稳定性？

---

## 31. 如果偏 LLM Alignment / RLHF / Agent

优先补：

- reinforcement learning；
- preference learning；
- causal inference；
- decision theory；
- game theory；
- off-policy evaluation；
- statistical evaluation。

典型问题：

- 什么样的偏好数据能稳定提升模型？
- reward model 为什么会被 hack？
- DPO、PPO、KTO、ORPO 等方法的目标函数差异是什么？
- agent 任务如何设计可靠评测？

---

## 32. 如果偏 Image / Video / Audio Generation

优先补：

- stochastic process；
- SDE/ODE；
- optimal transport；
- signal processing；
- Fourier/wavelet；
- variational inference；
- score matching；
- geometric deep learning。

典型问题：

- diffusion 和 flow matching 如何统一？
- guidance scale 如何影响生成质量？
- 视频生成如何处理时间一致性？
- latent space 中的信息瓶颈如何影响细节？

---

## 33. 如果偏 Mechanistic Interpretability

优先补：

- linear algebra；
- sparse coding；
- information geometry；
- causal interventions；
- graph theory；
- representation similarity；
- spectral analysis。

典型问题：

- Transformer 中某个 head 在做什么？
- residual stream 中如何存储特征？
- feature superposition 如何发生？
- 如何用 causal intervention 验证机制假设？

---

## 34. 如果偏 Efficient LLM / Systems-aware ML

优先补：

- numerical analysis；
- quantization math；
- low-rank approximation；
- randomized linear algebra；
- matrix multiplication complexity；
- memory/computation tradeoff；
- approximation theory。

典型问题：

- INT4 量化为什么会损失精度？
- KV cache 为什么成为推理瓶颈？
- FlashAttention 为什么能加速？
- LoRA、QLoRA、adapter 的参数效率如何比较？

---

# Part XII：推荐论文与资源

## 35. 核心教材

1. **Deep Learning**，Ian Goodfellow、Yoshua Bengio、Aaron Courville  
   适合补深度学习数学、优化、生成模型基础。  
   https://www.deeplearningbook.org/

2. **Convex Optimization**，Stephen Boyd、Lieven Vandenberghe  
   适合补优化、对偶、约束优化。  
   https://web.stanford.edu/~boyd/cvxbook/

3. **The Matrix Cookbook**  
   矩阵恒等式和矩阵求导速查。  
   https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf

4. **Understanding Deep Learning**，Simon J. D. Prince  
   适合从现代深度学习视角串联理论和实践。  
   https://udlbook.github.io/udlbook/

---

## 36. LLM 方向论文

1. **Attention Is All You Need**  
   Transformer 基础论文。  
   https://arxiv.org/abs/1706.03762

2. **Scaling Laws for Neural Language Models**  
   语言模型 scaling laws 经典论文。  
   https://arxiv.org/abs/2001.08361

3. **Training language models to follow instructions with human feedback**  
   InstructGPT / RLHF 经典路线。  
   https://arxiv.org/abs/2203.02155

4. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**  
   DPO 核心论文。  
   https://arxiv.org/abs/2305.18290

---

## 37. AIGC 生成模型论文

1. **Generative Adversarial Nets**  
   GAN 原始论文。  
   https://arxiv.org/abs/1406.2661

2. **Denoising Diffusion Probabilistic Models**  
   DDPM 经典论文。  
   https://arxiv.org/abs/2006.11239

3. **Score-Based Generative Modeling through Stochastic Differential Equations**  
   Score-based diffusion 与 SDE 统一视角。  
   https://arxiv.org/abs/2011.13456

4. **High-Resolution Image Synthesis with Latent Diffusion Models**  
   Latent Diffusion 经典论文。  
   https://arxiv.org/abs/2112.10752

5. **Flow Matching for Generative Modeling**  
   Flow Matching 代表性论文。  
   https://arxiv.org/abs/2210.02747

---

## 38. 课程资源

1. **Stanford CS224N: Natural Language Processing with Deep Learning**  
   NLP 与 LLM 系统课程。  
   https://web.stanford.edu/class/cs224n/

2. **Stanford CS25: Transformers United**  
   Transformer 前沿 seminar。  
   https://web.stanford.edu/class/cs25/

---

# Part XIII：给 Codex 的生成指令

下面这段可以直接复制给 Codex，让它根据本文档生成完整博客教程。

```text
请根据当前 Markdown 文档，生成一个面向 AI 算法研究员和高级工程师的中文博客教程系列。

要求：

1. 总体风格：深入浅出，不要只堆公式。每个概念都必须先讲直觉，再给数学定义，再说明它在 AIGC/LLM 中的作用。
2. 每篇文章都要包含：
   - 学习目标；
   - 背景动机；
   - 概念解释；
   - 关键公式；
   - 公式逐步推导；
   - 与 LLM/AIGC 的具体连接；
   - 最小代码或伪代码；
   - 常见误区；
   - 练习题；
   - 延伸阅读。
3. 不要假设读者已经熟悉高阶数学。遇到 Jacobian、KL、ELBO、SDE、DPO、PPO、RoPE、LoRA、FlashAttention 等术语时必须解释。
4. 每个公式后都要解释每个符号的含义。
5. 对 Transformer、Diffusion、DPO 三个主题要重点展开，不能只做概念列表。
6. 代码示例优先使用 Python + NumPy 或 PyTorch。
7. 所有文章使用 Markdown 输出，公式使用 LaTeX。
8. 每篇文章结尾给 5–10 道练习题，并包含至少 2 道推导题、2 道代码题、1 道思考题。
9. 最终生成一个 README.md，说明整个教程系列的学习路径。
10. 保留本文档中的论文与资源链接作为延伸阅读。
```

---

# Part XIV：博客写作示例模板

下面是单个概念的推荐写法模板。

## 示例：什么是 KL 散度？

### 1. 直觉

KL 散度衡量的是：如果真实分布是 \(p\)，但我们用 \(q\) 去近似它，会多付出多少信息代价。

### 2. 数学定义

\[
D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}
\]

其中：

- \(p_i\)：真实分布给第 \(i\) 个事件的概率；
- \(q_i\)：近似分布给第 \(i\) 个事件的概率。

### 3. 重要性质

1. \(D_{KL}(p\|q)\ge 0\)；
2. 只有当 \(p=q\) 时，KL 为 0；
3. KL 不对称，即：

\[
D_{KL}(p\|q)\ne D_{KL}(q\|p)
\]

### 4. 在 LLM 中的作用

在 RLHF 中，我们希望新模型 \(\pi_\theta\) 获得更高 reward，但又不要偏离原始模型 \(\pi_{ref}\) 太远，因此会加入 KL penalty：

\[
D_{KL}(\pi_\theta\|\pi_{ref})
\]

### 5. 常见误区

- KL 不是普通距离，因为它不对称；
- KL 很大时，常常意味着一个分布认为可能的事件，另一个分布认为几乎不可能；
- KL 的方向很重要，\(D_{KL}(p\|q)
\) 和 \(D_{KL}(q\|p)\) 的优化行为不同。

---

# Part XV：最终学习建议

最小闭环是：

```text
线代 + 概率 + 优化
  → 深度学习
  → Transformer / LLM
  → Diffusion / Flow
  → RLHF / DPO
  → 评测统计
```

更具体地说：

1. 先补线性代数、概率、矩阵微积分、优化。
2. 再学 backprop、normalization、regularization、optimizer。
3. 然后攻 Transformer、language modeling、scaling laws。
4. 再扩展 VAE、GAN、DDPM、score SDE、latent diffusion、flow matching。
5. 最后进入 RLHF、DPO、agent、evaluation、interpretability、efficient training。

判断优先级的一句话：

> 凡是能帮助你推导 loss、理解训练稳定性、解释生成过程、设计实验和读懂新论文的数学，优先学；纯粹形式化但短期不服务模型机制的数学，先放后面。
