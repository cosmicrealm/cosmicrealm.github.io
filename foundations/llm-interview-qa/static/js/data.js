(function () {
  window.LLM_INTERVIEW_SECTIONS = [
    { id: "transformer", title: "Transformer 核心架构", range: "Q1-Q4" },
    { id: "training", title: "训练优化技术", range: "Q5-Q7" },
    { id: "architecture", title: "模型架构设计", range: "Q8-Q10" },
    { id: "finetuning", title: "微调技术", range: "Q11-Q13" },
    { id: "inference", title: "推理部署", range: "Q14-Q15" },
    { id: "multimodal", title: "多模态与生成", range: "Q16-Q17" },
    { id: "rag-agent-eval", title: "RAG / Agent / 评估", range: "Q18-Q20" }
  ];

  window.LLM_INTERVIEW_QA_DATA = [
    {
      id: "q1",
      number: 1,
      sectionId: "transformer",
      section: "Transformer 核心架构",
      level: "核心",
      tags: ["Attention", "Softmax", "复杂度"],
      question: "自注意力机制的工作原理是什么？为什么需要在计算中进行缩放（除以根号 d_k）？",
      oneLiner: "Self-Attention 用 QK 点积给 token 间关系打分，再用 softmax 权重汇聚 V；除以 sqrt(d_k) 是为了把点积分布拉回稳定尺度，避免 softmax 饱和。",
      oralAnswer: "自注意力先把每个 token 投影成 Q、K、V。Q 和 K 的点积表示当前位置查询其他位置的匹配程度，经过除以 \\(\\sqrt{d_k}\\) 和 softmax 后得到注意力权重，再对 V 做加权求和。缩放项不是经验装饰：如果 Q、K 元素近似独立且方差为 1，点积方差约为 \\(d_k\\)，标准差约为 \\(\\sqrt{d_k}\\)。不缩放时 logits 过大，softmax 容易接近 one-hot，梯度在饱和区变小，训练会更不稳定。",
      formulas: [
        "\\[\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V\\]",
        "若 \\(q_i,k_i\\) 独立且 \\(\\operatorname{Var}(q_i)=\\operatorname{Var}(k_i)=1\\)，则 \\(\\operatorname{Var}(q^\\top k)=\\sum_i \\operatorname{Var}(q_i k_i)\\approx d_k\\)。"
      ],
      deepDive: [
        "Q 负责提出查询，K 负责被匹配，V 负责提供被汇聚的信息。分数只由 QK 决定，是因为注意力权重描述“该看哪里”；V 不参与打分，是为了让相关性估计和信息内容解耦。",
        "时间复杂度常写为 \\(O(n^2d)\\)，这里的 \\(n^2\\) 来自所有 token 两两打分，\\(d\\) 来自向量维度。显存瓶颈还包括注意力概率矩阵的 \\(n^2\\) 中间激活；FlashAttention 等方法主要减少这个中间矩阵的物化和 HBM 读写。"
      ],
      engineering: [
        "长上下文下，attention 的主要压力通常不是参数量，而是序列长度平方项和 KV cache。训练时关注 attention matrix 激活显存；推理时关注历史 K/V 存储和 decode 阶段带宽。",
        "Self-Attention 中 Q/K/V 来自同一序列；Cross-Attention 中 Q 来自解码端状态，K/V 来自编码端或外部条件，常用于 encoder-decoder、扩散模型文本条件和多模态特征注入。"
      ],
      followUps: [
        { q: "为什么不是除以 d_k？", a: "除以 \\(d_k\\) 会把标准差压到约 \\(1/\\sqrt{d_k}\\)，logits 过小会让注意力过于平均。除以 \\(\\sqrt{d_k}\\) 正好把标准差拉回常数量级。" },
        { q: "为什么 V 不参与打分？", a: "打分阶段只需要判断 query 与 key 是否匹配；V 是匹配后要读取的内容。把内容和地址分开，能让同一个相关性分布作用到不同 value 表示。" },
        { q: "Self-Attention 和 Cross-Attention 的区别？", a: "前者 Q/K/V 来自同一序列，负责序列内部依赖；后者 Q 来自当前状态，K/V 来自条件源，负责从外部序列读取信息。" }
      ],
      pitfalls: [
        "不要只说“缩放防止数值太大”，要说明点积方差随 \\(d_k\\) 增长以及 softmax 饱和导致梯度变小。",
        "复杂度要区分计算复杂度和显存中的注意力矩阵；二者都含 \\(n^2\\)，但瓶颈来源不同。"
      ],
      memoryHook: "QK 打分，V 汇总；缩放控方差，避免 softmax 锁死。"
    },
    {
      id: "q2",
      number: 2,
      sectionId: "transformer",
      section: "Transformer 核心架构",
      level: "核心",
      tags: ["Multi-Head", "GQA", "表示学习"],
      question: "多头注意力机制的设计原理是什么？为什么多头比单头更好？头数是否越多越好？",
      oneLiner: "多头注意力是在多个低维子空间里并行做注意力，再把结果拼接融合；它提升表示分解能力，但头数过多会压缩单头维度并影响效率。",
      oralAnswer: "多头注意力不是简单投票集成，而是同一层里的子空间分解。给定 hidden size \\(d_{model}\\)，通常把它拆成 \\(h\\) 个 head，每个 head 的维度是 \\(d_{model}/h\\)，各自学习不同的 Q/K/V 投影并并行计算注意力。多个头能同时建模局部依赖、长程指代、语法结构或特定语义关系。但头数不是越多越好，头太多会让单头维度过小，单头表达能力和矩阵乘硬件效率都可能下降，也可能产生冗余头。",
      formulas: [
        "\\[\\operatorname{MHA}(X)=\\operatorname{Concat}(head_1,\\ldots,head_h)W_O,\\quad head_i=\\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)\\]"
      ],
      deepDive: [
        "多头的关键不是“多个模型平均”，而是把同一个 token 表示投影到多个可学习子空间。每个头的注意力图可以不同，输出拼接后再由 \\(W_O\\) 融合。",
        "当 \\(h\\) 增大且 \\(d_{model}\\) 固定时，\\(head\\_dim\\) 会变小。过小的 head 可能不足以表达复杂匹配函数；同时某些 GPU kernel 对 head_dim 有偏好，过碎的维度会影响吞吐。"
      ],
      engineering: [
        "模型压缩或加速时可以分析 head 重要性，剪掉冗余头或共享 K/V。许多推理优化并不改变 Q 头数，而是减少 KV 头数，这就引出 MQA/GQA。",
        "MHA、GQA、MQA 可以看成连续谱：MHA 每个 Q 头独有 K/V，GQA 多个 Q 头共享一组 K/V，MQA 所有 Q 头共享同一组 K/V。"
      ],
      followUps: [
        { q: "某些头冗余怎么办？", a: "可以做 head importance 分析、剪枝或蒸馏；但剪枝要重新评估质量，因为冗余头在不同数据分布上可能承担备份或稳定训练的作用。" },
        { q: "Multi-Head 和 GQA/MQA 的关系？", a: "GQA/MQA 主要减少 K/V 投影和 KV cache，而 Q 仍可保留多头表达，是质量、显存和吞吐之间的折中。" },
        { q: "head_dim 为什么不能太小？", a: "点积匹配需要足够维度承载子空间信息；维度太小会限制每个头的表达，也可能不适合底层 kernel 的高效矩阵乘形状。" }
      ],
      pitfalls: [
        "不要把多头解释为多个独立模型 ensemble；它们共享层输入并在同一层内融合。",
        "头数选择要和 \\(d_{model}\\)、head_dim、硬件 kernel、KV cache 策略一起看。"
      ],
      memoryHook: "多头是子空间分解，不是投票；头数增加会压缩单头维度。"
    },
    {
      id: "q3",
      number: 3,
      sectionId: "transformer",
      section: "Transformer 核心架构",
      level: "深入",
      tags: ["RoPE", "位置编码", "长上下文"],
      question: "RoPE 旋转位置编码的原理是什么？它如何实现相对位置编码？外推问题有哪些解决方案？",
      oneLiner: "RoPE 对 Q/K 的二维维度对施加位置相关旋转，使 \\(q_m^\\top k_n\\) 自然依赖相对距离 \\(m-n\\)，但长上下文外推仍要处理频率和插值副作用。",
      oralAnswer: "RoPE 把向量维度两两成对，每对看作二维平面，然后按 token 位置 \\(m\\) 旋转角度 \\(m\\theta_i\\)。Q 和 K 都被旋转后再做点积。由于二维旋转矩阵满足 \\(R_m^\\top R_n=R_{n-m}\\)，位置 \\(m\\) 的 query 与位置 \\(n\\) 的 key 的内积会包含相对距离 \\(n-m\\)。因此 RoPE 同时给模型提供绝对位置相位和相对位置关系。长上下文扩展常见做法包括 PI、NTK-aware、NTK-by-parts 和 YaRN，但都可能改变短上下文行为或高频位置信息，不能写成无代价通用结论。",
      formulas: [
        "\\[R(m\\theta)=\\begin{bmatrix}\\cos(m\\theta)&-\\sin(m\\theta)\\\\ \\sin(m\\theta)&\\cos(m\\theta)\\end{bmatrix}\\]",
        "\\[(R_m q)^\\top(R_n k)=q^\\top R_m^\\top R_n k=q^\\top R_{n-m}k\\]"
      ],
      deepDive: [
        "正弦位置编码把位置加到 token 表示上；RoPE 则把位置作为旋转作用到 Q/K 上。这样 attention logit 在计算时直接携带相对距离，而不是依赖模型从加法位置向量中自行恢复相对关系。",
        "不同维度对使用不同频率。高频维度对局部距离敏感，低频维度对长距离变化更慢。外推超过训练窗口时，某些频率会进入训练中少见的角度范围，导致注意力模式不稳定。"
      ],
      engineering: [
        "PI 把位置索引压缩回训练窗口，简单但会挤压高频细节；NTK-aware 调整 base 以保护不同频率；NTK-by-parts 分频段处理；YaRN 在插值策略之外引入温度/缩放修正以缓解注意力分布变化。",
        "部署长上下文模型时要同时评估短文本回归、needle 检索、长文多跳和真实任务延迟。只看能否跑到更长长度，不等于模型真的能有效利用长上下文。"
      ],
      followUps: [
        { q: "RoPE 是绝对位置还是相对位置？", a: "旋转角由绝对位置决定，但 QK 点积中通过 \\(R_m^\\top R_n\\) 表现为相对距离依赖，所以它兼具绝对相位注入和相对关系建模。" },
        { q: "增大 base 为什么可能影响短上下文？", a: "base 改变频率分布，会让模型熟悉的局部相位变化变慢，短距离位置分辨率和高频信息可能受影响。" },
        { q: "YaRN 是否不需要微调？", a: "一些场景可少量或无需额外训练获得可用外推，但这不是普遍保证；模型、上下文长度、任务和采样设置都会影响结果。" }
      ],
      pitfalls: [
        "不要把“能外推”写成“无需验证一定有效”。长上下文外推必须看任务质量和短上下文回归。",
        "不要只背 PI/NTK/YaRN 名称，要说出它们分别在压缩位置、调整频率、分段处理和注意力温度上的差异。"
      ],
      memoryHook: "RoPE 旋转 QK，相对距离从 \\(R_m^\\top R_n\\) 里出来。"
    },
    {
      id: "q4",
      number: 4,
      sectionId: "transformer",
      section: "Transformer 核心架构",
      level: "核心",
      tags: ["FFN", "SwiGLU", "非线性"],
      question: "Transformer 中 FFN 前馈神经网络的作用是什么？SwiGLU 相比传统 ReLU 有哪些优势？",
      oneLiner: "FFN 是逐 token 的非线性通道变换，主要提升特征组合能力；SwiGLU 用门控乘法增强表达，但 SiLU 不是 0 到 1 的概率门。",
      oralAnswer: "Transformer 块里 Attention 负责跨 token 混合，FFN 则对每个 token 独立做通道维度的非线性变换，典型形式是升维、激活、降维。它不直接混合序列位置，却承担大量参数和表达能力。SwiGLU 把传统激活换成门控乘法：一条分支经过 SiLU，另一条分支提供内容，两者逐元素相乘后再投影回 hidden size。它比 ReLU 类 FFN 更平滑、门控更灵活，在许多现代 LLM 中表现稳定。但要注意 SiLU 输出不是严格 \\([0,1]\\)，所以 SwiGLU 不是概率意义上的开关门。",
      formulas: [
        "\\[\\operatorname{FFN}(x)=W_2\\,\\phi(W_1x)\\]",
        "\\[\\operatorname{SwiGLU}(x)=W_{down}\\left(\\operatorname{SiLU}(W_{gate}x)\\odot W_{up}x\\right)\\]"
      ],
      deepDive: [
        "Attention 的输出已经混合了上下文信息，但每个 token 的 hidden channel 仍需要更强的非线性组合。FFN 相当于对每个位置共享同一套 MLP，扩展通道后再压回原维度。",
        "GLU 类结构把“产生内容”和“调制内容”拆成两条线性路径。乘法交互比单一激活更有表达力，也更适合作为 MoE 中 expert 的基本单元。"
      ],
      engineering: [
        "FFN 常占 Transformer 参数和 FLOPs 的大头，MoE 通常把 FFN 替换为多个稀疏 expert。优化 FFN 会直接影响训练吞吐、推理延迟和模型容量。",
        "实现时要注意不同架构命名：LLaMA 类常见 gate_proj、up_proj、down_proj；有些框架把 gate/up 合并为一个矩阵再切分。"
      ],
      followUps: [
        { q: "FFN 会混合不同 token 吗？", a: "标准 FFN 不会，它对每个位置独立应用同一 MLP。序列位置之间的信息混合主要发生在 Attention。" },
        { q: "SwiGLU 的门是不是 0/1 开关？", a: "不是。SiLU(x)=x·sigmoid(x)，输出可为负，也可大于 1，更像连续调制信号，不是概率门。" },
        { q: "为什么现代模型常用 SwiGLU/GEGLU？", a: "它们提供平滑激活和乘法门控，通常在相近计算预算下提升表达能力，但具体收益仍取决于模型规模和训练设置。" }
      ],
      pitfalls: [
        "修订说明：SiLU 输出不是严格 \\([0,1]\\)，不能说门控值接近 1 就“顺利通过”、接近 0 就“完全关闭”。",
        "不要说 FFN 负责捕获 token 间依赖；标准 FFN 是逐位置通道变换。"
      ],
      memoryHook: "Attention 混位置，FFN 混通道；SwiGLU 是连续调制，不是概率开关。"
    },
    {
      id: "q5",
      number: 5,
      sectionId: "training",
      section: "训练优化技术",
      level: "核心",
      tags: ["AMP", "FP16", "BF16", "GradScaler"],
      question: "混合精度训练（AMP）的原理是什么？如何解决 FP16 带来的数值稳定性问题？",
      oneLiner: "AMP 让适合低精度的矩阵计算用 FP16/BF16 加速，让敏感归约和权重更新保留更高精度；GradScaler 主要解决 FP16 梯度下溢。",
      oralAnswer: "混合精度不是把整个模型粗暴改成 FP16，而是按算子选择精度。矩阵乘和卷积适合用 FP16 或 BF16 利用 Tensor Core，参数更新、某些归约、softmax、LayerNorm、loss 累计等数值敏感操作常保留 FP32。FP16 的指数范围较小，容易上溢或下溢；BF16 精度位少一些，但指数位接近 FP32，动态范围更好。PyTorch 中 autocast 负责自动选择前向算子精度，GradScaler 负责把 loss 放大后反传，检测 inf/NaN 后跳过更新并调低 scale。",
      formulas: [
        "AMP = autocast 选择算子精度 + GradScaler 动态 loss scaling + FP32 master/update 路径。"
      ],
      deepDive: [
        "FP16 有更多尾数精度于低位宽中保留，但指数范围有限；BF16 尾数较短，表示精度低于 FP16，但动态范围接近 FP32，所以在大模型训练中常更稳定。",
        "loss scaling 的作用是把小梯度放大到 FP16 可表示范围内，优化器更新前再按 scale 缩回。如果检测到溢出，当前 step 通常跳过并降低 scale。"
      ],
      engineering: [
        "训练脚本中要区分 compute dtype、parameter dtype、optimizer state dtype。显存节省不仅来自权重，也来自激活、梯度和优化器状态。",
        "AMP 相关 NaN 常出现在 softmax logits、归一化、loss spike 或自定义 kernel。定位时先关 AMP 做对照，再逐步缩小到具体算子。"
      ],
      followUps: [
        { q: "BF16 为什么常比 FP16 稳？", a: "BF16 指数位更多，动态范围接近 FP32，不容易因为数值太大或太小直接溢出/下溢。" },
        { q: "autocast 和 GradScaler 分工是什么？", a: "autocast 控制前向和部分反向算子的 dtype；GradScaler 控制 loss 放大、溢出检测、跳过 step 和动态调整 scale。" },
        { q: "哪些操作常保留 FP32？", a: "大范围归约、softmax、LayerNorm、loss 累计、优化器更新等对误差或溢出敏感的操作经常保留更高精度。" }
      ],
      pitfalls: [
        "不要把 AMP 等同于全模型 FP16；真正的混合精度是按算子和状态选择 dtype。",
        "FP16 与 BF16 的主要区别不只是显存相同，而是指数范围和尾数精度的取舍。"
      ],
      memoryHook: "autocast 管算子，GradScaler 管梯度尺度；AMP 不是全 FP16。"
    },
    {
      id: "q6",
      number: 6,
      sectionId: "training",
      section: "训练优化技术",
      level: "核心",
      tags: ["Gradient Checkpointing", "显存", "重算"],
      question: "梯度检查点（Gradient Checkpointing）的原理是什么？它是如何进行时间与空间的权衡的？",
      oneLiner: "梯度检查点少存前向激活，反向时从检查点重算中间结果，用额外计算换训练显存。",
      oralAnswer: "正常反向传播需要保存前向过程中的大量激活，因为梯度计算依赖这些中间值。梯度检查点只保存部分边界激活，丢弃检查点之间的中间激活；反向传播需要它们时，从最近的检查点重新跑一段前向，再计算梯度并释放临时激活。这就是“用算力换显存”的精确定义。吞吐下降来自额外前向重算、kernel 调度和可能的通信等待。理论上某些最优调度可达到 \\(O(\\sqrt n)\\) 激活存储，但普通框架按层切分并不自动保证这个复杂度。",
      formulas: [
        "memory_saved = 少保存中间激活；extra_compute = backward 中重复执行部分 forward。"
      ],
      deepDive: [
        "切分粒度决定权衡。每层都 checkpoint 会节省更多激活，但重算更多；只对大模块 checkpoint 则收益较小但吞吐损失也小。",
        "检查点与可逆网络、FlashAttention 的重算思想相似，但作用对象不同：gradient checkpointing 通常作用于任意模块激活，FlashAttention 重点避免注意力矩阵物化。"
      ],
      engineering: [
        "在大模型训练中，checkpoint 常和 ZeRO/FSDP、sequence parallel、FlashAttention、activation offload 一起使用。调参时要看 tokens/s，而不只看能否塞进显存。",
        "含 dropout 或随机层时要确保重算路径的随机状态一致，否则反向对应的前向值会变，导致梯度错误或不可复现。"
      ],
      followUps: [
        { q: "为什么吞吐会下降？", a: "反向阶段要重新执行部分前向，增加 FLOPs；同时更细粒度切分可能带来额外调度开销和并行效率下降。" },
        { q: "所有 checkpoint 都是 O(sqrt n) 显存吗？", a: "不是。\\(O(\\sqrt n)\\) 是特定调度理论结果，实际实现取决于切分策略、模块图和框架实现。" },
        { q: "它和 FlashAttention 有什么相同点？", a: "都用重算减少中间激活保存；不同点是 FlashAttention 针对 attention softmax 分块和 IO，checkpoint 是通用模块级策略。" }
      ],
      pitfalls: [
        "修订说明：不要把 \\(O(\\sqrt n)\\) 当成所有实现自动达到的结果。",
        "不要只说“节省显存”，还要说明反向时从哪些边界激活开始重算，以及为什么会降低吞吐。"
      ],
      memoryHook: "前向少存，反向重算；省显存的账要用吞吐偿还。"
    },
    {
      id: "q7",
      number: 7,
      sectionId: "training",
      section: "训练优化技术",
      level: "深入",
      tags: ["FlashAttention", "HBM", "Online Softmax"],
      question: "Flash Attention 是如何优化标准自注意力计算的？为什么它能同时节省显存并加速计算？",
      oneLiner: "FlashAttention 通过分块、SRAM 暂存和 online softmax 避免完整注意力矩阵写回 HBM；它是 exact attention，不是近似算法。",
      oralAnswer: "标准 attention 会把 \\(QK^\\top\\) 和 softmax 概率矩阵物化到 HBM，再读回来乘 V。长序列下，瓶颈常是 HBM 读写而不是纯计算。FlashAttention 把 Q/K/V 分块搬到更快但更小的 SRAM，在块内完成打分、online softmax 归一化和对 V 的累积输出，并维护每行的最大值与归一化因子来保证分块结果等价于全局 softmax。反向时它可以重算注意力概率而不是保存完整矩阵，所以显存更省；HBM 往返减少后也更快。",
      formulas: [
        "Online softmax 维护每行 \\(m=\\max logits\\) 与 \\(l=\\sum \\exp(logits-m)\\)，新块到来时按新的最大值重标定旧累积。"
      ],
      deepDive: [
        "FlashAttention 的核心是 IO-aware。矩阵乘本身 GPU 很擅长，但把 \\(n\\times n\\) 中间结果反复写入/读出 HBM 会吞掉大量时间。",
        "它不是 sparse attention 或 low-rank attention。只要 mask 和数值路径一致，输出应与标准 attention 在浮点误差范围内等价。"
      ],
      engineering: [
        "收益随序列长度、head_dim、GPU 架构和 kernel 支持而变。短序列或不匹配的形状下，收益可能不明显。",
        "训练中 FlashAttention 常与 AMP、checkpointing、sequence parallel 叠加；推理中还要结合 KV cache、paged attention 和 batch 调度看整体瓶颈。"
      ],
      followUps: [
        { q: "为什么它是 exact attention？", a: "它没有丢弃 token 或近似低秩分解，只是改变计算顺序，并用 online softmax 保持全局归一化一致。" },
        { q: "反向为什么省显存？", a: "标准实现保存 attention probability；FlashAttention 反向可根据 Q/K/V 和前向统计量重算概率，避免长期保存 \\(n^2\\) 矩阵。" },
        { q: "和 gradient checkpointing 的区别？", a: "二者都重算，但 FlashAttention 是注意力专用的 IO-aware kernel；checkpointing 是通用模块级激活重算策略。" }
      ],
      pitfalls: [
        "不要说 FlashAttention 是近似 attention；它主要改变内存访问模式和计算顺序。",
        "不要只说“分块所以快”，要点出 HBM/SRAM 层级和完整注意力矩阵物化的问题。"
      ],
      memoryHook: "不是少算，而是少搬；online softmax 保 exact。"
    },
    {
      id: "q8",
      number: 8,
      sectionId: "architecture",
      section: "模型架构设计",
      level: "核心",
      tags: ["Decoder-Only", "NTP", "KV Cache"],
      question: "为什么目前主流的 LLM 都采用 Decoder-Only 架构？与 Encoder-Only 和 Encoder-Decoder 架构相比有何优势和区别？",
      oneLiner: "Decoder-only 用 causal mask 做 next-token prediction，训练推理接口统一、上下文学习自然、KV cache 友好；但 encoder-decoder 在强条件生成中仍有价值。",
      oralAnswer: "Encoder-only 使用双向注意力，适合理解和分类，常见目标是 MLM；encoder-decoder 用 encoder 编码输入、decoder 自回归输出，适合翻译、摘要等输入输出结构强绑定任务；decoder-only 只用 causal self-attention，根据历史 token 预测下一个 token。开放式对话、代码和工具调用都可以统一成 next-token prediction，因此 decoder-only 的训练、推理和上下文学习接口很一致。推理时追加 token 只需缓存历史 K/V，也让服务系统更简单。但不能绝对化说 encoder-decoder 被淘汰，它在某些条件生成和编码效率场景仍有优势。",
      formulas: [
        "\\[\\max_\\theta \\sum_t \\log p_\\theta(x_t\\mid x_{<t})\\]"
      ],
      deepDive: [
        "训练目标决定接口。Decoder-only 把 prompt、示例、工具结果和答案都放进同一 token 序列，用同一个自回归目标建模，天然支持 in-context learning。",
        "Encoder-decoder 的 cross-attention 让输出显式关注输入表示，在翻译、摘要、语音识别等任务中仍很自然；encoder-only 则适合需要全局双向表示的打分或抽取任务。"
      ],
      engineering: [
        "Decoder-only serving 生态成熟：KV cache、continuous batching、speculative decoding、PagedAttention 等优化大多围绕自回归 decode 展开。",
        "架构选择还受数据管线、模型权重生态、推理框架和产品接口影响。工程生态越集中，decoder-only 的路径依赖越强。"
      ],
      followUps: [
        { q: "训练和推理一致性体现在哪里？", a: "训练时预测下一个 token，推理时也是不断预测下一个 token；没有额外 encoder/decoder 阶段切换。" },
        { q: "Encoder-decoder 还有哪些优势？", a: "对输入输出强对齐任务，它能把输入编码成可重复访问的表示，decoder 通过 cross-attention 读取，结构归纳更明确。" },
        { q: "为什么 KV cache 友好？", a: "Causal decode 每步只新增一个 token 的 K/V，历史 K/V 可复用；无需重复计算整个前缀。" }
      ],
      pitfalls: [
        "不要说 decoder-only 在所有任务上都必然优于 encoder-decoder；任务形态和成本约束会改变选择。",
        "不要把 causal mask 和双向 attention 混淆，decoder-only 不能在预测当前位置时看到未来 token。"
      ],
      memoryHook: "Decoder-only 把一切变成续写；生态围绕 KV cache 放大它的优势。"
    },
    {
      id: "q9",
      number: 9,
      sectionId: "architecture",
      section: "模型架构设计",
      level: "深入",
      tags: ["MoE", "Router", "Sparse Activation"],
      question: "MoE（Mixture-of-Experts）的架构原理是什么？DeepSeek MoE 做了哪些改进？",
      oneLiner: "MoE 用 router 为每个 token 选择少量 FFN expert，实现总参数量大、每 token 激活参数量小；核心难点是路由负载和溢出。",
      oralAnswer: "MoE 通常把 Transformer 中的 FFN 替换成多个 expert。每个 token 先经过 router 得到 expert 分数，再选择 Top-k expert 计算，最后按路由权重合并输出。它的价值在于稀疏激活：模型总参数量可以很大，但每个 token 只激活一小部分参数，所以计算量可控。工程难点包括负载均衡、capacity factor、token dropping、expert overflow、跨设备通信和路由稳定性。DeepSeek MoE 相关公开设计强调细粒度 expert 分割和 shared expert：前者提升知识粒度，后者保留通用能力；但具体配置不能泛化成所有 MoE 的通用规律。",
      formulas: [
        "\\[y=\\sum_{e\\in TopK(r(x))} p_e(x)\\,E_e(x)\\]"
      ],
      deepDive: [
        "总参数量和每 token 激活参数量必须分开说。MoE 的总容量来自许多 expert，但单个 token 的 FLOPs 主要由被选中的 Top-k expert 决定。",
        "负载均衡是训练稳定性的核心。如果 router 过度偏向少数 expert，会导致热门 expert 拥塞，其他 expert 训练不足，最终既浪费容量又影响质量。"
      ],
      engineering: [
        "capacity factor 决定每个 expert 能接收多少 token。容量不足会发生 overflow，系统可能丢 token、转发到备用 expert 或采用其他处理策略，这些都会影响训练信号。",
        "分布式 MoE 的性能瓶颈常在 all-to-all 通信。路由策略不只是模型问题，也是集群拓扑和 batch 组织问题。"
      ],
      followUps: [
        { q: "shared expert 有什么作用？", a: "shared expert 始终处理 token，承担通用模式；路由 expert 处理更专门的模式，能缓解所有知识都被稀疏路由切开的风险。" },
        { q: "细粒度 expert 分割的动机是什么？", a: "把大 expert 拆成更多小 expert，可让 router 组合更细的能力单元，但会增加路由和通信复杂度。" },
        { q: "MoE 推理一定更快吗？", a: "不一定。单 token 计算可能较低，但路由、通信、batch 不均衡和 expert 并行效率会决定真实延迟。" }
      ],
      pitfalls: [
        "不要只报总参数量来暗示每 token 成本；MoE 必须同时说明 active parameters。",
        "不要把某个模型的 expert 数、Top-k 或 shared expert 方案写成 MoE 的固定定义。"
      ],
      memoryHook: "MoE 大容量、小激活；router 省 FLOPs，也制造负载问题。"
    },
    {
      id: "q10",
      number: 10,
      sectionId: "architecture",
      section: "模型架构设计",
      level: "深入",
      tags: ["KV Cache", "MQA", "GQA", "MLA"],
      question: "KV Cache 的优化方案 MQA、GQA 和 MLA 各自的原理是什么？它们之间如何对比权衡？",
      oneLiner: "KV cache 显存随 batch、长度、层数、KV 头数、head_dim 和 dtype 线性增长；MQA/GQA 共享 K/V 头，MLA 则缓存低秩 latent 表示。",
      oralAnswer: "自回归推理中，prefill 会为整段 prompt 计算 K/V，decode 每步复用历史 K/V 并追加新 token。KV cache 显存约为 \\(2\\times B\\times L\\times N_{layer}\\times N_{kv\\_heads}\\times d_{head}\\times bytes\\)。MHA 每个 Q 头都有 K/V；MQA 让所有 Q 头共享一组 K/V，缓存最省但可能损失多样性；GQA 让一组 Q 头共享一组 K/V，是质量和缓存之间的折中；MLA 用低秩 latent cache 存压缩表示，再通过投影参与注意力，减少缓存但不等同于简单共享 K/V。",
      formulas: [
        "\\[KV\\ Cache\\ bytes \\approx 2\\cdot B\\cdot L\\cdot N_{layer}\\cdot N_{kv\\_heads}\\cdot d_{head}\\cdot bytes\\_per\\_value\\]",
        "MHA: \\(N_{kv}=N_q\\), GQA: \\(1<N_{kv}<N_q\\), MQA: \\(N_{kv}=1\\)."
      ],
      deepDive: [
        "prefill 阶段通常计算密集，因为一次处理整段 prompt；decode 阶段每步只处理一个新 token，却要读大量历史 K/V，常受内存带宽和调度影响。",
        "MLA 的思路是把 K/V 信息压到低维 latent，cache 存 latent 而不是完整 K/V。注意它不是把所有头共享一份 K/V，而是在参数化上引入低秩压缩和恢复路径。"
      ],
      engineering: [
        "KV cache 优化直接影响并发、上下文长度和 batch size。dtype 从 FP16 改到 FP8 或 INT8 cache 也会影响显存，但需要质量验证。",
        "GQA 往往是部署友好的折中：显著减少 cache，同时保留多组 K/V。MQA 更激进，适合对吞吐和内存极敏感的场景。"
      ],
      followUps: [
        { q: "batch 和 seq_len 如何影响 cache？", a: "它们都线性放大 KV cache；长上下文和高并发相乘时，cache 很快超过权重显存。" },
        { q: "MQA/GQA 会影响质量吗？", a: "可能会。共享 K/V 减少注意力多样性，但训练时适配得好可以获得较好的质量/效率折中。" },
        { q: "prefill 与 decode 瓶颈为什么不同？", a: "prefill 做大矩阵并行计算，偏 compute；decode 每步读历史 cache，串行且带宽敏感，偏 memory/scheduling。" }
      ],
      pitfalls: [
        "不要漏掉 batch、layer、kv_heads、head_dim 和 dtype；只写 \\(2\\times seq\\_len\\times heads\\) 不够工程化。",
        "不要把 MLA 简化成 MQA。MLA 的关键是 latent cache 和低秩投影设计。"
      ],
      memoryHook: "KV cache 看六个量：B、L、层、KV 头、head_dim、dtype。"
    },
    {
      id: "q11",
      number: 11,
      sectionId: "finetuning",
      section: "微调技术",
      level: "核心",
      tags: ["LoRA", "QLoRA", "NF4"],
      question: "LoRA 和 QLoRA 的原理是什么？LoRA 的秩 r 如何影响表达能力与效率的权衡？",
      oneLiner: "LoRA 冻结基座权重，只学习低秩增量 \\(\\Delta W=(\\alpha/r)BA\\)；QLoRA 再把基座以 4-bit NF4 等方式加载以节省显存。",
      oralAnswer: "LoRA 假设任务适配所需的权重更新近似低秩，因此冻结原权重 \\(W\\)，只训练两个小矩阵 \\(A\\) 和 \\(B\\)，前向等价于 \\(Wx+(\\alpha/r)BAx\\)。对一个 \\(d_{out}\\times d_{in}\\) 矩阵，LoRA 参数量是 \\(r(d_{in}+d_{out})\\)，所以 r 越大表达能力越强、显存和计算越高。alpha 控制增量尺度，target_modules 决定作用范围，dropout 影响正则。QLoRA 把冻结基座权重量化为 4-bit NF4，并使用 double quantization 和 paged optimizer 等技巧降低显存，但能否在单卡上微调超大模型取决于模型大小、序列长度、batch、优化器、offload 和硬件条件。",
      formulas: [
        "\\[W'=W+\\Delta W=W+\\frac{\\alpha}{r}BA\\]",
        "\\[\\#params_{LoRA}=r(d_{in}+d_{out})\\]"
      ],
      deepDive: [
        "LoRA 的低秩瓶颈来自 r。小 r 更省，但可表达的更新空间受限；大 r 更灵活，但逐步接近全量微调的成本。实践中还要看作用到 q_proj/v_proj 还是 attention+FFN 全覆盖。",
        "NF4 根据近似正态分布权重的分位数分配 4-bit 码值，比均匀 INT4 更贴合权重分布。QLoRA 前向时通常反量化到计算 dtype，反向只更新 adapter。"
      ],
      engineering: [
        "target_modules 是质量和成本的主要旋钮。只训 Q/V 常见且便宜，全覆盖 attention 与 FFN 更强但更耗显存。",
        "QLoRA 省的是冻结基座和优化器相关开销，不代表训练完全没有激活显存压力。长序列和大 batch 仍可能撑爆显存。"
      ],
      followUps: [
        { q: "B 为什么常零初始化？", a: "让 \\(BA\\) 初始为零，使微调开始时模型行为等价于基座，避免一开始扰动过大。" },
        { q: "r 和 alpha 如何配合？", a: "r 决定低秩容量，alpha/r 控制增量缩放。alpha 太大可能训练不稳，太小可能适配不足。" },
        { q: "QLoRA 是否等于训练 4-bit 权重？", a: "不是。通常 4-bit 基座冻结，只训练 LoRA adapter；计算时会临时反量化到 BF16/FP16 等 dtype。" }
      ],
      pitfalls: [
        "修订说明：不要无条件宣称“单张 24GB 可微调 65B”。这依赖量化、序列长度、batch、offload、优化器和实现细节。",
        "不要只说 LoRA 参数少，要能写出 \\(r(d_{in}+d_{out})\\) 并解释 r、alpha、target_modules 的作用。"
      ],
      memoryHook: "冻结 W，训练 BA；QLoRA 量化基座，不量化适配目标。"
    },
    {
      id: "q12",
      number: 12,
      sectionId: "finetuning",
      section: "微调技术",
      level: "核心",
      tags: ["SFT", "Chat Template", "Loss Mask"],
      question: "SFT（监督指令微调）的目的是什么？它与预训练的关系是什么？SFT 阶段会遇到哪些典型问题？",
      oneLiner: "SFT 仍是交叉熵训练，但用指令数据和 assistant-only loss 教模型按对话格式响应；它提升行为模式，不是注入新知识的唯一手段。",
      oralAnswer: "预训练让模型学习语言分布和大量知识，目标通常是 next-token prediction。SFT 继续用交叉熵，但数据变成 instruction-response 或多轮 chat，并通过 loss mask 只在 assistant 回复部分计算损失，用户输入和 system prompt 主要作为条件。SFT 的关键不是堆数量，而是 chat template 一致、EOS 学好、回答质量高、多样性足够。典型问题包括复读、过拟合固定格式、灾难性遗忘、格式污染和拒答风格被错误学习。若目标是更新事实知识，RAG 或持续预训练往往比单纯 SFT 更合适。",
      formulas: [
        "\\[\\mathcal{L}_{SFT}=-\\sum_t m_t\\log p_\\theta(y_t\\mid y_{<t}, prompt),\\quad m_t=1\\ only\\ for\\ assistant\\ tokens\\]"
      ],
      deepDive: [
        "SFT 与预训练都可看成 teacher forcing 下的交叉熵，但样本组织和 loss mask 不同。Chat template 定义角色边界，EOS 定义何时停止，二者出错会直接影响推理行为。",
        "SFT 更像行为和格式对齐：让模型学会回答方式、任务遵循和多轮对话约定。它不擅长把大量新事实可靠写入参数，尤其当事实频繁变化时。"
      ],
      engineering: [
        "数据清洗要去掉重复、低质、互相矛盾、模板污染和错误拒答样本。多轮数据要保证角色标记、工具结果和终止 token 一致。",
        "缓解遗忘可混入通用数据、降低学习率、缩短训练、使用 LoRA 或保留能力评测集做 early stopping。"
      ],
      followUps: [
        { q: "SFT 为什么只训 assistant 部分？", a: "用户内容是条件，不是希望模型复述的目标；assistant-only loss 让模型学习在给定指令下生成回答。" },
        { q: "EOS 没学好会怎样？", a: "模型可能停不下来、复读、把下一轮角色标记继续生成，或者过早结束回答。" },
        { q: "SFT 能注入新知识吗？", a: "可以记住一部分训练样本，但不是高可靠知识更新方案。动态知识更适合 RAG，领域语料适配可考虑持续预训练。" }
      ],
      pitfalls: [
        "不要把 SFT 说成“让模型拥有知识”的主要阶段；它更侧重指令遵循和回答格式。",
        "忽略 loss mask 和 chat template 是常见错误，实际训练质量很大程度取决于这些细节。"
      ],
      memoryHook: "预训练学续写，SFT 学按格式回答；mask 决定模型模仿谁。"
    },
    {
      id: "q13",
      number: 13,
      sectionId: "finetuning",
      section: "微调技术",
      level: "综合",
      tags: ["PPO", "DPO", "GRPO", "RLHF"],
      question: "RLHF 中 PPO、DPO、GRPO 三种算法的核心原理是什么？它们的优劣势如何对比？",
      oneLiner: "PPO 用奖励模型和 KL 约束做在线策略优化；DPO 直接从偏好对优化相对 logprob；GRPO 用组内相对奖励减少或去掉 critic。",
      oralAnswer: "PPO 版 RLHF 通常包含当前 policy、reference policy、reward model 和 value/critic 路径，用奖励分数加 KL 约束优化生成策略，并用 clip objective 限制更新幅度。critic 可以是独立模型，也可以是 value head，不一定是四个完整模型。DPO 从偏好对出发，直接提高 chosen 相对 rejected 的 logprob 差，并用 reference logprob 控制偏离，省去显式 reward model 训练和在线 RL 复杂度。GRPO 则对同一 prompt 采样一组回答，用组内均值和方差构造相对优势，减少或去掉 critic。三者主要权衡稳定性、成本、在线探索、奖励黑客和数据依赖。",
      formulas: [
        "PPO: \\(\\min(r_t A_t, \\operatorname{clip}(r_t,1-\\epsilon,1+\\epsilon)A_t) - \\beta KL(\\pi||\\pi_{ref})\\)",
        "DPO: \\(-\\log\\sigma\\left(\\beta[(\\log\\pi_\\theta(y_w)-\\log\\pi_{ref}(y_w))-(\\log\\pi_\\theta(y_l)-\\log\\pi_{ref}(y_l))]\\right)\\)",
        "GRPO advantage: \\(A_i=(r_i-\\operatorname{mean}(r))/\\operatorname{std}(r)\\) within sampled group."
      ],
      deepDive: [
        "PPO 能在线从当前策略采样，理论上可探索超出 SFT 数据的回答，但系统复杂、显存成本高，奖励模型漏洞会诱发 reward hacking。",
        "DPO 把偏好优化变成离线监督式训练，工程上简单稳定，但强依赖偏好数据覆盖；如果偏好对质量差或与当前策略分布差距大，收益会受限。"
      ],
      engineering: [
        "PPO 训练要监控 reward、KL、entropy、长度、拒答率和人工抽检样本；单看 reward 上升可能只是奖励模型被钻空子。",
        "DPO/GRPO 数据构造很关键。chosen/rejected 差异要体现真实偏好，不应只靠长度、格式或模板痕迹让模型学到捷径。"
      ],
      followUps: [
        { q: "PPO 为什么需要 reference model？", a: "reference 提供 KL 约束，防止 policy 为追求 reward 过度偏离原有语言分布和安全边界。" },
        { q: "DPO 完全不需要 reference 吗？", a: "通常仍需要 reference logprob 或等价约束来定义相对偏离；它省的是显式 reward model 和在线 RL 流程。" },
        { q: "GRPO 为什么能减少 critic？", a: "它用同一 prompt 多个样本的组内相对分数作为基线，构造优势估计，从而减少对单独 value model 的依赖。" }
      ],
      pitfalls: [
        "修订说明：critic 不一定是独立完整模型，可能是 policy 上的 value head；不要机械说 PPO 必须加载四个完整模型。",
        "不要说 DPO 没有 KL 思想。DPO 的 reference logprob 正是在控制策略偏离。"
      ],
      memoryHook: "PPO 在线问 reward，DPO 离线比偏好，GRPO 组内算优势。"
    },
    {
      id: "q14",
      number: 14,
      sectionId: "inference",
      section: "推理部署",
      level: "核心",
      tags: ["Quantization", "PTQ", "QAT", "NF4"],
      question: "模型量化的原理是什么？对称量化与非对称量化有何区别？QLoRA 中的 NF4 为什么比标准 Int4 精度更高？",
      oneLiner: "量化把浮点权重或激活映射到低位宽表示以省显存和带宽；对称量化围绕 0，非对称量化用 zero-point 对齐真实范围。",
      oralAnswer: "量化把 FP16/FP32 数值近似映射到 INT8、INT4 或特殊 4-bit 格式，以减少模型存储、显存带宽和部分计算成本。PTQ 是训练后校准或直接量化，QAT 在训练中模拟量化误差；weight-only 只量化权重，weight-activation quantization 同时处理激活，后者更难。对称量化用 \\([-a,a]\\) 和 scale 表示，zero-point 通常为 0；非对称量化用 \\([x_{min},x_{max}]\\) 映射到 \\([q_{min},q_{max}]\\)，常见公式是 \\(zero\\_point=round(q_{min}-x_{min}/scale)\\)。NF4 用正态分布分位数设计 4-bit 码值，更适合近似正态的权重，不等于普通均匀 INT4。",
      formulas: [
        "Symmetric: \\(scale=\\max(|x|)/q_{max},\\quad q=round(x/scale)\\)",
        "Asymmetric: \\(scale=(x_{max}-x_{min})/(q_{max}-q_{min}),\\quad zero\\_point=round(q_{min}-x_{min}/scale)\\)",
        "Dequant: \\(\\hat{x}=scale\\cdot(q-zero\\_point)\\)"
      ],
      deepDive: [
        "per-tensor 使用一个 scale，简单但误差大；per-channel 给每个输出通道 scale；group-wise 在若干权重一组内共享 scale，是 LLM 权重量化常见折中。",
        "NF4 的优势来自非均匀码本。权重通常集中在均值附近，NF4 在高密度区域分配更多刻度，尾部刻度较稀疏，量化误差更符合权重分布。"
      ],
      engineering: [
        "推理量化要区分存储节省和端到端加速。若 kernel 不支持低位计算，反量化开销可能抵消收益。",
        "量化评估不能只看平均分，要看长文本、数学、代码、函数调用和安全拒答等子任务是否有局部退化。"
      ],
      followUps: [
        { q: "PTQ 和 QAT 的差异？", a: "PTQ 在训练后量化或校准，成本低；QAT 训练中模拟量化误差，质量通常更稳但需要训练资源。" },
        { q: "weight-only 为什么更常见？", a: "权重分布固定，容易离线校准；激活动态依赖输入，量化难度和误差控制更复杂。" },
        { q: "zero-point 公式为什么不是 x_min/scale？", a: "zero-point 是整数域中实数 0 的位置，要考虑 q_min 偏移，常见形式是 \\(q_{min}-x_{min}/scale\\) 后取整。" }
      ],
      pitfalls: [
        "修订说明：非对称量化 zero-point 不能简单写成 \\(x_{min}/scale\\)，符号和 q_min 偏移都容易错。",
        "不要把 NF4 和 INT4 混为一谈；NF4 是非均匀 4-bit 码本，INT4 通常指整数均匀量化。"
      ],
      memoryHook: "scale 定步长，zero-point 定零点；NF4 按权重分布分刻度。"
    },
    {
      id: "q15",
      number: 15,
      sectionId: "inference",
      section: "推理部署",
      level: "深入",
      tags: ["vLLM", "PagedAttention", "KV Cache"],
      question: "vLLM 的 PagedAttention 机制如何解决传统 KV Cache 管理的显存碎片问题？",
      oneLiner: "PagedAttention 把 KV cache 切成固定大小 block，用 block table 映射逻辑序列到物理块，减少连续预分配造成的碎片和浪费。",
      oralAnswer: "传统推理系统常为每个请求预留一段连续 KV cache，按最大长度分配会造成内部碎片，多个不同长度请求进出又会形成外部碎片。PagedAttention 借鉴操作系统分页思想，把 KV cache 分成固定大小物理 block；每个请求看到的是逻辑连续的 token 序列，但底层通过 block table 映射到不连续物理块。生成时按需追加 block，请求结束后归还 block。多个请求共享 prefix 时可以共享物理块，分叉生成时用 copy-on-write。它优化的是 KV cache 管理和吞吐，不改变 attention 数学结果。",
      formulas: [
        "logical token blocks -> block table -> physical KV blocks"
      ],
      deepDive: [
        "内部碎片来自“预留但没用完”的空间，外部碎片来自显存中散落着大小不同的空洞。固定 block 池让回收和复用更简单。",
        "PagedAttention 的分页类比有边界：它不是 CPU 虚拟内存，也不是把显存换到磁盘；它是在 GPU KV cache 管理中使用固定块和映射表思想。"
      ],
      engineering: [
        "continuous batching 与 PagedAttention 配合后，请求可以动态加入和退出，提升在线服务吞吐。",
        "prefix caching 和并行采样能共享 prompt 部分 KV；当某条分支继续写入时，通过 copy-on-write 避免互相污染。"
      ],
      followUps: [
        { q: "PagedAttention 改变模型输出吗？", a: "理论上不改变。它改变 KV cache 的物理布局和访问方式，不改变 attention 公式。" },
        { q: "什么是 block table？", a: "它记录请求的逻辑块编号对应哪些物理 KV block，让逻辑序列连续而物理存储可分散。" },
        { q: "共享 prefix 如何工作？", a: "相同前缀的多个序列引用同一批物理 block；后续生成不同 token 时才为新内容分配或复制 block。" }
      ],
      pitfalls: [
        "不要说 PagedAttention 是新的注意力近似；它是 KV cache 内存管理机制。",
        "操作系统分页只是类比，不能延伸到磁盘换页、页表权限等不相关机制。"
      ],
      memoryHook: "KV 不必连续存；block table 让逻辑连续、物理分散。"
    },
    {
      id: "q16",
      number: 16,
      sectionId: "multimodal",
      section: "多模态与生成",
      level: "综合",
      tags: ["VLM", "LLaVA", "Qwen-VL", "视觉 Token"],
      question: "VLM（视觉语言模型）的主流架构设计模式有哪些？LLaVA 和 Qwen-VL 的架构有何异同？",
      oneLiner: "VLM 常见路线包括 dual encoder、fusion encoder、Q-Former/adapter 桥接和视觉 token 拼接到 LLM；具体模型特性必须按版本说明。",
      oralAnswer: "视觉语言模型要解决的是图像表示如何进入语言模型。CLIP-style dual encoder 分别编码图像和文本，适合检索和对齐；fusion encoder 让图文 token 早期融合，交互充分但计算更重；BLIP-2/Q-Former 用可学习 query 压缩视觉特征；LLaVA 类路线用视觉编码器加投影层，把图像特征变成 LLM 可读的视觉 token，再与文本 token 拼接。LLaVA 的特点是结构简洁、两阶段训练清晰。Qwen-VL 系列也遵循视觉编码器、适配层和 LLM 的大框架，但动态分辨率、视觉 token 压缩、OCR 或细粒度理解优化要按具体版本描述，不能混成统一架构。",
      formulas: [
        "image -> vision encoder -> projector/resampler -> visual tokens + text tokens -> LLM"
      ],
      deepDive: [
        "图像 token 过多会让 LLM 上下文成本暴涨，所以 VLM 需要 resampler、Q-Former、patch merge、动态分辨率或区域裁剪等压缩策略。",
        "视觉特征注入位置有多种：作为 prefix token 拼接到输入层、通过 cross-attention 被语言层读取、或在不同层注入多尺度视觉特征。"
      ],
      engineering: [
        "训练通常分阶段：先做图文对齐，让 projector 学会把视觉特征映射到语言空间；再用视觉指令数据训练模型在问答和推理中使用图像信息。",
        "OCR、表格、细粒度定位和长图理解对分辨率、patch 切分、token budget 和数据标注非常敏感，不能只靠一个“多模态”标签概括。"
      ],
      followUps: [
        { q: "图像 token 太多怎么办？", a: "可用动态分辨率、token merging、resampler/Q-Former、区域裁剪、分层读取或按任务选择不同视觉粒度。" },
        { q: "LLaVA 的两阶段训练是什么？", a: "先冻结大部分模块训练投影层做模态对齐，再用视觉指令数据训练投影层和语言侧适配能力。" },
        { q: "视觉特征注入 LLM 的位置有哪些？", a: "可作为输入 prefix、通过 cross-attention、通过 adapter 或在多层注入多尺度特征，不同路线成本和表达不同。" }
      ],
      pitfalls: [
        "修订说明：不要把某个后续版本的 Qwen-VL 特性写成所有 Qwen-VL 的统一架构；必须按版本限定。",
        "不要只说“视觉编码器 + LLM”，还要解释视觉 token 数量、压缩方式和训练阶段。"
      ],
      memoryHook: "VLM 的核心问题：图像特征怎么压缩、对齐、注入 LLM。"
    },
    {
      id: "q17",
      number: 17,
      sectionId: "multimodal",
      section: "多模态与生成",
      level: "核心",
      tags: ["Stable Diffusion", "LDM", "CFG"],
      question: "Stable Diffusion 中的潜在扩散模型（LDM）原理是什么？无分类器引导（CFG）是如何工作的？",
      oneLiner: "LDM 在 VAE latent 空间做扩散以降低计算；CFG 用条件和无条件噪声预测的差值加强文本条件，但 scale 过大可能破坏自然度。",
      oralAnswer: "像素空间扩散直接处理高维 RGB 图像，计算成本很高。Latent Diffusion 先用 VAE encoder 把图像压到低维 latent，再在 latent 上训练 U-Net 预测噪声，最后用 VAE decoder 还原图像。文本编码器提供 prompt embedding，U-Net 中的 cross-attention 让 latent 去噪过程读取文本条件。训练时随机加噪并最小化预测噪声和真实噪声的 MSE；推理时从随机噪声开始逐步去噪。CFG 训练时让模型见过有条件和空条件，推理时用 \\(\\epsilon_{uncond}+w(\\epsilon_{cond}-\\epsilon_{uncond})\\) 放大条件方向。w 过大可能导致过饱和、伪影和多样性下降。",
      formulas: [
        "\\[\\mathcal{L}=\\mathbb{E}_{z_0,t,\\epsilon}\\|\\epsilon-\\epsilon_\\theta(z_t,t,c)\\|_2^2\\]",
        "\\[\\epsilon_{guided}=\\epsilon_{uncond}+w(\\epsilon_{cond}-\\epsilon_{uncond})\\]"
      ],
      deepDive: [
        "VAE 决定像素和 latent 的接口质量；U-Net 决定去噪能力；Text Encoder 决定文本条件表示；Cross-Attention 是文本影响图像结构和语义的主要通道。",
        "Latent diffusion 不是无损压缩。VAE 的压缩率、重建质量和 latent 分布会限制最终图像细节，尤其是文字、细线和局部结构。"
      ],
      engineering: [
        "CFG scale、采样步数、scheduler、负面提示和分辨率会共同影响质量。scale 高并不总是更好，常需要按任务和模型调优。",
        "训练数据 caption 质量会强烈影响文本对齐。只调推理参数无法完全弥补训练阶段的文本-图像对齐缺陷。"
      ],
      followUps: [
        { q: "Pixel diffusion 和 latent diffusion 差异？", a: "前者在像素空间去噪，维度高但直接；后者在 VAE latent 空间去噪，成本低但受 VAE 表达限制。" },
        { q: "Cross-Attention 在哪里起作用？", a: "U-Net latent feature 作为 Q，文本 embedding 作为 K/V，让每步去噪都能参考 prompt 条件。" },
        { q: "CFG scale 太大会怎样？", a: "文本一致性可能增强，但图像可能过饱和、纹理伪影增加、多样性下降，甚至出现不自然结构。" }
      ],
      pitfalls: [
        "不要把 CFG 解释成外部分类器；classifier-free 的关键是同一模型同时学习条件和无条件预测。",
        "不要忽略 VAE。Stable Diffusion 的扩散主体在 latent 中，VAE 是像素和 latent 的边界。"
      ],
      memoryHook: "VAE 压缩，U-Net 去噪，文本靠 cross-attention，CFG 放大条件方向。"
    },
    {
      id: "q18",
      number: 18,
      sectionId: "rag-agent-eval",
      section: "RAG / Agent / 评估",
      level: "综合",
      tags: ["RAG", "Hybrid Search", "Rerank", "Evaluation"],
      question: "RAG（检索增强生成）系统的完整流程是什么？在各个环节中有哪些关键优化策略？",
      oneLiner: "RAG 分离线建库和在线检索生成：清洗切分、embedding/indexing、query rewrite、retrieve、rerank、context packing、generation 与 citation。",
      oralAnswer: "RAG 的目标是让模型回答时引用外部知识，而不是完全依赖参数记忆。离线阶段先清洗文档、按语义和结构切 chunk、生成 embedding、建立向量或混合索引，并保存 metadata。在线阶段对 query 做改写或分解，召回候选片段，再用 cross-encoder 或 reranker 精排，最后做 context packing，把相关内容、来源和约束放进 prompt 生成答案。优化点包括 chunk size/overlap、parent-child retrieval、metadata filter、BM25+dense hybrid search、RRF 融合、rerank、引用格式和答案忠实性评估。",
      formulas: [
        "Offline: documents -> clean -> chunk -> embed -> index",
        "Online: query -> rewrite -> retrieve -> rerank -> pack context -> generate -> cite",
        "\\[RRF(d)=\\sum_i \\frac{1}{k+rank_i(d)}\\]"
      ],
      deepDive: [
        "chunk 太小会丢上下文，太大又会稀释相关性并浪费窗口。overlap 能保留跨边界语义，但过多会引入重复。parent-child retrieval 用小块召回、大块提供上下文，是常见折中。",
        "dense retrieval 擅长语义相似，BM25 擅长关键词和实体匹配。RRF 用排名而不是原始分数融合，能避免不同检索器分数尺度不一致。"
      ],
      engineering: [
        "RAG 评估要拆开看：retrieval precision/recall、rerank 命中率、faithfulness、answer relevance、citation accuracy 和延迟成本。",
        "线上系统要记录 query、召回片段、最终上下文、引用和模型回答，方便诊断是没召回、排错序、上下文塞坏，还是生成阶段幻觉。"
      ],
      followUps: [
        { q: "query rewrite 有什么作用？", a: "把口语、指代或多意图问题改写成更适合检索的查询，也可把复杂问题分解为多个子查询。" },
        { q: "为什么需要 rerank？", a: "向量召回速度快但粗，cross-encoder rerank 能联合看 query 和文档，捕捉更细语义关系。" },
        { q: "如何评估 RAG 是否可信？", a: "同时看检索覆盖、答案忠实性、引用准确性和最终任务效果；只看回答流畅度不够。" }
      ],
      pitfalls: [
        "不要把 RAG 简化成“向量库 + prompt”。高质量 RAG 是数据治理、检索、重排、上下文组织和评估闭环。",
        "不要只优化生成模型。很多幻觉来自没有召回正确证据或上下文打包错误。"
      ],
      memoryHook: "RAG 先找证据再回答；错了要定位召回、重排、打包还是生成。"
    },
    {
      id: "q19",
      number: 19,
      sectionId: "rag-agent-eval",
      section: "RAG / Agent / 评估",
      level: "综合",
      tags: ["Agent", "ReAct", "Tool Use", "Safety"],
      question: "ReAct Agent 框架的设计原理是什么？为什么需要「推理 + 行动」的循环模式？Agent 比纯 LLM 多出了什么？",
      oneLiner: "ReAct 让模型在 Thought、Action、Observation、Final 循环中规划、调用工具、读取反馈并重规划；Agent 是 LLM 加工具、记忆、控制和安全边界。",
      oralAnswer: "纯 LLM 只能基于上下文生成文本，不能主动读取外部状态或执行动作。Agent 在 LLM 外加工具、记忆、规划和执行控制。ReAct 的循环是 Thought 分析当前状态，Action 调用搜索、数据库、代码执行或业务 API，Observation 接收工具结果，再继续推理直到 Final。这个循环的价值是让推理决定下一步行动，让行动带回新事实，并在失败时根据观察重规划。工程上还必须加入权限最小化、沙盒、审计日志、人类确认和 prompt injection 防护，否则工具调用能力会放大风险。",
      formulas: [
        "Thought -> Action(tool, args) -> Observation(result) -> Thought -> ... -> Final"
      ],
      deepDive: [
        "Agent 不等于一个 prompt。它至少包括可用工具 schema、执行器、状态管理、短期/长期记忆、停止条件、错误恢复和权限控制。",
        "Workflow、planner-executor 和 ReAct 的边界在控制权。固定 workflow 把步骤写死；planner-executor 先计划再执行；ReAct 更强调每次观察后即时调整。"
      ],
      engineering: [
        "工具失败要进入可恢复路径：解析错误、权限错误、网络失败、结果为空和结果冲突都需要反馈给模型或控制器，而不是静默吞掉。",
        "外部内容可能包含 prompt injection。系统应把工具返回视为不可信数据，限制可执行动作，并对写操作、发信、删数据等关键动作加确认。"
      ],
      followUps: [
        { q: "Agent 比 RAG 多什么？", a: "RAG 主要检索证据辅助回答；Agent 可以多步调用不同工具、修改外部状态并根据观察重规划。" },
        { q: "什么时候不用 Agent？", a: "任务步骤固定、无需外部状态或风险较高时，普通 workflow 或单次检索生成更稳定、更易审计。" },
        { q: "如何防 prompt injection？", a: "隔离工具输出和系统指令、最小权限、关键操作人工确认、记录审计日志，并把外部文本当作数据而非指令。" }
      ],
      pitfalls: [
        "不要把 Agent 说成“会思考的 LLM”即可；系统层的工具、状态、控制和安全才是差异。",
        "不要让模型无约束执行副作用工具。权限和确认是 Agent 设计的一部分。"
      ],
      memoryHook: "Agent = LLM + 工具 + 状态 + 控制；ReAct 每次观察后再行动。"
    },
    {
      id: "q20",
      number: 20,
      sectionId: "rag-agent-eval",
      section: "RAG / Agent / 评估",
      level: "综合",
      tags: ["Benchmark", "Evaluation", "Data Leakage"],
      question: "大模型评估中，一个好的 Benchmark 应该具备哪些标准？目前主流的 Benchmark 分别评测模型的哪些维度？",
      oneLiner: "好的 benchmark 要有区分度、鲁棒性、可复现、自动化、防泄漏并贴近真实任务；不能只看单一总分。",
      oralAnswer: "好的 Benchmark 首先要能区分模型能力，不能所有模型都接近满分；其次要可复现、对 prompt 和采样扰动鲁棒，并有清晰自动化评分流程；还要尽量防训练数据泄漏，并与真实任务相关。评测形态包括静态题库、动态题库、竞技场、人类偏好和真实工程任务。维度上可覆盖知识、推理、代码、Agent、多模态、安全、长上下文等。但具体榜单分数高度动态，如果没有日期、版本和评测设置，就不应写死。可靠评估要看任务分布、污染风险、采样参数、置信区间和错误样本，而不是只看一个平均分。",
      formulas: [
        "Report = score + date + model version + prompt/eval setting + sample size + confidence interval + contamination note"
      ],
      deepDive: [
        "静态选择题便宜、可复现，但容易被训练集污染并逐渐失去区分度。真实工程任务和动态评测更贴近使用场景，但成本高、噪声大、复现难度更高。",
        "LLM-as-a-Judge 灵活，但裁判模型可能有偏见、位置偏好、长度偏好或幻觉。人类偏好更贴近体验，但需要更严格抽样和一致性控制。"
      ],
      engineering: [
        "内部评测应按产品任务建集合：检索问答、工具调用、代码修复、长文分析、安全拒答等分别建指标和样本，而不是只引用公开榜单。",
        "上线前要固定模型版本、采样参数、prompt、评测脚本和随机种子；报告均值时尽量给置信区间或 bootstrap 估计。"
      ],
      followUps: [
        { q: "为什么不能只看榜单第一？", a: "榜单任务分布可能与真实业务不同，也可能受污染、采样参数和裁判偏差影响。需要看子任务和错误样本。" },
        { q: "如何防数据泄漏？", a: "使用动态题、私有集、时间切分、去重检测、canary 样本和污染审计，并避免把测试集进入训练或调参流程。" },
        { q: "固定答案、人类偏好、竞技场差异？", a: "固定答案便宜稳定，人类偏好贴近体验但贵，竞技场动态且抗泄漏较好但受用户分布和统计噪声影响。" }
      ],
      pitfalls: [
        "修订说明：具体分数必须附日期、模型版本和设置；没有来源时不要写成长期事实。",
        "不要把评估等同于单一 accuracy。任务分布、置信区间、污染风险和失败类型同样重要。"
      ],
      memoryHook: "Benchmark 看区分度和可信度；分数没有日期，就不是事实。"
    }
  ];
})();
