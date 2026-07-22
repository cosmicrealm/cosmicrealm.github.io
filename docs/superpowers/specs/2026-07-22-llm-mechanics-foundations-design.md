# LLM Mechanics Foundations 交互讲义设计

## 1. 目标与范围

在 `foundations/llm-mechanics/` 新建一篇中文、source-first、可交互的长篇技术讲义。页面沿一条真实的数据流解释：

```text
文本 → token ids → decoder block → next-token 训练 → prefill/decode → sampling/stop → 评测边界
```

首版只覆盖以下 12 个概念：

1. BPE / tokenizer；
2. decoder-only Transformer；
3. causal mask；
4. RoPE；
5. RMSNorm；
6. SwiGLU；
7. prefill 和 decode；
8. KV cache；
9. greedy、temperature、top-p；
10. repetition 和 EOS；
11. teacher forcing；
12. perplexity 的含义与局限。

不扩展为完整的预训练、SFT、RLHF、分布式训练或 nanochat speedrun 教程。页面目标是让读者建立机制闭环，而不是复刻某个大型训练工程。

## 2. 教学主张

整页不是十二篇词条，而是五章连续实验课。每一章均回答：

- 数据对象是什么，tensor shape 如何变化；
- 模型实际计算了什么；
- 公式为何这样写；
- 如何通过动态实验观察机制；
- 如何用最小代码复现；
- 破坏关键假设后会出现什么失败；
- Stanford CS336 2026 与 nanochat 当前实现分别支持什么结论。

每个机制站点采用统一节奏：

```text
提出问题 → 操作实验 → 解释机制 → 代码复现 → 故障注入 → 理解检查
```

页面正文使用中性、正式的技术表达，不设置“小白 / 专家”或类似显式受众分层。

## 3. 信息架构

### Chapter 01：文本如何变成模型输入

- byte-level BPE 的训练、merge rule、encode / decode；
- 词表大小、序列长度与压缩率的关系；
- UTF-8 byte、special token、round-trip 与 tokenizer 边界；
- 数据流：`text → bytes → token pieces → token ids → embeddings`。

### Chapter 02：一个 decoder block 如何计算

- decoder-only Transformer 的 residual stream；
- pre-norm block 中 attention 与 MLP 的顺序；
- causal mask 在 softmax 前屏蔽未来位置；
- RoPE 只旋转 Q/K，并通过相对相位编码位置；
- RMSNorm 的尺度控制、数值精度与 pre-norm 位置；
- SwiGLU 的双分支门控、SiLU 与 `d_ff`；
- 明确对照：nanochat 当前 MLP 使用 `ReLU²`，不是 SwiGLU。

### Chapter 03：模型如何学会预测下一个 token

- `inputs = tokens[:-1]` 与 `targets = tokens[1:]`；
- 使用金标准前缀并行计算所有位置的 next-token loss；
- teacher forcing 是该训练机制的标准术语，但 CS336 2026 核验材料没有把它作为独立术语系统讲授；
- cross-entropy、mean token NLL 与 `perplexity = exp(mean NLL)`；
- perplexity 对 tokenizer、数据域、长度口径、事实性和任务效用的限制。

### Chapter 04：生成为什么分成 prefill 和 decode

- prefill 处理完整 prompt，可在 token 维度并行，常偏 compute-bound；
- decode 每步只产生一个 token，常偏 memory-bound；
- KV cache 的 K/V shape、写入位置、缓存增长与显存公式；
- 对比 cached 与 uncached autoregressive generation；
- 明确 KV cache 复用的是每层 attention K/V，不是缓存最终答案或完整 hidden state。

### Chapter 05：概率分布如何变成最终文本

- greedy 是 argmax 的确定性选择；
- temperature 改变 logits 尺度，不是“随机度按钮”的黑盒；
- top-p 按累计概率动态构造候选集；
- repetition penalty 修改已出现 token 的 logits，会同时带来抑制退化与损伤必要重复的风险；
- EOS、最大长度与停止状态机；
- 同一 logits 下比较策略，并回看 perplexity 为何不能单独预测生成质量。

## 4. 动态实验

使用原生 HTML / CSS / JavaScript 实现 6 个稳定、可访问、无需外部服务的实验：

1. **BPE merge 工作台**：逐步或自动执行 merge；显示 pair frequency、token 序列、词表增量与压缩率；支持 reset 和自定义短文本。
2. **Decoder block 解剖台**：在 causal mask、RoPE、RMSNorm、SwiGLU 四个视图间切换；同步显示 tensor shape、矩阵 / 向量变化和 residual path。
3. **Teacher forcing 对齐器**：拖动序列位置；显示 input、target、logits、token NLL 与整段 perplexity；可切换错误的未 shift targets。
4. **Prefill / decode 时间线**：改变 prompt 与生成长度；比较并行 prefill 和逐 token decode 的计算单元、延迟构成与重复计算。
5. **KV cache 显存账本**：调节 layer、KV head、head dim、dtype、batch、context；实时计算每 token 与总 cache bytes；对比有 / 无缓存的注意力工作量。
6. **采样与停止游乐场**：在固定 logits 上切换 greedy、temperature、top-p 和 repetition penalty；显示概率柱状图、候选集、采样 token 与 EOS 状态。

每个实验必须具备：

- 可操作控件；
- 立即变化的视觉输出；
- 数值或公式读数；
- 对当前现象的文字解释；
- reset；
- 非法输入保护；
- 键盘操作、label 和 ARIA；
- 移动端稳定布局，不产生横向溢出。

## 5. 十二个最小可运行例子

每个概念提供一个独立文件、页面内代码块、复制按钮、下载链接、运行命令、预期输出、关键 assertion 和一个可观察的错误变体。

拟定文件：

```text
foundations/llm-mechanics/examples/
  README.md
  requirements.txt
  run_all.py
  01_bpe.py
  02_decoder_only.py
  03_causal_mask.py
  04_rope.py
  05_rmsnorm.py
  06_swiglu.py
  07_prefill_decode.py
  08_kv_cache.py
  09_sampling.py
  10_repetition_eos.py
  11_teacher_forcing.py
  12_perplexity.py
```

交付契约：

- 纯算法示例只用 Python 标准库；张量示例使用 PyTorch；
- 通常控制在 15–60 行，避免依赖预训练权重或下载数据；
- 固定 seed；
- 打印输入、关键 shape 和预期性质；
- assertion 直接检查核心机制；
- `run_all.py` 可批量执行全部示例并报告通过 / 失败；
- 页面中的变量名与浏览器实验保持一致。

## 6. 三层来源体系

每个概念均区分三类材料：

1. **机制基准**：给出定义、公式、shape 与适用条件；
2. **CS336 2026 课程映射**：精确到 lecture、handout 页码、assignment 或官方代码；
3. **nanochat 源码追踪**：固定官方 commit 并指出采用、替换或缺失。

### Stanford CS336 2026

核验入口（访问日期：2026-07-22）：

- [课程主页与 Spring 2026 schedule](https://cs336.stanford.edu/)
- [官方 lectures 仓库](https://github.com/stanford-cs336/lectures)
- [Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)

主要映射：

- BPE：Lecture 1 与 A1 handout pp.3–12；
- decoder-only、RMSNorm、SwiGLU、RoPE、causal attention：Lecture 3 与 A1 handout pp.13–26；
- temperature、top-p、EOS：A1 handout pp.37–42；
- prefill、decode、KV cache：Lecture 10；
- perplexity 定义：A1 handout p.29；语义与局限：Lecture 12。

课程边界：

- repetition penalty 没有被核验材料直接系统讲授；
- teacher forcing 的机制存在于 next-token objective，但该术语没有被核验材料单独展开；
- greedy decoding 没有作为独立课程术语重点定义；
- 不把课程默认选择写成“理论最优”或“唯一正确”。

### nanochat

固定上游版本：`karpathy/nanochat@92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`，核验日期 2026-07-22。

主要源码映射：

- tokenizer：`nanochat/tokenizer.py`、`scripts/tok_train.py`、`scripts/tok_eval.py`；
- decoder、RoPE、RMSNorm、causal attention：`nanochat/gpt.py`；
- prefill、decode、KV cache、greedy / temperature / top-k、EOS 状态：`nanochat/engine.py`；
- shifted targets：`nanochat/core_eval.py` 与 `scripts/chat_sft.py`；
- tokenizer-invariant 评测：`nanochat/loss_eval.py` 的 bits-per-byte。

必须显式呈现的差异：

- nanochat 使用 `ReLU²`，不是 SwiGLU；
- nanochat engine 当前实现 top-k，而不是 top-p；
- nanochat 没有通用 repetition penalty；
- nanochat 主要报告 bits-per-byte，不把 perplexity 作为唯一评测口径；
- nanochat 用 `<|assistant_end|>` 或 BOS 结束生成，而不是把所有停止逻辑都抽象成单一通用 EOS。

## 7. 视觉与导航

视觉方向为“可操作的工程讲义”：

- 暖色纸张背景、深墨正文、深绿机制强调、赭红警示与引用强调；
- 不使用通用渐变 hero、资源卡片墙或装饰性插图；
- 保留 `paper-header`、`chapter`、公式、算法面板和总结表格；
- 桌面端固定左侧目录，并留出真实 gutter，确保 `toc.right <= lecture.left`；
- 移动端目录折叠为按钮与章节进度，不阻塞正文；
- 顶部阅读进度条；
- 动态实验使用深色“实验台”表面，与正文纸张形成工作区层级；
- 代码块提供复制状态、文件链接和预期输出；
- source note 固定采用轻量边注 / 行内引用，不把页面变成参考文献卡片集合。

## 8. 文件与集成

预计新增或修改：

```text
_data/foundations.yml
foundations/llm-mechanics/index.html
foundations/llm-mechanics/static/css/index.css
foundations/llm-mechanics/static/js/index.js
foundations/llm-mechanics/examples/*
scripts/verify_llm_mechanics.py
docs/superpowers/specs/2026-07-22-llm-mechanics-foundations-design.md
```

MathJax 复用仓库内已有本地 vendor，不引入外部运行时依赖。页面内容、CSS 和 JavaScript 均局部命名，避免污染其他 Foundations 页面。

## 9. 错误处理与降级

- JavaScript 关闭时，正文、公式、静态矩阵、代码和来源仍完整可读；
- 实验初始化失败时显示局部错误提示，不影响其余章节；
- 滑块与数值输入统一 clamp，并处理 `NaN`、零长度序列和空候选集；
- top-p 至少保留概率最高 token；
- temperature 为零时显式走 greedy，不做除零；
- EOS 与 max tokens 同时存在时，页面解释哪个条件先触发；
- code copy 在 Clipboard API 不可用时退化为选中文本；
- canvas 按 device pixel ratio 绘制，resize 后重新布局。

## 10. 验收标准

内容：

- 12 个概念全部出现且定义、机制、实践、边界完整；
- 5 章主线连续，没有词条式割裂；
- 6 个实验可操作并给出数值解释；
- 12 个示例可独立运行，`run_all.py` 全部通过；
- Stanford 与 nanochat 的差异明确，不把缺失内容伪装成官方实现；
- 不出现显式受众分层标签。

工程：

- Jekyll / Docker 构建通过；
- verifier 检查 section、lab、example、source link 和禁用词；
- 桌面与移动端无横向溢出；
- 桌面目录不覆盖正文；
- 所有实验无控制台错误；
- 关键控件可键盘操作；
- 页面已加入 `_data/foundations.yml` 并能从 Foundations 列表进入；
- 浏览器截图覆盖桌面首屏、至少三个关键章节、移动端目录与交互状态。

## 11. 非目标

- 不训练真实语言模型；
- 不下载模型权重或数据集；
- 不在浏览器内运行完整 PyTorch；
- 不复刻 nanochat speedrun；
- 不扩展到 attention 优化、FlashAttention 内核、GQA/MQA/MLA、distributed training、SFT、RL 或 serving scheduler；
- 不把动态效果做成与机制无关的装饰。
