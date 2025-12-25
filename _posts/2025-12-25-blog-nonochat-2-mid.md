---
title: 'noao-chat-2-mid阶段训练'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-25-blog-nonochat-2-mid/
tags:
  - llm
---


### nonochat - LLM Mid 训练完整解析

[nonochat](https://github.com/karpathy/nanochat)


> 本文档详细解析 `mid_train.py`，讲解中间训练（Midtraining）的作用和实现细节。  
> 适合 LLM 领域的初学者，从 Base 模型到应用模型的关键过渡阶段。

---

## 目录

1. [什么是 Mid 训练](#1-什么是-mid-训练)
2. [数据来源的巨大变化](#2-数据来源的巨大变化)
3. [训练流程详解](#3-训练流程详解)
4. [与 Base 训练的对比](#4-与-base-训练的对比)
5. [实战案例分析](#5-实战案例分析)

---

## 1. 什么是 Mid 训练

### 1.1 训练阶段定位

```
Base Training → Mid Training → SFT → RL
     ↓              ↓            ↓     ↓
  通用能力      领域能力      对话   对齐
  (海量文本)  (结构化任务)   (指令)  (偏好)
```

**Mid Training（中间训练）** 是连接预训练和微调的桥梁阶段。

### 1.2 为什么需要 Mid Training？

**Base 模型的局限：**
- ✅ 有通用语言理解能力
- ❌ 不懂对话格式
- ❌ 不会使用工具
- ❌ 缺乏特定领域知识
- ❌ 在某些任务上表现不佳

**直接 SFT 的问题：**
- SFT 数据量通常很少（几千到几万条）
- 如果 Base 模型在某个能力上很弱，SFT 很难补救
- 某些基础能力需要更多数据才能学会

**Mid Training 的解决方案：**
- 🎯 使用**结构化对话数据**继续训练
- 🎯 注入**特定领域知识**（数学、对话、工具使用）
- 🎯 补齐 Base 模型的**能力短板**
- 🎯 为后续 SFT 打下良好基础

### 1.3 Mid Training 的核心目标

```python
# 训练数据组成（总共 ~850K 条对话）
train_dataset = TaskMixture([
    SmolTalk(split="train"),              # 460K 通用对话
    MMLU(subset="auxiliary_train"),        # 100K 知识问答
    GSM8K(subset="main", split="train"),   # 8K 数学推理
    CustomJSON(identity_file),             # 1K 身份对话 × 2 轮
    SimpleSpelling(size=200000),           # 200K 拼写
    SpellingBee(size=80000),               # 80K 字母计数
])
```

**关键能力注入：**

| 能力类型 | 数据来源 | 数量 | 目的 |
|---------|---------|------|------|
| 对话能力 | SmolTalk | 460K | 学会自然对话 |
| 知识储备 | MMLU | 100K | 多领域知识 |
| 数学推理 | GSM8K | 8K | 工具使用 + 推理 |
| 身份设定 | 自定义 | 1K×2 | 模型人格 |
| 拼写能力 | Spelling | 280K | token→字符映射 |

---

## 2. 数据来源的巨大变化

### 2.1 Base vs Mid 数据对比

#### Base Training 数据格式

```
原始文本（Parquet 文件）:
"The quick brown fox jumps over the lazy dog. This is a test..."

Tokenize → 连续的 token 流:
[15496, 995, 831, 374, 264, 1296, ...]

分批次:
inputs:  [15496, 995, 831, 374, ...]
targets: [995, 831, 374, 264, ...]
```

**特点：**
- 📄 纯文本，没有结构
- 🔄 文档边界被 `<|bos|>` 分隔
- 🎲 随机拼接文档

#### Mid Training 数据格式

```
结构化对话（来自任务数据集）:
{
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}

Tokenize → 带格式的 token 序列:
[<|bos|>, <|user_start|>, "What", "is", "2", "+", "2", "?", <|user_end|>, 
 <|assistant_start|>, "4", <|assistant_end|>]

分批次（保留对话结构）:
# 多个完整对话拼接在一起
```

**特点：**
- 💬 结构化对话
- 🏷️ 明确的角色标记
- 📚 来自精心策划的任务

### 2.2 任务数据集详解

#### 2.2.1 SmolTalk - 通用对话

**来源：** HuggingFace SmolTalk 数据集

**格式示例：**

```json
{
    "messages": [
        {"role": "user", "content": "Can you explain quantum computing?"},
        {"role": "assistant", "content": "Quantum computing uses quantum bits..."},
        {"role": "user", "content": "What are its applications?"},
        {"role": "assistant", "content": "Quantum computers can..."}
    ]
}
```

**特点：**
- 多轮对话
- 涵盖各种日常话题
- 自然的对话风格
- 训练集：460K 条，测试集：24K 条

**作用：** 让模型学会自然的多轮对话模式

#### 2.2.2 MMLU - 知识问答

**来源：** MMLU (Massive Multitask Language Understanding)

**格式示例：**

```json
{
    "messages": [
        {
            "role": "user", 
            "content": "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid"
        },
        {"role": "assistant", "content": "B"}
    ],
    "subject": "geography",
    "letters": ["A", "B", "C", "D"]
}
```

**特点：**
- 选择题格式
- 57 个学科领域
- 辅助训练集：100K 条

**学科分布：**
```
STEM: 数学、物理、化学、生物、计算机...
人文: 历史、哲学、法律、心理学...
社科: 经济学、社会学、政治学...
其他: 医学、商业、艺术...
```

**作用：** 注入大量结构化知识

#### 2.2.3 GSM8K - 数学推理

**来源：** GSM8K (Grade School Math 8K)

**格式示例：**

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Weng earns $12 an hour. Yesterday, she worked for 50 minutes. How much did she earn?"
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First, let's calculate per minute wage:\n"},
                {"type": "python", "text": "12/60"},
                {"type": "python_output", "text": "0.2"},
                {"type": "text", "text": "\nSo she earns $0.2 per minute. For 50 minutes:\n"},
                {"type": "python", "text": "0.2*50"},
                {"type": "python_output", "text": "10"},
                {"type": "text", "text": "\n#### 10"}
            ]
        }
    ]
}
```

**关键特性：**
- 🧮 包含工具调用（Python 计算器）
- 📝 步骤化推理
- ✅ 最终答案标记 `#### 10`

**Token 化后的格式：**

```
<|bos|>
<|user_start|> Weng earns $12 an hour... <|user_end|>
<|assistant_start|> 
    First, let's calculate per minute wage:
    <|python_start|> 12/60 <|python_end|>
    <|output_start|> 0.2 <|output_end|>
    So she earns $0.2 per minute. For 50 minutes:
    <|python_start|> 0.2*50 <|python_end|>
    <|output_start|> 10 <|output_end|>
    #### 10
<|assistant_end|>
```

**作用：** 
- 教会模型使用工具
- 学习步骤化推理
- 理解计算流程

#### 2.2.4 Spelling Tasks - 拼写能力

**为什么需要？**

LLM 的一个常见弱点：不知道 token 是如何拼写的。

```
问题示例：
"How many 'r' are in 'strawberry'?"

Base 模型可能回答: "2"  ❌
正确答案: "3"  ✅
```

**原因分析：**

```
Token 化:
"strawberry" → [str, aw, berry]  # 可能被切成多个 token

模型看到的是 token IDs，不是字符:
[24871, 675, 15717]

要回答"有几个 r"，模型需要知道:
- token 24871 包含哪些字母？
- token 675 包含哪些字母？
- token 15717 包含哪些字母？
```

这种 **token → 字符** 的映射需要专门训练！

**解决方案：SimpleSpelling**

```json
{
    "messages": [
        {"role": "user", "content": "Spell the word 'apple'"},
        {"role": "assistant", "content": "a-p-p-l-e"}
    ]
}
```

- 200K 个单词拼写任务
- 让模型学会 token 的字符组成

**解决方案：SpellingBee**

```json
{
    "messages": [
        {"role": "user", "content": "How many r are in strawberry"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me spell it:\n"},
                {"type": "python", "text": "'strawberry'.lower()"},
                {"type": "python_output", "text": "strawberry"},
                {"type": "text", "text": "\nNow count:\n"},
                {"type": "python", "text": "'strawberry'.count('r')"},
                {"type": "python_output", "text": "3"},
                {"type": "text", "text": "\n#### 3"}
            ]
        }
    ]
}
```

- 80K 个字母计数任务
- 结合拼写和工具使用

#### 2.2.5 Identity Conversations - 身份设定

**格式示例：**

```json
{
    "messages": [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am an AI assistant created by..."}
    ]
}
```

**作用：**
- 让模型知道自己是谁
- 统一回答身份问题
- 增强一致性

### 2.3 数据加载器对比

#### Base Training 数据加载

```python
def tokenizing_distributed_data_loader(B, T, split):
    # 读取 Parquet 文件
    pf = pq.ParquetFile(filepath)
    batch = rg.column('text').to_pylist()
    
    # Tokenize
    token_lists = tokenizer.encode(batch, prepend=bos_token)
    
    # 拼接成连续流
    for tokens in token_lists:
        token_buffer.extend(tokens)
    
    # 取出固定长度
    needed = B * T + 1
    tokens = [token_buffer.popleft() for _ in range(needed)]
    
    # 切分
    inputs = tokens[:-1].view(B, T)
    targets = tokens[1:].view(B, T)
```

#### Mid Training 数据加载

```python
def mid_data_generator(split):
    dataset = train_dataset  # TaskMixture
    
    # 遍历数据集
    for cursor in range(ddp_rank, len(dataset), ddp_world_size):
        # 获取一个对话
        conversation = dataset[cursor]
        
        # Tokenize（保留结构）
        ids, mask = tokenizer.render_conversation(conversation)
        # ids: [<|bos|>, <|user_start|>, ..., <|assistant_end|>]
        
        # 累积到缓冲区
        token_buffer.extend(ids)
        
        # 当缓冲区足够时，切分批次
        if len(token_buffer) >= needed_tokens:
            tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
            inputs = tokens[:-1].view(B, T)
            targets = tokens[1:].view(B, T)
            yield inputs, targets
```

**关键差异：**

| 维度 | Base Training | Mid Training |
|-----|---------------|--------------|
| 数据源 | Parquet 文件（纯文本） | Task 对象（结构化对话） |
| 分词方式 | `encode(text)` | `render_conversation(conv)` |
| 结构 | 无结构 | 带角色标记 |
| 边界 | 只有 `<|bos|>` | 完整的对话标记 |
| 对话完整性 | 文档可能被截断 | 尽量保持对话完整 |

---

## 3. 训练流程详解

### 3.1 完整数据流（图解）

```
步骤 1: 加载任务数据
┌─────────────────────────────────────┐
│ SmolTalk: 460K 对话                  │
│ MMLU: 100K 问答                      │
│ GSM8K: 8K 数学题                     │
│ Spelling: 280K 拼写任务              │
│ Identity: 1K 身份对话                │
└─────────────────────────────────────┘
            ↓
步骤 2: TaskMixture 混合
┌─────────────────────────────────────┐
│ 所有任务混合打乱                      │
│ 总计: ~850K 条对话                   │
│ 确定性随机打乱 (seed=42)              │
└─────────────────────────────────────┘
            ↓
步骤 3: 按索引遍历
┌─────────────────────────────────────┐
│ 每个 GPU 负责不同的子集                │
│ GPU 0: indices [0, 8, 16, 24, ...]   │
│ GPU 1: indices [1, 9, 17, 25, ...]   │
│ ...                                  │
└─────────────────────────────────────┘
            ↓
步骤 4: 获取对话
┌─────────────────────────────────────┐
│ conversation = dataset[cursor]       │
│ {                                    │
│   "messages": [                      │
│     {"role": "user", "content": ...} │
│     {"role": "assistant", ...}       │
│   ]                                  │
│ }                                    │
└─────────────────────────────────────┘
            ↓
步骤 5: Tokenize（保留结构）
┌─────────────────────────────────────┐
│ ids, mask = tokenizer.render_conversation(conv) │
│                                      │
│ ids: [<|bos|>, <|user_start|>, ...] │
│ mask: [0, 0, 0, 1, 1, ...]  # 训练掩码 │
└─────────────────────────────────────┘
            ↓
步骤 6: 累积到缓冲区
┌─────────────────────────────────────┐
│ token_buffer.extend(ids)             │
│ [1, 2, 3, 4, ..., 10000, 10001, ...] │
│ （多个对话连接在一起）                  │
└─────────────────────────────────────┘
            ↓
步骤 7: 切分批次
┌─────────────────────────────────────┐
│ needed = B * T + 1  # 32*2048+1      │
│ tokens = buffer[:needed]             │
│                                      │
│ inputs:  (B, T) = (32, 2048)         │
│ targets: (B, T) = (32, 2048)         │
└─────────────────────────────────────┘
            ↓
步骤 8: 送入模型训练
┌─────────────────────────────────────┐
│ loss = model(inputs, targets)        │
│ loss.backward()                      │
│ optimizer.step()                     │
└─────────────────────────────────────┘
```

### 3.2 TokenMixture 的作用

**问题：** 不同任务的数据量差异巨大

```
SmolTalk: 460K  ████████████████████████
MMLU:     100K  █████
GSM8K:      8K  ▌
Identity:   2K  ▏
```

**简单遍历的问题：**
```python
# 错误做法：依次训练每个任务
for task in tasks:
    for example in task:
        train(example)

# 结果：前期都在训练 SmolTalk，后期才见到 GSM8K
# 模型会遗忘早期学到的东西！
```

**TaskMixture 的解决方案：**

```python
class TaskMixture:
    def __init__(self, tasks):
        # 1. 构建索引映射
        self.index_map = []
        for task_idx, task in enumerate(tasks):
            for local_idx in range(len(task)):
                self.index_map.append((task_idx, local_idx))
        
        # 2. 打乱索引（确定性）
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        
        # 现在访问顺序是：
        # [SmolTalk_123, MMLU_45, GSM8K_7, SmolTalk_456, ...]
```

**效果：**
```
训练过程中看到的数据（混合后）：
[S, S, M, S, S, G, S, M, S, I, S, S, M, S, G, ...]
 ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
 SmolTalk, MMLU, GSM8K, Identity 混合出现

而不是：
[S, S, S, S, ..., M, M, M, ..., G, G, ...]
 ←── SmolTalk ───→  ←MMLU→  ←GSM8K→
```

**优点：**
- ✅ 所有任务在整个训练过程中都有曝光
- ✅ 避免灾难性遗忘
- ✅ 不同能力同步提升

### 3.3 训练循环详解

#### 3.3.1 进度追踪机制

Mid Training 的一个特殊之处：**我们不提前知道要训练多少步**

```python
# Base Training: 明确的迭代次数
num_iterations = 10000  # 确定
for step in range(num_iterations):
    ...

# Mid Training: 基于数据集大小
dataset_size = 850000  # 对话数量
# 每个 step 消耗多少对话？不确定！（因为对话长度不一）
# 所以用进度百分比
```

**进度计算：**

```python
# 全局变量
last_step = False       # 是否到达最后一步
approx_progress = 0.0   # 0 → 1

# 在数据生成器中更新
def mid_data_generator(split):
    cursor = ddp_rank
    it = 0
    
    while True:
        # 处理一个对话
        conversation = dataset[cursor]
        cursor += ddp_world_size
        
        # 更新进度
        if cursor >= dataset_size:
            cursor -= dataset_size  # 回绕（下一个 epoch）
            if split == "train":
                last_step = True  # 标记为最后一步
        
        # 计算近似进度
        approx_progress = cursor / dataset_size
        
        yield inputs, targets
```

**多卡同步：**

```python
# 问题：不同 GPU 可能处理速度不同
# GPU 0: last_step = True
# GPU 1: last_step = False  ← 还没完成

# 解决：同步 last_step
if ddp:
    last_step_tensor = torch.tensor(last_step, device=device)
    dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
    # 只要有一个 GPU 到达终点，所有 GPU 都停止
    last_step = bool(last_step_tensor.item())
```

#### 3.3.2 学习率调度

**与 Base Training 不同：**

```python
# Base Training: 基于步数
def get_lr_multiplier(step):
    if step < warmup_iters:
        return step / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        return linear_decay(...)

# Mid Training: 基于进度百分比
def get_lr_multiplier(progress):
    # progress: 0.0 → 1.0
    if progress < 0.8:
        return 1.0  # 前 80% 保持不变
    else:
        # 后 20% 线性衰减到 0
        return 1 - (progress - 0.8) / 0.2
```

**学习率曲线：**

```
LR Multiplier
    1.0 |════════════════════════════════╲
        |                                 ╲
        |                                  ╲
    0.5 |                                   ╲
        |                                    ╲
    0.0 |                                     ═
        +───────────────────────────────────────→
        0%      20%     40%     60%     80%   100% progress
        ←────────── 保持 1.0 ──────────→ ← 衰减 →
```

**为什么不同？**
- Base Training 知道总步数，可以精确规划
- Mid Training 不确定总步数，用进度百分比更灵活

#### 3.3.3 单步训练流程

```python
# 预取第一批数据
x, y = next(train_loader)

while True:
    # === 评估阶段 ===
    if step % eval_every == 0:
        model.eval()
        val_bpb = evaluate_bpb(...)
        model.train()
    
    # === 保存检查点 ===
    if last_step:
        save_checkpoint(...)
        break
    
    # === 训练阶段 ===
    synchronize()
    t0 = time.time()
    
    # 梯度累积
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        
        loss = loss / grad_accum_steps
        loss.backward()
        
        # 异步预取下一批
        x, y = next(train_loader)
        progress = max(progress, approx_progress)
    
    # 更新学习率
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    # 更新参数
    for opt in optimizers:
        opt.step()
    model.zero_grad()
    
    synchronize()
    t1 = time.time()
    
    # === 日志记录 ===
    step += 1
    dt = t1 - t0
    print(f"step {step} ({progress*100:.2f}%) | loss: {loss:.6f} ...")
```

### 3.4 维度分析

假设配置：`B=32, T=2048, vocab_size=32000, n_embd=768`

#### 单个对话的 Token 化

```python
# 输入：结构化对话
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}

# 输出：token IDs + mask
ids, mask = tokenizer.render_conversation(conversation)

# ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# mask: [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]

# 展开看：
# token_id | token_str           | mask | 说明
# ---------|---------------------|------|------
# 1        | <|bos|>             | 0    | 文档开始
# 2        | <|user_start|>      | 0    | 用户开始
# 3-7      | "What is 2+2?"      | 0    | 用户消息（不训练）
# 8        | <|user_end|>        | 0    | 用户结束
# 9        | <|assistant_start|> | 0    | 助手开始
# 10-11    | "4"                 | 1    | 助手回复（训练！）
# 12       | <|assistant_end|>   | 1    | 助手结束（训练）
```

#### 批次维度

```python
# 多个对话拼接
conversation_1: 200 tokens
conversation_2: 150 tokens
conversation_3: 180 tokens
...

# 累积到缓冲区
token_buffer: [tok1, tok2, ..., tok_N]  # N 很大

# 切分批次
needed = 32 * 2048 + 1 = 65537

tokens = token_buffer[:65537]
# [tok_1, tok_2, ..., tok_65537]

# 重塑
inputs = tokens[:-1].view(32, 2048)   # (32, 2048)
targets = tokens[1:].view(32, 2048)   # (32, 2048)
```

#### 模型前向传播

```python
# 输入
inputs: (32, 2048)  # int32

# Embedding
x = model.transformer.wte(inputs)
# (32, 2048) → (32, 2048, 768)

# Transformer Blocks
for block in model.transformer.h:
    x = block(x, ...)
# (32, 2048, 768) → (32, 2048, 768)

# LM Head
logits = model.lm_head(x)
# (32, 2048, 768) @ (768, 32000) → (32, 2048, 32000)

# 损失计算
loss = F.cross_entropy(
    logits.view(-1, 32000),   # (65536, 32000)
    targets.view(-1)          # (65536,)
)
# loss: 标量
```

---

## 4. 与 Base 训练的对比

### 4.1 核心差异总结

| 维度 | Base Training | Mid Training |
|-----|---------------|--------------|
| **数据来源** | Parquet 文件（网页抓取） | 任务数据集（精心策划） |
| **数据格式** | 纯文本 | 结构化对话 |
| **数据量** | 数十亿 tokens | 数百万对话 |
| **训练目标** | 通用语言理解 | 特定能力注入 |
| **Token 格式** | 连续文本流 | 带角色标记 |
| **进度追踪** | 固定步数 | 数据集进度 |
| **学习率** | 基于步数调度 | 基于进度百分比 |
| **训练时长** | 数天到数周 | 数小时到1天 |
| **检查点位置** | `base_checkpoints/` | `mid_checkpoints/` |

### 4.2 代码层面对比

#### 数据加载器

```python
# ===== Base Training =====
# 从 Parquet 读取
pf = pq.ParquetFile(filepath)
batch = rg.column('text').to_pylist()

# 简单 tokenize
token_lists = tokenizer.encode(batch, prepend=bos_token)

# 无结构，直接拼接
for tokens in token_lists:
    token_buffer.extend(tokens)

# ===== Mid Training =====
# 从 Task 对象读取
conversation = dataset[cursor]

# 结构化 tokenize
ids, mask = tokenizer.render_conversation(conversation)
# 包含 <|user_start|>, <|assistant_start|> 等特殊 token

# 保留对话结构
token_buffer.extend(ids)
```

#### 模型初始化

```python
# ===== Base Training =====
# 从零开始
with torch.device("meta"):
    model_config = GPTConfig(...)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()  # 随机初始化

# ===== Mid Training =====
# 从 Base 模型加载
model, tokenizer, meta = load_model(
    "base",  # 加载 base 模型
    device,
    phase="train",
    model_tag=model_tag,
    step=step
)
# 参数已经训练过，继续训练
```

#### 训练循环

```python
# ===== Base Training =====
for step in range(num_iterations):  # 固定次数
    loss = model(x, y)
    loss.backward()
    optimizer.step()

# ===== Mid Training =====
while True:  # 直到遍历完数据集
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    if last_step:  # 数据集遍历完
        break
```

### 4.3 超参数对比

```python
# ===== Base Training =====
base_config = {
    "total_batch_size": 524288,      # 512K tokens/step
    "device_batch_size": 32,
    "num_iterations": 50000,         # 明确指定
    "warmup_ratio": 0.0,             # 无 warmup
    "warmdown_ratio": 0.2,           # 最后 20% 衰减
    "embedding_lr": 0.2,
    "matrix_lr": 0.02,
    "unembedding_lr": 0.004,
    "init_lr_frac": 1.0,             # 从 100% 开始
}

# ===== Mid Training =====
mid_config = {
    "total_batch_size": 524288,      # 512K tokens/step (相同)
    "device_batch_size": 32,
    "num_iterations": -1,            # 自动推断
    "warmup_ratio": 0.0,             # 无 warmup
    "warmdown_ratio": 0.2,           # 最后 20% 衰减
    "embedding_lr": 0.2,
    "matrix_lr": 0.02,
    "unembedding_lr": 0.004,
    "init_lr_frac": 1.0,             # 从 100% 开始
}
```

**相同点：**
- 批次大小相同
- 学习率相同
- 优化器配置相同

**不同点：**
- Mid Training 从已训练的模型开始
- 数据源完全不同
- 训练目标不同

---

## 5. 实战案例分析

### 5.1 完整训练流程示例

假设我们训练一个 `depth=12` 的模型（约 200M 参数）。

#### 步骤 1: Base Training

```bash
# 8 卡 A100，训练 3 天
torchrun --nproc_per_node=8 -m scripts.base_train \
    --depth=12 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --num_iterations=50000 \
    --run=base_d12

# 输出：
# base_checkpoints/d12/step_00050000.pt
```

**结果：**
- ✅ 模型学会了通用语言理解
- ✅ 能够补全句子
- ❌ 不懂对话格式
- ❌ 不会使用工具

#### 步骤 2: Mid Training

```bash
# 8 卡 A100，训练 6 小时
torchrun --nproc_per_node=8 -m scripts.mid_train \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --run=mid_d12

# 输出：
# mid_checkpoints/d12/step_XXXXX.pt
```

**结果：**
- ✅ 理解对话格式（user/assistant）
- ✅ 学会了基本对话能力
- ✅ 能使用 Python 计算器
- ✅ 拼写能力显著提升
- ✅ 多领域知识增强
- ❌ 还不能很好地遵循指令

### 5.2 训练数据流转示例

让我们追踪一个实际的训练 batch：

#### Batch 构成（B=4, T=512 为例）

```python
# 假设 token_buffer 中有以下对话（简化）：

# 对话 1: SmolTalk（200 tokens）
[<|bos|>, <|user_start|>, "How", "are", "you", "?", <|user_end|>,
 <|assistant_start|>, "I'm", "doing", "well", "!", <|assistant_end|>, ...]

# 对话 2: MMLU（50 tokens）
[<|bos|>, <|user_start|>, "What", "is", "2+2", "?", "\n", "A.", "2", ...,
 <|user_end|>, <|assistant_start|>, "D", <|assistant_end|>]

# 对话 3: GSM8K（300 tokens）
[<|bos|>, <|user_start|>, "Weng", "earns", ..., <|user_end|>,
 <|assistant_start|>, "First", ",", "let's", ..., <|python_start|>, "12/60", ...]

# 对话 4: SpellingBee（100 tokens）
[<|bos|>, <|user_start|>, "How", "many", "r", "in", "strawberry", <|user_end|>,
 <|assistant_start|>, <|python_start|>, "'strawberry'.count('r')", ...]

# 从 buffer 中取出 4*512+1 = 2049 个 token
# 这些 token 来自上述对话的混合
```

#### 切分成 inputs 和 targets

```python
tokens = token_buffer[:2049]

inputs = tokens[:-1]   # 前 2048 个
targets = tokens[1:]   # 后 2048 个

# 重塑
inputs = inputs.view(4, 512)
targets = targets.view(4, 512)

# 现在每个样本包含多个对话的片段
# 样本 0: 对话1的一部分 + 对话2的开头
# 样本 1: 对话2的剩余 + 对话3的一部分
# 样本 2: 对话3的中间部分
# 样本 3: 对话3的结尾 + 对话4
```

#### 模型训练

```python
# 前向传播
logits = model(inputs)  # (4, 512, 32000)

# 计算损失
loss = F.cross_entropy(
    logits.view(-1, 32000),
    targets.view(-1)
)

# 反向传播
loss.backward()

# 更新参数
optimizer.step()
```

### 5.3 训练日志解读

```
step 00100 (12.50%) | loss: 2.345678 | lrm: 1.00 | dt: 245.32ms | 
tok/sec: 2,137,856 | mfu: 45.67 | total time: 8.52m

step 00200 (25.00%) | loss: 2.123456 | lrm: 1.00 | dt: 243.11ms | 
tok/sec: 2,156,234 | mfu: 46.12 | total time: 17.21m

...

step 01600 (80.00%) | loss: 1.876543 | lrm: 1.00 | dt: 242.55ms | 
tok/sec: 2,161,345 | mfu: 46.23 | total time: 137.45m

step 01700 (85.00%) | loss: 1.865432 | lrm: 0.75 | dt: 242.88ms | 
tok/sec: 2,159,876 | mfu: 46.19 | total time: 145.78m
                                      ^^^^
                                      学习率开始衰减

step 02000 (100.00%) | loss: 1.854321 | lrm: 0.00 | dt: 243.21ms | 
tok/sec: 2,157,234 | mfu: 46.14 | total time: 170.12m
```

**观察：**
- 进度从 0% → 100%
- Loss 从 2.3 → 1.8（显著下降）
- 在 80% 之后，lrm 从 1.0 开始衰减
- 总训练时间约 2.8 小时（8 卡）

### 5.4 能力提升对比

**测试任务：**

```python
prompts = [
    # 1. 对话能力
    "Hello! How are you?",
    
    # 2. 知识问答
    "What is the capital of France?",
    
    # 3. 数学推理
    "If I have 5 apples and buy 3 more, how many do I have?",
    
    # 4. 拼写能力
    "How many 'r' are in 'strawberry'?",
]
```

**模型对比：**

| 测试 | Base 模型 | Mid 模型 |
|-----|----------|---------|
| 对话 | "I are you" (语法错误) | "I'm doing well, thanks! How about you?" ✅ |
| 知识 | "Paris France city" (不连贯) | "Paris" ✅ |
| 数学 | "8 apples total" (直接猜) | "5 + 3 = 8" ✅ (使用计算) |
| 拼写 | "2" ❌ | "3" ✅ |

---

## 6. 关键要点总结

### 6.1 Mid Training 的本质

```
Mid Training = Base Model + Structured Tasks
             = 通用能力 + 特定技能
             = 为 SFT 做准备
```

### 6.2 何时需要 Mid Training？

**需要 Mid Training：**
- ✅ Base 模型在某些能力上很弱（如拼写、工具使用）
- ✅ 有大量结构化任务数据
- ✅ 想要注入特定领域知识
- ✅ 数据量够大（数十万级别）

**可以跳过 Mid Training：**
- ❌ Base 模型已经足够强
- ❌ 只有少量 SFT 数据
- ❌ 资源有限

### 6.3 Mid Training vs SFT

| 维度 | Mid Training | SFT |
|-----|-------------|-----|
| 数据量 | 大（数十万） | 小（数千） |
| 目标 | 能力扩展 | 指令对齐 |
| 学习率 | 较高 | 较低 |
| 训练轮数 | 1 epoch | 1-3 epochs |
| 数据来源 | 公开数据集 | 精心标注 |

### 6.4 实践建议

**数据配比：**
```python
# 推荐配比
train_dataset = TaskMixture([
    SmolTalk(...),          # 50-60% 通用对话
    Knowledge_QA(...),      # 15-20% 知识问答
    Math_Reasoning(...),    # 1-5% 数学推理
    Tool_Use(...),          # 1-5% 工具使用
    Spelling(...),          # 20-30% 拼写/基础能力
    Identity(...),          # <1% 身份设定（过采样）
])
```

**训练时长：**
- 小模型（<500M）：1-2 小时（8卡）
- 中模型（500M-2B）：4-8 小时（8卡）
- 大模型（>2B）：1-2 天（8卡）

**检查点保存：**
- 只保存最后的检查点
- Mid Training 通常不需要中间检查点

---

## 附录

### A. 完整训练命令

```bash
# 单 GPU 训练
python -m scripts.mid_train \
    --device_batch_size=16 \
    --run=mid_test

# 多 GPU 训练
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --run=mid_production
```

### B. 任务数据集详细统计

| 任务 | 训练集 | 测试集 | 平均长度 | 用途 |
|-----|-------|-------|----------|------|
| SmolTalk | 460K | 24K | ~150 tokens | 对话能力 |
| MMLU (aux) | 100K | 14K | ~100 tokens | 知识问答 |
| GSM8K | 8K | 1.3K | ~250 tokens | 数学推理 |
| SimpleSpelling | 200K | - | ~30 tokens | 拼写基础 |
| SpellingBee | 80K | - | ~100 tokens | 字母计数 |
| Identity | 1K | - | ~80 tokens | 身份设定 |

### C. 常见问题

**Q1: Mid Training 必须做吗？**
- 不是必须，但强烈推荐
- 特别是模型较小（<1B）时

**Q2: 可以做多轮 Mid Training 吗？**
- 可以，但通常 1 轮就够
- 更多轮可能过拟合

**Q3: 如何选择任务？**
- 根据应用场景选择
- 保证数据质量
- 注意任务间的平衡

**Q4: Mid Training 后 loss 应该是多少？**
- 通常比 Base Training 略低
- 典型值：1.5 - 2.0
- 重要的是相对下降，不是绝对值

---

## 9. 关键疑问解答：为什么 Mid 训练丢弃了 Mask？

### 9.1 Mask 机制的本质

在 tokenizer 中，`render_conversation` 函数会返回两个值：

```python
ids, mask = tokenizer.render_conversation(conversation)
# ids: [token_id1, token_id2, ...]  完整的 token 序列
# mask: [0, 0, 1, 1, 0, ...]        标记哪些 token 需要训练
```

**Mask 的含义**：
- `mask = 0`：不计算 loss（如 user 的输入、特殊 token）
- `mask = 1`：计算 loss（如 assistant 的回复）

**Mask 的生成逻辑**（来自 tokenizer.py）：

```python
def render_conversation(self, conversation):
    ids, mask = [], []
    
    def add_tokens(token_ids, mask_val):
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
    
    # BOS token: mask=0
    add_tokens(bos, 0)
    
    for message in messages:
        if message["role"] == "user":
            # User 的所有内容：mask=0
            add_tokens(user_start, 0)
            add_tokens(content_ids, 0)  # ← User 输入不训练
            add_tokens(user_end, 0)
        
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)  # Start token 不训练
            add_tokens(content_ids, 1)      # ← Assistant 回复训练！
            add_tokens(assistant_end, 1)    # End token 也训练
    
    return ids, mask
```

### 9.2 Mid Training 丢弃 Mask 的真相

**代码对比**：

```python
# Mid Training (mid_train.py 第 133 行)
ids, _ = tokenizer.render_conversation(conversation)
#      ↑ 直接丢弃 mask！

# SFT Training (chat_sft.py 第 123 行)
ids, mask = tokenizer.render_conversation(doc)
#      ↑ 保留并使用 mask
```

**为什么 Mid Training 丢弃 mask？**

答案：**Mid Training 在所有 token 上计算 loss，包括 User 的输入！**

### 9.3 Mid vs SFT：训练目标的根本区别

#### Mid Training 的目标：学习对话流

```python
# Mid Training 的数据处理
conversation = dataset[cursor]
ids, _ = tokenizer.render_conversation(conversation)
token_buffer.extend(ids)  # 直接拼接所有 token

# 生成 inputs, targets（没有 mask）
inputs = tokens[:-1]   # (B, T)
targets = tokens[1:]   # (B, T)

# 计算 loss（所有 token 都参与）
loss = model(inputs, targets)
# CrossEntropyLoss 会在所有 token 上计算梯度
```

**训练的内容**：
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>4<|assistant_end|>
↑      ↑                              ↑          ↑                    ↑
所有这些 token 都参与 loss 计算！
```

**学习的能力**：
- ✅ 对话格式（什么时候该是 user，什么时候该是 assistant）
- ✅ 上下文理解（user 说了什么）
- ✅ 回复生成（assistant 应该怎么回）
- ✅ 整体对话流（对话的开始、进行、结束）

#### SFT Training 的目标：只学习生成回复

```python
# SFT Training 的数据处理
ids, mask = tokenizer.render_conversation(doc)

# 关键步骤：将 mask=0 的位置设为 -1
targets = ids[1:]
mask_tensor = mask[1:]
targets[mask_tensor == 0] = -1  # ← 设为 ignore_index

# 计算 loss（只在 mask=1 的 token 上）
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=-1  # ← targets=-1 的位置不计算 loss
)
```

**训练的内容**：
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>4<|assistant_end|>
  ↓         ↓                        ↓             ↓            ↓      ↓
 -1        -1                       -1            -1            4      END
                                                               ↑       ↑
                                            只有这两个 token 参与 loss！
```

**学习的能力**：
- ✅ 只学习生成回复（assistant 的内容）
- ❌ 不学习理解问题（user 的内容不参与梯度）
- ❌ 不学习对话格式（特殊 token 不参与梯度）

### 9.4 为什么要这样设计？

#### Mid Training：为什么要训练所有 token？

**原因 1：从零开始学习对话**

在 Base 训练后，模型只见过纯文本：
```
The capital of France is Paris. It is a beautiful city...
```

从未见过对话格式：
```
<|user_start|>What is the capital of France?<|user_end|>
<|assistant_start|>The capital of France is Paris.<|assistant_end|>
```

**如果在 Mid 阶段就使用 mask**（只训练 assistant）：
- ❌ 模型不知道 `<|user_start|>` 后面应该是什么
- ❌ 模型不知道 user 说完后应该接 `<|user_end|>`
- ❌ 模型不知道 user 结束后应该接 `<|assistant_start|>`

**使用所有 token 训练的好处**：
- ✅ 学会对话的完整流程
- ✅ 学会在正确的时机切换角色
- ✅ 学会理解 user 的输入（通过预测下一个 token）

**原因 2：建立上下文理解能力**

```python
# 训练目标：预测下一个 token
输入：<|user_start|>What is 2+
目标：2

输入：<|user_start|>What is 2+2
目标：?

输入：<|user_start|>What is 2+2?
目标：<|user_end|>
```

通过预测 user 输入的每一个 token，模型学会了：
- 理解问题的结构
- 识别问题何时结束
- 准备生成回复

**原因 3：多任务混合的需要**

Mid Training 的数据混合了多种任务：
```python
TaskMixture([
    SmolTalk,        # 通用对话
    MMLU,            # 多选题
    GSM8K,           # 数学推理
    SpellingBee,     # 字符级任务
])
```

每种任务的格式可能不同，模型需要学会：
- 识别当前是哪种任务
- 理解不同任务的输入格式
- 生成符合任务要求的输出

**如果只训练 assistant 的回复**：
- 模型可能学不会区分不同任务类型
- 模型可能不理解输入的具体格式要求

#### SFT Training：为什么要使用 Mask？

**原因 1：防止遗忘**

在 Mid Training 后，模型已经学会了：
- ✅ 对话格式
- ✅ 任务类型识别
- ✅ 基础推理能力

SFT 的目标不是重新学习这些，而是：
- 🎯 精细调整输出风格
- 🎯 提升指令遵循能力
- 🎯 优化输出格式

**如果 SFT 仍然训练所有 token**：
- ❌ 模型可能"重新学习"对话格式（浪费计算）
- ❌ 可能破坏 Mid 阶段学到的知识
- ❌ 在小数据集上容易过拟合

**使用 Mask 的好处**：
- ✅ 只更新 assistant 的生成能力
- ✅ 保留 Mid 阶段的对话理解能力
- ✅ 配合小学习率（0.02x），防止遗忘

**原因 2：数据效率**

SFT 只有 23K 样本（vs Mid 的 850K）：
- 如果训练所有 token，数据量太少
- 只训练 assistant 回复，提高数据利用效率

**原因 3：避免记忆训练数据**

```python
# 问题：如果训练所有 token
输入：<|user_start|>What is the capital of France?
目标：<|user_end|>

# 风险：模型可能记住了这个问题
# 测试时看到类似问题，直接输出训练数据的答案
```

**使用 Mask**：
- 模型不会记住问题的具体内容
- 只学习如何生成合适的回答
- 泛化能力更强

### 9.5 数据流完整对比

#### Mid Training 数据流

```python
# Step 1: Tokenize（返回 mask 但丢弃）
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}

ids, _ = tokenizer.render_conversation(conversation)
# ids = [BOS, USER_START, W, h, a, t, ..., USER_END, 
#        ASST_START, 4, ASST_END]

# Step 2: 拼接到 token buffer
token_buffer.extend(ids)

# Step 3: 生成 batch（没有 mask）
inputs = [USER_START, W, h, a, t, ...]  # (B, T)
targets = [W, h, a, t, ..., USER_END, ASST_START, 4, ASST_END]

# Step 4: 计算 loss（所有 token）
logits = model(inputs)  # (B, T, V)
loss = F.cross_entropy(
    logits.view(-1, V),
    targets.view(-1)
)
# 每个 token 都有梯度：
# - W, h, a, t (user 的输入)
# - 4 (assistant 的输出)
# - 所有特殊 token
```

**梯度更新的内容**：
```
∂Loss/∂θ = 梯度来自所有 token 的预测误差

包括：
- 预测 "What" 中的 "h" 的误差
- 预测 "?" 后应该接 USER_END 的误差
- 预测 USER_END 后应该接 ASST_START 的误差
- 预测 ASST_START 后应该接 "4" 的误差
```

#### SFT Training 数据流

```python
# Step 1: Tokenize（保留 mask）
ids, mask = tokenizer.render_conversation(conversation)
# ids = [BOS, USER_START, W, h, a, t, ..., USER_END, 
#        ASST_START, 4, ASST_END]
# mask = [0, 0, 0, 0, 0, 0, ..., 0, 
#         0, 1, 1]
#                ↑  ↑  只有这两个是 1

# Step 2: 应用 mask（设置 ignore_index）
targets = ids[1:]
mask_tensor = mask[1:]
targets[mask_tensor == 0] = -1
# targets = [-1, -1, -1, -1, -1, ..., -1, 
#            -1, 4, ASST_END]
#                ↑   ↑  只有这两个保留

# Step 3: 计算 loss（只在 mask=1 的 token 上）
loss = F.cross_entropy(
    logits.view(-1, V),
    targets.view(-1),
    ignore_index=-1  # ← 关键！
)
# 只有 targets != -1 的位置有梯度
```

**梯度更新的内容**：
```
∂Loss/∂θ = 梯度只来自 mask=1 的 token

包括：
- 预测 "4" 的误差（assistant 的回复）
- 预测 ASST_END 的误差（结束 token）

不包括：
- User 输入的任何 token（mask=0）
- 特殊 token 如 BOS, USER_START, USER_END（mask=0）
```

### 9.6 实验验证

**假设实验 1：如果 Mid Training 也使用 Mask？**

可能的结果：
- ❌ 模型学不会对话格式（没见过 user 输入的训练信号）
- ❌ 生成时可能在错误的位置停止（不知道什么时候该结束）
- ❌ 无法处理多轮对话（不理解对话的流程）

**假设实验 2：如果 SFT Training 不使用 Mask？**

可能的结果：
- ❌ 过拟合训练数据的问题（记住了具体问题）
- ❌ 破坏 Mid 阶段的知识（重新学习对话格式）
- ❌ 泛化能力下降（在新问题上表现差）

### 9.7 设计哲学总结

```
┌─────────────────────────────────────────────────────────┐
│                    训练阶段的演变                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Base Training: 学习语言                                 │
│  ↓                                                      │
│  训练：所有 token（纯文本，无对话格式）                   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Mid Training: 学习对话流                                │
│  ↓                                                      │
│  训练：所有 token（包括 user 和 assistant）              │
│  目标：理解对话格式 + 学会生成回复                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  SFT Training: 精细调整回复                              │
│  ↓                                                      │
│  训练：只有 assistant 的 token（使用 mask）              │
│  目标：优化输出风格 + 指令遵循                           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  RL Training: 强化推理能力                               │
│  ↓                                                      │
│  训练：只有 assistant 的 token（使用 mask）              │
│  目标：多步推理 + 试错探索                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**核心原则**：
1. **Mid 阶段**：从零学习对话 → 需要完整的训练信号
2. **SFT/RL 阶段**：优化已有能力 → 只需要针对性的训练信号

**类比**：
- **Mid Training** 像学开车：需要学习方向盘、油门、刹车、换挡...（所有操作）
- **SFT Training** 像参加驾考：只需要优化驾驶技巧，不需要重新学基本操作

### 9.8 代码位置参考

```python
# Mask 的生成：nanochat/tokenizer.py 第 258-358 行
def render_conversation(self, conversation):
    """返回 ids 和 mask"""
    ids, mask = [], []
    # ... 详细的 mask 生成逻辑

# Mid Training 丢弃 mask：scripts/mid_train.py 第 133 行
ids, _ = tokenizer.render_conversation(conversation)
#      ↑ 下划线表示丢弃返回值

# SFT Training 使用 mask：scripts/chat_sft.py 第 123 行
ids, mask = tokenizer.render_conversation(doc)
# 后续在 collate_and_yield 中将 mask=0 的位置设为 -1
```

---

## 总结

**Mid Training 的三个关键作用：**

1. **能力扩展** - 从通用语言理解到特定任务能力
2. **数据桥接** - 从无结构文本到结构化对话
3. **技能注入** - 工具使用、推理、拼写等

**关键设计原则（新增）：**
- 📊 大规模混合（850K 对话）
- 📊 确定性 shuffle（seed=42）
- 📊 渐进式学习率调度
- 📊 高质量数据混合
- 📊 **训练所有 token（不使用 mask）** ← 核心差异！

**Mask 的使用时机：**
- ❌ Base Training：无对话格式，不需要 mask
- ❌ Mid Training：学习对话流，**不使用 mask**
- ✅ SFT Training：精细调整，**使用 mask**
- ✅ RL Training：强化学习，**使用 mask**

**训练流程：**
```
Base Model (通用能力)
    ↓
+ Structured Tasks (结构化任务)
    ↓ [训练所有 token，学习对话流]
Mid Model (扩展能力)
    ↓
准备好进行 SFT [只训练 assistant 回复]
```

**下一步：**
- SFT：教模型遵循指令（使用 mask）
- RL：通过强化学习优化行为（使用 mask）

---

*本文档基于 nanochat 项目分析生成*  
*适合 LLM 初学者理解 Mid Training 的完整流程*  
*更新时间: 2025年12月22日*
