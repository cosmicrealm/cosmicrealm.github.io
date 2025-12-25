---
title: 'noao-chat-sft阶段训练'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-25-blog-nonochat-sft/
tags:
  - llm
---

### LLM SFT 训练完整解析

[nonochat](https://github.com/karpathy/nanochat)



> 本文档详细解析 `chat_sft.py`，讲解监督微调（Supervised Fine-Tuning）如何将模型变成对话助手。  
> 适合 LLM 领域的初学者，从能力模型到应用模型的关键一步。

---

## 目录

1. [什么是 SFT](#1-什么是-sft)
2. [Mask 机制的奥秘](#2-mask-机制的奥秘)
3. [训练流程详解](#3-训练流程详解)
4. [评估体系](#4-评估体系)
5. [与前序阶段的对比](#5-与前序阶段的对比)

---

## 1. 什么是 SFT

### 1.1 训练阶段全景

```
Base Training    →    Mid Training    →    SFT           →    RL
     ↓                     ↓                  ↓                  ↓
通用语言能力         特定任务能力         指令遵循        偏好对齐
(海量文本)         (结构化任务)        (对话微调)      (强化学习)
  数周训练              数小时              数小时          数小时
  数十亿tokens        数百万对话          数万对话        数千对话
```

**SFT 的定位：** 训练流程的第三阶段，将模型从"能力型"转变为"应用型"。

### 1.2 SFT 要解决什么问题？

**Mid 模型的能力与限制：**

测试 Mid 模型：
```
输入: "What is 2+2?"
Mid 模型可能输出: "2+2=4. Four is a number..."
```

问题：
- ✅ 知道答案是 4
- ❌ 回答太随意，不够简洁
- ❌ 不遵循特定格式
- ❌ 可能继续生成无关内容

**我们期望的输出：**
```
输入: "What is 2+2?"
SFT 模型输出: "The answer is 4."
```

特点：
- ✅ 简洁明了
- ✅ 直接回答问题
- ✅ 遵循良好的对话礼仪
- ✅ 知道何时停止

### 1.3 SFT 的核心目标

**三个关键能力：**

1. **指令遵循 (Instruction Following)**
   ```
   用户: "用三个词总结这篇文章"
   模型: "创新、技术、未来"  ← 严格遵循"三个词"的要求
   ```

2. **格式控制 (Format Control)**
   ```
   用户: "以 JSON 格式回答"
   模型: {"answer": "42", "reason": "..."}  ← 精确的格式
   ```

3. **对话礼仪 (Conversational Etiquette)**
   ```
   用户: "谢谢！"
   模型: "不客气！有其他问题随时问我。"  ← 友好、得体
   ```

### 1.4 SFT 的数据特点

**数据量对比：**

| 阶段 | 数据量 | 数据类型 |
|-----|--------|---------|
| Base | 数十亿 tokens | 网页文本 |
| Mid | 数百万对话 | 任务数据集 |
| **SFT** | **数万对话** | **精心标注** |
| RL | 数千对话 | 环境反馈 |

**SFT 数据的黄金标准：**
- 📝 每条对话都是**人工编写或审核**
- ✨ 展示**最佳实践**（不是随便的对话）
- 🎯 涵盖**应用场景**（实际用户会问什么）
- 🏆 体现**期望行为**（模型应该怎么回答）

### 1.5 本项目的 SFT 配置

```python
# 训练数据混合 (总共 ~23K 对话)
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),      # 2.3K 简单推理
    ARC(subset="ARC-Challenge", split="train"), # 1.1K 困难推理
    GSM8K(subset="main", split="train"),        # 8K 数学题
    SmolTalk(split="train", stop=10_000),       # 10K 对话
    CustomJSON(identity_file),                  # 1K 身份对话
    SimpleSpelling(size=300, split="train"),    # 300 拼写
    SpellingBee(size=300, split="train"),       # 300 字母计数
])
```

**数据特点：**
- 相比 Mid Training (850K)，数据量减少了 **97%**
- 但都是**高质量**的对话示例
- 涵盖多种任务类型
- 每个任务都有明确的期望输出格式

---

## 2. Mask 机制的奥秘

### 2.1 为什么需要 Mask？

**问题场景：**

假设一个对话：
```
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."
```

**如果没有 Mask（全部训练）：**

```python
# 所有 token 都计算损失
tokens = [<|bos|>, <|user_start|>, "What", "is", "the", "capital", 
          "of", "France", "?", <|user_end|>, <|assistant_start|>, 
          "The", "capital", "of", "France", "is", "Paris", ".", 
          <|assistant_end|>]

# 模型需要学习：
# - 预测 <|user_start|> 之后是 "What"  ← 为什么要学这个？
# - 预测 "What" 之后是 "is"            ← 用户怎么说话不重要！
# - 预测 "?" 之后是 <|user_end|>        ← 这是固定格式
```

**问题：**
- ❌ 浪费训练资源学习用户的说话方式
- ❌ 模型可能学会生成用户消息（角色混淆）
- ❌ 训练信号被稀释

**Mask 的解决方案：**

```python
tokens = [<|bos|>, <|user_start|>, "What", "is", ..., <|user_end|>, 
          <|assistant_start|>, "The", "capital", ..., <|assistant_end|>]

mask   = [0,       0,              0,      0,   ..., 0,
          0,                       1,      1,        ..., 1]
          ↑                        ↑
          不训练                   训练！
```

**效果：**
- ✅ 只学习如何生成 Assistant 的回复
- ✅ 专注于重要的训练信号
- ✅ 避免角色混淆

### 2.2 Mask 的详细规则

#### 规则总览

| Token 类型 | Mask 值 | 是否训练 | 原因 |
|-----------|---------|----------|------|
| `<|bos|>` | 0 | ❌ | 文档开始标记 |
| `<|user_start|>` | 0 | ❌ | 用户消息开始 |
| 用户消息内容 | 0 | ❌ | 用户输入，不需要学习 |
| `<|user_end|>` | 0 | ❌ | 用户消息结束 |
| `<|assistant_start|>` | 0 | ❌ | 助手开始标记 |
| **助手文本** | **1** | **✅** | **核心训练内容！** |
| `<|python_start|>` | 1 | ✅ | 工具调用开始 |
| Python 代码 | 1 | ✅ | 学习何时/如何调用工具 |
| `<|python_end|>` | 1 | ✅ | 工具调用结束 |
| `<|output_start|>` | 0 | ❌ | 工具输出开始 |
| 工具输出 | 0 | ❌ | 环境返回，不需要学习 |
| `<|output_end|>` | 0 | ❌ | 工具输出结束 |
| `<|assistant_end|>` | 1 | ✅ | 学习何时停止 |

#### 代码实现

在 `tokenizer.py` 的 `render_conversation` 方法中：

```python
def render_conversation(self, conversation):
    ids, mask = [], []
    
    def add_tokens(token_ids, mask_val):
        """辅助函数：添加 token 和对应的 mask"""
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
    
    # 开始标记
    add_tokens(bos, 0)  # <|bos|> 不训练
    
    for message in messages:
        if message["role"] == "user":
            # 用户消息：全部不训练
            add_tokens(user_start, 0)
            add_tokens(user_message_ids, 0)  # ← mask=0
            add_tokens(user_end, 0)
            
        elif message["role"] == "assistant":
            # 助手消息：开始标记不训练，内容训练
            add_tokens(assistant_start, 0)  # ← 不训练开始标记
            
            if isinstance(content, str):
                # 纯文本回复
                add_tokens(content_ids, 1)  # ← mask=1，训练！
                
            elif isinstance(content, list):
                # 包含工具调用的回复
                for part in content:
                    if part["type"] == "text":
                        add_tokens(part_ids, 1)  # ← 文本部分训练
                    elif part["type"] == "python":
                        add_tokens(python_start, 1)
                        add_tokens(code_ids, 1)  # ← 代码部分训练
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        add_tokens(output_start, 0)
                        add_tokens(output_ids, 0)  # ← 输出不训练
                        add_tokens(output_end, 0)
            
            add_tokens(assistant_end, 1)  # ← 结束标记训练（学会停止）
    
    return ids, mask
```

### 2.3 Mask 的实际效果（图解）

#### 示例 1：简单问答

```
对话：
User: "What is 2+2?"
Assistant: "4"

Token 化：
<|bos|> <|user_start|> What is 2 + 2 ? <|user_end|> <|assistant_start|> 4 <|assistant_end|>
   0          0          0   0  0 0 0 0      0              0                1      1

                                                                              ↑      ↑
                                                                        只训练这两个！
```

**训练目标：**
- 给定 `<|assistant_start|>`，预测 `"4"`
- 给定 `"4"`，预测 `<|assistant_end|>`（学会停止）

#### 示例 2：带工具调用

```
对话：
User: "What is 12/60?"
Assistant: <|python_start|>12/60<|python_end|><|output_start|>0.2<|output_end|>The answer is 0.2

Token 化：
<|bos|> <|user_start|> What is 12/60 ? <|user_end|> 
   0          0          0   0    0   0      0

<|assistant_start|> <|python_start|> 12/60 <|python_end|> 
        0                   1            1          1
                            ↑            ↑          ↑
                        学习调用工具

<|output_start|> 0.2 <|output_end|> The answer is 0.2 <|assistant_end|>
       0          0         0            1    1     1  1       1
                                         ↑────────────↑────────↑
                                      学习如何使用工具结果
```

**训练要点：**
- ✅ 学习何时调用工具（`<|python_start|>`）
- ✅ 学习工具调用内容（`12/60`）
- ❌ 不学习工具输出（`0.2`）← 环境决定的
- ✅ 学习如何引用工具结果（"The answer is 0.2"）

### 2.4 Mask 在损失计算中的应用

#### 数据准备阶段

在 `chat_sft.py` 的 `sft_data_generator` 中：

```python
def collate_and_yield(batch):
    nrows = len(batch)
    ncols = max(len(ids) for ids, mask in batch) - 1
    
    # 初始化
    inputs = torch.full((nrows, ncols), pad_token_id)
    targets = torch.full((nrows, ncols), -1)  # ← -1 是关键！
    
    for i, (ids, mask) in enumerate(batch):
        n = len(ids)
        ids_tensor = torch.tensor(ids)
        
        # 输入和目标
        inputs[i, :n-1] = ids_tensor[:-1]
        targets[i, :n-1] = ids_tensor[1:]
        
        # 应用 mask：mask=0 的位置设为 -1
        mask_tensor = torch.tensor(mask[1:])
        targets[i, :n-1][mask_tensor == 0] = -1  # ← 屏蔽
    
    return inputs, targets
```

**关键点：**
- `targets` 中 mask=0 的位置被设为 `-1`
- `-1` 是 PyTorch CrossEntropyLoss 的 `ignore_index`

#### 损失计算

```python
# 在模型的 forward 方法中
def forward(self, inputs, targets):
    logits = self.lm_head(x)  # (B, T, vocab_size)
    
    # 交叉熵损失，自动忽略 target=-1 的位置
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-1  # ← 忽略 -1
    )
    return loss
```

**具体示例（B=2, T=8）：**

```python
# inputs (2, 8)
[[101, 202, 303, 404, 505, 606, 707, 808],
 [909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]]

# targets (2, 8) - 注意 -1 的位置
[[202, 303, 404, -1,  -1,  606, 707, 808],
 [1010, -1,  -1,  -1, 1313, 1414, 1515, -1]]
  ↑                   ↑
  训练               不训练

# 计算损失时：
# - 位置 [0,0]: 预测 202，计算损失 ✅
# - 位置 [0,3]: 目标是 -1，跳过 ❌
# - 位置 [1,1]: 目标是 -1，跳过 ❌
```

**效果：**
- 只有 mask=1 的位置贡献梯度
- 模型只学习 Assistant 的生成模式
- 训练效率更高，效果更好

---

## 3. 训练流程详解

### 3.1 完整数据流（从对话到梯度）

```
步骤 1: 加载对话数据
┌──────────────────────────────────────┐
│ ARC-Easy: 2.3K 科学推理              │
│ GSM8K: 8K 数学题                     │
│ SmolTalk: 10K 对话                   │
│ Spelling: 600 拼写任务               │
│ ...                                  │
│ 总计: ~23K 高质量对话                │
└──────────────────────────────────────┘
            ↓
步骤 2: TaskMixture 混合
┌──────────────────────────────────────┐
│ 所有任务混合打乱                      │
│ 确保多样性                            │
└──────────────────────────────────────┘
            ↓
步骤 3: 遍历对话
┌──────────────────────────────────────┐
│ for i in range(ddp_rank, len(ds), world_size): │
│     doc = dataset[i]                 │
│     conversation = {                 │
│         "messages": [...]            │
│     }                                │
└──────────────────────────────────────┘
            ↓
步骤 4: Tokenize + Mask
┌──────────────────────────────────────┐
│ ids, mask = tokenizer.render_conversation(doc) │
│                                      │
│ ids:  [1, 2, 3, 4, 5, 6, 7, 8, ...]  │
│ mask: [0, 0, 0, 1, 1, 0, 1, 1, ...]  │
│       ↑     ↑     ↑     ↑            │
│     不训练  训练 不训练  训练         │
└──────────────────────────────────────┘
            ↓
步骤 5: 组织批次
┌──────────────────────────────────────┐
│ 累积多个对话                          │
│ 每个批次: device_batch_size=4        │
│                                      │
│ inputs:  (4, T)                      │
│ targets: (4, T) ← 包含 -1 的屏蔽     │
└──────────────────────────────────────┘
            ↓
步骤 6: 模型前向传播
┌──────────────────────────────────────┐
│ logits = model(inputs)               │
│ # (4, T, vocab_size)                 │
└──────────────────────────────────────┘
            ↓
步骤 7: 计算损失（带 mask）
┌──────────────────────────────────────┐
│ loss = F.cross_entropy(              │
│     logits.view(-1, vocab_size),     │
│     targets.view(-1),                │
│     ignore_index=-1  ← 自动屏蔽      │
│ )                                    │
│ # loss: 标量                         │
└──────────────────────────────────────┘
            ↓
步骤 8: 反向传播
┌──────────────────────────────────────┐
│ loss.backward()                      │
│ # 只有 mask=1 的位置产生梯度         │
└──────────────────────────────────────┘
            ↓
步骤 9: 更新参数
┌──────────────────────────────────────┐
│ optimizer.step()                     │
│ model.zero_grad()                    │
└──────────────────────────────────────┘
```

### 3.2 超参数配置

#### 关键超参数

```python
# SFT 超参数
sft_config = {
    # 模型来源
    "source": "mid",  # 从 Mid 模型开始
    "model_tag": None,
    "step": None,
    
    # 训练规模
    "device_batch_size": 4,           # 每卡批次（小）
    "target_examples_per_step": 32,   # 目标批次大小
    "num_epochs": 1,                  # 训练轮数
    "num_iterations": -1,             # 自动计算
    
    # 学习率（层级化）
    "unembedding_lr": 0.004,  # 输出层（最小）
    "embedding_lr": 0.2,       # 输入层（最大）
    "matrix_lr": 0.02,         # 中间层
    "weight_decay": 0.0,
    "init_lr_frac": 0.02,      # 从 2% 开始！
    
    # 评估
    "eval_every": 100,
    "eval_steps": 100,
    "eval_metrics_every": 200,
}
```

#### 与 Base/Mid 的对比

| 超参数 | Base | Mid | SFT | 说明 |
|-------|------|-----|-----|------|
| 数据量 | 数十亿 | 850K | 23K | 越来越少 |
| batch_size | 32 | 32 | 4 | SFT 更小 |
| 学习率 | 0.02-0.2 | 0.02-0.2 | 0.004-0.2 | 相同 |
| init_lr_frac | 1.0 | 1.0 | **0.02** | SFT 从很小开始！ |
| 训练轮数 | - | 1 | 1 | 通常 1 轮 |
| 模型来源 | 随机初始化 | Base | **Mid** | 继承能力 |

**为什么 init_lr_frac=0.02？**

```python
# 实际学习率
actual_lr = base_lr * init_lr_frac

# 例如：
embedding_lr = 0.2 * 0.02 = 0.004   # 实际从 0.004 开始
matrix_lr = 0.02 * 0.02 = 0.0004    # 实际从 0.0004 开始
```

**原因：**
- Mid 模型已经很好了，不能破坏已有能力
- SFT 只是"微调"，不是重新训练
- 小学习率 = 轻柔调整 = 保留原有知识

### 3.3 学习率调度

```python
def get_lr_multiplier(it):
    # 简单的线性衰减
    lrm = 1.0 - it / num_iterations
    return lrm
```

**学习率曲线（num_iterations=1000）：**

```
实际学习率（以 matrix_lr 为例）
    
    0.0004 |╲                      ← 0.02 * 0.02 * 1.0
           | ╲
           |  ╲
    0.0002 |   ╲
           |    ╲
    0.0000 |     ╲________________ ← 0.02 * 0.02 * 0.0
           +──────────────────────→
           0    250   500   750   1000 step
           
初始就很小！然后逐渐减到 0
```

**对比 Base Training：**

```
Base Training 学习率：
    
    0.02 |         ══════════════════╲  ← 大学习率，长时间保持
         |                            ╲
         |                             ╲
    0.01 |                              ╲
         |                               ╲
    0.00 |                                ══
         +────────────────────────────────→

SFT 学习率：
    
  0.0004 |╲                              ← 小学习率，快速衰减
         | ╲
  0.0002 |  ╲
         |   ╲
  0.0000 |    ════════════════════════
         +────────────────────────────→
```

**设计哲学：**
- Base: "从零学习" → 需要大学习率
- SFT: "轻微调整" → 需要小学习率

### 3.4 训练循环详解

```python
# 预计算迭代次数
num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
# 例如：(23000 // 32) * 1 ≈ 718 步

for step in range(num_iterations):
    last_step = (step == num_iterations)
    
    # ===== 评估验证损失 =====
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad():
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        print(f"Step {step} | Val loss: {val_loss:.6f}")
        model.train()
    
    # ===== 评估任务指标 =====
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        # MMLU 准确率
        mmlu_acc = run_chat_eval("MMLU", model, ...)
        # ARC-Easy 准确率
        arc_acc = run_chat_eval("ARC-Easy", model, ...)
        print(f"Step {step} | MMLU: {mmlu_acc:.4f}, ARC: {arc_acc:.4f}")
        model.train()
    
    if last_step:
        break
    
    # ===== 计算梯度 =====
    num_tokens = 0
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        
        loss = loss / grad_accum_steps
        loss.backward()
        
        # 统计有效 token 数量
        num_tokens += (train_targets >= 0).sum()
    
    # ===== 更新学习率 =====
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    # ===== 更新参数 =====
    for opt in optimizers:
        opt.step()
    model.zero_grad()
    
    # ===== 日志 =====
    print(f"Step {step}/{num_iterations} | loss: {loss:.6f} | "
          f"lrm: {lrm:.4f} | num_tokens: {num_tokens}")
```

### 3.5 一个完整 Step 的维度追踪

假设：`device_batch_size=4, grad_accum_steps=8, max_seq_len=2048`

```python
# ===== Micro-step 1 =====
train_inputs, train_targets = next(train_loader)
# inputs:  (4, 2048) - int32
# targets: (4, 2048) - int64，包含 -1

# 前向传播
x = model.transformer.wte(train_inputs)  # (4, 2048, 768)
# ... Transformer blocks ...
logits = model.lm_head(x)  # (4, 2048, 32000)

# 损失计算
loss = F.cross_entropy(
    logits.view(-1, 32000),   # (8192, 32000)
    train_targets.view(-1),   # (8192,)
    ignore_index=-1
)
# loss: 标量，例如 2.345

# 归一化并反向传播
loss = loss / 8  # 2.345 / 8 = 0.293
loss.backward()  # 累积梯度

# ===== Micro-step 2-8 =====
# 重复上述过程，梯度持续累积

# ===== 更新参数 =====
for opt in optimizers:
    opt.step()  # 使用累积的梯度更新
model.zero_grad()  # 清零，准备下一步
```

**关键观察：**
- 每个 micro-step 处理 4 个样本
- 8 个 micro-steps = 32 个样本
- 只有 targets >= 0 的位置产生梯度
- 梯度累积后一次性更新

---

## 4. 评估体系

### 4.1 评估指标概览

SFT 有两种评估方式：

1. **验证损失 (Validation Loss)** - 衡量模型拟合度
2. **任务准确率 (Task Accuracy)** - 衡量实际能力

### 4.2 验证损失评估

```python
def evaluate_validation_loss():
    model.eval()
    val_loader = build_val_loader()
    losses = []
    
    for _ in range(eval_steps):  # 例如 100 步
        val_inputs, val_targets = next(val_loader)
        with torch.no_grad():
            loss = model(val_inputs, val_targets)
        losses.append(loss)
    
    val_loss = torch.stack(losses).mean()
    return val_loss
```

**验证数据来源：**
```python
val_ds = SmolTalk(split="test")  # 24K 测试对话
```

**Loss 的意义：**
- 越小越好
- 衡量模型对对话的"困惑度"
- 典型值：1.5 - 2.5

### 4.3 任务准确率评估

#### MMLU 评估

```python
def run_chat_eval(task_name, model, tokenizer, engine, 
                  batch_size=8, max_problems=1024):
    """
    评估模型在选择题任务上的表现
    """
    if task_name == "MMLU":
        task = MMLU(subset="all", split="test")
    elif task_name == "ARC-Easy":
        task = ARC(subset="ARC-Easy", split="test")
    
    correct = 0
    total = 0
    
    for idx in range(min(max_problems, len(task))):
        # 获取问题
        conversation = task[idx]
        # conversation["letters"] = ["A", "B", "C", "D"]
        
        # 生成回答
        tokens = tokenizer.render_for_completion(conversation)
        
        with torch.no_grad():
            # 限制输出只能是 A/B/C/D
            assistant_response = engine.generate_restricted(
                tokens,
                allowed_tokens=conversation["letters"]
            )
        
        # 评估正确性
        is_correct = task.evaluate(conversation, assistant_response)
        correct += is_correct
        total += 1
    
    accuracy = correct / total
    return accuracy
```

**关键技术：Restricted Generation**

```python
# 问题：模型可能生成 "The answer is B" 而不是 "B"
# 解决：强制只能输出 A/B/C/D 中的一个

def generate_restricted(tokens, allowed_tokens):
    # 获取 logits
    logits = model(tokens)[:, -1, :]  # (1, vocab_size)
    
    # 找到允许的 token IDs
    allowed_ids = [tokenizer.encode_special(t) for t in allowed_tokens]
    # 例如：[65, 66, 67, 68] 对应 A, B, C, D
    
    # 只保留这些位置的 logits
    restricted_logits = logits[:, allowed_ids]  # (1, 4)
    
    # 选择最大的
    best_idx = restricted_logits.argmax()
    return allowed_tokens[best_idx]  # "A" or "B" or "C" or "D"
```

#### ARC 评估

```python
# ARC (AI2 Reasoning Challenge)
# 类似 MMLU，也是选择题

arc_easy_acc = run_chat_eval("ARC-Easy", model, ...)
arc_challenge_acc = run_chat_eval("ARC-Challenge", model, ...)
```

**特点：**
- ARC-Easy: 较简单的科学推理
- ARC-Challenge: 较难的科学推理

### 4.4 评估频率

```python
# 验证损失：频繁评估
eval_every = 100  # 每 100 步

# 任务准确率：不太频繁
eval_metrics_every = 200  # 每 200 步

# 原因：
# - 验证损失快速（只是前向传播）
# - 任务准确率慢（需要生成 + 解析）
```

### 4.5 训练日志示例

```
Step 00000 | Validation loss: 2.456789
Step 00000 | mmlu_acc: 0.2543, arc_easy_acc: 0.4567

Step 00100/00718 | Training loss: 2.123456 | lrm: 0.8607 | num_tokens: 52,341
Step 00100 | Validation loss: 2.234567

Step 00200/00718 | Training loss: 1.987654 | lrm: 0.7215 | num_tokens: 51,892
Step 00200 | Validation loss: 2.098765
Step 00200 | mmlu_acc: 0.3245, arc_easy_acc: 0.5123  ← 提升了！

...

Step 00700/00718 | Training loss: 1.765432 | lrm: 0.0251 | num_tokens: 50,234
Step 00718 | Validation loss: 1.823456
Step 00718 | mmlu_acc: 0.4567, arc_easy_acc: 0.6234  ← 显著提升！
```

**观察：**
- Training loss 持续下降
- 任务准确率稳步提升
- 学习率逐渐减小

---

## 5. 与前序阶段的对比

### 5.1 四阶段全对比

| 维度 | Base | Mid | SFT | RL |
|-----|------|-----|-----|-----|
| **数据量** | 数十亿 tokens | 850K 对话 | 23K 对话 | 8K 对话 |
| **数据源** | 网页抓取 | 公开数据集 | 精选对话 | 环境反馈 |
| **数据质量** | 低 | 中 | 高 | 高 |
| **模型来源** | 随机初始化 | Base | Mid | SFT |
| **训练目标** | 语言建模 | 任务能力 | 指令遵循 | 偏好对齐 |
| **学习率** | 大 (0.02-0.2) | 中 (0.02-0.2) | 小 (0.0004-0.004) | 小 (0.0008-0.01) |
| **Batch Size** | 大 (32) | 大 (32) | 小 (4) | 中 (8) |
| **训练时长** | 数天-数周 | 数小时-1天 | 数小时 | 数小时 |
| **Loss 类型** | 全部 token | 全部 token | **Masked** | **Policy Gradient** |
| **评估方式** | BPB | BPB | 准确率 | Pass@k |

### 5.2 数据格式演变

#### Base Training

```
纯文本流：
"The quick brown fox jumps over the lazy dog. ..."

Token 化：
[15496, 995, 831, 374, 264, 1296, ...]

训练：全部 token
```

#### Mid Training

```
结构化对话（但仍然训练所有内容）：
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Token 化：
[<|bos|>, <|user_start|>, ..., <|assistant_start|>, ...]

训练：全部 token（包括 user 部分）
```

#### SFT

```
结构化对话 + Mask：
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Token 化 + Mask：
ids:  [<|bos|>, <|user_start|>, ..., <|assistant_start|>, ...]
mask: [0,       0,              ..., 0,                   ...]
      [                             1, 1, 1, ...]
      ↑                             ↑
    不训练                         只训练这部分！
```

**关键创新：Mask 机制！**

### 5.3 Loss 计算方式对比

```python
# ===== Base/Mid Training =====
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1)
    # 所有位置都计算损失
)

# ===== SFT =====
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=-1  # ← 关键差异！
)
# targets 中 mask=0 的位置是 -1，被忽略
```

### 5.4 模型能力演变

**测试对话：**

```
User: "Write a Python function to add two numbers."
```

#### Base 模型输出

```
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
...
```

**问题：**
- ❌ 继续生成无关函数
- ❌ 不知道何时停止
- ❌ 没有遵循"只要加法"的指令

#### Mid 模型输出

```
<|assistant_start|>def add(a, b):
    return a + b<|assistant_end|>

<|user_start|>Can you also write subtract?<|user_end|>
...
```

**问题：**
- ✅ 知道对话格式
- ✅ 能生成正确代码
- ❌ 继续生成虚构的用户消息
- ❌ 角色混淆

#### SFT 模型输出

```
Here's a Python function to add two numbers:

```python
def add(a, b):
    return a + b
```

This function takes two parameters and returns their sum.
```

**优点：**
- ✅ 直接回答问题
- ✅ 格式规范（Markdown）
- ✅ 简洁明了
- ✅ 适时停止
- ✅ 提供说明

### 5.5 训练效率对比

**参数更新次数：**

假设 200M 参数的模型：

| 阶段 | 数据量 | 步数 | 每个参数看到的数据 |
|-----|--------|------|-------------------|
| Base | 10B tokens | 50K | 50K 次更新 |
| Mid | 850K 对话 | 2K | 2K 次更新 |
| SFT | 23K 对话 | 718 | **718 次更新** |

**训练时间（8 卡 A100）：**

| 阶段 | 训练时间 | 每步时间 |
|-----|---------|---------|
| Base | 3 天 | ~5 秒 |
| Mid | 6 小时 | ~10 秒 |
| SFT | **2 小时** | ~10 秒 |

**为什么 SFT 这么快？**
- 数据量小（23K vs 850K）
- 轮数少（1 epoch）
- 只需要"微调"，不需要"重训练"

---

## 6. 实战案例分析

### 6.1 完整训练流程

#### 阶段 1: Base Training (已完成)

```bash
# 输出：base_checkpoints/d12/step_00050000.pt
```

#### 阶段 2: Mid Training (已完成)

```bash
# 输出：mid_checkpoints/d12/step_XXXXX.pt
```

#### 阶段 3: SFT (当前阶段)

```bash
# 单 GPU 调试
python -m scripts.chat_sft

# 8 GPU 训练
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --source=mid \
    --device_batch_size=4 \
    --num_epochs=1 \
    --run=sft_d12

# 输出：chatsft_checkpoints/d12/step_00718.pt
```

### 6.2 训练数据示例

#### 示例 1: ARC-Easy

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Which material is the best conductor of electricity?\nA. wood\nB. plastic\nC. copper\nD. rubber"
        },
        {
            "role": "assistant",
            "content": "C"
        }
    ],
    "letters": ["A", "B", "C", "D"]
}
```

**Tokenize 后：**
```
<|bos|> <|user_start|> Which material ... <|user_end|> 
<|assistant_start|> C <|assistant_end|>

Mask:
[0,     0,             0,    0,    ..., 0,
 0,                    1, 1]
                       ↑  ↑
                  只训练 "C" 和结束标记
```

#### 示例 2: GSM8K

```json
{
    "messages": [
        {
            "role": "user",
            "content": "If 5 apples cost $10, how much do 3 apples cost?"
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First, find the cost per apple:\n"},
                {"type": "python", "text": "10/5"},
                {"type": "python_output", "text": "2"},
                {"type": "text", "text": "\nSo each apple costs $2. For 3 apples:\n"},
                {"type": "python", "text": "2*3"},
                {"type": "python_output", "text": "6"},
                {"type": "text", "text": "\n#### 6"}
            ]
        }
    ]
}
```

**Mask 规则：**
```
<|user_start|> ... <|user_end|>                           [mask=0]
<|assistant_start|>                                        [mask=0]
    First, find the cost per apple:                        [mask=1] ✅
    <|python_start|> 10/5 <|python_end|>                  [mask=1] ✅
    <|output_start|> 2 <|output_end|>                     [mask=0] ❌
    So each apple costs $2. For 3 apples:                 [mask=1] ✅
    <|python_start|> 2*3 <|python_end|>                   [mask=1] ✅
    <|output_start|> 6 <|output_end|>                     [mask=0] ❌
    #### 6                                                  [mask=1] ✅
<|assistant_end|>                                          [mask=1] ✅
```

### 6.3 训练过程监控

#### 初始状态 (Step 0)

```
Step 00000 | Validation loss: 2.456
Step 00000 | mmlu_acc: 0.254, arc_easy_acc: 0.457
```

**解读：**
- Val loss ~2.5：Mid 模型的起点
- MMLU 25.4%：比随机猜测 (25%) 略好
- ARC 45.7%：还可以

#### 训练中期 (Step 350)

```
Step 00350/00718 | Training loss: 1.876 | lrm: 0.5125
Step 00350 | Validation loss: 2.012
Step 00400 | mmlu_acc: 0.367, arc_easy_acc: 0.589
```

**解读：**
- Training loss 下降到 1.87
- Val loss 下降到 2.01
- MMLU 提升到 36.7% (+11.3%)
- ARC 提升到 58.9% (+13.2%)

#### 训练完成 (Step 718)

```
Step 00718/00718 | Training loss: 1.654 | lrm: 0.0000
Step 00718 | Validation loss: 1.823
Step 00718 | mmlu_acc: 0.423, arc_easy_acc: 0.645
```

**最终收获：**
- Val loss: 2.456 → 1.823 (-25.8%)
- MMLU: 25.4% → 42.3% (+16.9%)
- ARC: 45.7% → 64.5% (+18.8%)

### 6.4 模型对比测试

#### 测试 1: 简单问答

```
输入: "What is the capital of France?"
```

| 模型 | 输出 | 评分 |
|-----|------|------|
| Base | "Paris is capital France city..." | ⭐⭐ |
| Mid | "The capital of France is Paris, which is..." | ⭐⭐⭐ |
| SFT | "Paris." | ⭐⭐⭐⭐⭐ |

**SFT 的优势：** 简洁、直接、准确

#### 测试 2: 指令遵循

```
输入: "List three colors in JSON format."
```

| 模型 | 输出 | 评分 |
|-----|------|------|
| Mid | "Red, blue, green are three colors." | ❌ |
| SFT | `{"colors": ["red", "blue", "green"]}` | ✅ |

**SFT 的优势：** 严格遵循格式要求

#### 测试 3: 工具使用

```
输入: "What is 123 * 456?"
```

| 模型 | 输出 | 评分 |
|-----|------|------|
| Mid | "123 * 456 = <|python_start|>123*456<|python_end|>..." 然后继续生成 | ⭐⭐⭐ |
| SFT | "<|python_start|>123*456<|python_end|><|output_start|>56088<|output_end|>The answer is 56,088." | ⭐⭐⭐⭐⭐ |

**SFT 的优势：** 
- 正确使用工具
- 等待工具输出
- 整合结果给出答案
- 适时停止

---

## 7. 关键要点总结

### 7.1 SFT 的本质

```
SFT = Mid Model + High-Quality Demonstrations + Mask
    = 能力模型 + 行为示范 + 精确训练
    = 应用就绪的对话助手
```

### 7.2 Mask 的重要性

**没有 Mask：**
```
模型学习：用户怎么提问 + 助手怎么回答
问题：浪费资源，可能角色混淆
```

**有 Mask：**
```
模型只学习：助手怎么回答
好处：高效、精准、效果好
```

### 7.3 SFT vs RL

| 维度 | SFT | RL |
|-----|-----|-----|
| 训练方式 | 监督学习 | 强化学习 |
| 需要什么 | 正确答案 | 奖励函数 |
| 学习内容 | 模仿示例 | 探索优化 |
| 适用场景 | 格式、礼仪 | 推理、偏好 |
| 训练难度 | 简单 | 复杂 |

### 7.4 实践建议

**数据准备：**
- ✅ 优先质量，不是数量
- ✅ 确保示例展示最佳实践
- ✅ 覆盖目标应用场景
- ✅ 包含多样化的任务类型

**训练配置：**
- 学习率：从很小开始 (init_lr_frac=0.02)
- 训练轮数：通常 1 轮就够
- Batch size：可以比较小 (4-8)
- 评估：频繁检查准确率

**常见错误：**
- ❌ 学习率太大 → 破坏已有能力
- ❌ 训练太久 → 过拟合
- ❌ 忘记 Mask → 效果差
- ❌ 数据质量低 → 白费力气

---

## 附录

### A. 完整训练命令

```bash
# 从 Mid 模型开始
python -m scripts.chat_sft \
    --source=mid \
    --device_batch_size=4 \
    --target_examples_per_step=32 \
    --num_epochs=1 \
    --eval_every=100 \
    --run=sft_experiment

# 从 Base 模型开始（不推荐）
python -m scripts.chat_sft \
    --source=base \
    --model_tag=d12 \
    --step=50000 \
    --device_batch_size=4

# 多 GPU
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --device_batch_size=4 \
    --run=sft_production
```

### B. 数据统计

| 数据集 | 训练量 | 测试量 | 任务类型 |
|-------|-------|-------|---------|
| ARC-Easy | 2.3K | 570 | 科学推理（简单） |
| ARC-Challenge | 1.1K | 299 | 科学推理（困难） |
| GSM8K | 7.5K | 1.3K | 数学推理 |
| SmolTalk | 10K | 24K | 通用对话 |
| Identity | 1K | - | 身份设定 |
| SimpleSpelling | 300 | - | 拼写基础 |
| SpellingBee | 300 | - | 字母计数 |
| **总计** | **~23K** | **~26K** | - |

### C. 常见问题

**Q1: SFT 可以跳过吗？**
- 理论上可以，但不推荐
- 没有 SFT，模型不会遵循指令

**Q2: 能不能多做几轮 SFT？**
- 可以，但通常 1 轮就够
- 多轮可能过拟合
- 更好的选择是增加数据多样性

**Q3: 为什么不直接从 Base 做 SFT？**
- 可以，但效果不如 Base→Mid→SFT
- Mid Training 提供了任务理解能力
- SFT 只需要调整行为，不需要学习能力

**Q4: SFT 数据怎么准备？**
- 人工编写高质量对话
- 从真实用户交互中筛选
- 使用更强模型生成（Distillation）
- 人工审核和修正

**Q5: Mask 是必须的吗？**
- 强烈推荐！
- 没有 Mask 也能训练，但效果差
- Mask 让训练更高效、更精准

---

## 总结

**SFT 的三个关键创新：**

1. **Mask 机制** - 只训练 Assistant 的回复
2. **小学习率** - 保护已有能力，轻柔调整
3. **高质量数据** - 少而精，展示最佳实践

**训练流程：**
```
Mid Model (任务能力)
    ↓
+ High-Quality Demonstrations (行为示范)
    ↓
+ Mask Mechanism (精确训练)
    ↓
SFT Model (应用就绪)
    ↓
准备好服务用户！
```

**下一步：RL**
- 通过强化学习进一步优化
- 处理复杂推理任务
- 对齐人类偏好

---

*本文档基于 nanochat 项目分析生成*  
*适合 LLM 初学者理解 SFT 的完整流程*  
*创建时间: 2025年12月21日*
