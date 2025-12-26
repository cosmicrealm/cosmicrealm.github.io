---
title: 'noao-chat-5-训练四阶段数据报告'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-25-blog-nonochat-5-dataset-report/
tags:
  - llm
---

### nanochat 项目四阶段训练数据完全报告


[nonochat](https://github.com/karpathy/nanochat)

## 📋 报告概览

本报告全面分析 nanochat 项目的四个训练阶段（Base → Mid → SFT → RL），深入剖析每个阶段使用的训练数据、验证数据、数据量、数据来源和评估指标。

---

## 🎯 四阶段数据总览表

| 阶段 | 训练数据来源 | 训练数据量 | 验证数据来源 | 验证数据量 | 主要评估指标 |
|------|-------------|-----------|------------|-----------|------------|
| **Base** | FineWeb-Edu | 100B tokens | FineWeb-Edu (最后一个shard) | ~10M tokens | BPB + CORE Metric |
| **Mid** | 7个任务混合 | 848K 对话 | 3个任务混合 | 39K 对话 | BPB |
| **SFT** | 7个任务混合 | 23K 对话 | SmolTalk | 24K 对话 | Loss + MMLU/ARC准确率 |
| **RL** | GSM8K | 7.5K 问题 | GSM8K | 1.3K 问题 | Pass@k (k=1,4,16) |

---

## 📊 第一阶段：Base Training（基础预训练）

### 训练数据

**数据集名称：FineWeb-Edu-100B**

```python
# 数据源配置
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # 共 1823 个 shard (shard_00000 到 shard_01822)
```

#### 数据规模
- **Token 数量**：~100B tokens
- **文件数量**：1823 个 Parquet 文件
- **文件格式**：`shard_00000.parquet` ~ `shard_01822.parquet`
- **存储大小**：每个文件约 50-100 MB
- **数据切分**：前 1822 个文件用于训练，最后 1 个文件用于验证

#### 数据特点
- **来源**：高质量教育网页（FineWeb-Edu 子集）
- **语言**：主要是英文
- **内容类型**：网页抓取的纯文本
- **预处理**：
  - 已经过质量过滤（教育相关内容）
  - 已去重和清洗
  - 存储格式：Parquet 文件，每行包含 `text` 字段

#### 数据加载机制

```python
# 从 dataset.py
def parquets_iter_batched(split, start=0, step=1):
    """
    - split: "train" 或 "val"
    - start/step: 用于 DDP 分布式训练
    """
    parquet_paths = list_parquet_files()
    
    if split == "train":
        parquet_paths = parquet_paths[:-1]  # 前 1822 个文件
    else:  # val
        parquet_paths = parquet_paths[-1:]  # 最后 1 个文件
    
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts
```

**关键特性**：
- ✅ **流式读取**：不需要一次性加载全部数据到内存
- ✅ **Row Group 级别**：利用 Parquet 的 row group 特性实现精细化分布式读取
- ✅ **DDP 友好**：通过 `start` 和 `step` 参数实现多卡训练时的数据分割

### 验证数据

**数据集：FineWeb-Edu（验证集）**

#### 规模
- **文件数量**：1 个 Parquet 文件（`shard_01822.parquet`）
- **Token 数量**：约 10-20M tokens
- **评估频率**：每 250 步（可配置 `--eval_every=250`）
- **评估 Token 数**：20 × 524288 = 10,485,760 tokens（可配置 `--eval_tokens`）

#### 评估指标 1：BPB (Bits Per Byte)

**公式**：
```
BPB = Loss / ln(2) / bytes_per_token
```

**计算过程**：
```python
# 从 loss_eval.py
def evaluate_bpb(model, val_loader, eval_steps, token_bytes):
    total_loss = 0.0
    total_tokens = 0
    
    for _ in range(eval_steps):
        inputs, targets = next(val_loader)  # (B, T)
        loss = model(inputs, targets)  # CrossEntropy loss
        
        # 统计非 padding token 的数量
        num_tokens = (targets != -1).sum()
        total_loss += loss * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens  # 平均 loss
    
    # token_bytes: (vocab_size,) 每个 token 对应的字节数
    bytes_per_token = token_bytes[targets[targets != -1]].float().mean()
    
    # BPB = Loss 转换为 bits，再除以每 token 的字节数
    bpb = avg_loss / math.log(2) / bytes_per_token
    
    return bpb
```

**BPB 的优势**：
- 🎯 **Tokenizer 无关**：不同的 tokenizer 可以比较
- 🎯 **字节级归一化**：更公平的压缩率度量
- 🎯 **可解释性**：表示每个字节需要多少 bits 来编码

#### 评估指标 2：CORE Metric

**CORE = Comprehensive Reasoning Evaluation**

CORE Metric 是 9 个不同任务的平均表现（经过 centered 归一化）：

| 任务类型 | 任务名称 | 样本数量 | 测试内容 |
|---------|---------|---------|---------|
| 多选题 | ARC-Easy | 500 | 小学科学推理 |
| 多选题 | ARC-Challenge | 500 | 中学科学推理 |
| 多选题 | MMLU | 500 | 57个学科的多选题 |
| 多选题 | HellaSwag | 500 | 常识推理（完成句子） |
| Schema | GSM8K (Zero-Shot) | 500 | 数学问题（不使用工具） |
| LM | GPQA | 500 | 研究生级别的科学问题 |
| LM | MuSR | 500 | 多步推理 |
| LM | Lambada | 500 | 语言建模（预测最后一个词） |
| LM | SQuAD | 500 | 阅读理解 |

**评估频率**：
- 每 2000 步（可配置 `--core_metric_every=2000`）
- 每个任务最多评估 500 个样本（可配置 `--core_metric_max_per_task=500`）

**CORE Metric 计算**：
```python
# 从 base_eval.py
def evaluate_model(model, tokenizer, device, max_per_task=500):
    results = {}
    
    # 评估 9 个任务
    for task_name, task_config in core_tasks.items():
        accuracy = evaluate_task(
            model, 
            tokenizer, 
            task_config, 
            max_examples=max_per_task
        )
        results[task_name] = accuracy
    
    # Centered normalization（减去基线模型的表现）
    centered_results = {}
    for task_name, accuracy in results.items():
        baseline = BASELINES[task_name]  # 基线模型（如 random guess）
        centered_results[task_name] = accuracy - baseline
    
    # CORE Metric = 所有 centered 结果的平均值
    core_metric = sum(centered_results.values()) / len(centered_results)
    
    return {
        "core_metric": core_metric,
        "centered_results": centered_results,
        "raw_results": results
    }
```

**CORE Metric 的意义**：
- 📊 **多样性**：覆盖推理、常识、数学、阅读理解等多个维度
- 📊 **基线归一化**：减去随机猜测的表现，更准确反映模型能力
- 📊 **早期指标**：在预训练阶段就能看到模型的推理能力

### 训练配置

```python
# base_train.py 默认配置
depth = 20                    # 模型深度
max_seq_len = 2048           # 最大序列长度
device_batch_size = 32       # 每张卡的 batch size
total_batch_size = 524288    # 总 batch size（tokens）
target_param_data_ratio = 20 # Chinchilla ratio（数据:参数 = 20:1）

# 学习率
embedding_lr = 0.2           # Embedding 层（AdamW）
unembedding_lr = 0.004       # Unembedding 层（AdamW）
matrix_lr = 0.02             # 其他矩阵参数（Muon）
weight_decay = 0.0

# 学习率调度
warmup_ratio = 0.0           # 预热阶段占比
warmdown_ratio = 0.2         # 衰减阶段占比
final_lr_frac = 0.0          # 最终学习率倍数
```

**训练长度计算**：
```python
num_params = 模型参数量（例如 120M）
target_tokens = 20 * num_params  # Chinchilla ratio
num_iterations = target_tokens // total_batch_size

# 例如：120M 参数的模型
# target_tokens = 20 * 120M = 2.4B tokens
# num_iterations = 2.4B / 524288 ≈ 4577 steps
```

---

## 📊 第二阶段：Mid Training（中间训练）

### 训练数据

**数据集：TaskMixture（7个任务混合）**

#### 数据组成

```python
# 从 mid_train.py
train_dataset = TaskMixture([
    SmolTalk(split="train"),                 # 460K 对话
    MMLU(subset="auxiliary_train", split="train"),  # 100K 问题
    GSM8K(subset="main", split="train"),     # 8K 问题
    CustomJSON(filepath=identity_conversations_filepath),  # 1K 对话
    CustomJSON(filepath=identity_conversations_filepath),  # 1K 对话（2 epochs）
    SimpleSpelling(size=200000, split="train"),  # 200K 对话
    SpellingBee(size=80000, split="train"),  # 80K 对话
])
# 总计：460K + 100K + 8K + 1K + 1K + 200K + 80K = 850K 对话
```

#### 各任务详情

##### 1. SmolTalk - 通用对话（460K）

**来源**：HuggingFaceTB/smol-smoltalk
- **内容**：高质量的多轮对话
- **格式**：标准对话格式（可选 system message + user/assistant 交替）
- **用途**：教会模型进行自然的多轮对话

**数据示例**：
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

##### 2. MMLU - 多学科多选题（100K）

**来源**：auxiliary_train 子集
- **内容**：从 ARC、MC_TEST、OBQA、RACE 抽取的多选题
- **格式**：4选1 多选题
- **用途**：教会模型理解和回答多选题格式

**数据示例**：
```python
{
  "messages": [
    {
      "role": "user",
      "content": "Multiple Choice question: What is the chemical symbol for gold?\n- Gold=A\n- Au=B\n- Ag=C\n- Fe=D\n\nRespond only with the letter of the correct answer."
    },
    {
      "role": "assistant",
      "content": "B"
    }
  ]
}
```

##### 3. GSM8K - 数学推理（8K）

**来源**：GSM8K main 训练集
- **内容**：小学数学应用题
- **格式**：对话式，带工具调用（calculator）
- **用途**：教会模型进行数学推理和使用工具

**数据示例**：
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. How many does she sell at the farmers' market daily?"
    },
    {
      "role": "assistant",
      "content": "Let me calculate this step by step.\n<tool>calculator(16-3-4)</tool>\nShe sells 9 eggs daily."
    }
  ]
}
```

##### 4. CustomJSON - 身份对话（2K，2 epochs）

**来源**：identity_conversations.jsonl（项目自定义）
- **内容**：关于模型身份、能力、限制的对话
- **格式**：标准对话格式
- **用途**：教会模型认识自己的身份（"I am nanochat"）
- **特殊处理**：2 epochs（出现两次）以强化记忆

**数据示例**：
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is your name?"
    },
    {
      "role": "assistant",
      "content": "I am nanochat, an AI assistant created to help answer questions and have conversations."
    }
  ]
}
```

##### 5. SimpleSpelling - 简单拼写（200K）

**来源**：程序化生成
- **内容**：简单拼写任务（如"spell the word 'apple'"）
- **格式**：对话式
- **用途**：教会模型理解和处理字符级任务

**数据示例**：
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Spell the word 'hello'."
    },
    {
      "role": "assistant",
      "content": "h-e-l-l-o"
    }
  ]
}
```

##### 6. SpellingBee - 字母计数（80K）

**来源**：程序化生成
- **内容**：字母计数任务（如"how many 'r' in 'strawberry'?"）
- **格式**：对话式，带工具调用
- **用途**：教会模型进行细粒度的字符级推理

**数据示例**：
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How many times does the letter 'r' appear in the word 'strawberry'?"
    },
    {
      "role": "assistant",
      "content": "<tool>count_letter('strawberry', 'r')</tool>\nThe letter 'r' appears 3 times in 'strawberry'."
    }
  ]
}
```

#### 数据混合策略

```python
# TaskMixture 实现（从 tasks/common.py）
class TaskMixture(Task):
    def __init__(self, tasks):
        self.tasks = tasks
        self.lengths = [len(task) for task in tasks]
        
        # 构建索引映射
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        
        # 确定性 shuffle（seed=42）
        rng = random.Random(42)
        rng.shuffle(self.index_map)
    
    def get_example(self, index):
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]
```

**混合特点**：
- ✅ **确定性 shuffle**：使用固定 seed=42，保证可复现
- ✅ **任务均匀分布**：不同任务的样本在整个 epoch 中均匀分布
- ✅ **支持 oversampling**：可以多次添加同一个任务（如 identity_conversations × 2）

### 验证数据

**数据集：TaskMixture（3个任务混合）**

```python
# 从 mid_train.py
val_dataset = TaskMixture([
    SmolTalk(split="test"),           # 24K 对话
    MMLU(subset="all", split="test", stop=5200),  # 5.2K 问题（从 14K 中取）
    GSM8K(subset="main", split="test", stop=420),  # 420 问题（从 1.3K 中取）
])
# 总计：24K + 5.2K + 0.42K ≈ 30K 对话
```

**验证数据特点**：
- 📊 **比例匹配**：验证集的任务比例与训练集相似
- 📊 **数据切片**：MMLU 和 GSM8K 使用 `stop` 参数控制样本数量
- 📊 **评估指标**：BPB（与 Base 训练相同的指标）

#### 评估配置

```python
# 从 mid_train.py
eval_every = 150              # 每 150 步评估一次
eval_tokens = 20 * 524288     # 评估 10M tokens
```

### 训练配置

```python
# mid_train.py 默认配置
max_seq_len = 2048
device_batch_size = 32
total_batch_size = 524288    # 与 Base 训练相同

# 学习率（与 Base 训练相同）
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 1.0           # 初始 LR 倍数

# 学习率调度（与 Base 不同！）
def get_lr_multiplier(progress):
    # 前 80% 不衰减，后 20% 线性衰减到 0
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2
```

**训练长度**：
- **默认**：1 epoch over 850K 对话
- **计算**：num_iterations = 由用户指定或自动计算（基于 dataset_size）

---

## 📊 第三阶段：SFT Training（监督微调）

### 训练数据

**数据集：TaskMixture（7个任务混合）**

#### 数据组成

```python
# 从 chat_sft.py
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),      # 2.3K 对话
    ARC(subset="ARC-Challenge", split="train"), # 1.1K 对话
    GSM8K(subset="main", split="train"),        # 8K 对话
    SmolTalk(split="train", stop=10_000),       # 10K 对话
    CustomJSON(filepath=identity_conversations_filepath),  # 1K 对话
    SimpleSpelling(size=300, split="train"),    # 300 对话
    SpellingBee(size=300, split="train"),       # 300 对话
])
# 总计：2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K 对话
```

#### 数据规模对比

| 任务 | Mid Training | SFT Training | 变化说明 |
|------|-------------|--------------|---------|
| SmolTalk | 460K | 10K | 大幅减少，只用 2% |
| MMLU | 100K (auxiliary) | 0 | 完全移除 |
| ARC | 0 | 3.4K (Easy+Challenge) | 新增 |
| GSM8K | 8K | 8K | 保持不变 |
| Identity | 2K | 1K | 减半 |
| SimpleSpelling | 200K | 300 | 大幅减少 |
| SpellingBee | 80K | 300 | 大幅减少 |
| **总计** | **850K** | **23K** | **减少 97%** |

**数据变化的原因**：
- 🎯 **精简高效**：SFT 只需要少量高质量数据
- 🎯 **去除重复**：模型在 Mid 阶段已经学会了对话格式
- 🎯 **聚焦能力**：重点放在推理能力（ARC、GSM8K）
- 🎯 **防止过拟合**：小数据集配合小学习率（0.02x）

#### ARC 任务详情

**来源**：AI2 Reasoning Challenge
- **ARC-Easy**：2.3K 小学级别科学推理题
- **ARC-Challenge**：1.1K 中学级别科学推理题
- **格式**：4选1 多选题
- **用途**：提升科学推理能力

**数据示例**：
```python
{
  "messages": [
    {
      "role": "user",
      "content": "Multiple Choice question: Which of the following is a renewable resource?\n- Coal=A\n- Oil=B\n- Solar energy=C\n- Natural gas=D\n\nRespond only with the letter of the correct answer."
    },
    {
      "role": "assistant",
      "content": "C"
    }
  ]
}
```

### SFT 的核心创新：Mask 机制

**关键问题**：在对话训练中，我们只想让模型学习 assistant 的回复，而不是 user 的问题。

**解决方案**：使用 mask 标记哪些 token 需要计算 loss。

#### Mask 生成过程

```python
# 从 tokenizer.py
def render_conversation(self, conversation):
    """
    返回：
    - ids: List[int]，完整的 token 序列
    - mask: List[int]，0/1 标记（1=计算loss，0=不计算）
    """
    ids = []
    mask = []
    
    # 添加 BOS token
    ids.append(self.encode_special("<|bos|>"))
    mask.append(0)  # BOS 不参与 loss
    
    for message in conversation["messages"]:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            # User message: 全部 mask=0
            ids.extend(self.encode_special("<|user_start|>"))
            ids.extend(self.encode(content))
            ids.extend(self.encode_special("<|user_end|>"))
            mask.extend([0] * (len(ids) - len(mask)))
        
        elif role == "assistant":
            # Assistant message: 只有 content 部分 mask=1
            ids.append(self.encode_special("<|assistant_start|>"))
            mask.append(0)  # start token 不参与 loss
            
            content_ids = self.encode(content)
            ids.extend(content_ids)
            mask.extend([1] * len(content_ids))  # ← 关键！只有这里是 1
            
            ids.append(self.encode_special("<|assistant_end|>"))
            mask.append(1)  # end token 参与 loss
    
    return ids, mask
```

#### Loss 计算过程

```python
# 从 chat_sft.py
def sft_data_generator(dataset, batch_size):
    batch = []
    for conversation in dataset:
        ids, mask = tokenizer.render_conversation(conversation)
        batch.append((ids, mask))
        
        if len(batch) == batch_size:
            # Collate 成张量
            inputs = ...   # (B, T)
            targets = ...  # (B, T)
            
            # 关键：将 mask=0 的位置设为 -1（ignore index）
            for i, (ids, mask) in enumerate(batch):
                row_targets = ids[1:]  # shift right
                mask_tensor = mask[1:]
                row_targets[mask_tensor == 0] = -1  # ← 设为 ignore index
                targets[i] = row_targets
            
            yield inputs, targets
```

**CrossEntropyLoss 的处理**：
```python
# PyTorch 内部
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # (B*T, V)
    targets.view(-1),              # (B*T,)
    ignore_index=-1                # ← targets=-1 的位置不计算 loss
)
```

**Mask 机制的效果**：
- ✅ **只学习生成**：模型只在 assistant 的回复上计算梯度
- ✅ **保留上下文**：user 的输入仍然参与 forward pass，提供上下文
- ✅ **精确控制**：可以灵活控制哪些 token 参与训练（如工具调用）

### 验证数据

**数据集：SmolTalk（测试集）**

```python
# 从 chat_sft.py
val_ds = SmolTalk(split="test")  # 24K 对话
```

**为什么只用 SmolTalk？**
- 📊 **通用性**：SmolTalk 涵盖各种对话场景
- 📊 **大规模**：24K 对话足够评估模型的对话能力
- 📊 **快速评估**：不需要运行慢速的生成式评估

#### 评估指标

##### 1. Validation Loss（每 100 步）

```python
# 从 chat_sft.py
eval_every = 100
eval_steps = 100

# 评估过程
for _ in range(eval_steps):
    val_inputs, val_targets = next(val_loader)
    with torch.no_grad():
        loss = model(val_inputs, val_targets)
    losses.append(loss)

val_loss = torch.stack(losses).mean()
```

##### 2. Task Accuracy（每 200 步）

```python
# 从 chat_sft.py
eval_metrics_every = 200
eval_metrics_max_problems = 1024

metrics = {
    "mmlu_acc": run_chat_eval("MMLU", ...),
    "arc_easy_acc": run_chat_eval("ARC-Easy", ...),
    "arc_challenge_acc": run_chat_eval("ARC-Challenge", ...),
    "gsm8k_maj1_acc": run_chat_eval("GSM8K", ..., num_samples=1),
    "gsm8k_maj16_acc": run_chat_eval("GSM8K", ..., num_samples=16),
}
```

**评估任务详情**：

| 任务 | 评估类型 | 样本数 | 指标含义 |
|------|---------|--------|---------|
| MMLU | Categorical | 1024 | 57个学科的准确率 |
| ARC-Easy | Categorical | 1024 | 小学科学推理准确率 |
| ARC-Challenge | Categorical | 1024 | 中学科学推理准确率 |
| GSM8K (maj@1) | Generative | 1024 | 单次采样的准确率 |
| GSM8K (maj@16) | Generative | 1024 | 16次采样的多数投票准确率 |

**Categorical vs Generative**：
- **Categorical**：比较 logits，不需要采样，速度快（~15 分钟）
- **Generative**：需要生成完整回答，解析答案，速度慢（~45 分钟）

### 训练配置

```python
# chat_sft.py 默认配置
device_batch_size = 4         # 小 batch size（避免 OOM）
target_examples_per_step = 32 # 每步处理 32 个对话
num_epochs = 1                # 1 epoch

# 学习率（非常小！）
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 0.02          # ← 关键！只用 2% 的 LR
# 实际 LR = 基础 LR × 0.02
# 例如：matrix_lr = 0.02 × 0.02 = 0.0004

weight_decay = 0.0

# 学习率调度（线性衰减到 0）
def get_lr_multiplier(it):
    return 1.0 - it / num_iterations
```

**为什么用这么小的学习率？**
- 🎯 **防止遗忘**：模型在 Base 和 Mid 阶段已经学到很多知识
- 🎯 **精细调整**：SFT 只是微调对话格式和输出风格
- 🎯 **小数据集**：23K 样本很容易过拟合

**训练长度计算**：
```python
dataset_size = 23000
target_examples_per_step = 32
num_epochs = 1

num_iterations = (dataset_size // target_examples_per_step) * num_epochs
# = (23000 // 32) * 1
# = 718 steps
```

---

## 📊 第四阶段：RL Training（强化学习）

### 训练数据

**数据集：GSM8K（训练集）**

```python
# 从 chat_rl.py
train_task = GSM8K(subset="main", split="train")
# 7473 个数学问题
```

#### 数据规模
- **问题数量**：7473 个小学数学应用题
- **格式**：单轮问答（user 提问，assistant 回答）
- **难度**：需要多步推理和工具使用

#### 数据特点

**GSM8K 问题示例**：
```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. 
She sells the remainder at the farmers' market daily for $2 per fresh 
duck egg. How much in dollars does she make every day at the farmers' market?

Answer: 18
```

**关键特性**：
- 🧮 **多步推理**：需要分解问题，逐步计算
- 🛠️ **工具使用**：需要调用 `<tool>calculator(...)</tool>`
- 📝 **自然语言**：答案需要用自然语言解释推理过程
- ✅ **明确答案**：每个问题有唯一的数值答案

### RL 训练机制

#### 核心思想：GRPO（简化版）

```python
"""
GRPO = Group Relative Policy Optimization

简化版：
1. 删除 trust region（无 KL 散度约束）
2. On-policy（无需 PPO ratio+clip）
3. Token-level 优势归一化（GAPO 风格）
4. 只用 (r - mu) 作为 advantage（不除以 sigma）
"""
```

#### 训练流程

```python
# 从 chat_rl.py
examples_per_step = 16  # 每步处理 16 个问题
num_samples = 16        # 每个问题生成 16 个回答

@torch.no_grad()
def get_batch():
    # 遍历训练集
    for example_idx in range(ddp_rank, len(train_task), ddp_world_size):
        
        # 1. 获取问题
        conversation = train_task[example_idx]
        
        # 2. Tokenize（保留 <|assistant_start|>，删除后面的内容）
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        
        # 3. 生成 16 个回答（批量生成，避免 OOM）
        model.eval()
        generated_token_sequences = []
        masks = []
        
        for sampling_step in range(num_samples // device_batch_size):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            
            sequences, masks_batch = engine.generate_batch(
                tokens,
                num_samples=device_batch_size,
                max_tokens=max_new_tokens,
                temperature=1.0,  # ← 高温度，鼓励探索
                top_k=50,
                seed=seed
            )
            
            generated_token_sequences.extend(sequences)
            masks.extend(masks_batch)
        
        # 4. 计算每个回答的奖励
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            
            # 奖励函数：答案是否正确
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)
        
        # 5. Padding 使所有序列等长
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [
            seq + [pad_token] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]
        
        # 6. 转换为张量
        ids = torch.tensor(padded_sequences, device=device)  # (16, T)
        mask_ids = torch.tensor(padded_masks, device=device)  # (16, T)
        
        inputs = ids[:, :-1]   # (16, T-1)
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1  # mask out
        
        rewards = torch.tensor(rewards, device=device)  # (16,)
        
        # 7. 计算 advantage（只减均值，不除标准差）
        mu = rewards.mean()
        advantages = rewards - mu  # (16,)
        
        # 8. Yield batch
        yield sequences, inputs, targets, rewards, advantages
```

#### 奖励函数

```python
# 从 tasks/gsm8k.py
def reward(self, problem, completion):
    """
    返回：1.0（正确）或 0.0（错误）
    """
    # 1. 从 completion 中提取答案
    predicted_answer = self.extract_answer(completion)
    
    # 2. 获取正确答案
    correct_answer = problem["answer"]
    
    # 3. 比较（需要处理数值格式：逗号、小数点等）
    return 1.0 if self.answers_match(predicted_answer, correct_answer) else 0.0

def extract_answer(self, completion):
    """
    从生成的文本中提取答案
    支持多种格式：
    - "The answer is 42"
    - "#### 42"
    - "<tool>calculator(...)</tool> = 42"
    """
    # 正则表达式匹配
    patterns = [
        r"#### (\d+)",
        r"The answer is (\d+)",
        r"= (\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, completion)
        if match:
            return match.group(1)
    return None
```

#### Policy Gradient 更新

```python
# 从 chat_rl.py
# Training loop
for step in range(num_steps):
    
    # 1. 获取 batch（包含 rollouts）
    sequences, inputs, targets, rewards, advantages = next(get_batch())
    # inputs: (16, T-1)
    # targets: (16, T-1)，mask 后
    # rewards: (16,)
    # advantages: (16,)
    
    # 2. Forward pass（计算 log probs）
    model.train()
    logits = model(inputs)  # (16, T-1, V)
    
    log_probs = F.log_softmax(logits, dim=-1)  # (16, T-1, V)
    
    # 3. 收集 targets 位置的 log probs
    # log_probs_taken: (16, T-1)
    log_probs_taken = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)
    
    # 4. Mask out padding 和 prompt
    mask = (targets != -1).float()  # (16, T-1)
    log_probs_taken = log_probs_taken * mask
    
    # 5. Policy Gradient Loss
    # L = -mean(log_prob * advantage)
    advantages_expanded = advantages.unsqueeze(-1)  # (16, 1)
    
    # Token-level loss
    token_losses = -log_probs_taken * advantages_expanded  # (16, T-1)
    
    # 只对非 mask 的 token 求均值
    loss = (token_losses * mask).sum() / mask.sum()
    
    # 6. Backward + Optimizer step
    loss.backward()
    
    # Gradient accumulation
    if (step + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Policy Gradient 的直觉**：
- 🎯 **Advantage > 0**：奖励高于平均 → 增大该 trajectory 的概率
- 🎯 **Advantage < 0**：奖励低于平均 → 减小该 trajectory 的概率
- 🎯 **Token-level**：每个 token 都用同一个 advantage，简化计算

### 验证数据

**数据集：GSM8K（测试集）**

```python
# 从 chat_rl.py
val_task = GSM8K(subset="main", split="test")
# 1319 个问题
```

#### 评估指标：Pass@k

**定义**：从 k 次采样中，至少有 1 次正确的概率。

**计算公式**：
```
Pass@k = Σ [1 if any(samples) is correct else 0] / num_examples
```

**评估过程**：

```python
# 从 chat_rl.py
def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    max_examples=400,
    num_samples=16,
    temperature=1.0,
):
    correct_counts = {1: 0, 4: 0, 16: 0}
    total = 0
    
    for problem in task[:max_examples]:
        conversation = problem
        tokens = tokenizer.render_for_completion(conversation)
        
        # 生成 num_samples 个回答
        sequences, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=256,
            temperature=temperature,
            top_k=50,
        )
        
        # 评估每个回答
        is_correct = []
        for seq in sequences:
            generated_text = tokenizer.decode(seq[len(tokens):])
            reward = task.reward(conversation, generated_text)
            is_correct.append(reward == 1.0)
        
        # 计算 Pass@k（k = 1, 4, 16）
        correct_counts[1] += int(is_correct[0])  # Pass@1 = 第 1 次正确
        correct_counts[4] += int(any(is_correct[:4]))  # Pass@4
        correct_counts[16] += int(any(is_correct[:16]))  # Pass@16
        
        total += 1
    
    return {
        "pass@1": correct_counts[1] / total,
        "pass@4": correct_counts[4] / total,
        "pass@16": correct_counts[16] / total,
    }
```

**Pass@k 的意义**：
- 📊 **Pass@1**：模型的"最佳猜测"准确率（类似 greedy decoding）
- 📊 **Pass@4**：给模型 4 次机会，能否找到正确答案
- 📊 **Pass@16**：给模型 16 次机会，测试探索能力
- 📊 **趋势**：Pass@1 < Pass@4 < Pass@16（模型的探索能力）

**评估配置**：
```python
eval_every = 60              # 每 60 步评估一次
eval_examples = 400          # 使用 400 个测试样本
num_samples = 16             # 每个问题生成 16 次
temperature = 1.0            # 高温度，保持探索性
```

### 训练配置

```python
# chat_rl.py 默认配置
device_batch_size = 8        # 每张卡的 batch size
examples_per_step = 16       # 每步处理 16 个问题
num_samples = 16             # 每个问题生成 16 个回答
max_new_tokens = 256         # 最大生成长度
temperature = 1.0            # 采样温度（高温度，鼓励探索）
top_k = 50                   # Top-k 采样

# 学习率（与 SFT 相似）
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05          # 5% 的基础 LR

num_epochs = 1               # 1 epoch over GSM8K
```

**训练长度计算**：
```python
dataset_size = 7473
examples_per_step = 16
num_epochs = 1

num_steps = (dataset_size // examples_per_step) * num_epochs
# = (7473 // 16) * 1
# = 467 steps
```

**为什么用高温度（temperature=1.0）？**
- 🎯 **鼓励探索**：生成多样化的回答
- 🎯 **Off-policy 学习**：即使某些回答不好，也能从中学习（通过负 advantage）
- 🎯 **避免模式崩塌**：防止模型只生成一种类型的回答

---

## 📊 四阶段数据流转图

```
┌─────────────────────────────────────────────────────────────┐
│                    Base Training (预训练)                    │
├─────────────────────────────────────────────────────────────┤
│ 训练数据: FineWeb-Edu (100B tokens, 1822 shards)            │
│ 验证数据: FineWeb-Edu (10M tokens, 1 shard)                 │
│ 评估指标: BPB + CORE Metric (9 tasks)                       │
│ 学习目标: 语言建模 + 基础推理能力                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Mid Training (中间训练)                   │
├─────────────────────────────────────────────────────────────┤
│ 训练数据: 7 任务混合 (850K 对话)                            │
│   - SmolTalk: 460K (通用对话)                               │
│   - MMLU: 100K (多选题)                                     │
│   - GSM8K: 8K (数学推理)                                    │
│   - Identity: 2K (身份认知)                                 │
│   - SimpleSpelling: 200K (拼写)                             │
│   - SpellingBee: 80K (字母计数)                             │
│ 验证数据: 3 任务混合 (30K 对话)                             │
│ 评估指标: BPB                                                │
│ 学习目标: 对话格式 + 任务能力 + 工具使用                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    SFT Training (监督微调)                   │
├─────────────────────────────────────────────────────────────┤
│ 训练数据: 7 任务混合 (23K 对话，精简 97%)                   │
│   - ARC: 3.4K (科学推理)                                    │
│   - GSM8K: 8K (数学推理)                                    │
│   - SmolTalk: 10K (通用对话)                                │
│   - Identity: 1K (身份认知)                                 │
│   - Spelling: 600 (拼写+计数)                               │
│ 验证数据: SmolTalk (24K 对话)                               │
│ 评估指标: Loss + MMLU/ARC/GSM8K 准确率                      │
│ 学习目标: 指令遵循 + 输出格式控制                            │
│ 核心技术: Mask 机制 (只在 assistant 回复上计算 loss)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    RL Training (强化学习)                    │
├─────────────────────────────────────────────────────────────┤
│ 训练数据: GSM8K (7.5K 问题)                                 │
│ 验证数据: GSM8K (1.3K 问题)                                 │
│ 评估指标: Pass@k (k=1,4,16)                                 │
│ 学习目标: 多步推理 + 试错探索                                │
│ 核心技术: Policy Gradient (每题采样 16 次)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 数据量变化趋势

### Token / 样本数量对比

| 阶段 | 训练数据量 | 验证数据量 | 数据类型 |
|------|-----------|-----------|---------|
| Base | 100B tokens | 10M tokens | 纯文本 |
| Mid | 850K 对话 ≈ 1.7B tokens | 30K 对话 ≈ 60M tokens | 结构化对话 |
| SFT | 23K 对话 ≈ 46M tokens | 24K 对话 ≈ 48M tokens | 结构化对话 |
| RL | 7.5K 问题 × 16 样本 = 120K 样本 | 1.3K 问题 × 16 样本 | 单轮问答 |

### 学习率变化趋势

| 阶段 | Matrix LR | Embedding LR | Unembedding LR | LR 倍数 |
|------|-----------|-------------|---------------|--------|
| Base | 0.02 | 0.2 | 0.004 | 1.0 |
| Mid | 0.02 | 0.2 | 0.004 | 1.0 |
| SFT | 0.02 | 0.2 | 0.004 | **0.02** |
| RL | 0.02 | 0.2 | 0.004 | **0.05** |

**学习率策略**：
- Base/Mid：大 LR，快速学习
- SFT：极小 LR（2%），防止遗忘
- RL：小 LR（5%），精细调整推理能力

### 评估指标演变

```
Base: BPB + CORE Metric
  ↓
  评估语言建模能力和基础推理
  
Mid: BPB
  ↓
  评估对话格式的掌握程度
  
SFT: Loss + Task Accuracy (MMLU/ARC/GSM8K)
  ↓
  评估指令遵循和格式控制
  
RL: Pass@k (k=1,4,16)
  ↓
  评估多步推理和探索能力
```

---

## 🎯 核心发现与洞察

### 1. 数据规模的阶梯式递减

**规律**：Base (100B) → Mid (1.7B) → SFT (46M) → RL (on-policy 生成)

**原因**：
- 🔍 **知识积累**：Base 需要大量数据学习语言和世界知识
- 🔍 **能力迁移**：Mid 复用 Base 的知识，只需学习新格式
- 🔍 **精细调整**：SFT 只需少量高质量数据调整输出风格
- 🔍 **在线学习**：RL 通过 on-policy 采样，无需离线数据

### 2. 任务混合的策略演变

**Base → Mid**：
- 从纯文本 → 结构化对话
- 引入任务数据（MMLU、GSM8K、Spelling）
- 教会模型工具使用

**Mid → SFT**：
- 大幅减少数据量（97% 减少）
- 去除大规模拼写任务（200K → 300）
- 引入 ARC，强化推理能力
- 使用 Mask 机制，只在 assistant 回复上学习

**SFT → RL**：
- 聚焦单一任务（GSM8K）
- 从监督学习 → 强化学习
- 从固定答案 → 试错探索

### 3. 评估指标的精细化

**BPB**：
- 适用于 Base 和 Mid 阶段
- 优势：tokenizer 无关，可比较不同模型
- 局限：只能评估语言建模，不能评估推理能力

**CORE Metric**：
- 适用于 Base 阶段
- 优势：多任务综合评估，早期发现推理能力
- 成本：需要 162MB 数据，评估较慢

**Task Accuracy**：
- 适用于 SFT 阶段
- 优势：直接评估任务表现
- 分类：Categorical（快）vs Generative（慢）

**Pass@k**：
- 适用于 RL 阶段
- 优势：评估探索能力，而非单次准确率
- 洞察：Pass@16 >> Pass@1 说明模型有很强的探索能力

### 4. Mask 机制的重要性

**问题**：如何让模型只学习生成回答，而不是记住问题？

**解决方案**：
```python
# User 的输入：mask = 0（不计算 loss）
# Assistant 的回复：mask = 1（计算 loss）
targets[mask == 0] = -1  # ignore_index
```

**效果**：
- ✅ 模型只在 assistant 回复上更新梯度
- ✅ 保留 user 输入的上下文信息
- ✅ 防止模型记住训练数据的问题

### 5. 强化学习的简化设计

**GRPO 的简化**：
- ❌ 删除 KL 散度约束（trust region）
- ❌ 删除 PPO ratio + clip
- ✅ 保留 token-level 优势归一化
- ✅ 使用 (r - mu) 而非 (r - mu) / sigma

**效果**：
- 🚀 **训练稳定**：On-policy 学习，无需担心策略偏移
- 🚀 **实现简单**：只需要计算 policy gradient
- 🚀 **效果良好**：Pass@k 持续提升

---

## 📋 快速参考表

### 运行命令

```bash
# Base Training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --run=base_run \
    --depth=20 \
    --device_batch_size=32

# Mid Training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    --run=mid_run \
    --model_tag=d20 \
    --step=4500 \
    --device_batch_size=32

# SFT Training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --run=sft_run \
    --source=mid \
    --model_tag=d20 \
    --device_batch_size=4

# RL Training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
    --run=rl_run \
    --source=sft \
    --device_batch_size=8
```

### 评估命令

```bash
# Base Model Evaluation
python -m scripts.base_eval --model_tag=d20 --step=4500

# Chat Model Evaluation
python -m scripts.chat_eval --source=sft --model_tag=d20

# GSM8K Pass@k Evaluation
# (集成在 RL 训练中，每 60 步自动运行)
```

### 数据准备

```bash
# 下载 Base 训练数据（自动下载）
# FineWeb-Edu 会在训练时按需下载

# 下载 CORE Metric 评估数据
# 评估时会自动下载 eval_bundle.zip

# 准备 Identity Conversations（需要手动创建）
# 文件位置: <base_dir>/identity_conversations.jsonl
```

---

## 🎓 总结

nanochat 项目的四阶段训练展示了现代 LLM 训练的完整流程：

1. **Base Training**：在海量文本上学习语言和知识
2. **Mid Training**：学习对话格式和任务能力
3. **SFT Training**：精细调整输出格式和指令遵循
4. **RL Training**：通过强化学习提升推理和探索能力

每个阶段都有明确的数据来源、评估指标和训练目标，形成了一个完整的训练流程。

**关键设计原则**：
- ✅ **数据质量 > 数量**：SFT 只用 23K 样本，但效果显著
- ✅ **学习率递减**：Base (1.0x) → SFT (0.02x)，防止遗忘
- ✅ **评估指标演变**：从 BPB → Accuracy → Pass@k，逐步精细化
- ✅ **渐进式学习**：每个阶段都基于前一阶段的 checkpoint

这份报告涵盖了所有训练和验证数据的详细信息，希望能帮助你全面理解 nanochat 的训练流程！🚀
