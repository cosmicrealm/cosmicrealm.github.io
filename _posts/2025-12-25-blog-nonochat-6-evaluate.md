---
title: 'noao-chat-6-训练评估指南'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-25-blog-nonochat-6-evaluate/
tags:
  - llm
---

### LLM 模型评估验证完全指南

[nonochat](https://github.com/karpathy/nanochat)

> 本文档全面解析 nanochat 项目中的模型评估体系，从评估指标到具体任务，从实现细节到最佳实践。

---

## 目录

1. [评估体系概览](#1-评估体系概览)
2. [评估指标详解](#2-评估指标详解)
3. [评估任务分类](#3-评估任务分类)
4. [Base 模型评估](#4-base-模型评估)
5. [Chat 模型评估](#5-chat-模型评估)
6. [实现细节剖析](#6-实现细节剖析)
7. [评估流程实战](#7-评估流程实战)
8. [最佳实践与优化](#8-最佳实践与优化)

---

## 1. 评估体系概览

### 1.1 评估金字塔

```
                    ┌─────────────────┐
                    │  Pass@k (RL)   │  最终能力
                    │  强化学习优化   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  Accuracy (SFT) │  任务准确率
                    │  MMLU, ARC等    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  BPB (Mid/SFT)  │  验证损失
                    │  每字节比特数   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  CORE (Base)    │  基础能力
                    │  9类任务平均    │
                    └─────────────────┘
```

### 1.2 四阶段评估对应

| 训练阶段 | 主要评估指标 | 辅助指标 | 评估频率 |
|---------|------------|---------|---------|
| **Base** | BPB, CORE Metric | 采样质量 | 每 250 步 BPB<br>每 2000 步 CORE |
| **Mid** | BPB | 任务准确率 | 每 250 步 |
| **SFT** | BPB, Task Accuracy | MMLU, ARC | 每 100 步 BPB<br>每 200 步 Accuracy |
| **RL** | Pass@k | 平均奖励 | 每 60 步 |

### 1.3 评估文件结构

```
nanochat/
├── loss_eval.py           # BPB 计算
├── core_eval.py           # CORE Metric 评估
└── engine.py              # 生成引擎（采样）

scripts/
├── base_eval.py           # Base 模型完整评估
├── base_loss.py           # 快速 BPB 评估
├── chat_eval.py           # Chat 模型评估入口
└── tok_eval.py            # Tokenizer 评估

tasks/
├── common.py              # Task 基类
├── gsm8k.py               # 数学推理
├── mmlu.py                # 多领域知识
├── arc.py                 # 科学推理
├── humaneval.py           # 代码生成
└── spellingbee.py         # 拼写计数
```

---

## 2. 评估指标详解

### 2.1 BPB (Bits Per Byte)

**定义：** 每字节的比特数，衡量模型的语言建模能力。

**公式：**

$$
\text{BPB} = \frac{\sum \text{loss} \times \text{token\_bytes}}{\sum \text{token\_bytes}}
$$

**特点：**
- ✅ **独立于词表大小**：不同 tokenizer 可比较
- ✅ **字节级归一化**：更公平的度量
- ✅ **排除特殊 token**：只计算实际文本

**代码实现：**

```python
@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    计算 Bits Per Byte
    
    Args:
        model: 要评估的模型
        batches: 数据迭代器
        steps: 评估步数
        token_bytes: (vocab_size,) 张量，每个 token 的字节数
    
    Returns:
        float: BPB 值
    """
    total_nats = 0.0  # 总自然对数损失
    total_bytes = 0   # 总字节数
    
    for _ in range(steps):
        x, y = next(batches)
        
        # 计算每个位置的损失（不降维）
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        
        # 获取每个 token 的字节数
        valid = y >= 0  # 排除 ignore_index=-1
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        num_bytes = token_bytes[y_safe]  # (B, T)
        
        # 累积
        total_nats += (loss2d * (num_bytes > 0)).sum()
        total_bytes += num_bytes.sum()
    
    # 转换：nats → bits, 归一化
    bpb = total_nats / (total_bytes * math.log(2))
    return bpb
```

**典型值：**

| 模型阶段 | 训练 BPB | 验证 BPB | 说明 |
|---------|---------|---------|------|
| Base 初始 | 4.5 | 4.6 | 随机初始化 |
| Base 训练后 | 3.2 | 3.45 | 基础语言能力 |
| Mid 训练后 | 3.0 | 3.2 | 增强任务能力 |
| SFT 训练后 | 2.8 | 3.0 | 指令遵循 |

### 2.2 CORE Metric

**定义：** Comprehensive Objective Reference Evaluation，综合基准测试。

**组成：** 9 类任务的平均准确率

```python
CORE_TASKS = [
    # 1. 多选题 (Multiple Choice)
    "arc_easy", "arc_challenge",
    "hellaswag", "piqa", "winogrande",
    
    # 2. 模式匹配 (Schema)
    "copa", "openbookqa",
    
    # 3. 语言建模 (Language Modeling)
    "lambada", "wikitext"
]

core_metric = mean([accuracy(task) for task in CORE_TASKS])
```

**任务类型：**

1. **Multiple Choice（多选题）**
   ```
   问题: What is the capital of France?
   选项: A) London  B) Paris  C) Berlin  D) Madrid
   模型: 选择损失最低的选项
   ```

2. **Schema（模式匹配）**
   ```
   多个上下文 → 同一结论
   选择最匹配的上下文
   ```

3. **Language Modeling（语言建模）**
   ```
   前缀 + 续写
   检查是否完全匹配参考答案
   ```

**评估流程：**

```python
def evaluate_task(model, tokenizer, data, device, task_meta):
    correct = torch.zeros(len(data))
    
    for idx in range(len(data)):
        item = data[idx]
        
        # 1. 采样 few-shot 示例
        fewshot_examples = sample_fewshot(data, idx, num_fewshot)
        
        # 2. 渲染 prompt
        prompts = render_prompts(item, fewshot_examples, task_type)
        
        # 3. Tokenize
        tokens = tokenizer(prompts)
        
        # 4. 前向传播，计算损失
        losses = model(tokens)
        
        # 5. 判断正确性
        if task_type == 'multiple_choice':
            # 选择损失最小的选项
            pred_idx = losses.argmin()
            is_correct = (pred_idx == item['gold'])
        elif task_type == 'language_modeling':
            # 检查生成是否匹配
            is_correct = (predictions == targets).all()
        
        correct[idx] = is_correct
    
    return correct.mean()
```

### 2.3 Task Accuracy

**定义：** 特定任务的准确率。

**计算方式：**

```python
accuracy = correct_count / total_count
```

**常见任务：**

| 任务 | 类型 | 评估方式 | 典型准确率 |
|-----|------|---------|----------|
| **MMLU** | 分类 | 选择题 (A/B/C/D) | 40-60% |
| **ARC** | 分类 | 科学问答 | 60-80% |
| **GSM8K** | 生成 | 数学答案匹配 | 30-50% |
| **HumanEval** | 生成 | 代码执行 | 20-40% |
| **SpellingBee** | 生成 | 字母计数 | 70-90% |

### 2.4 Pass@k

**定义：** 给 k 次机会，至少一次正确的概率。

**公式：**

$$
\text{Pass@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{any}([\text{eval}(s_{ij}) \text{ for } j \in [1,k]])]
$$

**实现：**

```python
def evaluate_pass_at_k(model, task, k=8):
    """
    评估 Pass@k
    
    Args:
        model: 模型
        task: 任务对象
        k: 采样次数
    
    Returns:
        float: Pass@k 准确率
    """
    correct = 0
    total = 0
    
    for question in task:
        # 生成 k 个答案
        answers = model.generate(
            question,
            num_samples=k,
            temperature=1.0  # 保持随机性
        )
        
        # 评估每个答案
        results = [task.evaluate(question, ans) for ans in answers]
        
        # 至少一个正确 → 算作 Pass
        passed = any(results)
        
        correct += int(passed)
        total += 1
    
    return correct / total
```

**典型值（GSM8K）：**

```python
# SFT 模型
Pass@1 = 0.45   # 第一次尝试的准确率
Pass@8 = 0.72   # 给 8 次机会的准确率

# RL 模型（训练后）
Pass@1 = 0.53   # 提升 8%
Pass@8 = 0.78   # 提升 6%
```

**Pass@k 的意义：**
- 衡量模型的**探索能力**
- 衡量模型的**鲁棒性**
- 对于复杂推理，高 Pass@k 说明模型能找到正确路径

---

## 3. 评估任务分类

### 3.1 任务分类体系

```
评估任务
├── 分类任务 (Categorical)
│   ├── 多选题 (Multiple Choice)
│   │   ├── MMLU (知识)
│   │   ├── ARC (科学)
│   │   └── CORE 任务
│   └── 二分类
│       └── 正确/错误判断
│
└── 生成任务 (Generative)
    ├── 数学推理
    │   └── GSM8K
    ├── 代码生成
    │   └── HumanEval
    └── 文本生成
        └── SpellingBee
```

### 3.2 MMLU (知识广度)

**数据集：** 57 个学科的多选题

**示例：**

```python
# 问题
question = "What is the primary function of mitochondria?"

# 选项
choices = [
    "Protein synthesis",
    "Energy production",      # ← 正确答案
    "DNA replication",
    "Cell division"
]

# 用户消息（渲染）
user_message = """
What is the primary function of mitochondria?
A) Protein synthesis
B) Energy production
C) DNA replication
D) Cell division
"""

# 助手回答
assistant_message = "B"
```

**评估方式：**

```python
def evaluate_mmlu(model, tokenizer):
    for problem in mmlu_dataset:
        # 1. 构造 4 个完整 prompt（每个选项一个）
        prompts = [
            f"{question}\n{choices}\nAnswer: A",
            f"{question}\n{choices}\nAnswer: B",
            f"{question}\n{choices}\nAnswer: C",
            f"{question}\n{choices}\nAnswer: D",
        ]
        
        # 2. 计算每个选项的损失
        losses = [model.compute_loss(prompt) for prompt in prompts]
        
        # 3. 选择损失最小的
        pred_letter = letters[losses.argmin()]
        
        # 4. 比较
        is_correct = (pred_letter == correct_letter)
```

**优势：**
- ✅ 不需要采样（快速）
- ✅ 确定性评估
- ✅ 覆盖广泛知识领域

### 3.3 GSM8K (数学推理)

**数据集：** 8.5K 小学数学应用题

**示例：**

```python
# 问题
question = """
Weng earns $12 an hour for babysitting. 
Yesterday, she just did 50 minutes of babysitting. 
How much did she earn?
"""

# 标准答案（带计算过程）
answer = """
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
"""
```

**评估方式：**

```python
def extract_answer(completion):
    """提取 #### 后的数字"""
    match = re.search(r"#### (\-?[0-9\.\,]+)", completion)
    if match:
        return match.group(1).replace(",", "")
    return None

def evaluate(conversation, completion):
    # 提取参考答案
    ref_answer = extract_answer(conversation['messages'][-1])
    
    # 提取生成的答案
    pred_answer = extract_answer(completion)
    
    # 比较
    return float(ref_answer == pred_answer)
```

**特点：**
- 需要多步推理
- 支持工具调用（计算器）
- 答案是数字（易于评估）

### 3.4 HumanEval (代码生成)

**数据集：** 164 个 Python 编程问题

**示例：**

```python
# Prompt（函数签名）
prompt = """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer
    to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

# 标准答案（函数实现）
canonical_solution = """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
"""
```

**评估方式：**

```python
def evaluate(conversation, completion):
    # 1. 提取代码
    code = extract_program(completion)
    
    # 2. 构造完整程序
    program = f"""
{imports}

{code}

{test_cases}

check({entry_point})
"""
    
    # 3. 执行代码
    result = execute_code(program, timeout=5)
    
    # 4. 返回是否成功
    return result.success
```

**挑战：**
- 代码需要执行（安全性）
- 可能有多种正确实现
- 需要通过所有测试用例

### 3.5 SpellingBee (拼写计数)

**任务：** 计算单词中字母出现次数

**示例：**

```python
# 用户问题
question = "How many r are in strawberry?"

# 助手回答（带推理）
answer = """
Let me spell out the word:
s-t-r-a-w-b-e-r-r-y

Now let me count the letter 'r':
<|python_start|>"strawberry".count("r")<|python_end|>
<|output_start|>3<|output_end|>

#### 3
"""
```

**评估方式：**

```python
def evaluate(conversation, completion):
    # 提取参考答案
    ref = extract_answer(conversation['messages'][-1])
    
    # 提取生成的答案
    pred = extract_answer(completion)
    
    # 比较数字
    return float(ref == pred)
```

**意义：**
- 测试模型的**字符级理解**
- 鼓励使用工具（Python）
- 对小模型尤其重要

---

## 4. Base 模型评估

### 4.1 评估指标

| 指标 | 频率 | 说明 |
|-----|------|------|
| **训练 Loss** | 每步 | 实时监控 |
| **验证 BPB** | 每 250 步 | 泛化能力 |
| **CORE Metric** | 每 2000 步 | 综合能力 |
| **采样质量** | 每 2000 步 | 定性评估 |

### 4.2 BPB 评估

**代码位置：** `scripts/base_loss.py`

```bash
# 快速评估 BPB
python -m scripts.base_loss --checkpoint_path=base_checkpoints/d12/step_50000.pt

# 多卡评估
torchrun --nproc_per_node=8 -m scripts.base_loss --checkpoint_path=...
```

**实现：**

```python
# scripts/base_loss.py
device_batch_size = 64
sequence_len = 2048
split_tokens = 20 * 524288  # 评估 20M tokens

for split_name in ['train', 'val']:
    # 创建数据加载器
    loader = tokenizing_distributed_data_loader(
        device_batch_size, 
        sequence_len, 
        split_name
    )
    
    # 计算步数
    steps = split_tokens // (device_batch_size * sequence_len * world_size)
    
    # 评估
    bpb = evaluate_bpb(model, loader, steps, token_bytes)
    
    print(f"{split_name} BPB: {bpb:.4f}")
```

### 4.3 CORE Metric 评估

**代码位置：** `scripts/base_eval.py`

```bash
# 完整 CORE 评估
python -m scripts.base_eval --checkpoint_path=base_checkpoints/d12/step_50000.pt

# 快速测试（每个任务只评估 100 个样本）
python -m scripts.base_eval --max_per_task=100
```

**实现流程：**

```python
# scripts/base_eval.py
def evaluate_model(model, tokenizer, device, max_per_task=-1):
    # 1. 加载 CORE 配置
    tasks = load_core_config()
    
    results = {}
    for task in tasks:
        # 2. 加载任务数据
        data = load_task_data(task)
        
        # 3. 评估
        accuracy = evaluate_task(
            model, 
            tokenizer, 
            data, 
            device, 
            task_meta
        )
        
        results[task['label']] = accuracy
    
    # 4. 计算 CORE Metric
    core_metric = mean(results.values())
    
    return core_metric, results
```

**输出示例：**

```
Evaluating: arc_easy (0-shot, type: multiple_choice)... 0.724
Evaluating: arc_challenge (25-shot, type: multiple_choice)... 0.423
Evaluating: hellaswag (10-shot, type: multiple_choice)... 0.567
Evaluating: piqa (5-shot, type: multiple_choice)... 0.781
Evaluating: winogrande (5-shot, type: multiple_choice)... 0.648
Evaluating: copa (0-shot, type: schema)... 0.920
Evaluating: openbookqa (0-shot, type: schema)... 0.348
Evaluating: lambada (0-shot, type: language_modeling)... 0.681
Evaluating: wikitext (0-shot, type: language_modeling)... 0.523

CORE Metric: 0.624
```

### 4.4 采样质量评估

**目的：** 定性检查模型生成质量

```python
# 在训练脚本中
if step % sample_every == 0:
    model.eval()
    
    # 生成样本
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    
    for token in model.generate(tokens, max_tokens=50):
        print(tokenizer.decode([token]), end='')
    
    print()
    model.train()
```

**示例输出：**

```
# Step 0（随机初始化）
The capital of France is !!#$%@#$... (乱码)

# Step 10000
The capital of France is the city of the United States, and the United...

# Step 30000
The capital of France is Paris, which is the largest city in France...

# Step 50000
The capital of France is Paris. It is located on the Seine River...
```

---

## 5. Chat 模型评估

### 5.1 评估任务

| 阶段 | 评估任务 | 指标 |
|-----|---------|------|
| **Mid** | BPB | 验证损失 |
| **SFT** | MMLU, ARC, SpellingBee | 准确率 |
| **RL** | GSM8K | Pass@k |

### 5.2 SFT 评估

**代码位置：** `scripts/chat_eval.py`

```bash
# 评估 MMLU
python -m scripts.chat_eval -a MMLU

# 评估 ARC-Easy
python -m scripts.chat_eval -a ARC-Easy

# 多卡评估
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a MMLU
```

**实现（分类任务）：**

```python
def run_categorical_eval(task_object, tokenizer, model, batch_size):
    """
    分类任务评估（不需要采样）
    """
    num_correct = 0
    total = 0
    
    # 按批次处理
    for batch_start in range(0, len(task_object), batch_size):
        batch = task_object[batch_start:batch_start + batch_size]
        
        # 1. 准备所有选项的 prompt
        all_prompts = []
        for problem in batch:
            # 每个问题有多个选项
            for letter in problem['letters']:
                prompt = tokenizer.render_with_answer(
                    problem, 
                    letter
                )
                all_prompts.append(prompt)
        
        # 2. Tokenize 并填充
        tokens = tokenizer(all_prompts)
        input_ids = pad_sequences(tokens)
        
        # 3. 前向传播，计算每个选项的 logits
        logits = model(input_ids)
        
        # 4. 计算每个选项在答案位置的 log prob
        # （细节：找到答案 token 的位置）
        answer_logprobs = extract_answer_logprobs(logits, tokens)
        
        # 5. 选择 logprob 最高的
        num_choices = len(problem['letters'])
        for i, problem in enumerate(batch):
            choice_logprobs = answer_logprobs[
                i * num_choices : (i+1) * num_choices
            ]
            pred_idx = choice_logprobs.argmax()
            correct_idx = problem['letters'].index(
                problem['correct_answer']
            )
            
            if pred_idx == correct_idx:
                num_correct += 1
            total += 1
    
    return num_correct / total
```

**实现（生成任务）：**

```python
def run_generative_eval(task_object, tokenizer, model, engine, 
                        num_samples, max_new_tokens, temperature, top_k):
    """
    生成任务评估（需要采样）
    """
    num_passed = 0
    total = 0
    
    for problem in task_object:
        # 1. Tokenize prompt
        encoded_prompt = tokenizer.render_for_completion(problem)
        
        # 2. 生成多个答案
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,    # 例如 8
            max_tokens=max_new_tokens,  # 例如 512
            temperature=temperature,    # 例如 0.8
            top_k=top_k,                # 例如 50
        )
        
        # 3. 解码
        prefix_length = len(encoded_prompt)
        completions = [
            tokenizer.decode(result[prefix_length:])
            for result in results
        ]
        
        # 4. 评估每个答案
        outcomes = [
            task_object.evaluate(problem, completion)
            for completion in completions
        ]
        
        # 5. Pass@k：至少一个正确
        passed = any(outcomes)
        
        num_passed += int(passed)
        total += 1
    
    return num_passed / total
```

### 5.3 RL 评估（Pass@k）

**在训练循环中：**

```python
# scripts/chat_rl.py
for step in range(num_steps):
    # 训练...
    
    # 定期评估
    if step % eval_every == 0:
        model.eval()
        
        # 评估多个 k 值
        pass_at_k = {}
        for k in [1, 4, 8, 16]:
            accuracy = evaluate_pass_at_k(
                model, 
                val_task, 
                k=k,
                max_problems=eval_examples
            )
            pass_at_k[f"pass@{k}"] = accuracy
        
        print(f"Step {step} | Pass@1: {pass_at_k['pass@1']:.3f} | Pass@8: {pass_at_k['pass@8']:.3f}")
        
        model.train()
```

**输出示例：**

```
Step 0000/0500 | Pass@1: 0.452 | Pass@8: 0.723
Step 0060/0500 | Pass@1: 0.468 | Pass@8: 0.735
Step 0120/0500 | Pass@1: 0.481 | Pass@8: 0.748
...
Step 0500/0500 | Pass@1: 0.534 | Pass@8: 0.782
```

---

## 6. 实现细节剖析

### 6.1 Few-shot 采样

**为什么需要 Few-shot？**
- 提供任务示例
- 帮助模型理解格式
- 提升评估准确性

**实现：**

```python
def evaluate_example(idx, model, tokenizer, data, task_meta):
    item = data[idx]
    num_fewshot = task_meta['num_fewshot']  # 例如 5
    
    # 从数据集中随机采样 few-shot 示例（排除当前样本）
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)  # 固定种子保证可复现
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]
    
    # 渲染 prompt
    prompt = render_prompt_with_fewshot(item, fewshot_examples)
    
    # ...评估
```

**Prompt 示例（5-shot）：**

```
[Example 1]
Question: What is 2+2?
Answer: 4

[Example 2]
Question: What is 5*3?
Answer: 15

[Example 3]
...

[Example 5]
...

[Actual Question]
Question: What is 7+8?
Answer: 
```

### 6.2 序列批处理

**挑战：** 不同选项的 prompt 长度不同

**解决方案：** Padding + Mask

```python
def stack_sequences(tokens, pad_token_id):
    """
    将多个序列堆叠成批次
    
    Args:
        tokens: List[List[int]]，每个序列的 token IDs
        pad_token_id: 填充 token
    
    Returns:
        input_ids: (B, max_len) 张量
    """
    bsz = len(tokens)
    seq_len = max(len(x) for x in tokens)
    
    # 创建填充后的张量
    input_ids = torch.full(
        (bsz, seq_len), 
        pad_token_id, 
        dtype=torch.long
    )
    
    # 填入实际 tokens
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    
    return input_ids
```

**示例：**

```python
tokens = [
    [1, 2, 3, 4],         # 长度 4
    [1, 2, 3, 4, 5, 6],   # 长度 6
    [1, 2],               # 长度 2
]

# Padding
input_ids = stack_sequences(tokens, pad_token_id=0)
# tensor([
#     [1, 2, 3, 4, 0, 0],
#     [1, 2, 3, 4, 5, 6],
#     [1, 2, 0, 0, 0, 0],
# ])
```

### 6.3 答案位置定位

**问题：** 如何找到答案在序列中的位置？

**方法 1：公共前缀（Multiple Choice）**

```python
def find_common_length(token_sequences, direction='left'):
    """
    找到所有序列的公共前缀长度
    """
    min_len = min(len(seq) for seq in token_sequences)
    
    for i in range(min_len):
        token = token_sequences[0][i]
        if not all(seq[i] == token for seq in token_sequences):
            return i  # 第一个不同的位置
    
    return min_len

# 示例
prompts = [
    "What is the capital of France? A) London",
    "What is the capital of France? B) Paris",
    "What is the capital of France? C) Berlin",
    "What is the capital of France? D) Madrid",
]

tokens = tokenizer(prompts)
answer_start = find_common_length(tokens)
# "What is the capital of France? " 的长度
# 答案从这里开始
```

**方法 2：公共后缀（Schema）**

```python
# 方向='right'
answer_length = find_common_length(tokens, direction='right')
# 从末尾往前数 answer_length 个 token
```

**方法 3：精确匹配（LM）**

```python
tokens_without = tokenizer(prompt_without)  # 不含答案
tokens_with = tokenizer(prompt_with)        # 含答案

# 答案位置
answer_start = len(tokens_without)
answer_end = len(tokens_with)
```

### 6.4 损失计算技巧

**只计算答案部分的损失：**

```python
def calculate_answer_loss(logits, targets, start_idx, end_idx):
    """
    只计算 [start_idx, end_idx) 范围内的损失
    """
    # logits: (B, T, vocab_size)
    # targets: (B, T)
    
    # 创建 mask
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, start_idx:end_idx] = True
    
    # 计算所有位置的损失
    losses = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction='none'
    ).view(B, T)
    
    # 只保留答案部分
    answer_losses = losses * mask
    
    # 平均
    mean_loss = answer_losses.sum() / mask.sum()
    
    return mean_loss
```

---

## 7. 评估流程实战

### 7.1 Base 模型完整评估

```bash
#!/bin/bash
# evaluate_base.sh

CHECKPOINT="base_checkpoints/d12/step_50000.pt"

echo "=== Evaluating Base Model ==="
echo "Checkpoint: $CHECKPOINT"

# 1. 快速 BPB 评估
echo -e "\n[1/3] BPB Evaluation..."
python -m scripts.base_loss --checkpoint_path=$CHECKPOINT

# 2. CORE Metric（完整）
echo -e "\n[2/3] CORE Metric Evaluation..."
python -m scripts.base_eval --checkpoint_path=$CHECKPOINT

# 3. CORE Metric（快速测试，每个任务 100 个样本）
echo -e "\n[3/3] CORE Metric (Quick Test)..."
python -m scripts.base_eval \
    --checkpoint_path=$CHECKPOINT \
    --max_per_task=100

echo -e "\n=== Evaluation Complete ==="
```

**输出示例：**

```
=== Evaluating Base Model ===
Checkpoint: base_checkpoints/d12/step_50000.pt

[1/3] BPB Evaluation...
train BPB: 3.234
val BPB: 3.451

[2/3] CORE Metric Evaluation...
arc_easy: 0.724
arc_challenge: 0.423
hellaswag: 0.567
piqa: 0.781
winogrande: 0.648
copa: 0.920
openbookqa: 0.348
lambada: 0.681
wikitext: 0.523
CORE Metric: 0.624

[3/3] CORE Metric (Quick Test)...
CORE Metric (100 samples): 0.618

=== Evaluation Complete ===
```

### 7.2 Chat 模型完整评估

```bash
#!/bin/bash
# evaluate_chat.sh

CHECKPOINT="chatsft_checkpoints/d12/step_00718.pt"

echo "=== Evaluating Chat Model ==="
echo "Checkpoint: $CHECKPOINT"

# 1. MMLU（知识）
echo -e "\n[1/5] MMLU..."
torchrun --nproc_per_node=8 -m scripts.chat_eval \
    --checkpoint_path=$CHECKPOINT \
    -a MMLU

# 2. ARC-Easy（科学推理）
echo -e "\n[2/5] ARC-Easy..."
torchrun --nproc_per_node=8 -m scripts.chat_eval \
    --checkpoint_path=$CHECKPOINT \
    -a ARC-Easy

# 3. ARC-Challenge
echo -e "\n[3/5] ARC-Challenge..."
torchrun --nproc_per_node=8 -m scripts.chat_eval \
    --checkpoint_path=$CHECKPOINT \
    -a ARC-Challenge

# 4. GSM8K（数学）
echo -e "\n[4/5] GSM8K..."
torchrun --nproc_per_node=8 -m scripts.chat_eval \
    --checkpoint_path=$CHECKPOINT \
    -a GSM8K \
    --num_samples=8 \
    --temperature=0.8

# 5. HumanEval（代码）
echo -e "\n[5/5] HumanEval..."
torchrun --nproc_per_node=8 -m scripts.chat_eval \
    --checkpoint_path=$CHECKPOINT \
    -a HumanEval \
    --num_samples=8 \
    --temperature=0.8

echo -e "\n=== Evaluation Complete ==="
```

### 7.3 RL 模型评估（Pass@k）

**已集成在训练脚本中：**

```python
# scripts/chat_rl.py
eval_every = 60
eval_examples = 400

for step in range(num_steps):
    # ... 训练 ...
    
    if step % eval_every == 0:
        # 评估多个 k 值
        for k in [1, 4, 8, 16]:
            accuracy = evaluate_pass_at_k(
                model, 
                val_task, 
                k=k,
                max_problems=eval_examples
            )
            wandb.log({f"pass@{k}": accuracy})
```

---

## 8. 最佳实践与优化

### 8.1 评估频率建议

| 阶段 | 评估类型 | 推荐频率 | 原因 |
|-----|---------|---------|------|
| **Base** | BPB | 每 250 步 | 快速反馈 |
| | CORE | 每 2000 步 | 计算成本高 |
| **Mid** | BPB | 每 100 步 | 训练时间短 |
| **SFT** | BPB | 每 100 步 | 快速迭代 |
| | Accuracy | 每 200 步 | 需要生成 |
| **RL** | Pass@k | 每 60 步 | 在线评估 |

### 8.2 加速技巧

#### 1. 减少评估样本

```python
# 开发阶段：快速测试
max_per_task = 100  # 只评估 100 个样本

# 生产阶段：完整评估
max_per_task = -1   # 评估所有样本
```

#### 2. 多卡并行

```bash
# 单卡：慢
python -m scripts.chat_eval -a MMLU

# 8 卡：快 8 倍
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a MMLU
```

#### 3. 批量处理

```python
# 分类任务：可以批量处理
batch_size = 32  # 同时处理 32 个问题

# 生成任务：逐个处理（因为长度不同）
batch_size = 1
```

#### 4. 缓存结果

```python
# 缓存 few-shot 示例
@lru_cache(maxsize=1000)
def get_fewshot_examples(task_name, example_idx, num_fewshot):
    # ...
    return fewshot_examples
```

### 8.3 调试技巧

#### 1. 打印 Prompt

```python
# 调试时打印实际的 prompt
if idx == 0:  # 只打印第一个样本
    print("=" * 50)
    print("Prompt:")
    print(prompt)
    print("=" * 50)
```

#### 2. 检查 Token 对齐

```python
# 验证答案位置是否正确
prompt_tokens = tokenizer(prompt)
print(f"Total length: {len(prompt_tokens)}")
print(f"Answer start: {answer_start_idx}")
print(f"Answer tokens: {prompt_tokens[answer_start_idx:]}")
```

#### 3. 逐步验证

```python
# 1. 先在单个样本上测试
result = evaluate_example(0, model, tokenizer, data, task_meta)
print(f"First example: {result}")

# 2. 再在小批次上测试
results = [evaluate_example(i, ...) for i in range(10)]
print(f"First 10 examples: {sum(results) / len(results)}")

# 3. 最后完整评估
accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
```

### 8.4 常见错误

#### 错误 1: Token 对齐问题

**症状：** 准确率异常低（如 0% 或 100%）

**原因：** 答案位置定位错误

**解决：**
```python
# 打印并检查
print(f"Prompt: {tokenizer.decode(tokens[:answer_start])}")
print(f"Answer: {tokenizer.decode(tokens[answer_start:])}")
```

#### 错误 2: Padding 影响

**症状：** 批量评估结果与逐个评估不一致

**原因：** Padding token 被计入损失

**解决：**
```python
# 使用 attention_mask
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=pad_token_id,  # ← 关键
    reduction='none'
)
```

#### 错误 3: 采样温度

**症状：** Pass@k 所有 k 值相同

**原因：** Temperature=0（贪婪解码）

**解决：**
```python
# Pass@k 必须用随机采样
temperature = 1.0  # 或 0.7, 0.8
top_k = 50
```

### 8.5 性能基准

**单卡 A100 评估时间：**

| 任务 | 样本数 | 时间 | 吞吐量 |
|-----|-------|------|--------|
| BPB | 20M tokens | 2 分钟 | 167K tokens/s |
| CORE | 全量 | 30 分钟 | - |
| MMLU | 14K | 15 分钟 | 15.6 样本/s |
| GSM8K (Pass@8) | 1.3K | 45 分钟 | 0.5 样本/s |
| HumanEval (Pass@8) | 164 | 20 分钟 | 0.14 样本/s |

**8 卡 A100 评估时间：**

| 任务 | 样本数 | 时间 | 加速比 |
|-----|-------|------|--------|
| BPB | 20M tokens | 20 秒 | 6x |
| CORE | 全量 | 5 分钟 | 6x |
| MMLU | 14K | 2 分钟 | 7.5x |
| GSM8K (Pass@8) | 1.3K | 6 分钟 | 7.5x |
| HumanEval (Pass@8) | 164 | 3 分钟 | 6.7x |

---

## 9. 总结

### 9.1 评估体系总览

```
评估金字塔（从底层到高层）:

Pass@k (RL)           ← 探索能力、鲁棒性
    ↑
Task Accuracy (SFT)   ← 任务特定能力
    ↑
BPB (Mid/SFT)         ← 泛化能力
    ↑
CORE Metric (Base)    ← 综合基础能力
    ↑
Training Loss         ← 学习信号
```

### 9.2 关键要点

1. **BPB vs Accuracy**
   - BPB：语言建模能力（低层）
   - Accuracy：任务完成能力（高层）
   - 两者不完全相关

2. **分类 vs 生成**
   - 分类：快速、确定性
   - 生成：慢速、需要采样

3. **Pass@k 的意义**
   - 不只是准确率
   - 衡量探索和恢复能力
   - 对 RL 尤其重要

4. **Few-shot 的作用**
   - 提供任务格式示例
   - 显著提升准确率
   - 需要固定随机种子保证可复现

### 9.3 快速参考

**快速评估命令：**

```bash
# Base 模型
python -m scripts.base_loss --checkpoint_path=<path>
python -m scripts.base_eval --checkpoint_path=<path> --max_per_task=100

# Chat 模型
python -m scripts.chat_eval --checkpoint_path=<path> -a MMLU
python -m scripts.chat_eval --checkpoint_path=<path> -a GSM8K --num_samples=8
```

**评估结果解读：**

| 指标 | 优秀 | 良好 | 需改进 |
|-----|------|------|--------|
| BPB (Base) | < 3.5 | 3.5-4.0 | > 4.0 |
| CORE | > 0.6 | 0.5-0.6 | < 0.5 |
| MMLU | > 50% | 40-50% | < 40% |
| GSM8K (Pass@1) | > 50% | 30-50% | < 30% |
| GSM8K (Pass@8) | > 75% | 60-75% | < 60% |

---

**参考资源：**
- [DCLM Paper (CORE Metric)](https://arxiv.org/abs/2406.11794)
- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
