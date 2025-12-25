---
title: 'noao-chat-rl'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-25-blog-nonochat-rl/
tags:
  - llm
---

### LLM RL 训练完整解析

[nonochat](https://github.com/karpathy/nanochat)


> 本文档详细解析 `chat_rl.py`，讲解强化学习如何优化模型的推理和决策能力。  
> 适合 LLM 领域的初学者，训练流程的最后一步，也是最复杂的一步。

---

## 目录

1. [什么是 RL](#1-什么是-rl)
2. [策略梯度的数学原理](#2-策略梯度的数学原理)
3. [Rollout 采样机制](#3-rollout-采样机制)
4. [优势函数与奖励](#4-优势函数与奖励)
5. [训练流程详解](#5-训练流程详解)
6. [与 SFT 的对比](#6-与-sft-的对比)

---

## 1. 什么是 RL

### 1.1 训练流程全景（最终章）

```
Base Training → Mid Training → SFT → RL
     ↓              ↓           ↓     ↓
 通用语言能力    任务能力   指令遵循  偏好对齐
   (语言)        (格式)     (行为)   (优化)
   数周          数小时     数小时   数小时
   监督学习      监督学习   监督学习  强化学习 ←
```

**RL (Reinforcement Learning)** 是训练的第四阶段，也是最后阶段。

### 1.2 为什么需要 RL？

**SFT 的局限性：**

示例场景：数学题求解

```
问题: "If 5 apples cost $10, how much do 3 apples cost?"

SFT 训练数据（需要完整答案）:
{
    "user": "If 5 apples cost $10...",
    "assistant": "First calculate per apple: 10/5=2, then 2*3=6"
}
```

**问题：**
- ❌ 需要**完整的正确答案**
- ❌ 如果答案格式稍有不同，无法学习
- ❌ 只能学会"模仿"示例，无法"创新"
- ❌ 对于复杂推理，很难提供完美示例

**RL 的解决方案：**

```
只需要：
1. 问题（不需要完整答案）
2. 评分函数（判断对错）

让模型：
1. 自己生成答案（探索）
2. 得到反馈（对/错）
3. 调整策略（学习）
```

**关键差异：**

| 维度 | SFT | RL |
|-----|-----|-----|
| 需要什么 | 完整的正确答案 | 问题 + 评分函数 |
| 学习方式 | 模仿示例 | 试错探索 |
| 适用场景 | 格式固定的任务 | 答案多样的任务 |
| 答案空间 | 受限于训练数据 | 可以探索新解法 |

### 1.3 RL 的核心目标

**三个关键能力：**

1. **探索优化 (Exploration & Optimization)**
   ```
   同一个问题，生成多个答案
   比较哪个更好
   强化好的策略
   ```

2. **复杂推理 (Complex Reasoning)**
   ```
   数学题、逻辑题、多步推理
   需要试错才能找到正确路径
   SFT 的示例可能不够
   ```

3. **偏好对齐 (Preference Alignment)**
   ```
   什么样的回答更好？
   更礼貌？更简洁？更准确？
   通过奖励函数定义"好"的标准
   ```

### 1.4 本项目的 RL 配置

```python
# RL 超参数
rl_config = {
    # 数据
    "source": "sft",              # 从 SFT 模型开始
    "task": GSM8K,                # 数学推理任务
    
    # 采样
    "device_batch_size": 8,       # 推理批次
    "examples_per_step": 16,      # 每步处理的问题数
    "num_samples": 16,            # 每个问题采样 16 个答案
    "max_new_tokens": 256,        # 最长生成 256 tokens
    "temperature": 1.0,           # 采样温度（探索）
    "top_k": 50,                  # Top-K 采样
    
    # 训练
    "unembedding_lr": 0.004,
    "embedding_lr": 0.2,
    "matrix_lr": 0.02,
    "init_lr_frac": 0.05,         # 从 5% 开始（比 SFT 高）
    
    # 评估
    "eval_every": 60,
    "eval_examples": 400,
}
```

**关键数字：**
- **16 个样本/问题** - 大量采样以探索不同解法
- **温度 1.0** - 保持随机性，鼓励探索
- **8K 训练数据** - 比 SFT 少，但通过多次采样增加多样性

### 1.5 简化版 GRPO

本项目使用的是**简化版 GRPO**：

**标准 RL 算法（PPO）的组件：**
1. ✅ 策略梯度
2. ✅ 优势函数
3. ❌ KL 散度约束（信任域）
4. ❌ PPO Clip（重要性采样比率裁剪）
5. ❌ 价值网络

**本项目的简化：**

```python
# 标准 PPO 目标函数：
L = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL(π||π_ref)
    ↑                                              ↑
    PPO Clip                                      KL 约束

# 简化后（本项目）：
L = E[log π(a|s) * A]
    ↑
    简单的策略梯度（REINFORCE）
```

**简化的原因：**
1. **On-policy** - 每次都用当前模型采样，不需要重要性采样
2. **小学习率** - 本身就限制了更新幅度，不需要额外约束
3. **效果相当** - 在 LLM 上，简单方法效果也很好

---

## 2. 策略梯度的数学原理

### 2.1 核心概念

**把 LLM 看作策略 (Policy)：**

```python
# LLM 的本质
def policy(state):
    """
    输入: state = 当前上下文 (问题 + 已生成的文本)
    输出: action = 下一个 token 的概率分布
    """
    logits = model(state)
    probs = softmax(logits)
    return probs
```

**RL 的三要素：**

1. **State (状态)** - 当前的上下文
   ```
   问题："What is 2+2?"
   已生成："The answer is"
   当前状态：["What", "is", "2", "+", "2", "?", "The", "answer", "is"]
   ```

2. **Action (动作)** - 选择下一个 token
   ```
   可能的动作：["4", "four", "2+2", ...]
   模型选择："4"
   ```

3. **Reward (奖励)** - 整个回答的得分
   ```
   生成完整回答："The answer is 4"
   检查答案：正确！
   奖励：r = 1.0
   ```

### 2.2 策略梯度公式

**目标：** 最大化期望奖励

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

其中：
- $\theta$ = 模型参数
- $\pi_\theta$ = 模型的策略（生成分布）
- $\tau$ = 一条完整的轨迹（采样的序列）
- $R(\tau)$ = 这条轨迹的总奖励

**梯度（策略梯度定理）：**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau)\right]
$$

**简化理解：**

```python
# 伪代码
for question in dataset:
    # 1. 采样多个答案
    answers = model.sample(question, num_samples=16)
    
    # 2. 计算每个答案的奖励
    rewards = [evaluate(question, answer) for answer in answers]
    
    # 3. 计算梯度
    for answer, reward in zip(answers, rewards):
        # log_prob: 生成这个答案的对数概率
        # reward: 这个答案的得分
        gradient += log_prob(answer) * reward
    
    # 4. 更新参数
    model.update(gradient)
```

**关键点：**
- 奖励高的答案 → 增加其概率
- 奖励低的答案 → 降低其概率

### 2.3 优势函数（Advantage）

**问题：** 如果所有奖励都是正数怎么办？

```python
rewards = [0.8, 0.9, 0.7, 0.85]  # 都是正数
# 所有答案的概率都会增加！
# 但我们只想增加"更好"的答案
```

**解决方案：** 使用相对奖励（优势）

$$
A = R - \mu(R)
$$

其中 $\mu(R)$ 是奖励的均值。

**示例：**

```python
rewards = [0.8, 0.9, 0.7, 0.85]
mean_reward = 0.8125

advantages = [
    0.8 - 0.8125 = -0.0125,   # 比平均差 → 降低概率
    0.9 - 0.8125 = +0.0875,   # 比平均好 → 增加概率
    0.7 - 0.8125 = -0.1125,   # 比平均差 → 降低概率
    0.85 - 0.8125 = +0.0375,  # 比平均好 → 增加概率
]
```

**效果：**
- 只有相对较好的答案才会被强化
- 避免所有答案都增加概率

### 2.4 Token-Level 的奖励分配

**问题：** 奖励是序列级别的，但训练是 token 级别的

```python
# 序列级别
answer = "The answer is 4"
reward = 1.0  # 整个答案的奖励

# Token 级别
tokens = ["The", "answer", "is", "4"]
# 每个 token 应该得到多少奖励？
```

**本项目的解决方案：** 均匀分配

```python
# 所有 token 获得相同的奖励
for token in tokens:
    token_reward = reward  # 1.0

# 实现中：
advantages = advantages.unsqueeze(-1)  # (B,) → (B, 1)
# 广播到 (B, T)，每个 token 位置都是相同值
```

**更复杂的方案（本项目未实现）：**
- GAE (Generalized Advantage Estimation)
- 根据 token 位置衰减奖励
- 使用价值函数估计

---

## 3. Rollout 采样机制

### 3.1 什么是 Rollout？

**Rollout = 让模型自己玩一遍游戏**

```
1. 给定问题（初始状态）
2. 模型生成答案（采样轨迹）
3. 评估答案（获得奖励）
4. 用这个经验更新模型
```

### 3.2 采样流程详解

#### 步骤 1: 获取问题

```python
@torch.no_grad()
def get_batch():
    for example_idx in range(ddp_rank, len(train_task), ddp_world_size):
        # 获取一个数学题
        conversation = train_task[example_idx]
        # {
        #     "messages": [
        #         {"role": "user", "content": "What is 2+2?"},
        #         {"role": "assistant", "content": "4"}
        #     ]
        # }
```

#### 步骤 2: 准备生成

```python
        # 去掉 assistant 的回答，只保留问题
        tokens = tokenizer.render_for_completion(conversation)
        
        # tokens:
        # [<|bos|>, <|user_start|>, "What", "is", "2", "+", "2", "?", 
        #  <|user_end|>, <|assistant_start|>]
        #                           ↑
        #                    到这里停止，让模型生成
        
        prefix_length = len(tokens)  # 记录前缀长度
```

#### 步骤 3: 采样多个答案

**关键：采样 16 个不同的答案！**

```python
        model.eval()  # 评估模式
        generated_token_sequences = []
        masks = []
        
        # 分批生成，避免 OOM
        num_sampling_steps = num_samples // device_batch_size
        # 16 // 8 = 2 步
        
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            
            with autocast_ctx:
                sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,  # 8
                    max_tokens=max_new_tokens,      # 256
                    temperature=1.0,                # 随机采样！
                    top_k=50,
                    seed=seed,
                )
            
            generated_token_sequences.extend(sequences_batch)
            masks.extend(masks_batch)
```

**为什么 temperature=1.0？**

```python
# temperature=0.0 (贪婪解码)
# 每次都选概率最高的 token
# 16 个样本会完全一样 ❌

# temperature=1.0 (采样)
# 按照概率分布随机选择
# 16 个样本各不相同 ✅
```

#### 步骤 4: 计算奖励

```python
        # 对每个生成的答案计算奖励
        rewards = []
        for sample_tokens in generated_token_sequences:
            # 提取生成的部分（去掉 prompt）
            generated_tokens = sample_tokens[prefix_length:]
            
            # 解码为文本
            generated_text = tokenizer.decode(generated_tokens)
            # "The answer is 4"
            
            # 计算奖励
            reward = train_task.reward(conversation, generated_text)
            # reward: 0.0 或 1.0（对于 GSM8K）
            
            rewards.append(reward)
```

**GSM8K 的奖励函数：**

```python
def reward(self, conversation, assistant_response):
    # 从标准答案中提取数字
    ref_num = extract_answer(conversation['messages'][-1])
    # "#### 10" → "10"
    
    # 从生成的回答中提取数字
    pred_num = extract_answer(assistant_response)
    # "The answer is 10" → "10"
    
    # 比较
    is_correct = float(pred_num == ref_num)
    return is_correct  # 0.0 或 1.0
```

#### 步骤 5: 准备训练数据

```python
        # 填充序列到相同长度
        max_length = max(len(seq) for seq in generated_token_sequences)
        
        padded_sequences = [
            seq + [pad_token] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]
        
        # 转为张量
        ids = torch.tensor(padded_sequences, device=device)  # (16, T)
        mask_ids = torch.tensor(padded_masks, device=device)  # (16, T)
        
        # 构造 inputs 和 targets
        inputs = ids[:, :-1]    # (16, T-1)
        targets = ids[:, 1:]    # (16, T-1)
        
        # 应用 mask：只训练生成的部分
        targets[mask_ids[:, 1:] == 0] = -1
        
        rewards = torch.tensor(rewards, device=device)  # (16,)
```

#### 步骤 6: 计算优势

```python
        # 计算优势函数
        mu = rewards.mean()  # 平均奖励
        advantages = rewards - mu  # 相对优势
        
        # 例如：
        # rewards = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
        # mu = 0.625
        # advantages = [-0.625, 0.375, -0.625, 0.375, ...]
```

### 3.3 采样的维度追踪

**单个问题的处理：**

```python
# 输入
question_tokens: [10, 20, 30, 40, 50]  # 问题 (5 tokens)

# 采样 16 次，每次生成不同的答案
sample_1: [10, 20, 30, 40, 50, 60, 70, 80]        # +3 tokens
sample_2: [10, 20, 30, 40, 50, 61, 72, 83, 94]    # +4 tokens
...
sample_16: [10, 20, 30, 40, 50, 65, 75]           # +2 tokens

# 填充到最长
max_length = 9  # 最长的序列
padded_samples: (16, 9)

# inputs & targets
inputs:  (16, 8)
targets: (16, 8)

# rewards & advantages
rewards: (16,)      # [0, 1, 0, 1, ...]
advantages: (16,)   # [-0.625, 0.375, ...]
```

---

## 4. 优势函数与奖励

### 4.1 奖励的设计

**GSM8K 的奖励函数：**

```python
def reward(conversation, assistant_response):
    # 提取答案
    ref = extract_answer(conversation)  # "#### 42"
    pred = extract_answer(assistant_response)  # "#### 42"
    
    # 比较
    return 1.0 if ref == pred else 0.0
```

**特点：**
- ✅ 稀疏奖励：只有最终答案对错
- ✅ 二元奖励：0 或 1
- ❌ 不考虑中间步骤
- ❌ 不考虑解题过程

**更复杂的奖励（可能的改进）：**

```python
def advanced_reward(conversation, assistant_response):
    reward = 0.0
    
    # 1. 最终答案正确 +1.0
    if is_correct_answer(conversation, assistant_response):
        reward += 1.0
    
    # 2. 使用工具 +0.2
    if has_calculator_call(assistant_response):
        reward += 0.2
    
    # 3. 步骤清晰 +0.1
    if has_clear_steps(assistant_response):
        reward += 0.1
    
    # 4. 格式规范 +0.1
    if has_proper_format(assistant_response):
        reward += 0.1
    
    return reward
```

### 4.2 优势函数的计算

**标准方法（Z-score 归一化）：**

$$
A = \frac{R - \mu}{\sigma}
$$

**本项目的简化：**

$$
A = R - \mu
$$

**原因：**
```python
# 标准方法
advantages = (rewards - rewards.mean()) / rewards.std()
# 问题：rewards 是 0/1，std 可能很小或为 0

# 简化方法
advantages = rewards - rewards.mean()
# 优点：简单，稳定，效果相当
```

**对比示例：**

```python
# 假设 16 个样本的奖励
rewards = torch.tensor([
    0, 1, 0, 1, 1, 0, 1, 1,  # 前 8 个
    0, 1, 0, 1, 1, 0, 1, 0   # 后 8 个
])  # 10 个正确，6 个错误

mean = 0.625
std = 0.484

# 标准方法
advantages_standard = (rewards - 0.625) / 0.484
# [-1.29, 0.77, -1.29, 0.77, 0.77, -1.29, 0.77, 0.77,
#  -1.29, 0.77, -1.29, 0.77, 0.77, -1.29, 0.77, -1.29]

# 简化方法
advantages_simple = rewards - 0.625
# [-0.625, 0.375, -0.625, 0.375, 0.375, -0.625, ...]

# 效果类似：正确的>0，错误的<0
```

### 4.3 Token-Level 的奖励分配

**问题：** 奖励是序列级别的标量

```python
reward = 1.0  # 标量
```

**需要：** Token 级别的梯度

```python
logits: (B, T, vocab_size)
targets: (B, T)
# 需要每个 token 位置都有奖励信号
```

**解决方案：** 广播

```python
# advantages: (B,) = (16,)
advantages = advantages.unsqueeze(-1)  # (16, 1)

# logp: (B, T) = (16, 256)
logp = -model(inputs, targets, loss_reduction='none')

# 计算策略梯度目标
pg_obj = (logp * advantages).sum()
#         ↑       ↑
#     (16, 256) * (16, 1)  → 广播为 (16, 256)
#     每个 token 都乘以相同的 advantage
```

**可视化：**

```python
# 假设一个样本，advantage = 0.375

logp = [-0.5, -0.3, -0.8, -0.2, -0.4]  # 5 个 token
advantage = 0.375

# 广播后
pg_obj_per_token = [
    -0.5 * 0.375 = -0.1875,
    -0.3 * 0.375 = -0.1125,
    -0.8 * 0.375 = -0.3000,
    -0.2 * 0.375 = -0.0750,
    -0.4 * 0.375 = -0.1500,
]

# 总和
pg_obj = -0.825
```

---

## 5. 训练流程详解

### 5.1 完整训练循环

```python
# 准备
batch_iterator = get_batch()  # 生成器

for step in range(num_steps):
    
    # ===== 评估阶段 =====
    if step % eval_every == 0:
        model.eval()
        # 评估 Pass@k
        passk = evaluate_pass_at_k(model, val_task)
        print(f"Pass@1: {passk[0]:.4f}, Pass@8: {passk[7]:.4f}")
        model.train()
    
    # ===== 训练阶段 =====
    rewards_list = []
    sequence_lengths = []
    
    # 处理多个问题
    for example_step in range(examples_per_rank):
        # 1. 获取一个问题的 16 个采样
        sequences, inputs, targets, rewards, advantages = next(batch_iterator)
        
        model.train()
        
        # 2. 分批处理（避免 OOM）
        num_passes = inputs.size(0) // device_batch_size
        # 16 // 8 = 2 passes
        
        for pass_idx in range(num_passes):
            # 取出一个 mini-batch
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs_batch = inputs[b0:b1]     # (8, T)
            targets_batch = targets[b0:b1]   # (8, T)
            advantages_batch = advantages[b0:b1]  # (8,)
            
            # 3. 计算 log 概率
            with autocast_ctx:
                logp = -model(inputs_batch, targets_batch, loss_reduction='none')
                # logp: (8, T)，每个 token 的 log P(token | context)
            
            # 4. 计算策略梯度目标
            pg_obj = (logp * advantages_batch.unsqueeze(-1)).sum()
            
            # 5. 归一化
            num_valid = (targets_batch >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            
            # 6. 转为损失（最小化）
            loss = -pg_obj
            loss.backward()
        
        # 记录统计
        rewards_list.append(rewards.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences)
    
    # ===== 更新参数 =====
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    for opt in optimizers:
        opt.step()
    model.zero_grad()
    
    # ===== 日志 =====
    mean_reward = sum(rewards_list) / len(rewards_list)
    print(f"Step {step} | Reward: {mean_reward:.4f}")
```

### 5.2 策略梯度计算详解

**单个 mini-batch 的处理：**

```python
# inputs:  (8, 256) - token IDs
# targets: (8, 256) - 目标 IDs（包含 -1）
# advantages: (8,) - 每个序列的优势

# 步骤 1: 前向传播
with autocast_ctx:
    # 模型计算每个 token 的 NLL
    nll = model(inputs, targets, loss_reduction='none')
    # nll: (8, 256)
    
    # 转为 log 概率
    logp = -nll  # (8, 256)

# 步骤 2: 应用优势函数
advantages = advantages.unsqueeze(-1)  # (8, 1)
weighted_logp = logp * advantages      # (8, 256)

# 步骤 3: 求和
pg_obj = weighted_logp.sum()  # 标量

# 步骤 4: 归一化
num_valid = (targets >= 0).sum()  # 有效 token 数量
pg_obj = pg_obj / num_valid

# 步骤 5: 转为损失
loss = -pg_obj  # 要最小化
```

**关键点：**

```python
# 正 advantage (reward 高)
advantage = +0.375
pg_obj = log P(tokens) * 0.375  # 正值
loss = -pg_obj  # 负值
# 最小化负值 = 最大化 log P(tokens)
# → 增加这些 token 的概率 ✅

# 负 advantage (reward 低)
advantage = -0.625
pg_obj = log P(tokens) * (-0.625)  # 负值
loss = -pg_obj  # 正值
# 最小化正值 = 减小 log P(tokens)
# → 降低这些 token 的概率 ✅
```

### 5.3 学习率调度

```python
def get_lr_multiplier(it):
    # 简单的线性衰减
    lrm = 1.0 - it / num_steps
    return lrm
```

**与 SFT 的对比：**

```python
# SFT 初始学习率
sft_initial_lr = matrix_lr * 0.02  # 0.02 * 0.02 = 0.0004

# RL 初始学习率
rl_initial_lr = matrix_lr * 0.05  # 0.02 * 0.05 = 0.001

# RL 的学习率是 SFT 的 2.5 倍！
```

**为什么 RL 学习率更高？**
- SFT 从 Mid 模型开始，需要小心
- RL 从 SFT 模型开始，已经很好了
- RL 需要探索新策略，可以更激进一些

### 5.4 Pass@k 评估

**定义：** 给 k 次机会，至少一次正确的概率

```python
def evaluate_pass_at_k(model, task, k=8):
    """
    对每个问题生成 k 个答案
    计算至少有一个正确的比例
    """
    correct_count = 0
    total_count = 0
    
    for question in task:
        # 生成 k 个答案
        answers = model.generate(question, num_samples=k)
        
        # 检查是否至少有一个正确
        is_any_correct = any(
            task.evaluate(question, answer)
            for answer in answers
        )
        
        if is_any_correct:
            correct_count += 1
        total_count += 1
    
    pass_at_k = correct_count / total_count
    return pass_at_k
```

**Pass@k 的意义：**

```python
# Pass@1: 首次尝试的准确率
# 例如：0.45 = 45% 的题第一次就做对

# Pass@8: 给 8 次机会的准确率
# 例如：0.72 = 72% 的题在 8 次尝试中至少对一次

# Pass@k 越高，说明模型：
# - 探索能力越强
# - 找到正确答案的概率越高
# - 即使不确定，也有机会试出来
```

---

## 6. 与 SFT 的对比

### 6.1 核心差异

| 维度 | SFT | RL |
|-----|-----|-----|
| **训练方式** | 监督学习 | 强化学习 |
| **数据需求** | 完整的正确答案 | 问题 + 评分函数 |
| **学习方式** | 模仿示例 | 试错探索 |
| **损失函数** | CrossEntropy | Policy Gradient |
| **采样** | 不需要 | 每步采样 16 次 |
| **温度** | 0.0 (确定性) | 1.0 (随机性) |
| **训练信号** | 每个 token 都有标签 | 只有最终奖励 |
| **计算成本** | 低 | 高（16x 采样） |
| **适用任务** | 格式、礼仪 | 推理、优化 |

### 6.2 数据流对比

#### SFT 数据流

```python
# 输入：完整对话
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}  # ← 有标准答案
    ]
}

# Tokenize
ids, mask = tokenizer.render_conversation(conversation)

# 训练
inputs = ids[:-1]
targets = ids[1:]  # 已知的正确答案
loss = model(inputs, targets)  # 直接计算损失
loss.backward()
```

#### RL 数据流

```python
# 输入：只有问题
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
        # 没有 assistant 答案！
    ]
}

# 生成 16 个答案
tokens = tokenizer.render_for_completion(conversation)
answers = model.generate(tokens, num_samples=16, temperature=1.0)
# ["4", "four", "2+2", "The answer is 4", ...]

# 评估每个答案
rewards = [evaluate(conv, ans) for ans in answers]
# [1.0, 0.0, 0.0, 1.0, ...]

# 计算优势
advantages = rewards - rewards.mean()

# 训练（策略梯度）
for answer, advantage in zip(answers, advantages):
    logp = log_prob(answer)
    loss = -logp * advantage
    loss.backward()
```

### 6.3 损失函数对比

#### SFT 损失

```python
# 标准交叉熵
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # (B*T, V)
    targets.view(-1),              # (B*T,)
    ignore_index=-1
)

# 每个 token 都有明确的目标
# 只要预测对了，loss 就低
```

#### RL 损失

```python
# 策略梯度
logp = -model(inputs, targets, loss_reduction='none')  # (B, T)
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
loss = -pg_obj / num_valid_tokens

# token 本身没有"对错"
# 只有整个序列的"好坏"（advantage）
# 好的序列 → 增加概率
# 坏的序列 → 降低概率
```

### 6.4 训练效果对比

**同一问题的不同方法：**

```
问题: "If 3 apples cost $6, how much do 5 apples cost?"
```

#### SFT 训练后

```
输出: "First, 6/3=2 per apple. Then 2*5=10."

特点:
✅ 格式规范
✅ 步骤清晰
❌ 可能过拟合训练示例的格式
❌ 缺乏灵活性
```

#### RL 训练后

```
输出 1: "Each apple is $2, so 5 apples cost $10."
输出 2: "6/3=2, 2*5=10, answer is $10."
输出 3: "Price per apple: $2. Total: $10."

特点:
✅ 答案正确（最重要）
✅ 多样化的解题方式
✅ 可能发现更优的策略
❌ 格式可能不如 SFT 规范
```

### 6.5 计算成本对比

**单个训练步骤：**

| 操作 | SFT | RL | 倍数 |
|-----|-----|-----|-----|
| 前向传播 | 1 次 | 16 次（采样） | 16x |
| 反向传播 | 1 次 | 1 次 | 1x |
| 总成本 | 低 | 高 | ~10x |

**训练时间（8 卡 A100，depth=12）：**

| 阶段 | 数据量 | 训练时间 | 原因 |
|-----|--------|---------|------|
| SFT | 23K 对话 | 2 小时 | 直接训练 |
| RL | 8K 问题 | 3-4 小时 | 需要大量采样 |

---

## 7. 实战案例分析

### 7.1 完整训练流程

#### 阶段 1-3: 已完成

```bash
# Base → Mid → SFT
# 输出：chatsft_checkpoints/d12/step_00718.pt
```

#### 阶段 4: RL

```bash
# 单 GPU
python -m scripts.chat_rl

# 8 GPU
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
    --source=sft \
    --device_batch_size=8 \
    --examples_per_step=16 \
    --num_samples=16 \
    --num_epochs=1 \
    --run=rl_d12

# 输出：chatrl_checkpoints/d12/step_XXXXX.pt
```

### 7.2 训练过程示例

#### 初始状态 (Step 0)

```
Step 0 | Pass@1: 0.452, Pass@8: 0.723
```

**解读：**
- Pass@1 45.2%: SFT 模型的基线
- Pass@8 72.3%: 给 8 次机会，正确率显著提高

#### 训练中期 (Step 250)

```
Step 250/500 | Reward: 0.487 | Pass@1: 0.498, Pass@8: 0.756
```

**解读：**
- 平均奖励 48.7%: 略好于初始
- Pass@1 49.8%: 提升了 4.6%
- Pass@8 75.6%: 提升了 3.3%

#### 训练完成 (Step 500)

```
Step 500/500 | Reward: 0.521 | Pass@1: 0.534, Pass@8: 0.782
```

**最终收获：**
- Pass@1: 45.2% → 53.4% (+8.2%)
- Pass@8: 72.3% → 78.2% (+5.9%)

### 7.3 采样示例分析

**问题：** "If 5 apples cost $10, how much do 3 apples cost?"

**SFT 模型（temperature=0.0，确定性）：**

```
输出（每次都一样）:
"First, calculate the price per apple:
<|python_start|>10/5<|python_end|><|output_start|>2<|output_end|>
Each apple costs $2. For 3 apples:
<|python_start|>2*3<|python_end|><|output_start|>6<|output_end|>
#### 6"
```

**RL 训练时（temperature=1.0，16 个样本）：**

```
Sample 1: "10/5=2, 2*3=6, #### 6" ✅
Sample 2: "Each apple is 10/5=$2, so 3 cost $6. #### 6" ✅
Sample 3: "Price per apple is $2. 3*2=6. #### 6" ✅
Sample 4: "10/5 gives 2, multiply by 3 is 6. #### 6" ✅
Sample 5: "Let me calculate: <|python_start|>10/5*3<|python_end|>..." ✅
Sample 6: "5 apples=$10, so 3 apples=$6. #### 6" ✅
Sample 7: "3/5 of 10 is 6. #### 6" ✅
Sample 8: "10*(3/5)=6. #### 6" ✅
Sample 9: "First 10/5... then 2... wait 2*3... #### 6" ✅
Sample 10: "The answer is 6 dollars. #### 6" ✅
Sample 11: "10 divided by 5 is 2, times 3 is 6. #### 6" ✅
Sample 12: "If 5→10 then 3→6. #### 6" ✅
Sample 13: "$10/5*3=$6. #### 6" ✅
Sample 14: "10 dollars for 5, so 6 for 3. #### 6" ✅
Sample 15: "3 apples would cost $6. #### 6" ✅
Sample 16: "5 apples at $10 means $2 each, 3*2=$6. #### 6" ✅
```

**观察：**
- 所有 16 个样本答案都对！
- 但解题方式各不相同
- 有的用工具，有的不用
- 有的详细，有的简洁

**奖励：**
```python
rewards = [1.0] * 16  # 全对
mean = 1.0
advantages = [0.0] * 16  # 都一样好

# 这种情况下，不会更新太多
# 因为没有"相对好坏"的区分
```

**另一个问题的采样（有对有错）：**

```
Sample 1: "Answer is 5. #### 5" ❌  reward=0.0, advantage=-0.4375
Sample 2: "Let me calculate... #### 6" ✅  reward=1.0, advantage=+0.5625
Sample 3: "I think it's 4. #### 4" ❌  reward=0.0, advantage=-0.4375
Sample 4: "10/5*3=6. #### 6" ✅  reward=1.0, advantage=+0.5625
...
Sample 16: "Not sure. #### 7" ❌  reward=0.0, advantage=-0.4375

# rewards = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
# mean = 0.5625
# advantages 让正确答案的概率增加，错误答案的概率降低
```

### 7.4 训练日志解读

```
Step 0000/0500 | Reward: 0.452 | Pass@1: 0.452 | Pass@8: 0.723
Step 0060/0500 | Reward: 0.471 | Pass@1: 0.468 | Pass@8: 0.735
Step 0120/0500 | Reward: 0.485 | Pass@1: 0.481 | Pass@8: 0.748
Step 0180/0500 | Reward: 0.493 | Pass@1: 0.492 | Pass@8: 0.759
Step 0240/0500 | Reward: 0.502 | Pass@1: 0.503 | Pass@8: 0.768
Step 0300/0500 | Reward: 0.509 | Pass@1: 0.512 | Pass@8: 0.774
Step 0360/0500 | Reward: 0.515 | Pass@1: 0.521 | Pass@8: 0.779
Step 0420/0500 | Reward: 0.519 | Pass@1: 0.528 | Pass@8: 0.781
Step 0480/0500 | Reward: 0.521 | Pass@1: 0.532 | Pass@8: 0.782
Step 0500/0500 | Reward: 0.521 | Pass@1: 0.534 | Pass@8: 0.782
```

**趋势：**
- 平均奖励稳步提升
- Pass@1 和 Pass@8 同步增长
- 后期增长放缓（收敛）

---

## 8. 关键要点总结

### 8.1 RL 的本质

```
RL = 试错学习 + 奖励反馈 + 策略优化
   = 让模型自己探索 + 告诉它哪些好 + 强化好的策略
   = 从行为模仿到智能决策
```

### 8.2 RL 的优势

**相比 SFT：**
- ✅ 不需要完整的正确答案
- ✅ 可以探索多种解法
- ✅ 更适合复杂推理任务
- ✅ 可以优化难以描述的目标

**代价：**
- ❌ 训练更复杂
- ❌ 计算成本更高（16x 采样）
- ❌ 可能不稳定
- ❌ 需要设计好的奖励函数

### 8.3 何时使用 RL？

**适合 RL：**
- ✅ 有明确的评价标准（对/错）
- ✅ 答案空间大，难以穷举示例
- ✅ 需要优化复杂目标
- ✅ 有充足的计算资源

**不适合 RL：**
- ❌ 格式要求严格（用 SFT）
- ❌ 已有大量高质量示例（用 SFT）
- ❌ 评价标准模糊
- ❌ 计算资源有限

### 8.4 实践建议

**奖励函数设计：**
- 简单明确比复杂模糊好
- 稀疏奖励也可以（本项目就是）
- 可以组合多个奖励（准确性+格式+效率）

**采样策略：**
- temperature=1.0 保证多样性
- num_samples=16 通常足够
- 可以动态调整采样数量

**训练技巧：**
- 从 SFT 模型开始（不要从 Base）
- 学习率可以略高于 SFT
- 监控 Pass@k 而不只是奖励

**常见错误：**
- ❌ 奖励函数设计不合理
- ❌ temperature 太低（缺乏探索）
- ❌ 学习率太高（破坏 SFT 能力）
- ❌ 训练太久（过拟合）

---

## 9. 四阶段训练总结

### 9.1 完整训练链

```
┌─────────────┐
│ Base Train  │  数周，数十亿 tokens
│ 通用语言能力  │  监督学习（语言建模）
└──────┬──────┘
       ↓
┌─────────────┐
│  Mid Train  │  数小时，850K 对话
│ 特定任务能力  │  监督学习（结构化对话）
└──────┬──────┘
       ↓
┌─────────────┐
│    SFT      │  数小时，23K 对话
│  指令遵循    │  监督学习（Masked）
└──────┬──────┘
       ↓
┌─────────────┐
│     RL      │  数小时，8K 问题
│  偏好对齐    │  强化学习（策略梯度）
└─────────────┘
```

### 9.2 四阶段对比表

| 维度 | Base | Mid | SFT | RL |
|-----|------|-----|-----|-----|
| **数据量** | 数十亿 | 850K | 23K | 8K |
| **数据类型** | 纯文本 | 结构化对话 | 精选对话 | 问题 |
| **学习方式** | 监督 | 监督 | 监督 | 强化 |
| **训练目标** | 语言建模 | 任务能力 | 指令遵循 | 偏好优化 |
| **关键技术** | 自回归 | 对话格式 | Mask | 策略梯度 |
| **学习率** | 大 | 中 | 小 | 中 |
| **训练时间** | 数周 | 数小时 | 数小时 | 数小时 |
| **模型来源** | 随机 | Base | Mid | SFT |
| **评估指标** | BPB | BPB + Acc | Acc | Pass@k |

### 9.3 能力演变

**Base 模型：**
```
能力：语言理解、补全句子
测试："The capital of France"
输出："is Paris, a beautiful city in Europe..."
问题：不知道何时停止
```

**Mid 模型：**
```
能力：对话格式、基本任务
测试：用户："What is 2+2?" 
输出：助手："The answer is 4."
问题：可能继续生成虚构对话
```

**SFT 模型：**
```
能力：指令遵循、格式控制
测试："Explain in one sentence"
输出："The answer is 4 because 2 plus 2 equals 4."
问题：可能不够灵活
```

**RL 模型：**
```
能力：优化推理、探索策略
测试：复杂数学题
输出：多种正确解法，选最优的
优点：准确率更高，方法更多样
```

---

## 附录

### A. 完整训练命令

```bash
# 阶段 1: Base Training
torchrun --nproc_per_node=8 -m scripts.base_train \
    --depth=12 --num_iterations=50000

# 阶段 2: Mid Training  
torchrun --nproc_per_node=8 -m scripts.mid_train

# 阶段 3: SFT
torchrun --nproc_per_node=8 -m scripts.chat_sft

# 阶段 4: RL
torchrun --nproc_per_node=8 -m scripts.chat_rl \
    --source=sft \
    --device_batch_size=8 \
    --num_samples=16 \
    --run=rl_production
```

### B. 超参数总结

```python
# RL 核心超参数
rl_config = {
    # 采样
    "num_samples": 16,         # 每个问题采样数
    "temperature": 1.0,        # 采样温度
    "top_k": 50,               # Top-K 采样
    
    # 训练
    "examples_per_step": 16,   # 每步处理的问题数
    "device_batch_size": 8,    # 推理批次大小
    "init_lr_frac": 0.05,      # 初始学习率系数
    
    # 评估
    "eval_every": 60,          # 评估频率
    "eval_examples": 400,      # 评估样本数
}
```

### C. 常见问题

**Q1: RL 必须做吗？**
- 不是必须，但对复杂任务很有用
- 简单应用可以只做到 SFT

**Q2: 为什么用 GSM8K？**
- 有明确的对错标准
- 适合评估推理能力
- 可以换成其他任务

**Q3: 能不能用更多样本？**
- 可以，但计算成本成正比
- 16 通常足够，更多边际收益递减

**Q4: 奖励函数怎么设计？**
- 简单二元（0/1）就很好
- 可以加入其他维度（格式、效率）
- 避免过于复杂

**Q5: RL 会破坏 SFT 能力吗？**
- 可能，如果学习率太高
- 所以要从 SFT 开始，小心调整
- 定期评估通用能力

---

## 总结

**RL 的三个核心创新：**

1. **策略梯度** - 从试错中学习
2. **多样化采样** - 探索不同策略
3. **奖励优化** - 强化好的行为

**完整训练流程：**
```
Base (语言) → Mid (任务) → SFT (行为) → RL (优化)
    ↓             ↓            ↓           ↓
 会说话        会做事       听指令      做得好
```

**下一步：**
- 部署模型服务用户
- 收集用户反馈
- 持续优化改进

恭喜你！现在你已经完整理解了 LLM 的四阶段训练流程！

---

*本文档基于 nanochat 项目分析生成*  
*适合 LLM 初学者理解 RL 的完整流程*  
*创建时间: 2025年12月21日*
