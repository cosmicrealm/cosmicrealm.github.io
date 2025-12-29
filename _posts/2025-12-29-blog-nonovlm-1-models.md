---
title: 'noao-vlm-1 架构详细分析'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-29-blog-nonovlm-1-models/
tags:
  - llm
---



### nanoVLM 模型架构与数据流转分析

[nonovlm](https://github.com/huggingface/nanoVLM)

## 目录
1. [模型整体架构](#模型整体架构)
2. [各组件详细分析](#各组件详细分析)
3. [数据流转与维度变化](#数据流转与维度变化)
4. [损失函数与训练目标](#损失函数与训练目标)
5. [前向传播完整流程](#前向传播完整流程)
6. [生成推理流程](#生成推理流程)

---

## 模型整体架构

### 核心组件概览

nanoVLM 是一个多模态视觉-语言模型，由三个主要组件构成：

```
┌──────────────────────────────────────────────────────────────┐
│                         nanoVLM                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │  Vision       │     │  Modality    │     │  Language    ││
│  │  Encoder      │ --> │  Projector   │ --> │  Decoder     ││
│  │  (ViT)        │     │  (MP)        │     │  (LLM)       ││
│  └───────────────┘     └──────────────┘     └──────────────┘│
│                                                               │
│  输入: 图像 + 文本      中间: 图像特征        输出: 文本生成  │
│                        转语言空间                             │
└──────────────────────────────────────────────────────────────┘
```

### 组件配置参数

| 组件 | 默认模型 | 参数量 | 输出维度 |
|------|---------|--------|---------|
| **Vision Encoder** | `google/siglip2-base-patch16-512` | ~86M | 768 |
| **Modality Projector** | Custom (Linear) | ~4M | 960 |
| **Language Decoder** | `HuggingFaceTB/SmolLM2-360M-Instruct` | ~360M | 960 |
| **总参数量** | - | **~450M** | - |

---

## 各组件详细分析

### 1. Vision Encoder (ViT)

#### 架构组成
```python
Vision Transformer (ViT)
├─ Patch Embedding Layer
│  ├─ Conv2d(3, 768, kernel=16, stride=16)  # 提取 patch
│  └─ Positional Embedding (learnable)
│
└─ Transformer Blocks (×12)
   ├─ Layer Norm
   ├─ Multi-Head Self-Attention (12 heads)
   │  ├─ QKV Projection: Linear(768, 768*3)
   │  └─ Output Projection: Linear(768, 768)
   ├─ Residual Connection
   ├─ Layer Norm
   ├─ MLP (Feed-Forward)
   │  ├─ Linear(768, 3072)  # 4x expansion
   │  ├─ GELU
   │  └─ Linear(3072, 768)
   └─ Residual Connection
```

#### 关键参数
- **图像大小**: 512×512 (训练时), 最大支持 2048×2048
- **Patch 大小**: 16×16
- **每张图的 patch 数**: (512/16)² = 1024
- **隐藏维度**: 768
- **注意力头数**: 12
- **Transformer 层数**: 12
- **无 CLS token**: `vit_cls_flag=False`

#### 输入输出维度
```
输入:  [B, 3, H, W]          # H, W ∈ [512, 2048]
      ↓ Patch Embedding
中间:  [B, num_patches, 768]  # num_patches = (H/16) × (W/16)
      ↓ Transformer Blocks (×12)
输出:  [B, num_patches, 768]  # 例: 512×512 → [B, 1024, 768]
```

### 2. Modality Projector (MP)

#### 核心算法：Pixel Shuffle

Modality Projector 的关键创新是使用 **Pixel Shuffle** 降低序列长度，同时增加特征维度。

```python
# 1. Pixel Shuffle 原理
输入: [B, H×W, D]           # H×W = 1024, D = 768
重塑: [B, H, W, D]          # [B, 32, 32, 768]
分组: [B, H/4, 4, W/4, 4, D]  # scale_factor = 4
重排: [B, H/4, W/4, 4, 4, D]
展平: [B, H×W/16, D×16]     # [B, 64, 768×16]

# 2. 线性投影
输入: [B, 64, 12288]         # 768 × 16 = 12288
投影: Linear(12288, 960)
输出: [B, 64, 960]           # 匹配语言模型维度
```

#### 维度变化详解
```
原始图像 token: 1024 个 (32×32 grid)
     ↓ Pixel Shuffle (factor=4)
压缩后 token:   64 个 (8×8 grid)
     ↓ 维度变化
特征维度: 768 → 12288 → 960
```

**设计优势**：
- ✅ **减少序列长度**: 1024 → 64 (压缩 16 倍)
- ✅ **保留空间信息**: 通过重排而非池化
- ✅ **降低计算成本**: 语言模型处理的 token 数大幅减少
- ✅ **信息无损**: 通过增加维度补偿序列长度减少

#### 参数量
```python
Linear(12288, 960) without bias
参数量 = 12288 × 960 = 11,796,480 ≈ 11.8M
```

### 3. Language Decoder (LLM)

#### 架构组成
```python
Language Model (SmolLM2-360M)
├─ Token Embedding
│  └─ Embedding(49218, 960)  # vocab_size = 49152 + 66 特殊 token
│
├─ Transformer Blocks (×32)
│  ├─ RMS Norm
│  ├─ Grouped Query Attention (GQA)
│  │  ├─ Query Projection: Linear(960, 960)     # 15 heads
│  │  ├─ Key Projection:   Linear(960, 320)     # 5 kv_heads
│  │  ├─ Value Projection: Linear(960, 320)     # 5 kv_heads
│  │  ├─ Rotary Position Embedding (RoPE)
│  │  └─ Output Projection: Linear(960, 960)
│  ├─ Residual Connection
│  ├─ RMS Norm
│  ├─ MLP (Feed-Forward)
│  │  ├─ Gate + Up: Linear(960, 2560×2)
│  │  ├─ SwiGLU Activation
│  │  └─ Down: Linear(2560, 960)
│  └─ Residual Connection
│
└─ LM Head
   └─ Linear(960, 49218)  # 输出 logits
```

#### 关键特性

##### Grouped Query Attention (GQA)
```
Query Heads:    15 个
KV Heads:       5 个
Grouping Ratio: 15 / 5 = 3

每个 KV head 被 3 个 Query head 共享
→ 减少 KV cache 大小
→ 提升推理效率
```

##### RoPE (Rotary Position Embedding)
```python
# 位置编码通过旋转矩阵注入
θ_i = 1 / (10000^(2i/d))  # 频率
Position m → Rotation angle: m × θ_i

Apply to Q, K:
q_rotated = (q * cos(θ)) + (rotate_half(q) * sin(θ))
k_rotated = (k * cos(θ)) + (rotate_half(k) * sin(θ))
```

**优势**：
- 相对位置编码
- 外推能力强
- 无需额外参数

##### RMS Normalization
```python
# 替代 LayerNorm
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

参数量: 只需 γ (可学习缩放因子)
比 LayerNorm 少了均值和偏置
```

#### 关键参数
- **隐藏维度**: 960
- **中间维度**: 2560
- **注意力头数**: 15 (Query), 5 (KV)
- **每个头维度**: 960/15 = 64
- **Transformer 层数**: 32
- **词表大小**: 49152 + 66 = 49218
- **最大序列长度**: 4096

---

## 数据流转与维度变化

### 训练阶段完整流程

```
┌─────────────────────────────────────────────────────────────┐
│                     输入数据准备                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────┐
        │  批次数据 (batch)                     │
        ├──────────────────────────────────────┤
        │  images: List[List[Tensor]]          │
        │    - 嵌套列表，支持多图像             │
        │    - 每个图像: [C, H, W]              │
        │  input_ids: [B, T_seq]                │
        │    - 包含 <|image|> 占位符            │
        │  attention_mask: [B, T_seq]           │
        │  labels: [B, T_seq]                   │
        │    - 非答案部分标记为 -100             │
        └──────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  第一步：图像处理                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
        展平图像列表 & 合并成批次
        images: List[List[Tensor]] → Tensor
        ↓
        [N_images, 3, H, W]  # N_images = 批次中所有图像数量
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Vision Encoder (ViT) 前向传播                      │
└─────────────────────────────────────────────────────────────┘
        [N_images, 3, H, W]
        ↓ Patch Embedding
        [N_images, (H/16)×(W/16), 768]
        ↓ Transformer Blocks (×12)
        [N_images, num_patches, 768]
        
        示例: H=W=512 → num_patches = 1024
        输出: [N_images, 1024, 768]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           Modality Projector (MP) 转换                        │
└─────────────────────────────────────────────────────────────┘
        [N_images, 1024, 768]
        ↓ Pixel Shuffle (factor=4)
        [N_images, 64, 12288]  # 64 = 1024/16, 12288 = 768×16
        ↓ Linear Projection
        [N_images, 64, 960]
        
        关键: mp_image_token_length = 64
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              第二步：文本 Token 嵌入                          │
└─────────────────────────────────────────────────────────────┘
        input_ids: [B, T_seq]
        ↓ Token Embedding
        token_embd: [B, T_seq, 960]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         第三步：图像嵌入替换文本中的占位符                      │
└─────────────────────────────────────────────────────────────┘
        
        操作: _replace_img_tokens_with_embd()
        
        # 1. 找到所有 <|image|> token 位置
        mask = (input_ids == tokenizer.image_token_id)
        # shape: [B, T_seq], dtype: bool
        
        # 2. 将图像嵌入填充到这些位置
        updated_token_embd = token_embd.clone()
        updated_token_embd[mask] = image_embd.view(-1, 960)
        
        # 详细示例:
        # 假设 batch_size = 2, T_seq = 4096
        # Sample 1: 2 个图像, 每个 64 tokens
        # Sample 2: 1 个图像, 64 tokens
        
        # input_ids:
        # [
        #   [text, text, <img>, <img>, ..., <img>, text, ...],  # 128 个 <img>
        #   [text, <img>, <img>, ..., <img>, text, text, ...],  # 64 个 <img>
        # ]
        
        # image_embd: [3, 64, 960]  # 3 个图像
        # 展平: [192, 960]  # 3×64 = 192
        
        # 按顺序填充到 mask 位置
        输出: [B, T_seq, 960]  # 混合了文本和图像的嵌入
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Language Decoder 前向传播                          │
└─────────────────────────────────────────────────────────────┘
        [B, T_seq, 960]
        ↓ Transformer Blocks (×32)
        │  每个 Block:
        │  ├─ RMS Norm
        │  ├─ GQA (Grouped Query Attention)
        │  │  ├─ 计算 Q: [B, 15, T_seq, 64]
        │  │  ├─ 计算 K, V: [B, 5, T_seq, 64]
        │  │  ├─ Apply RoPE to Q, K
        │  │  ├─ Repeat K, V for grouping: [B, 15, T_seq, 64]
        │  │  ├─ Scaled Dot-Product Attention
        │  │  │  Attention = softmax(Q @ K^T / sqrt(64))
        │  │  └─ Output = Attention @ V
        │  ├─ Residual + RMS Norm
        │  └─ MLP (SwiGLU)
        ↓
        [B, T_seq, 960]  # 上下文表示
        ↓ LM Head
        [B, T_seq, 49218]  # Logits
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   损失计算                                    │
└─────────────────────────────────────────────────────────────┘
        logits: [B, T_seq, 49218]
        labels: [B, T_seq]
        
        # 展平用于 cross_entropy
        logits_flat = logits.view(-1, 49218)  # [B×T_seq, 49218]
        labels_flat = labels.view(-1)         # [B×T_seq]
        
        # 计算交叉熵损失 (ignore_index=-100)
        loss = F.cross_entropy(
            logits_flat, 
            labels_flat, 
            ignore_index=-100
        )
        
        # -100 位置包括:
        # - 问题部分的所有 token
        # - 图像占位符 token
        # - Padding token
        # 只有答案部分的 token 计入损失
                            ↓
        返回: (logits, loss)
```

### 维度变化总结表

| 阶段 | 输入维度 | 操作 | 输出维度 |
|------|---------|------|---------|
| **原始数据** | - | 准备批次 | - |
| 图像 | `List[List[Tensor]]` | 展平合并 | `[N_img, 3, H, W]` |
| 文本 | `[B, T_seq]` | - | `[B, T_seq]` |
| **Vision Encoder** | | | |
| Patch Embed | `[N_img, 3, H, W]` | Conv2d + 展平 | `[N_img, N_patch, 768]` |
| ViT Blocks | `[N_img, N_patch, 768]` | Self-Attn×12 | `[N_img, N_patch, 768]` |
| **Modality Projector** | | | |
| Pixel Shuffle | `[N_img, 1024, 768]` | 重排维度 | `[N_img, 64, 12288]` |
| Linear Proj | `[N_img, 64, 12288]` | 线性层 | `[N_img, 64, 960]` |
| **Language Decoder** | | | |
| Token Embed | `[B, T_seq]` | 查表 | `[B, T_seq, 960]` |
| Replace Tokens | `[B, T_seq, 960]` | 替换占位符 | `[B, T_seq, 960]` |
| LM Blocks | `[B, T_seq, 960]` | GQA×32 | `[B, T_seq, 960]` |
| LM Head | `[B, T_seq, 960]` | 线性层 | `[B, T_seq, 49218]` |
| **损失计算** | | | |
| Flatten | `[B, T_seq, 49218]` | reshape | `[B×T_seq, 49218]` |
| CrossEntropy | logits + labels | 计算损失 | scalar |

### 特殊 Token 处理

#### 图像 Token
```python
vlm_extra_tokens = {
    "image_token": "<|image|>",          # 标准图像占位符
    "global_image_token": "<|global_image|>",  # 全局图像 token
    
    # 网格位置 token (8×8 = 64 个)
    "r1c1": "<row_1_col_1>", ..., "r1c8": "<row_1_col_8>",
    "r2c1": "<row_2_col_1>", ..., "r2c8": "<row_2_col_8>",
    ...
    "r8c1": "<row_8_col_1>", ..., "r8c8": "<row_8_col_8>",
}

# 总共 66 个额外 token
vocab_size = 49152 + 66 = 49218
```

#### 数据格式示例
```python
# 单图像对话
input_text = """
<|im_start|>user
<|image|><|image|>...<|image|>  # 64 个 <|image|> token
What is in this image?<|im_end|>
<|im_start|>assistant
"""

# 对应的 labels
labels = [
    -100, -100, ..., -100,  # user 部分和问题: ignore
    token_1, token_2, ...   # assistant 答案: 计算损失
]
```

---

## 损失函数与训练目标

### 损失函数详解

#### 交叉熵损失 (Cross-Entropy Loss)

```python
def forward(self, input_ids, images, attention_mask=None, targets=None):
    # ... 前向传播 ...
    
    if targets is not None:
        logits = self.decoder.head(logits)  # [B, T_seq, vocab_size]
        
        # 损失计算
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B×T_seq, vocab_size]
            targets.reshape(-1),                   # [B×T_seq]
            ignore_index=-100                      # 忽略非答案 token
        )
    
    return logits, loss
```

#### 数学公式

对于单个样本中的每个 token  $$ t $$：

$$
\text{Loss}_t = -\log P(y_t | x_{<t}, I)
$$

其中：
- $$ y_t $$: 真实标签（目标 token）
- $$ x_{<t} $$: 之前的文本上下文
- $$ I $$: 图像信息
- $$ P $$:  模型预测的概率分布

总损失（批次平均）：

$$
\mathcal{L} = \frac{1}{N_{valid}} \sum_{t \in \text{valid}} -\log \frac{\exp(z_{y_t})}{\sum_{j=1}^{V} \exp(z_j)}
$$

其中：

- $$ N_{valid} $$: 有效 token 数量（不包括 -100）
- $$ z_j $$: 第 $$ j $$ 个词的 logit 值
- $$ V $$: 词表大小 (49218)

### 训练目标

#### 主要目标：条件文本生成
```
给定: 
  - 图像 I
  - 问题文本 Q
  
目标:
  - 生成答案文本 A
  
优化:
  max P(A | I, Q) = max ∏ P(a_t | I, Q, a_{<t})
```

#### Label Masking 策略

```python
# 示例对话
conversation = [
    {
        "role": "user",
        "content": "<|image|>...<|image|> What do you see?"
    },
    {
        "role": "assistant", 
        "content": "I see a cat sitting on a table."
    }
]

# Token 化后的 labels
labels = [
    # System tokens
    -100,  # <|im_start|>
    -100,  # user
    
    # User message
    -100, -100, ..., -100,  # <|image|> tokens (64个)
    -100, -100, ..., -100,  # "What do you see?" tokens
    -100,  # <|im_end|>
    
    # Assistant prefix
    -100,  # <|im_start|>
    -100,  # assistant
    
    # Assistant answer (这部分计入损失!)
    40,    # "I"
    588,   # "see"
    247,   # "a"
    2566,  # "cat"
    ...    # 其余答案 tokens
]
```

**Masking 原则**：
1. ❌ **System tokens**: `-100` (ignore)
2. ❌ **User 消息**: `-100` (ignore)
3. ❌ **图像占位符**: `-100` (ignore)
4. ✅ **Assistant 答案**: 真实 token ID (计算损失)

### 训练损失组成

```
总损失 = 只计算 Assistant 生成部分的交叉熵

有效 token 比例 ≈ 20-30%
(取决于问答长度比例)
```

#### 损失计算示例

```python
# 假设一个批次
batch_size = 2
seq_len = 4096
vocab_size = 49218

# 模型输出
logits = model(...)  # [2, 4096, 49218]
labels = ...         # [2, 4096]

# labels 示例
# Sample 1: 前 3500 个 token 是 -100, 后 596 个是有效 token
# Sample 2: 前 3800 个 token 是 -100, 后 296 个是有效 token

# 损失计算
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # [8192, 49218]
    labels.view(-1),              # [8192]
    ignore_index=-100
)

# 实际计算:
# 有效 token 数 = 596 + 296 = 892
# 总 token 数 = 8192
# 有效比例 = 892 / 8192 ≈ 10.9%

# loss 是 892 个有效位置的平均损失
```

### 优化目标层次

#### 1. 低级目标：Next Token Prediction
```
给定前缀 token 序列，预测下一个 token
P(token_t | token_1, ..., token_{t-1}, image)
```

#### 2. 中级目标：视觉理解与语言对齐
```
- 学习将视觉特征映射到语言空间
- 理解图像内容
- 建立视觉-文本对应关系
```

#### 3. 高级目标：多模态对话能力
```
- 视觉问答 (VQA)
- 图像描述 (Image Captioning)
- 视觉推理 (Visual Reasoning)
- OCR 识别
- 图表理解
```

---

## 前向传播完整流程

### 训练模式 (forward)

```python
def forward(self, input_ids, images, attention_mask=None, targets=None):
    """
    参数:
        input_ids: [B, T_seq] - 包含文本和图像占位符
        images: List[List[Tensor]] - 批次中的所有图像
        attention_mask: [B, T_seq] - 1表示有效, 0表示padding
        targets: [B, T_seq] - 标签, -100表示ignore
        
    返回:
        logits: [B, T_seq, vocab_size] - 每个位置的预测分布
        loss: scalar - 交叉熵损失 (仅计算非-100位置)
    """
    
    # Step 1: 处理图像
    images_tensor = self._process_images(images, device)
    # [N_images, 3, H, W]
    
    # Step 2: 文本嵌入
    token_embd = self.decoder.token_embedding(input_ids)
    # [B, T_seq, 960]
    
    # Step 3: 图像编码
    if images_tensor is not None:
        image_embd = self.vision_encoder(images_tensor)
        # [N_images, num_patches, 768]
        
        image_embd = self.MP(image_embd)
        # [N_images, 64, 960]
        
        # Step 4: 替换图像占位符
        token_embd = self._replace_img_tokens_with_embd(
            input_ids, token_embd, image_embd
        )
        # [B, T_seq, 960] - 混合嵌入
    
    # Step 5: 语言模型前向
    logits, _ = self.decoder(token_embd, attention_mask=attention_mask)
    # [B, T_seq, 960]
    
    # Step 6: 计算损失
    loss = None
    if targets is not None:
        logits = self.decoder.head(logits)
        # [B, T_seq, vocab_size]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )
    
    return logits, loss
```

### 详细子步骤

#### _process_images
```python
def _process_images(self, images, device):
    """
    处理嵌套的图像列表
    
    输入: List[List[Tensor]] 或 List[Tensor]
        例: [[img1, img2], [img3], []]
        表示: 样本1有2张图, 样本2有1张图, 样本3无图
    
    输出: Tensor [N_total_images, 3, H, W]
        例: [3, 3, 512, 512]
    """
    if isinstance(images, list):
        if images and isinstance(images[0], list):
            # 展平嵌套列表
            images = [img for sublist in images for img in sublist]
        
        if not images:
            return None
        else:
            return torch.cat(images, dim=0).to(device)
    
    return images  # 已经是 tensor
```

#### _replace_img_tokens_with_embd
```python
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    """
    用图像嵌入替换占位符 token
    
    参数:
        input_ids: [B, T_seq]
        token_embd: [B, T_seq, D_lm]
        image_embd: [N_images, 64, D_lm]
    
    输出:
        updated_token_embd: [B, T_seq, D_lm]
    """
    updated_token_embd = token_embd.clone()
    
    # 找到所有图像 token 位置
    mask = (input_ids == self.tokenizer.image_token_id)
    # [B, T_seq], dtype=bool
    
    # 展平图像嵌入并填充
    image_embd_flat = image_embd.view(-1, image_embd.size(-1))
    # [N_images × 64, D_lm]
    
    updated_token_embd[mask] = image_embd_flat.to(updated_token_embd.dtype)
    
    return updated_token_embd
```

---

## 生成推理流程

### 推理模式 (generate)

```python
@torch.inference_mode()
def generate(self, input_ids, images, attention_mask=None, 
             max_new_tokens=5, top_k=50, top_p=0.9, 
             temperature=0.5, greedy=False):
    """
    自回归生成新 token
    
    参数:
        input_ids: [B, T_prompt] - 提示文本（含图像占位符）
        images: 图像列表
        max_new_tokens: 生成的最大 token 数
        top_k, top_p: 采样参数
        temperature: 温度系数
        greedy: 是否贪婪解码
        
    返回:
        generated_ids: [B, max_new_tokens] - 生成的 token 序列
    """
```

### 两阶段生成

#### 阶段 1: Prefill（预填充）

```python
# 处理整个提示（文本 + 图像）
# 1. 图像编码
image_embd = self.vision_encoder(images_tensor)
image_embd = self.MP(image_embd)

# 2. 文本嵌入 + 替换占位符
token_embd = self.decoder.token_embedding(input_ids)
token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

# 3. 一次性处理所有 prompt tokens
prefill_output, kv_cache_list = self.decoder(
    token_embd,                # [B, T_prompt, D]
    attention_mask=attention_mask,
    kv_cache=None,             # 初始为空
    start_pos=0
)
# prefill_output: [B, T_prompt, D]
# kv_cache_list: 缓存所有层的 K, V

# 4. 获取最后一个 token 的输出
last_token_output = prefill_output[:, -1, :]  # [B, D]
current_logits = self.decoder.head(last_token_output)  # [B, vocab_size]
```

**Prefill 阶段特点**：
- 处理所有提示 token（包括图像）
- 并行计算所有位置的注意力
- 建立初始 KV cache
- 输出第一个生成 token 的 logits

#### 阶段 2: Decode（自回归解码）

```python
newly_generated_ids_list = []

for step in range(max_new_tokens):
    # 1. 采样下一个 token
    if greedy:
        next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
    else:
        # Top-k, Top-p 采样
        filtered_logits = top_k_top_p_filtering(current_logits, top_k, top_p)
        probs = torch.softmax(filtered_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
    # next_token_id: [B, 1]
    
    newly_generated_ids_list.append(next_token_id)
    
    # 2. 嵌入新 token
    next_token_embed = self.decoder.token_embedding(next_token_id)
    # [B, 1, D]
    
    # 3. 更新 attention mask
    attention_mask = torch.cat([attention_mask, ones([B, 1])], dim=1)
    
    # 4. 使用 KV cache 解码（只处理新 token）
    decode_output, kv_cache_list = self.decoder(
        next_token_embed,          # [B, 1, D] - 只传入新 token!
        attention_mask=attention_mask,
        kv_cache=kv_cache_list,    # 使用缓存的 K, V
        start_pos=current_total_seq_len
    )
    # decode_output: [B, 1, D]
    
    # 5. 获取 logits
    last_token_output = decode_output[:, -1, :]
    current_logits = self.decoder.head(last_token_output)
    # [B, vocab_size]
    
    current_total_seq_len += 1

# 拼接所有生成的 token
generated_ids = torch.cat(newly_generated_ids_list, dim=1)
# [B, max_new_tokens]
```

**Decode 阶段特点**：
- 每步只处理一个新 token
- 利用 KV cache 避免重复计算
- 自回归：每个 token 依赖之前的所有 token

### KV Cache 机制

#### 原理
```python
# 不使用 cache (Prefill)
for layer in layers:
    Q = W_q @ X_all     # [B, n_heads, T_all, d]
    K = W_k @ X_all     # [B, n_kv_heads, T_all, d]
    V = W_v @ X_all     # [B, n_kv_heads, T_all, d]
    
    Attention = softmax(Q @ K^T) @ V

# 使用 cache (Decode)
for layer in layers:
    Q_new = W_q @ X_new      # [B, n_heads, 1, d] - 只计算新 token
    K_new = W_k @ X_new      # [B, n_kv_heads, 1, d]
    V_new = W_v @ X_new      # [B, n_kv_heads, 1, d]
    
    # 拼接历史 K, V
    K = cat([K_cached, K_new], dim=2)  # [B, n_kv_heads, T_all, d]
    V = cat([V_cached, V_new], dim=2)
    
    # 只计算新 token 的注意力
    Attention = softmax(Q_new @ K^T) @ V  # [B, n_heads, 1, d]
    
    # 更新 cache
    K_cached = K
    V_cached = V
```

#### 复杂度对比
```
不使用 cache:
  计算量 = O(T² × d) per step
  总计算 = O(n × T² × d) for n tokens
  
使用 cache:
  计算量 = O(T × d) per step
  总计算 = O(n × T × d) for n tokens
  
加速比 = T/1 ≈ 数百倍（对长序列）
```

### 采样策略

#### Top-k Sampling
```python
def top_k_filtering(logits, top_k):
    """
    只保留概率最高的 k 个 token
    """
    values, indices = torch.topk(logits, top_k)
    min_value = values[:, -1].unsqueeze(-1)
    logits[logits < min_value] = -float('Inf')
    return logits
```

#### Top-p (Nucleus) Sampling
```python
def top_p_filtering(logits, top_p):
    """
    保留累积概率达到 p 的最小 token 集合
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 移除低概率 token
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = -float('Inf')
    return logits
```

#### Temperature Scaling
```python
# 控制输出分布的"锐度"
probs = softmax(logits / temperature)

temperature < 1: 更确定性（分布更尖锐）
temperature = 1: 原始分布
temperature > 1: 更随机（分布更平滑）
```

---

## 模型特性总结

### 架构优势

| 特性 | 实现方式 | 优势 |
|------|---------|------|
| **高效图像编码** | ViT + Pixel Shuffle | 减少 94% token 数量 |
| **参数高效** | 冻结骨干网络 + 微调 | 降低训练成本 |
| **注意力优化** | GQA (15:5 ratio) | 减少 KV cache |
| **位置编码** | RoPE | 相对位置 + 外推 |
| **归一化** | RMS Norm | 更少参数 + 更快 |
| **激活函数** | SwiGLU | 更好性能 |
| **多图像支持** | 灵活占位符 | 每样本 1-4 张图 |

### 计算效率

```
Prefill 阶段 (T_prompt = 4096):
├─ Vision Encoder: ~50ms per image
├─ Modality Projector: ~5ms
├─ Language Decoder: ~200ms
└─ 总计: ~255ms (单图像)

Decode 阶段 (使用 KV cache):
├─ Per token: ~15ms
└─ 100 tokens: ~1.5s

总推理时间 (100 tokens):
~255ms + ~1.5s ≈ 1.8s
```

### 显存占用

```
推理显存 (batch_size=1):
├─ 模型参数: ~1.8 GB (fp32) / ~0.9 GB (fp16)
├─ 图像特征: ~50 MB per image
├─ KV Cache: ~800 MB (seq_len=4096)
└─ 总计: ~2.7 GB (fp32) / ~1.8 GB (fp16)

训练显存 (batch_size=2, grad_accum=8):
├─ 模型参数: ~1.8 GB
├─ 梯度: ~1.8 GB
├─ 优化器状态: ~3.6 GB (AdamW)
├─ 激活值: ~8 GB
└─ 总计: ~17 GB per GPU
```

---

## 关键设计决策分析

### 1. Pixel Shuffle vs 其他降维方法

| 方法 | Token 数 | 信息保留 | 计算成本 |
|------|---------|---------|---------|
| **Pixel Shuffle** | 64 | 高（无损重排） | 低 |
| Average Pooling | 64 | 中（信息损失） | 低 |
| Learnable Pooling | 64 | 中-高 | 中 |
| Perceiver Resampler | 64 | 高 | 高 |
| 直接使用 | 1024 | 最高 | 最高 |

**选择 Pixel Shuffle 的原因**：
- ✅ 计算效率高
- ✅ 信息保留完整
- ✅ 实现简单
- ✅ 适合预训练骨干网络

### 2. GQA vs 标准 MHA

```
标准 MHA (Multi-Head Attention):
Q, K, V 头数: 15, 15, 15
KV Cache 大小: 2 × 15 × T × d

GQA (Grouped Query Attention):
Q, K, V 头数: 15, 5, 5
KV Cache 大小: 2 × 5 × T × d

节省: (15-5)/15 = 66.7% KV cache
```

**权衡**：
- ✅ 显著减少显存
- ✅ 提升推理速度
- ⚠️ 轻微性能下降（可忽略）

### 3. 差异化学习率

```python
MP Layer:           lr = 0.00512  (新初始化)
Vision Backbone:    lr = 5e-5     (预训练)
Language Backbone:  lr = 5e-5     (预训练)
```

**设计理念**：
- MP 层需要快速学习跨模态映射
- 骨干网络已有良好表示，只需精细调整
- 避免灾难性遗忘

---

## 总结

### 模型核心特点

1. **模块化设计**: Vision Encoder, MP, Language Decoder 独立可替换
2. **参数高效**: 450M 总参数，通过冻结和差异化学习率优化
3. **序列压缩**: 图像从 1024 tokens 压缩到 64 tokens
4. **多图像支持**: 灵活处理每样本 1-4 张图像
5. **高效推理**: KV cache + GQA 实现快速生成

### 训练目标

```
核心目标: 学习视觉-语言对齐
├─ 给定图像和问题
├─ 生成准确、连贯的答案
└─ 通过 Next Token Prediction 优化

评估维度:
├─ 视觉理解能力 (VQA, Image Captioning)
├─ 视觉推理能力 (Visual Reasoning)
├─ OCR 能力 (Text Recognition)
└─ 图表理解能力 (Chart Understanding)
```

### 数据流转路径

```
Image → ViT → MP → [替换占位符] → LLM → Logits → Loss
[B,3,H,W] [N,1024,768] [N,64,960]  [B,T,960]  [B,T,V]
  ↓                                    ↑
Text Tokens → Token Embedding → [B,T,960]
[B,T]                            (混合嵌入)
```

---