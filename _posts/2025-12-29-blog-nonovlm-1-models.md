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



接下来档详细说明 nanoVLM 中图像的处理流程，以及最终送入 LLM 的 token 序列构建方式。

## 目录
- [1. 图像处理流程](#1-图像处理流程)
- [2. Vision Encoder 处理](#2-vision-encoder-处理)
- [3. Modality Projector](#3-modality-projector)
- [4. Token 序列构建](#4-token-序列构建)
- [5. 完整的输入序列结构](#5-完整的输入序列结构)
- [6. 关键数值总结](#6-关键数值总结)
- [7. 特殊情况处理](#7-特殊情况处理)

---

## 1. 图像处理流程

### 1.1 动态调整大小 (DynamicResize)

**相关代码**：`data/custom_transforms.py` 中的 `DynamicResize` 类

**配置参数**：
- `max_img_size = 2048` 像素（长边限制）
- `vit_img_size = 512` 像素（每个patch的大小）
- `resize_to_max_side_len = True`

**调整规则**：
- 保持图像宽高比
- 长边 ≤ 2048 且能被 512 整除
- 短边按比例缩放，也能被 512 整除

**示例1：处理 1920×1080 的图像**
```
原始尺寸：1920×1080
长边：1920 → 调整到 1536 (最接近的512倍数)
缩放比例：1536/1920 = 0.8
短边：1080 × 0.8 = 864 → 向上取整到 1024 (512的倍数)
调整后尺寸：1536×1024
```

**示例2：处理 2400×1600 的图像**
```
原始尺寸：2400×1600
长边：2400 → 调整到 2048 (受max_img_size限制)
缩放比例：2048/2400 ≈ 0.853
短边：1600 × 0.853 ≈ 1365 → 向上取整到 1536 (512的倍数)
调整后尺寸：2048×1536
```

### 1.2 切分成 Patches (GlobalAndSplitImages)

**相关代码**：`data/custom_transforms.py` 中的 `GlobalAndSplitImages` 类

图像被切分成多个 512×512 的 patches，并生成一个 global patch。

**示例1：1536×1024 图像的切分**
```
水平切分：1536 ÷ 512 = 3 个patches
垂直切分：1024 ÷ 512 = 2 个patches
Local patches总数：3×2 = 6 个
Global patch：整张图缩小到 512×512 = 1 个
最终patches总数：7 个 (1 global + 6 local)
Grid结构：(n_h=2, n_w=3)
```

**示例2：512×512 图像的切分**
```
水平切分：512 ÷ 512 = 1 个patch
垂直切分：512 ÷ 512 = 1 个patch
Grid结构：(n_h=1, n_w=1)
特殊情况：只有1个patch时，不生成额外的global patch
最终patches总数：1 个
```

**示例3：2048×1536 图像的切分**
```
水平切分：2048 ÷ 512 = 4 个patches
垂直切分：1536 ÷ 512 = 3 个patches
Local patches总数：4×3 = 12 个
Global patch：1 个
最终patches总数：13 个 (1 global + 12 local)
Grid结构：(n_h=3, n_w=4)
```

---

## 2. Vision Encoder 处理

**相关代码**：`models/vision_transformer.py` 中的 `ViT` 类

### 2.1 ViT 配置
```python
vit_hidden_dim: 768        # ViT的隐藏层维度
vit_patch_size: 16         # ViT将图像切成16×16的小patches
vit_img_size: 512          # 输入到ViT的图像大小
vit_n_blocks: 12           # Transformer层数
vit_n_heads: 12            # 注意力头数
```

### 2.2 处理流程

每个 512×512 的 patch 经过 ViT：

```
输入：512×512×3 的图像
切分：(512÷16) × (512÷16) = 32×32 = 1024 个小patches
每个小patch：16×16×3
经过ViT后：[1024, 768]
```

**以 1536×1024 图像为例（7个patches）**：
```
输入到ViT：[7, 3, 512, 512]
ViT输出：[7, 1024, 768]
```
其中：
- 7 = 1个global patch + 6个local patches
- 1024 = 每个512×512图像产生的token数量
- 768 = ViT的隐藏层维度

---

## 3. Modality Projector

**相关代码**：`models/modality_projector.py` 中的 `ModalityProjector` 类

### 3.1 Pixel Shuffle

**配置参数**：
```python
mp_pixel_shuffle_factor: 4    # 压缩因子
```

**作用**：将视觉tokens压缩，减少送入LLM的token数量

**计算过程**：
```
输入：[batch, 1024, 768]
Pixel Shuffle 4×：
  - Token数量：1024 → 1024 ÷ (4×4) = 64
  - 特征维度：768 → 768 × (4×4) = 12288
输出：[batch, 64, 12288]
```

### 3.2 Linear Projection

**配置参数**：
```python
vit_hidden_dim: 768           # ViT输出维度
lm_hidden_dim: 960            # LLM输入维度
mp_image_token_length: 64     # 每个patch的最终token数
```

**投影过程**：
```
输入：[batch, 64, 12288]
Linear层：12288 → 960
输出：[batch, 64, 960]
```

### 3.3 完整示例

**以 1536×1024 图像（7个patches）为例**：
```
ViT输出：     [7, 1024, 768]
Pixel Shuffle: [7, 64, 12288]
Linear投影：   [7, 64, 960]
展平后：       [448, 960]  (7×64=448个image tokens)
```

**关键结论**：每个512×512的图像patch最终产生 **64 个 image tokens**

---

## 4. Token 序列构建

**相关代码**：`data/processors.py` 中的 `get_image_string` 函数

### 4.1 特殊Token定义

```python
vlm_extra_tokens = {
    "image_token": "<|image|>",              # 实际的图像embedding占位符
    "global_image_token": "<|global_image|>", # 标记全局图像的开始
    "r1c1": "<row_1_col_1>",                 # 位置标记：第1行第1列
    "r1c2": "<row_1_col_2>",                 # 位置标记：第1行第2列
    "r1c3": "<row_1_col_3>",
    ...
    "r8c8": "<row_8_col_8>",                 # 最多支持 8×8 的grid
}
```

**Token功能**：
- `<|image|>`：会被替换成实际的image embedding（64维的向量）
- `<|global_image|>`：文本token，标识全局图像的开始
- `<row_i_col_j>`：文本token，标识局部patch的位置

### 4.2 单张图像的Token序列

**场景1：1536×1024 图像（2行3列grid）**

Token序列：
```
<|global_image|>                    (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
<row_1_col_1>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
<row_1_col_2>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
<row_1_col_3>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
<row_2_col_1>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
<row_2_col_2>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
<row_2_col_3>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
```

**Token统计**：
- Global部分：1（标记） + 64（embeddings） = 65 tokens
- Local部分：6个patches × (1（标记） + 64（embeddings）) = 390 tokens
- **总计**：65 + 390 = **455 tokens**

**场景2：512×512 图像（1×1 grid）**

Token序列：
```
<row_1_col_1>                       (1个文本token)
<|image|><|image|>...<|image|>      (64个image embeddings)
```

**Token统计**：
- 无global patch（因为只有1个patch）
- **总计**：1 + 64 = **65 tokens**

**场景3：2048×1536 图像（3行4列grid）**

Token序列结构：
```
<|global_image|> + 64个<|image|>
<row_1_col_1> + 64个<|image|>
<row_1_col_2> + 64个<|image|>
<row_1_col_3> + 64个<|image|>
<row_1_col_4> + 64个<|image|>
<row_2_col_1> + 64个<|image|>
<row_2_col_2> + 64个<|image|>
<row_2_col_3> + 64个<|image|>
<row_2_col_4> + 64个<|image|>
<row_3_col_1> + 64个<|image|>
<row_3_col_2> + 64个<|image|>
<row_3_col_3> + 64个<|image|>
<row_3_col_4> + 64个<|image|>
```

**Token统计**：
- Global部分：65 tokens
- Local部分：12个patches × 65 = 780 tokens
- **总计**：65 + 780 = **845 tokens**

### 4.3 多张图像的Token序列

**相关代码**：`data/processors.py` 中对多图像的处理

**场景：2张图像**
- 图1：1536×1024 (2×3 grid)
- 图2：1024×512 (1×2 grid，只有2个patches)

Token序列：
```
<image: 0>                          (1个文本token，标识第0张图)
<|global_image|>                    (1个文本token)
<|image|>×64                        (64个embeddings - global)
<row_1_col_1>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 1,1)
<row_1_col_2>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 1,2)
<row_1_col_3>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 1,3)
<row_2_col_1>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 2,1)
<row_2_col_2>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 2,2)
<row_2_col_3>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 2,3)
<image: 1>                          (1个文本token，标识第1张图)
<row_1_col_1>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 1,1)
<row_1_col_2>                       (1个文本token)
<|image|>×64                        (64个embeddings - patch 1,2)
```

**Token统计**：
- 图1：1（图像标识） + 1（global标记） + 64（global embeddings） + 6×(1+64) = **456 tokens**
- 图2：1（图像标识） + 2×(1+64) = **131 tokens**
- **图像总计**：456 + 131 = **587 tokens**

---

## 5. 完整的输入序列结构

**相关代码**：`data/datasets.py` 中的 `_get_messages` 和 `_prepare_inputs_and_loss_mask` 方法

### 5.1 Chat Template

使用 SmolLM2 的聊天格式：
```python
lm_chat_template = """
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
"""
```

格式化后：
```
<|im_start|>user
[图像tokens][问题文本]<|im_end|>
<|im_start|>assistant
[回答文本]<|im_end|>
```

### 5.2 具体示例

**场景**：用户上传2张图像，提问"这两张图有什么不同？"
- 图1：1536×1024
- 图2：1024×512

**完整Token序列**：

```
位置    Token内容                    类型        数量
----------------------------------------------------
0       <|im_start|>                文本token    1
1       user                        文本token    1
2       \n                          文本token    1
3       <image: 0>                  文本token    1
4       <|global_image|>            文本token    1
5-68    <|image|>...                embeddings   64
69      <row_1_col_1>               文本token    1
70-133  <|image|>...                embeddings   64
134     <row_1_col_2>               文本token    1
135-198 <|image|>...                embeddings   64
199     <row_1_col_3>               文本token    1
200-263 <|image|>...                embeddings   64
264     <row_2_col_1>               文本token    1
265-328 <|image|>...                embeddings   64
329     <row_2_col_2>               文本token    1
330-393 <|image|>...                embeddings   64
394     <row_2_col_3>               文本token    1
395-458 <|image|>...                embeddings   64
459     <image: 1>                  文本token    1
460     <row_1_col_1>               文本token    1
461-524 <|image|>...                embeddings   64
525     <row_1_col_2>               文本token    1
526-589 <|image|>...                embeddings   64
590     这                          文本token    1
591     两                          文本token    1
592     张                          文本token    1
593     图                          文本token    1
594     有                          文本token    1
595     什么                         文本token    1
596     不同                         文本token    1
597     ？                          文本token    1
598     <|im_end|>                  文本token    1
599     \n                          文本token    1
600     <|im_start|>                文本token    1
601     assistant                   文本token    1
602     \n                          文本token    1
603-    [模型生成的回答]             生成tokens   变长
```

**Token统计**：
- 系统格式tokens：约 10 个
- 图1的tokens：456 个
- 图2的tokens：131 个
- 问题文本tokens：约 8 个
- 助手前缀tokens：约 3 个
- **输入总计**：约 **608 tokens**

### 5.3 Labels (训练目标)

**相关代码**：`data/datasets.py` 中的 `_get_labels` 方法

```python
labels = input_ids.clone().masked_fill(~mask, -100)
labels = labels.roll(-1)  # 向左移1位，实现next token prediction
labels[-1] = -100
```

**Mask规则**（在 `_prepare_inputs_and_loss_mask` 中实现）：
```python
# 对于每条消息
for msg in messages:
    if msg["role"] == "assistant":
        # 只对助手的回答内容计算loss
        # 跳过 "<|im_start|>assistant\n" 这个前缀
        mask[start:end] = 1
```

**示例**（使用上面的场景）：
```
Position  Token                      Mask  Label
-------------------------------------------------
0-602     [所有用户输入和格式tokens]    0     -100
603       \n (assistant后的换行)       0     -100
604       第                          1     一
605       一                          1     张
606       张                          1     图
607       图                          1     是
...       [后续回答内容]               1     [下一个token]
最后      <|im_end|>                  1     -100
```

**关键点**：
- `mask=0` 的位置：label=-100，不计算loss
- `mask=1` 的位置：label=下一个token的ID，计算cross-entropy loss
- 所有图像tokens和用户问题都被mask掉
- 只有助手的实际回答内容参与loss计算

---

## 6. 关键数值总结

### 6.1 图像处理参数

| 参数名称 | 数值 | 说明 |
|---------|------|------|
| `vit_patch_size` | 16×16 | ViT将图像切分的最小单位 |
| `vit_img_size` | 512 | 送入ViT的图像patch大小 |
| `max_img_size` | 2048 | 输入图像长边的最大限制 |
| `vit_hidden_dim` | 768 | ViT的输出特征维度 |
| ViT每个patch的tokens | 1024 | (512÷16)² = 1024 |

### 6.2 Modality Projector参数

| 参数名称 | 数值 | 说明 |
|---------|------|------|
| `mp_pixel_shuffle_factor` | 4 | Pixel shuffle的压缩因子 |
| `mp_image_token_length` | 64 | 每个patch最终的token数量 |
| Pixel shuffle输入维度 | 768 | 来自ViT |
| Pixel shuffle输出维度 | 12288 | 768 × 4² |
| Linear层输出维度 | 960 | LLM的hidden_dim |

### 6.3 LLM参数

| 参数名称 | 数值 | 说明 |
|---------|------|------|
| `lm_hidden_dim` | 960 | SmolLM2-360M的隐藏层维度 |
| `lm_max_length` | 4096 | 训练时的最大序列长度 |
| `lm_base_vocab_size` | 49152 | 基础词汇表大小 |
| `extra_token_amount` | 66 | 额外的特殊tokens |
| `lm_vocab_size` | 49218 | 总词汇表大小 |

### 6.4 训练配置

| 参数名称 | 数值 | 说明 |
|---------|------|------|
| `max_sample_length` | 4096 | 单个样本的最大token数 |
| `max_images_per_example` | 4 | 每个样本最多包含的图像数 |
| `batch_size` | 2 | 每个GPU的batch size |

### 6.5 不同图像尺寸的Token消耗

| 原始尺寸 | 调整后尺寸 | Grid | Patches | Image Tokens | 总Tokens |
|---------|-----------|------|---------|-------------|---------|
| 512×512 | 512×512 | 1×1 | 1 | 64 | 65 |
| 800×600 | 1024×512 | 1×2 | 2 | 128 | 130 |
| 1024×768 | 1024×1024 | 2×2 | 4+1(g) | 320 | 325 |
| 1536×1024 | 1536×1024 | 2×3 | 6+1(g) | 448 | 455 |
| 1920×1080 | 1536×1024 | 2×3 | 6+1(g) | 448 | 455 |
| 2048×1536 | 2048×1536 | 3×4 | 12+1(g) | 832 | 845 |
| 2048×2048 | 2048×2048 | 4×4 | 16+1(g) | 1088 | 1105 |

**说明**：
- `(g)` 表示包含1个global patch
- Image Tokens = patches × 64
- 总Tokens = Image Tokens + 位置标记tokens

---

## 7. 特殊情况处理

### 7.1 单个Patch的图像（1×1 grid）

**代码位置**：`data/custom_transforms.py` 中的 `GlobalAndSplitImages`

```python
if grid == (1, 1):
    return patches, grid  # 不添加global patch
```

**原因**：当图像只有一个patch时，该patch本身已经代表了全局信息，无需额外的global patch。

**Token序列**：
```
<row_1_col_1><|image|>×64
```
仅 65 tokens（而不是 1+64+1+64=130）

### 7.2 无图像的纯文本对话

**代码位置**：`data/datasets.py` 中的 `_process_data`

```python
if item['images'] is None:
    images_data = []
```

**Token序列**：
```
<|im_start|>user
你好，请介绍一下你自己。<|im_end|>
<|im_start|>assistant
我是一个AI助手...<|im_end|>
```

没有任何图像相关的tokens。

### 7.3 多图像场景

**最大支持**：`max_images_per_example = 4`

**Token序列模式**：
```
<image: 0>[图1的所有tokens]<image: 1>[图2的所有tokens]<image: 2>[图3的tokens]<image: 3>[图4的tokens][文本内容]
```

**限制原因**：
- 避免序列过长超过 `lm_max_length = 4096`
- 4张 2048×2048 的图像就需要 4×1105 = 4420 tokens，已超过限制
- 实际训练中平均每张图约 300-500 tokens

### 7.4 位置标记的最大支持

**代码位置**：`models/config.py` 中的 `vlm_extra_tokens`

支持最大 8×8 的grid，即：
- 最大切分：8行 × 8列 = 64个local patches
- 加上1个global patch = 65个patches
- Token消耗：65 × 65 = 4225 tokens

**触发条件**：
- 图像尺寸达到 4096×4096 时会产生 8×8 grid
- 但受 `max_img_size = 2048` 限制，实际最大是 4×4 grid

### 7.5 图像预处理失败

**代码位置**：`data/datasets.py` 中的 `_process_images`

```python
if isinstance(image, Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # 转换为RGB模式
```

**处理方式**：
- RGBA、L（灰度）等格式自动转换为RGB
- 如果转换失败，抛出 `ValueError`
- 该样本会被跳过（在 `__getitem__` 中返回 `None`）

### 7.6 序列长度超限

**代码位置**：`data/collators.py` 中的 `_discard_samples_that_are_too_long`

```python
filtered = [
    (ids, label, attn, img)
    for ids, label, attn, img in zip(...)
    if len(ids) <= max_length
]
```

**处理方式**：
- 训练时：超过 `max_sample_length = 4096` 的样本直接丢弃
- Collator层面：超过 `max_length` 的样本被过滤掉
- 不会截断，避免破坏图像token的完整性

---

## 8. 代码实现要点

### 8.1 图像Token的替换机制

**代码位置**：`models/vision_language_model.py` 中的 `_replace_img_tokens_with_embd`

```python
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    updated_token_embd = token_embd.clone()
    mask = (input_ids == self.tokenizer.image_token_id)
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1))
    return updated_token_embd
```

**关键点**：
1. `<|image|>` token在tokenizer中有唯一的ID（`image_token_id`）
2. 找到所有 `<|image|>` token的位置
3. 用对应的image embedding替换这些位置
4. 其他文本token保持不变

**示例**：
```python
input_ids:    [10, 15, <img_id>, <img_id>, <img_id>, 20, 25]
token_embd:   [[e10], [e15], [et], [et], [et], [e20], [e25]]  # et是文本embedding
image_embd:   [[img1], [img2], [img3]]  # 实际的image embeddings
替换后:
token_embd:   [[e10], [e15], [img1], [img2], [img3], [e20], [e25]]
```

### 8.2 多图像的处理

**代码位置**：`models/vision_language_model.py` 中的 `_process_images`

```python
def _process_images(self, images, device):
    if isinstance(images, list):
        if images and isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]
        if not images:
            return None
        else:
            return torch.cat(images, dim=0).to(device)
    return images
```

**处理逻辑**：
1. Batch中每个样本的images是一个list
2. 将所有样本的所有图像patches展平成一维
3. 拼接成一个大tensor
4. 一次性送入Vision Encoder处理

**示例**（batch_size=2）：
```python
样本1: [img1的7个patches]
样本2: [img2的5个patches, img3的3个patches]

展平后: [12个patches]
Vision Encoder输出: [12, 64, 960]

替换时按顺序对应：
样本1的image tokens: 前7个patches (448个tokens)
样本2的image tokens: 后5+3个patches (512个tokens)
```

### 8.3 训练时的Loss计算

**代码位置**：`models/vision_language_model.py` 中的 `forward`

```python
if targets is not None:
    logits = self.decoder.head(logits)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )
```

**关键点**：
- `ignore_index=-100`：所有label为-100的位置不计入loss
- 图像tokens、用户问题、格式tokens的label都是-100
- 只有助手回答的实际内容参与loss计算

---

## 9. 性能考虑

### 9.1 Token效率

**Pixel Shuffle的作用**：
- 不使用Pixel Shuffle：每个patch需要 1024 tokens
- 使用4× Pixel Shuffle：每个patch只需 64 tokens
- **压缩比**：1024 ÷ 64 = 16倍

**示例对比**（1536×1024图像）：
```
无压缩：7个patches × 1024 = 7168 tokens
有压缩：7个patches × 64 = 448 tokens
节省：7168 - 448 = 6720 tokens (93.7%)
```

### 9.2 序列长度管理

**Batch中的实际长度分布**：
```python
样本1: 600 tokens (1张小图 + 问答)
样本2: 1200 tokens (2张中图 + 问答)
样本3: 2500 tokens (1张大图 + 长文本)
样本4: 800 tokens (无图 + 问答)

Padding后: 所有样本都padding到2500
实际计算量: 4 × 2500 = 10000 tokens
有效计算量: 600+1200+2500+800 = 5100 tokens
效率: 51%
```

**优化**（使用 `ConstantLengthDataset`）：
- 将长短样本打包在一起
- 减少padding浪费
- 提高训练效率

### 9.3 内存消耗估算

**单张2048×2048图像**：
```
原始图像: 2048×2048×3×4 bytes = 48 MB (float32)
调整后: 2048×2048×3×4 bytes = 48 MB
ViT patches: 17个 × 512×512×3×4 bytes = 51 MB
ViT输出: 17×1024×768×4 bytes = 53 MB
MP输出: 17×64×960×4 bytes = 4 MB
Image tokens: 1088×960×4 bytes = 4 MB
```

**Batch处理**（batch_size=2，每样本2张大图）：
```
总图像数: 2×2 = 4张
总patches: 4×17 = 68个
ViT输出内存: 68×1024×768×4 ≈ 213 MB
MP输出内存: 68×64×960×4 ≈ 17 MB
```

---

## 10. 总结

### 10.1 核心设计思想

1. **分块处理**：将大图切分成固定大小的patches，支持任意分辨率
2. **全局+局部**：global patch捕捉整体信息，local patches捕捉细节
3. **位置编码**：通过 `<row_i_col_j>` tokens显式表示空间关系
4. **高效压缩**：Pixel Shuffle将视觉tokens压缩16倍

### 10.2 Token流向总览

```
原始图像 (1920×1080)
    ↓ DynamicResize
调整后图像 (1536×1024)
    ↓ GlobalAndSplitImages
7个patches (512×512)
    ↓ Vision Encoder (ViT)
[7, 1024, 768]
    ↓ Pixel Shuffle (4×)
[7, 64, 12288]
    ↓ Linear Projection
[7, 64, 960]
    ↓ 构建Token序列
<|global_image|><|image|>×64<row_1_col_1><|image|>×64...<row_2_col_3><|image|>×64
    ↓ 与文本拼接
<|im_start|>user\n[图像tokens][问题文本]<|im_end|>\n<|im_start|>assistant\n
    ↓ 送入LLM
生成回答
```

### 10.3 实际应用建议

1. **图像尺寸**：
   - 推荐使用接近512倍数的尺寸（如512、1024、1536）
   - 避免极端宽高比，减少padding浪费

2. **多图场景**：
   - 控制图像数量和尺寸，避免超过序列长度限制
   - 4张512×512的图比1张2048×2048更高效

3. **Token预算**：
   - 单张图约占300-500 tokens
   - 为文本内容预留至少1000-2000 tokens
   - 4096长度可容纳2-3张中等尺寸图 + 充足文本

4. **性能优化**：
   - 使用batch内长度相近的样本
   - 开启 `ConstantLengthDataset` 减少padding
   - 考虑使用 `compile=True` 加速训练
