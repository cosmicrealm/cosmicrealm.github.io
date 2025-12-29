---
title: 'noao-vlm-2 数据集与评估系统分析'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-29-blog-nonovlm-2-dataset_and_evaluate/
tags:
  - llm
---

### 数据集与评估系统分析

## 目录
1. [训练数据集系统](#训练数据集系统)
2. [数据处理流程](#数据处理流程)
3. [评估系统](#评估系统)
4. [评估指标与基准](#评估指标与基准)

---

## 训练数据集系统

### 1. 数据集概览

#### 默认训练数据集
```python
# 来自 models/config.py
train_dataset_path = 'HuggingFaceM4/FineVision_concat_shuffled_2'
train_dataset_name = ("default", )
```

**FineVision 数据集**是一个多模态指令微调数据集，包含多个子数据集的组合：

| 子数据集 | 类型 | 说明 |
|---------|------|------|
| `allava_laion` | 图像描述 | LAION 数据集的 ALLaVA 变体 |
| `allava_vflan` | VQA | Visual FLAN 指令数据 |
| `cambrian(filtered)` | 多任务 | Cambrian 过滤数据 |
| `LLaVA_Instruct_150K` | 对话 | LLaVA 指令数据 |
| `mmevol` | 进化数据 | 多模态进化数据集 |
| `sharegpt4o` | 对话 | ShareGPT-4O 数据 |
| `sharegpt4v(coco)` | COCO VQA | ShareGPT4V COCO 子集 |
| `sharegpt4v(knowledge)` | 知识问答 | 知识密集型问答 |
| `sharegpt4v(llava)` | LLaVA 数据 | ShareGPT4V LLaVA 子集 |
| `sharegpt4v(sam)` | 分割 | SAM 相关数据 |

#### 数据集特性
- **流式加载**: `stream_dataset=True` - 支持大规模数据集无需全部加载到内存
- **多配置**: 支持加载单个或多个子数据集配置
- **Shard 支持**: 可以加载预分片的数据集 (total_shards=56)
- **灵活组合**: 可以通过配置选择不同的数据集组合

### 2. 数据格式

#### 标准数据样本结构
```python
{
    "images": [PIL.Image, PIL.Image, ...],  # 图像列表 (可以是多张)
    "texts": [
        {
            "user": "问题文本",
            "assistant": "答案文本"
        },
        # 可以包含多轮对话
    ],
    # 可选的质量评分 (用于过滤)
    "relevance_ratings": [4, 5, ...],           # 相关性评分
    "image_correspondence_ratings": [5, 4, ...], # 图像对应性评分
    "visual_dependency_ratings": [4, 5, ...],    # 视觉依赖性评分
    "formatting_ratings": [5, 5, ...]            # 格式质量评分
}
```

#### 对话模板格式
```python
# 使用 ChatML 模板
template = """
<|im_start|>user
<|image|><|image|>...<|image|>  # 64个图像token
{user_question}<|im_end|>
<|im_start|>assistant
{assistant_answer}<|im_end|>
"""
```

### 3. 数据质量过滤

#### 四维评分系统
```python
class BaseDataset:
    def __init__(
        self,
        relevance_min_rating=1,              # 默认: 1
        image_correspondence_min_rating=1,   # 默认: 1
        visual_dependency_min_rating=1,      # 默认: 1
        formatting_min_rating=1,             # 默认: 1
    ):
```

**评分维度说明**：

| 维度 | 含义 | 评分标准 |
|------|------|----------|
| **Relevance** | 答案相关性 | 答案是否直接回答问题 |
| **Image Correspondence** | 图像对应性 | 答案与图像内容的匹配度 |
| **Visual Dependency** | 视觉依赖性 | 答案是否真正需要图像信息 |
| **Formatting** | 格式质量 | 文本格式、标点、语法质量 |

**过滤逻辑**：
```python
# 在 _get_messages() 中
for index, text in enumerate(item['texts']):
    # 如果任何评分低于阈值，跳过该对话轮次
    if item['relevance_ratings'][index] < self.relevance_min_rating:
        continue
    if item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
        continue
    # ... 其他评分检查
    
    messages.append({"role": "user", "content": text['user']})
    messages.append({"role": "assistant", "content": text['assistant']})
```

---

## 数据处理流程

### 1. 数据集类架构

```
BaseDataset (基类)
    ↓
VQADataset (Visual Question Answering)
    ↓
ConstantLengthDataset (恒定长度包装)
    ↓
DataLoader (PyTorch)
```

### 2. VQADataset 处理流程

#### 完整数据处理管道
```
原始数据 → VQADataset
    ↓
1. 图像处理 (_process_images)
    ├─ 加载 PIL 图像
    ├─ RGB 转换
    ├─ 动态调整大小 (DynamicResize)
    ├─ 转换为 Tensor
    └─ 分割成 patches (GlobalAndSplitImages)
    
2. 消息构建 (_get_messages)
    ├─ 应用质量过滤
    ├─ 构建对话列表
    └─ 添加图像占位符
    
3. Tokenization (_prepare_inputs_and_loss_mask)
    ├─ 应用 chat template
    ├─ Tokenize 文本
    ├─ 创建 attention mask
    └─ 创建 loss mask (标记需要计算损失的位置)
    
4. Label 生成 (_get_labels)
    ├─ Clone input_ids
    ├─ Mask 非答案部分 (-100)
    └─ Shift labels (因果 LM)
    
输出:
{
    "images": List[Tensor],      # 处理后的图像
    "input_ids": Tensor,         # Token IDs
    "attention_mask": Tensor,    # Attention mask
    "labels": Tensor             # 训练标签
}
```

### 3. 图像处理详解

#### DynamicResize（动态调整大小）
```python
class DynamicResize:
    """
    智能调整图像大小:
    1. 保持宽高比
    2. 长边 ≤ max_side_len (2048)
    3. 短边按比例缩放
    4. 两边都能被 patch_size (16) 整除
    """
    
    def _get_new_hw(self, h, w):
        # 示例: 原图 1920×1080
        long = max(h, w)  # 1920
        short = min(h, w)  # 1080
        
        # 计算目标长边 (向上取整到 patch_size 倍数)
        target_long = min(2048, ceil(1920/16)*16) = 1920
        
        # 计算缩放比例
        scale = 1920 / 1920 = 1.0
        
        # 计算目标短边 (向上取整)
        target_short = ceil(1080 * 1.0 / 16) * 16 = 1088
        
        return (1920, 1088)  # 新尺寸
```

**设计优势**：
- ✅ 不改变宽高比，避免图像变形
- ✅ 保证能被 patch_size 整除
- ✅ 支持可选的上采样到 max_side_len
- ✅ 高效的双三次插值

#### GlobalAndSplitImages（全局+分割图像）
```python
class GlobalAndSplitImages:
    """
    将图像分割成多个 patch，并可选添加全局 patch
    
    输入: [B, C, H, W]
    输出: [N_patches, C, patch_size, patch_size], (n_h, n_w)
    """
    
    def forward(self, x):
        # 1. 分割成 patches
        # 例: [1, 3, 512, 512] → patches [32, 3, 16, 16]
        patches, grid = self.splitter(x)  # grid = (32, 32)
        
        # 2. 如果只有一个 patch，直接返回
        if grid == (1, 1):
            return patches, grid
        
        # 3. 创建全局 patch（缩放整个图像）
        global_patch = resize(x, [patch_size, patch_size])
        # [1, 3, 16, 16]
        
        # 4. 拼接全局 patch 和分割 patches
        return torch.cat([global_patch, patches], dim=0), grid
        # [1025, 3, 16, 16] = 1 (global) + 1024 (32×32)
```

**全局 Patch 的作用**：
- 捕获图像的整体语义信息
- 补充局部 patch 的细节信息
- 类似于多尺度特征提取

#### 图像 Token 生成
```python
def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length=64):
    """
    根据图像分割信息生成占位符字符串
    
    参数:
        splitted_image_counts: [(n_h, n_w), ...] - 每张图像的分割网格
        mp_image_token_length: 64 - 每个 patch 对应的 token 数
        
    示例输出:
        "<|global_image|><|image|>×64<row_1_col_1><|image|>×64<row_1_col_2><|image|>×64..."
    """
    image_string = ""
    
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        # 多图像标记
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        
        # 全局图像 token
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            
            if n_h == 1 and n_w == 1:
                continue  # 只有一个 patch，无需分割 tokens
        
        # 位置化的 patch tokens
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f'r{i+1}c{j+1}')
                image_string += tokenizer.image_token * mp_image_token_length
    
    return image_string
```

**Token 结构示例**（2×2 分割）：
```
<|global_image|>
<|image|><|image|>...<|image|>  (64个)
<row_1_col_1>
<|image|><|image|>...<|image|>  (64个)
<row_1_col_2>
<|image|><|image|>...<|image|>  (64个)
<row_2_col_1>
<|image|><|image|>...<|image|>  (64个)
<row_2_col_2>
<|image|><|image|>...<|image|>  (64个)

总计: 1 (global) + 4 (patches) = 5 个单元
每个单元 64 tokens = 320 个 <|image|> tokens
```

### 4. Label 生成与 Loss Masking

#### Loss Mask 机制
```python
def _prepare_inputs_and_loss_mask(self, messages):
    # 1. Tokenize 整个对话
    conv_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    
    # 2. 初始化 mask (全部为 0 = ignore)
    mask = [0] * len(conv_ids["input_ids"])
    
    # 3. 标记 assistant 部分
    cursor = 0
    for msg in messages:
        segment_ids = tokenizer.apply_chat_template([msg], tokenize=True)
        seg_len = len(segment_ids)
        
        if msg["role"] == "assistant":
            # 跳过前缀 (如 "<|im_start|>assistant\n")
            start = cursor + self.prefix_len
            end = cursor + seg_len
            mask[start:end] = [1] * (end - start)
        
        cursor += seg_len
    
    return input_ids, mask, attention_mask
```

**Mask 示例**：
```
Tokens:    [<|im_start|>, user, \n, What, ..., <|im_end|>, <|im_start|>, assistant, \n, This, is, ...]
Mask:      [     0     ,  0  ,  0,  0 , ...,     0     ,      0      ,    0    ,  0,  1  ,  1, ...]
                              ↑ User 部分全部 mask
                                                                                        ↑ Assistant 答案计入损失
```

#### Label 生成
```python
def _get_labels(self, input_ids, mask):
    # 1. Clone input_ids
    labels = input_ids.clone()
    
    # 2. Mask 非答案部分为 -100
    labels = labels.masked_fill(~mask, -100)
    
    # 3. Shift labels (因果 LM: 预测下一个 token)
    labels = labels.roll(-1)
    
    # 4. 最后一个 token 没有目标
    labels[-1] = -100
    
    return labels
```

**示例**：
```
input_ids: [101, 102, 103, 104, 105]
mask:      [ 0,   0,   1,   1,   1 ]

Step 1 - masked_fill:
labels:    [-100, -100, 103, 104, 105]

Step 2 - roll(-1):
labels:    [-100, 103, 104, 105, -100]
           ↑ predict token 103
                 ↑ predict token 104
                      ↑ predict token 105
                           ↑ predict EOS
```

### 5. ConstantLengthDataset（恒定长度数据集）

这是一个关键的优化组件，实现智能的样本打包。

#### 核心功能
```python
class ConstantLengthDataset(IterableDataset):
    """
    将可变长度的样本打包成固定长度的批次
    
    关键参数:
        seq_length: 4096 - 目标序列长度
        max_sample_length: 4096 - 单个样本的最大长度
        max_images_per_example: 4 - 每个样本最多图像数
        max_images_per_knapsack: 18 - 每个批次最多图像数
    """
```

#### 背包问题算法
```python
def _balanced_greedy_knapsack(self, buffer, L, max_images_per_knapsack):
    """
    贪婪背包算法，同时考虑长度和图像数量约束
    
    目标:
        1. 将样本分组，每组总长度 ≤ L (4096)
        2. 每组图像总数 ≤ max_images_per_knapsack (18)
        3. 最大化长度利用率
    """
    
    # 1. 按长度降序排序
    items = sorted(enumerate(zip(lengths, image_counts)), 
                   key=lambda x: x[1][0], reverse=True)
    
    # 2. 初始化多个背包
    min_knapsacks = (sum(lengths) + L - 1) // L + delta
    knapsack_load = [0] * min_knapsacks
    knapsack_image_counts = [0] * min_knapsacks
    knapsack_groups = [[] for _ in range(min_knapsacks)]
    
    # 3. 贪婪分配
    for idx, (item_len, item_image_count) in items:
        # 寻找满足约束的背包
        for ks_id in sorted(range(len(knapsack_load)), 
                           key=knapsack_load.__getitem__):
            length_fits = knapsack_load[ks_id] + item_len <= L
            image_fits = (knapsack_image_counts[ks_id] + item_image_count 
                         <= max_images_per_knapsack)
            
            if length_fits and image_fits:
                knapsack_groups[ks_id].append(idx)
                knapsack_load[ks_id] += item_len
                knapsack_image_counts[ks_id] += item_image_count
                break
        else:
            # 创建新背包
            create_new_knapsack()
    
    # 4. 随机打乱（避免顺序偏差）
    random.shuffle(knapsack_groups)
    
    return knapsack_groups
```

#### 打包示例
```
原始样本:
Sample 1: 512 tokens, 1 image
Sample 2: 1024 tokens, 2 images
Sample 3: 2048 tokens, 3 images
Sample 4: 1024 tokens, 1 image
Sample 5: 256 tokens, 1 image

背包算法结果:
Knapsack 1: [Sample 3, Sample 5]
  - 总长度: 2048 + 256 = 2304 tokens (< 4096)
  - 总图像: 3 + 1 = 4 images (< 18)
  
Knapsack 2: [Sample 2, Sample 4, Sample 1]
  - 总长度: 1024 + 1024 + 512 = 2560 tokens
  - 总图像: 2 + 1 + 1 = 4 images

打包后批次:
Batch 1: [Sample 3, Sample 5, <padding>]
  - 长度: 4096 (填充 1792 tokens)
  - 利用率: 56.3%
  
Batch 2: [Sample 2, Sample 4, Sample 1, <padding>]
  - 长度: 4096 (填充 1536 tokens)
  - 利用率: 62.5%
```

#### 生产者-消费者模式
```python
def __iter__(self):
    """
    使用多线程实现高效的数据预取
    """
    queue = Queue(maxsize=self.queue_size)  # 缓冲队列
    
    # 生产者线程：持续读取和打包数据
    producer = threading.Thread(
        target=self._producer, 
        args=(make_base_iterator, queue), 
        daemon=True
    )
    producer.start()
    
    # 消费者（主线程）：从队列获取数据
    while True:
        batch_of_batches = queue.get()
        if batch_of_batches is self._sentinel:
            break
        for batch in batch_of_batches:
            yield batch
```

**优势**：
- ✅ 异步数据预取，减少训练等待
- ✅ 缓冲队列平滑数据流
- ✅ 支持多 worker 并行

### 6. Collator（批次整理器）

#### VQACollator
```python
class VQACollator(BaseCollator):
    """
    将多个样本整理成一个批次
    
    功能:
        1. 过滤 None 样本
        2. 丢弃超长样本
        3. Padding 到相同长度
        4. Stack 成 batch tensor
    """
    
    def __call__(self, batch):
        # 1. 过滤
        batch = [s for s in batch if s is not None]
        
        # 2. 转换为 dict of lists
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        
        # 3. 丢弃超长样本
        batch = self._discard_samples_that_are_too_long(batch, max_length)
        
        # 4. Padding
        max_len = max(map(len, batch["input_ids"]))
        batch["input_ids"] = [
            F.pad(ids, (max_len - len(ids), 0), value=pad_token_id) 
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            F.pad(labels, (max_len - len(labels), 0), value=-100)  # 注意: -100
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            F.pad(mask, (max_len - len(mask), 0), value=0)
            for mask in batch["attention_mask"]
        ]
        
        # 5. Stack 成 tensor
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "images": batch["images"],  # 保持为列表
        }
```

#### Padding 示例
```
原始批次 (3 个样本):
Sample 1: input_ids=[101, 102, 103], labels=[-100, -100, 104]
Sample 2: input_ids=[101, 102, 103, 104, 105], labels=[-100, -100, 104, 105, 106]
Sample 3: input_ids=[101], labels=[-100]

Padding 后 (max_len=5):
input_ids:
  [[0, 0, 101, 102, 103],
   [101, 102, 103, 104, 105],
   [0, 0, 0, 0, 101]]

labels:
  [[-100, -100, -100, -100, 104],
   [-100, -100, 104, 105, 106],
   [-100, -100, -100, -100, -100]]

attention_mask:
  [[0, 0, 1, 1, 1],
   [1, 1, 1, 1, 1],
   [0, 0, 0, 0, 1]]
```

**关键设计**：
- Padding 在左侧（保持答案在右侧对齐）
- Labels padding 使用 -100（CrossEntropy 忽略）
- Attention mask padding 使用 0（不关注 padding）

---

## 评估系统

### 1. 评估框架概览

nanoVLM 使用 **lmms-eval** 框架进行标准化评估。

```
训练循环 (train.py)
    ↓ 每 500 步
保存检查点
    ↓
提交 SLURM 任务 (eval.slurm)
    ↓
运行评估 (evaluation.py + run_evaluation.py)
    ↓
生成结果 JSON
    ↓
合并结果 (merge_eval_results.py)
    ↓
自动记录到 wandb
```

### 2. 评估任务（Tasks）

#### 默认评估任务
```python
lmms_eval_tasks = 'mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,
                   scienceqa,mme,infovqa_val,chartqa'
```

#### 任务详情

| 任务 | 全称 | 类型 | 指标 | 说明 |
|------|------|------|------|------|
| **mmstar** | MMStar | 综合理解 | Accuracy | 多模态综合评估基准 |
| **mmmu_val** | MMMU | 大学知识 | Accuracy | 大学级别多学科问答 |
| **ocrbench** | OCRBench | OCR | F1 Score | 文字识别能力 |
| **textvqa_val** | TextVQA | 场景文字 | Accuracy | 场景文字理解 |
| **docvqa_val** | DocVQA | 文档理解 | ANLS | 文档问答 |
| **scienceqa** | ScienceQA | 科学问答 | Accuracy | 科学知识问答 |
| **mme** | MME | 细粒度评估 | Accuracy | 14个子任务综合评估 |
| **infovqa_val** | InfoVQA | 信息图表 | ANLS | 信息图表问答 |
| **chartqa** | ChartQA | 图表理解 | Accuracy | 图表数据问答 |

### 3. NanoVLMWrapper（LMMS-Eval 适配器）

#### 核心功能
```python
class NanoVLMWrapper(lmms):
    """
    将 nanoVLM 模型适配到 lmms-eval 框架
    
    主要方法:
        - generate_until: 生成式任务
        - loglikelihood: 似然计算任务（未实现）
    """
    
    def __init__(self, model, device="cuda", batch_size=32):
        if isinstance(model, str):
            self.model = VisionLanguageModel.from_pretrained(model)
        else:
            self.model = model
        
        self.tokenizer = get_tokenizer(...)
        self.image_processor = get_image_processor(...)
```

#### 生成任务实现
```python
def generate_until(self, requests: List[Instance]) -> List[str]:
    """
    批量生成回答
    
    流程:
        1. 准备输入 (文本 + 图像)
        2. 分批处理
        3. 调用模型生成
        4. 后处理输出
    """
    
    # 1. 准备输入
    contexts = []
    visuals = []
    for request in requests:
        contexts.append(request.args[0])      # 问题文本
        visuals.append(request.args[1])      # 图像
    
    # 2. 分批处理
    results = []
    for batch_start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[batch_start:batch_start+batch_size]
        batch_visuals = visuals[batch_start:batch_start+batch_size]
        
        # 3. 处理图像
        images, splitted_ratios = self._prepare_visual_input(batch_visuals)
        
        # 4. 构建提示
        batch_inputs = []
        for context, ratios in zip(batch_contexts, splitted_ratios):
            if ratios:
                image_string = get_image_string(
                    self.tokenizer, ratios, self.model.cfg.mp_image_token_length
                )
                context = image_string + context
            
            messages = [{"role": "user", "content": context}]
            batch_inputs.append(messages)
        
        # 5. Tokenize
        tokenized = [
            self.tokenizer.apply_chat_template(msgs, tokenize=True, ...)
            for msgs in batch_inputs
        ]
        
        # 6. 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        
        # 7. 解码
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend(outputs)
    
    return results
```

### 4. 评估流程

#### 评估脚本调用
```bash
# eval.slurm
python run_evaluation.py \
    --checkpoint_path {checkpoint_path} \
    --global_step {global_step} \
    --run_name {run_name} \
    --tasks {tasks} \
    --limit {limit} \
    --batch_size {batch_size}
```

#### run_evaluation.py
```python
def main():
    # 1. 加载模型
    model = VisionLanguageModel.from_pretrained(checkpoint_path)
    model.eval()
    
    # 2. 包装模型
    wrapped_model = NanoVLMWrapper(model, device="cuda", batch_size=128)
    
    # 3. 运行评估
    eval_results = cli_evaluate(
        model=wrapped_model,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
    )
    
    # 4. 处理结果
    output_data = {
        'global_step': global_step,
        'results': {}
    }
    
    for task_name, task_results in eval_results["results"].items():
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                key = f"{task_name}_{metric_name}"
                output_data['results'][key] = metric_value
    
    # 5. 保存结果
    output_path = f'eval_results/{run_name}/step_{global_step}_{tasks}.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
```

#### 结果文件结构
```json
{
    "global_step": 1000,
    "results": {
        "mmstar_accuracy": 0.456,
        "mmmu_val_accuracy": 0.342,
        "ocrbench_f1": 0.678,
        "textvqa_val_accuracy": 0.523,
        "docvqa_val_anls": 0.589,
        "scienceqa_accuracy": 0.701,
        "mme_accuracy": 0.812,
        "infovqa_val_anls": 0.445,
        "chartqa_accuracy": 0.389
    }
}
```

### 5. 结果合并（merge_eval_results.py）

#### 合并逻辑
```python
def merge_results():
    """
    合并同一步数的多个任务评估结果
    
    输入文件:
        step_1000_mmstar.json
        step_1000_mmmu_val.json
        step_1000_ocrbench.json
        ...
    
    输出文件:
        step_1000.json  (合并所有任务)
    """
    
    merged_data = {"global_step": global_step, "results": {}}
    
    # 查找所有部分结果
    files = glob.glob(f"step_{global_step}_*.json")
    
    # 合并
    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)
            if "results" in data:
                merged_data["results"].update(data["results"])
        
        # 删除部分文件
        os.remove(file_path)
    
    # 保存合并结果
    with open(f"step_{global_step}.json", "w") as f:
        json.dump(merged_data, f, indent=4)
```

### 6. 自动 Wandb 集成

#### 训练循环中的检测
```python
# 在 train.py 中
if global_step % stats_log_interval == 0:
    # 检查评估结果目录
    eval_results_dir = os.path.join('eval_results', run_name)
    
    if os.path.exists(eval_results_dir):
        for result_file in os.listdir(eval_results_dir):
            # 匹配 "step_1234.json"
            match = re.fullmatch(r"step_(\d+)\.json", result_file)
            
            if match:
                step = int(match.group(1))
                
                if step not in logged_eval_steps:
                    # 读取结果
                    with open(result_file) as f:
                        eval_data = json.load(f)
                    
                    # 记录到 wandb
                    metrics = {
                        f"lmms_eval/{key}": value 
                        for key, value in eval_data['results'].items()
                    }
                    wandb.log(metrics, step=global_step)
                    
                    logged_eval_steps.add(step)
```

---

## 评估指标与基准

### 1. 核心指标

#### Accuracy（准确率）
```python
# 大多数任务使用
accuracy = correct_predictions / total_predictions

# 适用任务: mmstar, mmmu_val, textvqa_val, scienceqa, mme, chartqa
```

#### ANLS (Average Normalized Levenshtein Similarity)
```python
# 文档和信息图表任务使用
def anls(prediction, ground_truth):
    """
    考虑编辑距离的相似度度量
    更宽容于 OCR 和格式化错误
    """
    edit_distance = levenshtein(prediction.lower(), ground_truth.lower())
    max_len = max(len(prediction), len(ground_truth))
    
    if max_len == 0:
        return 1.0
    
    nl = edit_distance / max_len
    return 1.0 - nl if nl < 0.5 else 0.0

# 适用任务: docvqa_val, infovqa_val
```

#### F1 Score
```python
# OCR 任务使用
f1 = 2 * (precision * recall) / (precision + recall)

# 适用任务: ocrbench
```

### 2. 基准对比

#### 性能目标（参考值）

| 任务 | 基准模型 | nanoVLM 目标 | SOTA |
|------|---------|-------------|------|
| MMStar | GPT-4V: 0.567 | 0.45+ | 0.65+ |
| MMMU | GPT-4V: 0.561 | 0.35+ | 0.60+ |
| OCRBench | GPT-4V: 0.645 | 0.50+ | 0.75+ |
| TextVQA | LLaVA-1.5: 0.588 | 0.45+ | 0.70+ |
| DocVQA | LLaVA-1.5: 0.604 | 0.50+ | 0.80+ |
| ScienceQA | GPT-3.5: 0.754 | 0.65+ | 0.90+ |
| MME | InstructBLIP: 1220 | 1000+ | 2000+ |
| ChartQA | LLaVA-1.5: 0.390 | 0.35+ | 0.60+ |

**注**：这些是估计的目标值，实际性能取决于训练数据和超参数。

### 3. 评估配置

#### 关键参数
```python
# 评估配置
batch_size = 64              # 评估批次大小
limit = None                 # 样本数量限制 (None = 全部)
num_fewshot = 0              # Few-shot 示例数量
device = "cuda"              # 设备
temperature = 0.5            # 生成温度
top_p = 0.9                  # Nucleus sampling
max_new_tokens = 512         # 最大生成长度
```

#### 评估频率
```python
# 训练中的评估
eval_interval = 500           # 每 500 步评估一次
eval_in_epochs = True         # 在 epoch 中评估

# 评估任务频率
# 基础评估: 每 500 步
# lmms-eval: 每 1000 步 (eval_interval * 2)
```

### 4. 性能监控

#### Wandb 可视化指标
```python
# 训练指标
- batch_loss
- val_loss
- grad_norm
- tokens_per_second

# 评估指标 (自定义 x 轴)
- lmms_eval/mmstar_accuracy
- lmms_eval/mmmu_val_accuracy
- lmms_eval/ocrbench_f1
- lmms_eval/textvqa_val_accuracy
- lmms_eval/docvqa_val_anls
- lmms_eval/scienceqa_accuracy
- lmms_eval/mme_accuracy
- lmms_eval/infovqa_val_anls
- lmms_eval/chartqa_accuracy

# Epoch 统计
- epoch_loss
- epoch_duration
- epoch_tokens_per_second
```

#### 自定义 X 轴
```python
# 为 lmms-eval 指标设置独立的 x 轴
lmms_eval_step = "<lmms-eval-step>"
wandb.run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

# 记录时指定步数
wandb.log({
    "lmms_eval/mmstar_accuracy": 0.456,
    "<lmms-eval-step>": 1000  # 评估对应的训练步数
}, step=current_training_step)
```

---

## 数据处理最佳实践

### 1. 数据集准备

#### 推荐配置
```python
# 大规模训练
stream_dataset = True        # 流式加载
max_images_per_example = 4   # 限制每样本图像数
max_images_per_knapsack = 18 # 限制每批次图像数

# 质量过滤（逐步提高）
# 初期训练
relevance_min_rating = 1
image_correspondence_min_rating = 1
visual_dependency_min_rating = 1
formatting_min_rating = 1

# 精调阶段
relevance_min_rating = 3
image_correspondence_min_rating = 3
visual_dependency_min_rating = 2
formatting_min_rating = 3
```

### 2. 数据增强

#### 当前实现
- 动态调整大小（保持宽高比）
- 双三次插值
- 归一化（ToTensor）

#### 可选增强（未实现）
```python
# 可以添加的增强
transforms.Compose([
    DynamicResize(...),
    transforms.ColorJitter(0.1, 0.1, 0.1),  # 颜色抖动
    transforms.RandomRotation(5),            # 小角度旋转
    transforms.ToTensor(),
    GlobalAndSplitImages(...),
])
```

### 3. 显存优化技巧

#### 图像数量控制
```python
# 显存不足时调整
max_images_per_example = 2      # 从 4 降至 2
max_images_per_knapsack = 10    # 从 18 降至 10
```

#### 序列长度控制
```python
# 减少序列长度
seq_length = 2048               # 从 4096 降至 2048
max_sample_length = 2048
```

#### 批次大小
```python
# 动态调整
batch_size = 1                  # 最小批次
gradient_accumulation_steps = 16 # 增加累积步数
```

### 4. 数据质量监控

#### 关键指标
```python
# 在训练循环中记录
accumulated_stats = {
    'images_per_sample': [],      # 每样本图像数
    'tokens_per_sample': [],       # 每样本 token 数
    'padding_ratio': [],           # Padding 比例
    'filtered_ratio': [],          # 质量过滤比例
}

# 监控异常
if avg_images_per_sample > max_images_per_example:
    warnings.warn("Average images per sample exceeds limit")

if padding_ratio > 0.5:
    warnings.warn("High padding ratio, consider adjusting seq_length")
```

---

## 评估最佳实践

### 1. 评估策略

#### 渐进式评估
```python
# 训练初期（前 5000 步）
eval_interval = 1000         # 更频繁评估
use_lmms_eval = False        # 只用验证损失

# 训练中期（5000-20000 步）
eval_interval = 500
use_lmms_eval = True
lmms_eval_tasks = 'mmstar,textvqa_val'  # 快速任务

# 训练后期（20000+ 步）
eval_interval = 500
lmms_eval_tasks = 'mmstar,mmmu_val,ocrbench,textvqa_val,...'  # 全部任务
```

### 2. 任务选择

#### 快速反馈任务
```python
# 训练中使用（快速评估）
quick_tasks = 'mmstar,textvqa_val,scienceqa'

# 执行时间: ~10-15 分钟
```

#### 完整评估任务
```python
# 检查点评估（完整评估）
full_tasks = 'mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa'

# 执行时间: ~1-2 小时
```

### 3. 结果解读

#### 关注的指标
```python
# 核心能力
mmstar_accuracy       # 综合理解能力
mmmu_val_accuracy     # 知识密集型任务

# 特定能力
ocrbench_f1          # OCR 能力
docvqa_val_anls      # 文档理解
chartqa_accuracy     # 数据图表理解

# 趋势分析
# 1. 所有指标同时提升 → 模型整体改进
# 2. 某些指标下降 → 可能过拟合或灾难性遗忘
# 3. 指标波动大 → 可能需要调整学习率或增加数据
```

---

### 评估问题

#### 问题：评估结果未记录
```python
# 症状
Wandb 中看不到 lmms-eval 结果

# 检查步骤
1. 确认评估任务已完成
2. 检查 eval_results/{run_name}/ 目录
3. 确认文件命名格式正确 (step_N.json)
4. 检查训练循环是否运行到 stats_log_interval
```

#### 问题：评估速度慢
```python
# 优化方案
1. 增加 eval batch_size (64 → 128)
2. 使用更快的设备
3. 减少评估任务数量
4. 使用 limit 参数限制样本数
```

---

## 总结

### 数据集系统特点

✅ **灵活性**
- 支持多数据集组合
- 流式和批量模式
- 质量过滤机制

✅ **效率性**
- 背包算法优化打包
- 异步数据预取
- 智能图像处理

✅ **可扩展性**
- 模块化设计
- 易于添加新数据集
- 支持自定义处理

### 评估系统特点

✅ **标准化**
- 使用 lmms-eval 框架
- 支持多种基准任务
- 统一的评估接口

✅ **自动化**
- 训练中自动评估
- 结果自动合并
- Wandb 自动记录

✅ **全面性**
- 多维度能力评估
- 9+ 标准基准
- 详细的性能分析

### 关键配置建议

```python
# 生产环境推荐配置
train_cfg = {
    'batch_size': 2,
    'gradient_accumulation_steps': 8,
    'max_images_per_example': 4,
    'max_images_per_knapsack': 18,
    'seq_length': 4096,
    'stream_dataset': True,
    'eval_interval': 500,
    'use_lmms_eval': True,
}

# 资源受限环境
train_cfg = {
    'batch_size': 1,
    'gradient_accumulation_steps': 16,
    'max_images_per_example': 2,
    'max_images_per_knapsack': 10,
    'seq_length': 2048,
    'stream_dataset': True,
    'eval_interval': 1000,
    'use_lmms_eval': False,  # 只用验证损失
}
```
