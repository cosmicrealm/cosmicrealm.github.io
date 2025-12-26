---
title: 'noao-chat-7-nonochat 多卡训练指南'
date: 2025-12-25
permalink: /posts/2025/12/2025-12-25-blog-nonochat-7-nonochat-ddp/
tags:
  - llm
---

#### LLM 多卡训练完全指南

[nonochat](https://github.com/karpathy/nanochat)



> 本文档全面解析 nanochat 项目中的分布式训练（DDP），从环境准备到代码实现，让小白也能轻松理解和使用多卡训练。

---

## 目录

1. [什么是多卡训练](#1-什么是多卡训练)
2. [为什么需要多卡训练](#2-为什么需要多卡训练)
3. [环境准备](#3-环境准备)
4. [多卡启动方式](#4-多卡启动方式)
5. [代码实现详解](#5-代码实现详解)
6. [数据并行机制](#6-数据并行机制)
7. [分布式优化器](#7-分布式优化器)
8. [实战案例](#8-实战案例)
9. [常见问题](#9-常见问题)

---

## 1. 什么是多卡训练

### 1.1 基本概念

**多卡训练 = 数据并行 (Data Parallel)**

简单来说：用多张 GPU 同时训练一个模型。

```
单卡训练:
GPU 0: [模型副本] → 处理 batch 1, 2, 3, 4, 5, 6, 7, 8
       时间: ████████

多卡训练 (8卡):
GPU 0: [模型副本] → 处理 batch 1
GPU 1: [模型副本] → 处理 batch 2
GPU 2: [模型副本] → 处理 batch 3
...
GPU 7: [模型副本] → 处理 batch 8
       时间: █
       速度提升: 8x (理论)
```

### 1.2 关键术语

| 术语 | 英文 | 解释 | 示例 |
|-----|------|------|------|
| **进程组** | Process Group | 所有参与训练的进程 | 8 卡 = 8 个进程 |
| **世界大小** | World Size | 总共有多少个进程 | 8 张卡 → world_size=8 |
| **全局排名** | Rank | 当前进程的全局编号 | 0, 1, 2, ..., 7 |
| **本地排名** | Local Rank | 当前节点内的进程编号 | 单节点 = Rank |
| **主进程** | Master Process | Rank 0，负责日志和保存 | ddp_rank == 0 |
| **DDP** | DistributedDataParallel | PyTorch 分布式训练框架 | torch.nn.parallel.DDP |

### 1.3 工作流程

```
初始化阶段:
1. 每个 GPU 加载相同的模型副本
2. 建立进程间通信（NCCL）
3. 同步所有进程

训练阶段:
每个步骤:
  1. 每个 GPU 处理不同的 mini-batch
  2. 各自计算梯度
  3. 通信：所有 GPU 的梯度取平均
  4. 所有 GPU 用相同的梯度更新模型
  5. 模型保持同步

结束阶段:
主进程保存模型检查点
```

---

## 2. 为什么需要多卡训练

### 2.1 训练加速

**实际效果（nanochat 项目）：**

| 配置 | 训练时间（Base, d=12） | 加速比 |
|-----|---------------------|--------|
| 单卡 A100 | ~16 小时 | 1x |
| 8卡 A100 | ~2 小时 | ~8x |

**为什么不是完美的 8x？**
- 通信开销（梯度同步）
- 数据加载开销
- 通常能达到 85-95% 的线性扩展

### 2.2 批次大小限制

**问题：** 单卡显存不够大

```python
# 单卡场景
device_batch_size = 32  # 受限于单卡显存
total_batch_size = 32 * 2048 = 65,536 tokens

# 8卡场景
device_batch_size = 32  # 每卡相同
total_batch_size = 32 * 2048 * 8 = 524,288 tokens
# 更大的批次 → 更稳定的梯度 → 更好的收敛
```

### 2.3 实验效率

**对比：**

```bash
# 单卡训练 - 需要 16 小时
python -m scripts.base_train --depth=12

# 8卡训练 - 只需 2 小时
torchrun --nproc_per_node=8 -m scripts.base_train --depth=12

# 节省时间 = 可以做更多实验！
```

---

## 3. 环境准备

### 3.1 硬件要求

**最低配置：**
- 2+ 张 NVIDIA GPU
- GPU 之间有 NVLink 或 PCIe 连接
- 推荐：相同型号的 GPU

**本项目测试配置：**
- 8x NVIDIA A100 80GB
- NVLink 互联

### 3.2 软件环境

**必需组件：**

```bash
# 1. PyTorch (支持分布式)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 验证 NCCL 后端
python -c "import torch; print(torch.distributed.is_nccl_available())"
# 应输出: True

# 3. 验证多卡可见
python -c "import torch; print(torch.cuda.device_count())"
# 应输出: 8 (如果你有 8 张卡)
```

### 3.3 环境变量

**自动设置（torchrun 会处理）：**

```bash
RANK          # 全局进程编号: 0, 1, 2, ..., 7
LOCAL_RANK    # 本地进程编号: 0, 1, 2, ..., 7
WORLD_SIZE    # 总进程数: 8
MASTER_ADDR   # 主节点地址: localhost (单节点)
MASTER_PORT   # 主节点端口: 29500 (默认)
```

**手动检查：**

```bash
# 查看当前环境
echo $CUDA_VISIBLE_DEVICES  # 可见的 GPU: 0,1,2,3,4,5,6,7
```

---

## 4. 多卡启动方式

### 4.1 torchrun 命令详解

**基本语法：**

```bash
torchrun [torchrun参数] -m [模块名] [脚本参数]
```

**常用参数：**

```bash
torchrun \
    --standalone \              # 单节点模式
    --nproc_per_node=8 \       # 每个节点的进程数（GPU数）
    -m scripts.base_train \    # 要运行的 Python 模块
    --depth=12 \               # 脚本参数
    --device_batch_size=32
```

### 4.2 完整示例

#### 训练 Base 模型

```bash
# 单卡（不使用 torchrun）
python -m scripts.base_train \
    --depth=12 \
    --device_batch_size=8 \
    --run=base_d12_single

# 8卡（使用 torchrun）
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=12 \
    --device_batch_size=32 \
    --run=base_d12_multi
```

#### 训练 Mid 模型

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train \
    --source=base \
    --device_batch_size=16
```

#### 训练 SFT 模型

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft \
    --source=mid \
    --device_batch_size=8
```

#### 训练 RL 模型

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl \
    --source=sft \
    --device_batch_size=8 \
    --num_samples=16
```

### 4.3 参数对比

| 参数 | 单卡 | 多卡 | 说明 |
|-----|------|------|------|
| 启动方式 | `python -m` | `torchrun` | 多卡需要 torchrun |
| device_batch_size | 8 | 32 | 多卡可以更大 |
| 梯度累积 | 多 | 少 | 多卡减少累积步数 |
| 训练时间 | 长 | 短 | 多卡显著加速 |

---

## 5. 代码实现详解

### 5.1 分布式初始化

**位置：** `nanochat/common.py`

```python
def compute_init(device_type="cuda"):
    """基础初始化，包括分布式设置"""
    
    # 1. 获取分布式信息
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # 2. 如果是分布式 + CUDA
    if ddp and device_type == "cuda":
        # 为每个进程指定不同的 GPU
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        
        # 初始化进程组（使用 NCCL 后端）
        dist.init_process_group(
            backend="nccl",      # NVIDIA GPU 专用
            device_id=device
        )
        
        # 同步所有进程
        dist.barrier()
    else:
        device = torch.device(device_type)
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
```

**关键点：**

1. **进程绑定 GPU**：
   ```python
   # Rank 0 → GPU 0
   # Rank 1 → GPU 1
   # ...
   # Rank 7 → GPU 7
   device = torch.device("cuda", ddp_local_rank)
   ```

2. **初始化通信**：
   ```python
   dist.init_process_group(backend="nccl")
   # 建立所有进程之间的通信通道
   ```

3. **同步屏障**：
   ```python
   dist.barrier()
   # 等待所有进程都到达这里
   ```

### 5.2 判断是否启用 DDP

```python
def is_ddp():
    """检查是否在分布式环境中"""
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    """获取分布式信息"""
    if is_ddp():
        # torchrun 会设置这些环境变量
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        # 单卡模式
        return False, 0, 0, 1
```

### 5.3 主进程判断

**为什么需要主进程？**
- 避免重复日志（8 个进程打印 8 遍）
- 避免重复保存（8 个进程保存 8 次）
- 避免重复评估（8 个进程评估 8 次）

```python
# 方法 1: 直接判断
if ddp_rank == 0:
    print("只有主进程打印这条消息")
    save_checkpoint(model)

# 方法 2: 使用 print0 工具
print0("自动判断主进程的打印")
```

**print0 实现：**

```python
def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
```

### 5.4 模型包装

**不需要显式包装！**

nanochat 项目**不使用** `torch.nn.parallel.DistributedDataParallel` 包装模型。

```python
# 传统 DDP 做法（其他项目）:
if ddp:
    model = DDP(model, device_ids=[local_rank])

# nanochat 做法:
# 不包装！直接使用原始模型
# 梯度同步在优化器中处理
```

**为什么？**
- 使用自定义的分布式优化器（DistMuon, DistAdamW）
- 优化器内部处理梯度通信
- 更灵活、更高效

---

## 6. 数据并行机制

### 6.1 数据分配策略

**核心思想：** 每个进程处理不同的数据

```python
# 循环分配（Round-Robin）
for example_idx in range(ddp_rank, len(dataset), ddp_world_size):
    process_example(example_idx)

# 示例：8 卡处理 24 个样本
# Rank 0: 处理 0, 8, 16
# Rank 1: 处理 1, 9, 17
# Rank 2: 处理 2, 10, 18
# ...
# Rank 7: 处理 7, 15, 23
```

### 6.2 Base 训练的数据加载

**位置：** `nanochat/dataloader.py`

```python
def tokenizing_distributed_data_loader(B, T, split, device="cuda"):
    """分布式数据加载器"""
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # 读取 Parquet 文件
    parquet_paths = list_parquet_files()
    
    # 每个 rank 从不同的 row group 开始
    rg_idx = ddp_rank  # 起始索引 = rank
    
    while rg_idx < pf.num_row_groups:
        # 读取数据
        rg = pf.read_row_group(rg_idx)
        batch = rg.column('text').to_pylist()
        
        # ... tokenize ...
        
        # 跳到下一个 row group（间隔 = world_size）
        rg_idx += ddp_world_size  # 0, 8, 16, ...
```

**可视化：**

```
Row Groups (Parquet 文件):
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] ...

8 卡分配:
Rank 0: [0]             [8]              [16] ...
Rank 1:     [1]             [9]              ...
Rank 2:         [2]             [10]         ...
...
Rank 7:                     [7]              ...

结果: 
- 所有 rank 处理不同的数据
- 不重复、不遗漏
- 负载均衡
```

### 6.3 SFT/RL 的数据加载

**对话数据的分配：**

```python
# SFT 训练
conversations = load_conversations()
for i in range(ddp_rank, len(conversations), ddp_world_size):
    conversation = conversations[i]
    # 处理这个对话...

# RL 训练
questions = load_questions()
for idx in range(ddp_rank, len(questions), ddp_world_size):
    question = questions[idx]
    # 采样 16 个答案...
```

---

## 7. 分布式优化器

### 7.1 为什么需要分布式优化器

**问题：** 每个 GPU 计算的梯度不同

```
GPU 0: grad_0 = [-0.1, 0.2, -0.3]
GPU 1: grad_1 = [-0.2, 0.1, -0.4]
...
GPU 7: grad_7 = [-0.15, 0.25, -0.35]

需要: 平均梯度
avg_grad = mean([grad_0, ..., grad_7])
```

**解决方案：** 分布式优化器自动同步梯度

### 7.2 DistAdamW 实现

**位置：** `nanochat/adamw.py`

```python
class DistAdamW(torch.optim.Optimizer):
    """分布式 AdamW"""
    
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 1. Reduce-Scatter: 梯度切片并平均
        for param in params:
            grad = param.grad
            rank_size = grad.shape[0] // world_size
            grad_slice = torch.empty_like(grad[:rank_size])
            
            # 分发梯度的一部分给每个 rank
            dist.reduce_scatter_tensor(
                grad_slice,      # 输出: 当前 rank 的切片
                grad,            # 输入: 完整梯度
                op=dist.ReduceOp.AVG  # 操作: 平均
            )
        
        # 2. 更新参数（每个 rank 只更新自己的切片）
        for param in params:
            p_slice = param[rank * rank_size:(rank + 1) * rank_size]
            # ... AdamW 更新逻辑 ...
            p_slice.add_(update, alpha=-lr)
        
        # 3. All-Gather: 收集所有 rank 的参数
        for param in params:
            dist.all_gather_into_tensor(
                param,      # 输出: 完整参数
                p_slice     # 输入: 当前 rank 的切片
            )
```

**三个关键步骤：**

1. **Reduce-Scatter（梯度平均）**
   ```
   Before:
   GPU 0: [grad_full]
   GPU 1: [grad_full]
   ...
   
   After:
   GPU 0: [avg_grad_slice_0]
   GPU 1: [avg_grad_slice_1]
   ...
   ```

2. **Update（本地更新）**
   ```
   GPU 0: 更新 param_slice_0
   GPU 1: 更新 param_slice_1
   ...
   ```

3. **All-Gather（参数同步）**
   ```
   Before:
   GPU 0: [param_slice_0]
   GPU 1: [param_slice_1]
   ...
   
   After:
   GPU 0: [param_full]
   GPU 1: [param_full]
   ...
   ```

### 7.3 DistMuon 实现

**位置：** `nanochat/muon.py`

```python
class DistMuon(torch.optim.Optimizer):
    """分布式 Muon 优化器"""
    
    @torch.no_grad()
    def step(self):
        # 1. Reduce-Scatter: 梯度平均
        for group in self.param_groups:
            for param in group["params"]:
                # 将 world_size 个梯度收集并平均
                dist.reduce_scatter(
                    rs_output,
                    rs_input,
                    op=dist.ReduceOp.AVG
                )
        
        # 2. 正交化 + 更新（每个 rank 的切片）
        for param in params:
            g = averaged_grad
            # ... Muon 逻辑 ...
            g = zeropower_via_newtonschulz5(g)
            param.add_(g, alpha=-lr)
        
        # 3. All-Gather: 同步参数
        for param in params:
            dist.all_gather(ag_output, ag_input)
```

**与 DistAdamW 的区别：**
- 梯度平均方式相同
- 更新算法不同（正交化 vs 自适应学习率）

### 7.4 通信开销分析

**通信量：**

```python
# 假设模型参数量: 100M
# 梯度大小: 100M * 4 bytes (fp32) = 400MB

# Reduce-Scatter: 400MB / 8 GPUs = 50MB per GPU
# All-Gather: 50MB * 8 = 400MB per GPU

# 总通信: 450MB per GPU per step
```

**通信时间：**

```
NVLink 带宽: ~300 GB/s
通信时间: 450MB / 300GB/s ≈ 1.5ms

前向+反向: ~50ms
通信占比: 1.5 / 50 = 3%

效率: 97% (非常高！)
```

---

## 8. 实战案例

### 8.1 完整训练流程（8 卡）

#### 步骤 1: Base 训练

```bash
# 启动 Base 训练
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=12 \
    --device_batch_size=32 \
    --num_iterations=50000 \
    --run=base_d12_8gpu

# 预期输出（每个 GPU）:
# Step 0: loss=4.234 (8 个进程同时打印)
# Step 1: loss=4.123 (但只有 rank 0 保存日志)
```

#### 步骤 2: 监控训练

```bash
# 查看 GPU 使用
watch -n 1 nvidia-smi

# 应该看到 8 张 GPU 都在运行
# GPU 0: Python (28GB / 80GB)
# GPU 1: Python (28GB / 80GB)
# ...
# GPU 7: Python (28GB / 80GB)
```

#### 步骤 3: Mid 训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train \
    --source=base \
    --device_batch_size=16
```

#### 步骤 4: SFT 训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft \
    --source=mid \
    --device_batch_size=8
```

#### 步骤 5: RL 训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl \
    --source=sft \
    --device_batch_size=8 \
    --num_samples=16
```

### 8.2 单卡 vs 多卡对比

#### 相同的总批次大小

```bash
# 目标: total_batch_size = 524,288 tokens

# ===== 单卡 =====
python -m scripts.base_train \
    --device_batch_size=8 \
    --max_seq_len=2048
# 计算: 8 * 2048 = 16,384 tokens/step
# 需要梯度累积: 524,288 / 16,384 = 32 steps
# 训练时间: ~16 小时

# ===== 8 卡 =====
torchrun --nproc_per_node=8 -m scripts.base_train \
    --device_batch_size=32 \
    --max_seq_len=2048
# 计算: 32 * 2048 * 8 = 524,288 tokens/step
# 需要梯度累积: 1 step (无累积！)
# 训练时间: ~2 小时
```

**对比表格：**

| 指标 | 单卡 | 8卡 | 提升 |
|-----|------|-----|------|
| device_batch_size | 8 | 32 | 4x |
| 梯度累积步数 | 32 | 1 | 32x 更快 |
| 训练时间 | 16h | 2h | 8x |
| 最终 Loss | 3.45 | 3.45 | 相同 |

### 8.3 调试技巧

#### 检查分布式状态

```python
# 在训练脚本中添加
print(f"Rank {ddp_rank}/{ddp_world_size} on GPU {ddp_local_rank}")

# 应该看到:
# Rank 0/8 on GPU 0
# Rank 1/8 on GPU 1
# ...
# Rank 7/8 on GPU 7
```

#### 验证数据不重复

```python
# 在数据加载器中
print(f"Rank {ddp_rank} processing examples: {list(range(ddp_rank, 100, ddp_world_size))}")

# 输出:
# Rank 0: [0, 8, 16, 24, ...]
# Rank 1: [1, 9, 17, 25, ...]
# ...
```

#### 验证梯度同步

```python
# 在训练循环中
if step % 100 == 0 and ddp:
    # 检查所有 rank 的第一个参数是否相同
    param = list(model.parameters())[0]
    local_sum = param.sum().item()
    print(f"Rank {ddp_rank}: param sum = {local_sum}")
    
    # 所有 rank 应该打印相同的值!
```

---

## 9. 常见问题

### 9.1 环境问题

#### Q: `NCCL error: unhandled system error`

**原因：** 网络配置问题

**解决：**
```bash
# 方法 1: 设置网络接口
export NCCL_SOCKET_IFNAME=eth0  # 或 ens3, ib0 等

# 方法 2: 使用 IB 网络（如果有）
export NCCL_IB_DISABLE=0

# 方法 3: 禁用 IB（如果不需要）
export NCCL_IB_DISABLE=1
```

#### Q: `RuntimeError: Address already in use`

**原因：** 端口被占用

**解决：**
```bash
# 更换端口
torchrun --master_port=29501 --nproc_per_node=8 ...

# 或者杀死之前的进程
pkill -9 python
```

### 9.2 性能问题

#### Q: 多卡训练没有加速

**可能原因 1：通信瓶颈**

```bash
# 检查 GPU 连接方式
nvidia-smi topo -m

# 理想情况: NVLink
# GPU0 <-> GPU1: NV12 (12条 NVLink)

# 不好: PCIe
# GPU0 <-> GPU1: PHB (通过 PCIe Host Bridge)
```

**可能原因 2：数据加载慢**

```python
# 增加数据加载线程
loader = DataLoader(
    dataset,
    num_workers=4,  # 增加到 4-8
    pin_memory=True
)
```

**可能原因 3：批次太小**

```python
# 批次太小，通信开销占比高
device_batch_size = 1  # 太小 ❌
device_batch_size = 32  # 合适 ✅
```

#### Q: 显存占用不均衡

**现象：**
```
GPU 0: 60GB / 80GB
GPU 1: 30GB / 80GB
```

**原因：** 数据不均衡或主进程额外操作

**解决：**
```python
# 确保数据均匀分配
assert len(dataset) % world_size == 0

# 主进程避免额外的显存操作
if ddp_rank == 0:
    # 在 CPU 上评估，不在 GPU 上
    model.cpu().eval()
```

### 9.3 训练问题

#### Q: Loss 在不同 rank 不一致

**原因：** 模型没有正确同步

**检查：**
```python
# 1. 确保使用分布式优化器
optimizer = DistAdamW(...)  # ✅
optimizer = torch.optim.AdamW(...)  # ❌ 不会同步

# 2. 确保调用 barrier
if ddp:
    dist.barrier()
```

#### Q: 检查点保存/加载问题

**最佳实践：**

```python
# 保存: 只有主进程保存
if ddp_rank == 0:
    torch.save(model.state_dict(), path)

# 加载: 所有进程都加载
model.load_state_dict(torch.load(path))

# 同步
if ddp:
    dist.barrier()
```

#### Q: 如何从单卡切换到多卡？

**非常简单！**

```bash
# 1. 单卡训练
python -m scripts.base_train --depth=12

# 2. 多卡训练（只需添加 torchrun）
torchrun --nproc_per_node=8 -m scripts.base_train --depth=12

# 代码完全不需要修改！
# nanochat 已经处理好了所有细节
```

---

## 10. 高级话题

### 10.1 多节点训练

**如果你有多台机器（例如 2 台，每台 8 卡）：**

```bash
# 节点 0（主节点）
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    --nproc_per_node=8 \
    -m scripts.base_train

# 节点 1
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    --nproc_per_node=8 \
    -m scripts.base_train
```

### 10.2 混合精度训练

nanochat 已经使用 bfloat16：

```python
# 自动混合精度
autocast_ctx = torch.amp.autocast(
    device_type="cuda",
    dtype=torch.bfloat16  # 训练更稳定
)

with autocast_ctx:
    loss = model(inputs, targets)
```

### 10.3 梯度检查点

对于更大的模型（depth > 20）：

```python
# 在模型中启用
model = torch.compile(model)
# PyTorch 2.0 自动优化内存使用
```

---

## 总结

### 多卡训练的核心要点

1. **启动方式**
   ```bash
   torchrun --nproc_per_node=8 -m scripts.xxx
   ```

2. **数据分配**
   ```python
   for i in range(ddp_rank, len(data), ddp_world_size):
       process(data[i])
   ```

3. **梯度同步**
   ```python
   # 自动由 DistAdamW / DistMuon 处理
   optimizer.step()
   ```

4. **主进程检查**
   ```python
   if ddp_rank == 0:
       save_checkpoint()
   ```

### 为什么 nanochat 的多卡训练这么简单？

1. **自动检测**：代码自动检测是否在分布式环境
2. **透明处理**：分布式优化器自动同步梯度
3. **无需修改**：单卡代码直接支持多卡
4. **工具函数**：`print0()`, `get_dist_info()` 等简化开发

### 下一步

- 实际运行多卡训练
- 监控 GPU 使用和训练速度
- 尝试不同的 batch size 配置
- 理解通信模式和优化策略

---

**参考资源：**
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
