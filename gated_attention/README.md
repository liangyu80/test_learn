# 门控注意力 Transformer (Gated Attention Transformer)

## 概述

本项目实现了**门控注意力 (Gated Attention)** 机制的教学演示，对比三种注意力变体：

| 模式 | 描述 | 特点 |
|------|------|------|
| **GAU** | Gated Attention Unit | 将 Attention 和 FFN 合二为一，用 relu² 实现稀疏注意力 |
| **Sigmoid-Gated** | Sigmoid 门控多头注意力 | 在标准 SDPA 上添加可学习的 sigmoid 门控 |
| **Standard** | 标准多头注意力 | 基线对比 (无门控) |

## 核心思想

### 为什么需要门控？

标准 Transformer 的注意力机制有一个问题：**所有注意力权重之和必须为 1**（softmax 约束）。这导致：

1. **Attention Sink**: 模型被迫将部分注意力分配给无意义的 token（如 `[BOS]`），浪费了注意力容量
2. **缺乏选择性**: 即使某些上下文不重要，注意力也必须"关注"它们
3. **信息过载**: FFN 层收到的是所有注意力输出的混合，无法有选择地过滤

门控注意力通过添加一个**可学习的门控信号**来解决这些问题：

```
标准注意力:  y = softmax(QK^T/√d) · V
门控注意力:  y = gate(x) ⊙ softmax(QK^T/√d) · V
```

`gate ≈ 0` 的位置，注意力输出被**抑制**；`gate ≈ 1` 的位置，注意力输出**保留**。

### GAU (Gated Attention Unit)

来源论文：*"Transformer Quality in Linear Time"* (Hua et al., 2022, Google)

GAU 的核心创新是将 Attention 和 FFN **合并为一个门控模块**：

```
U = SiLU(X·W_U)          ← 门控向量 (类似 FFN 的 up projection)
V = SiLU(X·W_V)          ← 值向量
Q = U·W_Q, K = V·W_K     ← 低维 query/key
A = relu²(Q·K^T/√d)      ← 稀疏注意力 (relu² 代替 softmax)
O = (U ⊙ A·V) · W_O      ← 门控输出 (类似 FFN 的 down projection)
```

关键设计：
- **relu²** 代替 softmax：自带稀疏性（负值变为 0），计算更高效
- **U ⊙ A·V**：门控 U 控制注意力输出的信息流
- **低维 QK**：注意力头的维度远小于扩展维度，节省计算

### Sigmoid-Gated Attention

在标准多头注意力上添加独立的 sigmoid 门控：

```
head_i = softmax(Q_i K_i^T / √d) · V_i    ← 标准注意力
g_i = σ(X · W_g_i)                          ← 门控信号
gated_head_i = g_i ⊙ head_i                 ← 门控过滤
```

每个注意力头有自己的门控，可以学习：
- 哪些头在当前输入下是重要的
- 哪些位置的注意力输出应该被抑制

## 项目结构

```
gated_attention/
├── README.md       # 本文档
├── model.py        # 模型实现 (GAU, SigmoidGated, Standard)
└── train.py        # 训练演示 + 对比实验
```

## 运行方式

```bash
# 安装依赖
pip install torch

# 模型结构测试
python gated_attention/model.py

# 完整训练演示 (对比三种模式)
cd gated_attention && python train.py
```

## 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 256 | 字符级词表大小 |
| `d_model` | 128 | 隐藏层维度 |
| `n_heads` | 4 | 注意力头数 (MHA 模式) |
| `d_head` | 32 | GAU 注意力头维度 |
| `n_layers` | 4 | Transformer 层数 |
| `max_seq_len` | 128 | 最大序列长度 |
| `expansion` | 2 | GAU 扩展倍数 |

## 参考论文

1. **GAU**: Hua et al., "Transformer Quality in Linear Time", 2022
2. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
3. **GLU Variants**: Shazeer, "GLU Variants Improve Transformer", 2020

## 与其他项目的关系

| 项目 | 主题 | 关联 |
|------|------|------|
| `speculative_vs_multitoken/` | 推理加速 | 门控注意力可减少推理计算量 |
| `RL/ppo/` | PPO-RLHF | 门控注意力可用于 RLHF 的 Actor 模型 |
| `RL/grpo/` | GRPO | 门控注意力可用于 GRPO 的策略模型 |
| `titans/` | 长期记忆 | Titans 的 MAC 也使用门控机制 |
