# 位置编码 (Positional Encoding) 全面对比

## 概述

Transformer 的自注意力机制本身是 **置换不变** (permutation-invariant) 的——打乱输入序列的顺序，输出不会改变。位置编码 (Positional Encoding, PE) 赋予模型感知 token 顺序的能力，是 Transformer 架构中至关重要的组件。

本项目从零实现 **9 种主流位置编码方法**，在统一框架下进行对比实验。

## 实现的位置编码方法

| 编码方法 | 类别 | 核心思想 | 代表模型 |
|:---------|:-----|:---------|:---------|
| **Sinusoidal PE** | 加性 / 绝对 | 固定的正弦余弦函数 | Transformer (原始) |
| **Learned PE** | 加性 / 绝对 | 每个位置一个可训练向量 | GPT-2, BERT |
| **RoPE** | 旋转 / 相对 | 复数旋转 Q/K，内积只依赖相对位置 | LLaMA, Qwen, Gemma |
| **ALiBi** | 偏置 / 相对 | 注意力分数加线性距离惩罚 | BLOOM, MPT |
| **Relative PE** | 偏置 / 相对 | 可学习的相对位置嵌入 | Transformer-XL, T5 |
| **Kerple** | 偏置 / 相对 | 核化的对数距离衰减 | Kerple (Chi et al.) |
| **FIRE** | 偏置 / 相对 | MLP 映射连续归一化相对位置 | FIRE (Li et al.) |
| **CoPE** | 偏置 / 上下文 | 注意力门加权的动态位置 | CoPE (Meta, 2024) |
| **NoPE** | 无 | 不加位置编码 (消融基线) | — |

## 位置编码分类体系

```
位置编码
├── 加性编码 (加到输入嵌入上)
│   ├── 绝对位置
│   │   ├── Sinusoidal PE — 固定，不可学习
│   │   └── Learned PE — 可训练嵌入表
│   └── (无)
├── Q/K 变换 (修改注意力的 Q 或 K)
│   └── RoPE — 旋转，天然具有相对位置性质
├── 注意力偏置 (加到注意力分数上)
│   ├── 固定偏置
│   │   └── ALiBi — 线性距离惩罚
│   ├── 可学习偏置
│   │   ├── Relative PE — 离散嵌入表
│   │   ├── Kerple — 核化参数
│   │   └── FIRE — MLP 映射
│   └── 上下文偏置
│       └── CoPE — 注意力门决定位置
└── 无编码
    └── NoPE — 消融基线
```

## 关键算法详解

### RoPE (旋转位置编码)

RoPE 的核心思想是将 Q/K 向量的每两个维度视为一个二维复数，然后乘以旋转矩阵：

```
q_rot = q * cos(m*θ) + rotate_half(q) * sin(m*θ)
k_rot = k * cos(n*θ) + rotate_half(k) * sin(n*θ)
```

其中 `θ_i = 1/10000^(2i/d)`。这样 `q_rot · k_rot` 的内积只依赖 `(m-n)`，即相对位置。

**优势**: 零额外参数，天然相对位置性质，支持长度外推 (配合 NTK-aware scaling)。

### ALiBi (线性偏置注意力)

ALiBi 不修改 Q/K/V，直接在注意力分数上加偏置：

```
attention = softmax(Q @ K^T / sqrt(d) - m * |i - j|)
```

斜率 `m` 按头的序号几何递减: `m_i = 2^{-8i/H}`。

**优势**: 零额外参数，训练短序列可直接外推到长序列。

### CoPE (上下文位置编码)

CoPE 让位置不再是固定的整数索引，而是由注意力权重动态决定：

```
gates = sigmoid(Q @ K^T / sqrt(d))
context_pos = gates @ [0, 1, 2, ..., n]    # 加权位置
bias = Q @ embed(context_pos)^T
```

**优势**: 对 token 粒度变化 (如 BPE 分词) 更鲁棒。

## 对比实验

### 实验1: 训练收敛速度
在字符级语言模型任务上训练，对比不同 PE 的收敛速度和最终损失。

### 实验2: 推理速度
对比各 PE 方法的额外计算开销 (tokens/sec)。

### 实验3: 长度泛化
短序列 (len=32) 训练 → 长序列 (len=64/96/128) 测试，检验外推能力。

### 实验4: 位置感知任务
- **序列复制**: `[x1, x2, ..., SEP, x1, x2, ...]` — 需要精确位置对应
- **序列反转**: `[x1, x2, ..., SEP, ..., x2, x1]` — 需要更强的位置理解

## 快速开始

```bash
# 运行单模块演示
python positional_encoding.py

# 运行完整对比实验
python compare.py
```

## 依赖

- Python 3.8+
- PyTorch >= 1.10

## 参考文献

- Vaswani et al. (2017). *Attention Is All You Need* — Sinusoidal PE
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding* — RoPE
- Press et al. (2022). *Train Short, Test Long: Attention with Linear Biases* — ALiBi
- Shaw et al. (2018). *Self-Attention with Relative Position Representations* — Relative PE
- Chi et al. (2022). *Kerple: Kernelized Relative Positional Embedding* — Kerple
- Li et al. (2023). *Functional Interpolation for Relative Positions Improves Long Context Transformers* — FIRE
- Golovneva et al. (2024). *Contextual Position Encoding: Learning to Count What's Important* — CoPE
