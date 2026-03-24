# Mamba vs Transformer vs 混合模型

从零实现 **Mamba (选择性状态空间模型)**、**标准 Transformer**，以及参考 **Jamba / Qwen3.5 / Zamba** 的 **Mamba+Transformer 混合架构**，并进行全面对比。

## 核心概念

### 状态空间模型 (SSM) 基础

```
连续形式:
  h'(t) = A·h(t) + B·x(t)     ← 状态转移方程
  y(t)  = C·h(t)               ← 观测方程

离散化 (Zero-Order Hold):
  h_t = Ā·h_{t-1} + B̄·x_t    ← 循环形式 (推理用)
  y_t = C·h_t
```

### Mamba 的创新: 选择性机制 (Selection Mechanism)

传统 SSM (如 S4) 的参数 A, B, C 是**固定的**，Mamba 让它们**随输入变化**：

```
Δ = softplus(Linear(x))   ← 输入决定 "更新步长"
B = Linear(x)              ← 输入决定 "写入什么"
C = Linear(x)              ← 输入决定 "读取什么"

Δ 大 → 状态更新幅度大 (记住当前输入)
Δ 小 → 状态几乎不变   (忽略当前输入)
```

### Mamba Block 结构

```
input ─→ [Linear 2×expand] ──┬── Conv1D → SiLU → SSM ──→ × ──→ [Linear project] → output
                              └── SiLU ─────────────────┘ (门控)
```

## 三种混合策略

### 1. Jamba 风格 (AI21, 稀疏注意力)

```
[Mamba] → [Mamba] → [Mamba] → [Attn] → [Mamba] → [Mamba] → [Mamba] → [Attn]
```

- 每 N 层 (默认 4) 插入一层 Attention
- 大部分计算走 Mamba (高效)，少数 Attention 层保证精确回忆
- KV cache 极小 (只有 1-2 层 Attention)

### 2. Qwen/Alternate 风格 (交替排列)

```
[Mamba] → [Attn] → [Mamba] → [Attn] → [Mamba] → [Attn] → [Mamba] → [Attn]
```

- Mamba 和 Attention 等比例交替
- 每一步都融合两种能力
- 参考 Qwen3.5 的做法

### 3. Zamba 风格 (共享注意力)

```
[Mamba] → [Mamba] → [SharedAttn] → [Mamba] → [Mamba] → [SharedAttn] → ...
```

- 核心假设: "One attention layer is all you need"
- 单个 Attention 模块在多个位置共享权重
- 参数效率最高

## 架构对比

| 维度 | Transformer | Mamba | 混合模型 |
|------|-------------|-------|----------|
| 序列建模 | Self-Attention | Selective SSM | 两者结合 |
| 训练复杂度 | O(N²·d) | O(N·d·n) | 介于两者之间 |
| 推理空间 | O(N) KV cache | O(1) 状态 | 少量 KV + 状态 |
| 位置编码 | 需要 | 不需要 | 可选 |
| 精确回忆 | 强 (直接 attend) | 弱 (压缩到状态) | 中-强 |
| 长序列效率 | 差 (二次方) | 好 (线性) | 好 |

## 文件说明

| 文件 | 内容 |
|------|------|
| `mamba.py` | Mamba S6 实现: 选择性扫描、MambaBlock、MambaLM |
| `transformer.py` | 标准 Transformer: 因果注意力、FFN、TransformerLM |
| `hybrid.py` | 三种混合策略: Jamba / Alternate / Zamba |
| `compare.py` | 统一对比: 参数量、收敛速度、吞吐量、能力测试 |
| `rnn_transformer_mamba.py` | RNN/LSTM/Transformer/Mamba 四代序列模型对比 |
| `linear_attention.py` | Linear Attention 方法汇总: Linear Attn / RetNet / RWKV / GLA vs Mamba |
| `qwen_mamba_hybrid.py` | 仿 Qwen3 风格 Transformer+Mamba 混合模型 (RoPE+GQA+SwiGLU) |

## 运行

```bash
# 单独运行各模型
python mamba.py
python transformer.py
python hybrid.py

# 运行完整对比实验
python compare.py

# RNN vs Transformer vs Mamba 四代对比
python rnn_transformer_mamba.py

# Linear Attention 方法对比
python linear_attention.py

# Qwen-Mamba 混合模型演示
python qwen_mamba_hybrid.py
```

## 实际应用

| 模型 | 架构 | 规模 | 特点 |
|------|------|------|------|
| **Jamba** (AI21) | Mamba + 稀疏 Attention + MoE | 398B (94B active) | 256K 上下文, 4GB KV cache |
| **Qwen3.5** (阿里) | Mamba + Transformer 交替 | 多种规模 | 高效长上下文推理 |
| **Zamba** (Zyphra) | Mamba + 共享 Attention | 7B | 7B 规模最快推理 |
| **Granite 4** (IBM) | Mamba + Transformer 混合 | 多种规模 | 企业级部署 |

## 参考

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) (AI21, 2024)
- [Zamba: A Compact 7B SSM Hybrid Model](https://arxiv.org/abs/2405.16712) (Zyphra, 2024)
- [Mamba-2: State Space Duality](https://arxiv.org/abs/2405.21060) (Dao & Gu, 2024)
