# 激活函数全面对比

从 Sigmoid 到 SwiGLU —— 20+ 种激活函数的深度对比与分析。

## 目录

- [概述](#概述)
- [激活函数分类](#激活函数分类)
- [数学公式汇总](#数学公式汇总)
- [大模型中的使用情况](#大模型中的使用情况)
- [SwiGLU 为何成为标准](#swiglu-为何成为标准)
- [运行实验](#运行实验)

---

## 概述

激活函数是神经网络的核心组件，赋予模型非线性拟合能力。从 2012 年 AlexNet 的 ReLU 到 2024 年大模型标配的 SwiGLU，激活函数的演进反映了深度学习的发展脉络：

```
Sigmoid (1990s) → ReLU (2012) → GELU (2018/BERT) → SwiGLU (2022/PaLM) → 事实标准 (2024)
```

## 激活函数分类

### 1. 经典激活函数

| 函数 | 公式 | 值域 | 特点 |
|------|------|------|------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | (0, 1) | 概率输出，梯度消失 |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | 零中心，梯度消失 |
| **ReLU** | max(0, x) | [0, +∞) | 快速，死亡神经元 |

### 2. ReLU 改进

| 函数 | 公式 | 解决的问题 |
|------|------|------------|
| **LeakyReLU** | max(αx, x), α=0.01 | 死亡 ReLU |
| **PReLU** | max(αx, x), α 可学习 | 自适应负区间斜率 |
| **ELU** | x if x>0, else α(eˣ-1) | 零中心 + 平滑 |
| **SELU** | λ·ELU(x) | 自归一化 |

### 3. Transformer 时代

| 函数 | 公式 | 代表模型 |
|------|------|----------|
| **GELU** | x·Φ(x) | BERT, GPT-2/3 |
| **SiLU/Swish** | x·σ(x) | EfficientNet, SwiGLU 组件 |
| **Mish** | x·tanh(softplus(x)) | YOLOv4/v5 |

### 4. GLU 门控变体 (大模型标配)

| 函数 | 门控激活 | 使用模型 |
|------|----------|----------|
| **GLU** | Sigmoid | 原始门控 (2017) |
| **ReGLU** | ReLU | 研究用 |
| **GeGLU** | GELU | Gemma, Gemma-2 |
| **SwiGLU** | SiLU/Swish | LLaMA, PaLM, Mistral, DeepSeek, Qwen, Yi, Phi-3 |

## 数学公式汇总

### 逐元素激活函数

```
Sigmoid:    σ(x) = 1 / (1 + e^(-x))
Tanh:       tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
ReLU:       f(x) = max(0, x)
LeakyReLU:  f(x) = max(αx, x)
ELU:        f(x) = x if x>0, else α(e^x - 1)
GELU:       f(x) = x · Φ(x) = x · 0.5 · (1 + erf(x/√2))
SiLU:       f(x) = x · σ(x)
Mish:       f(x) = x · tanh(ln(1 + e^x))
```

### GLU 门控变体

```
GLU:    f(x) = (xW₁) ⊙ σ(xW₂)          门控用 Sigmoid
ReGLU:  f(x) = (xW₁) ⊙ ReLU(xW₂)      门控用 ReLU
GeGLU:  f(x) = (xW₁) ⊙ GELU(xW₂)      门控用 GELU
SwiGLU: f(x) = (xW₂) ⊙ SiLU(xW₁)      门控用 SiLU/Swish
```

## 大模型中的使用情况

### 演进时间线

```
2017 │ Transformer (原始)     → ReLU      简单高效
2018 │ BERT                   → GELU      平滑，概率解释
2019 │ GPT-2                  → GELU      成为标配
2020 │ GPT-3 / T5             → GELU/ReLU 大规模验证
2020 │ Shazeer 论文            → 提出 SwiGLU/GeGLU
2022 │ PaLM (540B)            → SwiGLU    Google 率先采用
2023 │ LLaMA / Mistral        → SwiGLU    开源生态转向
2024 │ LLaMA-3 / DeepSeek-V3  → SwiGLU    事实标准
```

### 当前主流模型使用情况

| 模型 | 参数量 | 年份 | 机构 | 激活函数 |
|------|--------|------|------|----------|
| BERT | 340M | 2018 | Google | GELU |
| GPT-3 | 175B | 2020 | OpenAI | GELU |
| PaLM | 540B | 2022 | Google | **SwiGLU** |
| LLaMA | 65B | 2023 | Meta | **SwiGLU** |
| LLaMA-2 | 70B | 2023 | Meta | **SwiGLU** |
| Mistral | 7B | 2023 | Mistral AI | **SwiGLU** |
| Qwen-2 | 72B | 2024 | 阿里 | **SwiGLU** |
| Gemma-2 | 27B | 2024 | Google | **GeGLU** |
| DeepSeek-V3 | 671B | 2024 | DeepSeek | **SwiGLU** |
| LLaMA-3 | 405B | 2024 | Meta | **SwiGLU** |
| Phi-3 | 14B | 2024 | Microsoft | **SwiGLU** |

### 使用频率

```
SwiGLU   ████████████████ (13 个模型)
GELU     ██████           (5 个模型)
GeGLU    ██               (2 个模型)
ReLU     ██               (2 个模型)
```

## SwiGLU 为何成为标准

### 1. 更好的 Perplexity

Shazeer (2020) 的实验表明，在控制参数量相同的条件下，SwiGLU 和 GeGLU 在语言建模中取得了最低的 perplexity，优于 ReLU、GELU 和原始 GLU。

### 2. 门控机制的优势

```python
# 标准 FFN (ReLU/GELU)
FFN(x) = W₂ · Activation(W₁ · x)     # 2 个矩阵

# SwiGLU FFN
FFN(x) = W₃ · (SiLU(W₁ · x) ⊙ W₂ · x)  # 3 个矩阵
```

门控机制让模型可以**选择性地传递信息**:
- `W₁` 学习"哪些特征应该被激活" (门控)
- `W₂` 学习"特征应该是什么" (内容)
- 两者相乘 = 只保留"值得保留的特征"

### 3. 参数量调整

为保持与标准 FFN 相同的参数量 (因为多了一个矩阵):
- 标准 FFN 隐藏维度: `4 × d_model`
- SwiGLU 隐藏维度: `4 × d_model × 2/3 ≈ 8/3 × d_model`

例如 LLaMA-7B:
- `d_model = 4096`
- FFN 隐藏维度 = `11008 ≈ 4096 × 8/3`

### 4. SiLU 的自门控特性

SiLU(x) = x · σ(x) 本身就是一种"自门控":
- 大正值: σ(x) ≈ 1 → SiLU(x) ≈ x (完全通过)
- 大负值: σ(x) ≈ 0 → SiLU(x) ≈ 0 (完全抑制)
- 零附近: 平滑过渡

SwiGLU = SiLU (自门控) + GLU (外部门控) = **双重门控**

### 5. 工业验证

- **PaLM (Google, 2022)**: 首个在超大规模 (540B) 上验证 SwiGLU 的模型
- **LLaMA (Meta, 2023)**: 开源后引发了整个生态的跟进
- **此后**: Mistral, Qwen, Yi, DeepSeek 等全部采用

## 其他重要趋势

### Squared ReLU

```python
Squared_ReLU(x) = ReLU(x)²
```

- Primer (2021, Google) 使用
- 更强的稀疏性，某些场景下优于 GELU
- 但数值不稳定，大值会被平方放大

### 硬件友好性

- **ReLU**: 最快 (简单比较+选择)
- **GELU/SiLU**: 现代 GPU/TPU 有优化 kernel，几乎与 ReLU 同速
- **Mish**: 最慢 (tanh + softplus + 乘法)
- **SwiGLU**: 矩阵多 50%，但隐藏维度缩小 1/3，总 FLOPs 相当

## 运行实验

```bash
# 运行激活函数 demo (数值性质)
python activations.py

# 运行完整对比实验 (数值性质 + 梯度流 + 训练 + 速度 + LLM 分析)
python compare.py
```

### 实验内容

1. **数值性质**: 输出分布、稀疏度、梯度均值、死亡率
2. **梯度流**: 5/10/20/50 层网络中的梯度传播
3. **训练对比**: 小型 Transformer LM 用不同激活函数训练
4. **速度对比**: 前向/反向计算时间
5. **LLM 使用调查**: 主流大模型的激活函数选择

## 参考文献

1. Hendrycks & Gimpel. "Gaussian Error Linear Units (GELUs)." 2016.
2. Ramachandran et al. "Searching for Activation Functions." 2017. (SiLU/Swish)
3. Dauphin et al. "Language Modeling with Gated Convolutional Networks." 2017. (GLU)
4. Shazeer. "GLU Variants Improve Transformer." 2020. (SwiGLU/GeGLU/ReGLU)
5. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models." 2023.
6. Misra. "Mish: A Self Regularized Non-Monotonic Activation Function." 2019.
7. Klambauer et al. "Self-Normalizing Neural Networks." 2017. (SELU)
