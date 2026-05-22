# Diffusion LLM — 离散扩散语言模型

## 什么是 Diffusion LLM？

传统大语言模型 (GPT, LLaMA) 是**自回归**的：从左到右一个 token 一个 token 地生成文本。

Diffusion LLM 是**非自回归**的：所有 token 同时生成，通过多轮迭代从噪声中恢复出文本。

```
自回归 LM:  [BOS] → "The" → "cat" → "sat" → "on" → "the" → "mat"
                    ↓         ↓        ↓       ↓       ↓        ↓
              (6步，每步预测1个token)

Diffusion LM: [MASK][MASK][MASK][MASK][MASK][MASK]
                ↓ (第1轮去噪)
              [MASK] "cat" [MASK][MASK] "the" [MASK]
                ↓ (第2轮去噪)
              "The" "cat" "sat" [MASK] "the" "mat"
                ↓ (第3轮去噪)
              "The" "cat" "sat" "on" "the" "mat"
```

## 核心原理

### 前向过程 (加噪)

将干净文本逐步破坏为噪声：

```
t=0: "the cat sat on the mat"     ← 干净文本
t=1: "the [M] sat on the [M]"    ← 开始加噪
t=2: "[M] [M] sat [M] the [M]"
...
t=T: "[M] [M] [M] [M] [M] [M]"  ← 纯噪声
```

### 逆向过程 (去噪)

训练一个神经网络，学习从噪声中恢复干净文本：

```
输入: (噪声文本 x_t, 时间步 t)
输出: 预测的干净文本 x_0
```

### 训练目标

```python
# 1. 随机采样时间步
t = random(1, T)
# 2. 前向加噪
x_t = add_noise(x_0, t)
# 3. 去噪网络预测
x_0_pred = denoiser(x_t, t)
# 4. 计算损失
loss = cross_entropy(x_0_pred, x_0)
```

## 三种实现方法

### 1. 吸收态扩散 (Absorbing Diffusion / MDLM)

- **噪声类型**: [MASK] token
- **前向过程**: 随机将 token 替换为 [MASK]
- **终态**: 全部变成 [MASK]
- **类比**: 类似 BERT MLM，但迭代多步进行生成
- **代表工作**: MDLM (Sahoo et al., 2024), GenBERT

### 2. 多项式扩散 (Multinomial Diffusion / D3PM)

- **噪声类型**: 随机 token (从词表均匀采样)
- **前向过程**: 以一定概率将 token 替换为随机 token
- **终态**: 均匀随机序列
- **类比**: 类似连续扩散中加高斯噪声
- **代表工作**: D3PM (Austin et al., 2021), Argmax Flows

### 3. 嵌入空间扩散 (Embedding Diffusion / Diffusion-LM)

- **噪声类型**: 高斯噪声 (在连续嵌入空间)
- **前向过程**: 对 token 嵌入向量加高斯噪声
- **终态**: 标准高斯分布
- **最终输出**: 对去噪嵌入做最近邻查找 → 回到离散 token
- **代表工作**: Diffusion-LM (Li et al., 2022), CDCD

## Diffusion LLM vs 自回归 LLM

| 维度 | Diffusion LLM | 自回归 LLM (GPT) |
|------|--------------|------------------|
| 生成方式 | 并行 (所有 token 同时) | 顺序 (从左到右) |
| 注意力 | 双向 (看所有位置) | 因果 (只看左侧) |
| Infilling | ✓ 原生支持 | ✗ 需要特殊处理 |
| 困惑度 | 通常较差 | 通常更优 |
| 可控生成 | ✓ 适合约束生成 | 需要引导技巧 |
| 编辑能力 | ✓ 可修改任意位置 | 只能 append |

## 项目结构

```
diffusion_llm/
├── README.md               # 本文件
├── diffusion_llm.py        # 三种 Diffusion LLM 实现
│   ├── AbsorbingDiffusionLM    # 吸收态扩散
│   ├── MultinomialDiffusionLM  # 多项式扩散
│   └── EmbeddingDiffusionLM    # 嵌入空间扩散
└── train.py                # 训练与对比实验
    ├── 实验1: 训练收敛对比
    ├── 实验2: 去噪过程可视化
    ├── 实验3: 生成质量分析
    ├── 实验4: Diffusion vs 自回归对比
    └── 实验5: 三种扩散方法机制对比
```

## 运行

```bash
# 运行模型演示 (前向/逆向过程展示)
python diffusion_llm/diffusion_llm.py

# 运行完整对比实验
python diffusion_llm/train.py
```

## 为什么 Diffusion LLM 重要？

1. **Infilling 和编辑**: 可以填充文本中的任意空白位置，这对代码补全、文档编辑等场景很有价值
2. **可控生成**: 因为是并行生成，可以同时施加多个约束 (长度、格式、关键词等)
3. **理论统一**: 将图像生成的扩散模型框架扩展到 NLP，统一了生成建模方法
4. **潜在加速**: 在特定配置下，去噪步数 T 可以远小于序列长度 L，实现加速

## 当前局限

- 在 perplexity (困惑度) 指标上仍落后于自回归 LM
- 需要多步迭代，推理开销仍然较大
- 对长序列的建模能力有待验证
- 工程优化和 scaling law 研究较少

## 参考文献

- Li et al. "Diffusion-LM Improves Controllable Text Generation" (NeurIPS 2022)
- Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces" (NeurIPS 2021)
- Sahoo et al. "Simple and Effective Masked Diffusion Language Models" (MDLM, 2024)
- Lou et al. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (SEDD, 2024)
- Shi et al. "Simplified and Generalized Masked Diffusion for Discrete Data" (2024)
