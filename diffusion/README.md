# 扩散模型 (Diffusion Model) —— DDPM & DDIM 从零实现

## 概述

本项目从零实现了两种核心的扩散模型算法:
- **DDPM** (Denoising Diffusion Probabilistic Model) —— 基础扩散模型
- **DDIM** (Denoising Diffusion Implicit Model) —— 加速采样方法

并提供了与 VAE/GAN 的横向对比实验。

## 核心原理

### 扩散模型的两个过程

```
前向过程 (加噪, 固定):
x_0 ──→ x_1 ──→ x_2 ──→ ... ──→ x_T (纯噪声)
    +ε₁     +ε₂     +ε₃            +ε_T

反向过程 (去噪, 需要学习):
x_T ──→ x_{T-1} ──→ ... ──→ x_1 ──→ x_0 (生成样本)
   UNet      UNet           UNet     UNet
```

### DDPM 数学框架

**前向过程** (封闭解):
```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
```

**训练目标** (简化):
```
L = E_{t,ε} [‖ε - ε_θ(x_t, t)‖²]
```
即让网络预测每步添加的噪声。

### DDIM 加速采样

DDIM 的核心洞察: 反向过程可以在任意时间步子序列上运行!

```
DDPM: x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0  (T 步)
DDIM: x_T → x_{T/S·(S-1)} → ... → x_{T/S} → x_0   (S 步, S << T)
```

## DDPM vs DDIM 对比

| 特性 | DDPM | DDIM |
|------|------|------|
| 采样步数 | T (如 1000) | S << T (如 50) |
| 采样速度 | 慢 | 快 10-100x |
| 训练方式 | 标准 | 无需重新训练! |
| 随机性 | 有 (马尔可夫) | 可控 (η 参数) |
| 可复现性 | 不可 | 可 (η=0 时) |

## 三种生成模型横向对比

| 维度 | Diffusion | VAE | GAN |
|------|-----------|-----|-----|
| 训练稳定性 | ✅ 最稳定 | ✅ 稳定 | ❌ 不稳定 |
| 生成质量 | ✅ 最高 | ❌ 模糊 | ✅ 清晰 |
| 采样速度 | ❌ 最慢 | ✅ 最快 | ✅ 快 |
| 模式崩塌 | ✅ 无 | ✅ 无 | ❌ 有 |
| 代表应用 | Stable Diffusion | β-VAE | StyleGAN |

## 文件结构

```
diffusion/
├── README.md     # 本文件
├── ddpm.py       # DDPM 实现 (噪声调度 + 噪声预测网络 + 采样)
├── ddim.py       # DDIM 加速采样器 (复用 DDPM 模型)
└── train.py      # 训练脚本 + 对比实验 (β调度 + DDPM vs DDIM + vs VAE/GAN)
```

## 运行方式

```bash
# 安装依赖
pip install torch

# 运行 DDPM 演示
python diffusion/ddpm.py

# 运行 DDIM 加速采样演示
cd diffusion && python ddim.py

# 运行完整对比实验 (含 VAE/GAN 横向对比)
cd diffusion && python train.py
```

## 实现亮点

### 噪声调度 (Noise Schedule)
- **Linear**: β 从 1e-4 线性增长到 0.02
- **Cosine**: 后期保留更多信号，生成质量更好 (Improved DDPM)

### 噪声预测网络
- 正弦时间步嵌入 (与 Transformer 位置编码相同原理)
- 带时间条件的残差块
- 梯度裁剪稳定训练

### DDIM 采样技巧
- η=0: 完全确定性采样，可复现
- η=1: 等价于 DDPM 随机采样
- 均匀子序列选取时间步
