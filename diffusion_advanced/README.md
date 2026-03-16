# 前沿扩散模型算法 (Advanced Diffusion Models)

## 概述

本项目实现了 4 种最新的扩散模型算法，涵盖了 2023-2025 年最重要的研究方向:

| 算法 | 论文/来源 | 核心贡献 |
|------|-----------|----------|
| Flow Matching | Lipman et al., ICLR 2023 | 学习速度场替代噪声预测 |
| Consistency Model | Song et al., ICML 2023 | 1-2 步高质量生成 |
| Shortcut Model | arXiv 2024 | 单次训练实现 1 步生成 |
| DiT / DyDiT | Peebles & Xie / ICLR 2025 | Transformer 架构 + 动态优化 |

## 算法详解

### 1. Flow Matching (Rectified Flow)

**核心区别 vs DDPM**: 不再预测噪声，而是直接学习从噪声到数据的速度场。

```
DDPM: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε  (复杂的噪声调度)
FM:   x_t = (1-t) · x_0 + t · ε          (简单线性插值!)
```

- 训练目标: `L = ‖v_θ(x_t, t) - (ε - x_0)‖²`
- 采样: ODE 求解 (Euler / 中点法)
- 优势: 代码极简，无需噪声调度超参

### 2. Consistency Model

**核心思想**: ODE 轨迹上任意一点都应映射到同一个 x_0。

```
f_θ(x_t, t) = f_θ(x_t', t')  (自一致性)
```

- 使用 EMA 目标网络 + Pseudo-Huber 损失
- 边界条件: f(x, σ_min) = x (通过 skip connection 保证)
- 课程学习: 离散化步数逐渐增大
- **1-2 步即可生成!**

### 3. Shortcut Model

**核心创新**: 网络额外接受步长 d 作为输入。

```
标准: v_θ(x_t, t)        → 只能小步走
Shortcut: s_θ(x_t, t, d) → 可以跳任意步长!
```

- 阶段 1: Flow Matching 基础训练 (d ≈ 0)
- 阶段 2: 自蒸馏 — 两个小步 = 一个大步
- 步长逐渐翻倍: 1/128 → 1/64 → ... → 1/2 → 1
- **单次训练，无需额外蒸馏!**

### 4. DiT / DyDiT

**DiT (Diffusion Transformer)**:
- 用 Transformer 替代 U-Net (Stable Diffusion 3, Sora 的基础)
- 核心创新: adaLN (自适应 LayerNorm) + adaLN-Zero
- Patch Embedding 将数据 token 化

**DyDiT (Dynamic DiT)**:
- 时间步自适应: 高噪声用少计算，低噪声用全计算
- Token 选择: 只处理重要的 token，跳过不重要的
- FLOPs 降低 51%，速度提升 1.73x

## 发展路线

```
DDPM (2020) ─→ DDIM (2020) ─→ Flow Matching (2022)
     │              │               │
     │              ↓               ↓
     │     Consistency Model    Shortcut Model
     │         (2023)             (2024)
     │
     ↓
 U-Net ─────────→ DiT (2023) ──→ DyDiT (2024)
(SD 1/2)          (SD3, Sora)    (高效 DiT)
```

## 文件结构

```
diffusion_advanced/
├── README.md              # 本文件
├── flow_matching.py       # Flow Matching (Rectified Flow)
├── consistency_model.py   # Consistency Model (CT 训练)
├── shortcut_model.py      # Shortcut Model (自蒸馏 1 步生成)
├── dynamic_dit.py         # DiT + DyDiT (Transformer 架构)
└── train.py               # 统一对比实验
```

## 运行方式

```bash
# 安装依赖
pip install torch

# 单独运行
python diffusion_advanced/flow_matching.py
python diffusion_advanced/consistency_model.py
python diffusion_advanced/shortcut_model.py
python diffusion_advanced/dynamic_dit.py

# 统一对比实验
cd diffusion_advanced && python train.py
```
