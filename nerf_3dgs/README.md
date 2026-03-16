# NeRF vs 3D Gaussian Splatting 对比实验

## 概述

本项目从零实现了两种主流的神经3D场景表示方法:
- **NeRF** (Neural Radiance Fields) —— 隐式神经场景表示
- **3DGS** (3D Gaussian Splatting) —— 显式高斯点云表示

并在相同的合成场景上进行对比实验。

## 核心原理

### NeRF (Neural Radiance Fields)

用 MLP 网络隐式表示整个 3D 场景:

```
F_θ: (x, y, z, θ, φ) → (r, g, b, σ)
```

渲染通过**体渲染** (Volume Rendering) 实现:
```
C(r) = Σ_i T_i · α_i · c_i
T_i = Π_{j<i} (1 - α_j)       ← 累积透射率
α_i = 1 - exp(-σ_i · δ_i)     ← 不透明度
```

关键技术:
- **位置编码**: `γ(p) = [sin(2^k·πp), cos(2^k·πp)]` 帮助学习高频细节
- **分层采样**: 粗网络 + 细网络 (本实现为简化版)
- **视角相关颜色**: 密度只依赖位置，颜色依赖位置+方向

### 3D Gaussian Splatting

用一组 3D 高斯椭球体显式表示场景:

每个高斯: `{μ, Σ, α, color}`
- **μ**: 3D 中心位置
- **Σ = R·S·Sᵀ·Rᵀ**: 协方差矩阵 (由四元数旋转 + 缩放分解)
- **α**: 不透明度
- **color**: RGB 颜色

渲染通过**光栅化** (Rasterization) 实现:
```
1. 3D 高斯 → 投影到 2D (EWA Splatting)
2. 按深度排序
3. 前到后 Alpha 混合
```

## 核心对比

| 维度 | NeRF | 3DGS |
|------|------|------|
| 场景表示 | 隐式 (MLP 权重) | 显式 (高斯点云) |
| 渲染方式 | Volume Rendering | Rasterization |
| 渲染速度 | 慢 (逐光线采样) | 快 (实时可达) |
| 训练速度 | 慢 (MLP 优化) | 快 (直接参数优化) |
| 内存 | 小 (只有 MLP) | 大 (百万高斯) |
| 编辑能力 | 难 (隐式) | 易 (操作点云) |
| 代表应用 | 新视角合成 | 实时渲染/VR |

## 文件结构

```
nerf_3dgs/
├── README.md              # 本文件
├── nerf.py                # NeRF 完整实现
├── gaussian_splatting.py  # 3DGS 完整实现
└── compare.py             # 对比实验脚本
```

## 运行方式

```bash
pip install torch

# 单独运行 NeRF
python nerf_3dgs/nerf.py

# 单独运行 3DGS
cd nerf_3dgs && python gaussian_splatting.py

# 运行对比实验
cd nerf_3dgs && python compare.py
```

## 实现亮点

### NeRF
- 完整的位置编码 (多频率 sin/cos)
- Skip connection (第 4 层)
- 分层采样 (训练时随机扰动)
- 密度和颜色的分离设计 (密度只依赖位置)

### 3DGS
- 四元数参数化旋转 (避免万向锁)
- 协方差分解 Σ = RSSᵀRᵀ (保证半正定)
- EWA Splatting 投影 (透视投影的局部线性化)
- 分参数学习率 (位置、缩放、颜色各不同)
- 前到后 Alpha 混合 + 早停优化
