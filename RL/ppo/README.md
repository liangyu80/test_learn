# PPO-RLHF: 用 PPO 训练大语言模型

本项目从零实现 PPO (Proximal Policy Optimization) 算法用于 LLM 的强化学习对齐 (RLHF)。所有代码均可在 Mac CPU 上运行，使用 ~1M 参数的自定义小型 GPT 模型。

## 目录

- [RLHF 背景](#rlhf-背景)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [架构详解](#架构详解)
- [PPO 算法详解](#ppo-算法详解)
- [损失函数全解析](#损失函数全解析)
- [关键超参数指南](#关键超参数指南)

---

## RLHF 背景

RLHF (Reinforcement Learning from Human Feedback) 是将人类偏好融入 LLM 的核心技术，分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                      RLHF 三阶段                                │
│                                                                 │
│  阶段 1: SFT (监督微调)                                         │
│  ──────────────────                                             │
│  在高质量 (prompt, response) 数据上微调预训练模型                 │
│  目标: 让模型学会基本的指令遵循能力                              │
│                                                                 │
│  阶段 2: 奖励模型训练                                            │
│  ──────────────────                                             │
│  收集人类偏好数据: (prompt, response_好, response_差)             │
│  训练奖励模型学会区分好回复和差回复                               │
│                                                                 │
│  阶段 3: PPO 优化   ← 本项目实现                                │
│  ──────────────────                                             │
│  用 PPO 算法最大化奖励模型的分数                                 │
│  同时通过 KL 惩罚防止偏离 SFT 模型太远                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
RL/ppo/
├── README.md          # 本文件
├── model.py           # GPT 模型定义 (Actor + Critic + Reward Model)
├── ppo_trainer.py     # PPO 训练器 (GAE, Clipped Objective, KL Penalty)
└── train.py           # 主训练脚本 (SFT + PPO 完整流程)
```

---

## 快速开始

```bash
# 安装依赖
pip install torch

# 运行完整训练 (SFT + PPO)
cd RL/ppo
python train.py

# 单独测试模型
python model.py
```

预计运行时间：2-5 分钟 (CPU)

---

## 架构详解

### 模型组件

| 组件 | 类名 | 参数量 | 作用 |
|------|------|--------|------|
| 策略模型 (Actor) | `GPTLanguageModel` | ~800K | 生成回复，输出词表上的概率分布 |
| 价值模型 (Critic) | `GPTValueModel` | ~800K | 估计状态价值 V(s)，用于计算优势 |
| 参考模型 (Ref) | 冻结的 Actor 副本 | ~800K | KL 约束的参考点，不更新 |
| 奖励模型 (Reward) | `GPTRewardModel` | ~800K | 评估回复质量（本项目用规则替代） |

### GPT 模型架构

```
输入 tokens → [Token Embedding + Position Embedding]
                         ↓
              ┌─── Transformer Block ×4 ───┐
              │  LayerNorm → CausalAttn     │
              │     + Residual              │
              │  LayerNorm → FFN (GELU)     │
              │     + Residual              │
              └─────────────────────────────┘
                         ↓
                    Final LayerNorm
                         ↓
                 LM Head (→ vocab_size)    ← Actor
                 Value Head (→ 1)          ← Critic
```

### RL 术语与 LLM 的对应

| RL 概念 | LLM 对应 |
|---------|----------|
| 环境 (Environment) | prompt + 已生成的 token |
| 状态 (State) | 当前序列 [x₁, ..., xₜ] |
| 动作 (Action) | 下一个 token xₜ₊₁ |
| 策略 (Policy) | π_θ(xₜ₊₁ \| x₁..xₜ) = softmax(logits) |
| 奖励 (Reward) | 奖励模型的打分（通常序列级） |
| 价值 (Value) | V(sₜ) = E[未来累计奖励 \| sₜ] |
| Episode | 一次完整的 prompt → response 生成 |

---

## PPO 算法详解

### 1. 经验收集 (Rollout)

```python
# 伪代码
for each prompt:
    response = policy.generate(prompt)           # π_θ_old 生成回复
    log_probs = policy.log_prob(response)         # 记录旧策略概率
    ref_log_probs = ref_policy.log_prob(response) # 参考策略概率
    reward = reward_model(prompt + response)      # 奖励打分
    values = value_model(prompt + response)       # 价值估计
```

### 2. KL 惩罚奖励

```
r_final(t) = r_reward(t) - β · KL_t

KL_t ≈ log π_θ(aₜ|sₜ) - log π_ref(aₜ|sₜ)
```

β 的自适应调整（InstructGPT 方式）：
- KL > 1.5 × target → β × 1.5（加大惩罚）
- KL < target / 1.5 → β × 0.67（放松惩罚）

### 3. GAE 优势估计

```
δₜ = rₜ + γ · V(sₜ₊₁) - V(sₜ)        # TD 误差
Aₜ = δₜ + (γλ) · δₜ₊₁ + (γλ)² · δₜ₊₂ + ...  # GAE

递推: Aₜ = δₜ + γλ · Aₜ₊₁  (从后往前计算)
```

### 4. PPO-Clip 优化

```
rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)    # 重要性采样比率

L_CLIP = E[min(rₜ · Aₜ, clip(rₜ, 1-ε, 1+ε) · Aₜ)]

L_total = -L_CLIP + c₁ · L_value
```

---

## 损失函数全解析

### 1. SFT 预训练损失

标准交叉熵（Next Token Prediction）：

```
L_SFT = -1/T × Σₜ log P(xₜ₊₁ | x₁, ..., xₜ)
```

### 2. PPO 策略损失 (Policy Loss)

```
L_policy = -E[min(rₜ · Aₜ, clip(rₜ, 1-ε, 1+ε) · Aₜ)]
```

Clip 的作用图示：

```
        目标函数值
        ^
        |         /
        |        / (unclipped)
   1+ε  |-------/-------
        |      /|
   1.0  |-----/-|-------
        |    /  |
   1-ε  |---/--|-------
        |  /   |
        +--+---+------→ ratio rₜ(θ)
         1-ε  1+ε

当 Aₜ > 0: 裁剪上界，防止过度鼓励
当 Aₜ < 0: 裁剪下界，防止过度惩罚
```

### 3. 价值损失 (Value Loss)

```
L_value = E[(V_ψ(sₜ) - returnsₜ)²]

其中 returnsₜ = Aₜ + V_old(sₜ)  (GAE 计算的回报)
```

### 4. KL 惩罚

```
KL(π_θ || π_ref) ≈ E_{a~π_θ}[log π_θ(a|s) - log π_ref(a|s)]
```

逐 token 计算，作为奖励的惩罚项。

### 5. 奖励模型损失（参考）

Bradley-Terry 偏好损失（本项目未训练奖励模型，但 `model.py` 中有架构）：

```
L_reward = -log σ(r(x, y_w) - r(x, y_l))

y_w = 人类偏好的回复
y_l = 人类不偏好的回复
```

---

## 关键超参数指南

| 参数 | 默认值 | 作用 | 调参建议 |
|------|--------|------|----------|
| `clip_eps` | 0.2 | PPO 裁剪范围 | 0.1-0.3，越小越保守 |
| `gamma` | 1.0 | 折扣因子 | RLHF 中通常为 1.0 |
| `lam` (GAE λ) | 0.95 | 偏差-方差权衡 | 0.9-0.99 |
| `kl_coef` (β) | 0.1 | KL 惩罚强度 | 太大→策略不变，太小→reward hacking |
| `target_kl` | 0.02 | 自适应 KL 目标 | InstructGPT 建议 ~6，本项目因模型小用 0.02 |
| `ppo_epochs` | 4 | 每批数据优化轮数 | 1-10，越多数据利用率越高 |
| `lr` | 1e-5 | 学习率 | RLHF 阶段应比 SFT 小 5-10 倍 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 | 0.5-1.0 |
| `vf_coef` | 0.5 | 价值损失权重 | 0.5-1.0 |
| `temperature` | 0.7 | 采样温度 | 0.6-1.0 |

---

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| Python | ≥ 3.8 | 运行环境 |
| PyTorch | ≥ 1.12 | 深度学习框架 |

无需 GPU，无需下载预训练模型，~1M 参数可在任何 Mac 上运行。
