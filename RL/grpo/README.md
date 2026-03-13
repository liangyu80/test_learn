# GRPO — Group Relative Policy Optimization

GRPO（组相对策略优化）是 DeepSeek 提出的强化学习算法，用于大语言模型的对齐优化。它是 PPO 的简化变体，核心创新在于**无需 Critic 模型**，通过组内相对比较来估计优势函数。

## 核心思想

### PPO 的问题

PPO 需要 3 个模型：Actor（策略）、Critic（价值）、Reference（参考），其中 Critic 需要单独训练来估计 V(s)，计算 GAE 优势。

### GRPO 的解决方案

GRPO 去掉 Critic，只需 2 个模型：Actor + Reference。

**关键创新**：对每个 prompt 生成一**组**（Group）回复，通过组内奖励的**相对排名**来计算优势：

```
Â_i = (r_i - mean(r_group)) / std(r_group)
```

直觉：不需要知道"绝对好坏"，只需知道"相对好坏"。

## 算法流程

```
对每个训练迭代:
    1. 采样一批 prompt: {q_1, ..., q_B}
    2. 对每个 q_j, 用策略 π_θ_old 生成 G 条回复: {o_1, ..., o_G}
    3. 用奖励模型对每条回复打分: {r_1, ..., r_G}
    4. 计算组内相对优势: Â_i = (r_i - μ) / σ
    5. 用 PPO-Clip 优化策略:
       L = min(ratio·Â, clip(ratio, 1-ε, 1+ε)·Â) - β·KL(π_θ || π_ref)
    6. 更新策略参数 θ
```

## 与 PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 模型数量 | Actor + Critic + Ref (3个) | Actor + Ref (2个) |
| 优势估计 | GAE (需要 Critic) | 组内归一化 |
| 显存开销 | 高 | 低 (~33% 减少) |
| 奖励粒度 | Token 级 | 序列级 |
| 采样策略 | 1 prompt → 1 回复 | 1 prompt → G 回复 |
| 实现复杂度 | 较复杂 | 较简单 |
| 典型应用 | InstructGPT, ChatGPT | DeepSeek-R1 |

## 项目结构

```
RL/grpo/
├── model.py           # 轻量 GPT 模型 (~1M 参数)
├── grpo_trainer.py    # GRPO 训练器 (组采样、相对优势、Clip 优化)
├── train.py           # 主训练脚本 (SFT + GRPO)
└── README.md          # 本文件
```

## 运行

```bash
cd RL/grpo
python train.py
```

预计运行时间：1-3 分钟 (CPU)

## 数学公式

### 组内相对优势

$$\hat{A}_i = \frac{r_i - \mu_{group}}{\sigma_{group}}$$

### GRPO 目标函数

$$L_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r_t(\theta) \hat{A}_i, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

### KL 散度

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{\pi_{ref}}\left[\frac{\pi_{ref}}{\pi_\theta} \log \frac{\pi_{ref}}{\pi_\theta}\right]$$

## 参考论文

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (Shao et al., 2024)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025)
