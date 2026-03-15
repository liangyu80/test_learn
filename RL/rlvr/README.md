# RLVR — Reinforcement Learning with Verifiable Rewards

## 核心思想

RLVR 是一种利用**可验证奖励**进行强化学习的方法。与传统 RLHF 使用学习的奖励模型不同，RLVR 直接通过**程序化验证**（如检查数学答案、运行代码测试）来判断模型输出的好坏。

```
传统 RLHF:
    模型输出 → 奖励模型(学习的) → 连续奖励分数 → 策略优化
                   ↑
              可能不准确, 可被 hack

RLVR:
    模型输出 → 验证器(规则/程序) → 二值奖励 {0, 1} → 策略优化
                   ↑
              精确无误, 无法被 hack
```

## 算法流程

```
┌──────────────────────────────────────────────────────────┐
│                    RLVR Training Loop                     │
│                                                          │
│  输入: 问题集 D = {(q_i, a*_i)}  (问题 + 标准答案)       │
│                                                          │
│  for each iteration:                                     │
│    1. 采样一批问题 {q_1, ..., q_B}                        │
│    2. 对每个问题生成 G 条回答:                             │
│       {o_{i,1}, ..., o_{i,G}} ~ π_θ(·|q_i)              │
│    3. 验证每条回答:                                       │
│       r_{i,j} = verify(o_{i,j}, a*_i) ∈ {0, 1}          │
│    4. 计算组内相对优势:                                    │
│       Â_{i,j} = (r_{i,j} - μ_i) / (σ_i + ε)            │
│    5. PPO-Clip 优化 + KL 约束:                           │
│       L = L_clip(Â) + β · D_KL(π_θ || π_ref)            │
│  end for                                                 │
└──────────────────────────────────────────────────────────┘
```

## 与 GRPO/PPO 的对比

| 维度 | PPO | GRPO | RLVR |
|------|-----|------|------|
| **奖励来源** | 学习的奖励模型 | 学习的奖励模型 | 可验证的规则/程序 |
| **奖励类型** | 连续值 | 连续值 | 二值 {0, 1} |
| **模型数量** | 3 (Actor+Critic+Ref) | 2 (Actor+Ref) | 2 (Actor+Ref) |
| **需要 RM 训练** | 是 | 是 | 否 |
| **Reward Hacking** | 容易发生 | 容易发生 | 不可能 |
| **适用场景** | 通用对齐 | 通用对齐 | 数学/代码/推理 |
| **数据可扩展性** | 受限于人工标注 | 受限于人工标注 | 可无限自动生成 |
| **奖励噪声** | 高 (RM 不完美) | 高 (RM 不完美) | 零 (规则精确) |

## RLVR 的关键优势

### 1. 零奖励噪声
```
RLHF 奖励模型: "这个数学解答看起来有 0.73 分的正确性" (主观、有噪声)
RLVR 验证器:   "答案是 42, 模型输出 42, 正确!" (客观、精确)
```

### 2. 无法 Reward Hacking
```
RLHF: 模型可能学会"看起来正确"的回答 (骗过奖励模型)
RLVR: 答案要么对要么错, 没有空间可以 hack
```

### 3. 无限可扩展的训练数据
```
RLHF: 需要人工标注偏好数据 (昂贵、有限)
RLVR: 可以程序化生成无限多的 (问题, 答案) 对
      例: 随机生成数学题 → 用符号计算得到标准答案
```

## 二值奖励下的组采样

RLVR 使用二值奖励 {0, 1}，这对组采样有特殊影响:

```
混合组 (有效): r = [1, 0, 1, 0]  → 有正负优势, 可以学习
全对组 (无效): r = [1, 1, 1, 1]  → 所有优势=0, 无学习信号
全错组 (无效): r = [0, 0, 0, 0]  → 所有优势=0, 无学习信号
```

**混合组比例**是 RLVR 训练中的关键监控指标:
- 如果准确率 p 太高 (接近 1): 几乎全是全对组, 学不动
- 如果准确率 p 太低 (接近 0): 几乎全是全错组, 也学不动
- 理想范围: p ∈ [0.2, 0.8], 混合组比例最高

| 准确率 p | G=4 混合组比例 | G=8 混合组比例 |
|----------|---------------|---------------|
| 0.1 | 34% | 57% |
| 0.3 | 76% | 94% |
| 0.5 | 88% | 99% |
| 0.7 | 76% | 94% |
| 0.9 | 34% | 57% |

## 数学公式

### 目标函数

$$L_{\text{RLVR}}(\theta) = \mathbb{E}_{q \sim P, \{o_i\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r_t(\theta) \hat{A}_i, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) \right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

### 可验证优势

$$\hat{A}_i = \frac{\text{verify}(o_i, a^*) - \mu_{\text{group}}}{\sigma_{\text{group}} + \varepsilon}$$

其中 $\text{verify}(o_i, a^*) \in \{0, 1\}$ 是验证器的输出。

## 项目结构

```
rlvr/
├── README.md           # 本文档
├── model.py            # 轻量 GPT 模型 (~1M 参数)
├── rlvr_trainer.py     # RLVR 训练器 (组采样 + 验证 + PPO-Clip)
└── train.py            # 主训练脚本 (SFT + RLVR)
```

## 快速开始

```bash
cd RL/rlvr
python train.py
```

预计运行时间: 1-3 分钟 (CPU)

## 实际应用场景

| 场景 | 验证器类型 | 示例 |
|------|-----------|------|
| **数学推理** | 精确匹配 / 符号等价 | GSM8K, MATH, Olympiad |
| **代码生成** | 单元测试执行 | HumanEval, MBPP, SWE-bench |
| **逻辑推理** | 规则检查 | ARC, LogiQA |
| **事实问答** | 知识库查询 | TriviaQA, NaturalQuestions |
| **翻译** | BLEU/精确匹配 | WMT (阈值化) |

## 关键论文

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (2025)
  - 在大规模模型上使用 RLVR (GRPO + 可验证奖励) 训练推理能力
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al., 2023)
  - 过程奖励模型 (PRM) 在数学推理中的应用
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
  - 使用验证器解决数学文字题

## 超参数参考

| 参数 | 本项目 | 推荐范围 | 说明 |
|------|--------|---------|------|
| group_size | 4 | 4-64 | 组大小, 越大混合组越多但采样成本越高 |
| correct_bonus | 1.0 | 1.0 | 正确奖励 (归一化后不影响) |
| incorrect_penalty | 0.0 | 0.0 | 错误惩罚 |
| clip_eps | 0.2 | 0.1-0.3 | PPO clip 范围 |
| kl_coef | 0.04 | 0.01-0.1 | KL 惩罚系数 |
| temperature | 0.8 | 0.6-1.0 | 采样温度 |
| lr | 1e-5 | 1e-6 ~ 5e-5 | 学习率 |
