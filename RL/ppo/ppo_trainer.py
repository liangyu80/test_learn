"""
PPO (Proximal Policy Optimization) 训练器 —— 用于 RLHF

完整实现 PPO 算法用于语言模型的强化学习对齐 (RLHF)。

RLHF 三阶段回顾:
    阶段 1: 预训练 (SFT) — 在高质量数据上监督微调
    阶段 2: 奖励模型训练  — 用人类偏好数据训练奖励模型
    阶段 3: PPO 优化     — 用 PPO 最大化奖励模型的分数 ← 本文件实现的部分

PPO 在 RLHF 中的工作流程:
    ┌──────────────────────────────────────────────────────────────┐
    │                    PPO Training Loop                         │
    │                                                              │
    │  1. 策略模型 π_θ 根据 prompt 生成回复                        │
    │  2. 奖励模型 R_φ 对 (prompt, 回复) 给出奖励 r               │
    │  3. 计算 KL 惩罚: r_final = r - β · KL(π_θ || π_ref)        │
    │  4. 用 GAE 计算优势函数 A_t                                  │
    │  5. PPO Clipped 目标函数优化 π_θ                             │
    │  6. 同时优化价值模型 V_ψ 的价值损失                          │
    │                                                              │
    │  循环直到策略收敛                                            │
    └──────────────────────────────────────────────────────────────┘

关键论文:
    - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    - "Training language models to follow instructions with human feedback" (Ouyang et al., 2022, InstructGPT)
    - "Learning to summarize from human feedback" (Stiennon et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import copy


# ==============================================================================
# PPO 训练配置
# ==============================================================================

@dataclass
class PPOConfig:
    """
    PPO 训练超参数。

    每个参数的设计考量:
    """
    # --- 生成参数 ---
    max_gen_len: int = 32           # 生成回复的最大长度
    temperature: float = 0.7        # 采样温度

    # --- PPO 核心参数 ---
    ppo_epochs: int = 4             # 每批数据上 PPO 迭代的轮数
    # 为什么要多轮迭代:
    #   每次收集一批经验后，PPO 可以在这批数据上多次优化。
    #   这提高了数据利用效率（相比 REINFORCE 只用一次就丢弃）。
    #   Clip 机制保证即使多轮迭代也不会走太远。

    clip_eps: float = 0.2           # PPO clip 范围 ε
    # clip_eps 控制策略更新的幅度:
    #   ratio = π_θ(a|s) / π_θ_old(a|s)
    #   clipped_ratio = clip(ratio, 1-ε, 1+ε)
    #   ε = 0.2 → ratio 被限制在 [0.8, 1.2]
    #   太大: 策略变化过快，训练不稳定
    #   太小: 更新过于保守，收敛慢

    gamma: float = 1.0              # 折扣因子
    # 在 RLHF 中通常设为 1.0 (不折扣):
    #   因为奖励通常只在序列结束时给出一次，
    #   所有 token 对最终奖励的贡献被视为同等重要。

    lam: float = 0.95               # GAE λ 参数
    # GAE (Generalized Advantage Estimation) 的偏差-方差权衡:
    #   λ = 0: A_t = r_t + γV(s_{t+1}) - V(s_t)  (TD(0), 低方差高偏差)
    #   λ = 1: A_t = Σ γ^k r_{t+k} - V(s_t)      (MC,    高方差低偏差)
    #   λ = 0.95: 在两者之间取得良好平衡

    vf_coef: float = 0.5            # 价值损失系数
    # 价值损失在总损失中的权重:
    #   L_total = L_policy + vf_coef * L_value
    #   0.5 是常用值，平衡策略和价值的学习速度

    # --- KL 惩罚参数 ---
    kl_coef: float = 0.1            # KL 散度惩罚系数 β
    # β 控制策略不偏离参考模型太远:
    #   r_final = r_reward - β · KL(π_θ || π_ref)
    #   太大: 策略几乎不变，无法从奖励中学习
    #   太小: 策略可能"过度优化"奖励模型，产生 reward hacking
    #   InstructGPT 使用自适应 β (KL 目标值控制)，这里用固定值简化

    target_kl: Optional[float] = None  # KL 目标值（可选的自适应 KL）
    # 如果设置了 target_kl:
    #   当 KL > 1.5 * target_kl → 增大 β（约束更强）
    #   当 KL < target_kl / 1.5 → 减小 β（放松约束）

    # --- 优化参数 ---
    lr: float = 1e-5                # 学习率（RLHF 阶段通常较小）
    max_grad_norm: float = 0.5      # 梯度裁剪阈值

    # --- 训练参数 ---
    batch_size: int = 8             # 每批 prompt 数
    num_iterations: int = 50        # 总训练迭代数


# ==============================================================================
# 经验缓冲区 (Experience Buffer)
# ==============================================================================

@dataclass
class PPOExperience:
    """
    PPO 训练的一批经验数据。

    在 RLHF 上下文中，一条"经验"包含:
        - 一个 prompt
        - 策略模型根据 prompt 生成的回复
        - 生成过程中每个 token 的 log 概率
        - 奖励模型给出的奖励
        - 价值模型估计的状态价值
        - 通过 GAE 计算的优势估计

    RL 术语与 LLM 的对应关系:
        状态 (state):   已生成的 token 序列 [x_1, ..., x_t]
        动作 (action):  下一个 token x_{t+1}
        策略 (policy):  π_θ(x_{t+1} | x_1, ..., x_t) = softmax(logits)
        奖励 (reward):  通常只在序列结束时给出
        价值 (value):   V(s_t) = E[R | s_t]，期望未来累计奖励
    """
    # 完整序列 (prompt + response), shape = (B, prompt_len + gen_len)
    sequences: torch.Tensor = None

    # prompt 长度（用于区分 prompt 和 response 部分）
    prompt_lens: List[int] = field(default_factory=list)

    # 生成时每个 token 的 log 概率 (旧策略), shape = (B, gen_len)
    old_log_probs: torch.Tensor = None

    # 奖励序列, shape = (B, gen_len)
    # 通常只有最后一个位置有非零值 (序列级奖励)
    rewards: torch.Tensor = None

    # 价值估计, shape = (B, gen_len)
    values: torch.Tensor = None

    # GAE 优势估计, shape = (B, gen_len)
    advantages: torch.Tensor = None

    # GAE 回报 (returns), shape = (B, gen_len)
    returns: torch.Tensor = None

    # 参考模型的 log 概率 (用于 KL 惩罚), shape = (B, gen_len)
    ref_log_probs: torch.Tensor = None


# ==============================================================================
# PPO 训练器
# ==============================================================================

class PPOTrainer:
    """
    PPO-RLHF 训练器。

    核心组件:
        1. 策略模型 (Actor, π_θ):     生成回复
        2. 价值模型 (Critic, V_ψ):    估计状态价值
        3. 参考模型 (Reference, π_ref): 冻结的初始策略，用于 KL 约束
        4. 奖励函数 (Reward):          评估回复质量

    PPO 目标函数:
        L^{CLIP}(θ) = E[ min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t) ]

        其中:
            r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)    (重要性采样比率)
            A_t = 优势函数 (通过 GAE 计算)
            ε = clip_eps

    参数:
        policy_model:  策略模型 (Actor)
        value_model:   价值模型 (Critic)
        reward_fn:     奖励函数 (接受 token 序列，返回标量奖励)
        config:        PPO 配置
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reward_fn,
        config: PPOConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy_model.to(device)
        self.value_model = value_model.to(device)
        self.reward_fn = reward_fn
        self.config = config
        self.device = device

        # ---------------------------------------------------------------
        # 参考模型 (Reference Model):
        # 冻结的策略模型副本，PPO 训练过程中不更新。
        #
        # 作用: 通过 KL 散度惩罚防止策略模型偏离太远。
        # 如果没有 KL 约束，策略可能会"过度优化"奖励模型，
        # 找到奖励模型的漏洞而不是真正提升回复质量 (reward hacking)。
        #
        # 例子: 奖励模型可能对"非常长的回复"给高分，
        # 没有 KL 约束时策略会学会生成冗长但无用的回复。
        # ---------------------------------------------------------------
        self.ref_policy = copy.deepcopy(policy_model).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # 优化器：同时优化策略和价值模型
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_model.parameters()),
            lr=config.lr,
        )

        # 自适应 KL 系数
        self.kl_coef = config.kl_coef

        # 训练统计
        self.train_stats: List[Dict] = []

    # ==================================================================
    # 阶段 1: 收集经验 (Rollout / Experience Collection)
    # ==================================================================

    @torch.no_grad()
    def collect_experience(self, prompts: torch.Tensor) -> PPOExperience:
        """
        收集一批经验：用策略模型生成回复，计算奖励和价值。

        这对应 PPO 的"采样阶段"：
            1. 用当前策略 π_θ_old 与环境交互（生成回复）
            2. 记录所有相关信号（概率、奖励、价值）
            3. 这些数据将在后续的"优化阶段"中使用

        参数:
            prompts: 一批 prompt, shape = (B, prompt_len)

        返回:
            experience: 包含所有经验数据的 PPOExperience 对象
        """
        self.policy.eval()
        self.value_model.eval()

        B = prompts.shape[0]
        prompt_len = prompts.shape[1]

        # ---- 步骤 1: 生成回复 ----
        # 策略模型自回归生成，同时记录每个 token 的 log 概率
        generated = prompts.clone()
        all_log_probs = []

        for t in range(self.config.max_gen_len):
            logits, _ = self.policy(generated)
            # 只取最后一个位置的 logits
            next_logits = logits[:, -1, :] / self.config.temperature

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 记录选中 token 的 log 概率
            log_probs = F.log_softmax(next_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, next_token)  # (B, 1)
            all_log_probs.append(selected_log_probs)

            generated = torch.cat([generated, next_token], dim=-1)

        # 堆叠 log 概率: shape = (B, gen_len)
        old_log_probs = torch.cat(all_log_probs, dim=-1)

        # ---- 步骤 2: 计算参考模型的 log 概率 (用于 KL 惩罚) ----
        ref_log_probs = self._compute_log_probs(
            self.ref_policy, generated, prompt_len
        )

        # ---- 步骤 3: 计算奖励 ----
        rewards = self._compute_rewards(generated, prompt_len, ref_log_probs, old_log_probs)

        # ---- 步骤 4: 计算价值估计 ----
        values = self.value_model(generated)
        # 只取 response 部分的价值: (B, gen_len)
        response_values = values[:, prompt_len - 1 : -1]

        # ---- 步骤 5: 计算 GAE 优势估计 ----
        advantages, returns = self._compute_gae(rewards, response_values)

        experience = PPOExperience(
            sequences=generated,
            prompt_lens=[prompt_len] * B,
            old_log_probs=old_log_probs,
            rewards=rewards,
            values=response_values,
            advantages=advantages,
            returns=returns,
            ref_log_probs=ref_log_probs,
        )

        return experience

    def _compute_log_probs(
        self, model: nn.Module, sequences: torch.Tensor, prompt_len: int
    ) -> torch.Tensor:
        """
        计算给定模型对 response 部分每个 token 的 log 概率。

        参数:
            model:      语言模型
            sequences:  完整序列 (prompt + response)
            prompt_len: prompt 的长度

        返回:
            log_probs: shape = (B, gen_len)
        """
        logits, _ = model(sequences[:, :-1])  # 预测位置 1..T-1
        log_probs = F.log_softmax(logits, dim=-1)

        # 取 response 部分 (从 prompt_len 开始) 对应的实际 token 的 log prob
        response_tokens = sequences[:, prompt_len:]  # (B, gen_len)
        response_logprobs = log_probs[:, prompt_len - 1 :, :]  # (B, gen_len, vocab)

        # gather: 取出实际 token 的 log 概率
        selected = response_logprobs.gather(
            2, response_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return selected

    def _compute_rewards(
        self,
        sequences: torch.Tensor,
        prompt_len: int,
        ref_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算带 KL 惩罚的奖励。

        奖励组成:
            r_final(t) = r_reward(t) - β · KL_t

        其中:
            r_reward(t): 原始奖励（通常只在最后一个 token 给出）
            KL_t:        每个 token 位置的 KL 散度近似
            β:           KL 惩罚系数

        KL 散度的逐 token 近似:
            KL_t ≈ log π_θ(a_t|s_t) - log π_ref(a_t|s_t)

            这是真正 KL 散度 KL(π_θ || π_ref) 的无偏估计。
            完整的 KL 散度:
                KL(π_θ || π_ref) = Σ_x π_θ(x) · log(π_θ(x) / π_ref(x))
            我们用采样近似:
                E_{a~π_θ}[log π_θ(a) - log π_ref(a)] ≈ log π_θ(a_t) - log π_ref(a_t)

        为什么需要 KL 惩罚:
            防止 reward hacking —— 策略找到奖励模型的"漏洞"，生成奖励分数高
            但实际质量差的回复。KL 惩罚确保策略不偏离 SFT 模型太远。

        参数:
            sequences:     完整序列
            prompt_len:    prompt 长度
            ref_log_probs: 参考模型的 log 概率
            old_log_probs: 当前策略的 log 概率

        返回:
            rewards: shape = (B, gen_len)，带 KL 惩罚的奖励
        """
        B, gen_len = old_log_probs.shape

        # 1. 获取原始奖励 (序列级)
        raw_rewards = self.reward_fn(sequences)  # (B,)

        # 2. 构造 token 级奖励
        # 原始奖励只放在最后一个 token，其余为 0
        token_rewards = torch.zeros(B, gen_len, device=self.device)
        token_rewards[:, -1] = raw_rewards

        # 3. 计算每个 token 的 KL 惩罚
        # KL_t = log π_θ(a_t|s_t) - log π_ref(a_t|s_t)
        kl_per_token = old_log_probs - ref_log_probs  # (B, gen_len)

        # 4. 最终奖励 = 原始奖励 - KL 惩罚
        rewards = token_rewards - self.kl_coef * kl_per_token

        return rewards

    # ==================================================================
    # 阶段 2: 计算 GAE 优势估计
    # ==================================================================

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        广义优势估计 (Generalized Advantage Estimation, GAE)。

        核心公式:
            δ_t = r_t + γ · V(s_{t+1}) - V(s_t)     (TD 误差)
            A_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}  (GAE 优势)

        展开形式:
            A_t = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...

        递推计算 (从后往前):
            A_{T-1} = δ_{T-1}
            A_t = δ_t + γλ · A_{t+1}

        GAE 的 λ 参数控制偏差-方差权衡:
            ┌──────────────────────────────────────────────────────┐
            │ λ = 0:  A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)     │
            │         只看一步 TD 误差                              │
            │         ✓ 低方差（只依赖一步随机性）                  │
            │         ✗ 高偏差（严重依赖 V 的准确性）               │
            │                                                      │
            │ λ = 1:  A_t = R_t - V(s_t)  (蒙特卡洛回报 - 基线)    │
            │         看整个轨迹的实际回报                          │
            │         ✓ 低偏差（使用实际回报，无近似）              │
            │         ✗ 高方差（整个轨迹的随机性叠加）              │
            │                                                      │
            │ λ = 0.95: 在两者之间取得实用的平衡                   │
            └──────────────────────────────────────────────────────┘

        为什么优势函数很重要:
            策略梯度: ∇_θ J = E[A_t · ∇_θ log π_θ(a_t|s_t)]
            - A_t > 0: 增大 π_θ(a_t|s_t)（鼓励这个动作）
            - A_t < 0: 减小 π_θ(a_t|s_t)（抑制这个动作）
            - A_t = 0: 不改变（符合预期的动作）

            使用优势而非原始奖励的好处:
            减去基线 V(s_t) 可以大幅降低方差，而不引入偏差。

        参数:
            rewards: token 级奖励, shape = (B, gen_len)
            values:  价值估计, shape = (B, gen_len)

        返回:
            advantages: GAE 优势, shape = (B, gen_len)
            returns:    回报（优势 + 价值）, shape = (B, gen_len)
        """
        B, T = rewards.shape
        gamma = self.config.gamma
        lam = self.config.lam

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        # 从后往前递推计算 GAE
        for t in reversed(range(T)):
            if t == T - 1:
                # 最后一步: 没有下一个状态，V(s_{T}) = 0 (episode 结束)
                next_value = 0.0
            else:
                next_value = values[:, t + 1]

            # TD 误差: δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + gamma * next_value - values[:, t]

            # GAE 递推: A_t = δ_t + γλ · A_{t+1}
            advantages[:, t] = delta + gamma * lam * last_gae
            last_gae = advantages[:, t]

        # 回报 = 优势 + 价值
        # 用于价值函数的训练目标: L_value = (V(s_t) - returns_t)²
        returns = advantages + values

        return advantages, returns

    # ==================================================================
    # 阶段 3: PPO 优化
    # ==================================================================

    def ppo_update(self, experience: PPOExperience) -> Dict[str, float]:
        """
        PPO 优化步骤：在收集的经验上进行多轮优化。

        PPO-Clip 目标函数详解:
        =====================

        L^{CLIP}(θ) = E_t[ min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t) ]

        其中:
            r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
                   = exp(log π_θ - log π_θ_old)

        r_t(θ) 是"重要性采样比率"，衡量新旧策略的差异:
            r_t = 1:   新旧策略对动作 a_t 的概率相同
            r_t > 1:   新策略比旧策略更倾向选择 a_t
            r_t < 1:   新策略比旧策略更不倾向选择 a_t

        Clip 的作用 (核心创新):
        ─────────────────────
        考虑两种情况:

        情况 1: A_t > 0 (好动作，应该鼓励)
            min(r_t · A_t, clip(r_t, 0.8, 1.2) · A_t)
            = min(r_t · A_t, min(r_t, 1.2) · A_t)

            即使 r_t 很大（策略大幅倾向这个动作），
            目标函数的增益被限制在 1.2 · A_t。
            防止一步更新走太远。

        情况 2: A_t < 0 (坏动作，应该抑制)
            min(r_t · A_t, clip(r_t, 0.8, 1.2) · A_t)
            = min(r_t · A_t, max(r_t, 0.8) · A_t)   [注意 A_t < 0 翻转不等式]

            即使策略已经大幅远离这个动作 (r_t << 1)，
            损失也不会超过 0.8 · A_t。
            防止过度惩罚。

        直觉总结:
            Clip 是一个"安全带"，防止策略更新过猛。
            不管优势是正是负，策略的变化幅度都被限制在 [1-ε, 1+ε] 内。

        价值损失 (Value Loss):
        ────────────────────
            L_value = E_t[ (V_ψ(s_t) - returns_t)² ]

            即让价值模型的预测 V_ψ(s_t) 接近实际的回报 returns_t。
            也可以使用 clipped 版本的价值损失（这里使用简单版本）。

        参数:
            experience: 收集的经验数据

        返回:
            stats: 训练统计信息
        """
        self.policy.train()
        self.value_model.train()

        sequences = experience.sequences
        old_log_probs = experience.old_log_probs
        advantages = experience.advantages
        returns = experience.returns
        prompt_len = experience.prompt_lens[0]

        # 优势归一化 (Advantage Normalization):
        # 减均值除标准差，使优势的分布近似标准正态。
        # 这不会改变最优策略（因为只是线性变换），但能:
        #   1. 稳定训练（梯度大小更一致）
        #   2. 减少对奖励尺度的敏感性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 记录统计信息
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0

        for epoch in range(self.config.ppo_epochs):
            # ---- 步骤 1: 前向传播，获取当前策略的 log 概率 ----
            new_log_probs = self._compute_log_probs(
                self.policy, sequences, prompt_len
            )

            # ---- 步骤 2: 计算重要性采样比率 ----
            # r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            # 在 log 空间: log r_t = log π_θ - log π_θ_old
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)  # (B, gen_len)

            # ---- 步骤 3: 计算 PPO-Clip 策略损失 ----
            # 未裁剪的目标: r_t · A_t
            surr1 = ratio * advantages

            # 裁剪后的目标: clip(r_t, 1-ε, 1+ε) · A_t
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
            ) * advantages

            # 取 min → 悲观估计 (pessimistic bound)
            # 然后取负号（因为我们要最大化目标函数，但优化器做的是最小化）
            policy_loss = -torch.min(surr1, surr2).mean()

            # ---- 步骤 4: 计算价值损失 ----
            # L_value = MSE(V(s_t), returns_t)
            new_values = self.value_model(sequences)
            response_values = new_values[:, prompt_len - 1: -1]
            value_loss = F.mse_loss(response_values, returns)

            # ---- 步骤 5: 计算熵奖励 (可选) ----
            # 熵鼓励策略保持一定的随机性，防止过早收敛到确定性策略
            # H(π) = -Σ π(a|s) log π(a|s)
            # 加入负熵作为损失的一部分: L_total -= entropy_coef * H
            logits, _ = self.policy(sequences[:, :-1])
            response_logits = logits[:, prompt_len - 1:, :]
            dist = torch.distributions.Categorical(logits=response_logits)
            entropy = dist.entropy().mean()

            # ---- 步骤 6: 总损失 ----
            # L_total = L_policy + vf_coef · L_value - entropy_coef · H
            loss = policy_loss + self.config.vf_coef * value_loss

            # ---- 步骤 7: 反向传播 + 梯度裁剪 + 更新 ----
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪: 防止梯度爆炸
            # 当梯度范数超过 max_grad_norm 时，等比例缩小所有梯度
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_model.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            # ---- 统计 ----
            with torch.no_grad():
                # KL 散度近似
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                # Clip 比例: 有多少比率被裁剪了
                clip_frac = (
                    (torch.abs(ratio - 1.0) > self.config.clip_eps).float().mean().item()
                )

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += approx_kl
            total_clip_frac += clip_frac

        n_epochs = self.config.ppo_epochs
        stats = {
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
            "approx_kl": total_kl / n_epochs,
            "clip_fraction": total_clip_frac / n_epochs,
            "mean_reward": experience.rewards.sum(dim=-1).mean().item(),
            "mean_advantage": experience.advantages.mean().item(),
            "kl_coef": self.kl_coef,
        }

        # ---- 自适应 KL 系数 ----
        if self.config.target_kl is not None:
            avg_kl = stats["approx_kl"]
            if avg_kl > 1.5 * self.config.target_kl:
                self.kl_coef *= 1.5  # KL 太大 → 加大惩罚
            elif avg_kl < self.config.target_kl / 1.5:
                self.kl_coef *= 0.67  # KL 太小 → 减小惩罚
            stats["kl_coef"] = self.kl_coef

        return stats

    # ==================================================================
    # 完整训练循环
    # ==================================================================

    def train(self, prompt_generator) -> List[Dict]:
        """
        完整的 PPO-RLHF 训练循环。

        参数:
            prompt_generator: 一个可调用对象，每次调用返回一批 prompt
                              shape = (batch_size, prompt_len)

        返回:
            all_stats: 所有迭代的训练统计
        """
        all_stats = []

        for iteration in range(self.config.num_iterations):
            # 1. 获取一批 prompt
            prompts = prompt_generator().to(self.device)

            # 2. 收集经验 (rollout)
            experience = self.collect_experience(prompts)

            # 3. PPO 优化
            stats = self.ppo_update(experience)
            stats["iteration"] = iteration + 1

            all_stats.append(stats)

            # 4. 日志
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(
                    f"  [PPO] Iter {iteration + 1:3d}/{self.config.num_iterations} | "
                    f"Reward: {stats['mean_reward']:7.3f} | "
                    f"Policy Loss: {stats['policy_loss']:7.4f} | "
                    f"Value Loss: {stats['value_loss']:7.4f} | "
                    f"KL: {stats['approx_kl']:.4f} | "
                    f"Clip%: {stats['clip_fraction']:.2%} | "
                    f"Entropy: {stats['entropy']:.3f}"
                )

        return all_stats
