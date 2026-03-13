"""
GRPO (Group Relative Policy Optimization) 训练器

GRPO 是 DeepSeek 提出的一种强化学习算法，用于语言模型的对齐优化。
它是 PPO 的简化变体，核心创新在于"组内相对优势估计"，无需价值模型 (Critic)。

GRPO vs PPO 的核心区别:
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │     维度         │        PPO           │        GRPO          │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 优势函数计算     │ 需要 Critic 模型      │ 组内相对比较          │
    │                 │ A_t = GAE(r, V)      │ A_i = (r_i-μ)/σ     │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 模型数量         │ Actor + Critic + Ref │ Actor + Ref          │
    │                 │ (3 个模型)           │ (2 个模型)           │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 显存需求         │ 高 (需要额外 Critic) │ 低 (省掉 Critic)     │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 奖励粒度         │ Token 级奖励 + GAE   │ 序列级奖励           │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 采样方式         │ 每个 prompt 生成 1 条 │ 每个 prompt 生成 G 条 │
    │                 │ 回复                 │ 回复 (一个组)        │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ KL 约束方式      │ KL 惩罚项加入奖励    │ KL 散度直接作为损失   │
    │                 │ r' = r - β·KL        │ L = L_clip + β·KL   │
    └─────────────────┴──────────────────────┴──────────────────────┘

GRPO 算法流程:
    ┌──────────────────────────────────────────────────────────┐
    │                    GRPO Training Loop                     │
    │                                                          │
    │  对每个 prompt q:                                         │
    │  1. 用策略模型 π_θ_old 生成 G 条回复 {o_1, ..., o_G}      │
    │  2. 奖励模型对每条回复打分: {r_1, ..., r_G}               │
    │  3. 组内归一化计算优势:                                   │
    │     Â_i = (r_i - mean(r)) / std(r)                       │
    │  4. 对回复中的每个 token 应用相同的序列级优势              │
    │  5. PPO-Clip 优化:                                       │
    │     L = E[ min(ratio·Â, clip(ratio)·Â) ]                │
    │     - β · KL(π_θ || π_ref)                              │
    │                                                          │
    │  循环直到策略收敛                                         │
    └──────────────────────────────────────────────────────────┘

关键论文:
    - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
       in Open Language Models" (Shao et al., 2024)
    - 在 DeepSeek-R1 中也大量使用 GRPO 进行推理能力的对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import copy


# ==============================================================================
# GRPO 训练配置
# ==============================================================================

@dataclass
class GRPOConfig:
    """
    GRPO 训练超参数。

    与 PPO 对比，GRPO 新增了 group_size 参数，去掉了 GAE 相关参数 (gamma, lam)。
    """
    # --- 生成参数 ---
    max_gen_len: int = 32           # 生成回复的最大长度
    temperature: float = 0.7        # 采样温度

    # --- GRPO 核心参数 ---
    group_size: int = 4             # 每个 prompt 生成的回复数量 G
    # group_size 的选择:
    #   太小 (G=2): 组内统计不稳定，优势估计噪声大
    #   太大 (G=16): 采样成本高，每轮需要更多前向传播
    #   实践中 G=4~8 效果最好
    #   DeepSeek-R1 使用 G=64，但那是在大规模集群上

    grpo_epochs: int = 1            # 每批数据上 GRPO 迭代的轮数
    # GRPO 论文中通常只迭代 1 轮 (mu=1)
    # 因为每个 prompt 已经有 G 条回复，数据利用已经很充分

    clip_eps: float = 0.2           # PPO clip 范围 ε
    # 与 PPO 相同: ratio 被限制在 [1-ε, 1+ε]

    # --- KL 约束参数 ---
    kl_coef: float = 0.04           # KL 散度惩罚系数 β
    # GRPO 中的 KL 约束:
    #   GRPO 直接在损失函数中加入 KL 项（而非加入奖励中）
    #   L_total = L_clip - β · D_KL(π_θ || π_ref)
    #
    #   DeepSeek 论文中 β 通常较小 (0.01~0.05)
    #   因为 GRPO 的组内归一化本身就有稳定效果

    # --- 优化参数 ---
    lr: float = 1e-5                # 学习率
    max_grad_norm: float = 1.0      # 梯度裁剪阈值

    # --- 训练参数 ---
    batch_size: int = 4             # 每批 prompt 数
    # 注意: 实际前向传播的 batch = batch_size × group_size
    # 例如 batch_size=4, group_size=4 → 每轮实际处理 16 条序列

    num_iterations: int = 50        # 总训练迭代数


# ==============================================================================
# GRPO 经验数据
# ==============================================================================

@dataclass
class GRPOExperience:
    """
    GRPO 训练的一批经验数据。

    与 PPO 的区别:
        - 没有 values (无 Critic 模型)
        - 没有 GAE advantages/returns
        - 新增: group_rewards (组内奖励，用于计算相对优势)
        - 新增: group_advantages (组内归一化后的优势)

    数据组织:
        假设 batch_size=B, group_size=G, gen_len=L
        sequences:       shape = (B*G, prompt_len + L)
        old_log_probs:   shape = (B*G, L)
        group_rewards:   shape = (B, G) → 每个 prompt 的 G 个回复的奖励
        advantages:      shape = (B*G,)  → 展开后的组内优势
    """
    sequences: torch.Tensor = None          # (B*G, prompt_len + gen_len)
    prompt_len: int = 0
    old_log_probs: torch.Tensor = None      # (B*G, gen_len)
    ref_log_probs: torch.Tensor = None      # (B*G, gen_len)
    group_rewards: torch.Tensor = None      # (B, G)
    advantages: torch.Tensor = None         # (B*G,) 展平的组内优势


# ==============================================================================
# GRPO 训练器
# ==============================================================================

class GRPOTrainer:
    """
    GRPO 训练器。

    核心组件 (比 PPO 少一个 Critic):
        1. 策略模型 (Policy, π_θ):      生成回复
        2. 参考模型 (Reference, π_ref):  冻结的初始策略，用于 KL 约束
        3. 奖励函数 (Reward):            评估回复质量

    GRPO 目标函数:
        L_GRPO(θ) = E_{q~P, {o_i}~π_θ_old}[
            1/G Σ_{i=1}^G  1/|o_i| Σ_{t=1}^{|o_i|}
            min(r_t(θ) · Â_i, clip(r_t(θ), 1-ε, 1+ε) · Â_i)
        ] - β · D_KL(π_θ || π_ref)

        其中:
            G = group_size (每个 prompt 生成的回复数)
            r_t(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})
            Â_i = (r_i - mean({r_j})) / std({r_j})  (组内相对优势)

    组内相对优势 (Group Relative Advantage) 的直觉:
        假设对 prompt "什么是AI?" 生成 4 条回复:
            o_1: 奖励 = 5.0  → Â_1 > 0 (好回复，鼓励)
            o_2: 奖励 = 3.0  → Â_2 ≈ 0 (一般回复)
            o_3: 奖励 = 1.0  → Â_3 < 0 (差回复，抑制)
            o_4: 奖励 = 3.0  → Â_4 ≈ 0 (一般回复)

        归一化后:
            Â_i = (r_i - 3.0) / std  → 好回复正优势，差回复负优势

        这种相对比较的好处:
            1. 不需要绝对奖励的准确校准
            2. 自动适应奖励的尺度变化
            3. 完全不需要 Critic 模型估计基线

    参数:
        policy_model:  策略模型
        reward_fn:     奖励函数
        config:        GRPO 配置
        device:        计算设备
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_fn,
        config: GRPOConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy_model.to(device)
        self.reward_fn = reward_fn
        self.config = config
        self.device = device

        # ---------------------------------------------------------------
        # 参考模型 (Reference Model):
        # 与 PPO 相同，冻结的策略模型副本，PPO 训练过程中不更新。
        # 用于 KL 散度约束，防止 reward hacking。
        # ---------------------------------------------------------------
        self.ref_policy = copy.deepcopy(policy_model).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # 优化器：只优化策略模型（无 Critic）
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.lr,
        )

        self.train_stats: List[Dict] = []

    # ==================================================================
    # 阶段 1: 收集经验 (Group Sampling)
    # ==================================================================

    @torch.no_grad()
    def collect_experience(self, prompts: torch.Tensor) -> GRPOExperience:
        """
        收集一批经验：对每个 prompt 生成一组回复。

        GRPO 的核心采样策略:
            每个 prompt 不只生成 1 条回复（PPO 的做法），
            而是生成 G 条回复，形成一个"组"。

        实现细节:
            - 将每个 prompt 复制 G 次
            - 批量生成 B*G 条回复
            - 计算奖励后，按组进行归一化

        参数:
            prompts: 一批 prompt, shape = (B, prompt_len)

        返回:
            experience: 包含组内经验的 GRPOExperience 对象
        """
        self.policy.eval()

        B = prompts.shape[0]
        G = self.config.group_size
        prompt_len = prompts.shape[1]

        # ---- 步骤 1: 复制 prompt，每个重复 G 次 ----
        # (B, prompt_len) → (B*G, prompt_len)
        # 例如 prompts = [[p1], [p2]], G=3
        # → expanded = [[p1], [p1], [p1], [p2], [p2], [p2]]
        expanded_prompts = prompts.repeat_interleave(G, dim=0)  # (B*G, prompt_len)

        # ---- 步骤 2: 批量生成 B*G 条回复 ----
        generated = expanded_prompts.clone()
        all_log_probs = []

        for t in range(self.config.max_gen_len):
            logits = self.policy(generated)
            next_logits = logits[:, -1, :] / self.config.temperature

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B*G, 1)

            log_probs = F.log_softmax(next_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, next_token)  # (B*G, 1)
            all_log_probs.append(selected_log_probs)

            generated = torch.cat([generated, next_token], dim=-1)

        # 堆叠 log 概率: shape = (B*G, gen_len)
        old_log_probs = torch.cat(all_log_probs, dim=-1)

        # ---- 步骤 3: 计算参考模型的 log 概率 ----
        ref_log_probs = self._compute_log_probs(self.ref_policy, generated, prompt_len)

        # ---- 步骤 4: 计算奖励 ----
        # reward_fn 返回 (B*G,) 的标量奖励
        raw_rewards = self.reward_fn(generated)  # (B*G,)

        # 重塑为组: (B, G)
        group_rewards = raw_rewards.view(B, G)

        # ---- 步骤 5: 计算组内相对优势 ----
        advantages = self._compute_group_advantages(group_rewards)  # (B*G,)

        experience = GRPOExperience(
            sequences=generated,
            prompt_len=prompt_len,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            group_rewards=group_rewards,
            advantages=advantages,
        )

        return experience

    def _compute_log_probs(
        self, model: nn.Module, sequences: torch.Tensor, prompt_len: int
    ) -> torch.Tensor:
        """
        计算给定模型对 response 部分每个 token 的 log 概率。

        参数:
            model:      语言模型
            sequences:  完整序列 (prompt + response), shape = (N, T)
            prompt_len: prompt 的长度

        返回:
            log_probs: shape = (N, gen_len)
        """
        logits = model(sequences[:, :-1])  # 预测位置 1..T-1
        log_probs = F.log_softmax(logits, dim=-1)

        response_tokens = sequences[:, prompt_len:]     # (N, gen_len)
        response_logprobs = log_probs[:, prompt_len - 1:, :]  # (N, gen_len, vocab)

        selected = response_logprobs.gather(
            2, response_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return selected

    def _compute_group_advantages(self, group_rewards: torch.Tensor) -> torch.Tensor:
        """
        计算组内相对优势 (Group Relative Advantage)。

        这是 GRPO 的核心创新！

        公式:
            Â_i = (r_i - μ_group) / σ_group

            其中:
                μ_group = mean({r_1, ..., r_G})  组内奖励均值
                σ_group = std({r_1, ..., r_G})   组内奖励标准差

        直觉:
            组内相对比较，将"绝对好坏"转换为"相对好坏"。
            - 即使所有回复奖励都很高 (r=[8,9,10])，
              也会产生正负优势 (Â=[-1,0,1])
            - 即使所有回复奖励都很低 (r=[1,2,3])，
              也会产生类似的正负优势 (Â=[-1,0,1])

        这种设计的好处:
            1. 自然的基线减除: μ_group 相当于一个自适应的基线 (baseline)
               在 PPO 中，基线由 Critic 模型提供: A_t = R_t - V(s_t)
               在 GRPO 中，基线就是组内均值: Â_i = r_i - μ_group
               → 无需训练 Critic！

            2. 奖励尺度不变性: 除以 σ_group 使优势归一化
               无论奖励模型的绝对分数如何变化，优势始终在合理范围

            3. 相对排序: 只关心回复之间的相对好坏，不关心绝对分数
               这与人类偏好的本质一致（人类也是通过比较来判断好坏）

        参数:
            group_rewards: shape = (B, G), 每个 prompt 的 G 个回复的奖励

        返回:
            advantages: shape = (B*G,), 展平的组内优势
        """
        # 计算组内均值和标准差
        group_mean = group_rewards.mean(dim=-1, keepdim=True)  # (B, 1)
        group_std = group_rewards.std(dim=-1, keepdim=True)    # (B, 1)

        # 归一化: Â_i = (r_i - μ) / σ
        # 加 1e-8 防止除零（当所有回复奖励相同时 σ=0）
        advantages = (group_rewards - group_mean) / (group_std + 1e-8)  # (B, G)

        # 展平: (B, G) → (B*G,)
        return advantages.view(-1)

    # ==================================================================
    # 阶段 2: GRPO 优化
    # ==================================================================

    def grpo_update(self, experience: GRPOExperience) -> Dict[str, float]:
        """
        GRPO 优化步骤。

        GRPO 目标函数详解:
        ==================

        L_GRPO(θ) = 1/G Σ_{i=1}^G [
            1/|o_i| Σ_{t=1}^{|o_i|}
            min(r_t(θ)·Â_i, clip(r_t(θ), 1-ε, 1+ε)·Â_i)
        ] - β · D_KL(π_θ || π_ref)

        分解来看:

        1. PPO-Clip 部分 (与 PPO 相同):
           对每个 token 计算 clipped surrogate objective
           - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
           - 取 min(unclipped, clipped) 确保保守更新

        2. 序列级优势 (GRPO 特色):
           同一条回复的所有 token 共享相同的优势 Â_i
           这与 PPO 不同: PPO 中每个 token 有自己的 GAE 优势

        3. 长度归一化:
           除以 |o_i| (回复长度)，避免长回复主导损失
           → 每个 token 的贡献被均等化

        4. KL 惩罚 (直接加入损失):
           GRPO 直接在损失函数中加入 KL 散度
           而非像 PPO 那样加入奖励信号中
           D_KL 的计算:
           D_KL(π_θ || π_ref) = E_{π_ref}[
               π_ref(o|q)/π_θ(o|q) · log(π_ref(o|q)/π_θ(o|q))
           ]
           实际实现中用 KL 的近似:
           D_KL ≈ exp(log_ref - log_new) · (log_ref - log_new) - 1
           ≈ (ratio_ref · log_ratio_ref - 1) 的均值

        参数:
            experience: 收集的组内经验数据

        返回:
            stats: 训练统计信息
        """
        self.policy.train()

        sequences = experience.sequences           # (B*G, T)
        old_log_probs = experience.old_log_probs   # (B*G, gen_len)
        ref_log_probs = experience.ref_log_probs   # (B*G, gen_len)
        advantages = experience.advantages         # (B*G,)
        prompt_len = experience.prompt_len
        gen_len = old_log_probs.shape[1]

        # 将序列级优势扩展到 token 级
        # (B*G,) → (B*G, gen_len): 同一条回复的所有 token 共享相同优势
        token_advantages = advantages.unsqueeze(-1).expand(-1, gen_len)

        total_policy_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0

        for epoch in range(self.config.grpo_epochs):
            # ---- 步骤 1: 计算当前策略的 log 概率 ----
            new_log_probs = self._compute_log_probs(
                self.policy, sequences, prompt_len
            )

            # ---- 步骤 2: 计算重要性采样比率 ----
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)  # (B*G, gen_len)

            # ---- 步骤 3: PPO-Clip 策略损失 ----
            surr1 = ratio * token_advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
            ) * token_advantages

            # min → 悲观估计 (与 PPO 相同)
            # 除以 gen_len → 长度归一化 (GRPO 特色)
            policy_loss = -torch.min(surr1, surr2).mean()

            # ---- 步骤 4: KL 散度惩罚 ----
            # GRPO 使用的 KL 散度近似:
            # D_KL(π_θ || π_ref) ≈ E[exp(log π_ref - log π_θ) · (log π_ref - log π_θ) - 1]
            #
            # 更简单的近似 (本实现使用):
            # KL ≈ 0.5 * (log π_θ - log π_ref)² + higher order terms
            # 或者用标准近似: KL ≈ (ratio - 1) - log(ratio)
            log_ratio_ref = ref_log_probs - new_log_probs  # log(π_ref / π_θ)
            # 使用 DeepSeek 论文中的公式:
            # D_KL = exp(log_ratio_ref) * log_ratio_ref - (exp(log_ratio_ref) - 1)
            # 简化后等价于: D_KL = ratio_ref * log(ratio_ref) - ratio_ref + 1
            ratio_ref = torch.exp(log_ratio_ref)
            kl_divergence = (ratio_ref * log_ratio_ref - (ratio_ref - 1)).mean()

            # ---- 步骤 5: 总损失 ----
            # L_total = L_clip + β · D_KL
            # 注意符号: policy_loss 已取负号，kl_divergence 正值表示偏离
            loss = policy_loss + self.config.kl_coef * kl_divergence

            # ---- 步骤 6: 反向传播 + 梯度裁剪 + 更新 ----
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            # ---- 统计 ----
            with torch.no_grad():
                approx_kl = kl_divergence.item()
                clip_frac = (
                    (torch.abs(ratio - 1.0) > self.config.clip_eps).float().mean().item()
                )

            total_policy_loss += policy_loss.item()
            total_kl += approx_kl
            total_clip_frac += clip_frac

        n_epochs = self.config.grpo_epochs
        stats = {
            "policy_loss": total_policy_loss / n_epochs,
            "kl_divergence": total_kl / n_epochs,
            "clip_fraction": total_clip_frac / n_epochs,
            "mean_reward": experience.group_rewards.mean().item(),
            "reward_std": experience.group_rewards.std().item(),
            "mean_advantage": experience.advantages.mean().item(),
            "advantage_std": experience.advantages.std().item(),
            # 组内奖励统计 (理解 GRPO 效果的关键指标)
            "group_reward_spread": (
                experience.group_rewards.max(dim=-1).values
                - experience.group_rewards.min(dim=-1).values
            ).mean().item(),
        }

        return stats

    # ==================================================================
    # 完整训练循环
    # ==================================================================

    def train(self, prompt_generator) -> List[Dict]:
        """
        完整的 GRPO 训练循环。

        参数:
            prompt_generator: 可调用对象，每次调用返回一批 prompt
                              shape = (batch_size, prompt_len)

        返回:
            all_stats: 所有迭代的训练统计
        """
        all_stats = []

        for iteration in range(self.config.num_iterations):
            # 1. 获取一批 prompt
            # 关键: 确保 prompt 在正确的设备上
            prompts = prompt_generator().to(self.device)

            # 2. 收集组内经验 (每个 prompt 生成 G 条回复)
            experience = self.collect_experience(prompts)

            # 3. GRPO 优化
            stats = self.grpo_update(experience)
            stats["iteration"] = iteration + 1

            all_stats.append(stats)

            # 4. 日志
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(
                    f"  [GRPO] Iter {iteration + 1:3d}/{self.config.num_iterations} | "
                    f"Reward: {stats['mean_reward']:7.3f} | "
                    f"Policy Loss: {stats['policy_loss']:7.4f} | "
                    f"KL: {stats['kl_divergence']:.4f} | "
                    f"Clip%: {stats['clip_fraction']:.2%} | "
                    f"Spread: {stats['group_reward_spread']:.3f}"
                )

        return all_stats
