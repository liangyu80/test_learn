"""
RLVR (Reinforcement Learning with Verifiable Rewards) 训练器

RLVR 是一种利用"可验证奖励"进行强化学习的方法，特别适用于
数学推理、代码生成、逻辑推理等有明确正误标准的任务。

RLVR 与 GRPO/PPO 的核心区别:
    ┌─────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
    │     维度         │        PPO           │        GRPO          │        RLVR          │
    ├─────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
    │ 奖励来源         │ 学习的奖励模型        │ 学习的奖励模型        │ 可验证的规则          │
    │                 │ (Reward Model)       │ (Reward Model)       │ (Verifiable Rules)   │
    ├─────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
    │ 奖励类型         │ 连续值 (-∞, +∞)      │ 连续值 (-∞, +∞)      │ 二值/离散 {0, 1}     │
    │                 │ 主观偏好评分          │ 主观偏好评分          │ 客观正误判定          │
    ├─────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
    │ 优势函数         │ GAE (Critic)         │ 组内相对归一化        │ 组内相对归一化        │
    │                 │ 需要 Value Model      │ 无需 Critic          │ 无需 Critic          │
    ├─────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
    │ 适用场景         │ 通用对齐              │ 通用对齐              │ 数学/代码/推理        │
    │                 │ 对话、摘要等          │ 对话、推理            │ 有标准答案的任务      │
    ├─────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
    │ 奖励噪声         │ 高 (RM 不完美)       │ 高 (RM 不完美)       │ 零 (规则精确)        │
    │                 │ 容易 reward hacking   │ 容易 reward hacking   │ 无法被 hack          │
    └─────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

RLVR 的关键优势:
    1. 零奖励噪声: 可验证奖励是精确的，不存在 reward hacking 的问题
    2. 无需奖励模型训练: 省去了 RLHF 第二阶段 (奖励模型训练)
    3. 可扩展: 可以自动生成无限多的训练数据 (如数学题)
    4. 信号清晰: 二值奖励 (对/错) 提供了最直接的学习信号

RLVR 算法流程:
    ┌──────────────────────────────────────────────────────────┐
    │                    RLVR Training Loop                     │
    │                                                          │
    │  对每个问题 q (含标准答案 a*):                             │
    │  1. 用策略模型 π_θ 生成 G 条回答 {o_1, ..., o_G}          │
    │  2. 验证器检查每条回答的正确性:                             │
    │     r_i = verify(o_i, a*) ∈ {0, 1}                       │
    │  3. 组内归一化计算优势:                                    │
    │     Â_i = (r_i - mean(r)) / (std(r) + ε)                 │
    │  4. 对正确回答增大生成概率，对错误回答降低概率             │
    │  5. PPO-Clip 优化 + KL 约束                              │
    │                                                          │
    │  循环直到策略收敛                                          │
    └──────────────────────────────────────────────────────────┘

关键论文:
    - "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
       via Reinforcement Learning" (DeepSeek, 2025)
    - "Let's Verify Step by Step" (Lightman et al., 2023)
    - "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import copy


# ==============================================================================
# RLVR 训练配置
# ==============================================================================

@dataclass
class RLVRConfig:
    """
    RLVR 训练超参数。

    与 GRPO 的区别:
        - 奖励是二值的 (0/1)，不是连续值
        - 新增 correct_bonus / incorrect_penalty 控制奖励尺度
        - 新增 outcome_supervision 选择结果监督还是过程监督
    """
    # --- 生成参数 ---
    max_gen_len: int = 32           # 生成回复的最大长度
    temperature: float = 0.7        # 采样温度

    # --- 组采样参数 (与 GRPO 类似) ---
    group_size: int = 4             # 每个问题生成的回答数量 G
    # RLVR 中 group_size 的意义:
    #   在二值奖励下，组内必须同时存在正确和错误的回答，
    #   才能计算有意义的组内优势。
    #   如果 G 太小 (G=2)，可能一组全对或全错，优势为零。
    #   如果 G 太大 (G=16)，采样成本太高。
    #   推荐 G=4~8，确保组内有足够的对比。

    # --- RLVR 奖励参数 ---
    correct_bonus: float = 1.0      # 回答正确时的奖励
    incorrect_penalty: float = 0.0  # 回答错误时的惩罚 (通常为 0)
    # 在 RLVR 中，奖励的绝对值不重要 (组内归一化会处理)，
    # 重要的是正确和错误之间有区分度。
    # 简单设置为 correct=1, incorrect=0 即可。

    # --- 优化参数 ---
    rlvr_epochs: int = 1            # 每批数据上 RLVR 迭代的轮数
    clip_eps: float = 0.2           # PPO clip 范围 ε
    kl_coef: float = 0.04           # KL 散度惩罚系数 β
    lr: float = 1e-5                # 学习率
    max_grad_norm: float = 1.0      # 梯度裁剪阈值

    # --- 训练参数 ---
    batch_size: int = 4             # 每批问题数
    num_iterations: int = 50        # 总训练迭代数


# ==============================================================================
# RLVR 经验数据
# ==============================================================================

@dataclass
class RLVRExperience:
    """
    RLVR 训练的一批经验数据。

    与 GRPO 的主要区别:
        - verification_results: 二值验证结果 (0/1)，这是 RLVR 的核心
        - group_rewards 来自验证器而非奖励模型

    数据组织:
        假设 batch_size=B, group_size=G, gen_len=L
        sequences:              shape = (B*G, prompt_len + L)
        old_log_probs:          shape = (B*G, L)
        verification_results:   shape = (B, G) → 每条回答的正确性 {0, 1}
        advantages:             shape = (B*G,) → 展开后的组内优势
    """
    sequences: torch.Tensor = None              # (B*G, prompt_len + gen_len)
    prompt_len: int = 0
    old_log_probs: torch.Tensor = None          # (B*G, gen_len)
    ref_log_probs: torch.Tensor = None          # (B*G, gen_len)
    verification_results: torch.Tensor = None   # (B, G) 二值: 0=错误, 1=正确
    group_rewards: torch.Tensor = None          # (B, G) 基于验证结果的奖励
    advantages: torch.Tensor = None             # (B*G,) 展平的组内优势


# ==============================================================================
# RLVR 训练器
# ==============================================================================

class RLVRTrainer:
    """
    RLVR 训练器。

    核心组件:
        1. 策略模型 (Policy, π_θ):      根据问题生成回答
        2. 参考模型 (Reference, π_ref):  冻结的初始策略，用于 KL 约束
        3. 验证器 (Verifier):            验证回答的正确性

    RLVR 目标函数 (基于 GRPO 框架，但奖励来自验证器):
        L_RLVR(θ) = E_{q~P, {o_i}~π_θ_old}[
            1/G Σ_{i=1}^G  1/|o_i| Σ_{t=1}^{|o_i|}
            min(r_t(θ) · Â_i, clip(r_t(θ), 1-ε, 1+ε) · Â_i)
        ] - β · D_KL(π_θ || π_ref)

        其中:
            Â_i = (verify(o_i, a*) - mean) / std   (基于验证结果的组内优势)

    与 GRPO 的训练器相比:
        1. reward_fn 替换为 verifier (验证器返回二值结果)
        2. 新增 answer_key 传入正确答案
        3. 需要处理"全对"和"全错"组的特殊情况

    参数:
        policy_model:  策略模型
        verifier:      验证器函数 (sequences, answers) → {0, 1}
        config:        RLVR 配置
        device:        计算设备
    """

    def __init__(
        self,
        policy_model: nn.Module,
        verifier: Callable,
        config: RLVRConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy_model.to(device)
        self.verifier = verifier
        self.config = config
        self.device = device

        # ---------------------------------------------------------------
        # 参考模型 (Reference Model):
        # 冻结的策略模型副本，用于 KL 约束，防止策略偏离太远。
        # 在 RLVR 中，KL 约束尤其重要:
        #   因为二值奖励信号较稀疏，模型可能会"记住"答案模式
        #   而非真正学会推理。KL 约束防止这种过拟合。
        # ---------------------------------------------------------------
        self.ref_policy = copy.deepcopy(policy_model).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # 优化器：只优化策略模型
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.lr,
        )

        self.train_stats: List[Dict] = []

    # ==================================================================
    # 阶段 1: 收集经验 (Group Sampling + Verification)
    # ==================================================================

    @torch.no_grad()
    def collect_experience(
        self,
        prompts: torch.Tensor,
        answers: List,
    ) -> RLVRExperience:
        """
        收集一批经验: 对每个问题生成一组回答，然后验证正确性。

        RLVR 的采样流程:
            1. 对每个问题生成 G 条回答 (与 GRPO 相同)
            2. 用验证器检查每条回答的正确性 (RLVR 特有)
            3. 将验证结果转化为奖励
            4. 通过组内归一化计算优势

        参数:
            prompts: 一批问题, shape = (B, prompt_len)
            answers: 标准答案列表, 长度 = B

        返回:
            experience: 包含验证结果和组内优势的 RLVRExperience 对象
        """
        self.policy.eval()

        B = prompts.shape[0]
        G = self.config.group_size
        prompt_len = prompts.shape[1]

        # ---- 步骤 1: 复制 prompt，每个重复 G 次 ----
        expanded_prompts = prompts.repeat_interleave(G, dim=0)  # (B*G, prompt_len)

        # ---- 步骤 2: 批量生成 B*G 条回答 ----
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

        old_log_probs = torch.cat(all_log_probs, dim=-1)  # (B*G, gen_len)

        # ---- 步骤 3: 计算参考模型的 log 概率 ----
        ref_log_probs = self._compute_log_probs(self.ref_policy, generated, prompt_len)

        # ---- 步骤 4: 验证回答正确性 (RLVR 核心!) ----
        # 将答案也扩展为 B*G
        expanded_answers = []
        for ans in answers:
            expanded_answers.extend([ans] * G)

        # 验证器返回二值结果: 1=正确, 0=错误
        verification_results = self.verifier(generated, expanded_answers, prompt_len)
        # shape: (B*G,) → 重塑为 (B, G)
        verification_matrix = verification_results.view(B, G)

        # ---- 步骤 5: 将验证结果转化为奖励 ----
        # 简单映射: 正确 → correct_bonus, 错误 → incorrect_penalty
        group_rewards = torch.where(
            verification_matrix == 1,
            torch.tensor(self.config.correct_bonus, device=self.device),
            torch.tensor(self.config.incorrect_penalty, device=self.device),
        )

        # ---- 步骤 6: 计算组内相对优势 ----
        advantages = self._compute_group_advantages(group_rewards)  # (B*G,)

        experience = RLVRExperience(
            sequences=generated,
            prompt_len=prompt_len,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            verification_results=verification_matrix,
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

        在 RLVR 中的特殊处理:
            由于奖励是二值的 (0 或 1)，组内可能出现:
            1. 全对组: r = [1,1,1,1] → std = 0 → 优势 = 0 (无学习信号)
            2. 全错组: r = [0,0,0,0] → std = 0 → 优势 = 0 (无学习信号)
            3. 混合组: r = [1,0,1,0] → std > 0 → 有学习信号

            只有混合组能提供有效的学习信号!
            这就是为什么 RLVR 中 group_size 要足够大:
            如果模型准确率 p=0.8, G=4, 全对概率 = 0.8^4 = 0.41
            有效组比例 ≈ 59%，仍然有足够的学习信号。

        公式:
            Â_i = (r_i - μ_group) / (σ_group + ε)

        参数:
            group_rewards: shape = (B, G)

        返回:
            advantages: shape = (B*G,)
        """
        group_mean = group_rewards.mean(dim=-1, keepdim=True)  # (B, 1)
        group_std = group_rewards.std(dim=-1, keepdim=True)    # (B, 1)

        # 归一化: 加 1e-8 防止除零 (全对或全错组)
        advantages = (group_rewards - group_mean) / (group_std + 1e-8)  # (B, G)

        return advantages.view(-1)

    # ==================================================================
    # 阶段 2: RLVR 优化
    # ==================================================================

    def rlvr_update(self, experience: RLVRExperience) -> Dict[str, float]:
        """
        RLVR 优化步骤。

        与 GRPO 的 grpo_update 高度相似，核心区别在于:
            1. 优势来自二值验证结果而非连续奖励
            2. 新增准确率等 RLVR 特有指标
            3. 混合组比例是重要的监控指标

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
        token_advantages = advantages.unsqueeze(-1).expand(-1, gen_len)

        total_policy_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0

        for epoch in range(self.config.rlvr_epochs):
            # ---- 步骤 1: 计算当前策略的 log 概率 ----
            new_log_probs = self._compute_log_probs(
                self.policy, sequences, prompt_len
            )

            # ---- 步骤 2: 重要性采样比率 ----
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)  # (B*G, gen_len)

            # ---- 步骤 3: PPO-Clip 策略损失 ----
            surr1 = ratio * token_advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
            ) * token_advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # ---- 步骤 4: KL 散度惩罚 ----
            log_ratio_ref = ref_log_probs - new_log_probs
            ratio_ref = torch.exp(log_ratio_ref)
            kl_divergence = (ratio_ref * log_ratio_ref - (ratio_ref - 1)).mean()

            # ---- 步骤 5: 总损失 ----
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

        # ---- RLVR 特有指标 ----
        verification = experience.verification_results  # (B, G)
        accuracy = verification.float().mean().item()
        B = verification.shape[0]

        # 混合组比例: 组内既有正确又有错误回答的组
        group_sums = verification.sum(dim=-1)  # (B,)
        G = verification.shape[1]
        mixed_groups = ((group_sums > 0) & (group_sums < G)).float().mean().item()

        n_epochs = self.config.rlvr_epochs
        stats = {
            "policy_loss": total_policy_loss / n_epochs,
            "kl_divergence": total_kl / n_epochs,
            "clip_fraction": total_clip_frac / n_epochs,
            "mean_reward": experience.group_rewards.mean().item(),
            # RLVR 特有指标
            "accuracy": accuracy,                # 回答准确率
            "mixed_group_ratio": mixed_groups,   # 有效学习信号的比例
            "mean_advantage": experience.advantages.mean().item(),
            "advantage_std": experience.advantages.std().item(),
        }

        return stats

    # ==================================================================
    # 完整训练循环
    # ==================================================================

    def train(self, problem_generator: Callable) -> List[Dict]:
        """
        完整的 RLVR 训练循环。

        与 GRPO 的 train 方法对比:
            GRPO: prompt_generator() → prompts
            RLVR: problem_generator() → (prompts, answers)
            RLVR 需要额外提供标准答案用于验证。

        参数:
            problem_generator: 可调用对象，每次调用返回:
                - prompts: 一批问题, shape = (batch_size, prompt_len)
                - answers: 标准答案列表, 长度 = batch_size

        返回:
            all_stats: 所有迭代的训练统计
        """
        all_stats = []

        for iteration in range(self.config.num_iterations):
            # 1. 获取一批问题和标准答案
            prompts, answers = problem_generator()
            prompts = prompts.to(self.device)

            # 2. 收集经验 (生成 + 验证)
            experience = self.collect_experience(prompts, answers)

            # 3. RLVR 优化
            stats = self.rlvr_update(experience)
            stats["iteration"] = iteration + 1

            all_stats.append(stats)

            # 4. 日志 (包含 RLVR 特有指标)
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(
                    f"  [RLVR] Iter {iteration + 1:3d}/{self.config.num_iterations} | "
                    f"Acc: {stats['accuracy']:.2%} | "
                    f"Reward: {stats['mean_reward']:7.3f} | "
                    f"Loss: {stats['policy_loss']:7.4f} | "
                    f"KL: {stats['kl_divergence']:.4f} | "
                    f"Mixed%: {stats['mixed_group_ratio']:.2%}"
                )

        return all_stats
