"""
GRPO 训练主脚本

本脚本演示完整的 GRPO 训练流程:
    1. 预训练 (SFT): 用模拟数据进行监督微调
    2. GRPO 训练:    用奖励函数 + GRPO 优化策略 (无需 Critic)

与 PPO 训练脚本的区别:
    - 无需创建 Value Model (省显存)
    - 每个 prompt 生成多条回复 (组采样)
    - 优势函数通过组内比较计算 (无需 GAE)

由于是教学目的，我们使用:
    - 字节级 token (vocab_size=256)，无需 tokenizer
    - 基于规则的奖励函数（替代真实奖励模型）
    - ~1M 参数的小模型，可在 Mac CPU/MPS 上运行

用法:
    python train.py

预计运行时间: 1-3 分钟 (CPU), ~1 分钟 (MPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List

from model import GPTConfig, GPTLanguageModel, count_parameters
from grpo_trainer import GRPOConfig, GRPOTrainer


# ==============================================================================
# 设备选择工具
# ==============================================================================

def get_device() -> torch.device:
    """
    自动选择最佳计算设备。

    优先级: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

    注意事项:
        - MPS 设备上某些 PyTorch 操作可能不支持，会自动回退到 CPU
        - 所有 tensor 必须在同一设备上，否则会报错
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ==============================================================================
# 奖励函数设计
# ==============================================================================

class RuleBasedReward:
    """
    基于规则的奖励函数。

    在真实 RLHF 中，奖励来自训练好的奖励模型 (Reward Model)。
    这里用简单规则替代，方便观察 GRPO 的训练效果。

    奖励规则:
        1. 多样性奖励: 鼓励生成不重复的 token
        2. 重复惩罚:   惩罚连续重复相同 token
        3. 模式奖励:   鼓励生成特定模式（如升序片段）

    GRPO 特别适合这种奖励函数:
        因为 GRPO 通过组内比较来计算优势，
        即使奖励函数的绝对值不够准确，
        只要它能正确反映回复之间的相对好坏，训练就能正常进行。
    """

    def __init__(self, prompt_len: int, device: torch.device):
        self.prompt_len = prompt_len
        self.device = device

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        计算奖励。

        参数:
            sequences: 完整序列 (prompt + response), shape = (N, T)

        返回:
            rewards: 每个序列的标量奖励, shape = (N,)
        """
        N = sequences.shape[0]
        # 关键: 在正确的设备上创建 tensor
        rewards = torch.zeros(N, device=self.device)

        for i in range(N):
            response = sequences[i, self.prompt_len:]
            rewards[i] = self._score_single(response)

        return rewards

    def _score_single(self, response: torch.Tensor) -> float:
        """对单条回复打分。"""
        tokens = response.tolist()
        score = 0.0

        # 1. 多样性奖励: unique tokens / total tokens
        if len(tokens) > 0:
            diversity = len(set(tokens)) / len(tokens)
            score += diversity * 2.0

        # 2. 重复惩罚: 连续重复相同 token
        repeat_count = 0
        for j in range(1, len(tokens)):
            if tokens[j] == tokens[j - 1]:
                repeat_count += 1
        if len(tokens) > 1:
            repeat_ratio = repeat_count / (len(tokens) - 1)
            score -= repeat_ratio * 3.0

        # 3. 模式奖励: 鼓励局部递增模式
        ascending_count = 0
        for j in range(1, len(tokens)):
            if tokens[j] == tokens[j - 1] + 1:
                ascending_count += 1
        if len(tokens) > 1:
            score += (ascending_count / (len(tokens) - 1)) * 1.0

        return score


# ==============================================================================
# 预训练 (SFT) 阶段
# ==============================================================================

def pretrain_sft(
    model: GPTLanguageModel,
    config: GPTConfig,
    device: torch.device,
    num_samples: int = 200,
    epochs: int = 10,
    lr: float = 3e-4,
) -> List[float]:
    """
    监督微调 (Supervised Fine-Tuning, SFT)。

    参数:
        model:       要训练的模型
        config:      模型配置
        device:      计算设备
        num_samples: 训练样本数
        epochs:      训练轮数
        lr:          学习率

    返回:
        losses: 每个 epoch 的平均损失
    """
    print("=" * 60)
    print("阶段 1: 监督微调 (SFT)")
    print("=" * 60)

    # 生成训练数据: 包含局部递增模式的序列
    seq_len = 48
    data = []
    for _ in range(num_samples):
        seq = []
        pos = torch.randint(0, 200, (1,)).item()
        for j in range(seq_len):
            seq.append(pos % config.vocab_size)
            if torch.rand(1).item() < 0.7:
                pos += 1
            else:
                pos = torch.randint(0, 200, (1,)).item()
        data.append(seq)

    # 关键: 训练数据必须移动到目标设备
    train_data = torch.tensor(data, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(num_samples, device=device)
        batch_size = 16

        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch = train_data[idx]

            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            logits = model(input_ids)
            loss = criterion(
                logits.reshape(-1, config.vocab_size),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (num_samples // batch_size)
        losses.append(avg_loss)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  [SFT] Epoch {epoch + 1:2d}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


# ==============================================================================
# 主训练流程
# ==============================================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "GRPO 训练演示".center(48) + "║")
    print("║" + "(Group Relative Policy Optimization)".center(48) + "║")
    print("╚" + "═" * 58 + "╝")

    # ---- 设备选择 ----
    device = get_device()
    print(f"\n使用设备: {device}")

    # ---- 随机种子 ----
    torch.manual_seed(42)

    # ---- 模型配置 ----
    config = GPTConfig(
        vocab_size=256,     # 字节级 token
        max_seq_len=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
    )

    print(f"\n模型配置:")
    print(f"  词表大小:   {config.vocab_size}")
    print(f"  隐藏维度:   {config.d_model}")
    print(f"  层数:       {config.n_layers}")
    print(f"  注意力头:   {config.n_heads}")

    # ---- 创建策略模型 (GRPO 不需要 Critic!) ----
    policy_model = GPTLanguageModel(config).to(device)

    print(f"\n参数量:")
    print(f"  策略模型: {count_parameters(policy_model):,}")
    print(f"  (GRPO 无需 Critic，比 PPO 少约 50% 参数!)")

    # ==========================
    # 阶段 1: SFT 预训练
    # ==========================
    sft_losses = pretrain_sft(
        policy_model, config, device=device, num_samples=200, epochs=10, lr=3e-4
    )

    # 生成示例（SFT 后）
    policy_model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, 8), device=device)
        sft_output = policy_model.generate(prompt, max_new_tokens=24, temperature=0.8)
        print(f"\n  SFT 后生成示例:")
        print(f"    Prompt:   {prompt[0].tolist()}")
        print(f"    Response: {sft_output[0, 8:].tolist()}")

    # ==========================
    # 阶段 2: GRPO 训练
    # ==========================
    print("\n" + "=" * 60)
    print("阶段 2: GRPO 强化学习优化")
    print("=" * 60)

    prompt_len = 8
    reward_fn = RuleBasedReward(prompt_len=prompt_len, device=device)

    grpo_config = GRPOConfig(
        max_gen_len=24,          # 生成长度
        temperature=0.8,
        group_size=4,            # 每个 prompt 生成 4 条回复
        grpo_epochs=1,           # 每批数据上 GRPO 迭代 1 轮
        clip_eps=0.2,
        kl_coef=0.04,            # KL 惩罚系数 (GRPO 中通常较小)
        lr=1e-5,
        max_grad_norm=1.0,
        batch_size=4,            # 每批 4 个 prompt
        # 实际每轮处理: 4 prompts × 4 responses = 16 条序列
        num_iterations=30,
    )

    print(f"\nGRPO 配置:")
    print(f"  Group Size:    {grpo_config.group_size} (每个 prompt 生成 {grpo_config.group_size} 条回复)")
    print(f"  Clip ε:        {grpo_config.clip_eps}")
    print(f"  KL 系数:       {grpo_config.kl_coef}")
    print(f"  GRPO Epochs:   {grpo_config.grpo_epochs}")
    print(f"  迭代数:        {grpo_config.num_iterations}")
    print(f"  每轮采样量:    {grpo_config.batch_size} × {grpo_config.group_size} = {grpo_config.batch_size * grpo_config.group_size} 条序列")

    # 创建 GRPO 训练器
    trainer = GRPOTrainer(
        policy_model=policy_model,
        reward_fn=reward_fn,
        config=grpo_config,
        device=device,
    )

    # Prompt 生成器: 每次返回一批随机 prompt
    # 注意: 不在这里 .to(device)，因为 trainer.train() 内部会处理
    def prompt_generator():
        return torch.randint(0, 50, (grpo_config.batch_size, prompt_len))

    # 开始 GRPO 训练
    print(f"\n开始 GRPO 训练...")
    start_time = time.time()
    train_stats = trainer.train(prompt_generator)
    elapsed = time.time() - start_time
    print(f"\n  训练完成! 耗时: {elapsed:.1f}s")

    # ==========================
    # 训练结果分析
    # ==========================
    print("\n" + "=" * 60)
    print("训练结果分析")
    print("=" * 60)

    # 奖励变化趋势
    rewards = [s["mean_reward"] for s in train_stats]
    print(f"\n  奖励变化:")
    print(f"    初始:   {rewards[0]:.3f}")
    print(f"    最终:   {rewards[-1]:.3f}")
    print(f"    最高:   {max(rewards):.3f}")
    print(f"    变化:   {rewards[-1] - rewards[0]:+.3f}")

    # KL 散度变化
    kls = [s["kl_divergence"] for s in train_stats]
    print(f"\n  KL 散度:")
    print(f"    初始:   {kls[0]:.4f}")
    print(f"    最终:   {kls[-1]:.4f}")

    # 组内奖励分布
    spreads = [s["group_reward_spread"] for s in train_stats]
    print(f"\n  组内奖励差距 (Spread):")
    print(f"    初始:   {spreads[0]:.3f}")
    print(f"    最终:   {spreads[-1]:.3f}")
    print(f"    (Spread 减小说明策略变得更稳定，不同回复质量更接近)")

    # 生成示例（GRPO 后）
    trainer.policy.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, prompt_len), device=device)
        grpo_output = trainer.policy.generate(
            prompt, max_new_tokens=24, temperature=0.8
        )
        print(f"\n  GRPO 后生成示例:")
        print(f"    Prompt:   {prompt[0].tolist()}")
        print(f"    Response: {grpo_output[0, prompt_len:].tolist()}")

        reward = reward_fn(grpo_output)
        print(f"    奖励:     {reward.item():.3f}")

    # 组采样演示 —— 展示 GRPO 的核心: 一个 prompt 多条回复
    print(f"\n  GRPO 组采样演示 (Group Sampling):")
    print(f"  对同一个 prompt 生成 {grpo_config.group_size} 条回复:")
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, prompt_len), device=device)
        print(f"    Prompt: {prompt[0].tolist()}")
        print()

        group_rewards = []
        for g in range(grpo_config.group_size):
            output = trainer.policy.generate(
                prompt, max_new_tokens=24, temperature=0.8
            )
            response = output[0, prompt_len:].tolist()
            r = reward_fn(output).item()
            group_rewards.append(r)

            unique_ratio = len(set(response)) / len(response) if response else 0
            print(
                f"    回复 {g + 1}: Reward={r:+.2f}, "
                f"Unique={unique_ratio:.0%}, "
                f"Tokens={response[:12]}..."
            )

        mean_r = sum(group_rewards) / len(group_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        print(f"\n    组内均值: {mean_r:.2f}, 标准差: {std_r:.2f}")
        print(f"    组内优势: {[(r - mean_r) / (std_r + 1e-8) for r in group_rewards]}")

    # 多次生成对比
    print(f"\n  多次独立生成对比:")
    for trial in range(3):
        with torch.no_grad():
            prompt = torch.randint(0, 50, (1, prompt_len), device=device)
            output = trainer.policy.generate(
                prompt, max_new_tokens=24, temperature=0.8
            )
            reward = reward_fn(output)
            response = output[0, prompt_len:].tolist()
            unique_ratio = len(set(response)) / len(response) if response else 0

            repeats = sum(
                1 for j in range(1, len(response)) if response[j] == response[j - 1]
            )

            print(
                f"    Trial {trial + 1}: "
                f"Reward={reward.item():+.2f}, "
                f"Unique={unique_ratio:.0%}, "
                f"Repeats={repeats}, "
                f"Tokens={response[:12]}..."
            )

    # ---- GRPO vs PPO 对比总结 ----
    print("\n" + "=" * 60)
    print("GRPO vs PPO 关键对比")
    print("=" * 60)
    comparison = """
    ┌─────────────────┬─────────────────────┬─────────────────────┐
    │     维度         │        PPO          │        GRPO         │
    ├─────────────────┼─────────────────────┼─────────────────────┤
    │ 模型数量         │ Actor + Critic + Ref│ Actor + Ref         │
    │ 显存需求         │ 高 (3个模型)        │ 低 (2个模型)        │
    │ 优势估计         │ GAE (需要 Critic)   │ 组内归一化          │
    │ 奖励粒度         │ Token 级            │ 序列级              │
    │ 采样策略         │ 1 prompt → 1 回复   │ 1 prompt → G 回复   │
    │ 采样效率         │ 高 (数据复用)       │ 中 (需要组采样)     │
    │ 实现复杂度       │ 较复杂              │ 较简单              │
    │ 典型应用         │ InstructGPT, ChatGPT│ DeepSeek-R1         │
    └─────────────────┴─────────────────────┴─────────────────────┘

    GRPO 的核心洞察:
        "不需要一个精确的价值估计，只需要知道哪条回复相对更好。"

    这与人类评估偏好的方式一致:
        人类也不会给回复打"绝对分数"，而是在比较中判断好坏。
        GRPO 将这种"比较"的思想直接融入算法设计中。
"""
    print(comparison)

    # ---- 可视化训练曲线 (ASCII) ----
    print("  奖励训练曲线 (ASCII):")
    _ascii_plot(rewards, width=50, height=8, label="Mean Reward")


def _ascii_plot(values: List[float], width: int = 50, height: int = 8, label: str = ""):
    """简单的 ASCII 折线图。"""
    if not values:
        return

    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v if max_v != min_v else 1.0

    n = len(values)
    resampled = []
    for i in range(width):
        idx = int(i * (n - 1) / max(width - 1, 1))
        resampled.append(values[idx])

    print(f"    {label}")
    print(f"    {max_v:7.2f} ┤")
    for row in range(height - 2):
        threshold = max_v - (row + 1) * range_v / (height - 1)
        line = "    " + " " * 8 + "│"
        for v in resampled:
            if v >= threshold:
                line += "█"
            else:
                line += " "
        print(line)
    print(f"    {min_v:7.2f} ┤" + "─" * width)
    print(f"    " + " " * 8 + "0" + " " * (width - 5) + f"{len(values)}")


if __name__ == "__main__":
    main()
