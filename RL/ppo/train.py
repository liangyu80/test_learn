"""
PPO-RLHF 训练主脚本

本脚本演示完整的 RLHF 训练流程:
    1. 预训练 (SFT): 用模拟数据进行监督微调
    2. PPO 训练:     用奖励函数 + PPO 优化策略

由于是教学目的，我们使用:
    - 字节级 token (vocab_size=256)，无需 tokenizer
    - 基于规则的奖励函数（替代真实奖励模型）
    - ~1M 参数的小模型，可在 Mac CPU 上运行

用法:
    python train.py

预计运行时间: 2-5 分钟 (CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List

from model import GPTConfig, GPTLanguageModel, GPTValueModel, count_parameters
from ppo_trainer import PPOConfig, PPOTrainer


# ==============================================================================
# 奖励函数设计
# ==============================================================================

class RuleBasedReward:
    """
    基于规则的奖励函数。

    在真实 RLHF 中，奖励来自训练好的奖励模型 (Reward Model)。
    这里用简单规则替代，方便观察 PPO 的训练效果。

    奖励规则 (可以随意修改来观察不同行为):
        1. 多样性奖励: 鼓励生成不重复的 token（unique token 比例越高越好）
        2. 长度奖励:   轻微鼓励较长的非零序列
        3. 模式奖励:   鼓励生成特定模式（如升序片段）
        4. 重复惩罚:   惩罚连续重复相同 token

    设计哲学:
        奖励函数决定了模型会学到什么行为。
        在真实 RLHF 中，奖励模型从人类偏好中学习，
        因此模型最终会学到"人类喜欢"的回复风格。
    """

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        计算奖励。

        参数:
            sequences: 完整序列 (prompt + response), shape = (B, T)

        返回:
            rewards: 每个序列的标量奖励, shape = (B,)
        """
        B = sequences.shape[0]
        rewards = torch.zeros(B, device=sequences.device)

        for i in range(B):
            response = sequences[i, self.prompt_len:]
            rewards[i] = self._score_single(response)

        return rewards

    def _score_single(self, response: torch.Tensor) -> float:
        """对单条回复打分。"""
        tokens = response.tolist()
        score = 0.0

        # 1. 多样性奖励: unique tokens / total tokens
        # 鼓励模型生成丰富多样的内容
        if len(tokens) > 0:
            diversity = len(set(tokens)) / len(tokens)
            score += diversity * 2.0

        # 2. 重复惩罚: 连续重复相同 token
        # 惩罚模式: "aaaaaa..."（退化输出）
        repeat_count = 0
        for j in range(1, len(tokens)):
            if tokens[j] == tokens[j - 1]:
                repeat_count += 1
        if len(tokens) > 1:
            repeat_ratio = repeat_count / (len(tokens) - 1)
            score -= repeat_ratio * 3.0  # 重复越多，惩罚越重

        # 3. 模式奖励: 鼓励局部递增模式
        # 检查是否存在 x, x+1, x+2 这样的升序片段
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
    num_samples: int = 200,
    epochs: int = 10,
    lr: float = 3e-4,
) -> List[float]:
    """
    监督微调 (Supervised Fine-Tuning, SFT)。

    在 RLHF 的阶段 1 中，模型在高质量的 (prompt, response) 数据上
    进行监督微调。这确保模型在 PPO 训练之前就有基本的语言能力。

    这里用简单的模式数据替代真实数据:
        - 生成包含局部递增模式的序列
        - 让模型学会基本的序列建模能力

    参数:
        model:       要训练的模型
        config:      模型配置
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
    # 这模拟了"高质量数据"——包含结构化模式的序列
    seq_len = 48
    data = []
    for _ in range(num_samples):
        seq = []
        pos = torch.randint(0, 200, (1,)).item()
        for j in range(seq_len):
            seq.append(pos % config.vocab_size)
            # 70% 概率递增，30% 概率跳转（增加随机性）
            if torch.rand(1).item() < 0.7:
                pos += 1
            else:
                pos = torch.randint(0, 200, (1,)).item()
        data.append(seq)

    train_data = torch.tensor(data, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        # 随机打乱
        perm = torch.randperm(num_samples)
        batch_size = 16

        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch = train_data[idx]

            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            logits, _ = model(input_ids)
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
    print("║" + "PPO-RLHF 训练演示".center(46) + "║")
    print("╚" + "═" * 58 + "╝")

    # ---- 设备选择 ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print(f"\n使用设备: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n使用设备: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n使用设备: CPU")

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

    # ---- 创建模型 ----
    policy_model = GPTLanguageModel(config).to(device)
    value_model = GPTValueModel(config).to(device)

    print(f"\n参数量:")
    print(f"  策略模型: {count_parameters(policy_model):,}")
    print(f"  价值模型: {count_parameters(value_model):,}")
    print(f"  总计:     {count_parameters(policy_model) + count_parameters(value_model):,}")

    # ==========================
    # 阶段 1: SFT 预训练
    # ==========================
    sft_losses = pretrain_sft(policy_model, config, num_samples=200, epochs=10, lr=3e-4)

    # 生成示例（SFT 后）
    policy_model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, 8), device=device)
        sft_output = policy_model.generate(prompt, max_new_tokens=24, temperature=0.8)
        print(f"\n  SFT 后生成示例:")
        print(f"    Prompt:   {prompt[0].tolist()}")
        print(f"    Response: {sft_output[0, 8:].tolist()}")

    # ==========================
    # 阶段 2: PPO 训练
    # ==========================
    print("\n" + "=" * 60)
    print("阶段 2: PPO 强化学习优化")
    print("=" * 60)

    prompt_len = 8
    reward_fn = RuleBasedReward(prompt_len=prompt_len)

    ppo_config = PPOConfig(
        max_gen_len=24,          # 生成长度
        temperature=0.8,
        ppo_epochs=4,            # 每批数据上 PPO 迭代轮数
        clip_eps=0.2,            # Clip 范围
        gamma=1.0,               # 折扣因子 (序列级奖励不折扣)
        lam=0.95,                # GAE λ
        vf_coef=0.5,             # 价值损失权重
        kl_coef=0.1,             # KL 惩罚系数
        target_kl=0.02,          # 自适应 KL 目标
        lr=1e-5,                 # 学习率 (RLHF 阶段用较小 lr)
        max_grad_norm=0.5,
        batch_size=8,
        num_iterations=30,       # 总迭代数（增大可观察到更明显的训练效果）
    )

    print(f"\nPPO 配置:")
    print(f"  Clip ε:        {ppo_config.clip_eps}")
    print(f"  GAE λ:         {ppo_config.lam}")
    print(f"  KL 系数:       {ppo_config.kl_coef}")
    print(f"  KL 目标:       {ppo_config.target_kl}")
    print(f"  PPO Epochs:    {ppo_config.ppo_epochs}")
    print(f"  迭代数:        {ppo_config.num_iterations}")

    # 创建 PPO 训练器
    trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reward_fn=reward_fn,
        config=ppo_config,
        device=device,
    )

    # Prompt 生成器: 每次返回一批随机 prompt
    def prompt_generator():
        return torch.randint(0, 50, (ppo_config.batch_size, prompt_len))

    # 开始 PPO 训练
    print(f"\n开始 PPO 训练...")
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
    kls = [s["approx_kl"] for s in train_stats]
    print(f"\n  KL 散度:")
    print(f"    初始:   {kls[0]:.4f}")
    print(f"    最终:   {kls[-1]:.4f}")
    print(f"    最终 KL 系数: {train_stats[-1]['kl_coef']:.4f}")

    # 生成示例（PPO 后）
    trainer.policy.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, prompt_len), device=device)
        ppo_output = trainer.policy.generate(
            prompt, max_new_tokens=24, temperature=0.8
        )
        print(f"\n  PPO 后生成示例:")
        print(f"    Prompt:   {prompt[0].tolist()}")
        print(f"    Response: {ppo_output[0, prompt_len:].tolist()}")

        # 计算奖励
        reward = reward_fn(ppo_output)
        print(f"    奖励:     {reward.item():.3f}")

    # 多次生成对比
    print(f"\n  多次生成对比:")
    for trial in range(3):
        with torch.no_grad():
            prompt = torch.randint(0, 50, (1, prompt_len), device=device)
            output = trainer.policy.generate(
                prompt, max_new_tokens=24, temperature=0.8
            )
            reward = reward_fn(output)
            response = output[0, prompt_len:].tolist()
            unique_ratio = len(set(response)) / len(response) if response else 0

            # 计算连续重复数
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

    # ---- 可视化训练曲线 (ASCII) ----
    print("\n  奖励训练曲线 (ASCII):")
    _ascii_plot(rewards, width=50, height=8, label="Mean Reward")


def _ascii_plot(values: List[float], width: int = 50, height: int = 8, label: str = ""):
    """简单的 ASCII 折线图。"""
    if not values:
        return

    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v if max_v != min_v else 1.0

    # 重采样到 width 个点
    n = len(values)
    resampled = []
    for i in range(width):
        idx = int(i * (n - 1) / max(width - 1, 1))
        resampled.append(values[idx])

    # 绘制
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
