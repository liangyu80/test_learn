"""
RLVR 训练主脚本

本脚本演示完整的 RLVR (Reinforcement Learning with Verifiable Rewards) 训练流程:
    1. 预训练 (SFT): 用数学加法数据进行监督微调
    2. RLVR 训练:    用可验证奖励 (答案正确性) 优化策略

RLVR 与 GRPO/PPO 训练脚本的核心区别:
    - 奖励来源: 不使用奖励模型，而是用验证器检查答案正确性
    - 数据格式: 每条样本包含 (问题, 标准答案), 不仅仅是 prompt
    - 奖励类型: 二值 (0/1 = 错/对), 不是连续值

任务设计:
    我们使用"序列求和"作为可验证任务:
        输入:  一个 token 序列 [a, b, c, ...]
        目标:  生成的序列中包含正确的求和模式
        验证:  检查模型输出是否满足特定的数学规则

    这个任务具备 RLVR 的核心特点:
        1. 有明确的正误标准 (可验证)
        2. 可以自动生成无限训练数据
        3. 奖励信号清晰 (对就是对，错就是错)

由于是教学目的，我们使用:
    - 字节级 token (vocab_size=256)，无需 tokenizer
    - 简单的数学规则验证器
    - ~1M 参数的小模型，可在 Mac CPU/MPS 上运行

用法:
    python train.py

预计运行时间: 1-3 分钟 (CPU), ~1 分钟 (MPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Tuple

from model import GPTConfig, GPTLanguageModel, count_parameters
from rlvr_trainer import RLVRConfig, RLVRTrainer


# ==============================================================================
# 设备选择工具
# ==============================================================================

def get_device() -> torch.device:
    """
    自动选择最佳计算设备。

    优先级: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ==============================================================================
# 可验证任务: 序列模式匹配
# ==============================================================================

# 任务编码:
# 我们定义一个简单但有明确正误标准的任务:
#
# 输入 prompt: [a1, a2, a3, ..., a_k]  (k 个 token)
# 目标: 模型生成的序列中，尽可能多地包含 "升序对" (a_{t+1} = a_t + 1)
#       并且避免 "重复对" (a_{t+1} = a_t)
#
# 验证规则:
#   1. 升序比例 ≥ 阈值 → 正确 (奖励 = 1)
#   2. 否则 → 错误 (奖励 = 0)
#
# 这模拟了真实 RLVR 中数学题的 "答案验证":
#   - 答案要么对要么错，没有中间状态
#   - 验证是精确的，不依赖主观判断


class SequenceVerifier:
    """
    序列模式验证器 —— RLVR 的核心组件。

    在真实的 RLVR 应用中，验证器可以是:
        - 数学答案检查器 (exact match / symbolic equivalence)
        - 代码执行器 (运行单元测试)
        - 逻辑推理验证器 (检查推理链的每一步)

    本验证器的规则:
        检查模型生成的序列是否满足"升序模式":
        - 计算升序对 (a_{t+1} = a_t + 1) 的比例
        - 如果升序比例 ≥ threshold → 正确 (1)
        - 否则 → 错误 (0)

    为什么选择这个任务:
        1. 有明确的正误判定 (升序比例是否达标)
        2. 模型可以通过 RL 学会这个模式
        3. 简单直观，便于理解 RLVR 的工作原理
    """

    def __init__(self, threshold: float = 0.3):
        """
        参数:
            threshold: 升序对比例阈值，超过则判定为正确
                       设得太高: 初始正确率太低，学不起来
                       设得太低: 初始正确率太高，没有挑战性
                       0.3 是一个合理的起点
        """
        self.threshold = threshold

    def __call__(
        self,
        sequences: torch.Tensor,
        answers: List,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        验证一批生成序列的正确性。

        参数:
            sequences:  完整序列 (prompt + response), shape = (N, T)
            answers:    标准答案列表 (本任务中是目标升序比例阈值)
            prompt_len: prompt 长度

        返回:
            results: 二值验证结果, shape = (N,), 0=错误, 1=正确
        """
        N = sequences.shape[0]
        device = sequences.device
        results = torch.zeros(N, device=device)

        for i in range(N):
            response = sequences[i, prompt_len:].tolist()
            # 使用每个样本自己的阈值 (来自 answers)
            threshold = answers[i] if isinstance(answers[i], float) else self.threshold
            results[i] = self._verify_single(response, threshold)

        return results

    def _verify_single(self, response: List[int], threshold: float) -> float:
        """
        验证单条回复。

        返回:
            1.0 如果升序对比例 ≥ threshold
            0.0 否则
        """
        if len(response) < 2:
            return 0.0

        # 计算升序对比例
        ascending_count = 0
        for j in range(1, len(response)):
            if response[j] == response[j - 1] + 1:
                ascending_count += 1

        ascending_ratio = ascending_count / (len(response) - 1)

        # 额外惩罚: 如果过多重复，直接判错
        repeat_count = sum(
            1 for j in range(1, len(response)) if response[j] == response[j - 1]
        )
        repeat_ratio = repeat_count / (len(response) - 1)

        if repeat_ratio > 0.5:
            return 0.0

        # 二值判定: 达标 = 正确, 不达标 = 错误
        return 1.0 if ascending_ratio >= threshold else 0.0


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

    在 RLVR 中，SFT 阶段特别重要:
        - 必须让模型先学会基本的序列生成能力
        - 如果模型完全随机，RLVR 阶段几乎所有回答都是错的
        - 初始准确率至少要有 10-30% 才能有效学习

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

    # 生成训练数据: 包含升序模式的序列
    # 这样 SFT 后模型就有一定概率生成升序片段
    seq_len = 48
    data = []
    for _ in range(num_samples):
        seq = []
        pos = torch.randint(0, 200, (1,)).item()
        for j in range(seq_len):
            seq.append(pos % config.vocab_size)
            if torch.rand(1).item() < 0.7:  # 70% 概率升序
                pos += 1
            else:
                pos = torch.randint(0, 200, (1,)).item()
        data.append(seq)

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
# 问题生成器
# ==============================================================================

class ProblemGenerator:
    """
    可验证问题生成器。

    在真实 RLVR 中，问题来自:
        - GSM8K (小学数学)
        - MATH (竞赛数学)
        - MBPP/HumanEval (代码生成)
        - ARC (科学推理)

    本演示中:
        生成随机 prompt，标准答案是固定的升序比例阈值。
        模型需要学会在给定 prompt 后生成满足阈值的升序序列。
    """

    def __init__(
        self,
        batch_size: int,
        prompt_len: int,
        threshold: float = 0.3,
    ):
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.threshold = threshold

    def __call__(self) -> Tuple[torch.Tensor, List[float]]:
        """
        生成一批问题和标准答案。

        返回:
            prompts: shape = (batch_size, prompt_len)
            answers: 标准答案列表 (升序比例阈值)
        """
        # 生成随机 prompt (模拟不同的"数学题")
        prompts = torch.randint(0, 50, (self.batch_size, self.prompt_len))
        # 每个问题的"标准答案"是升序比例阈值
        answers = [self.threshold] * self.batch_size
        return prompts, answers


# ==============================================================================
# 主训练流程
# ==============================================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "RLVR 训练演示".center(48) + "║")
    print("║" + "(RL with Verifiable Rewards)".center(48) + "║")
    print("╚" + "═" * 58 + "╝")

    # ---- 设备选择 ----
    device = get_device()
    print(f"\n使用设备: {device}")

    # ---- 随机种子 ----
    torch.manual_seed(42)

    # ---- 模型配置 ----
    config = GPTConfig(
        vocab_size=256,
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

    # ---- 创建策略模型 ----
    policy_model = GPTLanguageModel(config).to(device)

    print(f"\n参数量:")
    print(f"  策略模型: {count_parameters(policy_model):,}")
    print(f"  (RLVR 无需 Critic 和 Reward Model!)")

    # ==========================
    # 阶段 1: SFT 预训练
    # ==========================
    sft_losses = pretrain_sft(
        policy_model, config, device=device, num_samples=200, epochs=10, lr=3e-4
    )

    # 生成示例（SFT 后）
    policy_model.eval()
    prompt_len = 8
    verifier = SequenceVerifier(threshold=0.3)

    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, prompt_len), device=device)
        sft_output = policy_model.generate(prompt, max_new_tokens=24, temperature=0.8)
        response = sft_output[0, prompt_len:].tolist()
        # 计算升序比例
        asc = sum(1 for j in range(1, len(response)) if response[j] == response[j-1]+1)
        asc_ratio = asc / (len(response) - 1) if len(response) > 1 else 0

        print(f"\n  SFT 后生成示例:")
        print(f"    Prompt:   {prompt[0].tolist()}")
        print(f"    Response: {response}")
        print(f"    升序比例: {asc_ratio:.2%} (阈值: 30%)")
        print(f"    验证结果: {'✓ 正确' if asc_ratio >= 0.3 else '✗ 错误'}")

    # ==========================
    # 阶段 2: RLVR 训练
    # ==========================
    print("\n" + "=" * 60)
    print("阶段 2: RLVR 强化学习优化")
    print("=" * 60)

    rlvr_config = RLVRConfig(
        max_gen_len=24,
        temperature=0.8,
        group_size=4,
        correct_bonus=1.0,
        incorrect_penalty=0.0,
        rlvr_epochs=1,
        clip_eps=0.2,
        kl_coef=0.04,
        lr=1e-5,
        max_grad_norm=1.0,
        batch_size=4,
        num_iterations=30,
    )

    print(f"\nRLVR 配置:")
    print(f"  Group Size:       {rlvr_config.group_size}")
    print(f"  Correct Bonus:    {rlvr_config.correct_bonus}")
    print(f"  Incorrect Penalty:{rlvr_config.incorrect_penalty}")
    print(f"  Clip ε:           {rlvr_config.clip_eps}")
    print(f"  KL 系数:          {rlvr_config.kl_coef}")
    print(f"  迭代数:           {rlvr_config.num_iterations}")
    print(f"  每轮采样量:       {rlvr_config.batch_size} × {rlvr_config.group_size} = {rlvr_config.batch_size * rlvr_config.group_size} 条序列")

    # 创建 RLVR 训练器
    trainer = RLVRTrainer(
        policy_model=policy_model,
        verifier=verifier,
        config=rlvr_config,
        device=device,
    )

    # 问题生成器
    problem_gen = ProblemGenerator(
        batch_size=rlvr_config.batch_size,
        prompt_len=prompt_len,
        threshold=0.3,
    )

    # 开始 RLVR 训练
    print(f"\n开始 RLVR 训练...")
    start_time = time.time()
    train_stats = trainer.train(problem_gen)
    elapsed = time.time() - start_time
    print(f"\n  训练完成! 耗时: {elapsed:.1f}s")

    # ==========================
    # 训练结果分析
    # ==========================
    print("\n" + "=" * 60)
    print("训练结果分析")
    print("=" * 60)

    # 准确率变化 (RLVR 最核心的指标)
    accuracies = [s["accuracy"] for s in train_stats]
    print(f"\n  准确率变化 (RLVR 核心指标):")
    print(f"    初始:   {accuracies[0]:.2%}")
    print(f"    最终:   {accuracies[-1]:.2%}")
    print(f"    最高:   {max(accuracies):.2%}")
    print(f"    变化:   {accuracies[-1] - accuracies[0]:+.2%}")

    # 混合组比例 (衡量学习信号强度)
    mixed_ratios = [s["mixed_group_ratio"] for s in train_stats]
    print(f"\n  混合组比例 (有效学习信号):")
    print(f"    初始:   {mixed_ratios[0]:.2%}")
    print(f"    最终:   {mixed_ratios[-1]:.2%}")
    print(f"    (混合组 = 组内既有对又有错的组, 才能提供学习信号)")

    # KL 散度
    kls = [s["kl_divergence"] for s in train_stats]
    print(f"\n  KL 散度:")
    print(f"    初始:   {kls[0]:.4f}")
    print(f"    最终:   {kls[-1]:.4f}")

    # 奖励变化
    rewards = [s["mean_reward"] for s in train_stats]
    print(f"\n  平均奖励:")
    print(f"    初始:   {rewards[0]:.3f}")
    print(f"    最终:   {rewards[-1]:.3f}")

    # 生成示例（RLVR 后）
    trainer.policy.eval()
    print(f"\n  RLVR 后生成与验证:")
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, prompt_len), device=device)
        rlvr_output = trainer.policy.generate(
            prompt, max_new_tokens=24, temperature=0.8
        )
        response = rlvr_output[0, prompt_len:].tolist()
        asc = sum(1 for j in range(1, len(response)) if response[j] == response[j-1]+1)
        asc_ratio = asc / (len(response) - 1) if len(response) > 1 else 0

        print(f"    Prompt:   {prompt[0].tolist()}")
        print(f"    Response: {response}")
        print(f"    升序比例: {asc_ratio:.2%}")
        print(f"    验证结果: {'✓ 正确' if asc_ratio >= 0.3 else '✗ 错误'}")

    # 组采样 + 验证演示
    print(f"\n  RLVR 组采样与验证演示:")
    print(f"  对同一个问题生成 {rlvr_config.group_size} 条回答并验证:")
    with torch.no_grad():
        prompt = torch.randint(0, 50, (1, prompt_len), device=device)
        print(f"    Prompt: {prompt[0].tolist()}")
        print()

        group_results = []
        for g in range(rlvr_config.group_size):
            output = trainer.policy.generate(
                prompt, max_new_tokens=24, temperature=0.8
            )
            response = output[0, prompt_len:].tolist()
            asc = sum(1 for j in range(1, len(response)) if response[j] == response[j-1]+1)
            asc_ratio = asc / (len(response) - 1) if len(response) > 1 else 0
            is_correct = asc_ratio >= 0.3

            group_results.append(is_correct)

            print(
                f"    回答 {g + 1}: "
                f"{'✓' if is_correct else '✗'} "
                f"升序={asc_ratio:.0%}, "
                f"Tokens={response[:12]}..."
            )

        correct_count = sum(group_results)
        print(f"\n    组内正确率: {correct_count}/{rlvr_config.group_size}")
        is_mixed = 0 < correct_count < rlvr_config.group_size
        print(f"    是否为混合组: {'是 (有效学习信号)' if is_mixed else '否 (无学习信号)'}")

    # 多次生成对比
    print(f"\n  多次独立生成验证:")
    correct_total = 0
    n_trials = 10
    for trial in range(n_trials):
        with torch.no_grad():
            prompt = torch.randint(0, 50, (1, prompt_len), device=device)
            output = trainer.policy.generate(
                prompt, max_new_tokens=24, temperature=0.8
            )
            response = output[0, prompt_len:].tolist()
            asc = sum(1 for j in range(1, len(response)) if response[j] == response[j-1]+1)
            asc_ratio = asc / (len(response) - 1) if len(response) > 1 else 0
            is_correct = asc_ratio >= 0.3
            correct_total += int(is_correct)

            if trial < 5:  # 只打印前 5 个
                print(
                    f"    Trial {trial + 1:2d}: "
                    f"{'✓' if is_correct else '✗'} "
                    f"升序={asc_ratio:.0%}, "
                    f"Tokens={response[:12]}..."
                )

    print(f"    ...")
    print(f"    总准确率: {correct_total}/{n_trials} = {correct_total/n_trials:.0%}")

    # ---- RLVR vs GRPO vs PPO 对比总结 ----
    print("\n" + "=" * 60)
    print("RLVR vs GRPO vs PPO 关键对比")
    print("=" * 60)
    comparison = """
    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │     维度         │      PPO        │      GRPO       │      RLVR       │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ 奖励来源         │ 学习的 RM       │ 学习的 RM       │ 可验证规则       │
    │ 奖励类型         │ 连续值          │ 连续值          │ 二值 {0, 1}     │
    │ 模型数量         │ 3 (A+C+R)      │ 2 (A+R)        │ 2 (A+R)        │
    │ 需要 RM 训练     │ 是              │ 是              │ 否              │
    │ Reward Hacking  │ 容易            │ 容易            │ 不可能          │
    │ 适用场景         │ 通用对齐        │ 通用对齐        │ 数学/代码/推理  │
    │ 数据可扩展性     │ 受限于人工标注  │ 受限于人工标注  │ 无限自动生成    │
    │ 代表性工作       │ InstructGPT     │ DeepSeek-R1     │ DeepSeek-R1     │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘

    RLVR 的核心洞察:
        "与其训练一个不完美的奖励模型来猜测答案质量，
         不如直接验证答案的正确性 —— 当验证是可行的时候。"

    这限制了 RLVR 的适用范围 (必须有可验证的标准),
    但在适用的场景中, RLVR 比 RLHF 更高效、更可靠:
        - 数学: 答案可以精确验证
        - 代码: 可以运行单元测试
        - 推理: 可以检查逻辑链
"""
    print(comparison)

    # ---- 可视化训练曲线 (ASCII) ----
    print("  准确率训练曲线 (ASCII):")
    _ascii_plot(accuracies, width=50, height=8, label="Accuracy")

    print("\n  奖励训练曲线 (ASCII):")
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
