"""
投机采样 vs 多 Token 预测 —— 对比实验

本脚本对两种方法进行端到端比较:
    1. 训练各自的模型
    2. 使用不同策略生成文本
    3. 对比推理效率（前向传播次数、接受率等）
    4. 可视化对比结果

用法:
    python compare.py
"""

import time
import torch
import torch.nn as nn

from speculative_decoding import (
    SimpleTransformerLM,
    SpeculativeDecoder,
    train_standard_lm,
)
from multitoken_prediction import (
    MultiTokenTransformerLM,
    MultiTokenDecoder,
    train_multitoken_lm,
)


def measure_autoregressive_baseline(
    model: SimpleTransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> tuple:
    """
    标准自回归解码基线。

    这是最基本的解码方式：每次前向传播只生成一个 token。
    用作性能对比的基线。

    参数:
        model:          语言模型
        prompt:         输入 prompt
        max_new_tokens: 最大生成 token 数
        temperature:    采样温度

    返回:
        output_ids: 生成的序列
        stats:      统计信息
    """
    current_ids = prompt.clone()
    initial_len = prompt.shape[1]

    stats = {
        "method": "自回归基线 (Autoregressive Baseline)",
        "num_forward_passes": 0,
        "total_tokens_generated": 0,
    }

    start_time = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            probs = model.get_next_token_probs(current_ids, temperature)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            stats["num_forward_passes"] += 1
            stats["total_tokens_generated"] += 1

    stats["wall_time"] = time.time() - start_time
    stats["tokens_per_forward"] = 1.0  # 始终为 1

    return current_ids, stats


def run_comparison():
    """运行完整的对比实验。"""

    print("╔" + "═" * 68 + "╗")
    print("║" + "投机采样 vs 多 Token 预测 —— 对比实验".center(50) + "║")
    print("╚" + "═" * 68 + "╝")

    # ===========================
    # 实验设置
    # ===========================
    torch.manual_seed(42)
    device = torch.device("cpu")

    VOCAB_SIZE = 500
    SEQ_LEN = 32
    MAX_NEW_TOKENS = 30
    TRAIN_EPOCHS = 5
    N_TRAIN_SAMPLES = 80

    print("\n实验参数:")
    print(f"  词表大小:       {VOCAB_SIZE}")
    print(f"  训练序列长度:   {SEQ_LEN}")
    print(f"  生成 token 数:  {MAX_NEW_TOKENS}")
    print(f"  训练样本数:     {N_TRAIN_SAMPLES}")
    print(f"  训练轮数:       {TRAIN_EPOCHS}")

    # 生成训练数据
    train_data = torch.randint(0, VOCAB_SIZE, (N_TRAIN_SAMPLES, SEQ_LEN), device=device)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 5), device=device)

    all_stats = []

    # ===========================
    # 方法 1: 标准自回归 (基线)
    # ===========================
    print("\n" + "=" * 70)
    print("方法 1: 标准自回归解码 (基线)")
    print("=" * 70)
    print("  模型: 4层, 256维 Transformer")

    baseline_model = SimpleTransformerLM(
        vocab_size=VOCAB_SIZE, d_model=256, n_heads=4, n_layers=4
    ).to(device)

    print("\n  训练中...")
    train_standard_lm(baseline_model, train_data, epochs=TRAIN_EPOCHS, lr=1e-3)

    print("\n  生成中...")
    output_baseline, stats_baseline = measure_autoregressive_baseline(
        baseline_model, prompt, max_new_tokens=MAX_NEW_TOKENS
    )
    all_stats.append(stats_baseline)

    print(f"  生成序列长度: {output_baseline.shape[1]}")
    print(f"  前向传播次数: {stats_baseline['num_forward_passes']}")
    print(f"  每次前向传播生成 token 数: {stats_baseline['tokens_per_forward']:.1f}")

    # ===========================
    # 方法 2: 投机采样
    # ===========================
    print("\n" + "=" * 70)
    print("方法 2: 投机采样 (Speculative Decoding)")
    print("=" * 70)

    K = 4  # 投机长度
    print(f"  草稿模型: 2层, 128维 | 目标模型: 4层, 256维 | 投机长度 K={K}")

    draft_model = SimpleTransformerLM(
        vocab_size=VOCAB_SIZE, d_model=128, n_heads=4, n_layers=2
    ).to(device)

    # 目标模型复用基线模型（它们架构相同）
    target_model = baseline_model

    print("\n  训练草稿模型...")
    train_standard_lm(draft_model, train_data, epochs=TRAIN_EPOCHS, lr=1e-3)

    print("\n  投机采样生成中...")
    spec_decoder = SpeculativeDecoder(draft_model, target_model, K=K)

    start_time = time.time()
    output_spec, stats_spec = spec_decoder.generate(
        prompt, max_new_tokens=MAX_NEW_TOKENS
    )
    stats_spec["wall_time"] = time.time() - start_time
    stats_spec["method"] = "投机采样 (Speculative Decoding)"
    all_stats.append(stats_spec)

    print(f"  生成序列长度:     {output_spec.shape[1]}")
    print(f"  投机迭代次数:     {stats_spec['num_iterations']}")
    print(f"  目标模型调用次数: {stats_spec['target_model_calls']}")
    print(f"  草稿模型调用次数: {stats_spec['draft_model_calls']}")
    print(f"  接受率:           {stats_spec['acceptance_rate']:.2%}")

    # ===========================
    # 方法 3: 多 Token 预测 (贪心并行)
    # ===========================
    print("\n" + "=" * 70)
    print("方法 3: 多 Token 预测 — 贪心并行解码")
    print("=" * 70)

    N_PREDICT = 4
    print(f"  模型: 4层, 256维 | 预测头数: {N_PREDICT}")

    mt_model = MultiTokenTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_heads=4,
        n_layers=4,
        n_predict=N_PREDICT,
    ).to(device)

    print("\n  训练中...")
    train_multitoken_lm(
        mt_model,
        train_data,
        epochs=TRAIN_EPOCHS,
        lr=1e-3,
        loss_weights=[0.4, 0.3, 0.2, 0.1],
    )

    print("\n  贪心并行解码生成中...")
    mt_decoder_greedy = MultiTokenDecoder(mt_model, strategy="greedy_parallel")

    start_time = time.time()
    output_mt_greedy, stats_mt_greedy = mt_decoder_greedy.generate(
        prompt, max_new_tokens=MAX_NEW_TOKENS
    )
    stats_mt_greedy["wall_time"] = time.time() - start_time
    stats_mt_greedy["method"] = "多Token预测 — 贪心并行 (Greedy Parallel)"
    all_stats.append(stats_mt_greedy)

    print(f"  生成序列长度:   {output_mt_greedy.shape[1]}")
    print(f"  前向传播次数:   {stats_mt_greedy['num_forward_passes']}")
    print(f"  平均每步 token: {stats_mt_greedy.get('avg_tokens_per_step', 'N/A')}")

    # ===========================
    # 方法 4: 多 Token 预测 (自验证)
    # ===========================
    print("\n" + "=" * 70)
    print("方法 4: 多 Token 预测 — 自验证解码")
    print("=" * 70)
    print(f"  复用上面训练好的多 Token 预测模型")

    mt_decoder_spec = MultiTokenDecoder(mt_model, strategy="self_speculative")

    start_time = time.time()
    output_mt_spec, stats_mt_spec = mt_decoder_spec.generate(
        prompt, max_new_tokens=MAX_NEW_TOKENS
    )
    stats_mt_spec["wall_time"] = time.time() - start_time
    stats_mt_spec["method"] = "多Token预测 — 自验证 (Self-Speculative)"
    all_stats.append(stats_mt_spec)

    print(f"  生成序列长度:   {output_mt_spec.shape[1]}")
    print(f"  前向传播次数:   {stats_mt_spec['num_forward_passes']}")
    print(f"  草稿接受数:     {stats_mt_spec['accepted_from_draft']}")
    print(f"  草稿拒绝数:     {stats_mt_spec['rejected_from_draft']}")

    # ===========================
    # 汇总对比
    # ===========================
    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)

    print(f"\n{'方法':<40} {'前向传播次数':>12} {'生成token数':>12}")
    print("-" * 70)

    for s in all_stats:
        method = s["method"]
        if "target_model_calls" in s:
            # 投机采样: 目标模型调用是主要瓶颈
            fwd = f"{s['target_model_calls']}(大)+{s['draft_model_calls']}(小)"
        else:
            fwd = str(s["num_forward_passes"])
        tokens = str(s["total_tokens_generated"])
        print(f"  {method:<38} {fwd:>12} {tokens:>12}")

    # ===========================
    # 关键差异分析
    # ===========================
    print("\n" + "=" * 70)
    print("关键差异分析")
    print("=" * 70)

    analysis = """
┌─────────────────┬────────────────────────────┬────────────────────────────┐
│     维度         │     投机采样               │     多 Token 预测           │
│                 │ (Speculative Decoding)     │ (Multi-Token Prediction)   │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 核心思路        │ 用小模型猜测，大模型验证   │ 改变训练目标，同时预测多个  │
│                 │ (推理阶段的优化)           │ token (训练+推理都改变)    │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 是否需要额外    │ 是（需要一个草稿模型）     │ 否（仅增加预测头，模型      │
│ 模型            │                            │ 自包含）                   │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 训练方式        │ 草稿/目标模型独立训练      │ 统一训练，多头损失函数      │
│                 │ 标准 next-token prediction  │ 联合优化                   │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 损失函数        │ 标准交叉熵 (每个模型独立)  │ 多头加权交叉熵:             │
│                 │ L = -log P(x_{t+1}|x_≤t)  │ L = Σ λ_k * CE(head_k)    │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 输出质量        │ 与目标模型完全一致         │ 贪心并行可能有质量损失；    │
│                 │ (数学上无损)               │ 自验证模式接近无损          │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 加速原理        │ 大模型并行验证多个 token   │ 一次前向传播输出多个 token  │
│                 │ (减少大模型调用次数)       │ (减少总前向传播次数)        │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 典型加速比      │ 2x - 3x                   │ 2x - 4x (取决于接受率)     │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 模型质量提升    │ 无 (纯推理优化)            │ 有 (多任务学习带来更好的    │
│                 │                            │ 内部表示)                  │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 额外显存开销    │ 需要同时加载两个模型       │ 仅增加预测头参数            │
│                 │                            │ (通常 < 5% 额外参数)       │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 适用场景        │ 已有大小模型对，不想重新   │ 从头训练或微调，希望同时    │
│                 │ 训练，只想加速推理         │ 提升质量和速度              │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ 组合使用        │ 可以将多Token预测模型作为   │ 可以用自身的多预测头进行    │
│                 │ 投机采样的目标模型         │ "自投机采样"               │
└─────────────────┴────────────────────────────┴────────────────────────────┘
"""
    print(analysis)


if __name__ == "__main__":
    run_comparison()
