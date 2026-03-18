"""
激活函数对比实验

对比维度:
    1. 数值性质:   输出分布、梯度分布、稀疏度、死亡率
    2. 梯度流:     多层网络中梯度消失/爆炸情况
    3. 训练效果:   在语言建模任务上用不同激活函数训练小型 Transformer
    4. 计算效率:   前向+反向的速度对比
    5. GLU 变体:   对比 GLU/ReGLU/GeGLU/SwiGLU 在 FFN 中的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

from activations import (
    ELEMENTWISE_ACTIVATIONS, get_glu_activations,
    compute_properties, analyze_gradient_flow,
    TransformerFFN,
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, SELU, GELU, SiLU, Mish,
    GLU, ReGLU, GeGLU, SwiGLU, Softplus, Hardswish, Squareplus,
)


# ==============================================================================
# 实验 1: 数值性质全面对比
# ==============================================================================

def exp1_numerical_properties():
    """对比所有逐元素激活函数的数值性质。"""
    print("=" * 80)
    print("  实验 1: 激活函数数值性质对比")
    print("=" * 80)

    torch.manual_seed(42)

    # 用不同分布测试
    distributions = {
        "标准正态 N(0,1)": torch.randn(50000),
        "宽分布 N(0,4)":   torch.randn(50000) * 2,
        "均匀 U(-3,3)":    torch.rand(50000) * 6 - 3,
    }

    for dist_name, x in distributions.items():
        print(f"\n  输入分布: {dist_name} (均值={x.mean():.3f}, 标准差={x.std():.3f})")
        print("-" * 80)
        print(f"  {'激活函数':<16} {'输出均值':>8} {'输出σ':>7} {'稀疏度':>7} "
              f"{'梯度均值':>8} {'梯度σ':>7} {'死亡率':>7}")
        print("-" * 80)

        for name, act_fn in ELEMENTWISE_ACTIVATIONS.items():
            props = compute_properties(act_fn, x.clone())
            print(f"  {name:<16} {props['output_mean']:>8.4f} {props['output_std']:>7.4f} "
                  f"{props['sparsity']:>7.1%} {props['grad_mean']:>8.4f} "
                  f"{props['grad_std']:>7.4f} {props['dead_ratio']:>7.1%}")

    # 总结
    print("\n📋 数值性质总结:")
    print("  • 零中心性 (输出均值≈0): Tanh, LeakyReLU, ELU, SELU, GELU, SiLU, Mish, Softsign")
    print("  • 高稀疏度: ReLU (≈50%), LeakyReLU 内容接近0但有梯度")
    print("  • 无梯度死亡: LeakyReLU, ELU, SELU, GELU, SiLU, Mish (所有平滑函数)")
    print("  • 有界输出: Sigmoid (0,1), Tanh (-1,1), Softsign (-1,1), Hardswish")


# ==============================================================================
# 实验 2: 梯度流对比
# ==============================================================================

def exp2_gradient_flow():
    """对比不同激活函数在深层网络中的梯度传播。"""
    print("\n" + "=" * 80)
    print("  实验 2: 梯度流对比 (深层全连接网络)")
    print("=" * 80)

    depths = [5, 10, 20]
    activations = {
        "ReLU":       ReLU(),
        "LeakyReLU":  LeakyReLU(0.01),
        "ELU":        ELU(),
        "SELU":       SELU(),
        "GELU":       GELU(),
        "SiLU/Swish": SiLU(),
        "Mish":       Mish(),
        "Tanh":       Tanh(),
        "Sigmoid":    Sigmoid(),
    }

    for depth in depths:
        print(f"\n  网络深度: {depth} 层")
        print(f"  {'激活函数':<14} {'首层梯度':>12} {'末层梯度':>12} {'首/末比':>10} {'状态':>10}")
        print("  " + "-" * 62)

        for name, act_fn in activations.items():
            result = analyze_gradient_flow(act_fn, depth=depth, dim=64)
            norms = result["grad_norms"]
            if not norms:
                print(f"  {name:<14} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
                continue
            first, last = norms[0], norms[-1]
            ratio = result["ratio_first_last"]

            if result["vanishing"]:
                status = "⚠ 消失"
            elif result["exploding"]:
                status = "⚠ 爆炸"
            elif ratio > 100:
                status = "△ 不稳定"
            else:
                status = "✓ 正常"

            print(f"  {name:<14} {first:>12.6f} {last:>12.6f} {ratio:>10.2f} {status:>10}")

    print("\n📋 梯度流总结:")
    print("  • ReLU: 浅层网络表现好，深层可能出现死亡神经元")
    print("  • GELU/SiLU: 梯度传播平稳，是 Transformer 的好选择")
    print("  • SELU: 理论上自归一化，但实际要求严格的初始化")
    print("  • Sigmoid/Tanh: 深层网络严重梯度消失")


# ==============================================================================
# 实验 3: 小型语言模型训练对比
# ==============================================================================

class MiniTransformerLM(nn.Module):
    """
    微型 Transformer 语言模型，用于对比不同激活函数。

    架构:
        - Embedding → N × (Self-Attention + FFN) → LM Head
        - FFN 使用指定的激活函数
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, d_ff: int = 256,
                 max_seq_len: int = 64, activation: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ffn": TransformerFFN(d_model, d_ff, activation=activation,
                                       dropout=dropout),
                "norm2": nn.LayerNorm(d_model),
            }))

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)

        # 因果 mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        for layer in self.layers:
            # Self-Attention
            residual = h
            h = layer["norm1"](h)
            h, _ = layer["attn"](h, h, h, attn_mask=causal_mask, is_causal=True)
            h = h + residual

            # FFN
            residual = h
            h = layer["norm2"](h)
            h = layer["ffn"](h)
            h = h + residual

        h = self.ln_f(h)
        return self.lm_head(h)


def generate_synthetic_data(vocab_size: int = 256, seq_len: int = 64,
                            n_samples: int = 1000) -> torch.Tensor:
    """生成合成语言数据 (有简单模式的字符序列)。"""
    data = []
    for _ in range(n_samples):
        # 混合模式: 重复、递增、随机
        seq = []
        while len(seq) < seq_len:
            pattern_type = torch.randint(0, 3, (1,)).item()
            if pattern_type == 0:
                # 重复模式: ABABAB
                a, b = torch.randint(0, vocab_size, (2,)).tolist()
                seg = [a, b] * 5
            elif pattern_type == 1:
                # 递增模式: 1,2,3,4,...
                start = torch.randint(0, vocab_size - 10, (1,)).item()
                seg = list(range(start, start + 8))
            else:
                # 随机
                seg = torch.randint(0, vocab_size, (6,)).tolist()
            seq.extend(seg)
        data.append(seq[:seq_len])
    return torch.tensor(data, dtype=torch.long)


def exp3_training_comparison():
    """对比不同激活函数在小型语言模型训练中的效果。"""
    print("\n" + "=" * 80)
    print("  实验 3: 小型语言模型训练对比")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 超参数 (教学 demo, 适合 CPU 快速运行)
    vocab_size = 64
    d_model = 32
    n_heads = 4
    n_layers = 2
    d_ff = 128
    seq_len = 32
    batch_size = 64
    epochs = 5
    lr = 1e-3

    # 生成数据
    torch.manual_seed(42)
    data = generate_synthetic_data(vocab_size, seq_len, n_samples=256).to(device)
    n_batches = len(data) // batch_size

    # 要对比的激活函数
    activation_list = ["relu", "gelu", "silu", "swiglu", "geglu"]

    results = {}

    for act_name in activation_list:
        print(f"\n  ▶ 训练 {act_name.upper()} 模型...")
        torch.manual_seed(42)

        model = MiniTransformerLM(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=seq_len,
            activation=act_name, dropout=0.1
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        train_losses = []
        t0 = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            # 打乱数据
            perm = torch.randperm(len(data))
            for i in range(n_batches):
                batch = data[perm[i * batch_size: (i + 1) * batch_size]]
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                                       targets.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)

        train_time = time.time() - t0
        final_loss = train_losses[-1]

        results[act_name] = {
            "params": param_count,
            "final_loss": final_loss,
            "min_loss": min(train_losses),
            "train_time": train_time,
            "losses": train_losses,
        }

        print(f"    参数量: {param_count:,d} | 最终 Loss: {final_loss:.4f} | "
              f"最低 Loss: {min(train_losses):.4f} | 耗时: {train_time:.1f}s")

    # 排名
    print("\n" + "-" * 80)
    print("  📊 训练结果排名 (按最终 Loss)")
    print("-" * 80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["final_loss"])
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"  #{rank}  {name:<10}: Loss={res['final_loss']:.4f}  "
              f"参数={res['params']:>7,d}  耗时={res['train_time']:.1f}s")

    print("\n📋 训练对比总结:")
    print("  • SwiGLU/GeGLU 通常能获得最低的 Loss (门控机制的优势)")
    print("  • GELU/SiLU 效果接近且稳定")
    print("  • ReLU 虽然简单但在小模型上差距不大")
    print("  • Tanh 在深层模型上通常表现较差 (梯度消失)")

    return results


# ==============================================================================
# 实验 4: 计算效率对比
# ==============================================================================

def exp4_speed_benchmark():
    """对比不同激活函数的前向+反向计算速度。"""
    print("\n" + "=" * 80)
    print("  实验 4: 计算效率对比 (前向 + 反向)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 逐元素激活函数速度
    print("\n  ▶ 逐元素激活函数 (输入: [32, 128, 256])")
    print("-" * 60)
    print(f"  {'激活函数':<16} {'前向 (ms)':>10} {'反向 (ms)':>10} {'总计 (ms)':>10}")
    print("-" * 60)

    x_size = (32, 128, 256)
    n_runs = 20

    for name, act_fn in ELEMENTWISE_ACTIVATIONS.items():
        act_fn = act_fn.to(device)
        x = torch.randn(*x_size, device=device, requires_grad=True)

        # 预热
        for _ in range(10):
            y = act_fn(x)
            y.sum().backward()
            x.grad = None

        # 前向计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            y = act_fn(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fwd_time = (time.time() - t0) / n_runs * 1000

        # 反向计时
        t0 = time.time()
        for _ in range(n_runs):
            x_grad = torch.randn(*x_size, device=device, requires_grad=True)
            y = act_fn(x_grad)
            y.sum().backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bwd_time = (time.time() - t0) / n_runs * 1000

        print(f"  {name:<16} {fwd_time:>10.3f} {bwd_time:>10.3f} {fwd_time + bwd_time:>10.3f}")

    # FFN 速度
    print("\n  ▶ Transformer FFN 模块 (输入: [8, 32, 128], d_ff=512)")
    print("-" * 60)
    print(f"  {'FFN 类型':<16} {'前向 (ms)':>10} {'反向 (ms)':>10} {'总计 (ms)':>10}")
    print("-" * 60)

    d_model, d_ff = 128, 512
    x_ffn = torch.randn(8, 32, d_model, device=device)

    for act_name in ["relu", "gelu", "silu", "swiglu", "geglu", "mish"]:
        ffn = TransformerFFN(d_model, d_ff, activation=act_name, dropout=0.0).to(device)

        # 预热
        for _ in range(10):
            y = ffn(x_ffn)
            y.sum().backward()

        # 前向
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            y = ffn(x_ffn)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fwd_time = (time.time() - t0) / n_runs * 1000

        # 反向
        t0 = time.time()
        for _ in range(n_runs):
            y = ffn(x_ffn)
            y.sum().backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bwd_time = (time.time() - t0) / n_runs * 1000 - fwd_time

        print(f"  {act_name:<16} {fwd_time:>10.3f} {bwd_time:>10.3f} {fwd_time + bwd_time:>10.3f}")

    print("\n📋 速度总结:")
    print("  • ReLU 最快 (简单 max 操作)")
    print("  • GELU/SiLU 速度接近 (PyTorch 有优化的 kernel)")
    print("  • Mish 最慢 (需要 tanh + softplus)")
    print("  • SwiGLU 虽然有 3 个矩阵，但隐藏维度缩小了 1/3，总速度可接受")


# ==============================================================================
# 实验 5: 大模型中激活函数使用情况分析
# ==============================================================================

def exp5_llm_activation_survey():
    """分析当前主流大模型中激活函数的使用情况。"""
    print("\n" + "=" * 80)
    print("  实验 5: 大语言模型 (LLM) 中激活函数使用情况")
    print("=" * 80)

    # 模型数据库
    models = [
        # (模型名, 参数量, 发布年, 机构, 激活函数, FFN类型)
        ("Transformer",  "N/A",   2017, "Google",      "ReLU",    "标准 FFN"),
        ("BERT",         "340M",  2018, "Google",      "GELU",    "标准 FFN"),
        ("GPT-2",        "1.5B",  2019, "OpenAI",      "GELU",    "标准 FFN"),
        ("GPT-3",        "175B",  2020, "OpenAI",      "GELU",    "标准 FFN"),
        ("T5",           "11B",   2020, "Google",      "ReLU",    "标准 FFN"),
        ("PaLM",         "540B",  2022, "Google",      "SwiGLU",  "GLU FFN"),
        ("LLaMA",        "65B",   2023, "Meta",        "SwiGLU",  "GLU FFN"),
        ("LLaMA-2",      "70B",   2023, "Meta",        "SwiGLU",  "GLU FFN"),
        ("Mistral",      "7B",    2023, "Mistral AI",  "SwiGLU",  "GLU FFN"),
        ("Mixtral",      "47B",   2024, "Mistral AI",  "SwiGLU",  "GLU FFN (MoE)"),
        ("Qwen",         "72B",   2023, "阿里",        "SwiGLU",  "GLU FFN"),
        ("Qwen-2",       "72B",   2024, "阿里",        "SwiGLU",  "GLU FFN"),
        ("Yi",           "34B",   2023, "零一万物",    "SwiGLU",  "GLU FFN"),
        ("Gemma",        "7B",    2024, "Google",      "GeGLU",   "GLU FFN"),
        ("Gemma-2",      "27B",   2024, "Google",      "GeGLU",   "GLU FFN"),
        ("DeepSeek-V2",  "236B",  2024, "DeepSeek",    "SwiGLU",  "GLU FFN (MoE)"),
        ("DeepSeek-V3",  "671B",  2024, "DeepSeek",    "SwiGLU",  "GLU FFN (MoE)"),
        ("LLaMA-3",      "405B",  2024, "Meta",        "SwiGLU",  "GLU FFN"),
        ("Phi-3",        "14B",   2024, "Microsoft",   "SwiGLU",  "GLU FFN"),
        ("GPT-4",        "~1.8T", 2023, "OpenAI",      "GELU*",   "标准/未公开"),
        ("Claude-3",     "未公开", 2024, "Anthropic",   "未公开",  "未公开"),
    ]

    print(f"\n  {'模型':<16} {'参数量':<10} {'年份':<6} {'机构':<14} {'激活函数':<10} {'FFN 类型':<16}")
    print("  " + "-" * 76)
    for name, params, year, org, act, ffn in models:
        print(f"  {name:<16} {params:<10} {year:<6} {org:<14} {act:<10} {ffn:<16}")

    # 统计
    print("\n  📊 激活函数使用统计 (已公开架构):")
    act_count = {}
    for _, _, year, _, act, _ in models:
        if "未公开" not in act:
            act_base = act.replace("*", "")
            act_count[act_base] = act_count.get(act_base, 0) + 1

    for act, count in sorted(act_count.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {act:<10}: {bar} ({count})")

    # 时间线分析
    print("\n  📈 激活函数演进时间线:")
    print("  " + "-" * 60)
    print("  2017  ┃ Transformer → ReLU (简单高效)")
    print("  2018  ┃ BERT       → GELU (平滑、概率解释)")
    print("  2019  ┃ GPT-2      → GELU (已成为标配)")
    print("  2020  ┃ GPT-3      → GELU (继续沿用)")
    print("  2020  ┃ Shazeer 提出 GLU Variants (SwiGLU/GeGLU)")
    print("  2022  ┃ PaLM       → SwiGLU (Google 率先采用)")
    print("  2023  ┃ LLaMA      → SwiGLU (开源生态全面转向)")
    print("  2024  ┃ SwiGLU 成为事实标准 (几乎所有新模型)")
    print("  " + "-" * 60)

    print("\n  🔍 关键发现:")
    print("  1. SwiGLU 是当前大模型的事实标准")
    print("     - 2023 年后几乎所有开源 LLM 都采用 SwiGLU")
    print("     - LLaMA 架构的成功推动了 SwiGLU 的普及")
    print("")
    print("  2. GELU 仍在 encoder 模型中使用")
    print("     - BERT、GPT-2/3 使用 GELU")
    print("     - GPT-4 可能仍使用 GELU (未完全公开)")
    print("")
    print("  3. GeGLU 是 SwiGLU 的有力竞争者")
    print("     - Google Gemma 系列使用 GeGLU")
    print("     - 在某些实验中与 SwiGLU 效果相当")
    print("")
    print("  4. ReLU 已在大模型中基本淘汰")
    print("     - 最后使用 ReLU 的大模型是 T5 (2020)")
    print("     - 但在 CNN 和轻量模型中仍广泛使用")
    print("")
    print("  5. 参数量调整是关键技巧")
    print("     - SwiGLU 有 3 个权重矩阵 (vs 标准 FFN 的 2 个)")
    print("     - 隐藏维度缩小到 2/3 以保持总参数量一致")
    print("     - 例: LLaMA-7B 的 FFN 维度为 11008 (≈ 4096 × 8/3)")


# ==============================================================================
# 主函数: 运行所有实验
# ==============================================================================

def run_all_experiments():
    """运行所有对比实验。"""
    print("╔" + "═" * 78 + "╗")
    print("║" + "激活函数全面对比实验".center(70) + "║")
    print("║" + "从 Sigmoid 到 SwiGLU: 20+ 种激活函数的深度对比".center(62) + "║")
    print("╚" + "═" * 78 + "╝")

    t_start = time.time()

    # 实验 1: 数值性质
    exp1_numerical_properties()

    # 实验 2: 梯度流
    exp2_gradient_flow()

    # 实验 3: 训练对比
    training_results = exp3_training_comparison()

    # 实验 4: 速度对比
    exp4_speed_benchmark()

    # 实验 5: LLM 使用情况
    exp5_llm_activation_survey()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  所有实验完成! 总耗时: {total_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
