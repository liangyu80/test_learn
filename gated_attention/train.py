"""
门控注意力 Transformer 训练演示

本脚本演示:
    1. 三种注意力模式的对比训练 (GAU vs Sigmoid-Gated vs Standard)
    2. 字符级语言模型的训练过程
    3. 门控机制对注意力稀疏性的影响分析

任务设计:
    字符级语言模型 — 在合成文本数据上训练，预测下一个字符。
    这个任务简单但足够展示门控注意力的特性:
        - 门控如何学会"忽略"不相关的上下文
        - 不同模式在相同参数预算下的表现差异

用法:
    cd gated_attention && python train.py

预计运行时间: 2-5 分钟 (CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import List, Tuple, Dict

from model import GatedAttnConfig, GatedAttentionTransformer, count_parameters


# ==============================================================================
# 设备选择
# ==============================================================================

def get_device() -> torch.device:
    """自动选择最佳设备 (MPS > CUDA > CPU)。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ==============================================================================
# 数据生成
# ==============================================================================

def generate_synthetic_text(n_samples: int, seq_len: int, vocab_size: int = 256) -> torch.Tensor:
    """
    生成合成文本数据 (带有可学习模式)。

    数据特点:
        - 包含重复模式 (便于模型学习规律)
        - 有局部依赖 (相邻字符有关联)
        - 有长距离依赖 (周期性模式重复)

    这比纯随机数据更有意义:
        - 纯随机: 交叉熵损失 = log(vocab_size) ≈ 5.55 (对于 256)
        - 有模式: 损失应该远低于此

    参数:
        n_samples: 样本数
        seq_len:   序列长度
        vocab_size: 词表大小

    返回:
        data: shape = (n_samples, seq_len), dtype = long
    """
    data = []
    for _ in range(n_samples):
        seq = []
        # 生成几个基础模式
        n_patterns = torch.randint(3, 8, (1,)).item()
        patterns = []
        for _ in range(n_patterns):
            pat_len = torch.randint(3, 12, (1,)).item()
            pat = torch.randint(0, vocab_size, (pat_len,))
            patterns.append(pat)

        # 通过重复和拼接模式构建序列
        while len(seq) < seq_len:
            # 随机选一个模式
            pat = patterns[torch.randint(0, len(patterns), (1,)).item()]
            # 有时加点噪声
            if torch.rand(1).item() < 0.3:
                noise = torch.randint(0, vocab_size, (torch.randint(1, 5, (1,)).item(),))
                seq.extend(noise.tolist())
            seq.extend(pat.tolist())

        seq = seq[:seq_len]
        data.append(torch.tensor(seq, dtype=torch.long))

    return torch.stack(data)


def generate_copy_task_data(
    n_samples: int,
    seq_len: int,
    pattern_len: int = 8,
    vocab_size: int = 256,
) -> torch.Tensor:
    """
    生成"复制回忆"任务数据。

    数据结构:
        [模式 A] [分隔符] [填充...] [分隔符] [模式 A 的副本]

    这个任务测试模型是否能:
        1. 记住远处出现的模式
        2. 在合适的时候回忆并复制

    门控注意力的优势:
        - 门控可以学会在"填充"区域关闭注意力 (节省容量)
        - 在"回忆"区域打开注意力 (利用记忆)

    参数:
        n_samples:   样本数
        seq_len:     序列长度
        pattern_len: 模式长度
        vocab_size:  词表大小

    返回:
        data: shape = (n_samples, seq_len), dtype = long
    """
    SEP = 0  # 分隔符 token
    data = []

    for _ in range(n_samples):
        # 生成模式 (避免使用分隔符 token)
        pattern = torch.randint(1, vocab_size, (pattern_len,))

        # 填充长度
        fill_len = seq_len - 2 * pattern_len - 2  # 减去两个分隔符
        fill = torch.randint(1, vocab_size, (max(fill_len, 1),))

        # 拼接: [pattern] [SEP] [fill...] [SEP] [pattern]
        seq = torch.cat([
            pattern,
            torch.tensor([SEP]),
            fill,
            torch.tensor([SEP]),
            pattern,
        ])

        # 截断或填充到 seq_len
        if len(seq) >= seq_len:
            seq = seq[:seq_len]
        else:
            padding = torch.randint(1, vocab_size, (seq_len - len(seq),))
            seq = torch.cat([seq, padding])

        data.append(seq)

    return torch.stack(data)


# ==============================================================================
# 训练逻辑
# ==============================================================================

def train_model(
    model: GatedAttentionTransformer,
    train_data: torch.Tensor,
    epochs: int = 15,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """
    训练门控注意力 Transformer。

    参数:
        model:      模型
        train_data: 训练数据, shape = (n_samples, seq_len)
        epochs:     训练轮数
        lr:         学习率
        batch_size: 批大小
        device:     计算设备

    返回:
        losses: 每个 epoch 的平均损失
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # 余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    n_samples = train_data.shape[0]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_samples, device=device)

        for start in range(0, n_samples, batch_size):
            idx = perm[start: start + batch_size]
            batch = train_data[idx]

            # 输入和目标 (自回归: 输入是前 T-1 个 token, 目标是后 T-1 个 token)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            _, loss = model(input_ids, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch + 1:3d}/{epochs}, Loss: {avg_loss:.4f}, LR: {lr_now:.6f}")

    return losses


# ==============================================================================
# 门控分析
# ==============================================================================

@torch.no_grad()
def analyze_gate_sparsity(model: GatedAttentionTransformer, data: torch.Tensor) -> Dict:
    """
    分析门控的稀疏性。

    对于 Sigmoid-Gated 模式:
        统计门控值的分布，了解多少信息被"关闭"了。

    对于 GAU 模式:
        分析 relu² 注意力权重的稀疏性。

    参数:
        model: 训练好的模型
        data:  评估数据

    返回:
        分析结果字典
    """
    model.eval()
    mode = model.config.attn_mode
    results = {}

    if mode == "sigmoid_gated":
        # 收集所有层的门控值
        all_gates = []

        # Hook 收集门控值
        hooks = []
        gate_values = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                x = input[0]
                x_norm = module.norm1(x)
                gate = torch.sigmoid(module.W_gate(x_norm))
                gate_values.append(gate.cpu())
            return hook_fn

        for i, layer in enumerate(model.layers):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        # 前向传播
        input_ids = data[:, :-1]
        model(input_ids)

        # 清理 hooks
        for h in hooks:
            h.remove()

        if gate_values:
            all_gates = torch.cat(gate_values, dim=0)
            results["mean_gate"] = all_gates.mean().item()
            results["std_gate"] = all_gates.std().item()
            # 稀疏性: 接近 0 的门控比例 (阈值 0.1)
            results["sparsity_01"] = (all_gates < 0.1).float().mean().item()
            results["sparsity_05"] = (all_gates < 0.5).float().mean().item()

    elif mode == "gau":
        # 对于 GAU, 分析 relu² 注意力的稀疏性
        attn_values = []

        def make_gau_hook(layer_idx):
            def hook_fn(module, input, output):
                x = input[0]
                x_norm = module.norm(x)
                u = F.silu(module.W_U(x_norm))
                v = F.silu(module.W_V(x_norm))
                q = module.W_Q(u)
                k = module.W_K(v)
                attn = torch.matmul(q, k.transpose(-2, -1)) * module.scale
                attn = F.relu(attn) ** 2
                attn_values.append(attn.cpu())
            return hook_fn

        hooks = []
        for i, layer in enumerate(model.layers):
            h = layer.register_forward_hook(make_gau_hook(i))
            hooks.append(h)

        input_ids = data[:, :-1]
        model(input_ids)

        for h in hooks:
            h.remove()

        if attn_values:
            all_attn = torch.cat(attn_values, dim=0)
            results["mean_attn"] = all_attn.mean().item()
            # relu² 稀疏性: 值为 0 的注意力权重比例
            results["zero_attn_ratio"] = (all_attn == 0).float().mean().item()
            results["near_zero_ratio"] = (all_attn < 1e-6).float().mean().item()

    return results


# ==============================================================================
# ASCII 可视化
# ==============================================================================

def ascii_plot(values: List[float], width: int = 50, height: int = 8, label: str = ""):
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
    print(f"    {max_v:8.4f} ┤")
    for row in range(height - 2):
        threshold = max_v - (row + 1) * range_v / (height - 1)
        line = "    " + " " * 9 + "│"
        for v in resampled:
            if v >= threshold:
                line += "█"
            else:
                line += " "
        print(line)
    print(f"    {min_v:8.4f} ┤" + "─" * width)
    print(f"    " + " " * 9 + "0" + " " * (width - 5) + f"{len(values)}")


# ==============================================================================
# 主训练流程
# ==============================================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "门控注意力 Transformer 训练演示".center(38) + "║")
    print("║" + "(GAU vs Sigmoid-Gated vs Standard)".center(46) + "║")
    print("╚" + "═" * 58 + "╝")

    # ---- 设备选择 ----
    device = get_device()
    print(f"\n使用设备: {device}")
    torch.manual_seed(42)

    # ---- 生成数据 ----
    print("\n" + "=" * 60)
    print("数据准备")
    print("=" * 60)

    n_train = 500
    seq_len = 65  # 64 + 1 (输入 64, 目标偏移 1)
    vocab_size = 64  # 小词表, 便于快速训练

    train_data = generate_synthetic_text(n_train, seq_len, vocab_size).to(device)
    eval_data = generate_synthetic_text(50, seq_len, vocab_size).to(device)
    print(f"  训练集: {train_data.shape} ({n_train} 样本)")
    print(f"  评估集: {eval_data.shape}")
    print(f"  词表大小: {vocab_size}")
    print(f"  随机基线损失: {math.log(vocab_size):.4f}")

    # ---- 对比训练三种模式 ----
    modes = ["gau", "sigmoid_gated", "standard"]
    all_losses = {}
    all_models = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"训练模式: {mode}")
        print("=" * 60)

        config = GatedAttnConfig(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            d_head=32,
            n_layers=4,
            max_seq_len=64,
            expansion=2,
            dropout=0.1,
            attn_mode=mode,
        )

        model = GatedAttentionTransformer(config).to(device)
        n_params = count_parameters(model)
        print(f"  参数量: {n_params:,}")

        start_time = time.time()
        losses = train_model(
            model, train_data,
            epochs=15, lr=3e-4, batch_size=32, device=device,
        )
        elapsed = time.time() - start_time

        print(f"  训练完成! 耗时: {elapsed:.1f}s")
        print(f"  初始损失: {losses[0]:.4f}")
        print(f"  最终损失: {losses[-1]:.4f}")

        all_losses[mode] = losses
        all_models[mode] = model

    # ---- 评估对比 ----
    print(f"\n{'=' * 60}")
    print("评估对比")
    print("=" * 60)

    eval_input = eval_data[:, :-1]
    eval_target = eval_data[:, 1:]

    print(f"\n  {'模式':<20} {'训练损失':>10} {'评估损失':>10} {'参数量':>12}")
    print(f"  {'─' * 55}")

    for mode in modes:
        model = all_models[mode]
        model.eval()
        with torch.no_grad():
            _, eval_loss = model(eval_input, eval_target)

        train_loss = all_losses[mode][-1]
        n_params = count_parameters(model)
        print(f"  {mode:<20} {train_loss:10.4f} {eval_loss.item():10.4f} {n_params:12,}")

    # ---- 门控稀疏性分析 ----
    print(f"\n{'=' * 60}")
    print("门控稀疏性分析")
    print("=" * 60)

    for mode in ["sigmoid_gated", "gau"]:
        model = all_models[mode]
        results = analyze_gate_sparsity(model, eval_data)

        print(f"\n  [{mode}]")
        if mode == "sigmoid_gated":
            print(f"    门控均值:        {results.get('mean_gate', 0):.4f}")
            print(f"    门控标准差:      {results.get('std_gate', 0):.4f}")
            print(f"    稀疏性 (<0.1):   {results.get('sparsity_01', 0):.2%}")
            print(f"    稀疏性 (<0.5):   {results.get('sparsity_05', 0):.2%}")
        elif mode == "gau":
            print(f"    注意力均值:      {results.get('mean_attn', 0):.6f}")
            print(f"    零值比例:        {results.get('zero_attn_ratio', 0):.2%}")
            print(f"    近零比例:        {results.get('near_zero_ratio', 0):.2%}")

    # ---- 生成演示 ----
    print(f"\n{'=' * 60}")
    print("文本生成演示")
    print("=" * 60)

    for mode in modes:
        model = all_models[mode]
        model.eval()

        # 用一小段数据作为 prompt
        prompt = eval_data[0, :8].unsqueeze(0)
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)

        prompt_tokens = prompt[0].tolist()
        gen_tokens = generated[0, 8:].tolist()

        print(f"\n  [{mode}]")
        print(f"    Prompt:    {prompt_tokens}")
        print(f"    Generated: {gen_tokens}")

    # ---- 训练曲线对比 ----
    print(f"\n{'=' * 60}")
    print("训练曲线对比 (ASCII)")
    print("=" * 60)

    for mode in modes:
        ascii_plot(all_losses[mode], width=40, height=6, label=f"{mode}")
        print()

    # ---- 总结 ----
    print("=" * 60)
    print("门控注意力核心思想总结")
    print("=" * 60)
    summary = """
    ┌─────────────────────────────────────────────────────────────┐
    │               门控注意力 (Gated Attention)                   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  核心公式:                                                   │
    │                                                             │
    │  GAU:                                                       │
    │    U = SiLU(X·W_U)          ← 门控向量                     │
    │    V = SiLU(X·W_V)          ← 值向量                       │
    │    A = relu²(Q·K^T/√d)      ← 稀疏注意力                   │
    │    O = (U ⊙ A·V) · W_O      ← 门控 × 注意力               │
    │                                                             │
    │  Sigmoid-Gated:                                             │
    │    g = σ(X·W_g)             ← sigmoid 门控                 │
    │    A = softmax(Q·K^T/√d)    ← 标准注意力                   │
    │    O = g ⊙ A·V              ← 门控过滤                     │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  门控的三大好处:                                              │
    │                                                             │
    │  1. 稀疏化: 门控学会"关闭"不重要的注意力输出                   │
    │     → 减少 attention sink (不需要浪费注意力在无意义 token)    │
    │                                                             │
    │  2. 参数效率: GAU 将 Attention + FFN 合并                    │
    │     → 一个模块完成两个子层的工作                               │
    │                                                             │
    │  3. 动态选择: 门控根据输入动态决定哪些信息通过                  │
    │     → 比固定的 softmax 更灵活                                │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  实际应用:                                                   │
    │    - Qwen 系列模型使用 sigmoid 门控注意力                     │
    │    - Google 的 FLASH 模型使用 GAU 架构                       │
    │    - 门控注意力是现代高效 Transformer 的重要组件               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
"""
    print(summary)


if __name__ == "__main__":
    main()
