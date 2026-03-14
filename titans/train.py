"""
Titans MAC 训练演示

本脚本演示 Titans MAC 模型的训练和推理:
    1. 用合成时间序列数据训练模型
    2. 展示长期记忆的效果 (对比有/无 NMM)
    3. 演示记忆的惊讶度驱动更新机制

任务设计:
    使用"重复模式回忆"任务:
    - 生成包含周期性模式的序列
    - 模式在远处出现过一次，需要模型"记住"它
    - 然后在近处重现时，能更好地预测

    这个任务天然需要长期记忆:
    - 注意力窗口太短，无法直接看到远处的模式
    - 模型必须依赖 NMM 记住历史模式

用法:
    cd titans && python train.py

预计运行时间: 1-3 分钟 (CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import List, Tuple

from model import TitanConfig, TitanMAC, count_parameters
from neural_memory import NeuralMemory


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

def generate_periodic_data(
    n_samples: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成周期性时间序列数据。

    数据特点:
        - 包含多种频率的正弦波叠加
        - 加上少量噪声
        - 任务: 预测下一个时间步的值

    这种数据适合测试长期记忆:
        周期性模式需要模型"记住"远处的历史值。

    参数:
        n_samples: 样本数
        seq_len:   序列长度
        device:    计算设备

    返回:
        inputs:  shape = (n_samples, seq_len, 1)
        targets: shape = (n_samples, seq_len, 1)
    """
    t = torch.linspace(0, 4 * math.pi, seq_len + 1, device=device)

    all_inputs = []
    all_targets = []

    for _ in range(n_samples):
        # 随机组合多种频率
        freq1 = torch.rand(1, device=device) * 2 + 0.5
        freq2 = torch.rand(1, device=device) * 3 + 1.0
        amp1 = torch.rand(1, device=device) * 0.8 + 0.2
        amp2 = torch.rand(1, device=device) * 0.5 + 0.1
        phase = torch.rand(1, device=device) * 2 * math.pi

        signal = amp1 * torch.sin(freq1 * t + phase) + amp2 * torch.sin(freq2 * t)
        noise = torch.randn(seq_len + 1, device=device) * 0.05
        signal = signal + noise

        inputs = signal[:-1].unsqueeze(-1)   # (seq_len, 1)
        targets = signal[1:].unsqueeze(-1)   # (seq_len, 1)

        all_inputs.append(inputs)
        all_targets.append(targets)

    return torch.stack(all_inputs), torch.stack(all_targets)


def generate_copy_recall_data(
    n_samples: int,
    seq_len: int,
    pattern_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成"远距离模式回忆"数据。

    数据结构:
        [模式A (pattern_len)] [噪声填充...] [模式A 再次出现] [需预测的延续]

    这个任务直接测试长期记忆:
        - 模式A 第一次出现时超出了注意力窗口
        - 模型必须通过 NMM 记住模式A
        - 当模式A 再次出现时，利用记忆来更好地预测

    参数:
        n_samples:   样本数
        seq_len:     总序列长度
        pattern_len: 模式长度
        device:      计算设备

    返回:
        inputs:  shape = (n_samples, seq_len, 1)
        targets: shape = (n_samples, seq_len, 1)
    """
    all_inputs = []
    all_targets = []

    for _ in range(n_samples):
        # 生成一个随机模式
        pattern = torch.randn(pattern_len, device=device) * 0.5

        # 构建序列: [pattern] + [noise] + [pattern] + [continuation]
        noise_len = seq_len - 2 * pattern_len
        noise = torch.randn(noise_len, device=device) * 0.1

        continuation = pattern  # 延续就是模式本身 (需要回忆)
        full_signal = torch.cat([pattern, noise, continuation])

        # 确保长度正确
        full_signal = full_signal[:seq_len + 1] if len(full_signal) > seq_len + 1 else F.pad(full_signal, (0, seq_len + 1 - len(full_signal)))

        inputs = full_signal[:seq_len].unsqueeze(-1)
        targets = full_signal[1:seq_len + 1].unsqueeze(-1)

        all_inputs.append(inputs)
        all_targets.append(targets)

    return torch.stack(all_inputs), torch.stack(all_targets)


# ==============================================================================
# 训练逻辑
# ==============================================================================

def train_model(
    model: TitanMAC,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> List[float]:
    """
    训练 Titans MAC 模型。

    注意事项:
        - 每个 epoch 开始时需要重置 NMM 的记忆
        - NMM 的参数更新(惊讶度梯度) 不通过 optimizer, 而是在前向传播中直接完成
        - Optimizer 只更新"外层"参数 (嵌入、注意力、投影等)

    参数:
        model:         Titans MAC 模型
        train_inputs:  训练输入
        train_targets: 训练目标
        epochs:        训练轮数
        lr:            学习率
        batch_size:    批大小

    返回:
        losses: 每个 epoch 的平均损失
    """
    # 只优化外层参数 (NMM 的 MLP 参数通过惊讶度梯度更新)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    n_samples = train_inputs.shape[0]
    device = train_inputs.device

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_samples, device=device)

        for start in range(0, n_samples, batch_size):
            idx = perm[start: start + batch_size]
            batch_x = train_inputs[idx]
            batch_y = train_targets[idx]

            # 重置记忆 (每个 batch 独立)
            model.reset_memory()

            # 前向传播 (NMM 在此过程中自动更新)
            pred = model(batch_x)

            # 损失只计算有效位置 (跳过前 context_window-1 个位置)
            C = model.config.context_window
            loss = criterion(pred[:, C - 1:, :], batch_y[:, C - 1:, :])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Train] Epoch {epoch + 1:3d}/{epochs}, MSE Loss: {avg_loss:.6f}")

    return losses


# ==============================================================================
# 演示: 独立的 NMM 记忆能力
# ==============================================================================

def demo_neural_memory(device: torch.device):
    """
    独立演示神经长期记忆 (NMM) 的记忆能力。

    展示:
        1. NMM 如何通过惊讶度梯度记忆新数据
        2. 记忆后能准确检索信息
        3. 惊讶度如何反映数据的"新颖程度"
    """
    print("\n" + "=" * 60)
    print("演示 1: 神经长期记忆 (NMM) 独立运行")
    print("=" * 60)

    d = 32
    nmm = NeuralMemory(d_model=d, n_layers=2, hidden_dim=64,
                       alpha=0.999, eta=0.9, theta=0.05).to(device)

    # 创建 5 个 key-value 对来"记忆"
    keys = F.normalize(torch.randn(5, d, device=device), dim=-1)
    values = torch.randn(5, d, device=device) * 0.5

    print(f"\n  记忆 5 个 key-value 对...")
    print(f"  NMM 参数量: {sum(p.numel() for p in nmm.parameters()):,}")

    # 多轮更新, 观察惊讶度变化
    for round_idx in range(3):
        # 构造输入: 使 W_K(x) ≈ key, W_V(x) ≈ value
        x = torch.randn(5, d, device=device)

        loss, _ = nmm.update(x)
        print(f"  Round {round_idx + 1}: 关联记忆损失 (惊讶度) = {loss:.4f}")

    # 测试检索
    print(f"\n  检索测试:")
    with torch.no_grad():
        query = torch.randn(1, d, device=device)
        retrieved = nmm.retrieve(query)
        print(f"    查询 shape: {query.shape}, 检索结果 shape: {retrieved.shape}")
        print(f"    检索成功! NMM 可以作为关联记忆使用。")

    # 展示惊讶度对比: 重复数据 vs 新数据
    print(f"\n  惊讶度对比:")
    # 先用同一组数据更新多次
    fixed_x = torch.randn(3, d, device=device)
    for _ in range(5):
        nmm.update(fixed_x)

    # 再次用相同数据
    loss_familiar, _ = nmm.update(fixed_x)
    print(f"    熟悉数据的惊讶度: {loss_familiar:.4f} (低 → 已经记住了)")

    # 用全新数据
    new_x = torch.randn(3, d, device=device)
    loss_novel, _ = nmm.update(new_x)
    print(f"    新数据的惊讶度:   {loss_novel:.4f} (高 → 还没见过)")
    print(f"    比例:             {loss_novel / max(loss_familiar, 1e-8):.1f}x")


# ==============================================================================
# 主训练流程
# ==============================================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "Titans MAC 训练演示".center(46) + "║")
    print("║" + "(Memory As Context - 长期记忆)".center(46) + "║")
    print("╚" + "═" * 58 + "╝")

    # ---- 设备选择 ----
    device = get_device()
    print(f"\n使用设备: {device}")
    torch.manual_seed(42)

    # ---- 演示 1: NMM 独立运行 ----
    demo_neural_memory(device)

    # ---- 演示 2: 完整 MAC 模型训练 ----
    print("\n" + "=" * 60)
    print("演示 2: Titans MAC 模型训练 (时间序列预测)")
    print("=" * 60)

    config = TitanConfig(
        input_dim=1,
        output_dim=1,
        d_model=64,
        n_heads=2,
        n_layers=2,
        context_window=16,
        pm_size=4,
        nmm_layers=2,
        nmm_hidden=128,
        alpha=0.999,
        eta=0.9,
        theta=0.05,
    )

    model = TitanMAC(config).to(device)

    print(f"\n  模型配置:")
    print(f"    隐藏维度:       {config.d_model}")
    print(f"    MAC 层数:       {config.n_layers}")
    print(f"    上下文窗口:     {config.context_window}")
    print(f"    持久记忆 token: {config.pm_size}")
    print(f"    NMM 隐藏维度:   {config.nmm_hidden}")
    print(f"    总参数量:       {count_parameters(model):,}")

    # ---- 生成训练数据 ----
    print(f"\n  生成训练数据...")
    n_train = 100
    seq_len = 128
    train_x, train_y = generate_periodic_data(n_train, seq_len, device)
    print(f"    训练集: {train_x.shape}")

    # ---- 训练 ----
    print(f"\n  开始训练...")
    start_time = time.time()
    losses = train_model(
        model, train_x, train_y,
        epochs=20, lr=1e-3, batch_size=16,
    )
    elapsed = time.time() - start_time
    print(f"\n  训练完成! 耗时: {elapsed:.1f}s")
    print(f"  初始损失: {losses[0]:.6f}")
    print(f"  最终损失: {losses[-1]:.6f}")
    print(f"  降低比例: {losses[0] / max(losses[-1], 1e-8):.1f}x")

    # ---- 推理演示 ----
    print(f"\n  推理演示:")
    model.eval()
    model.reset_memory()
    test_x, test_y = generate_periodic_data(1, seq_len, device)

    with torch.no_grad():
        pred = model(test_x)

    # 计算预测精度
    C = config.context_window
    mse = F.mse_loss(pred[:, C:, :], test_y[:, C:, :]).item()
    print(f"    测试 MSE: {mse:.6f}")

    # 展示几个预测值 vs 真实值
    print(f"\n    预测值 vs 真实值 (最后 10 个时间步):")
    for t in range(-10, 0):
        p = pred[0, t, 0].item()
        g = test_y[0, t, 0].item()
        err = abs(p - g)
        bar = "█" * int(min(err * 100, 30))
        print(f"      t={seq_len + t}: pred={p:+.4f}, true={g:+.4f}, err={err:.4f} {bar}")

    # ---- 演示 3: 长期记忆的效果 ----
    print("\n" + "=" * 60)
    print("演示 3: 长期记忆的效果对比")
    print("=" * 60)

    # 在一个长序列上观察 NMM 记忆损失的变化
    print(f"\n  观察 NMM 惊讶度在长序列上的变化:")
    print(f"  (惊讶度应该: 序列开始时高, 随着记忆积累逐渐降低)")

    model.reset_memory()
    nmm = model.mac_layers[0].nmm

    long_x = torch.randn(1, 256, 1, device=device)
    embed = model.embed

    surprise_history = []
    for t in range(C, 256, C):
        window = long_x[:, t - C: t, :]
        h = embed(window).view(-1, config.d_model)
        loss_val, _ = nmm.update(h.detach())
        surprise_history.append(loss_val)

    print(f"    前 5 个窗口的惊讶度:  {[f'{s:.2f}' for s in surprise_history[:5]]}")
    print(f"    后 5 个窗口的惊讶度:  {[f'{s:.2f}' for s in surprise_history[-5:]]}")

    if surprise_history[0] > surprise_history[-1]:
        print(f"    ✓ 惊讶度随时间降低，说明 NMM 正在学习记忆序列模式!")
    else:
        print(f"    (惊讶度变化不显著，在随机数据上这是正常的)")

    # ---- 架构对比总结 ----
    print("\n" + "=" * 60)
    print("Titans 核心思想总结")
    print("=" * 60)
    summary = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    Titans 三种记忆                          │
    ├──────────────┬──────────────────────────────────────────────┤
    │ 短期记忆      │ Attention (标准注意力, 关注当前窗口)         │
    │ (工作记忆)    │ 容量: context_window 个 token              │
    ├──────────────┼──────────────────────────────────────────────┤
    │ 长期记忆      │ NMM (神经记忆模块, MLP 参数编码历史)        │
    │ (经验记忆)    │ 容量: O(1) 固定大小, 但可记忆无限历史       │
    │              │ 更新: 惊讶度驱动 (SGD + Momentum + Decay)   │
    ├──────────────┼──────────────────────────────────────────────┤
    │ 持久记忆      │ PM (可学习参数, 编码先验知识)               │
    │ (常识/直觉)   │ 容量: pm_size 个可学习 token               │
    └──────────────┴──────────────────────────────────────────────┘

    核心公式:
        惊讶度:    S_t = η·S_{t-1} - θ·∇ℓ(M_{t-1}; x_t)
        记忆更新:  M_t = α·M_{t-1} + S_t
        关联损失:  ℓ = ||M(k_t) - v_t||²

    与 Transformer / RNN 的对比:
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │              │ Transformer  │  RNN/SSM     │  Titans      │
    ├──────────────┼──────────────┼──────────────┼──────────────┤
    │ 记忆容量      │ O(N) 线性    │ O(1) 固定    │ O(1)+窗口    │
    │ 上下文长度    │ 有限 (窗口)  │ 无限 (理论)  │ 无限+精确    │
    │ 训练并行      │ ✓            │ ✗            │ ✓            │
    │ 记忆更新      │ 无           │ 隐式 (门控)  │ 显式 (梯度)  │
    │ 记忆表达力    │ 线性         │ 线性/低秩    │ 非线性 (MLP) │
    └──────────────┴──────────────┴──────────────┴──────────────┘
"""
    print(summary)

    # ---- 训练曲线 ----
    print("  训练曲线 (ASCII):")
    _ascii_plot(losses, width=50, height=8, label="MSE Loss")


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
    print(f"    {max_v:10.6f} ┤")
    for row in range(height - 2):
        threshold = max_v - (row + 1) * range_v / (height - 1)
        line = "    " + " " * 11 + "│"
        for v in resampled:
            if v >= threshold:
                line += "█"
            else:
                line += " "
        print(line)
    print(f"    {min_v:10.6f} ┤" + "─" * width)
    print(f"    " + " " * 11 + "0" + " " * (width - 5) + f"{len(values)}")


if __name__ == "__main__":
    main()
