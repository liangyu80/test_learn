"""
Mamba vs Transformer vs 混合模型 统一对比实验

对比维度:
  1. 参数量与结构
  2. 训练损失收敛速度
  3. 训练/推理吞吐量 (tokens/sec)
  4. 不同序列长度的缩放行为
  5. 特定能力测试:
     - 重复模式学习 (pattern memorization)
     - 长距离依赖 (long-range dependency)
     - 精确回忆 (exact recall / copy task)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from mamba import MambaLM, MambaConfig
from transformer import TransformerLM, TransformerConfig
from hybrid import HybridLM, HybridConfig


# ==============================================================================
# 数据生成
# ==============================================================================

def generate_pattern_data(batch_size: int, seq_len: int,
                          pattern_len: int = 8, vocab_size: int = 50,
                          device: torch.device = torch.device('cpu')):
    """
    重复模式数据: [a,b,c,d,a,b,c,d,a,b,c,d,...]
    测试模型学习周期性模式的能力。
    """
    pattern = torch.randint(0, vocab_size, (batch_size, pattern_len), device=device)
    data = pattern.repeat(1, seq_len // pattern_len + 1)[:, :seq_len + 1]
    return data[:, :-1], data[:, 1:]


def generate_copy_data(batch_size: int, copy_len: int = 16,
                       delay: int = 32, vocab_size: int = 50,
                       device: torch.device = torch.device('cpu')):
    """
    复制任务: [a,b,c,...,0,0,...,0,a,b,c,...]
    测试模型精确回忆远处信息的能力。

    序列结构:
      [source tokens] [padding (delay)] [should copy source tokens]
    """
    # 源序列
    src = torch.randint(1, vocab_size, (batch_size, copy_len), device=device)
    # 分隔/填充 (用 0)
    pad = torch.zeros(batch_size, delay, dtype=torch.long, device=device)
    # 拼接: src + pad + src
    data = torch.cat([src, pad, src], dim=1)
    total_len = data.shape[1]
    return data[:, :-1], data[:, 1:]


def generate_longrange_data(batch_size: int, seq_len: int = 128,
                            vocab_size: int = 50,
                            device: torch.device = torch.device('cpu')):
    """
    长距离依赖: 第一个 token 决定最后一段的模式。
    token[0] 的值映射到不同的结尾模式, 测试长距离信息传播。
    """
    data = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    # 最后 8 个 token 由第一个 token 决定
    for b in range(batch_size):
        key = data[b, 0].item() % 5
        data[b, -8:] = torch.arange(key, key + 8) % vocab_size
    return data[:, :-1], data[:, 1:]


# ==============================================================================
# 训练与评估
# ==============================================================================

def train_and_evaluate(model: nn.Module, data_fn, name: str,
                       n_steps: int = 200, lr: float = 1e-3,
                       device: torch.device = torch.device('cpu')):
    """训练模型并返回损失曲线。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        inp, tgt = data_fn()
        _, loss = model(inp.to(device), tgt.to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return losses


def measure_throughput(model: nn.Module, batch_size: int, seq_len: int,
                       n_warmup: int = 5, n_measure: int = 20,
                       device: torch.device = torch.device('cpu')):
    """测量训练吞吐量 (tokens/sec)。"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    vocab_size = 256

    # 预热
    for _ in range(n_warmup):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        _, loss = model(x, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 测量
    start = time.time()
    for _ in range(n_measure):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        _, loss = model(x, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = n_measure * batch_size * seq_len
    return total_tokens / elapsed


def measure_inference_throughput(model: nn.Module, batch_size: int,
                                 seq_len: int, n_measure: int = 20,
                                 device: torch.device = torch.device('cpu')):
    """测量推理吞吐量。"""
    model.eval()
    vocab_size = 256

    # 预热
    with torch.no_grad():
        for _ in range(3):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_measure):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = n_measure * batch_size * seq_len
    return total_tokens / elapsed


# ==============================================================================
# 主对比实验
# ==============================================================================

def run_comparison():
    """运行完整对比实验。"""
    print("=" * 70)
    print("  Mamba vs Transformer vs 混合模型 对比实验")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ====================== 模型配置 ======================
    d_model = 64
    n_layers = 4
    vocab_size = 256
    max_seq_len = 256
    batch_size = 8
    seq_len = 64

    # 创建模型
    models = {}

    # 1. 纯 Mamba
    mamba_config = MambaConfig(
        d_model=d_model, n_layers=n_layers, vocab_size=vocab_size,
        d_state=16, d_conv=4, expand=2, max_seq_len=max_seq_len,
    )
    models['Mamba'] = MambaLM(mamba_config).to(device)

    # 2. 纯 Transformer
    tf_config = TransformerConfig(
        d_model=d_model, n_layers=n_layers, n_heads=4,
        vocab_size=vocab_size, max_seq_len=max_seq_len,
    )
    models['Transformer'] = TransformerLM(tf_config).to(device)

    # 3. 混合: Jamba 风格
    jamba_config = HybridConfig(
        d_model=d_model, n_layers=n_layers * 2, vocab_size=vocab_size,
        max_seq_len=max_seq_len, n_heads=4,
        d_state=16, d_conv=4, expand=2,
        strategy="jamba", attn_interval=4,
    )
    models['Hybrid-Jamba'] = HybridLM(jamba_config).to(device)

    # 4. 混合: 交替风格
    alt_config = HybridConfig(
        d_model=d_model, n_layers=n_layers * 2, vocab_size=vocab_size,
        max_seq_len=max_seq_len, n_heads=4,
        d_state=16, d_conv=4, expand=2,
        strategy="alternate",
    )
    models['Hybrid-Alternate'] = HybridLM(alt_config).to(device)

    # 5. 混合: Zamba 风格
    zamba_config = HybridConfig(
        d_model=d_model, n_layers=n_layers * 2, vocab_size=vocab_size,
        max_seq_len=max_seq_len, n_heads=4,
        d_state=16, d_conv=4, expand=2,
        strategy="zamba",
    )
    models['Hybrid-Zamba'] = HybridLM(zamba_config).to(device)

    # ====================== 1. 参数量对比 ======================
    print("\n" + "=" * 70)
    print("  1. 模型参数量对比")
    print("=" * 70)
    print(f"\n{'模型':<20s} {'参数量':>12s} {'结构'}")
    print("─" * 70)
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        if hasattr(model, 'get_architecture_info'):
            arch = model.get_architecture_info()
        elif isinstance(model, MambaLM):
            arch = " → ".join([f"[M{i}]" for i in range(mamba_config.n_layers)])
        else:
            arch = " → ".join([f"[T{i}]" for i in range(tf_config.n_layers)])
        print(f"  {name:<18s} {n_params:>10,}   {arch}")

    # ====================== 2. 训练收敛对比 ======================
    print("\n" + "=" * 70)
    print("  2. 训练收敛速度 (重复模式学习)")
    print("=" * 70)

    n_train_steps = 300
    train_results = {}

    for name, model in models.items():
        # 重置模型参数
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        data_fn = lambda: generate_pattern_data(
            batch_size, seq_len, pattern_len=8, vocab_size=50, device=device)
        losses = train_and_evaluate(
            model, data_fn, name, n_steps=n_train_steps, device=device)
        train_results[name] = losses

        # 打印关键节点
        print(f"\n  {name}:")
        for step in [50, 100, 200, 300]:
            if step <= len(losses):
                avg = sum(losses[max(0,step-10):step]) / min(10, step)
                print(f"    Step {step:>3d}: loss = {avg:.4f}")

    # ====================== 3. 吞吐量对比 ======================
    print("\n" + "=" * 70)
    print("  3. 吞吐量对比 (tokens/sec)")
    print("=" * 70)

    print(f"\n{'模型':<20s} {'训练吞吐':>15s} {'推理吞吐':>15s}")
    print("─" * 55)
    for name, model in models.items():
        train_tp = measure_throughput(
            model, batch_size=4, seq_len=64, device=device)
        infer_tp = measure_inference_throughput(
            model, batch_size=4, seq_len=64, device=device)
        print(f"  {name:<18s} {train_tp:>12,.0f}/s {infer_tp:>12,.0f}/s")

    # ====================== 4. 序列长度缩放 ======================
    print("\n" + "=" * 70)
    print("  4. 不同序列长度的推理吞吐")
    print("=" * 70)

    seq_lengths = [32, 64, 128, 256]
    print(f"\n{'模型':<20s}", end="")
    for sl in seq_lengths:
        print(f"  L={sl:>3d}", end="")
    print()
    print("─" * 60)

    # 只测试核心模型
    for name in ['Mamba', 'Transformer', 'Hybrid-Jamba']:
        model = models[name]
        print(f"  {name:<18s}", end="")
        for sl in seq_lengths:
            try:
                tp = measure_inference_throughput(
                    model, batch_size=4, seq_len=sl, n_measure=10, device=device)
                print(f"  {tp:>5.0f}k", end="")
            except Exception:
                print(f"    OOM", end="")
        print(" tokens/s")

    # ====================== 5. 能力测试 ======================
    print("\n" + "=" * 70)
    print("  5. 特定能力测试")
    print("=" * 70)

    # 重新初始化模型
    for name in models:
        for p in models[name].parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    tasks = {
        "重复模式": lambda: generate_pattern_data(
            batch_size, seq_len, pattern_len=8, vocab_size=50, device=device),
        "长距离依赖": lambda: generate_longrange_data(
            batch_size, seq_len=64, vocab_size=50, device=device),
        "精确复制": lambda: generate_copy_data(
            batch_size, copy_len=8, delay=16, vocab_size=50, device=device),
    }

    for task_name, data_fn in tasks.items():
        print(f"\n  任务: {task_name}")
        print(f"  {'模型':<20s} {'初始 loss':>10s} {'训练后 loss':>12s} {'提升':>8s}")
        print(f"  {'─' * 55}")

        for name, model in models.items():
            # 重置
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            # 初始损失
            model.eval()
            with torch.no_grad():
                inp, tgt = data_fn()
                _, init_loss = model(inp, tgt)
                init_loss = init_loss.item()

            # 训练
            model.train()
            losses = train_and_evaluate(
                model, data_fn, name, n_steps=200, device=device)
            final_loss = sum(losses[-10:]) / 10

            improvement = (init_loss - final_loss) / init_loss * 100
            print(f"  {name:<20s} {init_loss:>10.4f} {final_loss:>12.4f} "
                  f"{improvement:>7.1f}%")

    # ====================== 6. 总结 ======================
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │                      架构特点对比                                  │
  ├──────────────┬──────────────┬──────────────┬──────────────────────┤
  │              │  Transformer │    Mamba     │      混合模型         │
  ├──────────────┼──────────────┼──────────────┼──────────────────────┤
  │ 训练复杂度    │  O(N²·d)     │  O(N·d·n)    │  介于两者之间         │
  │ 推理复杂度    │  O(N) KV     │  O(1) 状态   │  少量 KV + 状态      │
  │ 长序列效率    │  差 (二次方)  │  好 (线性)   │  好                  │
  │ 精确回忆      │  强           │  弱          │  中-强               │
  │ 模式学习      │  强           │  强          │  强                  │
  │ 位置编码      │  必需         │  不需要      │  可选                │
  │ 参数效率      │  中           │  高          │  取决于策略           │
  ├──────────────┴──────────────┴──────────────┴──────────────────────┤
  │                                                                    │
  │ 混合策略推荐:                                                       │
  │  • 追求效率 → Jamba 风格 (少数 Attention 层)                        │
  │  • 追求质量 → Alternate 风格 (Attention 和 Mamba 交替)              │
  │  • 追求参数效率 → Zamba 风格 (共享 Attention)                       │
  │                                                                    │
  │ 实际应用趋势 (2024-2025):                                          │
  │  • Jamba (AI21): 398B 参数, 256K 上下文                             │
  │  • Qwen3.5: Mamba + Transformer 交替, 高效长上下文                  │
  │  • Zamba: 7B 规模最快的非 Transformer 模型                          │
  │  • 混合架构正在成为下一代 LLM 的主流趋势                             │
  └────────────────────────────────────────────────────────────────────┘
""")

    print("[对比实验完成]")


if __name__ == '__main__':
    run_comparison()
