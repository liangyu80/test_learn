"""
位置编码对比实验

对比 9 种位置编码在以下维度的表现:
1. 训练收敛速度 (字符级语言模型)
2. 参数量开销
3. 推理速度 (tokens/sec)
4. 长度泛化能力 (短序列训练 -> 长序列测试)
5. 位置感知任务 (复制/反转序列)
"""

import time
import torch
import torch.nn as nn
from positional_encoding import PositionalEncodingLM


# =============================================================================
# 数据生成
# =============================================================================
def generate_lm_data(
    n_samples: int, seq_len: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    生成带有位置依赖模式的序列数据
    模式: 每个位置的 token 值 = (前一个 token + position) % vocab_size
    这样模型需要理解位置信息才能正确预测
    """
    data = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
    data[:, 0] = torch.randint(0, vocab_size, (n_samples,), device=device)
    for t in range(1, seq_len):
        data[:, t] = (data[:, t - 1] + t) % vocab_size
    return data


def generate_copy_data(
    n_samples: int, half_len: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    生成复制任务数据: [x1, x2, ..., xn, SEP, x1, x2, ..., xn]
    模型需要精确的位置感知来完成复制

    SEP token 用 vocab_size - 1 表示
    """
    sep_token = vocab_size - 1
    src = torch.randint(0, vocab_size - 1, (n_samples, half_len), device=device)
    sep = torch.full((n_samples, 1), sep_token, dtype=torch.long, device=device)
    return torch.cat([src, sep, src], dim=1)  # (n_samples, 2*half_len+1)


def generate_reverse_data(
    n_samples: int, half_len: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    生成反转任务数据: [x1, x2, ..., xn, SEP, xn, ..., x2, x1]
    更强的位置理解需求
    """
    sep_token = vocab_size - 1
    src = torch.randint(0, vocab_size - 1, (n_samples, half_len), device=device)
    sep = torch.full((n_samples, 1), sep_token, dtype=torch.long, device=device)
    rev = src.flip(dims=[1])
    return torch.cat([src, sep, rev], dim=1)


# =============================================================================
# 实验1: 训练收敛对比
# =============================================================================
def exp_training_convergence():
    """对比各位置编码的训练收敛速度"""
    print("\n" + "=" * 70)
    print("实验1: 训练收敛对比 (字符级语言模型)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe_types = [
        "sinusoidal", "learned", "rope", "alibi",
        "relative", "kerple", "fire", "cope", "none",
    ]

    # 超参数
    vocab_size = 64
    d_model = 64
    n_heads = 4
    n_layers = 2
    seq_len = 64
    n_train = 512
    batch_size = 64
    n_epochs = 30
    lr = 3e-3

    # 生成数据
    train_data = generate_lm_data(n_train, seq_len, vocab_size, device)
    criterion = nn.CrossEntropyLoss()

    results = {}

    for pe_type in pe_types:
        print(f"\n--- {pe_type} ---")
        model = PositionalEncodingLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=d_model * 2,
            max_len=seq_len * 2,
            pe_type=pe_type,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                batch = train_data[i : i + batch_size]
                logits = model(batch)
                loss = criterion(
                    logits[:, :-1].reshape(-1, vocab_size),
                    batch[:, 1:].reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

        results[pe_type] = {
            "final_loss": losses[-1],
            "losses": losses,
            "params": model.count_params(),
        }

    # 汇总
    print(f"\n{'类型':<15} {'参数量':>10} {'初始损失':>10} {'最终损失':>10} {'收敛速度':>10}")
    print("-" * 60)
    for pe_type in pe_types:
        r = results[pe_type]
        # 收敛速度: 首次低于阈值的 epoch
        threshold = r["losses"][0] * 0.5  # 损失降到初始的 50%
        conv_epoch = n_epochs
        for i, l in enumerate(r["losses"]):
            if l < threshold:
                conv_epoch = i + 1
                break
        print(
            f"{pe_type:<15} {r['params']:>10,} "
            f"{r['losses'][0]:>10.4f} {r['final_loss']:>10.4f} "
            f"{'epoch '+str(conv_epoch):>10}"
        )

    return results


# =============================================================================
# 实验2: 推理速度对比
# =============================================================================
def exp_inference_speed():
    """对比各位置编码的推理速度"""
    print("\n" + "=" * 70)
    print("实验2: 推理速度对比")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe_types = [
        "sinusoidal", "learned", "rope", "alibi",
        "relative", "kerple", "fire", "cope", "none",
    ]

    vocab_size = 64
    d_model = 64
    n_heads = 4
    seq_len = 128
    batch_size = 32
    n_warmup = 5
    n_runs = 20

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"\n设备: {device} | seq_len={seq_len} | batch_size={batch_size}")
    print(f"\n{'类型':<15} {'平均耗时(ms)':>12} {'tokens/sec':>12}")
    print("-" * 42)

    for pe_type in pe_types:
        model = PositionalEncodingLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            ff_dim=d_model * 2,
            max_len=seq_len * 2,
            pe_type=pe_type,
        ).to(device).eval()

        # 预热
        with torch.no_grad():
            for _ in range(n_warmup):
                model(input_ids)

        # 计时
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                model(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        total_tokens = batch_size * seq_len
        tokens_per_sec = total_tokens / (avg_ms / 1000)

        print(f"{pe_type:<15} {avg_ms:>12.2f} {tokens_per_sec:>12,.0f}")


# =============================================================================
# 实验3: 长度泛化能力
# =============================================================================
def exp_length_generalization():
    """测试在短序列上训练、长序列上推理的泛化能力"""
    print("\n" + "=" * 70)
    print("实验3: 长度泛化能力 (训练 seq=32 → 测试 seq=64,96,128)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 只测可能支持外推的方法
    pe_types = ["sinusoidal", "learned", "rope", "alibi", "kerple", "fire", "none"]

    vocab_size = 64
    d_model = 64
    n_heads = 4
    train_len = 32
    test_lens = [32, 64, 96, 128]
    n_train = 512
    batch_size = 64
    n_epochs = 30
    lr = 3e-3

    criterion = nn.CrossEntropyLoss()
    train_data = generate_lm_data(n_train, train_len, vocab_size, device)

    print(f"\n训练序列长度: {train_len}")
    header = f"{'类型':<15} {'训练损失':>10}"
    for tl in test_lens:
        header += f" {'len='+str(tl):>10}"
    print(header)
    print("-" * (15 + 10 + 10 * len(test_lens) + 2))

    for pe_type in pe_types:
        model = PositionalEncodingLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            ff_dim=d_model * 2,
            max_len=256,  # 大于最长测试序列
            pe_type=pe_type,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # 训练
        for epoch in range(n_epochs):
            model.train()
            for i in range(0, n_train, batch_size):
                batch = train_data[i : i + batch_size]
                logits = model(batch)
                loss = criterion(
                    logits[:, :-1].reshape(-1, vocab_size),
                    batch[:, 1:].reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 测试不同长度
        model.eval()
        row = f"{pe_type:<15} {loss.item():>10.4f}"

        for test_len in test_lens:
            test_data = generate_lm_data(128, test_len, vocab_size, device)
            with torch.no_grad():
                logits = model(test_data)
                test_loss = criterion(
                    logits[:, :-1].reshape(-1, vocab_size),
                    test_data[:, 1:].reshape(-1),
                )
            row += f" {test_loss.item():>10.4f}"

        print(row)


# =============================================================================
# 实验4: 位置感知任务 (复制 & 反转)
# =============================================================================
def exp_positional_tasks():
    """测试位置敏感任务: 序列复制和反转"""
    print("\n" + "=" * 70)
    print("实验4: 位置感知任务 (序列复制 & 反转)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe_types = [
        "sinusoidal", "learned", "rope", "alibi",
        "relative", "kerple", "fire", "cope", "none",
    ]

    vocab_size = 32
    d_model = 64
    n_heads = 4
    half_len = 8  # 序列半长
    n_train = 1024
    batch_size = 64
    n_epochs = 50
    lr = 3e-3

    criterion = nn.CrossEntropyLoss()

    tasks = {
        "复制": lambda n, hl, vs, d: generate_copy_data(n, hl, vs, d),
        "反转": lambda n, hl, vs, d: generate_reverse_data(n, hl, vs, d),
    }

    for task_name, data_fn in tasks.items():
        print(f"\n--- 任务: {task_name} (half_len={half_len}) ---")
        train_data = data_fn(n_train, half_len, vocab_size, device)
        test_data = data_fn(256, half_len, vocab_size, device)
        full_len = 2 * half_len + 1

        print(f"{'类型':<15} {'最终损失':>10} {'准确率(%)':>10}")
        print("-" * 38)

        for pe_type in pe_types:
            model = PositionalEncodingLM(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=3,
                ff_dim=d_model * 2,
                max_len=full_len * 2,
                pe_type=pe_type,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # 训练
            for epoch in range(n_epochs):
                model.train()
                for i in range(0, n_train, batch_size):
                    batch = train_data[i : i + batch_size]
                    logits = model(batch)
                    # 只在 SEP 之后的位置计算损失
                    loss = criterion(
                        logits[:, half_len:-1].reshape(-1, vocab_size),
                        batch[:, half_len + 1 :].reshape(-1),
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # 测试准确率
            model.eval()
            with torch.no_grad():
                logits = model(test_data)
                preds = logits[:, half_len:-1].argmax(dim=-1)
                targets = test_data[:, half_len + 1 :]
                accuracy = (preds == targets).float().mean().item() * 100

                test_loss = criterion(
                    logits[:, half_len:-1].reshape(-1, vocab_size),
                    test_data[:, half_len + 1 :].reshape(-1),
                )

            print(f"{pe_type:<15} {test_loss.item():>10.4f} {accuracy:>10.1f}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    print("=" * 70)
    print("  位置编码 (Positional Encoding) 全面对比实验")
    print("  对比: Sinusoidal / Learned / RoPE / ALiBi / Relative")
    print("        Kerple / FIRE / CoPE / NoPE")
    print("=" * 70)

    exp_training_convergence()
    exp_inference_speed()
    exp_length_generalization()
    exp_positional_tasks()

    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
