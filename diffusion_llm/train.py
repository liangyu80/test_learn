"""
Diffusion LLM — 训练与对比实验

实验内容:
1. 序列复制任务: 三种方法学习复制输入序列的能力对比
2. 生成质量分析: 训练后的生成样本分析
3. 去噪过程可视化: 展示从噪声到文本的逐步去噪
4. 与自回归 LM 对比: Diffusion LLM vs 标准自回归 Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from diffusion_llm import (
    AbsorbingDiffusionLM,
    MultinomialDiffusionLM,
    EmbeddingDiffusionLM,
    count_parameters,
)


# ============================================================
# 简单自回归 LM (对比基线)
# ============================================================

class SimpleAutoRegressiveLM(nn.Module):
    """
    标准自回归语言模型 (GPT 风格)

    从左到右逐个预测下一个 token。
    作为对比基线: Diffusion LLM 的非自回归 vs 这里的自回归。
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_seq_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        # 因果 mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        return self.head(h)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x[:, :-1])
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), x[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, batch_size: int = 1, seq_len: int = 16,
                 device: str = 'cpu') -> torch.Tensor:
        self.eval()
        x = torch.randint(0, self.vocab_size, (batch_size, 1), device=device)
        for _ in range(seq_len - 1):
            logits = self.forward(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token], dim=1)
        return x


# ============================================================
# 数据集: 简单模式序列
# ============================================================

def create_pattern_dataset(num_samples: int, seq_len: int, vocab_size: int,
                           device: str = 'cpu') -> torch.Tensor:
    """
    创建有规律的序列数据集

    模式:
    - 重复模式: [a, b, c, a, b, c, a, b, c, ...]
    - 递增模式: [a, a+1, a+2, a+3, ...]
    - 镜像模式: [a, b, c, d, d, c, b, a]

    这些简单模式方便观察模型是否真正学会了数据分布。
    """
    data = []
    for i in range(num_samples):
        pattern_type = i % 3
        if pattern_type == 0:
            # 重复模式: 长度 2~4 的 pattern 重复
            pat_len = (i % 3) + 2
            pat = torch.randint(0, vocab_size, (pat_len,))
            seq = pat.repeat(seq_len // pat_len + 1)[:seq_len]
        elif pattern_type == 1:
            # 递增模式
            start = torch.randint(0, vocab_size - seq_len, (1,)).item()
            seq = torch.arange(start, start + seq_len) % vocab_size
        else:
            # 镜像模式
            half = seq_len // 2
            first_half = torch.randint(0, vocab_size, (half,))
            seq = torch.cat([first_half, first_half.flip(0)])
            if seq_len % 2 == 1:
                seq = torch.cat([seq, first_half[:1]])
        data.append(seq)
    return torch.stack(data).to(device)


# ============================================================
# 实验 1: 训练收敛对比
# ============================================================

def experiment_training_comparison():
    """对比三种 Diffusion LLM 和自回归 LM 的训练收敛"""
    print("=" * 70)
    print("实验 1: 训练收敛对比")
    print("=" * 70)

    vocab_size = 30
    seq_len = 16
    d_model = 64
    n_layers = 2
    n_heads = 4
    num_timesteps = 50
    batch_size = 32
    num_steps = 150
    device = 'cpu'

    dataset = create_pattern_dataset(512, seq_len, vocab_size, device)

    models = {
        '吸收态扩散 (MDLM)': AbsorbingDiffusionLM(
            vocab_size, d_model, n_heads, n_layers, seq_len, num_timesteps),
        '多项式扩散 (D3PM)': MultinomialDiffusionLM(
            vocab_size, d_model, n_heads, n_layers, seq_len, num_timesteps),
        '嵌入扩散 (Diff-LM)': EmbeddingDiffusionLM(
            vocab_size, d_model, n_heads, n_layers, seq_len, num_timesteps),
        '自回归 (GPT)': SimpleAutoRegressiveLM(
            vocab_size, d_model, n_heads, n_layers, seq_len),
    }

    print(f"\n{'模型':<24} {'参数量':>10} {'初始损失':>10} {'最终损失':>10} {'训练时间':>10}")
    print("-" * 70)

    for name, model in models.items():
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        params = count_parameters(model)

        losses = []
        start_time = time.time()

        for step in range(num_steps):
            idx = torch.randint(0, len(dataset), (batch_size,))
            batch = dataset[idx]

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        elapsed = time.time() - start_time
        print(f"{name:<24} {params:>10,} {losses[0]:>10.4f} {losses[-1]:>10.4f} {elapsed:>9.1f}s")

    print()


# ============================================================
# 实验 2: 去噪过程可视化
# ============================================================

def experiment_denoising_visualization():
    """可视化吸收态扩散的逐步去噪过程"""
    print("=" * 70)
    print("实验 2: 去噪过程可视化 (吸收态扩散)")
    print("=" * 70)

    vocab_size = 20
    seq_len = 12
    num_timesteps = 30
    device = 'cpu'

    model = AbsorbingDiffusionLM(
        vocab_size, d_model=64, n_heads=4, n_layers=2,
        max_seq_len=seq_len, num_timesteps=num_timesteps
    )

    # 简单训练
    dataset = create_pattern_dataset(256, seq_len, vocab_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("训练中...")
    for step in range(200):
        idx = torch.randint(0, len(dataset), (32,))
        batch = dataset[idx]
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
    print(f"训练完成, 最终损失: {loss.item():.4f}")

    # 逐步去噪可视化
    print("\n--- 逐步去噪过程 ---")
    model.eval()
    L = seq_len
    x = torch.full((1, L), model.mask_token_id, dtype=torch.long)
    steps_to_show = set([num_timesteps] + list(range(num_timesteps, 0, -max(1, num_timesteps // 8))) + [1])

    with torch.no_grad():
        for step in range(num_timesteps, 0, -1):
            t = torch.tensor([step])
            logits = model.denoiser(x, t)
            probs = F.softmax(logits[:, :, :vocab_size], dim=-1)
            pred_tokens = probs.argmax(dim=-1)

            next_mask_prob = (step - 1) / num_timesteps
            n_to_mask = int(next_mask_prob * L)

            if step - 1 > 0:
                confidence = probs.max(dim=-1).values
                is_masked = (x == model.mask_token_id)
                confidence[~is_masked] = 2.0
                x[is_masked] = pred_tokens[is_masked]
                if n_to_mask > 0:
                    _, indices = confidence.sort(dim=-1)
                    x[0, indices[0, :n_to_mask]] = model.mask_token_id
            else:
                x[x == model.mask_token_id] = pred_tokens[x == model.mask_token_id]

            if step in steps_to_show:
                display = []
                for tok in x[0].tolist():
                    if tok == model.mask_token_id:
                        display.append("[M]")
                    else:
                        display.append(f"{tok:2d}")
                n_mask = sum(1 for t in x[0].tolist() if t == model.mask_token_id)
                print(f"  t={step:2d}: {' '.join(display):>50s}  (mask={n_mask})")

    print(f"\n最终生成: {x[0].tolist()}")
    print()


# ============================================================
# 实验 3: 生成质量分析
# ============================================================

def experiment_generation_quality():
    """训练后分析生成样本的质量 — 检查是否学到了数据中的模式"""
    print("=" * 70)
    print("实验 3: 生成质量分析")
    print("=" * 70)

    vocab_size = 20
    seq_len = 12
    num_timesteps = 30
    device = 'cpu'

    # 只用重复模式训练 — 容易检验
    print("创建重复模式数据: [a,b,a,b,...] 或 [a,b,c,a,b,c,...]")
    data = []
    for _ in range(256):
        pat_len = torch.randint(2, 4, (1,)).item()
        pat = torch.randint(0, vocab_size, (pat_len,))
        seq = pat.repeat(seq_len // pat_len + 1)[:seq_len]
        data.append(seq)
    dataset = torch.stack(data)

    models_config = [
        ("吸收态扩散", AbsorbingDiffusionLM(
            vocab_size, 64, 4, 2, seq_len, num_timesteps)),
        ("多项式扩散", MultinomialDiffusionLM(
            vocab_size, 64, 4, 2, seq_len, num_timesteps)),
    ]

    for name, model in models_config:
        print(f"\n--- {name} ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(300):
            idx = torch.randint(0, len(dataset), (32,))
            batch = dataset[idx]
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
        print(f"训练完成, 损失: {loss.item():.4f}")

        # 生成 8 个样本
        samples = model.generate(batch_size=8, seq_len=seq_len)
        print("生成样本:")
        repeat_count = 0
        for i, s in enumerate(samples):
            seq = s.tolist()
            # 检查是否有重复模式
            has_repeat = False
            for pat_len in [2, 3, 4]:
                pat = seq[:pat_len]
                expected = (pat * (seq_len // pat_len + 1))[:seq_len]
                if seq == expected:
                    has_repeat = True
                    break
            marker = " ✓ 重复模式" if has_repeat else ""
            print(f"  [{i}] {seq}{marker}")
            if has_repeat:
                repeat_count += 1
        print(f"重复模式占比: {repeat_count}/8 = {repeat_count/8:.0%}")

    print()


# ============================================================
# 实验 4: Diffusion LLM vs 自回归 LM
# ============================================================

def experiment_diffusion_vs_autoregressive():
    """
    对比 Diffusion LLM 和自回归 LM 的核心区别

    对比维度:
    1. 生成方式: 并行 vs 顺序
    2. 生成速度: 多少次前向传播才能生成完整序列
    3. 可编辑性: 能否修改已生成的内容
    """
    print("=" * 70)
    print("实验 4: Diffusion LLM vs 自回归 LM 核心对比")
    print("=" * 70)

    vocab_size = 30
    seq_len = 16
    device = 'cpu'

    diff_model = AbsorbingDiffusionLM(
        vocab_size, d_model=64, n_heads=4, n_layers=2,
        max_seq_len=seq_len, num_timesteps=20
    )
    ar_model = SimpleAutoRegressiveLM(
        vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=seq_len
    )

    print(f"\n{'维度':<20} {'Diffusion LLM':<30} {'自回归 LLM (GPT)':<30}")
    print("-" * 80)
    print(f"{'生成方式':<20} {'并行 (所有token同时生成)':<30} {'顺序 (从左到右逐个)':<30}")
    print(f"{'注意力类型':<20} {'双向 (看到所有位置)':<30} {'因果 (只看到左侧)':<30}")
    print(f"{'迭代次数':<20} {'T步去噪 (可调节)':<30} {'L步 (=序列长度)':<30}")
    print(f"{'可编辑性':<20} {'✓ 可 infill/修改任意位置':<30} {'✗ 只能 append':<30}")
    print(f"{'训练目标':<20} {'去噪/重建':<30} {'下一个token预测':<30}")
    print(f"{'困惑度':<20} {'通常略逊':<30} {'通常更优':<30}")

    # 测量生成速度
    print("\n--- 生成速度对比 ---")

    diff_model.eval()
    ar_model.eval()

    with torch.no_grad():
        # Diffusion: T 步去噪
        start = time.time()
        for _ in range(10):
            diff_model.generate(1, seq_len)
        diff_time = (time.time() - start) / 10

        # AR: L 步自回归
        start = time.time()
        for _ in range(10):
            ar_model.generate(1, seq_len)
        ar_time = (time.time() - start) / 10

    print(f"  Diffusion LLM (T={diff_model.num_timesteps}步): {diff_time*1000:.1f}ms/序列")
    print(f"  自回归 LM (L={seq_len}步):         {ar_time*1000:.1f}ms/序列")

    # 演示 infilling — Diffusion LLM 的独特能力
    print("\n--- Infilling 演示 (Diffusion LLM 独有) ---")
    print("场景: 给定 '_ _ _ 5 6 7 _ _ _ 5 6 7', 填充 [MASK] 位置")

    x = torch.full((1, seq_len), diff_model.mask_token_id, dtype=torch.long)
    # 已知位置
    known_positions = {3: 5, 4: 6, 5: 7, 9: 5, 10: 6, 11: 7}
    for pos, val in known_positions.items():
        if pos < seq_len:
            x[0, pos] = val

    display = ["[M]" if t == diff_model.mask_token_id else f"{t:2d}" for t in x[0].tolist()]
    print(f"  输入: {' '.join(display)}")

    # 只去噪 mask 位置
    with torch.no_grad():
        for step in range(diff_model.num_timesteps, 0, -1):
            t = torch.tensor([step])
            logits = diff_model.denoiser(x, t)
            pred = logits[:, :, :vocab_size].argmax(dim=-1)
            # 只更新 mask 位置
            mask = (x == diff_model.mask_token_id)
            if step > 1:
                # 逐步揭示
                probs = F.softmax(logits[:, :, :vocab_size], dim=-1)
                confidence = probs.max(dim=-1).values
                confidence[~mask] = 2.0
                x[mask] = pred[mask]
                n_to_mask = int((step - 1) / diff_model.num_timesteps * mask.sum().item())
                if n_to_mask > 0:
                    _, indices = confidence.sort(dim=-1)
                    for b in range(1):
                        re_mask_count = 0
                        for idx in indices[b]:
                            if idx.item() not in known_positions and re_mask_count < n_to_mask:
                                x[b, idx] = diff_model.mask_token_id
                                re_mask_count += 1
            else:
                x[mask] = pred[mask]

    print(f"  输出: {x[0].tolist()}")
    print("  (自回归 LM 无法做 infilling — 只能从左到右生成)")

    print()


# ============================================================
# 实验 5: 三种扩散方法的核心机制对比
# ============================================================

def experiment_mechanism_comparison():
    """直观对比三种扩散方法的加噪和去噪机制"""
    print("=" * 70)
    print("实验 5: 三种扩散方法的核心机制对比")
    print("=" * 70)

    vocab_size = 20
    seq_len = 8

    # 创建一个明确的序列: [0, 1, 2, 3, 4, 5, 6, 7]
    x_0 = torch.arange(seq_len).unsqueeze(0)  # (1, 8)
    print(f"原始序列: {x_0[0].tolist()}")

    # --- 吸收态扩散 ---
    print("\n方法 1: 吸收态扩散 (Absorbing)")
    print("  噪声类型: [MASK] token")
    print("  信息去向: 被吸收到单一 [MASK] 状态")
    abs_model = AbsorbingDiffusionLM(vocab_size, num_timesteps=10, max_seq_len=seq_len)
    print("  加噪过程:")
    for t_val in [2, 5, 8, 10]:
        t = torch.tensor([t_val])
        x_t = abs_model.forward_process(x_0, t)
        display = ["[M]" if tok == abs_model.mask_token_id else f" {tok} " for tok in x_t[0].tolist()]
        print(f"    t={t_val:2d}: [{', '.join(display)}]")

    # --- 多项式扩散 ---
    print("\n方法 2: 多项式扩散 (Multinomial)")
    print("  噪声类型: 随机 token (均匀分布)")
    print("  信息去向: 扩散到整个词表")
    multi_model = MultinomialDiffusionLM(vocab_size, num_timesteps=10, max_seq_len=seq_len)
    print("  加噪过程:")
    for t_val in [2, 5, 8, 10]:
        t = torch.tensor([t_val])
        x_t = multi_model.forward_process(x_0, t)
        diff = ["*" if x_t[0, i] != x_0[0, i] else " " for i in range(seq_len)]
        print(f"    t={t_val:2d}: {x_t[0].tolist()}  变化: {''.join(diff)}")

    # --- 嵌入扩散 ---
    print("\n方法 3: 嵌入扩散 (Embedding)")
    print("  噪声类型: 高斯噪声 (在嵌入空间)")
    print("  信息去向: 被高斯噪声淹没")
    emb_model = EmbeddingDiffusionLM(vocab_size, num_timesteps=10, max_seq_len=seq_len)
    e_0 = emb_model.tokens_to_embeddings(x_0)
    print("  加噪过程 (嵌入空间信噪比):")
    for t_val in [2, 5, 8, 10]:
        t = torch.tensor([t_val])
        e_t, noise = emb_model.forward_process(e_0, t)
        signal_power = (emb_model.sqrt_alpha_cumprod[t_val - 1] ** 2).item()
        noise_power = (emb_model.sqrt_one_minus_alpha_cumprod[t_val - 1] ** 2).item()
        snr_db = 10 * torch.log10(torch.tensor(signal_power / max(noise_power, 1e-10))).item()
        recovered = emb_model.embeddings_to_tokens(e_t)
        acc = (recovered[0] == x_0[0]).float().mean().item()
        print(f"    t={t_val:2d}: SNR={snr_db:+.1f}dB, 信号={signal_power:.3f}, "
              f"噪声={noise_power:.3f}, 恢复率={acc:.0%}")

    # 总结
    print("\n" + "-" * 70)
    print("总结:")
    print(f"  {'方法':<16} {'噪声空间':<12} {'优点':<24} {'缺点':<24}")
    print(f"  {'吸收态':<16} {'离散(MASK)':<12} {'简单直观,训练稳定':<24} {'mask信息全丢失':<24}")
    print(f"  {'多项式':<16} {'离散(随机)':<12} {'保留部分语义线索':<24} {'训练较难收敛':<24}")
    print(f"  {'嵌入':<16} {'连续(高斯)':<12} {'可用DDPM/DDIM理论':<24} {'rounding误差':<24}")
    print()


if __name__ == '__main__':
    print("Diffusion LLM — 训练与对比实验")
    print("=" * 70)
    torch.manual_seed(42)

    experiment_training_comparison()
    experiment_denoising_visualization()
    experiment_generation_quality()
    experiment_diffusion_vs_autoregressive()
    experiment_mechanism_comparison()

    print("\n全部实验完成!")
