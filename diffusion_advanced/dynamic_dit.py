"""
Dynamic Diffusion Transformer (DyDiT) —— 从零实现

核心思想:
    DiT (Diffusion Transformer) 是 Stable Diffusion 3 / Sora 等模型的基础架构，
    用 Transformer 替代传统的 U-Net 作为扩散模型的骨干网络。

    DyDiT 发现: DiT 在不同时间步和空间区域存在大量冗余计算。
    - 高噪声时 (t 大): 需要的特征较简单，不需要全部宽度
    - 低激活区域: 不需要与高激活区域相同的精度

    DyDiT 引入两种动态机制:
    1. 时间步自适应宽度 (Timestep-wise Width): 根据 t 动态选择使用多少维度
    2. 空间自适应 Token 选择: 根据重要性动态选择处理哪些 token

    效果: FLOPs 降低 51%, 速度提升 1.73x, FID 2.07 (ImageNet 256×256)

DiT 基础架构:
    ┌─────────────────────────────────────────┐
    │            Patch Embedding               │
    │  (图像 → token 序列 + 位置编码)         │
    ├─────────────────────────────────────────┤
    │        DiT Block × N                    │
    │  ┌───────────────────────────────────┐  │
    │  │  LayerNorm (自适应, 由 t 控制)    │  │
    │  │  Multi-Head Self-Attention        │  │
    │  │  + Residual                       │  │
    │  ├───────────────────────────────────┤  │
    │  │  LayerNorm (自适应)               │  │
    │  │  FFN (MLP)                        │  │
    │  │  + Residual                       │  │
    │  └───────────────────────────────────┘  │
    ├─────────────────────────────────────────┤
    │         Final LayerNorm + Linear         │
    │         (→ 预测噪声/速度)               │
    └─────────────────────────────────────────┘

DyDiT 的动态优化:
    ┌─────────────────────────────────────────┐
    │ 高噪声 (t 大)     → 使用 50% 宽度      │
    │ 低噪声 (t 小)     → 使用 100% 宽度     │
    │ 不重要的 token    → 跳过 (Token Pruning) │
    │ 重要的 token      → 正常计算            │
    └─────────────────────────────────────────┘

与其他扩散模型架构的关系:
    U-Net (Stable Diffusion 1/2) → DiT (SD3, Sora) → DyDiT (高效 DiT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class DyDiTConfig:
    """Dynamic DiT 配置。"""
    data_dim: int = 200           # 数据维度
    num_tokens: int = 25          # 将数据切分为多少个 token (patch)
    token_dim: int = 8            # 每个 token 的维度 (data_dim / num_tokens)
    d_model: int = 128            # Transformer 隐藏维度
    n_heads: int = 4              # 注意力头数
    n_layers: int = 4             # Transformer 层数
    d_ff: int = 256               # FFN 维度
    time_emb_dim: int = 64        # 时间嵌入维度
    dropout: float = 0.0
    # DyDiT 动态参数
    width_ratios: tuple = (0.5, 0.75, 1.0)  # 可选宽度比例
    token_keep_ratio: float = 0.7             # Token 保留比例


# ==============================================================================
# 基础组件
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ==============================================================================
# Adaptive LayerNorm (adaLN) —— DiT 的核心创新
# ==============================================================================

class AdaptiveLayerNorm(nn.Module):
    """
    自适应 LayerNorm (adaLN)。

    DiT 的核心创新之一: 用时间步条件来调制 LayerNorm 的 γ (scale) 和 β (shift)。

    标准 LN:  y = γ · (x - μ) / σ + β     (γ, β 是可学习参数)
    adaLN:    y = γ_t · (x - μ) / σ + β_t  (γ_t, β_t 由时间步 t 控制)

    这样不同时间步会有不同的归一化行为!
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # 从条件向量生成 γ 和 β
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    输入, shape (batch, seq_len, d_model)
            cond: 条件向量, shape (batch, cond_dim)
        """
        # 生成 scale 和 shift
        params = self.proj(cond)  # (batch, 2*d_model)
        gamma, beta = params.chunk(2, dim=-1)  # 各 (batch, d_model)
        gamma = gamma.unsqueeze(1)  # (batch, 1, d_model)
        beta = beta.unsqueeze(1)

        # 自适应归一化
        return self.norm(x) * (1 + gamma) + beta


# ==============================================================================
# DiT Block (标准版)
# ==============================================================================

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block。

    结构:
        x → adaLN → Multi-Head Attention → + Residual
          → adaLN → FFN (MLP)             → + Residual

    与标准 Transformer 的区别:
        使用 adaLN 替代普通 LayerNorm，让时间步控制归一化。
    """

    def __init__(self, config: DyDiTConfig):
        super().__init__()
        d = config.d_model

        # 自注意力
        self.ada_ln1 = AdaptiveLayerNorm(d, config.time_emb_dim)
        self.attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout,
                                          batch_first=True)

        # FFN
        self.ada_ln2 = AdaptiveLayerNorm(d, config.time_emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, d),
        )

        # adaLN-Zero: 用 gate 控制残差连接的强度
        # 这是 DiT 的另一个关键设计，初始化为 0 使训练更稳定
        self.gate_attn = nn.Parameter(torch.zeros(1, 1, d))
        self.gate_ffn = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     输入, shape (batch, seq_len, d_model)
            t_emb: 时间嵌入, shape (batch, time_emb_dim)
        """
        # 自注意力
        h = self.ada_ln1(x, t_emb)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gate_attn * h

        # FFN
        h = self.ada_ln2(x, t_emb)
        h = self.ffn(h)
        x = x + self.gate_ffn * h

        return x


# ==============================================================================
# Dynamic DiT Block (DyDiT 版本)
# ==============================================================================

class DynamicDiTBlock(nn.Module):
    """
    动态 DiT Block (DyDiT 的核心)。

    两种动态机制:
    1. 时间步自适应宽度: 根据 t 选择使用多少维度
       - 高噪声 (t 大): 特征简单，使用较少维度 (如 50%)
       - 低噪声 (t 小): 需要精细特征，使用全部维度 (100%)

    2. Token 选择: 根据重要性分数选择保留哪些 token
       - 使用一个轻量预测器评估每个 token 的重要性
       - 只对重要 token 做 attention + FFN
       - 不重要的 token 直接跳过 (用残差连接)
    """

    def __init__(self, config: DyDiTConfig):
        super().__init__()
        d = config.d_model
        self.d_model = d
        self.token_keep_ratio = config.token_keep_ratio

        # 宽度路由器 (Width Router): 根据 t 决定使用多少宽度
        self.width_router = nn.Sequential(
            nn.Linear(config.time_emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, len(config.width_ratios)),
        )
        self.width_ratios = config.width_ratios

        # Token 重要性预测器 (Token Router)
        self.token_router = nn.Sequential(
            nn.Linear(d, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # 标准 DiT Block 组件
        self.ada_ln1 = AdaptiveLayerNorm(d, config.time_emb_dim)
        self.attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout,
                                          batch_first=True)
        self.ada_ln2 = AdaptiveLayerNorm(d, config.time_emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, d),
        )
        self.gate_attn = nn.Parameter(torch.zeros(1, 1, d))
        self.gate_ffn = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                use_dynamic: bool = True) -> torch.Tensor:
        """
        Args:
            x:           输入, shape (batch, seq_len, d_model)
            t_emb:       时间嵌入, shape (batch, time_emb_dim)
            use_dynamic: 是否启用动态计算
        """
        batch_size, seq_len, d = x.shape

        if use_dynamic:
            # ---- 宽度路由: 记录统计信息 (实际宽度裁剪需要独立网络) ----
            width_logits = self.width_router(t_emb)  # (batch, num_ratios)
            # 训练时使用 softmax 加权 (避免不可微的 argmax)
            width_weights = F.softmax(width_logits, dim=-1)  # 用于辅助损失

            # ---- Token 路由: 选择重要的 token ----
            importance = self.token_router(x.detach()).squeeze(-1)  # (batch, seq_len)
            num_keep = max(int(seq_len * self.token_keep_ratio), 1)

            # 选择 top-k 重要的 token
            _, keep_indices = importance.topk(num_keep, dim=-1)
            keep_indices_sorted, _ = keep_indices.sort(dim=-1)

            # 提取重要 token
            batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(-1)
            x_selected = x[batch_idx, keep_indices_sorted]  # (batch, num_keep, d)

            # ---- 在选中的 token 上执行 attention + FFN ----
            h = self.ada_ln1(x_selected, t_emb)
            h, _ = self.attn(h, h, h, need_weights=False)
            x_selected = x_selected + self.gate_attn * h

            h = self.ada_ln2(x_selected, t_emb)
            h = self.ffn(h)
            x_selected = x_selected + self.gate_ffn * h

            # ---- 将结果放回原位置 (未选中的 token 保持不变) ----
            output = x.clone()
            output[batch_idx, keep_indices_sorted] = x_selected
            return output

        else:
            # 标准 DiT Block (无动态优化)
            h = self.ada_ln1(x, t_emb)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.gate_attn * h

            h = self.ada_ln2(x, t_emb)
            h = self.ffn(h)
            x = x + self.gate_ffn * h
            return x


# ==============================================================================
# 完整 DiT / DyDiT 模型
# ==============================================================================

class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (支持标准 DiT 和 DyDiT 模式)。

    将 1D 数据切分为 "patch" (类似图像中的 patch)，
    作为 token 输入 Transformer。

    数据处理流程:
        data (200,) → reshape (25, 8) → linear proj → (25, d_model)
        即: 25 个 token，每个 token 代表数据的一个区域
    """

    def __init__(self, config: DyDiTConfig, dynamic: bool = False):
        super().__init__()
        self.config = config
        self.dynamic = dynamic

        # Token 化: 将数据切分为 patch
        self.patch_embed = nn.Linear(config.token_dim, config.d_model)

        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_tokens, config.d_model) * 0.02
        )

        # 时间嵌入
        self.time_emb = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.time_emb_dim),
            nn.SiLU(),
            nn.Linear(config.time_emb_dim, config.time_emb_dim),
        )

        # Transformer 层
        if dynamic:
            self.blocks = nn.ModuleList([
                DynamicDiTBlock(config) for _ in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(config) for _ in range(config.n_layers)
            ])

        # 输出
        self.final_norm = AdaptiveLayerNorm(config.d_model, config.time_emb_dim)
        self.output_proj = nn.Linear(config.d_model, config.token_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入数据, shape (batch, data_dim)
            t: 时间, shape (batch,)
        Returns:
            预测, shape (batch, data_dim)
        """
        batch_size = x.size(0)
        config = self.config

        # 1. Token 化: (batch, data_dim) → (batch, num_tokens, token_dim)
        x = x.view(batch_size, config.num_tokens, config.token_dim)
        x = self.patch_embed(x)  # (batch, num_tokens, d_model)

        # 2. 加位置编码
        x = x + self.pos_embed

        # 3. 时间嵌入
        t_emb = self.time_emb(t * 1000)
        t_emb = self.time_proj(t_emb)  # (batch, time_emb_dim)

        # 4. Transformer 层
        for block in self.blocks:
            if self.dynamic:
                x = block(x, t_emb, use_dynamic=True)
            else:
                x = block(x, t_emb)

        # 5. 输出
        x = self.final_norm(x, t_emb)
        x = self.output_proj(x)  # (batch, num_tokens, token_dim)

        # 6. 还原: (batch, num_tokens, token_dim) → (batch, data_dim)
        x = x.view(batch_size, -1)
        return x


# ==============================================================================
# 训练函数 (使用 Flow Matching)
# ==============================================================================

def train_dit(config: DyDiTConfig = None, dynamic: bool = False,
              epochs: int = 50, batch_size: int = 128, lr: float = 1e-3,
              device: torch.device = None) -> dict:
    """
    训练 DiT / DyDiT 模型 (使用 Flow Matching 框架)。

    为什么用 Flow Matching 而不是 DDPM?
    因为 DiT 论文和后续工作 (如 SD3) 普遍采用 Flow Matching。
    """
    if config is None:
        config = DyDiTConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = "DyDiT" if dynamic else "DiT"
    print(f"[{model_name}] 设备: {device}")

    # 数据
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.data_dim, device)
    print(f"[{model_name}] 合成数据: {data.shape}")

    # 模型
    model = DiffusionTransformer(config, dynamic=dynamic).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name}] 参数量: {num_params:,}")

    # 训练 (Flow Matching)
    history = {'loss': []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            x_0 = data_shuffled[i:i + batch_size]
            bs = x_0.size(0)

            # Flow Matching
            t = torch.rand(bs, device=device)
            noise = torch.randn_like(x_0)
            t_expand = t.unsqueeze(-1)
            x_t = (1 - t_expand) * x_0 + t_expand * noise
            target_v = noise - x_0

            predicted_v = model(x_t, t)
            loss = F.mse_loss(predicted_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    # 采样
    model.eval()
    generated = _sample_euler(model, 16, config.data_dim, 50, device)
    print(f"[{model_name}] 生成样本: 范围 [{generated.min():.3f}, {generated.max():.3f}]")

    return {'model': model, 'history': history, 'generated_samples': generated}


@torch.no_grad()
def _sample_euler(model, num_samples, data_dim, num_steps, device):
    """Euler 采样。"""
    x = torch.randn(num_samples, data_dim, device=device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t = torch.full((num_samples,), t_val, device=device)
        v = model(x, t)
        x = x - dt * v
    return x


def _generate_structured_data(num_samples, dim, device):
    """生成合成数据 ([-1, 1])。"""
    data = torch.zeros(num_samples, dim, device=device)
    segment = dim // 5
    for i in range(num_samples):
        mode = i % 5
        start = mode * segment
        end = start + segment
        data[i, start:end] = torch.rand(segment, device=device) * 0.8 + 0.2
        data[i] += torch.rand(dim, device=device) * 0.05
    return data * 2 - 1


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_dynamic_dit():
    """DiT vs DyDiT 对比演示。"""
    import time

    print("=" * 60)
    print("DiT vs DyDiT (Dynamic DiT) 对比演示")
    print("=" * 60)

    config = DyDiTConfig(
        data_dim=200,
        num_tokens=25,
        token_dim=8,
        d_model=64,        # 演示用较小的模型
        n_heads=4,
        n_layers=3,
        d_ff=128,
        time_emb_dim=32,
        width_ratios=(0.5, 0.75, 1.0),
        token_keep_ratio=0.7,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 训练标准 DiT ----
    print("\n▶ 训练标准 DiT...")
    t0 = time.time()
    dit_results = train_dit(config=config, dynamic=False, epochs=50,
                            batch_size=64, device=device)
    dit_time = time.time() - t0

    # ---- 训练 DyDiT ----
    print("\n▶ 训练 DyDiT (Dynamic DiT)...")
    t0 = time.time()
    dydit_results = train_dit(config=config, dynamic=True, epochs=50,
                              batch_size=64, device=device)
    dydit_time = time.time() - t0

    # ---- 对比 ----
    print("\n" + "=" * 60)
    print("  DiT vs DyDiT 对比结果")
    print("=" * 60)

    dit_model = dit_results['model']
    dydit_model = dydit_results['model']

    dit_params = sum(p.numel() for p in dit_model.parameters())
    dydit_params = sum(p.numel() for p in dydit_model.parameters())

    print(f"\n📊 模型规模:")
    print(f"  DiT:   {dit_params:,} 参数")
    print(f"  DyDiT: {dydit_params:,} 参数")
    print(f"  DyDiT 额外参数: {dydit_params - dit_params:,} (路由器开销)")

    print(f"\n📊 训练时间:")
    print(f"  DiT:   {dit_time:.2f}s")
    print(f"  DyDiT: {dydit_time:.2f}s")

    print(f"\n📊 最终训练损失:")
    print(f"  DiT:   {dit_results['history']['loss'][-1]:.6f}")
    print(f"  DyDiT: {dydit_results['history']['loss'][-1]:.6f}")

    # 采样对比
    real_data = _generate_structured_data(500, config.data_dim, device)
    num_eval = 200

    print(f"\n📊 采样质量对比:")
    for steps in [50, 20, 10]:
        t0 = time.time()
        dit_samples = _sample_euler(dit_model, num_eval, config.data_dim, steps, device)
        t_dit = time.time() - t0

        t0 = time.time()
        dydit_samples = _sample_euler(dydit_model, num_eval, config.data_dim, steps, device)
        t_dydit = time.time() - t0

        dit_err = abs(dit_samples.mean().item() - real_data.mean().item())
        dydit_err = abs(dydit_samples.mean().item() - real_data.mean().item())

        print(f"  {steps:3d} 步 | DiT: err={dit_err:.4f} ({t_dit:.3f}s) | "
              f"DyDiT: err={dydit_err:.4f} ({t_dydit:.3f}s)")

    # DyDiT 计算量分析
    print(f"\n📊 DyDiT 计算量优化:")
    print(f"  宽度比例: {config.width_ratios}")
    print(f"  Token 保留率: {config.token_keep_ratio:.0%}")
    theoretical_savings = 1 - config.token_keep_ratio * config.width_ratios[0]
    print(f"  最大理论 FLOPs 节省: {theoretical_savings:.0%} "
          f"(保留{config.token_keep_ratio:.0%} token × "
          f"{config.width_ratios[0]:.0%} 宽度)")

    print("""
    DiT 架构要点:
      1. adaLN (自适应 LayerNorm) —— 用时间步控制归一化
      2. adaLN-Zero —— gate 参数初始化为 0，稳定训练
      3. Patch Embedding —— 数据切分为 token 序列

    DyDiT 动态优化:
      1. 时间步自适应宽度 —— 高噪声用少维度，低噪声用全维度
      2. Token 选择 —— 只处理重要的 token
      3. 路由器开销极小，但节省大量计算
    """)

    print("[DiT / DyDiT 演示完成]")
    return {'dit': dit_results, 'dydit': dydit_results}


if __name__ == '__main__':
    demo_dynamic_dit()
