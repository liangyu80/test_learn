"""
Mamba: 选择性状态空间模型 (Selective State Space Model, S6)

论文: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
作者: Albert Gu, Tri Dao (2023)

核心创新: 将 SSM 的参数从 "固定" 变为 "输入相关 (input-dependent)"
使模型能根据当前输入选择性地记忆或遗忘历史信息

┌──────────────────────────────────────────────────────────────┐
│                   State Space Model (SSM)                     │
│                                                               │
│  连续形式:                                                     │
│    h'(t) = A·h(t) + B·x(t)       (状态转移方程)              │
│    y(t)  = C·h(t)                 (观测方程)                  │
│                                                               │
│  离散化 (ZOH):                                                │
│    Ā = exp(Δ·A)                                               │
│    B̄ = (Δ·A)⁻¹ · (exp(Δ·A) - I) · Δ·B  ≈ Δ·B              │
│    h_t = Ā·h_{t-1} + B̄·x_t                                  │
│    y_t = C·h_t                                                │
│                                                               │
│  Mamba 的关键区别 (选择性机制):                                │
│    Δ = softplus(Linear(x))    ← 输入相关的步长                │
│    B = Linear(x)              ← 输入相关的输入矩阵            │
│    C = Linear(x)              ← 输入相关的输出矩阵            │
│    A = diag(exp(-exp(log_A))) ← 可学习但固定的衰减            │
│                                                               │
│  这使得模型可以:                                               │
│    - 看到重要 token → 增大 Δ → 更多地更新状态                 │
│    - 看到无关 token → 减小 Δ → 保持状态不变                   │
│    - 通过 B,C 控制写入/读取什么信息                            │
└──────────────────────────────────────────────────────────────┘

与 Transformer 对比:
  - Transformer: O(N²) 注意力, 完美的 "位置寻址" 能力
  - Mamba:       O(N)  线性, 通过选择性压缩历史到固定大小状态
  - Mamba 在长序列上更高效, 但可能在需要精确回忆的任务上弱于 Transformer

Mamba Block 结构:
  input → [Linear (expand)] → [Conv1D] → [SiLU] → [SSM] → × → [Linear (project)] → output
  input → [Linear (expand)] ─────────→ [SiLU] ──────────┘  (门控分支)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class MambaConfig:
    """Mamba 模型配置"""
    d_model: int = 128          # 模型维度
    n_layers: int = 4           # Mamba 层数
    vocab_size: int = 1000      # 词表大小
    d_state: int = 16           # SSM 状态维度 N (论文中 N=16)
    d_conv: int = 4             # 局部卷积核大小
    expand: int = 2             # 内部扩展倍数 (d_inner = expand × d_model)
    dt_rank: str = "auto"       # Δ 投影的秩 ("auto" = ceil(d_model/16))
    max_seq_len: int = 512      # 最大序列长度
    dropout: float = 0.1        # Dropout 率

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# ==============================================================================
# 选择性 SSM (Selective Scan)
# ==============================================================================

def selective_scan(x: torch.Tensor, delta: torch.Tensor,
                   A: torch.Tensor, B: torch.Tensor,
                   C: torch.Tensor, D: torch.Tensor
                   ) -> torch.Tensor:
    """
    选择性扫描算法 (Selective Scan / S6)。

    这是 Mamba 的核心: 输入相关的离散化 + 循环扫描。

    Args:
        x:     输入序列         (batch, seq_len, d_inner)
        delta: 步长参数         (batch, seq_len, d_inner)  ← 输入相关!
        A:     状态转移矩阵     (d_inner, d_state)         ← 可学习固定参数
        B:     输入矩阵         (batch, seq_len, d_state)  ← 输入相关!
        C:     输出矩阵         (batch, seq_len, d_state)  ← 输入相关!
        D:     跳跃连接参数     (d_inner,)

    Returns:
        y: 输出序列 (batch, seq_len, d_inner)

    算法流程:
        对每个时间步 t:
            1. 离散化: Ā = exp(Δ·A),  B̄ = Δ·B
            2. 状态更新: h_t = Ā·h_{t-1} + B̄·x_t
            3. 输出: y_t = C·h_t + D·x_t
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # 离散化 A: Ā = exp(Δ·A)
    # delta: (B, L, D) → (B, L, D, 1)
    # A: (D, N) → (1, 1, D, N)
    # deltaA: (B, L, D, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

    # 离散化 B: B̄ = Δ·B (简化的 ZOH 近似)
    # delta: (B, L, D) → (B, L, D, 1)
    # B: (B, L, N) → (B, L, 1, N)
    # deltaB: (B, L, D, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

    # 循环扫描 (这里用简单循环实现; 实际 Mamba 用并行扫描/CUDA kernel)
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    outputs = []

    for t in range(seq_len):
        # h_t = Ā_t · h_{t-1} + B̄_t · x_t
        # deltaA[:, t]: (B, D, N)
        # h: (B, D, N)
        h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)

        # y_t = C_t · h_t
        # C[:, t]: (B, N) → (B, 1, N)
        # h: (B, D, N)
        # 内积 → (B, D)
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        outputs.append(y_t)

    # (B, L, D)
    y = torch.stack(outputs, dim=1)

    # 跳跃连接: y += D · x
    y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y


# ==============================================================================
# Mamba Block
# ==============================================================================

class MambaBlock(nn.Module):
    """
    Mamba 块: 替代 Transformer 中的 Self-Attention + FFN。

    结构:
        x → Norm → [分两路]
                    ├─ Linear(expand) → Conv1D → SiLU → SSM ──→ ×  → Linear(project) → + → output
                    └─ Linear(expand) ────────→ SiLU ──────────┘      (门控)            │
        x ─────────────────────────────────────────────────────────────────────── residual ┘

    门控机制 (类似 GLU): 让模型控制信息流通量
    Conv1D: 捕获局部依赖 (类似位置编码的作用)
    SSM: 捕获长距离依赖 (通过选择性状态压缩)
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        d_inner = config.d_inner
        d_state = config.d_state
        d_conv = config.d_conv
        dt_rank = config.dt_rank

        # 输入归一化
        self.norm = nn.LayerNorm(d)

        # 投影: d_model → 2 × d_inner (SSM 分支 + 门控分支)
        self.in_proj = nn.Linear(d, d_inner * 2, bias=False)

        # 局部因果卷积 (在 SSM 之前捕获局部模式)
        # groups=d_inner: depthwise conv, 每个通道独立卷积
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # 因果填充
            groups=d_inner,
            bias=True,
        )

        # SSM 参数投影
        # x → (B, C, Δ): 将输入映射到 SSM 的输入相关参数
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # Δ 投影: 低秩投影 → d_inner 维的 Δ
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A 参数: 固定的状态转移矩阵 (通过 log 参数化确保负值)
        # 初始化为 -log(1, 2, ..., N) (S4D-Lin 初始化)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_inner, -1)  # (d_inner, d_state)
        self.log_A = nn.Parameter(torch.log(A))

        # D 参数: 跳跃连接 (类似 residual)
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # 投影到 2×d_inner, 分为 SSM 分支和门控分支
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # ---- SSM 分支 ----
        # Conv1D (因果卷积, 捕获局部模式)
        x_ssm = x_ssm.transpose(1, 2)  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]  # 因果: 截断未来
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # 生成输入相关的 SSM 参数 (B, C, Δ)
        x_proj = self.x_proj(x_ssm)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_proj,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )

        # Δ: 低秩投影 + softplus (确保正值)
        delta = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # A: exp(-exp(log_A)) 确保 A 为负 (稳定衰减)
        A = -torch.exp(self.log_A)  # (d_inner, d_state), 负值

        # 选择性扫描
        y = selective_scan(x_ssm, delta, A, B, C, self.D)

        # ---- 门控 ----
        # z 分支经过 SiLU 激活后与 SSM 输出相乘 (GLU 变体)
        y = y * F.silu(z)

        # 输出投影
        output = self.out_proj(y)
        output = self.dropout(output)

        return output + residual


# ==============================================================================
# Mamba 语言模型
# ==============================================================================

class MambaLM(nn.Module):
    """
    基于 Mamba 的语言模型。

    用同构的 Mamba Block 堆叠 (不需要位置编码!):
        Token Embedding → [MambaBlock] × N_layers → LayerNorm → LM Head

    关键特点:
      - 无需位置编码 (因果卷积 + SSM 自然具有位置感知)
      - 推理时 O(1) 每步 (只需维护 h 状态, 不需 KV cache)
      - 训练时 O(N) (并行扫描, 对比 Transformer 的 O(N²))
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Token 嵌入 (不需要位置编码)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 堆叠 Mamba Block
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layers)
        ])

        # 最终归一化
        self.norm = nn.LayerNorm(config.d_model)

        # 语言模型头 (与嵌入权重绑定)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # 权重绑定

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len)
            targets:   (batch, seq_len) 用于计算损失
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   标量损失 (如果提供 targets)
        """
        x = self.embedding(input_ids)  # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0) -> torch.Tensor:
        """自回归生成 (简化版, 非增量推理)。"""
        for _ in range(max_new_tokens):
            # 截断到最大长度
            x = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==============================================================================
# 演示
# ==============================================================================

def demo_mamba():
    """Mamba 模型演示。"""
    print("=" * 60)
    print("  Mamba (Selective State Space Model) 演示")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MambaConfig(
        d_model=64, n_layers=4, vocab_size=256,
        d_state=16, d_conv=4, expand=2, max_seq_len=128,
    )

    model = MambaLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")
    print(f"配置: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"d_state={config.d_state}, d_inner={config.d_inner}")

    # ---- 1. 前向传播测试 ----
    print("\n--- 前向传播测试 ---")
    batch_size, seq_len = 4, 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    logits, loss = model(x, targets)
    print(f"输入: {x.shape}, 输出: {logits.shape}, 损失: {loss.item():.4f}")

    # ---- 2. 选择性扫描可视化 ----
    print("\n--- 选择性机制分析 ---")
    print("Mamba 的核心: 参数 Δ, B, C 随输入变化")
    print("  Δ 大 → 状态更新幅度大 (记住当前输入)")
    print("  Δ 小 → 状态几乎不变 (忽略当前输入)")

    # 展示不同输入对 Δ 的影响
    layer = model.layers[0]
    with torch.no_grad():
        test_x = model.embedding(x[:1])
        test_x = layer.norm(test_x)
        xz = layer.in_proj(test_x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm_t = x_ssm.transpose(1, 2)
        x_ssm_t = layer.conv1d(x_ssm_t)[:, :, :seq_len].transpose(1, 2)
        x_ssm_t = F.silu(x_ssm_t)
        x_proj = layer.x_proj(x_ssm_t)
        dt_raw = x_proj[:, :, :config.dt_rank]
        delta = F.softplus(layer.dt_proj(dt_raw))
        print(f"\n  Δ 统计 (第 1 层):")
        print(f"    均值: {delta.mean().item():.4f}")
        print(f"    标准差: {delta.std().item():.4f}")
        print(f"    范围: [{delta.min().item():.4f}, {delta.max().item():.4f}]")

    # ---- 3. 复杂度分析 ----
    print("\n--- 复杂度对比 ---")
    print(f"  Transformer Attention: O(N²·d) = O({seq_len}²·{config.d_model}) "
          f"= O({seq_len**2 * config.d_model:,})")
    print(f"  Mamba SSM:             O(N·d·n) = O({seq_len}·{config.d_inner}·{config.d_state}) "
          f"= O({seq_len * config.d_inner * config.d_state:,})")
    ratio = (seq_len ** 2 * config.d_model) / (seq_len * config.d_inner * config.d_state)
    print(f"  Mamba 效率提升: {ratio:.1f}× (序列越长优势越大)")

    # ---- 4. 训练演示 ----
    print("\n--- 简单训练演示 ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 生成简单模式数据: 重复序列
    for step in range(100):
        # 数据: [a, b, c, d, a, b, c, d, ...] 重复模式
        pattern = torch.randint(0, 50, (batch_size, 8), device=device)
        data = pattern.repeat(1, seq_len // 8 + 1)[:, :seq_len + 1]
        inp = data[:, :-1]
        tgt = data[:, 1:]

        logits, loss = model(inp, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: loss = {loss.item():.4f}")

    print("\n[Mamba 演示完成]")
    return model, config


if __name__ == '__main__':
    demo_mamba()
