"""
Mamba + Transformer 混合模型

参考架构:
  1. Jamba (AI21): 每 8 层中 1 层用 Transformer Attention, 其余用 Mamba
  2. Zamba: 纯 Mamba 骨干 + 单个全局共享注意力模块
  3. Qwen3.5: 每隔一层交替使用 Mamba 和 Transformer

本实现提供三种混合策略:

┌──────────────────────────────────────────────────────────────┐
│ 策略 A: Jamba 风格 (稀疏注意力)                              │
│   [Mamba][Mamba][Mamba][Attn][Mamba][Mamba][Mamba][Attn]    │
│   大部分层用 Mamba (效率), 少数层用 Attention (精确回忆)      │
│   优点: 兼顾效率和长距离精确检索                              │
│                                                              │
│ 策略 B: Qwen 风格 (交替排列)                                 │
│   [Mamba][Attn][Mamba][Attn][Mamba][Attn][Mamba][Attn]      │
│   Mamba 和 Attention 等比例交替                               │
│   优点: 每一层都有两种能力的融合                              │
│                                                              │
│ 策略 C: Zamba 风格 (共享注意力)                              │
│   [Mamba][SharedAttn][Mamba][SharedAttn][Mamba][SharedAttn]  │
│   共享同一个注意力层的权重 (参数更少)                         │
│   优点: 参数效率最高                                          │
└──────────────────────────────────────────────────────────────┘

混合架构的核心洞察:
  - Mamba 擅长: 长序列建模, 高效推理, 捕获渐进式的上下文
  - Attention 擅长: 精确的位置检索 (needle-in-a-haystack), 长距离复制
  - 混合 → 取两者之长, 避各自之短
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from mamba import MambaBlock, MambaConfig, selective_scan
from transformer import CausalSelfAttention, FeedForward, TransformerConfig


# ==============================================================================
# 混合模型配置
# ==============================================================================

@dataclass
class HybridConfig:
    """Mamba + Transformer 混合模型配置"""
    # 通用参数
    d_model: int = 128
    n_layers: int = 8
    vocab_size: int = 1000
    max_seq_len: int = 512
    dropout: float = 0.1

    # Mamba 参数
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Transformer 参数
    n_heads: int = 4
    d_ff: int = 0  # 0 = 4 × d_model

    # 混合策略
    # "jamba": 每 attn_interval 层插入一层 Attention
    # "alternate": Mamba 和 Attention 交替
    # "zamba": Mamba 为主 + 共享 Attention
    strategy: str = "jamba"
    attn_interval: int = 4  # Jamba 风格: 每 N 层一个 Attention

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model

    def get_mamba_config(self) -> MambaConfig:
        return MambaConfig(
            d_model=self.d_model, n_layers=1, vocab_size=self.vocab_size,
            d_state=self.d_state, d_conv=self.d_conv, expand=self.expand,
            max_seq_len=self.max_seq_len, dropout=self.dropout,
        )

    def get_transformer_config(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model, n_layers=1, n_heads=self.n_heads,
            vocab_size=self.vocab_size, max_seq_len=self.max_seq_len,
            d_ff=self.d_ff, dropout=self.dropout,
        )


# ==============================================================================
# Attention Block (用于混合模型)
# ==============================================================================

class AttentionBlock(nn.Module):
    """
    Transformer 注意力块 (Pre-Norm):
        x → LayerNorm → Attention → +x → LayerNorm → FFN → +x
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        t_config = config.get_transformer_config()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(t_config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(t_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# Jamba 风格混合模型
# ==============================================================================

class JambaModel(nn.Module):
    """
    Jamba 风格: 以 Mamba 为主, 每隔 N 层插入一层 Attention。

    例如 n_layers=8, attn_interval=4:
      Layer 0: Mamba
      Layer 1: Mamba
      Layer 2: Mamba
      Layer 3: Attention  ← 每 4 层一个
      Layer 4: Mamba
      Layer 5: Mamba
      Layer 6: Mamba
      Layer 7: Attention  ← 每 4 层一个

    设计理念 (来自 Jamba 论文):
      - 大部分层用 Mamba: 高效处理长序列, 线性复杂度
      - 少数层用 Attention: 提供精确的位置检索能力
      - Attention 层的 KV cache 很小 (只有 1-2 层), 内存友好
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        m_config = config.get_mamba_config()

        # 构建层列表
        self.layers = nn.ModuleList()
        self.layer_types = []  # 记录每层类型

        for i in range(config.n_layers):
            if (i + 1) % config.attn_interval == 0:
                # Attention 层
                self.layers.append(AttentionBlock(config))
                self.layer_types.append("attn")
            else:
                # Mamba 层
                self.layers.append(MambaBlock(m_config))
                self.layer_types.append("mamba")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_layer_info(self) -> str:
        return " → ".join(
            f"[{'A' if t == 'attn' else 'M'}{i}]"
            for i, t in enumerate(self.layer_types)
        )


# ==============================================================================
# Alternate (交替) 风格混合模型
# ==============================================================================

class AlternateModel(nn.Module):
    """
    Qwen 风格: Mamba 和 Attention 层交替排列。

      [Mamba][Attn][Mamba][Attn][Mamba][Attn]...

    每一层都能获得:
      - Mamba: 高效的序列压缩和长距离依赖
      - Attention: 精确的位置检索和全局关联

    参考 Qwen3.5 的做法: 每隔一层替换为 Mamba。
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        m_config = config.get_mamba_config()

        self.layers = nn.ModuleList()
        self.layer_types = []

        for i in range(config.n_layers):
            if i % 2 == 0:
                self.layers.append(MambaBlock(m_config))
                self.layer_types.append("mamba")
            else:
                self.layers.append(AttentionBlock(config))
                self.layer_types.append("attn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_layer_info(self) -> str:
        return " → ".join(
            f"[{'A' if t == 'attn' else 'M'}{i}]"
            for i, t in enumerate(self.layer_types)
        )


# ==============================================================================
# Zamba 风格: 共享注意力层
# ==============================================================================

class ZambaModel(nn.Module):
    """
    Zamba 风格: Mamba 骨干 + 单个共享的全局注意力块 (GSA)。

    核心假设: "One attention layer is all you need"
    同一个 Attention 层在多个位置重复使用, 但有独立的 KV 投影。

      [Mamba₀] → [SharedAttn] → [Mamba₁] → [SharedAttn] → [Mamba₂] → ...

    共享权重大幅减少参数量, 同时保持精确检索能力。
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        m_config = config.get_mamba_config()

        # Mamba 层 (每层独立)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(m_config) for _ in range(config.n_layers)
        ])

        # 单个共享注意力层 (参数共享!)
        self.shared_attn = AttentionBlock(config)

        # 在哪些 Mamba 层之后插入共享注意力 (每隔 2 层)
        self.attn_positions = set(range(1, config.n_layers, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, mamba_layer in enumerate(self.mamba_layers):
            x = mamba_layer(x)
            if i in self.attn_positions:
                x = self.shared_attn(x)  # 共享同一个 Attention
        return x

    def get_layer_info(self) -> str:
        parts = []
        for i in range(self.config.n_layers):
            parts.append(f"[M{i}]")
            if i in self.attn_positions:
                parts.append("[SA]")  # SA = Shared Attention
        return " → ".join(parts)


# ==============================================================================
# 统一混合语言模型
# ==============================================================================

class HybridLM(nn.Module):
    """
    Mamba + Transformer 混合语言模型。

    支持三种混合策略:
      - "jamba":     Jamba 风格 (稀疏注意力)
      - "alternate": Qwen 风格 (交替排列)
      - "zamba":     Zamba 风格 (共享注意力)
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # 位置嵌入 (因为有 Attention 层需要位置信息)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # 根据策略选择骨干
        if config.strategy == "jamba":
            self.backbone = JambaModel(config)
        elif config.strategy == "alternate":
            self.backbone = AlternateModel(config)
        elif config.strategy == "zamba":
            self.backbone = ZambaModel(config)
        else:
            raise ValueError(f"未知策略: {config.strategy}")

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)
        # 注意: 不使用权重绑定, 因为维度已经匹配
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device

        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(L, device=device))
        x = self.drop(tok + pos)

        x = self.backbone(x)

        x = self.norm(x)
        logits = self.lm_head(x)

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
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def get_architecture_info(self) -> str:
        return self.backbone.get_layer_info()


# ==============================================================================
# 演示
# ==============================================================================

def demo_hybrid():
    """混合模型演示。"""
    print("=" * 60)
    print("  Mamba + Transformer 混合模型演示")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    strategies = ["jamba", "alternate", "zamba"]
    batch_size, seq_len = 4, 64

    for strategy in strategies:
        print(f"\n{'─' * 60}")
        print(f"策略: {strategy}")
        print(f"{'─' * 60}")

        config = HybridConfig(
            d_model=64, n_layers=8, vocab_size=256,
            max_seq_len=128, n_heads=4,
            d_state=16, d_conv=4, expand=2,
            strategy=strategy, attn_interval=4,
        )

        model = HybridLM(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params:,}")
        print(f"  层结构: {model.get_architecture_info()}")

        # 前向传播
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        logits, loss = model(x, targets)
        print(f"  前向传播: loss = {loss.item():.4f}")

        # 快速训练
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(50):
            pattern = torch.randint(0, 50, (batch_size, 8), device=device)
            data = pattern.repeat(1, seq_len // 8 + 1)[:, :seq_len + 1]
            inp, tgt = data[:, :-1], data[:, 1:]
            _, loss = model(inp, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  50 步训练后: loss = {loss.item():.4f}")

    # 参数量对比
    print(f"\n{'─' * 60}")
    print("混合策略参数量对比:")
    print(f"{'─' * 60}")
    for strategy in strategies:
        config = HybridConfig(
            d_model=64, n_layers=8, vocab_size=256,
            max_seq_len=128, strategy=strategy, attn_interval=4,
        )
        model = HybridLM(config)
        n = sum(p.numel() for p in model.parameters())
        n_attn_layers = sum(1 for t in model.backbone.layer_types
                            if t == "attn") if hasattr(model.backbone, 'layer_types') else "shared"
        print(f"  {strategy:12s}: {n:>8,} 参数, Attn 层数: {n_attn_layers}")

    print("\n[混合模型演示完成]")


if __name__ == '__main__':
    demo_hybrid()
