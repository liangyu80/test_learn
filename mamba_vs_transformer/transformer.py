"""
标准 Transformer 语言模型 (作为 Mamba 的对比基线)

经典 Transformer 架构 (Decoder-only, GPT 风格):
  Token Embedding + Position Embedding → [TransformerBlock] × N → LayerNorm → LM Head

TransformerBlock:
  x → LayerNorm → Multi-Head Attention → Dropout → +residual
    → LayerNorm → FFN (expand → GELU → project) → Dropout → +residual

与 Mamba 的关键对比:
┌─────────────────────┬────────────────────┬────────────────────┐
│                     │ Transformer        │ Mamba              │
├─────────────────────┼────────────────────┼────────────────────┤
│ 序列建模方式         │ Self-Attention     │ Selective SSM      │
│ 时间复杂度 (训练)    │ O(N²·d)            │ O(N·d·n)           │
│ 空间复杂度 (推理)    │ O(N) KV cache      │ O(1) 状态          │
│ 位置编码            │ 需要 (sin/学习/RoPE)│ 不需要             │
│ 长距离依赖          │ 精确 (直接 attend)  │ 压缩 (通过状态)    │
│ 推理延迟            │ 随序列增长          │ 恒定              │
│ 并行训练            │ 完全并行            │ 并行扫描           │
└─────────────────────┴────────────────────┴────────────────────┘
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
class TransformerConfig:
    """Transformer 模型配置"""
    d_model: int = 128          # 模型维度
    n_layers: int = 4           # Transformer 层数
    n_heads: int = 4            # 注意力头数
    vocab_size: int = 1000      # 词表大小
    max_seq_len: int = 512      # 最大序列长度
    d_ff: int = 0               # FFN 中间维度 (0 = 4 × d_model)
    dropout: float = 0.1        # Dropout 率

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


# ==============================================================================
# 多头因果自注意力
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力 (Causal Multi-Head Self-Attention)。

    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    因果掩码确保每个位置只能看到自己和之前的 token。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # Q, K, V 合并投影
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # 因果掩码 (下三角矩阵)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # QKV 投影
        qkv = self.qkv_proj(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # 分头: (B, L, D) → (B, n_heads, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # 注意力分数: QK^T / √d_k
        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # 因果掩码: 屏蔽未来位置
        attn = attn.masked_fill(
            self.causal_mask[:, :, :L, :L] == 0, float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        out = torch.matmul(attn, v)  # (B, H, L, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.resid_drop(self.out_proj(out))


# ==============================================================================
# 前馈网络 (FFN)
# ==============================================================================

class FeedForward(nn.Module):
    """
    前馈网络: Linear → GELU → Linear

    作用: 对每个位置独立地进行非线性变换 (点级变换)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    标准 Transformer 块 (Pre-Norm 风格):
        x → LayerNorm → Attention → +x → LayerNorm → FFN → +x
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# Transformer 语言模型
# ==============================================================================

class TransformerLM(nn.Module):
    """
    基于 Transformer 的语言模型 (GPT 风格)。

    Token Embedding + Position Embedding → [TransformerBlock] × N → Norm → LM Head
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token 嵌入 + 位置嵌入
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        self.drop = nn.Dropout(config.dropout)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重绑定

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device

        # Token 嵌入 + 位置嵌入
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(L, device=device))
        x = self.drop(tok + pos)

        for layer in self.layers:
            x = layer(x)

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
        """自回归生成。"""
        for _ in range(max_new_tokens):
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

def demo_transformer():
    """Transformer 模型演示。"""
    print("=" * 60)
    print("  Transformer (Causal LM) 演示")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TransformerConfig(
        d_model=64, n_layers=4, n_heads=4,
        vocab_size=256, max_seq_len=128,
    )

    model = TransformerLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")
    print(f"配置: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, d_ff={config.d_ff}")

    # 前向传播
    batch_size, seq_len = 4, 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    logits, loss = model(x, targets)
    print(f"\n输入: {x.shape}, 输出: {logits.shape}, 损失: {loss.item():.4f}")

    # 训练
    print("\n--- 简单训练演示 ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(100):
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

    print("\n[Transformer 演示完成]")
    return model, config


if __name__ == '__main__':
    demo_transformer()
