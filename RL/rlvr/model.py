"""
轻量级 GPT 语言模型 —— 用于 RLVR 训练

设计目标:
    1. 足够小，能在 Mac CPU/MPS 上流畅运行（~1M 参数）
    2. 架构完整，包含真实 GPT 的所有关键组件
    3. 所有 tensor 操作严格保持 device 一致性

架构概览:
    ┌────────────────────────────────────────┐
    │           Token Embedding              │
    │         + Position Embedding           │
    ├────────────────────────────────────────┤
    │     Transformer Decoder Block × N      │
    │  ┌──────────────────────────────────┐  │
    │  │  Multi-Head Causal Self-Attention │  │
    │  │  + Residual + LayerNorm          │  │
    │  ├──────────────────────────────────┤  │
    │  │  Feed-Forward Network (FFN)      │  │
    │  │  + Residual + LayerNorm          │  │
    │  └──────────────────────────────────┘  │
    ├────────────────────────────────────────┤
    │           Final LayerNorm              │
    ├────────────────────────────────────────┤
    │           LM Head (→ vocab)            │
    └────────────────────────────────────────┘

与 GRPO/PPO 实现的区别:
    RLVR 与 GRPO 类似，不需要 Critic (Value Model)，因为:
    1. 奖励来自可验证的规则（如数学答案正确性），无需学习
    2. 优势函数通过组内相对比较计算
    因此本文件只包含策略模型，与 GRPO 的 model.py 结构一致。
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
class GPTConfig:
    """
    GPT 模型配置。

    默认参数设计为 ~1M 参数量，可在 Mac 上快速训练和推理。
    作为参考:
        - GPT-2 Small:  124M 参数 (12层, 768维, 12头)
        - 我们的模型:   ~1M 参数 (4层, 128维, 4头)
    """
    vocab_size: int = 256       # 词表大小（字节级 token，256 个可能值）
    max_seq_len: int = 128      # 最大序列长度
    d_model: int = 128          # 隐藏层维度
    n_heads: int = 4            # 注意力头数
    n_layers: int = 4           # Transformer 层数
    d_ff: int = 512             # FFN 内部维度 (通常 4 × d_model)
    dropout: float = 0.1        # Dropout 率
    bias: bool = False          # 是否在线性层中使用 bias


# ==============================================================================
# 多头因果自注意力
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力 (Multi-Head Causal Self-Attention)。

    因果注意力确保位置 t 只能关注 t 及之前的位置 (≤ t)，
    这是自回归语言模型的核心约束。

    数学公式:
        Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # QKV 合并投影
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 预计算因果掩码 (下三角矩阵)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale

        # 因果掩码
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


# ==============================================================================
# 前馈网络 (Feed-Forward Network)
# ==============================================================================

class FeedForward(nn.Module):
    """
    位置级前馈网络 (Position-wise FFN)。

    结构: Linear → GELU → Linear → Dropout
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    单个 Transformer 解码器块 (Pre-Norm 架构)。

        x → LayerNorm → Attention → + x (残差) → LayerNorm → FFN → + x (残差)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ==============================================================================
# GPT 语言模型 (Policy Model)
# ==============================================================================

class GPTLanguageModel(nn.Module):
    """
    完整的 GPT 语言模型，作为 RLVR 中的策略模型。

    在 RLVR 中的角色:
        - 策略模型 π_θ: 根据 prompt (数学题) 生成回复 (答案)
        - 输出: 词表上的概率分布 → 用于采样 token
        - RLVR 优化目标: 最大化可验证奖励（答案正确性）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            input_ids: 输入 token 序列, shape = (B, T)

        返回:
            logits: 词表上的 logits, shape = (B, T, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.config.max_seq_len, f"序列长度 {T} 超过最大限制 {self.config.max_seq_len}"

        positions = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成。

        参数:
            input_ids:      prompt 序列, shape = (B, T)
            max_new_tokens: 最大生成 token 数
            temperature:    采样温度
            top_k:          Top-K 采样

        返回:
            生成的完整序列 (prompt + response)
        """
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits = self.forward(idx_cond)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


# ==============================================================================
# 工具函数
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = GPTConfig()

    model = GPTLanguageModel(config)
    print(f"模型参数量: {count_parameters(model):,}")

    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(dummy_input)
    print(f"输入 shape:   {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"生成 shape:   {generated.shape}")
    print(f"生成序列:     {generated[0].tolist()}")
