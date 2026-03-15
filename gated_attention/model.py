"""
门控注意力 Transformer (Gated Attention Transformer)

来源论文:
    1. "Transformer Quality in Linear Time" (Hua et al., 2022, Google)
       — 提出 GAU (Gated Attention Unit)，将注意力和 FFN 合并为一个门控模块
    2. "GatedAttention: Gated Multi-Head Attention with Sigmoid Gating"
       — 在标准注意力输出上添加 sigmoid 门控，实现稀疏化

核心思想:
    标准 Transformer 中，注意力层和 FFN 层是分离的:
        y = Attention(x) + x     (注意力)
        z = FFN(y) + y           (前馈网络)

    门控注意力将两者合并，用门控机制控制信息流:
        y = Gate(x) ⊙ Attention(x)   (门控 × 注意力)

    这样做的好处:
        1. 参数效率更高 (合并两个子层)
        2. 门控可以学会"忽略"不重要的注意力输出 → 隐式稀疏化
        3. 减少 attention sink 现象 (不需要浪费注意力在 [BOS] 等无意义 token 上)

本文件实现三种变体:
    1. GatedAttentionUnit (GAU): 完整的 GAU，合并注意力和 FFN
    2. SigmoidGatedAttention:    在标准 SDPA 上添加 sigmoid 门控
    3. GatedAttentionTransformer: 完整的语言模型

模型参数设计:
    教学 demo，参数量约 500K-2M，可在 CPU 上运行。
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
class GatedAttnConfig:
    """
    门控注意力 Transformer 配置。

    参数设计:
        - vocab_size=256:   字符级词表 (教学用)
        - d_model=128:      足够小以在 CPU 上快速运行
        - n_layers=4:       4 层门控注意力
        - max_seq_len=128:  较短序列，方便演示
    """
    vocab_size: int = 256       # 词表大小 (字符级)
    d_model: int = 128          # 隐藏层维度
    n_heads: int = 4            # 注意力头数 (仅 SigmoidGated 模式使用)
    d_head: int = 32            # 每头维度 (GAU 模式)
    n_layers: int = 4           # 层数
    max_seq_len: int = 128      # 最大序列长度
    expansion: int = 2          # GAU 内部扩展倍数
    dropout: float = 0.1        # Dropout 概率
    attn_mode: str = "gau"      # 注意力模式: "gau" | "sigmoid_gated" | "standard"


# ==============================================================================
# RoPE 旋转位置编码
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)。

    来源: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

    核心思想:
        不是将位置信息"加"到 token embedding 上，
        而是通过旋转 query 和 key 向量来编码相对位置。

    数学公式:
        对于维度 i 的一对元素 (x_{2i}, x_{2i+1}):
            [cos(mθ_i)  -sin(mθ_i)] [x_{2i}  ]
            [sin(mθ_i)   cos(mθ_i)] [x_{2i+1}]

        其中 m 是位置索引，θ_i = 10000^{-2i/d}
    """

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        # 频率: θ_i = 10000^{-2i/d}
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算 cos/sin 表
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())    # (seq_len, dim)
        self.register_buffer("sin_cached", emb.sin())    # (seq_len, dim)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (cos, sin)，shape = (seq_len, dim)。"""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码。

    参数:
        x:   输入张量, shape = (..., seq_len, dim)
        cos: 余弦表, shape = (seq_len, dim)
        sin: 正弦表, shape = (seq_len, dim)

    返回:
        旋转后的张量, shape 不变
    """
    # 将 x 拆分为两半
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]

    cos = cos[:, :d // 2]
    sin = sin[:, :d // 2]

    # 旋转: [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)

    return rotated


# ==============================================================================
# GAU (Gated Attention Unit)
# ==============================================================================

class GatedAttentionUnit(nn.Module):
    """
    门控注意力单元 (Gated Attention Unit, GAU)。

    来源: "Transformer Quality in Linear Time" (Hua et al., 2022)

    GAU 将传统 Transformer 的两个子层 (Attention + FFN) 合并为一个:

    传统 Transformer:
        ┌─────────────┐     ┌─────────────┐
        │  Attention   │ →→→ │     FFN     │
        │  (Q,K,V)     │     │ (expand→shrink) │
        └─────────────┘     └─────────────┘

    GAU:
        ┌───────────────────────────────────────┐
        │              GAU                       │
        │                                        │
        │   U = SiLU(X·W_U)    ← 门控分支       │
        │   V = SiLU(X·W_V)    ← 值分支         │
        │                                        │
        │   Q = U·W_Q           (低维 query)     │
        │   K = V·W_K           (低维 key)       │
        │   A = relu²(Q·K^T/√d) (注意力权重)     │
        │                                        │
        │   O = (U ⊙ A·V) · W_O  ← 门控输出     │
        └───────────────────────────────────────┘

    关键设计:
        1. U 和 V 先扩展到 e×d (类似 FFN 的扩展)
        2. Q 和 K 是低维的 (d_head << e×d)，节省计算
        3. 用 relu² 代替 softmax (更高效，效果相当)
        4. U ⊙ A·V: 门控 (U) 控制注意力输出 (A·V) 的信息流

    参数:
        config: 模型配置
    """

    def __init__(self, config: GatedAttnConfig):
        super().__init__()
        d = config.d_model
        e = d * config.expansion      # 扩展维度
        s = config.d_head              # 注意力头维度

        # ---- 门控分支 (类似 FFN 的 up projection) ----
        # U = SiLU(X·W_U): 控制哪些信息可以通过
        self.W_U = nn.Linear(d, e, bias=False)

        # ---- 值分支 ----
        # V = SiLU(X·W_V): 提供候选信息
        self.W_V = nn.Linear(d, e, bias=False)

        # ---- 低维注意力 ----
        # Q, K 维度远小于 U, V (节省计算)
        self.W_Q = nn.Linear(e, s, bias=False)
        self.W_K = nn.Linear(e, s, bias=False)

        # 位置编码
        self.rope = RotaryEmbedding(s, config.max_seq_len)

        # ---- 输出投影 ----
        self.W_O = nn.Linear(e, d, bias=False)

        # ---- 归一化 ----
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

        self.scale = 1.0 / math.sqrt(s)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GAU 前向传播。

        参数:
            x:    输入, shape = (B, T, d_model)
            mask: 因果掩码, shape = (T, T) 或 None

        返回:
            输出, shape = (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        # 1. 计算门控向量 U 和值向量 V (扩展维度)
        u = F.silu(self.W_U(x))   # (B, T, e) — 门控
        v = F.silu(self.W_V(x))   # (B, T, e) — 值

        # 2. 低维 query 和 key (用于注意力)
        q = self.W_Q(u)           # (B, T, s)
        k = self.W_K(v)           # (B, T, s)

        # 3. 应用 RoPE 位置编码
        cos, sin = self.rope(q, T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # 4. 计算注意力权重 (使用 relu² 代替 softmax)
        # relu²(x) = max(0, x)²
        # 优点: 比 softmax 更高效; 自带稀疏性 (负值变为 0)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, T, T)

        # 应用因果掩码 (确保不看到未来的 token)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # relu² 激活 — 关键创新
        # softmax 强制所有权重之和为 1, relu² 允许稀疏 (大部分权重为 0)
        attn = F.relu(attn) ** 2
        attn = self.dropout(attn)

        # 5. 注意力加权聚合
        attn_out = torch.matmul(attn, v)  # (B, T, e)

        # 6. 门控: U ⊙ Attention(V)
        # 这是 GAU 的核心 — 用门控 U 控制注意力输出的信息流
        # 类似 GLU (Gated Linear Unit)，但用注意力替代了简单的线性变换
        gated = u * attn_out  # (B, T, e) — 逐元素门控

        # 7. 输出投影 (类似 FFN 的 down projection)
        output = self.W_O(gated)  # (B, T, d)
        output = self.dropout(output)

        return output + residual


# ==============================================================================
# Sigmoid-Gated Multi-Head Attention
# ==============================================================================

class SigmoidGatedAttention(nn.Module):
    """
    Sigmoid 门控多头注意力。

    在标准 SDPA (Scaled Dot-Product Attention) 基础上，
    添加一个可学习的 sigmoid 门控，控制每个注意力头的输出。

    架构:
        标准 MHA:
            y = Concat(head_1, ..., head_h) · W_O
            head_i = softmax(Q_i K_i^T / √d) · V_i

        Sigmoid 门控 MHA:
            y = Concat(g_1 ⊙ head_1, ..., g_h ⊙ head_h) · W_O
            g_i = σ(X · W_g_i)    ← 每个头独立的门控

    门控的作用:
        1. 抑制 attention sink: 无意义 token 的注意力权重被门控"关闭"
        2. 隐式稀疏化: sigmoid 输出接近 0 时，等效于跳过该注意力头
        3. 动态头选择: 不同输入可以激活不同的注意力头

    参数:
        config: 模型配置
    """

    def __init__(self, config: GatedAttnConfig):
        super().__init__()
        d = config.d_model
        h = config.n_heads
        d_k = d // h  # 每头维度

        assert d % h == 0, f"d_model ({d}) 必须能被 n_heads ({h}) 整除"

        self.n_heads = h
        self.d_k = d_k

        # ---- 标准 QKV 投影 ----
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

        # ---- 门控投影 ----
        # 每个头一个独立的门控: σ(X · W_gate)
        # shape: (d_model) → (n_heads * d_k) = (d_model)
        self.W_gate = nn.Linear(d, d, bias=True)
        # bias 初始化为正值，确保训练初期门控大部分打开 (避免信号消失)
        nn.init.constant_(self.W_gate.bias, 1.0)

        # ---- 位置编码 ----
        self.rope = RotaryEmbedding(d_k, config.max_seq_len)

        # ---- 归一化和 FFN ----
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sigmoid 门控注意力前向传播。

        参数:
            x:    输入, shape = (B, T, d_model)
            mask: 因果掩码, shape = (T, T) 或 None

        返回:
            输出, shape = (B, T, d_model)
        """
        B, T, d = x.shape
        h, d_k = self.n_heads, self.d_k

        # ---- 注意力子层 ----
        residual = x
        x_norm = self.norm1(x)

        # 1. QKV 投影 + 分头
        q = self.W_Q(x_norm).view(B, T, h, d_k).transpose(1, 2)  # (B, h, T, d_k)
        k = self.W_K(x_norm).view(B, T, h, d_k).transpose(1, 2)
        v = self.W_V(x_norm).view(B, T, h, d_k).transpose(1, 2)

        # 2. RoPE
        cos, sin = self.rope(q, T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # 3. 标准注意力 (SDPA)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v)  # (B, h, T, d_k)

        # 4. Sigmoid 门控 — 核心创新
        # 计算每个头的门控信号
        gate = torch.sigmoid(self.W_gate(x_norm))             # (B, T, d)
        gate = gate.view(B, T, h, d_k).transpose(1, 2)        # (B, h, T, d_k)

        # 门控: 逐元素控制注意力输出
        # gate ≈ 0 的位置: 注意力输出被抑制 (不重要的信息被过滤)
        # gate ≈ 1 的位置: 注意力输出完整保留
        gated_out = attn_out * gate  # (B, h, T, d_k)

        # 5. 合并头 + 输出投影
        gated_out = gated_out.transpose(1, 2).contiguous().view(B, T, d)
        output = self.W_O(gated_out)
        output = self.dropout(output) + residual

        # ---- FFN 子层 ----
        residual = output
        output = self.ffn(self.norm2(output)) + residual

        return output


# ==============================================================================
# 标准多头注意力 (基线对比)
# ==============================================================================

class StandardAttention(nn.Module):
    """
    标准多头注意力 (无门控，用于基线对比)。

    和 SigmoidGatedAttention 结构相同，但去掉了门控部分。
    用于对比实验，观察门控带来的效果。
    """

    def __init__(self, config: GatedAttnConfig):
        super().__init__()
        d = config.d_model
        h = config.n_heads
        d_k = d // h

        self.n_heads = h
        self.d_k = d_k

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

        self.rope = RotaryEmbedding(d_k, config.max_seq_len)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, d = x.shape
        h, d_k = self.n_heads, self.d_k

        residual = x
        x_norm = self.norm1(x)

        q = self.W_Q(x_norm).view(B, T, h, d_k).transpose(1, 2)
        k = self.W_K(x_norm).view(B, T, h, d_k).transpose(1, 2)
        v = self.W_V(x_norm).view(B, T, h, d_k).transpose(1, 2)

        cos, sin = self.rope(q, T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v)

        # 无门控 — 直接合并
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d)
        output = self.W_O(attn_out)
        output = self.dropout(output) + residual

        residual = output
        output = self.ffn(self.norm2(output)) + residual

        return output


# ==============================================================================
# 完整模型: 门控注意力 Transformer
# ==============================================================================

class GatedAttentionTransformer(nn.Module):
    """
    门控注意力 Transformer 语言模型。

    架构:
        Token Embedding → [GatedAttn Layer × N] → LayerNorm → LM Head

    支持三种注意力模式:
        - "gau":            GAU (Gated Attention Unit)，注意力和 FFN 合二为一
        - "sigmoid_gated":  Sigmoid 门控 MHA + FFN
        - "standard":       标准 MHA + FFN (基线)

    三种模式的对比:
    ┌──────────────────┬──────────────┬──────────────┬──────────────┐
    │                  │  Standard    │ SigmoidGated │     GAU      │
    ├──────────────────┼──────────────┼──────────────┼──────────────┤
    │ 子层数 (每层)     │ 2 (Attn+FFN)│ 2 (Attn+FFN)│ 1 (合并)     │
    │ 门控机制          │ 无           │ σ(X·W_g)    │ U ⊙ A·V     │
    │ 注意力类型        │ softmax      │ softmax      │ relu²        │
    │ 稀疏性           │ 低           │ 中 (sigmoid) │ 高 (relu²)   │
    │ 参数效率          │ 基线         │ +门控参数     │ 更少参数     │
    └──────────────────┴──────────────┴──────────────┴──────────────┘

    参数:
        config: 模型配置
    """

    def __init__(self, config: GatedAttnConfig):
        super().__init__()
        self.config = config

        # Token 嵌入
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # 选择注意力模式
        layer_cls = {
            "gau": GatedAttentionUnit,
            "sigmoid_gated": SigmoidGatedAttention,
            "standard": StandardAttention,
        }[config.attn_mode]

        self.layers = nn.ModuleList([layer_cls(config) for _ in range(config.n_layers)])

        # 最终归一化 + LM Head
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定 (embedding 和 lm_head 共享权重)
        self.lm_head.weight = self.embed.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """权重初始化 (Xavier Uniform)。"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。

        参数:
            input_ids: 输入 token ID, shape = (B, T)
            targets:   目标 token ID, shape = (B, T) 或 None

        返回:
            logits: 预测 logits, shape = (B, T, vocab_size)
            loss:   交叉熵损失 (如果提供了 targets) 或 None
        """
        B, T = input_ids.shape
        device = input_ids.device

        # 因果掩码 (下三角矩阵，确保不看未来)
        mask = torch.tril(torch.ones(T, T, device=device))

        # Token 嵌入
        x = self.embed(input_ids)  # (B, T, d_model)

        # 通过所有层
        for layer in self.layers:
            x = layer(x, mask)

        # 最终归一化 + LM Head
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失 (如果有目标)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        自回归生成。

        参数:
            input_ids:      起始 token, shape = (1, T)
            max_new_tokens: 最多生成 token 数
            temperature:    采样温度 (越高越随机)
            top_k:          Top-K 采样

        返回:
            生成的完整序列, shape = (1, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx = input_ids[:, -self.config.max_seq_len:]

            # 前向传播
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature  # 只取最后一个 token

            # Top-K 采样
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ==============================================================================
# 工具函数
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# 独立运行测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("门控注意力 Transformer — 模型结构测试")
    print("=" * 60)

    for mode in ["gau", "sigmoid_gated", "standard"]:
        config = GatedAttnConfig(attn_mode=mode)
        model = GatedAttentionTransformer(config)
        n_params = count_parameters(model)

        # 测试前向传播
        B, T = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))
        logits, loss = model(input_ids, targets)

        print(f"\n  模式: {mode}")
        print(f"    参数量:     {n_params:,}")
        print(f"    输入 shape: ({B}, {T})")
        print(f"    输出 shape: {logits.shape}")
        print(f"    损失:       {loss.item():.4f}")

    # 测试生成
    print(f"\n  生成测试 (GAU 模式):")
    config = GatedAttnConfig(attn_mode="gau")
    model = GatedAttentionTransformer(config)
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"    输入长度: {prompt.shape[1]}, 生成后长度: {generated.shape[1]}")
    print(f"\n  所有测试通过!")
