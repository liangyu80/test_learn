"""
位置编码 (Positional Encoding) 全面对比

实现以下位置编码方法:
1. Sinusoidal PE      - Transformer 原始正弦余弦位置编码
2. Learned PE         - 可学习的绝对位置嵌入
3. RoPE               - 旋转位置编码 (Rotary Position Embedding)
4. ALiBi              - 线性偏置注意力 (Attention with Linear Biases)
5. Relative PE        - 相对位置编码 (Shaw et al.)
6. Kerple             - 核化相对位置编码
7. FIRE               - 函数化插值相对位置编码
8. CoPE               - 上下文位置编码 (Contextual Position Encoding)
9. NoPE               - 无显式位置编码基线

每种编码都集成到一个轻量 Transformer block 中进行公平对比。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Sinusoidal PE (Vaswani et al., 2017)
# =============================================================================
class SinusoidalPE(nn.Module):
    """
    经典正弦余弦位置编码
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算频率分母: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> 加上位置编码"""
        return x + self.pe[:, : x.size(1), :]


# =============================================================================
# 2. Learned PE (可学习绝对位置嵌入)
# =============================================================================
class LearnedPE(nn.Module):
    """
    可学习位置嵌入 —— 每个位置对应一个可训练向量
    GPT-2, BERT 使用此方案
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


# =============================================================================
# 3. RoPE (Su et al., 2021 — Rotary Position Embedding)
# =============================================================================
class RoPE(nn.Module):
    """
    旋转位置编码 —— 通过复数旋转将位置信息注入 Q/K

    核心思想: 将 Q/K 的每两个维度视为复数, 乘以 e^{i * pos * theta}
    使得 Q_m · K_n 的内积只依赖于相对位置 (m - n)

    优势: 天然具有相对位置性质, 支持长度外推
    """

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        # 频率: theta_i = 1 / base^(2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        """预计算 cos/sin 缓存"""
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, d_head)
        self.register_buffer("cos_cache", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将 (x1, x2) 变为 (-x2, x1), 实现复数旋转"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (batch, n_heads, seq_len, d_head)
        返回旋转后的 q, k
        """
        seq_len = q.size(2)
        cos = self.cos_cache[:, :, :seq_len, :]
        sin = self.sin_cache[:, :, :seq_len, :]
        # 旋转公式: x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# =============================================================================
# 4. ALiBi (Press et al., 2022 — Attention with Linear Biases)
# =============================================================================
class ALiBi(nn.Module):
    """
    线性偏置注意力 —— 不修改 Q/K/V, 直接在注意力分数上加线性距离偏置

    bias(q_pos, k_pos) = -m * |q_pos - k_pos|
    其中 m 是每个头的斜率, 按几何级数递减: m_i = 2^{-8i/n_heads}

    优势: 零额外参数, 训练短序列可外推到长序列
    """

    def __init__(self, n_heads: int, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        # 每个头的斜率: 2^{-8/n_heads}, 2^{-16/n_heads}, ...
        slopes = torch.tensor(
            [2 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)]
        )
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))
        # 预计算距离矩阵
        positions = torch.arange(max_len)
        # |q_pos - k_pos| 距离矩阵
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        self.register_buffer("distance", distance.unsqueeze(0).unsqueeze(0))

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        attn_scores: (batch, n_heads, seq_len, seq_len)
        返回加了 ALiBi 偏置的注意力分数
        """
        seq_len = attn_scores.size(-1)
        bias = -self.slopes * self.distance[:, :, :seq_len, :seq_len]
        return attn_scores + bias


# =============================================================================
# 5. Relative PE (Shaw et al., 2018)
# =============================================================================
class RelativePE(nn.Module):
    """
    相对位置编码 —— 在注意力计算中加入可学习的相对位置嵌入

    Attention(Q, K) = QK^T + Q * R_{k-q}^T
    其中 R 是可学习的相对位置嵌入, 裁剪到 [-max_rel, max_rel]

    优势: 直接建模相对距离关系
    """

    def __init__(self, d_head: int, max_rel: int = 128):
        super().__init__()
        self.max_rel = max_rel
        # 可学习的相对位置嵌入, 共 2*max_rel+1 个
        self.rel_embedding = nn.Embedding(2 * max_rel + 1, d_head)

    def forward(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        q: (batch, n_heads, seq_len, d_head)
        返回相对位置偏置: (batch, n_heads, seq_len, seq_len)
        """
        # 构建相对位置索引矩阵
        positions = torch.arange(seq_len, device=q.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        # 裁剪到 [-max_rel, max_rel] 并偏移到 [0, 2*max_rel]
        rel_pos = rel_pos.clamp(-self.max_rel, self.max_rel) + self.max_rel
        rel_emb = self.rel_embedding(rel_pos)  # (seq, seq, d_head)
        # Q 与相对位置嵌入的内积
        # q: (B, H, S, D), rel_emb: (S, S, D)
        bias = torch.einsum("bhqd,qkd->bhqk", q, rel_emb)
        return bias


# =============================================================================
# 6. Kerple (Chi et al., 2022 — Kernelized Relative PE)
# =============================================================================
class Kerple(nn.Module):
    """
    核化相对位置编码 —— 使用可学习的核函数建模位置关系

    bias(i, j) = -r1 * log(1 + r2 * |i - j|)
    其中 r1, r2 > 0 是每个头的可学习参数

    优势: 比固定斜率 (ALiBi) 更灵活, 可学习衰减曲线
    """

    def __init__(self, n_heads: int, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        # 可学习参数 r1, r2 (对数空间, 保证正值)
        self.log_r1 = nn.Parameter(torch.zeros(n_heads))
        self.log_r2 = nn.Parameter(torch.zeros(n_heads))
        # 预计算距离矩阵
        positions = torch.arange(max_len).float()
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        self.register_buffer("distance", distance)

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        attn_scores: (batch, n_heads, seq_len, seq_len)
        """
        seq_len = attn_scores.size(-1)
        r1 = self.log_r1.exp().view(1, -1, 1, 1)  # (1, H, 1, 1)
        r2 = self.log_r2.exp().view(1, -1, 1, 1)
        dist = self.distance[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        # 核化偏置: -r1 * log(1 + r2 * distance)
        bias = -r1 * torch.log1p(r2 * dist)
        return attn_scores + bias


# =============================================================================
# 7. FIRE (Li et al., 2023 — Functional Interpolation Relative PE)
# =============================================================================
class FIRE(nn.Module):
    """
    函数化插值相对位置编码 —— 用 MLP 将连续化的相对位置映射为偏置

    步骤:
    1. 将相对位置归一化到 [0, 1]: ĩ = (k-q) / max(seq_len, threshold)
    2. 用 MLP 将归一化位置映射为标量偏置: bias = MLP(ĩ)

    优势: 连续映射使长度外推更平滑, 不依赖离散位置嵌入表
    """

    def __init__(self, n_heads: int, hidden_dim: int = 32, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.threshold = max_len  # 归一化阈值
        # 每个头一个 MLP: 输入 1 维相对位置 -> 输出 1 维偏置
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads),
        )

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        attn_scores: (batch, n_heads, seq_len, seq_len)
        """
        seq_len = attn_scores.size(-1)
        positions = torch.arange(seq_len, device=attn_scores.device).float()
        # 相对位置矩阵, 归一化到 [0, 1]
        rel_pos = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        normalizer = max(seq_len, self.threshold)
        rel_pos_norm = (rel_pos / normalizer).unsqueeze(-1)  # (S, S, 1)
        # MLP 映射
        bias = self.mlp(rel_pos_norm)  # (S, S, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, S, S)
        return attn_scores + bias


# =============================================================================
# 8. CoPE (Golovneva et al., 2024 — Contextual Position Encoding)
# =============================================================================
class CoPE(nn.Module):
    """
    上下文位置编码 —— 位置由注意力权重动态决定, 而非固定索引

    步骤:
    1. 计算注意力门: gates = sigmoid(Q @ K^T)
    2. 上下文位置: pos_i = sum_j(gates_{ij} * j)  (加权位置)
    3. 位置嵌入: pe = 插值查找(pos_i)
    4. 偏置: bias = Q @ pe^T

    优势: 位置感知上下文相关, 对 token 密度变化更鲁棒
    """

    def __init__(self, d_head: int, max_len: int = 2048):
        super().__init__()
        self.d_head = d_head
        # 位置嵌入表 (用于插值)
        self.pos_embedding = nn.Embedding(max_len, d_head)
        self.max_len = max_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q, k: (batch, n_heads, seq_len, d_head)
        返回上下文位置偏置: (batch, n_heads, seq_len, seq_len)
        """
        seq_len = q.size(2)
        # 步骤1: 注意力门
        gates = torch.sigmoid(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        )
        # 步骤2: 上下文位置 —— 门加权的位置期望
        positions = torch.arange(seq_len, device=q.device).float()
        ctx_pos = torch.matmul(gates, positions)  # (B, H, S)

        # 步骤3: 位置嵌入插值
        ctx_pos_clamped = ctx_pos.clamp(0, self.max_len - 2)
        pos_floor = ctx_pos_clamped.long()
        pos_frac = (ctx_pos_clamped - pos_floor.float()).unsqueeze(-1)  # (B,H,S,1)
        emb_floor = self.pos_embedding(pos_floor)  # (B,H,S,D)
        emb_ceil = self.pos_embedding((pos_floor + 1).clamp(max=self.max_len - 1))
        pos_emb = emb_floor + pos_frac * (emb_ceil - emb_floor)  # 线性插值

        # 步骤4: Q 与上下文位置嵌入的内积作为偏置
        bias = torch.matmul(q, pos_emb.transpose(-2, -1))
        return bias


# =============================================================================
# 9. NoPE — 无显式位置编码 (基线)
# =============================================================================
class NoPE(nn.Module):
    """无位置编码基线 —— 不加任何位置信息, 作为消融对照"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# =============================================================================
# 统一的 Transformer Block
# =============================================================================
class TransformerBlock(nn.Module):
    """
    轻量 Transformer block, 支持切换不同位置编码

    结构: LayerNorm -> Multi-Head Attention (带位置编码) -> 残差
          -> LayerNorm -> FFN -> 残差

    pe_type 决定位置编码方式:
    - "sinusoidal", "learned": 加到输入上 (additive)
    - "rope": 旋转 Q/K
    - "alibi", "kerple", "fire": 加到注意力分数上 (bias)
    - "relative": Q 与相对位置嵌入内积
    - "cope": 上下文位置偏置
    - "none": 不加位置编码
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        max_len: int = 1024,
        pe_type: str = "sinusoidal",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.pe_type = pe_type

        # Q/K/V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # 位置编码模块
        if pe_type == "sinusoidal":
            self.pe = SinusoidalPE(d_model, max_len)
        elif pe_type == "learned":
            self.pe = LearnedPE(d_model, max_len)
        elif pe_type == "rope":
            self.pe = RoPE(self.d_head, max_len)
        elif pe_type == "alibi":
            self.pe = ALiBi(n_heads, max_len)
        elif pe_type == "relative":
            self.pe = RelativePE(self.d_head)
        elif pe_type == "kerple":
            self.pe = Kerple(n_heads, max_len)
        elif pe_type == "fire":
            self.pe = FIRE(n_heads, max_len=max_len)
        elif pe_type == "cope":
            self.pe = CoPE(self.d_head, max_len)
        elif pe_type == "none":
            self.pe = NoPE()
        else:
            raise ValueError(f"不支持的位置编码类型: {pe_type}")

        # LayerNorm + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """多头注意力 + 位置编码"""
        B, S, D = x.shape

        # --- 加性位置编码: 先加到输入上 ---
        if self.pe_type in ("sinusoidal", "learned"):
            x = self.pe(x)

        # Q/K/V 投影 + reshape 为多头
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # --- RoPE: 旋转 Q/K ---
        if self.pe_type == "rope":
            q, k = self.pe(q, k)

        # 注意力分数: Q @ K^T / sqrt(d_head)
        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # --- 注意力偏置类位置编码 ---
        if self.pe_type in ("alibi", "kerple", "fire"):
            attn_scores = self.pe(attn_scores)
        elif self.pe_type == "relative":
            attn_scores = attn_scores + self.pe(q, S)
        elif self.pe_type == "cope":
            attn_scores = attn_scores + self.pe(q, k)

        # 因果掩码
        if causal:
            mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        out = torch.matmul(attn_weights, v)  # (B, H, S, D_head)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        # Pre-norm Transformer block
        x = x + self.dropout(self._attention(self.norm1(x), causal))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =============================================================================
# 语言模型包装器 (用于对比训练)
# =============================================================================
class PositionalEncodingLM(nn.Module):
    """
    轻量语言模型: Embedding -> N * TransformerBlock -> LM Head

    用于在字符级/token 级任务上对比不同位置编码的效果
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        max_len: int = 1024,
        pe_type: str = "sinusoidal",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, max_len, pe_type, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq_len) 整数 token id
        返回 logits: (batch, seq_len, vocab_size)
        """
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        for layer in self.layers:
            x = layer(x, causal=True)
        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Demo
# =============================================================================
def demo_positional_encodings():
    """演示各种位置编码"""
    print("=" * 70)
    print("位置编码 (Positional Encoding) 全面对比")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe_types = [
        "sinusoidal", "learned", "rope", "alibi",
        "relative", "kerple", "fire", "cope", "none",
    ]

    batch_size = 2
    seq_len = 64
    vocab_size = 256

    # 随机输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"\n设备: {device}")
    print(f"输入: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    print(f"\n{'类型':<15} {'参数量':>10} {'输出形状':<25} {'损失 (随机)':<12}")
    print("-" * 65)

    criterion = nn.CrossEntropyLoss()

    for pe_type in pe_types:
        model = PositionalEncodingLM(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2,
            ff_dim=256,
            max_len=512,
            pe_type=pe_type,
        ).to(device)

        logits = model(input_ids)
        # 计算语言模型损失 (预测下一个 token)
        loss = criterion(logits[:, :-1].reshape(-1, vocab_size), input_ids[:, 1:].reshape(-1))

        print(
            f"{pe_type:<15} {model.count_params():>10,} "
            f"{str(tuple(logits.shape)):<25} {loss.item():<12.4f}"
        )

    print("\n各位置编码模块说明:")
    descriptions = {
        "sinusoidal": "Transformer 原始固定正弦余弦编码",
        "learned": "可训练的绝对位置嵌入 (GPT-2/BERT)",
        "rope": "旋转位置编码, 天然相对位置性质 (LLaMA/Qwen)",
        "alibi": "注意力线性偏置, 零参数, 支持长度外推",
        "relative": "可学习相对位置嵌入 (Shaw et al.)",
        "kerple": "核化相对位置编码, 可学习衰减曲线",
        "fire": "MLP 映射连续相对位置, 平滑外推",
        "cope": "上下文位置编码, 位置由注意力动态决定",
        "none": "无位置编码基线 (消融对照)",
    }
    for name, desc in descriptions.items():
        print(f"  {name:<15} — {desc}")


if __name__ == "__main__":
    demo_positional_encodings()
