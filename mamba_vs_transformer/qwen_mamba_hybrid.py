"""
仿 Qwen3 风格的 Transformer + Mamba 混合模型

==========================================================================
背景: Qwen3 / Qwen3.5 的混合架构
==========================================================================

Qwen3 系列在部分规格中引入了 Mamba 层来替换一部分 Transformer 层:
  - 交替排列: Mamba 层和 Attention 层按一定比例交替
  - 共享机制: 部分层可能共享参数以提高效率
  - 位置编码: Attention 层使用 RoPE, Mamba 层天然有序无需位置编码

本实现参考 Qwen-Next 的设计理念, 从零构建一个混合模型, 展示:
  1. 如何在同一个模型中交替使用 Transformer 和 Mamba
  2. 两种层如何共享 embedding 和 LM head
  3. 不同的混合比例对性能的影响

==========================================================================
架构设计
==========================================================================

┌──────────────────────────────────────────────────────────────────┐
│                    Qwen-Mamba Hybrid Model                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Token IDs                                                 │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐                                                 │
│  │ Token Embed  │ (共享, 不加位置编码)                            │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │  Mamba Block │  ← 线性复杂度, 捕获长距离依赖                  │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ Attn + RoPE  │  ← 精确位置检索 (自带 RoPE 位置编码)          │
│  │ + FFN        │                                                │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │  Mamba Block │  ← 高效处理已经 attend 过的特征                │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ Attn + RoPE  │  ← 再次精确检索                                │
│  │ + FFN        │                                                │
│  └──────┬──────┘                                                 │
│         │                                                        │
│        ...  (重复 N 次)                                          │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │  LayerNorm   │                                                │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │   LM Head    │ (与 Token Embed 权重绑定)                     │
│  └─────────────┘                                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

层间的设计考量:
  - Mamba 层不需要位置编码 (因果卷积 + SSM 天然有序)
  - Attention 层使用 RoPE (旋转位置编码), 提供精确的位置感知
  - 两种层交替出现, 使模型既有高效的长距离建模能力, 又有精确的检索能力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class QwenMambaConfig:
    """Qwen-Mamba 混合模型配置"""
    d_model: int = 128          # 模型维度
    n_layers: int = 8           # 总层数
    vocab_size: int = 1000      # 词表大小
    max_seq_len: int = 512      # 最大序列长度
    dropout: float = 0.1

    # Attention 参数
    n_heads: int = 4            # 注意力头数
    n_kv_heads: int = 4         # KV 头数 (GQA: n_kv_heads < n_heads)
    rope_base: float = 10000.0  # RoPE 频率基数

    # Mamba 参数
    d_state: int = 16           # SSM 状态维度
    d_conv: int = 4             # 因果卷积核大小
    expand: int = 2             # 内部扩展倍数

    # 混合比例
    mamba_ratio: float = 0.5    # Mamba 层占比 (0.5 = 1:1 交替)
    # 层排列方式: "alternate" (交替), "mamba_heavy" (Mamba为主), "custom"
    layer_pattern: str = "alternate"

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        self.d_head = self.d_model // self.n_heads
        self.dt_rank = math.ceil(self.d_model / 16)
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads


# ==============================================================================
# RoPE 旋转位置编码
# ==============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding)
    Qwen 系列的标配位置编码, 只用在 Attention 层

    核心: 将 Q/K 的每两个维度视为复数, 乘以旋转矩阵
    使得 Q·K 内积只依赖相对位置 (m - n)
    """

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """q, k: (B, H, L, d_head) → 旋转后的 q, k"""
        seq_len = q.size(2)
        cos = self.cos_cache[:, :, :seq_len, :]
        sin = self.sin_cache[:, :, :seq_len, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ==============================================================================
# Transformer Attention Layer (带 RoPE + GQA)
# ==============================================================================

class QwenAttentionLayer(nn.Module):
    """
    Qwen 风格的 Attention 层:
      - Pre-RMSNorm
      - Multi-Head Attention + RoPE
      - 支持 GQA (Grouped Query Attention)
      - SwiGLU FFN (Qwen 标配)
      - 残差连接

    结构:
      x → RMSNorm → Attention(RoPE) → +x → RMSNorm → SwiGLU FFN → +x
    """

    def __init__(self, config: QwenMambaConfig):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.n_rep = config.n_heads // config.n_kv_heads  # GQA 重复次数

        # Attention
        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, config.n_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(d, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(d, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, d, bias=False)
        self.rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len, config.rope_base)
        self.attn_drop = nn.Dropout(config.dropout)

        # SwiGLU FFN: x → gate * up → down
        ff_dim = int(d * 8 / 3)  # Qwen 风格 FFN 维度
        self.norm2 = RMSNorm(d)
        self.gate_proj = nn.Linear(d, ff_dim, bias=False)
        self.up_proj = nn.Linear(d, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, d, bias=False)
        self.ffn_drop = nn.Dropout(config.dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """GQA: 将 KV 头重复 n_rep 次以匹配 Q 头数"""
        if self.n_rep == 1:
            return x
        B, H, L, D = x.shape
        return x.unsqueeze(2).expand(B, H, self.n_rep, L, D).reshape(
            B, H * self.n_rep, L, D
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        # ---- Attention ----
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)

        # RoPE 旋转位置编码
        q, k = self.rope(q, k)

        # GQA: 重复 KV 头
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 缩放点积注意力
        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        x = residual + self.o_proj(out)

        # ---- SwiGLU FFN ----
        residual = x
        h = self.norm2(x)
        # SwiGLU: gate * up, 其中 gate 过 SiLU 激活
        x = residual + self.ffn_drop(
            self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        )
        return x


# ==============================================================================
# Mamba Layer
# ==============================================================================

class QwenMambaLayer(nn.Module):
    """
    Qwen 风格的 Mamba 层:
      - Pre-RMSNorm
      - Mamba Block (Conv1D + Selective SSM + 门控)
      - 残差连接

    注意: 不使用位置编码! Mamba 通过因果卷积和 SSM 天然感知位置

    结构:
      x → RMSNorm → [分两路]
                      ├─ Linear → Conv1D → SiLU → SSM ──→ × ──→ Linear → +x
                      └─ Linear ────────→ SiLU ──────────┘
    """

    def __init__(self, config: QwenMambaConfig):
        super().__init__()
        d = config.d_model
        d_inner = config.d_inner
        d_state = config.d_state
        d_conv = config.d_conv
        dt_rank = config.dt_rank

        self.norm = RMSNorm(d)

        # 投影: d → 2×d_inner (SSM + 门控)
        self.in_proj = nn.Linear(d, d_inner * 2, bias=False)

        # 因果深度卷积
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # SSM 参数投影
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A 参数 (S4D-Lin 初始化)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A))

        # D (跳跃连接)
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.d_state = d_state
        self.dt_rank = dt_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # 分两路: SSM 分支 + 门控分支
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1D
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # 生成输入相关的 SSM 参数
        x_proj = self.x_proj(x_ssm)
        dt, B_mat, C_mat = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.log_A)

        # 选择性扫描
        d_inner = x_ssm.shape[-1]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)

        h = torch.zeros(B, d_inner, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * x_ssm[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x_ssm * self.D

        # 门控
        y = y * F.silu(z)

        return residual + self.dropout(self.out_proj(y))


# ==============================================================================
# RMSNorm (Qwen 使用 RMSNorm 而非 LayerNorm)
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    比 LayerNorm 快 (不需要计算均值), Qwen/LLaMA 标配
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ==============================================================================
# Qwen-Mamba 混合模型
# ==============================================================================

class QwenMambaHybridLM(nn.Module):
    """
    仿 Qwen3 风格的 Transformer + Mamba 混合语言模型

    层的排列方式 (layer_pattern):
      "alternate":   [Mamba, Attn, Mamba, Attn, ...]  (1:1 交替)
      "mamba_heavy": [Mamba, Mamba, Mamba, Attn, ...]  (Mamba 为主)
      "custom":       自定义 (用 _build_custom_layers)

    关键设计:
      1. Token Embedding 不加位置编码 (Mamba 层不需要)
      2. Attention 层内部自带 RoPE
      3. 使用 RMSNorm (更快) 而非 LayerNorm
      4. SwiGLU FFN (只在 Attention 层)
      5. 权重绑定: LM Head 与 Embedding 共享
    """

    def __init__(self, config: QwenMambaConfig):
        super().__init__()
        self.config = config

        # Token Embedding (无位置编码!)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # 构建混合层
        self.layers = nn.ModuleList()
        self.layer_types = []
        self._build_layers(config)

        # 输出
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重绑定

        self._init_weights()

    def _build_layers(self, config: QwenMambaConfig):
        """根据 layer_pattern 构建层列表"""
        if config.layer_pattern == "alternate":
            # 1:1 交替: [Mamba, Attn, Mamba, Attn, ...]
            for i in range(config.n_layers):
                if i % 2 == 0:
                    self.layers.append(QwenMambaLayer(config))
                    self.layer_types.append("mamba")
                else:
                    self.layers.append(QwenAttentionLayer(config))
                    self.layer_types.append("attn")

        elif config.layer_pattern == "mamba_heavy":
            # Mamba 为主 (类似 Jamba): 每 4 层 1 个 Attention
            attn_interval = max(2, int(1 / (1 - config.mamba_ratio)))
            for i in range(config.n_layers):
                if (i + 1) % attn_interval == 0:
                    self.layers.append(QwenAttentionLayer(config))
                    self.layer_types.append("attn")
                else:
                    self.layers.append(QwenMambaLayer(config))
                    self.layer_types.append("mamba")

        elif config.layer_pattern == "custom":
            # 自定义: 底部 Mamba, 中间交替, 顶部 Attention
            # 这种设计的理由:
            #   底部: Mamba 高效处理原始 token 嵌入, 建立初步上下文
            #   中间: 交替使用, 逐步精炼表示
            #   顶部: Attention 层做最终的精确检索
            n = config.n_layers
            for i in range(n):
                if i < n // 4:
                    # 底部: 纯 Mamba
                    self.layers.append(QwenMambaLayer(config))
                    self.layer_types.append("mamba")
                elif i < 3 * n // 4:
                    # 中间: 交替
                    if i % 2 == 0:
                        self.layers.append(QwenMambaLayer(config))
                        self.layer_types.append("mamba")
                    else:
                        self.layers.append(QwenAttentionLayer(config))
                        self.layer_types.append("attn")
                else:
                    # 顶部: 纯 Attention
                    self.layers.append(QwenAttentionLayer(config))
                    self.layer_types.append("attn")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None):
        """
        input_ids: (batch, seq_len)
        targets: (batch, seq_len), 可选
        """
        # 只有 Token Embedding, 不加位置编码
        # (Attention 层内部自带 RoPE, Mamba 层不需要位置编码)
        x = self.tok_emb(input_ids)

        # 逐层处理
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1), ignore_index=-1,
            )
        return logits, loss

    def get_layer_info(self) -> str:
        """返回层结构的可视化字符串"""
        return " → ".join(
            f"[{'A' if t == 'attn' else 'M'}{i}]"
            for i, t in enumerate(self.layer_types)
        )

    def count_params(self) -> dict:
        """分别统计 Mamba 层和 Attention 层的参数量"""
        mamba_params = 0
        attn_params = 0
        for layer, ltype in zip(self.layers, self.layer_types):
            p = sum(p.numel() for p in layer.parameters())
            if ltype == "mamba":
                mamba_params += p
            else:
                attn_params += p
        shared = sum(p.numel() for p in self.tok_emb.parameters()) + \
                 sum(p.numel() for p in self.norm.parameters())
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "mamba": mamba_params,
            "attn": attn_params,
            "shared": shared,
        }


# ==============================================================================
# 对比实验
# ==============================================================================

def demo_qwen_mamba():
    """演示 Qwen-Mamba 混合模型"""
    print("=" * 70)
    print("  仿 Qwen3 风格: Transformer + Mamba 混合模型")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    # ---- 1. 三种混合策略对比 ----
    print("1. 三种层排列策略对比")
    print("-" * 60)

    patterns = ["alternate", "mamba_heavy", "custom"]
    pattern_names = {
        "alternate": "交替排列 (Qwen 风格)",
        "mamba_heavy": "Mamba 为主 (Jamba 风格)",
        "custom": "自定义 (底Mamba+中交替+顶Attn)",
    }

    models = {}
    for pat in patterns:
        config = QwenMambaConfig(
            d_model=64, n_layers=8, vocab_size=256,
            max_seq_len=128, n_heads=4, n_kv_heads=2,
            d_state=16, d_conv=4, expand=2,
            layer_pattern=pat, mamba_ratio=0.75,
        )
        model = QwenMambaHybridLM(config).to(device)
        models[pat] = model
        params = model.count_params()

        n_mamba = sum(1 for t in model.layer_types if t == "mamba")
        n_attn = sum(1 for t in model.layer_types if t == "attn")

        print(f"\n  {pattern_names[pat]}")
        print(f"    层结构: {model.get_layer_info()}")
        print(f"    Mamba层: {n_mamba}, Attention层: {n_attn}")
        print(f"    参数量: 总={params['total']:,}  "
              f"(Mamba={params['mamba']:,}, Attn={params['attn']:,})")

    # ---- 2. 前向传播验证 ----
    print(f"\n\n2. 前向传播验证")
    print("-" * 60)

    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

    for pat in patterns:
        model = models[pat]
        logits, loss = model(input_ids, targets)
        print(f"  {pat:<15} output={tuple(logits.shape)}  loss={loss.item():.4f}")

    # ---- 3. 训练对比 ----
    print(f"\n\n3. 训练收敛对比 (200 步)")
    print("-" * 60)

    vocab_size = 256
    criterion = nn.CrossEntropyLoss()

    def make_data():
        """位置依赖模式: token[t] = (token[t-1] + t) % vocab_size"""
        data = torch.zeros(batch_size, seq_len + 1, dtype=torch.long, device=device)
        data[:, 0] = torch.randint(0, vocab_size, (batch_size,), device=device)
        for t in range(1, seq_len + 1):
            data[:, t] = (data[:, t - 1] + t) % vocab_size
        return data[:, :-1], data[:, 1:]

    for pat in patterns:
        model = models[pat]
        # 重新初始化
        model._init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        losses = []
        for step in range(200):
            inp, tgt = make_data()
            _, loss = model(inp, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        print(f"  {pat:<15} 初始={losses[0]:.3f}  "
              f"第100步={losses[99]:.3f}  最终={losses[-1]:.3f}")

    # ---- 4. 关键设计解读 ----
    print(f"""

4. 关键设计解读
{'─' * 60}

  为什么 Qwen/Jamba 要混合 Transformer 和 Mamba?

  ┌────────────────────────────────────────────────────────────┐
  │ Mamba 的优势:                                              │
  │   • O(N) 复杂度, 长序列高效                                │
  │   • O(1) 推理内存, 不需要 KV cache                         │
  │   • 天然因果, 不需要位置编码                                │
  │                                                            │
  │ Mamba 的劣势:                                              │
  │   • 精确回忆弱 (信息被压缩到固定大小状态)                  │
  │   • 无法精确 "看到" 远处的特定 token                       │
  │                                                            │
  │ 解决方案: 加入少量 Attention 层                             │
  │   • Attention 层提供精确的 token-to-token 检索              │
  │   • RoPE 编码只在 Attention 层使用                          │
  │   • Mamba 层负责高效的上下文压缩                            │
  │   → 两者互补, 既高效又精确                                 │
  └────────────────────────────────────────────────────────────┘

  Qwen 风格 vs Jamba 风格:

    Qwen (交替):  [M][A][M][A][M][A]...
      → 每一层都能结合两种能力
      → Attention 层较多, 精确度高, 但 KV cache 也大

    Jamba (稀疏):  [M][M][M][A][M][M][M][A]...
      → 大部分计算在 Mamba (高效)
      → 只有 1-2 层 Attention, KV cache 极小
      → 适合超长上下文 (256K+)

  实际部署考量:
    • 短上下文 (<4K): 纯 Transformer 足够
    • 中等上下文 (4K~32K): Qwen 交替风格
    • 超长上下文 (32K+): Jamba 稀疏风格
""")
    print("[演示完成]")


if __name__ == "__main__":
    demo_qwen_mamba()
