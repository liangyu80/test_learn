"""
Linear Attention / Linear Transformer 方法汇总与 Mamba 对比

核心问题: Transformer 的 softmax(QK^T)V 复杂度是 O(N²·d), 能否降到 O(N)?

==========================================================================
方法汇总
==========================================================================

1. Linear Attention (Katharopoulos et al., 2020)
   - 去掉 softmax, 用核函数近似: Attn = φ(Q)·(φ(K)^T·V)
   - 利用结合律先算 KV (O(d²·N)) 而非 QK^T (O(N²·d))
   - 可以写成 RNN 形式: S_t = S_{t-1} + φ(k_t)·v_t^T

2. RetNet (Sun et al., 2023 — Retentive Network)
   - 注意力 = 线性注意力 + 指数衰减 (类似 SSM 的 A 矩阵)
   - 三种计算模式: 并行 (训练) / 循环 (推理) / 分块 (长序列)

3. RWKV (Peng et al., 2023)
   - WKV 机制: 用时间衰减替代 softmax 归一化
   - 核心: 维护 (加权分子, 加权分母) 两个状态
   - 完全可以写成 RNN 递推形式

4. GLA (Gated Linear Attention, Yang et al., 2024)
   - 在 Linear Attention 基础上加数据相关的门控衰减
   - 门控: G_t = σ(W_g · x_t) —— 每步可学习的遗忘率
   - 结合了 Linear Attention 和 Mamba 的优点

5. Lightning Attention (Qin et al., 2024)
   - 分块 Linear Attention, 块内用常规注意力, 块间用累积 KV 状态
   - 解决 Linear Attention 在 IO 上的瓶颈

==========================================================================
与 Mamba 对比
==========================================================================

┌───────────────────┬──────────────────────┬──────────────────────┐
│                   │   Linear Attention    │       Mamba          │
├───────────────────┼──────────────────────┼──────────────────────┤
│ 状态形式           │ S = Σ φ(k)·v^T       │ h = Ā·h + B̄·x       │
│                   │ 外积累加 (d×d 矩阵)   │ 对角矩阵乘 (d×n 向量)│
│ 状态大小           │ O(d²)                │ O(d·n), n<<d         │
│ 选择性             │ 无 (固定核映射)       │ 有 (B,C,Δ 输入相关)  │
│ 衰减机制           │ 无衰减/固定衰减       │ exp(Δ·A) 自适应衰减   │
│ 训练并行性         │ 完全并行              │ 并行扫描              │
│ 代表模型           │ RetNet, RWKV, GLA     │ Mamba, Mamba-2        │
│ 核心差异           │ 注意力的线性近似       │ 状态空间模型的选择性化 │
└───────────────────┴──────────────────────┴──────────────────────┘

殊途同归:
  Linear Attention 和 Mamba 在循环推理形式下等价于更新一个 "状态矩阵"!
  - Linear Attn: S_t = S_{t-1} + k_t ⊗ v_t     (无衰减外积累加)
  - RetNet:      S_t = γ·S_{t-1} + k_t ⊗ v_t   (固定衰减)
  - GLA:         S_t = G_t·S_{t-1} + k_t ⊗ v_t  (数据相关门控衰减)
  - Mamba:       h_t = Ā_t·h_{t-1} + B̄_t·x_t    (数据相关结构化衰减)

  Mamba-2 论文 (Dao & Gu, 2024) 证明了 SSM 和 Linear Attention 的对偶性!
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Linear Attention (Katharopoulos et al., 2020)
# ==============================================================================

class LinearAttention(nn.Module):
    """
    线性注意力: 用 elu+1 核函数替代 softmax

    标准 Attention: y = softmax(QK^T / √d) · V           O(N²·d)
    Linear Attention: y = φ(Q) · (φ(K)^T · V) / (φ(Q) · Σφ(k))  O(N·d²)

    关键技巧 — 结合律:
      原始: (φ(Q) · φ(K)^T) · V  → 先算 N×N 矩阵 → O(N²)
      改写: φ(Q) · (φ(K)^T · V)  → 先算 d×d 矩阵 → O(N·d²)

    循环形式 (推理):
      S_t = S_{t-1} + φ(k_t)·v_t^T          # 外积累加到 d×d 状态
      z_t = z_{t-1} + φ(k_t)                 # 归一化累积
      y_t = (S_t · φ(q_t)) / (z_t · φ(q_t)) # 从状态中读取

    局限: 没有衰减机制, 旧信息永远不会被遗忘 → 状态可能 "饱和"
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
        """核映射 φ(x) = elu(x) + 1 (保证非负)"""
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        因果 Linear Attention (循环形式)
        x: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 各 (B, H, L, d_head)

        # 核映射
        q = self._elu_feature_map(q)
        k = self._elu_feature_map(k)

        # 因果循环形式: 确保只看到过去
        # S_t = S_{t-1} + k_t ⊗ v_t  (d_head × d_head 的状态矩阵)
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        z = torch.zeros(B, self.n_heads, self.d_head, device=x.device)
        outputs = []

        for t in range(L):
            k_t = k[:, :, t]  # (B, H, d)
            v_t = v[:, :, t]  # (B, H, d)
            q_t = q[:, :, t]  # (B, H, d)

            # 更新状态: S += k_t ⊗ v_t
            S = S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            z = z + k_t

            # 输出: y_t = S · q_t / (z · q_t)
            num = torch.einsum("bhde,bhd->bhe", S, q_t)    # (B, H, d)
            den = (z * q_t).sum(-1, keepdim=True).clamp(min=1e-6)  # (B, H, 1)
            outputs.append(num / den)

        out = torch.stack(outputs, dim=2)  # (B, H, L, d)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ==============================================================================
# 2. RetNet (Sun et al., 2023 — Retentive Network)
# ==============================================================================

class RetNetAttention(nn.Module):
    """
    RetNet 的 Retention 机制: 线性注意力 + 指数衰减

    与 Linear Attention 的关键区别: 加了衰减因子 γ
      S_t = γ · S_{t-1} + k_t ⊗ v_t     (γ < 1, 旧信息指数衰减)

    每个注意力头使用不同的衰减率 γ:
      γ_head = 1 - 2^{-5-head_id}

    三种计算模式 (本实现用循环模式):
      - 并行: 构造 γ^{|i-j|} 衰减矩阵, 与标准 Attention 类似  (训练)
      - 循环: S_t = γ·S + kv^T, 类似 RNN  (推理)
      - 分块: 块内并行, 块间循环  (长序列训练)

    与 Mamba 对比:
      - RetNet: γ 固定 (超参数), 所有时间步衰减率一样
      - Mamba:  exp(Δ·A) 随输入变化, 可以选择性地快衰减或慢衰减
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(n_heads, d_model)

        # 每个头不同的衰减率
        gammas = 1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), n_heads))
        self.register_buffer("gammas", gammas)  # (n_heads,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # 循环模式 (推理友好)
        gammas = self.gammas.view(1, self.n_heads, 1, 1)  # (1, H, 1, 1)
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        outputs = []

        for t in range(L):
            k_t = k[:, :, t]  # (B, H, d)
            v_t = v[:, :, t]
            q_t = q[:, :, t]

            # 核心: 带衰减的状态更新
            S = gammas * S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            y_t = torch.einsum("bhde,bhd->bhe", S, q_t)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)  # (B, H, L, d)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.out(out)


# ==============================================================================
# 3. RWKV (Peng et al., 2023)
# ==============================================================================

class RWKVAttention(nn.Module):
    """
    RWKV 的 WKV (Weighted Key-Value) 机制

    核心思想: 用时间衰减权重替代 softmax 归一化

    WKV_t = Σ_{i=1}^{t} e^{-(t-i)·w + k_i} · v_i  /  Σ e^{-(t-i)·w + k_i}

    其中 w > 0 是可学习的时间衰减参数

    递推形式:
      a_t = e^{-w} · a_{t-1} + e^{k_t} · v_t     (加权分子)
      b_t = e^{-w} · b_{t-1} + e^{k_t}            (加权分母)
      wkv_t = a_t / b_t

    (实际用 log-space 技巧避免数值溢出)

    与 Mamba 对比:
      - RWKV 的衰减 w 是可学习但与输入无关 (固定衰减率)
      - Mamba 的 Δ·A 随输入变化 (选择性衰减率)
      - RWKV 有 token shift (时间混合), 类似 Mamba 的 Conv1D
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 可学习的时间衰减 (正值, 通过 exp 保证)
        self.time_decay = nn.Parameter(torch.randn(d_model) * 0.1 - 5)  # 初始化为较大衰减
        # 当前 token 的额外加成 (bonus)
        self.time_first = nn.Parameter(torch.randn(d_model) * 0.01)

        # 投影层
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Token shift (时间混合: 当前 token 和前一个 token 的线性混合)
        self.time_mix_k = nn.Parameter(torch.ones(d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Token shift: x_shifted = lerp(x_{t-1}, x_t, mix)
        x_prev = F.pad(x, (0, 0, 1, 0))[:, :L]  # 前移一步 (0 填充)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)           # (B, L, D)
        v = self.value(xv)         # (B, L, D)
        r = torch.sigmoid(self.receptance(xr))  # 门控 (类似 Mamba 的门控分支)

        # WKV 计算 (简化版, 实际 RWKV 用 log-space)
        w = -torch.exp(self.time_decay)  # 负衰减率
        u = self.time_first               # 当前 token 加成

        # 循环计算
        outputs = []
        a = torch.zeros(B, D, device=x.device)  # 加权分子
        b = torch.zeros(B, D, device=x.device)  # 加权分母
        max_prev = torch.full((B, D), -1e38, device=x.device)  # 数值稳定

        for t in range(L):
            k_t = k[:, t]  # (B, D)
            v_t = v[:, t]

            # 数值稳定的 log-space 计算
            # 当前 token 贡献: e^{u + k_t} · v_t
            wkv_num = a + torch.exp(u + k_t) * v_t
            wkv_den = b + torch.exp(u + k_t)
            wkv = wkv_num / wkv_den.clamp(min=1e-6)

            outputs.append(wkv)

            # 状态更新: 乘以衰减因子
            a = torch.exp(w) * a + torch.exp(k_t) * v_t
            b = torch.exp(w) * b + torch.exp(k_t)

        wkv = torch.stack(outputs, dim=1)  # (B, L, D)
        return self.output(r * wkv)


# ==============================================================================
# 4. GLA (Gated Linear Attention, Yang et al., 2024)
# ==============================================================================

class GatedLinearAttention(nn.Module):
    """
    门控线性注意力 (GLA): 在 Linear Attention 上加数据相关的门控衰减

    S_t = G_t ⊙ S_{t-1} + k_t ⊗ v_t
    y_t = S_t · q_t

    G_t = σ(W_g · x_t)  —— 数据相关的门控 (0~1)

    对比:
      - Linear Attention: S_t = S_{t-1} + kv^T           (无衰减, 会饱和)
      - RetNet:           S_t = γ · S_{t-1} + kv^T       (固定衰减)
      - GLA:              S_t = G_t · S_{t-1} + kv^T     (数据相关衰减)
      - Mamba:            h_t = Ā_t · h_{t-1} + B̄_t · x_t (数据相关结构化衰减)

    GLA 和 Mamba 在 "数据相关衰减" 这一点上最为接近!
    主要区别:
      - GLA 的状态是 d×d 矩阵 (外积), Mamba 是 d×n 向量
      - GLA 沿用 QKV 框架, Mamba 用 SSM 框架
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # 数据相关的门控
        self.gate_proj = nn.Linear(d_model, n_heads)  # 每个头一个门控值

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # 数据相关的衰减门控
        gates = torch.sigmoid(self.gate_proj(x))  # (B, L, H)
        gates = gates.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, H, L, 1, 1)

        # 循环计算
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        outputs = []

        for t in range(L):
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            q_t = q[:, :, t]
            g_t = gates[:, :, t]  # (B, H, 1, 1) — 门控衰减

            # 核心: 数据相关的衰减!
            S = g_t * S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            y_t = torch.einsum("bhde,bhd->bhe", S, q_t)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ==============================================================================
# 5. Mamba SSM (简化版, 用于对比)
# ==============================================================================

class MambaSSMForComparison(nn.Module):
    """简化版 Mamba SSM, 用于与 Linear Attention 方法直接对比"""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_model, -1)
        self.log_A = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        B_mat = self.B_proj(x)
        C_mat = self.C_proj(x)
        delta = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.log_A)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)

        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1) + x * self.D


# ==============================================================================
# 统一语言模型包装
# ==============================================================================

class LinearMethodLM(nn.Module):
    """统一的 LM 包装, 用于对比各种 Linear Attention 方法"""

    def __init__(self, method: str, vocab_size: int = 256,
                 d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.method = method

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            norm = nn.LayerNorm(d_model)
            if method == "linear_attn":
                attn = LinearAttention(d_model)
            elif method == "retnet":
                attn = RetNetAttention(d_model)
            elif method == "rwkv":
                attn = RWKVAttention(d_model)
            elif method == "gla":
                attn = GatedLinearAttention(d_model)
            elif method == "mamba":
                attn = MambaSSMForComparison(d_model)
            else:
                raise ValueError(f"未知方法: {method}")
            self.layers.append(nn.ModuleDict({"norm": norm, "attn": attn}))

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer["attn"](layer["norm"](x))
        return self.lm_head(self.norm(x))


# ==============================================================================
# 对比实验
# ==============================================================================

def compare_linear_methods():
    """对比各种 Linear Attention 方法与 Mamba"""
    print("=" * 70)
    print("  Linear Attention 方法与 Mamba 对比")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    vocab_size = 64
    d_model = 64
    n_layers = 2
    seq_len = 64
    batch_size = 32
    n_epochs = 40

    methods = ["linear_attn", "retnet", "rwkv", "gla", "mamba"]
    method_names = {
        "linear_attn": "Linear Attn",
        "retnet": "RetNet",
        "rwkv": "RWKV",
        "gla": "GLA",
        "mamba": "Mamba",
    }

    # ---- 1. 参数量 ----
    print("1. 参数量对比")
    print("-" * 50)
    models = {}
    for m in methods:
        model = LinearMethodLM(m, vocab_size, d_model, n_layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        models[m] = model
        print(f"  {method_names[m]:<15} {n_params:>10,} 参数")

    # ---- 2. 训练收敛 ----
    print(f"\n2. 训练收敛 (位置依赖模式, {n_epochs} epochs)")
    print("-" * 50)

    criterion = nn.CrossEntropyLoss()

    def make_data():
        data = torch.zeros(batch_size, seq_len + 1, dtype=torch.long, device=device)
        data[:, 0] = torch.randint(0, vocab_size, (batch_size,), device=device)
        for t in range(1, seq_len + 1):
            data[:, t] = (data[:, t - 1] + t) % vocab_size
        return data[:, :-1], data[:, 1:]

    for m in methods:
        model = models[m]
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        for epoch in range(n_epochs):
            inp, tgt = make_data()
            logits = model(inp)
            loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        print(f"  {method_names[m]:<15} 初始={losses[0]:.3f}  最终={losses[-1]:.3f}")

    # ---- 3. 推理速度 ----
    print(f"\n3. 推理速度 (seq_len={seq_len})")
    print("-" * 50)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for m in methods:
        model = models[m].eval()
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)

        n_runs = 20
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model(input_ids)
        elapsed = (time.perf_counter() - start) / n_runs * 1000
        tokens_sec = batch_size * seq_len / (elapsed / 1000)
        print(f"  {method_names[m]:<15} {elapsed:>8.2f} ms  {tokens_sec:>10,.0f} tokens/s")

    # ---- 4. 总结 ----
    print(f"""
4. 方法特点总结
{'─' * 60}

  Linear Attention:
    • 去掉 softmax, 用核函数近似
    • 状态: d×d 外积矩阵 (无衰减, 会饱和)
    • 局限: 没有遗忘机制, 不适合很长的序列

  RetNet:
    • Linear Attention + 固定指数衰减
    • 三种计算模式: 并行/循环/分块
    • 局限: 衰减率固定, 无法根据内容调整

  RWKV:
    • Token shift + 时间衰减的 WKV 机制
    • 100% RNN: 可增量推理, 无需 KV cache
    • 局限: 衰减率与输入无关

  GLA:
    • Linear Attention + 数据相关的门控衰减
    • G_t = σ(W·x_t) 每步动态决定遗忘率
    • 最接近 Mamba 的 Linear Attention 变体

  Mamba:
    • SSM 框架: h = Ā·h + B̄·x (非 QKV 框架)
    • B, C, Δ 全部输入相关 (最强选择性)
    • 状态 d×n (n<<d, 更紧凑)
    • A 矩阵提供结构化的多尺度衰减

  共同点: 都是 O(N) 训练, O(1) 推理的序列模型
  趋势: GLA 和 Mamba 的融合 (如 Mamba-2 的 SSD 对偶性)
""")


if __name__ == "__main__":
    compare_linear_methods()
