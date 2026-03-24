"""
RNN vs Transformer vs Mamba: 三代序列模型全面对比

本文件通过代码 + 详细注释，讲解三种架构的核心区别，
以及 Mamba 如何平衡 RNN 和 Transformer 的计算量。

==========================================================================
一、三代序列模型概览
==========================================================================

┌────────────────────────────────────────────────────────────────────────┐
│ 时间线:                                                                │
│   RNN (1986) → LSTM (1997) → Transformer (2017) → Mamba (2023)        │
│                                                                        │
│ 核心矛盾:                                                              │
│   训练效率 (并行化) vs 推理效率 (增量计算) vs 长距离建模能力            │
└────────────────────────────────────────────────────────────────────────┘

┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│              │      RNN/LSTM    │   Transformer    │     Mamba        │
├──────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 训练方式      │ 串行 (BPTT)      │ 完全并行          │ 并行扫描         │
│ 训练复杂度    │ O(N·d²)          │ O(N²·d)          │ O(N·d·n)         │
│ 推理方式      │ 增量 (隐状态)     │ 全量 (KV cache)   │ 增量 (SSM状态)   │
│ 推理每步      │ O(d²)            │ O(N·d)           │ O(d·n)           │
│ 推理内存      │ O(d) 隐状态      │ O(N·d) KV cache  │ O(d·n) 状态      │
│ 长距离建模    │ 弱 (梯度消失)     │ 强 (直接attend)   │ 中 (选择性压缩)  │
│ 位置感知      │ 天然有序          │ 需位置编码        │ 天然有序          │
│ 精确回忆      │ 弱               │ 强               │ 弱-中             │
│ 并行性        │ 差               │ 优               │ 良 (并行扫描)     │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘

其中:
  N = 序列长度, d = 模型维度, n = SSM 状态维度 (通常 n << d)

==========================================================================
二、各架构详解
==========================================================================

1. RNN: 通过隐状态传递信息 (时间维度的压缩)
   h_t = f(W_h·h_{t-1} + W_x·x_t)        # 固定压缩规则
   问题: 信息瓶颈 —— 所有历史必须压缩进固定大小的 h

2. Transformer: 通过注意力直接访问所有历史 (不压缩)
   Attn(Q,K,V) = softmax(QK^T/√d)·V       # 全量访问
   问题: 计算量随序列长度二次增长, KV cache 线性增长

3. Mamba: 输入相关的选择性压缩 (智能压缩)
   h_t = Ā(x_t)·h_{t-1} + B̄(x_t)·x_t     # 选择性压缩
   关键: Ā 和 B̄ 随输入变化, 模型可以选择记忆什么、遗忘什么

==========================================================================
三、Mamba 如何平衡 RNN 和 Transformer
==========================================================================

Mamba 的核心洞察: 取 RNN 的 O(1) 推理 + Transformer 的上下文感知能力

1. 继承 RNN 的优点:
   - 推理时每步 O(1) 内存 (只维护固定大小状态, 不需要 KV cache)
   - 天然因果性 (不需要因果掩码)
   - 天然位置感知 (不需要位置编码)

2. 继承 Transformer 的优点:
   - 训练时可并行 (通过并行扫描算法, 类似前缀和)
   - 输入相关的计算 (B, C, Δ 都是输入的函数, 类似 Q, K, V)

3. 解决 RNN 的缺点:
   - RNN 用固定函数压缩 → Mamba 用输入相关函数选择性压缩
   - RNN 的遗忘门只有标量 → Mamba 的 Δ 是向量 + A 矩阵结构化

4. 解决 Transformer 的缺点:
   - Transformer O(N²) 计算 → Mamba O(N) 线性
   - Transformer O(N) KV cache → Mamba O(1) 固定状态

==========================================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 简易 RNN / LSTM
# ==============================================================================

class SimpleRNN(nn.Module):
    """
    简单 RNN: h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)

    问题:
      - 梯度消失: 长序列反向传播时梯度指数衰减
      - 固定压缩: 无论输入重要与否, 都用同一个 W_h 压缩
      - 无法并行: 训练必须按时间步串行计算
    """

    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden
        self.W_h = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_x = nn.Linear(d_input, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_input)
        返回: (batch, seq_len, d_hidden)
        """
        B, L, _ = x.shape
        h = torch.zeros(B, self.d_hidden, device=x.device)
        outputs = []
        for t in range(L):
            # h_t = tanh(W_h·h_{t-1} + W_x·x_t)
            h = torch.tanh(self.W_h(h) + self.W_x(x[:, t]))
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class SimpleLSTM(nn.Module):
    """
    LSTM: 通过门控机制缓解梯度消失

    f_t = σ(W_f·[h_{t-1}, x_t])        遗忘门: 决定丢弃哪些旧信息
    i_t = σ(W_i·[h_{t-1}, x_t])        输入门: 决定写入哪些新信息
    c̃_t = tanh(W_c·[h_{t-1}, x_t])     候选记忆
    c_t = f_t * c_{t-1} + i_t * c̃_t    记忆更新 (遗忘 + 写入)
    o_t = σ(W_o·[h_{t-1}, x_t])        输出门: 决定输出什么
    h_t = o_t * tanh(c_t)

    与 Mamba 的类比:
      - LSTM 的遗忘门 f_t ↔ Mamba 的 exp(Δ·A) —— 控制 "保留多少旧状态"
      - LSTM 的输入门 i_t ↔ Mamba 的 Δ·B       —— 控制 "写入多少新信息"
      - LSTM 的输出门 o_t ↔ Mamba 的 C           —— 控制 "读取什么"

    关键区别:
      - LSTM: 门控是标量 (对整个隐状态统一缩放)
      - Mamba: A 是结构化矩阵 (对不同状态维度有不同的衰减率)
      - LSTM: 隐状态维度 = 输出维度 (通常几百到几千)
      - Mamba: 状态维度 n 可以很大 (因为不直接输出, 只通过 C 读取)
    """

    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden
        # 4 个门合并计算: [f, i, o, c̃]
        self.gates = nn.Linear(d_input + d_hidden, 4 * d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = torch.zeros(B, self.d_hidden, device=x.device)
        c = torch.zeros(B, self.d_hidden, device=x.device)
        outputs = []

        for t in range(L):
            combined = torch.cat([h, x[:, t]], dim=-1)
            gates = self.gates(combined)
            f, i, o, c_tilde = gates.chunk(4, dim=-1)

            f = torch.sigmoid(f)              # 遗忘门
            i = torch.sigmoid(i)              # 输入门
            o = torch.sigmoid(o)              # 输出门
            c_tilde = torch.tanh(c_tilde)     # 候选记忆

            c = f * c + i * c_tilde           # 记忆更新
            h = o * torch.tanh(c)             # 输出
            outputs.append(h)

        return torch.stack(outputs, dim=1)


# ==============================================================================
# 2. 简易 Transformer Attention
# ==============================================================================

class SimpleAttention(nn.Module):
    """
    标准 Self-Attention: Attn(Q,K,V) = softmax(QK^T / √d_k) · V

    计算量分析:
      - QK^T: O(N² · d)      ← 二次方瓶颈!
      - softmax: O(N²)
      - Attn·V: O(N² · d)
      - 总计: O(N² · d)

    推理时的 KV Cache:
      - 需要缓存所有历史的 K, V: 内存 O(N · d)
      - 每生成一个新 token: 需要与所有缓存的 K 计算注意力 O(N · d)
      - 序列越长, 每步推理越慢, 内存越大
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 各 (B, H, L, d_head)

        # QK^T / √d —— O(N²·d) 这里是瓶颈
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 因果掩码
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(mask, -1e9)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ==============================================================================
# 3. 简易 Mamba SSM
# ==============================================================================

class SimpleMambaSSM(nn.Module):
    """
    简化版 Mamba 选择性 SSM —— 展示核心思想

    与 RNN 对比:
      RNN:   h_t = tanh(W·h_{t-1} + U·x_t)          # 固定参数
      Mamba: h_t = exp(Δ(x_t)·A)·h_{t-1} + Δ(x_t)·B(x_t)·x_t  # 输入相关

    与 Attention 对比:
      Attention: y = softmax(QK^T)·V         # 全量看所有历史, O(N²)
      Mamba:     y = C(x_t)·h_t              # 只看压缩后的状态, O(1)

    Mamba 的平衡策略:
    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │   RNN 的 O(1) 推理          Transformer 的上下文感知          │
    │        │                           │                           │
    │        └────────┐    ┌─────────────┘                           │
    │                 ▼    ▼                                          │
    │            Mamba: 选择性 SSM                                   │
    │                                                                │
    │   - 固定大小状态 h (像 RNN) → O(1) 推理                       │
    │   - 但 A, B, C 随输入变化 (像 Attention 的 Q, K, V)           │
    │   - 并行扫描算法 → 训练和 Transformer 一样快                  │
    │   - Δ 控制 "步长":                                             │
    │     · 重要 token → Δ 大 → 状态大幅更新 (类似 Attention)       │
    │     · 无关 token → Δ 小 → 状态几乎不变 (跳过, RNN做不到)     │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 输入相关的参数映射 (这是与 RNN 的关键区别)
        self.B_proj = nn.Linear(d_model, d_state)    # 输入→B: 写入什么
        self.C_proj = nn.Linear(d_model, d_state)    # 输入→C: 读取什么
        self.dt_proj = nn.Linear(d_model, d_model)   # 输入→Δ: 更新多少

        # A: 固定的衰减矩阵 (对数空间参数化)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_model, -1)
        self.log_A = nn.Parameter(torch.log(A))

        # D: 跳跃连接
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        B_batch, L, D = x.shape

        # 输入相关的参数 (核心创新!)
        B = self.B_proj(x)                        # (B, L, n) — 写入什么
        C = self.C_proj(x)                        # (B, L, n) — 读取什么
        delta = F.softplus(self.dt_proj(x))       # (B, L, D) — 更新多少

        A = -torch.exp(self.log_A)                # (D, n) — 衰减率

        # 离散化: Ā = exp(Δ·A)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)   # (B, L, D, n)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, n)

        # 循环扫描 (训练时可用并行扫描加速)
        h = torch.zeros(B_batch, D, self.d_state, device=x.device)
        outputs = []

        for t in range(L):
            # 状态更新: h_t = Ā_t · h_{t-1} + B̄_t · x_t
            # 对比 RNN: h_t = tanh(W_h·h_{t-1} + W_x·x_t)
            # 区别: Ā_t, B̄_t 随输入变化 (RNN 的 W_h, W_x 固定)
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)

            # 输出: y_t = C_t · h_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y + x * self.D  # 加跳跃连接


# ==============================================================================
# 4. 语言模型包装器
# ==============================================================================

class SequenceModelLM(nn.Module):
    """统一的语言模型包装, 用于对比 RNN / LSTM / Attention / Mamba"""

    def __init__(self, model_type: str, vocab_size: int = 256,
                 d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.model_type = model_type

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            if model_type == "rnn":
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(d_model),
                    SimpleRNN(d_model, d_model),
                ))
            elif model_type == "lstm":
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(d_model),
                    SimpleLSTM(d_model, d_model),
                ))
            elif model_type == "transformer":
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(d_model),
                    SimpleAttention(d_model),
                ))
            elif model_type == "mamba":
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(d_model),
                    SimpleMambaSSM(d_model),
                ))
            else:
                raise ValueError(f"未知模型类型: {model_type}")

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)  # 残差连接
        return self.lm_head(self.norm(x))


# ==============================================================================
# 对比实验
# ==============================================================================

def compare_architectures():
    """对比 RNN / LSTM / Transformer / Mamba 四种架构"""
    print("=" * 70)
    print("  RNN vs Transformer vs Mamba: 三代序列模型对比")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    vocab_size = 64
    d_model = 64
    n_layers = 2
    seq_len = 64
    batch_size = 32
    n_epochs = 30

    model_types = ["rnn", "lstm", "transformer", "mamba"]

    # ---- 1. 参数量对比 ----
    print("1. 参数量对比")
    print("-" * 50)
    models = {}
    for mt in model_types:
        model = SequenceModelLM(mt, vocab_size, d_model, n_layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        models[mt] = model
        print(f"  {mt:<15} {n_params:>10,} 参数")

    # ---- 2. 训练收敛速度 ----
    print(f"\n2. 训练收敛速度 (位置依赖模式任务, {n_epochs} epochs)")
    print("-" * 50)

    criterion = nn.CrossEntropyLoss()

    # 生成位置依赖数据: token[t] = (token[t-1] + t) % vocab_size
    def make_data():
        data = torch.zeros(batch_size, seq_len + 1, dtype=torch.long, device=device)
        data[:, 0] = torch.randint(0, vocab_size, (batch_size,), device=device)
        for t in range(1, seq_len + 1):
            data[:, t] = (data[:, t - 1] + t) % vocab_size
        return data[:, :-1], data[:, 1:]

    for mt in model_types:
        model = models[mt]
        # 重新初始化
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

        print(f"  {mt:<15} 初始={losses[0]:.3f}  最终={losses[-1]:.3f}")

    # ---- 3. 推理速度对比 ----
    print(f"\n3. 推理速度对比 (seq_len={seq_len})")
    print("-" * 50)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for mt in model_types:
        model = models[mt].eval()
        # 预热
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
        print(f"  {mt:<15} {elapsed:>8.2f} ms/batch  {tokens_sec:>10,.0f} tokens/s")

    # ---- 4. 复杂度分析 ----
    print(f"\n4. 计算复杂度分析 (d={d_model}, N={seq_len}, n=16)")
    print("-" * 50)
    n_state = 16
    rnn_flops = seq_len * d_model ** 2
    tf_flops = seq_len ** 2 * d_model
    mamba_flops = seq_len * d_model * n_state

    print(f"  RNN/LSTM 训练:     O(N·d²) = {rnn_flops:>12,}")
    print(f"  Transformer 训练:  O(N²·d) = {tf_flops:>12,}")
    print(f"  Mamba 训练:        O(N·d·n) = {mamba_flops:>12,}")
    print(f"\n  推理每步 (生成 1 个 token):")
    print(f"  RNN/LSTM:          O(d²)   = {d_model**2:>12,}   内存: O(d)={d_model}")
    print(f"  Transformer:       O(N·d)  = {seq_len*d_model:>12,}   内存: O(N·d)={seq_len*d_model}")
    print(f"  Mamba:             O(d·n)  = {d_model*n_state:>12,}   内存: O(d·n)={d_model*n_state}")

    print(f"""
5. 核心总结
{'─' * 50}

  RNN/LSTM:
    ✓ 推理 O(1), 内存 O(d)
    ✗ 训练串行, 梯度消失, 固定压缩规则
    → 无法选择性记忆, 长序列表现差

  Transformer:
    ✓ 训练完全并行, 长距离建模强, 精确回忆
    ✗ O(N²) 计算, O(N) KV cache, 推理随序列变慢
    → 短-中长度序列的最佳选择, 但长序列成本爆炸

  Mamba (平衡方案):
    ✓ 推理 O(1) (像 RNN), 训练可并行 (像 Transformer)
    ✓ 选择性压缩: Δ大→记忆, Δ小→遗忘 (超越 RNN 的固定压缩)
    ✓ 线性复杂度 O(N·d·n), 长序列友好
    ✗ 精确回忆不如 Transformer (信息被压缩到固定状态)
    → 长序列的高效方案, 配合少量 Attention 层可弥补精确回忆弱点
""")


if __name__ == "__main__":
    compare_architectures()
