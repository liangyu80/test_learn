"""
轻量级 GPT 语言模型 —— 用于 PPO/RLHF 训练

设计目标:
    1. 足够小，能在 Mac CPU/MPS 上流畅运行（~1M 参数）
    2. 架构完整，包含真实 GPT 的所有关键组件
    3. 提供 Actor (策略模型) 和 Critic (价值模型) 两个变体

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
    │  LM Head (→ vocab)  或  Value Head (→ 1) │
    └────────────────────────────────────────┘
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
    vocab_size: int = 256       # 词表大小（用字节级 token 简化，256 个可能值）
    max_seq_len: int = 128      # 最大序列长度
    d_model: int = 128          # 隐藏层维度
    n_heads: int = 4            # 注意力头数
    n_layers: int = 4           # Transformer 层数
    d_ff: int = 512             # FFN 内部维度 (通常 4 × d_model)
    dropout: float = 0.1        # Dropout 率
    bias: bool = False          # 是否在线性层中使用 bias（现代 LLM 倾向不用）


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

        其中:
            Q = X · W_Q    (Query: "我在找什么")
            K = X · W_K    (Key:   "我有什么")
            V = X · W_V    (Value: "我提供什么")
            d_k = d_model / n_heads (每个头的维度)

        因果掩码: 将上三角 (未来位置) 的注意力分数设为 -∞，
        经过 softmax 后变为 0，从而阻断未来信息。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads  # 每个头的维度 d_k

        # QKV 合并投影（一个矩阵同时计算 Q, K, V，更高效）
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # 输出投影
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 预计算因果掩码 (下三角矩阵)
        # 注册为 buffer 使其不被视为可训练参数，但会随模型一起保存/移动设备
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: shape = (batch, seq_len, d_model)
        返回:
            out: shape = (batch, seq_len, d_model)
        """
        B, T, C = x.shape  # batch, seq_len, d_model

        # 1. 计算 Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 2. 拆分为多头: (B, T, d_model) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数: QKᵀ / √d_k
        # 缩放因子 √d_k 防止在 d_k 较大时，点积值过大导致 softmax 梯度消失
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # 4. 应用因果掩码: 未来位置设为 -inf
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        # 5. Softmax 归一化 → 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 加权聚合 Value
        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # 7. 合并多头: (B, n_heads, T, head_dim) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. 输出投影
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

    每个位置独立地进行非线性变换，是 Transformer 中引入非线性的关键组件。
    第一个 Linear 将维度从 d_model 扩展到 d_ff (通常 4×d_model)，
    第二个 Linear 将维度压缩回 d_model。

    为什么用 GELU 而不是 ReLU:
        GELU(x) = x · Φ(x)   (Φ 是标准正态分布的 CDF)
        GELU 在原点附近是光滑的（不像 ReLU 有不可微点），
        在实践中能带来更好的训练效果。GPT-2/3, BERT 等均采用 GELU。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)          # GELU 激活
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    单个 Transformer 解码器块。

    采用 Pre-Norm 架构 (LayerNorm 在子层之前):
        x → LayerNorm → Attention → + x (残差) → LayerNorm → FFN → + x (残差)

    Pre-Norm vs Post-Norm:
        - Post-Norm (原始 Transformer): x → Attention → + x → LayerNorm
        - Pre-Norm (GPT-2 及之后):     x → LayerNorm → Attention → + x
        Pre-Norm 训练更稳定，不需要 learning rate warmup，是现代 LLM 的标配。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接: 梯度可以直接通过残差路径反向传播，缓解深层网络的梯度消失
        x = x + self.attn(self.ln1(x))  # 注意力子层 + 残差
        x = x + self.ffn(self.ln2(x))   # FFN 子层 + 残差
        return x


# ==============================================================================
# GPT 语言模型 (Actor / Policy Model)
# ==============================================================================

class GPTLanguageModel(nn.Module):
    """
    完整的 GPT 语言模型，作为 PPO 中的 Actor (策略模型)。

    在 RLHF 中的角色:
        - Actor / Policy (π_θ): 根据 prompt 生成回复
        - 输出: 词表上的概率分布 → 用于采样 token
        - PPO 优化目标: 最大化奖励的同时不偏离参考策略太远
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer 主干
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)

        # LM Head: 映射到词表
        # 注意: 很多实现中 LM Head 与 token_emb 共享权重 (weight tying)
        # 这里为简单起见不共享，但实际中共享能减少参数量并提升效果
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        参数初始化。

        使用较小的标准差 (0.02) 初始化，这是 GPT-2 的默认策略。
        原因: Transformer 中有很多残差连接，如果初始值太大，
        随着层数加深，残差累加会导致激活值爆炸。
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。

        参数:
            input_ids:     输入 token 序列, shape = (B, T)
            return_hidden: 是否同时返回最后一层的隐藏状态（Value Head 需要）

        返回:
            logits:  词表上的 logits, shape = (B, T, vocab_size)
            hidden:  最后一层隐藏状态 (如果 return_hidden=True), shape = (B, T, d_model)
        """
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.config.max_seq_len, f"序列长度 {T} 超过最大限制 {self.config.max_seq_len}"

        # 位置索引
        positions = torch.arange(T, device=device)

        # 嵌入
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 最终 LayerNorm
        hidden = self.ln_f(x)

        # LM Head
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        return logits, None

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
            top_k:          Top-K 采样（仅保留概率最高的 K 个候选）

        返回:
            生成的完整序列
        """
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            logits, _ = self.forward(idx_cond)
            next_logits = logits[:, -1, :] / temperature

            # Top-K 采样: 将概率最低的 token 屏蔽掉
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


# ==============================================================================
# Value Model (Critic / 价值模型)
# ==============================================================================

class GPTValueModel(nn.Module):
    """
    价值模型 (Critic / Value Model)，用于 PPO 中估计状态价值 V(s)。

    在 RLHF/PPO 中的角色:
        - Critic: 评估当前状态（已生成的序列）的"好坏程度"
        - 输出: 标量价值 V(s_t)，表示从状态 s_t 开始期望获得的累计奖励
        - 用途:
          1. 计算优势函数 A(s_t, a_t) = R_t - V(s_t)（或通过 GAE 计算）
          2. 优势函数用于 PPO 的策略梯度更新，决定哪些动作应被鼓励/抑制

    架构:
        与策略模型共享 Transformer backbone，但将 LM Head 替换为 Value Head。
        Value Head 将每个位置的隐藏状态映射为标量值。

    为什么需要独立的价值模型:
        - Actor-Critic 架构中，Actor 和 Critic 需要分离
        - 如果共享参数，策略优化和价值估计的梯度会互相干扰
        - 在大规模 RLHF 中通常从同一个预训练模型初始化两个副本
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 与策略模型相同的 Transformer backbone
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        # ---------------------------------------------------------------
        # Value Head: 将 d_model 维隐藏状态映射为标量价值
        #
        # 与 LM Head 的区别:
        #   LM Head:    d_model → vocab_size (预测下一个 token 的分布)
        #   Value Head: d_model → 1          (估计当前状态的价值)
        # ---------------------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),   # Tanh 限制输出范围，稳定价值估计
            nn.Linear(config.d_model, 1),
        )

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
        前向传播，输出每个位置的价值估计。

        参数:
            input_ids: shape = (B, T)

        返回:
            values: shape = (B, T)，每个位置的状态价值 V(s_t)
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)

        # Value Head: (B, T, d_model) → (B, T, 1) → (B, T)
        values = self.value_head(hidden).squeeze(-1)

        return values


# ==============================================================================
# 奖励模型 (Reward Model)
# ==============================================================================

class GPTRewardModel(nn.Module):
    """
    奖励模型 (Reward Model)，用于评估生成文本的质量。

    在 RLHF 中的角色:
        - 替代人类打分，对模型生成的回复给出标量奖励
        - 通常通过人类偏好数据训练: 给定 (prompt, response_A, response_B)，
          人类标注哪个更好，模型学习拟合这个偏好

    训练损失 (Bradley-Terry 模型):
        L = -log σ(r(x, y_w) - r(x, y_l))

        其中:
            r(x, y) = 奖励模型对 (prompt x, response y) 的打分
            y_w     = 人类偏好的回复 (winner)
            y_l     = 人类不偏好的回复 (loser)
            σ       = sigmoid 函数

        直觉: 最大化 "好回复分数 - 差回复分数" 的 sigmoid，
        即让好回复的分数尽可能高于差回复。

    本实现中的简化:
        由于没有真实的人类偏好数据，我们使用一个简单的基于规则的
        奖励函数来替代（见 train.py 中的实现）。
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

        # Reward Head: 输出标量奖励
        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

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
        对整个序列给出一个标量奖励。

        参数:
            input_ids: shape = (B, T)

        返回:
            rewards: shape = (B,)，每个序列的奖励分数
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)

        # 取最后一个 token 的隐藏状态作为整个序列的表示
        # (类似 BERT 的 [CLS] token，但在 GPT 中取最后一个位置)
        last_hidden = hidden[:, -1, :]  # (B, d_model)
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)

        return rewards


# ==============================================================================
# 工具函数
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(config: GPTConfig):
    """打印模型配置和参数量摘要。"""
    print(f"\n模型配置:")
    print(f"  词表大小:      {config.vocab_size}")
    print(f"  最大序列长度:  {config.max_seq_len}")
    print(f"  隐藏维度:      {config.d_model}")
    print(f"  注意力头数:    {config.n_heads}")
    print(f"  层数:          {config.n_layers}")
    print(f"  FFN 维度:      {config.d_ff}")

    policy = GPTLanguageModel(config)
    value = GPTValueModel(config)
    reward = GPTRewardModel(config)

    print(f"\n参数量:")
    print(f"  策略模型 (Actor):  {count_parameters(policy):,}")
    print(f"  价值模型 (Critic): {count_parameters(value):,}")
    print(f"  奖励模型 (Reward): {count_parameters(reward):,}")
    print(f"  总计:              {count_parameters(policy) + count_parameters(value):,} (Actor+Critic)")


if __name__ == "__main__":
    config = GPTConfig()
    print_model_summary(config)

    # 简单的前向传播测试
    policy = GPTLanguageModel(config)
    value_model = GPTValueModel(config)

    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    logits, _ = policy(dummy_input)
    values = value_model(dummy_input)

    print(f"\n前向传播测试:")
    print(f"  输入 shape:    {dummy_input.shape}")
    print(f"  Logits shape:  {logits.shape}")
    print(f"  Values shape:  {values.shape}")

    # 生成测试
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = policy.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"  生成 shape:    {generated.shape}")
    print(f"  生成序列:      {generated[0].tolist()}")
