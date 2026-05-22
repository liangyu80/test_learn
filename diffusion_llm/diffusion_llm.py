"""
Diffusion LLM — 离散扩散语言模型

三种主流方法的 PyTorch 实现:
1. 吸收态扩散 (Absorbing Diffusion / MDLM 风格) — 用 [MASK] 逐步遮蔽文本
2. 多项式扩散 (Multinomial Diffusion / D3PM 风格) — 用随机 token 替换
3. 连续嵌入扩散 (Embedding Diffusion / Diffusion-LM 风格) — 在嵌入空间加高斯噪声

核心思想:
  传统 LLM (GPT) 是自回归的 — 一个 token 一个 token 从左到右生成。
  Diffusion LLM 是非自回归的 — 所有 token 同时生成，通过多轮迭代去噪。

  前向过程: 干净文本 → 逐步加噪 → 纯噪声
  逆向过程: 纯噪声 → 逐步去噪 → 干净文本 (由神经网络学习)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# 通用组件
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """时间步嵌入 — 将离散时间步 t 编码为连续向量"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class DenoisingTransformerBlock(nn.Module):
    """去噪 Transformer 块 — 双向注意力 + 时间步条件"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.time_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # 时间步条件: 通过加法注入 (AdaLN 的简化版)
        t_bias = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, D)
        h = self.norm1(x + t_bias)
        h = x + self.attn(h, h, h, need_weights=False)[0]
        h = h + self.ffn(self.norm2(h))
        return h


class DenoisingTransformer(nn.Module):
    """
    去噪网络 — 给定噪声序列和时间步，预测干净 token

    注意: 这里用双向注意力 (非因果)，因为 Diffusion LLM 可以看到所有位置。
    这是 Diffusion LLM 与自回归 LLM 的关键区别之一。
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.blocks = nn.ModuleList([
            DenoisingTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: 噪声 token 序列 (B, L) — 整数 token IDs
            t:   时间步 (B,) — 整数, 0 = 干净, T = 纯噪声
        Returns:
            logits: (B, L, V) — 对每个位置预测干净 token 的 logits
        """
        B, L = x_t.shape
        pos = torch.arange(L, device=x_t.device).unsqueeze(0)
        h = self.token_emb(x_t) + self.pos_emb(pos)
        t_emb = self.time_emb(t)  # (B, D)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.norm(h)
        return self.head(h)


class ContinuousDenoisingTransformer(nn.Module):
    """
    连续嵌入空间的去噪网络 — 输入是连续向量而非离散 token

    用于 Embedding Diffusion (Diffusion-LM 风格):
    输入 x_t 是加噪后的连续嵌入向量。
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.blocks = nn.ModuleList([
            DenoisingTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: 加噪嵌入 (B, L, D) — 连续向量
            t:   时间步 (B,)
        Returns:
            x_0_pred: 预测的干净嵌入 (B, L, D)
        """
        B, L, D = x_t.shape
        pos = torch.arange(L, device=x_t.device).unsqueeze(0)
        h = self.input_proj(x_t) + self.pos_emb(pos)
        t_emb = self.time_emb(t)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.norm(h)
        return self.output_proj(h)


# ============================================================
# 方法 1: 吸收态扩散 (Absorbing Diffusion / MDLM)
# ============================================================

class AbsorbingDiffusionLM(nn.Module):
    """
    吸收态离散扩散语言模型 (Masked Diffusion Language Model)

    前向过程: 每一步随机选择一些 token 替换为 [MASK]
      t=0: "the cat sat on the mat"
      t=1: "the [M] sat on the [M]"
      t=2: "[M] [M] sat [M] the [M]"
      ...
      t=T: "[M] [M] [M] [M] [M] [M]"

    逆向过程: 神经网络预测被 mask 的 token，逐步恢复
      t=T: "[M] [M] [M] [M] [M] [M]"
      t=T-1: "[M] [M] sat [M] [M] [M]"  (网络预测出 "sat")
      ...
      t=0: "the cat sat on the mat"

    这与 BERT 的 MLM 预训练非常相似，但有完整的扩散框架:
    - BERT: 一步预测 15% 的 mask (不迭代)
    - MDLM: 从全 mask 开始，迭代去噪多步 (生成式)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 64, num_timesteps: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size  # 额外的 [MASK] token
        self.num_timesteps = num_timesteps
        self.max_seq_len = max_seq_len

        self.denoiser = DenoisingTransformer(
            vocab_size=vocab_size + 1,  # +1 for [MASK]
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )

        # 噪声调度: 线性 mask 概率 — t/T 比例的 token 被 mask
        # t=0 → mask_prob=0 (干净), t=T → mask_prob=1 (全 mask)

    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """计算时间步 t 对应的 mask 概率"""
        return t.float() / self.num_timesteps

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向加噪: 将干净 token 按概率替换为 [MASK]

        Args:
            x_0: 干净 token (B, L)
            t: 时间步 (B,)
        Returns:
            x_t: 加噪 token (B, L) — 部分位置变成 [MASK]
        """
        mask_prob = self.get_mask_prob(t).unsqueeze(-1)  # (B, 1)
        # 每个位置独立地以 mask_prob 概率被 mask
        mask = torch.rand_like(x_0.float()) < mask_prob  # (B, L)
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        训练损失: 随机采样时间步，对被 mask 的位置计算交叉熵

        只在被 mask 的位置计算损失 (类似 BERT MLM)，
        因为未被 mask 的位置网络可以直接看到答案。
        """
        B, L = x_0.shape
        # 随机采样时间步 (1 到 T，不包括 0 因为 t=0 时没有噪声)
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=x_0.device)
        x_t = self.forward_process(x_0, t)
        logits = self.denoiser(x_t, t)  # (B, L, V+1)

        # 只在被 mask 的位置计算损失
        mask = (x_t == self.mask_token_id)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_0.device)

        logits_masked = logits[mask]  # (N_masked, V+1)
        targets_masked = x_0[mask]    # (N_masked,)
        loss = F.cross_entropy(logits_masked, targets_masked)
        return loss

    @torch.no_grad()
    def generate(self, batch_size: int = 1, seq_len: Optional[int] = None,
                 device: str = 'cpu') -> torch.Tensor:
        """
        逆向生成: 从全 [MASK] 开始，逐步去噪

        策略: 每一步选置信度最高的 token 揭示
        """
        self.eval()
        L = seq_len or self.max_seq_len
        # 初始化: 全部 [MASK]
        x = torch.full((batch_size, L), self.mask_token_id, dtype=torch.long, device=device)

        for step in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=device)
            logits = self.denoiser(x, t)  # (B, L, V+1)

            # 对被 mask 的位置，选最可能的 token
            probs = F.softmax(logits[:, :, :self.vocab_size], dim=-1)  # 排除 [MASK] token
            pred_tokens = probs.argmax(dim=-1)  # (B, L)

            # 计算当前步和下一步应该保持 mask 的比例
            current_mask_prob = step / self.num_timesteps
            next_mask_prob = (step - 1) / self.num_timesteps

            if next_mask_prob > 0:
                # 计算每个位置的置信度
                confidence = probs.max(dim=-1).values  # (B, L)
                is_masked = (x == self.mask_token_id)
                # 对非 mask 位置设置高置信度 (不会被重新 mask)
                confidence[~is_masked] = 2.0

                # 按置信度排序，保留置信度最低的位置继续 mask
                n_to_mask = int(next_mask_prob * L)
                if n_to_mask > 0:
                    _, indices = confidence.sort(dim=-1)
                    # 先把所有 mask 位置替换为预测
                    x[is_masked] = pred_tokens[is_masked]
                    # 再把最不确定的位置重新 mask
                    for b in range(batch_size):
                        re_mask_idx = indices[b, :n_to_mask]
                        x[b, re_mask_idx] = self.mask_token_id
                else:
                    x[x == self.mask_token_id] = pred_tokens[x == self.mask_token_id]
            else:
                # 最后一步: 揭示所有 mask
                x[x == self.mask_token_id] = pred_tokens[x == self.mask_token_id]

        return x


# ============================================================
# 方法 2: 多项式扩散 (Multinomial Diffusion / D3PM)
# ============================================================

class MultinomialDiffusionLM(nn.Module):
    """
    多项式离散扩散语言模型 (D3PM 风格)

    前向过程: 每一步以一定概率将 token 替换为随机 token
      t=0: "the cat sat on the mat"
      t=1: "the xyz sat on abc mat"    (部分 token 被随机替换)
      t=2: "qrs xyz pqr lmn abc mat"
      ...
      t=T: "随机 token 序列" (均匀分布)

    与吸收态扩散的区别:
    - 吸收态: 噪声 = [MASK], 信息被"吸收"到一个特殊状态
    - 多项式: 噪声 = 随机 token, 信息被"扩散"到整个词表

    转移矩阵: Q_t = (1 - β_t) * I + β_t * (1/V) * 11^T
    即: 每个 token 以 β_t 概率变成随机 token，以 (1-β_t) 概率保持不变
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 64, num_timesteps: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.max_seq_len = max_seq_len

        self.denoiser = DenoisingTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )

        # 噪声调度: β_t 线性增加
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        # 累积保留概率: 到时间 t 时，原始 token 保持不变的概率
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_cumprod', alpha_cumprod)

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向加噪: 以 (1 - alpha_cumprod_t) 概率替换为随机 token
        """
        B, L = x_0.shape
        alpha_t = self.alpha_cumprod[t - 1].unsqueeze(-1)  # (B, 1)

        # 每个位置: 以 alpha_t 概率保持原始 token，以 (1-alpha_t) 概率变随机
        keep_mask = torch.rand(B, L, device=x_0.device) < alpha_t
        random_tokens = torch.randint(0, self.vocab_size, (B, L), device=x_0.device)

        x_t = torch.where(keep_mask, x_0, random_tokens)
        return x_t

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        训练损失: 在所有位置计算交叉熵 (因为任何位置都可能被替换)
        """
        B, L = x_0.shape
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=x_0.device)
        x_t = self.forward_process(x_0, t)
        logits = self.denoiser(x_t, t)

        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), x_0.reshape(-1))
        return loss

    @torch.no_grad()
    def generate(self, batch_size: int = 1, seq_len: Optional[int] = None,
                 device: str = 'cpu', temperature: float = 0.8) -> torch.Tensor:
        """
        逆向生成: 从随机 token 开始，逐步去噪
        """
        self.eval()
        L = seq_len or self.max_seq_len

        # 初始化: 随机 token (均匀分布 — 纯噪声)
        x = torch.randint(0, self.vocab_size, (batch_size, L), device=device)

        for step in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=device)
            logits = self.denoiser(x, t)

            # 用 temperature 采样来增加多样性
            probs = F.softmax(logits / temperature, dim=-1)
            pred_tokens = torch.multinomial(probs.reshape(-1, self.vocab_size), 1)
            pred_tokens = pred_tokens.reshape(batch_size, L)

            if step > 1:
                # 混合: 部分位置用预测值，部分保持当前值 (模拟逆向转移)
                alpha_t = self.alpha_cumprod[step - 1]
                alpha_prev = self.alpha_cumprod[step - 2]
                # 简化的逆向步骤: 增加保留预测 token 的概率
                update_prob = 1.0 - (alpha_t / alpha_prev).item()
                update_mask = torch.rand(batch_size, L, device=device) < update_prob
                x = torch.where(update_mask, pred_tokens, x)
            else:
                x = pred_tokens

        return x


# ============================================================
# 方法 3: 连续嵌入扩散 (Embedding Diffusion / Diffusion-LM)
# ============================================================

class EmbeddingDiffusionLM(nn.Module):
    """
    连续嵌入空间扩散语言模型 (Diffusion-LM 风格)

    核心思想: 将离散 token 映射到连续嵌入空间，在嵌入空间做标准高斯扩散

    前向过程 (在嵌入空间):
      e_0 = Embed("the cat sat")           # 干净嵌入
      e_1 = √α₁ · e_0 + √(1-α₁) · ε      # 加少量高斯噪声
      e_2 = √α₂ · e_0 + √(1-α₂) · ε      # 加更多高斯噪声
      ...
      e_T ≈ N(0, I)                         # 纯高斯噪声

    逆向过程:
      e_T ~ N(0, I)                         # 从噪声开始
      → 去噪网络预测 e_0                     # 预测干净嵌入
      → 最近邻查找 → token                   # 映射回离散 token

    优点: 可以复用连续扩散的全部理论工具 (DDPM/DDIM 等)
    缺点: 连续-离散往返可能引入误差 (rounding error)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 64, num_timesteps: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.max_seq_len = max_seq_len

        # 嵌入层 — 同时用于 token→embedding 和 embedding→token (最近邻)
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.denoiser = ContinuousDenoisingTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )

        # DDPM 噪声调度
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))

    def tokens_to_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """token IDs → 嵌入向量"""
        return self.embedding(x)

    def embeddings_to_tokens(self, emb: torch.Tensor) -> torch.Tensor:
        """嵌入向量 → 最近邻 token (rounding)"""
        # 计算与所有 token 嵌入的余弦距离，取最近的
        weight = self.embedding.weight  # (V, D)
        # 归一化
        emb_norm = F.normalize(emb, dim=-1)
        weight_norm = F.normalize(weight, dim=-1)
        # 余弦相似度
        sim = torch.matmul(emb_norm, weight_norm.T)  # (B, L, V)
        return sim.argmax(dim=-1)

    def forward_process(self, e_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向加噪 (标准 DDPM):
        e_t = √(ᾱ_t) · e_0 + √(1-ᾱ_t) · ε

        Returns: (e_t, ε)
        """
        sqrt_alpha = self.sqrt_alpha_cumprod[t - 1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t - 1].unsqueeze(-1).unsqueeze(-1)

        noise = torch.randn_like(e_0)
        e_t = sqrt_alpha * e_0 + sqrt_one_minus * noise
        return e_t, noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        训练损失: 预测干净嵌入 (x_0 prediction) 的 MSE

        可以选择预测:
        - ε (噪声预测, DDPM 原始方式)
        - x_0 (干净数据预测, Diffusion-LM 方式)

        这里用 x_0 prediction，因为我们最终需要映射回离散 token，
        预测 x_0 更直接。
        """
        B, L = x_0.shape
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=x_0.device)

        e_0 = self.tokens_to_embeddings(x_0)  # (B, L, D)
        e_t, _ = self.forward_process(e_0, t)

        e_0_pred = self.denoiser(e_t, t)  # (B, L, D)

        # MSE 损失 (嵌入空间)
        mse_loss = F.mse_loss(e_0_pred, e_0)

        # 辅助交叉熵损失 — 帮助嵌入对齐到 token (rounding loss)
        logits = torch.matmul(e_0_pred, self.embedding.weight.T)  # (B, L, V)
        ce_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), x_0.reshape(-1))

        return mse_loss + 0.1 * ce_loss

    @torch.no_grad()
    def generate(self, batch_size: int = 1, seq_len: Optional[int] = None,
                 device: str = 'cpu') -> torch.Tensor:
        """
        逆向生成: DDPM 反向采样 → 嵌入 → token

        从标准高斯噪声开始，逐步去噪得到干净嵌入，最后 rounding 回 token。
        """
        self.eval()
        L = seq_len or self.max_seq_len

        # 初始化: 标准高斯噪声
        e_t = torch.randn(batch_size, L, self.d_model, device=device)

        for step in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=device)
            e_0_pred = self.denoiser(e_t, t)

            if step > 1:
                # DDPM 逆向采样公式
                alpha_t = self.alphas[step - 1]
                alpha_bar_t = self.alpha_cumprod[step - 1]
                alpha_bar_prev = self.alpha_cumprod[step - 2]
                beta_t = self.betas[step - 1]

                # 后验均值
                coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
                coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
                mean = coef1 * e_0_pred + coef2 * e_t

                # 后验方差
                var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                noise = torch.randn_like(e_t)
                e_t = mean + torch.sqrt(var) * noise
            else:
                e_t = e_0_pred

        # 最终: 嵌入 → token (最近邻 rounding)
        tokens = self.embeddings_to_tokens(e_t)
        return tokens


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_absorbing():
    """演示吸收态扩散的前向/逆向过程"""
    print("=" * 60)
    print("吸收态扩散 (Absorbing / MDLM)")
    print("=" * 60)

    vocab_size = 50
    model = AbsorbingDiffusionLM(
        vocab_size=vocab_size, d_model=64, n_heads=4,
        n_layers=2, max_seq_len=16, num_timesteps=20
    )
    print(f"参数量: {count_parameters(model):,}")

    # 模拟一个 batch
    x_0 = torch.randint(0, vocab_size, (2, 16))
    print(f"\n干净输入 x_0:\n{x_0[0].tolist()}")

    # 展示前向过程 (逐步 mask)
    print("\n--- 前向过程 (逐步加噪) ---")
    for t_val in [1, 5, 10, 15, 20]:
        t = torch.tensor([t_val, t_val])
        x_t = model.forward_process(x_0, t)
        n_masked = (x_t[0] == model.mask_token_id).sum().item()
        display = [f"[M]" if tok == model.mask_token_id else f"{tok:3d}" for tok in x_t[0].tolist()]
        print(f"  t={t_val:2d}: {' '.join(display)}  (mask了{n_masked}个)")

    # 训练一步
    loss = model.compute_loss(x_0)
    print(f"\n训练损失: {loss.item():.4f}")

    # 生成
    print("\n--- 逆向生成 ---")
    generated = model.generate(batch_size=1, seq_len=16)
    print(f"生成结果: {generated[0].tolist()}")


def demo_multinomial():
    """演示多项式扩散的前向/逆向过程"""
    print("\n" + "=" * 60)
    print("多项式扩散 (Multinomial / D3PM)")
    print("=" * 60)

    vocab_size = 50
    model = MultinomialDiffusionLM(
        vocab_size=vocab_size, d_model=64, n_heads=4,
        n_layers=2, max_seq_len=16, num_timesteps=20
    )
    print(f"参数量: {count_parameters(model):,}")

    x_0 = torch.randint(0, vocab_size, (2, 16))
    print(f"\n干净输入 x_0:\n{x_0[0].tolist()}")

    # 展示前向过程 (逐步随机替换)
    print("\n--- 前向过程 (逐步加噪) ---")
    for t_val in [1, 5, 10, 15, 20]:
        t = torch.tensor([t_val, t_val])
        x_t = model.forward_process(x_0, t)
        n_changed = (x_t[0] != x_0[0]).sum().item()
        print(f"  t={t_val:2d}: {x_t[0].tolist()}  (变了{n_changed}个)")

    loss = model.compute_loss(x_0)
    print(f"\n训练损失: {loss.item():.4f}")

    print("\n--- 逆向生成 ---")
    generated = model.generate(batch_size=1, seq_len=16)
    print(f"生成结果: {generated[0].tolist()}")


def demo_embedding():
    """演示连续嵌入扩散的前向/逆向过程"""
    print("\n" + "=" * 60)
    print("连续嵌入扩散 (Embedding / Diffusion-LM)")
    print("=" * 60)

    vocab_size = 50
    d_model = 64
    model = EmbeddingDiffusionLM(
        vocab_size=vocab_size, d_model=d_model, n_heads=4,
        n_layers=2, max_seq_len=16, num_timesteps=20
    )
    print(f"参数量: {count_parameters(model):,}")

    x_0 = torch.randint(0, vocab_size, (2, 16))
    print(f"\n干净输入 x_0:\n{x_0[0].tolist()}")

    # 展示前向过程 (嵌入空间加噪)
    e_0 = model.tokens_to_embeddings(x_0)
    print(f"嵌入向量 e_0 形状: {e_0.shape}, 范数: {e_0.norm(dim=-1).mean():.4f}")

    print("\n--- 前向过程 (嵌入空间加噪) ---")
    for t_val in [1, 5, 10, 15, 20]:
        t = torch.tensor([t_val, t_val])
        e_t, _ = model.forward_process(e_0, t)
        # 尝试 round 回 token, 看能恢复多少
        recovered = model.embeddings_to_tokens(e_t)
        acc = (recovered[0] == x_0[0]).float().mean().item()
        print(f"  t={t_val:2d}: 嵌入范数={e_t.norm(dim=-1).mean():.4f}, "
              f"rounding恢复率={acc:.1%}")

    loss = model.compute_loss(x_0)
    print(f"\n训练损失: {loss.item():.4f}")

    print("\n--- 逆向生成 ---")
    generated = model.generate(batch_size=1, seq_len=16)
    print(f"生成结果: {generated[0].tolist()}")


if __name__ == '__main__':
    print("Diffusion LLM — 离散扩散语言模型演示")
    print("三种方法: 吸收态扩散 / 多项式扩散 / 连续嵌入扩散\n")
    torch.manual_seed(42)
    demo_absorbing()
    demo_multinomial()
    demo_embedding()
