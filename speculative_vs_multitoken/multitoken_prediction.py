"""
多 Token 预测 (Multi-Token Prediction) 实现

核心思想:
    多 Token 预测是一种训练范式的改进。传统语言模型在每个位置只预测下一个 token
    (Next Token Prediction)，而多 Token 预测让模型同时预测未来的 N 个 token。

    关键区别：这不仅仅是推理时的加速技巧（虽然也能加速推理），更重要的是
    它改变了训练目标本身，让模型学到更好的内部表示。

    架构设计:
    - 共享主干网络 (Shared Backbone): Transformer 主体保持不变
    - 多个独立预测头 (Independent Prediction Heads): 每个头负责预测未来第 k 个 token
      例如: Head-1 预测 t+1, Head-2 预测 t+2, ..., Head-N 预测 t+N

    训练损失:
        L_total = Σ_{k=1}^{N} λ_k * L_k

        其中 L_k 是第 k 个预测头的交叉熵损失，λ_k 是权重系数。
        通常 λ_1（下一个 token 的权重）最大，后续权重递减。

参考论文:
    - "Better & Faster Large Language Models via Multi-token Prediction" (Gloeckle et al., 2024, Meta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ==============================================================================
# 多 Token 预测 Transformer 模型
# ==============================================================================

class MultiTokenTransformerLM(nn.Module):
    """
    多 Token 预测语言模型。

    与标准语言模型相比，关键区别在于:
    1. 共享的 Transformer backbone 提取通用表示
    2. N 个独立的预测头，每个预测未来第 k 个 token
    3. 训练时同时优化所有预测头的损失

    架构图 (N=3 为例):

        输入: [x_1, x_2, x_3, ..., x_T]
               ↓
        ┌──────────────────────┐
        │  Shared Transformer  │  ← 共享主干网络
        │     Backbone         │
        └──────┬───────────────┘
               ↓
           hidden states
          ↙    ↓    ↘
    ┌──────┐ ┌──────┐ ┌──────┐
    │Head 1│ │Head 2│ │Head 3│  ← 独立预测头
    └──┬───┘ └──┬───┘ └──┬───┘
       ↓        ↓        ↓
    预测 t+1  预测 t+2  预测 t+3

    参数:
        vocab_size:     词表大小
        d_model:        模型隐藏层维度
        n_heads:        注意力头数
        n_layers:       Transformer 层数
        n_predict:      同时预测的未来 token 数量 (N)
        max_seq_len:    最大序列长度
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_predict: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_predict = n_predict  # 同时预测的 token 数

        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 共享 Transformer 主干网络
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ---------------------------------------------------------------
        # 多预测头 (Multi-Prediction Heads)
        #
        # Meta 论文中的设计选择:
        # - 每个预测头是一个独立的线性层 (或小型 MLP)
        # - 头之间不共享参数（保持预测的独立性）
        # - Head-k 的输入是共享 backbone 的隐藏状态 h_t
        # - Head-k 的输出是对 token x_{t+k} 的预测
        #
        # 为什么用独立的头而不是一个大的输出层:
        #   预测 x_{t+1} 和 x_{t+2} 是本质不同的任务:
        #   - x_{t+1} 的预测高度依赖局部上下文
        #   - x_{t+2} 的预测需要更多全局推理能力
        #   独立的头让每个预测任务有自己的参数空间
        # ---------------------------------------------------------------
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),       # 非线性激活增强表达能力
                nn.LayerNorm(d_model),
                nn.Linear(d_model, vocab_size),
            )
            for _ in range(n_predict)
        ])

    def forward(
        self, input_ids: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        前向传播，返回所有预测头的 logits。

        参数:
            input_ids: 输入 token 序列, shape = (batch_size, seq_len)

        返回:
            all_logits: 列表，包含 N 个 tensor，每个 shape = (batch_size, seq_len, vocab_size)
                        all_logits[k] 是第 k 个预测头的输出（预测 t+k+1 位置的 token）
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. 嵌入
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 2. 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # 3. Transformer 主干网络（共享表示）
        hidden_states = self.transformer(x, memory=x, tgt_mask=causal_mask)

        # 4. 每个预测头独立预测
        all_logits = []
        for head in self.prediction_heads:
            logits = head(hidden_states)  # shape = (batch_size, seq_len, vocab_size)
            all_logits.append(logits)

        return all_logits

    @torch.no_grad()
    def get_next_token_probs(
        self, input_ids: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        获取下一个 token 的概率分布（仅使用第一个预测头）。

        在推理时如果只需要下一个 token，使用 Head-1 即可。

        参数:
            input_ids:   输入序列
            temperature: 采样温度

        返回:
            probs: 下一个 token 的概率分布
        """
        all_logits = self.forward(input_ids)
        # 使用第一个预测头（预测 t+1）
        next_logits = all_logits[0][:, -1, :] / temperature
        return F.softmax(next_logits, dim=-1)

    @torch.no_grad()
    def get_multi_token_probs(
        self, input_ids: torch.Tensor, temperature: float = 1.0
    ) -> List[torch.Tensor]:
        """
        获取未来 N 个 token 的概率分布（使用所有预测头）。

        这是多 Token 预测在推理时的核心优势——一次前向传播同时预测多个 token。

        参数:
            input_ids:   输入序列
            temperature: 采样温度

        返回:
            probs_list: 列表，包含 N 个概率分布，分别对应 t+1, t+2, ..., t+N
        """
        all_logits = self.forward(input_ids)
        probs_list = []
        for logits in all_logits:
            next_logits = logits[:, -1, :] / temperature
            probs_list.append(F.softmax(next_logits, dim=-1))
        return probs_list


# ==============================================================================
# 多 Token 预测训练
# ==============================================================================

def train_multitoken_lm(
    model: MultiTokenTransformerLM,
    data: torch.Tensor,
    epochs: int = 5,
    lr: float = 1e-3,
    loss_weights: Optional[List[float]] = None,
) -> List[float]:
    """
    多 Token 预测模型的训练函数。

    损失函数详解 (Loss Function):
    =============================

    总损失是所有预测头损失的加权和:

        L_total = Σ_{k=1}^{N} λ_k * L_k

    其中:
        N     = 预测头数量（即同时预测的未来 token 数）
        λ_k   = 第 k 个预测头的权重
        L_k   = 第 k 个预测头的交叉熵损失

    每个预测头的损失:
        L_k = -1/T * Σ_{t=1}^{T-k} log P_k(x_{t+k} | x_1, ..., x_t)

    其中 P_k(x_{t+k} | x_1, ..., x_t) 是第 k 个预测头在给定前缀 x_1..x_t
    条件下预测 x_{t+k} 的概率。

    权重设计 (Loss Weights):
        - 常见策略 1: 均匀权重 λ_k = 1/N
        - 常见策略 2: 递减权重 λ_k = 1/k（对近距离预测给更高权重）
        - 常见策略 3: λ_1 = 1, λ_k = α (k > 1)，其中 α < 1

    为什么多 Token 预测能提升模型质量:
        1. 正则化效果: 预测更远的 token 迫使模型学习更抽象、更鲁棒的表示，
           而不是依赖表面上的局部模式。
        2. 更丰富的梯度信号: 每个训练步骤提供 N 倍的监督信号，
           让 backbone 的表示更信息丰富。
        3. 减少"暴露偏差" (Exposure Bias): 模型在训练时就学会考虑更长
           距离的依赖关系，推理时的误差累积问题得到缓解。
        4. 隐式规划能力: 为了准确预测 x_{t+3}，模型必须在内部"想明白"
           x_{t+1} 和 x_{t+2} 是什么，促进了隐式推理能力。

    参数:
        model:        要训练的多 Token 预测模型
        data:         训练数据, shape = (num_samples, seq_len)
        epochs:       训练轮数
        lr:           学习率
        loss_weights: 各预测头的权重列表，长度 = n_predict

    返回:
        losses: 每个 epoch 的平均总损失
    """
    n_predict = model.n_predict
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 设置损失权重
    if loss_weights is None:
        # 默认策略: 递减权重 (1/k 归一化)
        # Head-1 权重最高，因为预测下一个 token 是最重要的任务
        loss_weights = [1.0 / (k + 1) for k in range(n_predict)]
        # 归一化使得权重之和为 1
        weight_sum = sum(loss_weights)
        loss_weights = [w / weight_sum for w in loss_weights]

    print(f"   损失权重: {[f'{w:.3f}' for w in loss_weights]}")

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        per_head_losses = [0.0] * n_predict  # 跟踪每个预测头的损失

        for i in range(data.shape[0]):
            sequence = data[i : i + 1]  # shape = (1, seq_len)
            seq_len = sequence.shape[1]

            # 前向传播：获取所有预测头的 logits
            all_logits = model(sequence)

            total_loss = torch.tensor(0.0, device=sequence.device)

            for k in range(n_predict):
                # -------------------------------------------------------
                # 第 k 个预测头 (Head-k):
                # - 输入位置:  t = 0, 1, ..., T-k-2
                # - 目标 token: x_{t+k+1} = 序列中位置 t+k+1 的 token
                #
                # 例如 k=0 (标准 next-token):
                #   input:  位置 0..T-2 的 hidden states
                #   target: 位置 1..T-1 的 token (即 x_2..x_T)
                #
                # 例如 k=1 (预测 t+2):
                #   input:  位置 0..T-3 的 hidden states
                #   target: 位置 2..T-1 的 token (即 x_3..x_T)
                #
                # 例如 k=2 (预测 t+3):
                #   input:  位置 0..T-4 的 hidden states
                #   target: 位置 3..T-1 的 token (即 x_4..x_T)
                #
                # 注意: k 越大，可用的训练样本越少（序列末尾的位置不够用）
                # -------------------------------------------------------
                logits_k = all_logits[k]  # shape = (1, seq_len, vocab_size)

                # 确保有足够的位置用于计算损失
                if seq_len <= k + 1:
                    continue

                # 预测位置: 0 到 seq_len - k - 2
                # 目标位置: k+1 到 seq_len - 1
                pred_logits = logits_k[:, : seq_len - k - 1, :]  # 预测端
                target_tokens = sequence[:, k + 1 : seq_len]      # 目标端

                # 计算该预测头的交叉熵损失
                loss_k = criterion(
                    pred_logits.reshape(-1, model.vocab_size),
                    target_tokens.reshape(-1),
                )

                # 加权累加到总损失
                total_loss = total_loss + loss_weights[k] * loss_k
                per_head_losses[k] += loss_k.item()

            # 反向传播 + 梯度更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / data.shape[0]
        losses.append(avg_loss)

        # 打印每个预测头的平均损失
        per_head_avg = [l / data.shape[0] for l in per_head_losses]
        head_info = ", ".join([f"H{k+1}={l:.4f}" for k, l in enumerate(per_head_avg)])
        print(
            f"  [多Token预测训练] Epoch {epoch + 1}/{epochs}, "
            f"Total Loss: {avg_loss:.4f} | {head_info}"
        )

    return losses


# ==============================================================================
# 多 Token 预测推理 (并行解码)
# ==============================================================================

class MultiTokenDecoder:
    """
    多 Token 预测解码器。

    推理策略:
        1. 贪心并行解码 (Greedy Parallel Decoding):
           一次前向传播同时从所有预测头获取结果，直接采用所有头的预测。
           速度最快，但可能有质量损失（因为各头独立预测，可能不连贯）。

        2. 自验证并行解码 (Self-Speculative Decoding):
           类似投机采样，但草稿模型就是自己的多预测头。
           Head-1 作为"验证者"，Head-2..N 作为"猜测者"。
           先采纳所有头的预测，然后用 Head-1 逐个验证。

    参数:
        model:    多 Token 预测模型
        strategy: 解码策略 ("greedy_parallel" 或 "self_speculative")
    """

    def __init__(
        self,
        model: MultiTokenTransformerLM,
        strategy: str = "greedy_parallel",
    ):
        self.model = model
        self.strategy = strategy
        self.n_predict = model.n_predict

    @torch.no_grad()
    def generate_greedy_parallel(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        贪心并行解码。

        每次前向传播同时生成 N 个 token，大幅减少前向传播次数。

        注意事项:
            这种方法的质量取决于各预测头的准确度。Head-k (k>1) 在没有看到
            x_{t+1}..x_{t+k-1} 的情况下直接预测 x_{t+k}，可能不如自回归准确。
            但在实际应用中，经过充分训练的模型可以达到不错的效果。

        参数:
            input_ids:      初始输入序列
            max_new_tokens: 最大生成 token 数
            temperature:    采样温度

        返回:
            output_ids: 生成的完整序列
            stats:      统计信息
        """
        current_ids = input_ids.clone()
        initial_len = input_ids.shape[1]

        stats = {
            "num_forward_passes": 0,   # 前向传播次数
            "total_tokens_generated": 0,  # 生成的总 token 数
            "tokens_per_step": [],     # 每步生成的 token 数
        }

        while current_ids.shape[1] - initial_len < max_new_tokens:
            remaining = max_new_tokens - (current_ids.shape[1] - initial_len)
            if remaining <= 0:
                break

            # 一次前向传播获取所有预测头的结果
            probs_list = self.model.get_multi_token_probs(current_ids, temperature)
            stats["num_forward_passes"] += 1

            # 从每个预测头采样
            new_tokens = []
            for k, probs in enumerate(probs_list):
                if len(new_tokens) >= remaining:
                    break
                token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                new_tokens.append(token)

            # 将新 token 拼接到序列
            if new_tokens:
                new_tokens_tensor = torch.cat(new_tokens, dim=-1)
                current_ids = torch.cat([current_ids, new_tokens_tensor], dim=-1)
                stats["tokens_per_step"].append(len(new_tokens))
                stats["total_tokens_generated"] += len(new_tokens)

        # 截断到目标长度
        output_ids = current_ids[:, : initial_len + max_new_tokens]

        if stats["num_forward_passes"] > 0:
            stats["avg_tokens_per_step"] = (
                stats["total_tokens_generated"] / stats["num_forward_passes"]
            )
        else:
            stats["avg_tokens_per_step"] = 0

        return output_ids, stats

    @torch.no_grad()
    def generate_self_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        自验证并行解码 (Self-Speculative Decoding)。

        将多预测头的结果当作"草稿"，然后用模型自身的 Head-1 进行验证。
        这种方法不需要额外的草稿模型，是多 Token 预测和投机采样的结合。

        工作流程:
            1. 一次前向传播获取所有预测头的输出: Head-1 预测 x_{t+1},
               Head-2 预测 x_{t+2}, ..., Head-N 预测 x_{t+N}
            2. 采纳所有预测作为草稿
            3. 将草稿序列送入模型，用 Head-1 逐个验证
            4. 接受连续正确的 token，在第一个不一致处截断

        参数:
            input_ids:      初始输入序列
            max_new_tokens: 最大生成 token 数
            temperature:    采样温度

        返回:
            output_ids: 生成的完整序列
            stats:      统计信息
        """
        current_ids = input_ids.clone()
        initial_len = input_ids.shape[1]

        stats = {
            "num_forward_passes": 0,
            "num_draft_steps": 0,
            "num_verify_steps": 0,
            "total_tokens_generated": 0,
            "accepted_from_draft": 0,
            "rejected_from_draft": 0,
        }

        while current_ids.shape[1] - initial_len < max_new_tokens:
            remaining = max_new_tokens - (current_ids.shape[1] - initial_len)
            if remaining <= 0:
                break

            # --- 草稿阶段：获取所有预测头的输出 ---
            probs_list = self.model.get_multi_token_probs(current_ids, temperature)
            stats["num_forward_passes"] += 1
            stats["num_draft_steps"] += 1

            # 从各预测头贪心采样
            draft_tokens = []
            for k, probs in enumerate(probs_list):
                token = torch.argmax(probs, dim=-1, keepdim=True)  # 贪心选择
                draft_tokens.append(token)

            # 第一个 token (Head-1 的结果) 总是接受
            current_ids = torch.cat([current_ids, draft_tokens[0]], dim=-1)
            stats["total_tokens_generated"] += 1

            if len(draft_tokens) <= 1 or remaining <= 1:
                continue

            # --- 验证阶段：用 Head-1 验证后续草稿 token ---
            # 将草稿 token 加入序列后，再做一次前向传播
            # Head-1 在位置 t+1 的输出预测 x_{t+2}，
            # 如果与 Head-2 之前预测的 x_{t+2} 一致，则接受
            verify_seq = torch.cat(
                [current_ids] + [t for t in draft_tokens[1:]], dim=-1
            )
            verify_logits = self.model(verify_seq)
            stats["num_forward_passes"] += 1
            stats["num_verify_steps"] += 1

            # 用 Head-1 的结果验证 draft_tokens[1], [2], ...
            n = current_ids.shape[1]  # 第一个草稿 token 已被接受后的序列长度

            for k in range(1, len(draft_tokens)):
                if current_ids.shape[1] - initial_len >= max_new_tokens:
                    break

                # Head-1 在位置 (n-1+k-1) 的预测是对位置 (n+k-1) 的 token 的预测
                verify_probs = F.softmax(
                    verify_logits[0][:, n - 1 + k - 1, :] / temperature, dim=-1
                )
                verified_token = torch.argmax(verify_probs, dim=-1, keepdim=True)
                draft_token = draft_tokens[k]

                if verified_token.item() == draft_token.item():
                    # 验证通过，接受
                    current_ids = torch.cat([current_ids, draft_token], dim=-1)
                    stats["accepted_from_draft"] += 1
                    stats["total_tokens_generated"] += 1
                else:
                    # 验证失败，用 Head-1 的结果替换，并停止
                    current_ids = torch.cat([current_ids, verified_token], dim=-1)
                    stats["rejected_from_draft"] += 1
                    stats["total_tokens_generated"] += 1
                    break

        output_ids = current_ids[:, : initial_len + max_new_tokens]

        if stats["num_forward_passes"] > 0:
            stats["avg_tokens_per_forward"] = (
                stats["total_tokens_generated"] / stats["num_forward_passes"]
            )

        return output_ids, stats

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """根据策略选择解码方法。"""
        if self.strategy == "greedy_parallel":
            return self.generate_greedy_parallel(
                input_ids, max_new_tokens, temperature
            )
        elif self.strategy == "self_speculative":
            return self.generate_self_speculative(
                input_ids, max_new_tokens, temperature
            )
        else:
            raise ValueError(f"未知策略: {self.strategy}")


# ==============================================================================
# 演示与测试
# ==============================================================================

def demo_multitoken_prediction():
    """演示多 Token 预测的完整流程。"""
    print("=" * 70)
    print("多 Token 预测 (Multi-Token Prediction) 演示")
    print("=" * 70)

    torch.manual_seed(42)

    # 自动选择最佳设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n使用设备: {device}")

    vocab_size = 500
    seq_len = 32
    n_predict = 4  # 同时预测 4 个未来 token

    # 创建模型
    print(f"\n1. 创建多 Token 预测模型 (n_predict={n_predict})...")
    model = MultiTokenTransformerLM(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        n_predict=n_predict,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "prediction_heads" not in name
    )
    head_params = total_params - backbone_params
    print(f"   总参数量:      {total_params:,}")
    print(f"   主干网络参数:  {backbone_params:,}")
    print(f"   预测头参数:    {head_params:,} ({n_predict} 个头)")

    # 训练
    print("\n2. 训练模型...")
    train_data = torch.randint(0, vocab_size, (50, seq_len), device=device)

    # 使用递减权重
    loss_weights = [0.4, 0.3, 0.2, 0.1]  # Head-1 权重最高
    print(f"   使用自定义权重: {loss_weights}")

    train_multitoken_lm(
        model, train_data, epochs=3, lr=1e-3, loss_weights=loss_weights
    )

    # 贪心并行解码
    print("\n3. 贪心并行解码...")
    decoder_greedy = MultiTokenDecoder(model, strategy="greedy_parallel")
    prompt = torch.randint(0, vocab_size, (1, 5), device=device)
    print(f"   输入序列: {prompt[0].tolist()}")

    output_greedy, stats_greedy = decoder_greedy.generate(
        prompt, max_new_tokens=20, temperature=1.0
    )
    print(f"   输出序列: {output_greedy[0].tolist()}")
    print(f"   前向传播次数: {stats_greedy['num_forward_passes']}")
    print(f"   每步平均 token: {stats_greedy.get('avg_tokens_per_step', 'N/A')}")

    # 自验证解码
    print("\n4. 自验证并行解码...")
    decoder_spec = MultiTokenDecoder(model, strategy="self_speculative")
    output_spec, stats_spec = decoder_spec.generate(
        prompt, max_new_tokens=20, temperature=1.0
    )
    print(f"   输出序列: {output_spec[0].tolist()}")
    print(f"   前向传播次数: {stats_spec['num_forward_passes']}")
    print(f"   草稿接受数:   {stats_spec['accepted_from_draft']}")
    print(f"   草稿拒绝数:   {stats_spec['rejected_from_draft']}")

    return stats_greedy, stats_spec


if __name__ == "__main__":
    demo_multitoken_prediction()
