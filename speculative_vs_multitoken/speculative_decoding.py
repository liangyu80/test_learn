"""
投机采样 (Speculative Decoding) 实现

核心思想:
    投机采样是一种加速大语言模型 (LLM) 推理的方法。它利用一个小型的"草稿模型"(draft model)
    快速生成多个候选 token，然后用大型的"目标模型"(target model) 并行验证这些候选 token。
    由于目标模型可以一次性并行评估多个 token（前向传播一次即可），比逐个生成 token 更高效。

    关键数学原理 —— 接受/拒绝采样:
    对于草稿模型生成的每个候选 token x:
      - 令 q(x) = 草稿模型给出的概率
      - 令 p(x) = 目标模型给出的概率
      - 接受概率 = min(1, p(x) / q(x))

    这个接受准则保证了最终输出的分布与仅使用目标模型时完全一致（无损加速）。

参考论文:
    - "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
    - "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ==============================================================================
# 简化的 Transformer 模型（用于演示）
# ==============================================================================

class SimpleTransformerLM(nn.Module):
    """
    简化的 Transformer 语言模型。

    这是一个用于演示投机采样的简化模型。在实际应用中，草稿模型和目标模型
    通常是不同规模的预训练模型（如 GPT-2 Small 作为草稿，GPT-2 XL 作为目标）。

    参数:
        vocab_size:  词表大小
        d_model:     模型隐藏层维度
        n_heads:     注意力头数
        n_layers:    Transformer 层数
        max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token 嵌入层：将离散 token 映射到连续向量空间
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置嵌入层：为每个位置提供位置信息（可学习的位置编码）
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer 解码器层
        # 每一层包含:
        #   1. 多头自注意力 (Multi-Head Self-Attention) —— 捕获序列内 token 之间的依赖关系
        #   2. 前馈网络 (Feed-Forward Network) —— 对每个位置独立进行非线性变换
        #   3. 残差连接 + 层归一化 (Residual + LayerNorm)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # FFN 内部维度通常是 d_model 的 4 倍
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 语言模型头 (LM Head)：将隐藏状态映射回词表空间，输出每个 token 的 logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            input_ids: 输入 token 序列, shape = (batch_size, seq_len)

        返回:
            logits: 每个位置对应词表上的未归一化分数, shape = (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 2. Token 嵌入 + 位置嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 3. 生成因果掩码 (Causal Mask)
        # 因果掩码确保每个位置只能关注它自己及之前的位置，防止信息"穿越时间"泄漏
        # True 表示被屏蔽的位置（不可见）
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # 4. Transformer 解码器前向传播
        # 注意：这里使用 memory=x 作为简化处理（自回归模式）
        # 在标准的编码器-解码器架构中，memory 来自编码器的输出
        x = self.transformer(x, memory=x, tgt_mask=causal_mask)

        # 5. LM Head：映射到词表维度
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def get_next_token_probs(
        self, input_ids: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        获取下一个 token 的概率分布。

        参数:
            input_ids:   输入序列, shape = (batch_size, seq_len)
            temperature: 采样温度。
                         temperature < 1.0 → 分布更尖锐（更确定性）
                         temperature > 1.0 → 分布更平坦（更随机）
                         temperature = 1.0 → 原始分布

        返回:
            probs: 下一个 token 的概率分布, shape = (batch_size, vocab_size)
        """
        logits = self.forward(input_ids)
        # 只取最后一个位置的 logits（自回归预测下一个 token）
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        return probs


# ==============================================================================
# 投机采样核心算法
# ==============================================================================

class SpeculativeDecoder:
    """
    投机采样解码器。

    工作流程:
        1. 草稿阶段 (Draft Phase):
           使用小模型（快但不太准）自回归生成 K 个候选 token。
           同时记录每个 token 被草稿模型选中的概率 q(x_i)。

        2. 验证阶段 (Verification Phase):
           将包含所有候选 token 的序列一次性送入大模型（慢但准），
           获得大模型在每个位置的概率 p(x_i)。
           大模型只需做一次前向传播就能并行评估所有候选位置。

        3. 接受/拒绝阶段 (Accept/Reject Phase):
           对每个候选 token x_i，按照以下规则决定是否接受:
             - 生成均匀随机数 r ~ Uniform(0, 1)
             - 如果 r < min(1, p(x_i) / q(x_i))，则接受该 token
             - 否则拒绝该 token，并从修正分布中重新采样
             - 一旦某个 token 被拒绝，后续所有候选 token 也被拒绝

        4. 修正采样 (Correction Sampling):
           当第 i 个 token 被拒绝时，从修正分布中采样一个新 token:
             p'(x) = max(0, p(x) - q(x)) / Z
           其中 Z 是归一化常数，确保 p'(x) 是一个有效的概率分布。
           这个修正分布保证最终输出与目标模型的分布一致。

    参数:
        draft_model:  草稿模型（小而快）
        target_model: 目标模型（大而准）
        K:            每次投机猜测的 token 数量（投机长度）
    """

    def __init__(
        self,
        draft_model: SimpleTransformerLM,
        target_model: SimpleTransformerLM,
        K: int = 4,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.K = K  # 投机长度：每次草稿模型生成多少个候选 token

    @torch.no_grad()
    def draft_step(
        self, input_ids: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        草稿阶段：使用草稿模型自回归生成 K 个候选 token。

        参数:
            input_ids:   当前已有的序列, shape = (1, current_len)
            temperature: 采样温度

        返回:
            draft_tokens:      草稿模型生成的候选 token 列表
            draft_probs_list:  每个候选 token 对应的草稿模型概率分布
        """
        draft_tokens = []
        draft_probs_list = []
        current_ids = input_ids.clone()

        for _ in range(self.K):
            # 获取草稿模型的下一个 token 概率分布
            probs = self.draft_model.get_next_token_probs(current_ids, temperature)
            draft_probs_list.append(probs)

            # 从概率分布中采样一个 token
            next_token = torch.multinomial(probs, num_samples=1)  # shape = (1, 1)
            draft_tokens.append(next_token)

            # 将新 token 追加到序列中，供下一步使用
            current_ids = torch.cat([current_ids, next_token], dim=-1)

        return draft_tokens, draft_probs_list

    @torch.no_grad()
    def verify_step(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[torch.Tensor],
        draft_probs_list: List[torch.Tensor],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        验证阶段 + 接受/拒绝 + 修正采样。

        这是投机采样的核心步骤。目标模型一次前向传播评估所有候选 token，
        然后逐个决定是否接受。

        参数:
            input_ids:        原始输入序列
            draft_tokens:     草稿模型生成的候选 token
            draft_probs_list: 草稿模型对每个候选 token 的概率分布
            temperature:      采样温度

        返回:
            accepted_ids:     经过验证后接受的完整序列
        """
        # ============================
        # 步骤 1: 构建包含所有候选 token 的完整序列
        # ============================
        # 将草稿 token 拼接到原始序列后面
        full_sequence = torch.cat([input_ids] + draft_tokens, dim=-1)

        # ============================
        # 步骤 2: 目标模型一次前向传播
        # ============================
        # 这是投机采样提速的关键——目标模型只需一次前向传播
        # 就能同时获得所有位置的概率分布
        target_logits = self.target_model(full_sequence)
        target_logits = target_logits / temperature

        # 获取每个候选位置上目标模型给出的概率分布
        # 原始序列长度为 n，候选 token 位于位置 n, n+1, ..., n+K-1
        # 但 logits[i] 预测的是位置 i+1 的 token，因此:
        #   target_probs[n-1] → 预测位置 n 的 token（即第一个草稿 token 的位置）
        n = input_ids.shape[1]

        # ============================
        # 步骤 3: 逐个验证候选 token（接受/拒绝）
        # ============================
        accepted_ids = input_ids.clone()

        for i in range(self.K):
            # 获取目标模型在第 i 个候选位置的概率分布
            target_probs_i = F.softmax(target_logits[:, n - 1 + i, :], dim=-1)

            # 获取草稿模型在第 i 个候选位置的概率分布
            draft_probs_i = draft_probs_list[i]

            # 获取草稿模型选择的 token
            candidate_token = draft_tokens[i]  # shape = (1, 1)
            token_id = candidate_token.item()

            # ---------------------------------------------------------------
            # 接受准则 (Acceptance Criterion):
            #
            #   接受概率 = min(1, p(x) / q(x))
            #
            # 其中:
            #   p(x) = 目标模型给 token x 的概率
            #   q(x) = 草稿模型给 token x 的概率
            #
            # 直觉理解:
            #   - 如果 p(x) >= q(x)：目标模型比草稿模型更看好这个 token → 一定接受
            #   - 如果 p(x) < q(x)：目标模型没那么看好 → 按比例概率接受
            #
            # 为什么这样做是正确的 (数学证明的核心):
            #   考虑 token x 最终被输出的概率:
            #     P(output x) = q(x) * min(1, p(x)/q(x))
            #                 = q(x) * p(x)/q(x)   [当 p(x) < q(x)]
            #                 = p(x)
            #   或
            #     P(output x) = q(x) * 1            [当 p(x) >= q(x)]
            #                 = q(x)
            #   再加上修正采样的贡献，最终总概率恰好等于 p(x)。
            # ---------------------------------------------------------------
            p_x = target_probs_i[0, token_id].item()
            q_x = draft_probs_i[0, token_id].item()

            # 计算接受概率
            if q_x == 0:
                # 避免除零错误；如果草稿模型概率为 0，理论上不会采到这个 token
                acceptance_prob = 1.0 if p_x > 0 else 0.0
            else:
                acceptance_prob = min(1.0, p_x / q_x)

            # 生成均匀随机数，决定是否接受
            r = torch.rand(1).item()

            if r < acceptance_prob:
                # ✓ 接受这个候选 token
                accepted_ids = torch.cat([accepted_ids, candidate_token], dim=-1)
            else:
                # ✗ 拒绝这个候选 token
                # -------------------------------------------------------
                # 修正采样 (Correction Sampling / Rejection Resampling):
                #
                # 当草稿 token 被拒绝时，我们需要从修正分布中重新采样：
                #   p'(x) = max(0, p(x) - q(x)) / Z
                #
                # 其中 Z = Σ_x max(0, p(x) - q(x)) 是归一化常数
                #
                # 直觉理解：
                #   修正分布 p'(x) 补偿了接受采样中"漏掉"的概率质量。
                #   对于 p(x) > q(x) 的 token，它们在接受阶段被全部接受
                #   （概率为 1），但实际应有更高的概率（p(x) > q(x)），
                #   因此修正分布给它们额外的概率质量 p(x) - q(x)。
                #
                # 数学上可以证明：接受采样 + 修正采样的组合，
                # 最终输出分布严格等于目标分布 p(x)。
                # -------------------------------------------------------
                corrected_probs = torch.clamp(target_probs_i - draft_probs_i, min=0)

                # 归一化（确保概率和为 1）
                correction_sum = corrected_probs.sum()
                if correction_sum > 0:
                    corrected_probs = corrected_probs / correction_sum
                else:
                    # 极端情况：修正分布为零向量（p 和 q 完全相同）
                    # 退化为均匀分布
                    corrected_probs = torch.ones_like(corrected_probs) / corrected_probs.shape[-1]

                # 从修正分布中采样一个新 token
                new_token = torch.multinomial(corrected_probs, num_samples=1)
                accepted_ids = torch.cat([accepted_ids, new_token], dim=-1)

                # 一旦拒绝，后续所有候选 token 都不再验证
                break
        else:
            # 如果所有 K 个候选 token 都被接受了，
            # 我们还可以额外从目标模型在最后一个位置的分布中采样一个 token
            # （因为目标模型已经计算了这个位置的概率，不采白不采）
            bonus_probs = F.softmax(target_logits[:, n - 1 + self.K, :], dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            accepted_ids = torch.cat([accepted_ids, bonus_token], dim=-1)

        return accepted_ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        使用投机采样生成序列。

        参数:
            input_ids:      初始输入序列
            max_new_tokens: 最大生成 token 数
            temperature:    采样温度

        返回:
            output_ids: 生成的完整序列
            stats:      统计信息（用于性能分析）
        """
        current_ids = input_ids.clone()
        initial_len = input_ids.shape[1]

        # 统计信息
        stats = {
            "total_draft_tokens": 0,     # 草稿模型总共生成的候选 token 数
            "total_accepted_tokens": 0,  # 被目标模型接受的 token 数
            "num_iterations": 0,         # 投机采样的迭代次数
            "target_model_calls": 0,     # 目标模型前向传播次数
            "draft_model_calls": 0,      # 草稿模型前向传播次数
        }

        while current_ids.shape[1] - initial_len < max_new_tokens:
            len_before = current_ids.shape[1]

            # 草稿阶段：小模型快速生成 K 个候选
            draft_tokens, draft_probs = self.draft_step(current_ids, temperature)
            stats["draft_model_calls"] += self.K

            # 验证阶段：大模型并行验证
            current_ids = self.verify_step(
                current_ids, draft_tokens, draft_probs, temperature
            )
            stats["target_model_calls"] += 1

            # 统计本轮接受了多少 token
            newly_added = current_ids.shape[1] - len_before
            stats["total_draft_tokens"] += self.K
            stats["total_accepted_tokens"] += newly_added
            stats["num_iterations"] += 1

        # 截断到目标长度
        output_ids = current_ids[:, : initial_len + max_new_tokens]

        # 计算接受率
        if stats["total_draft_tokens"] > 0:
            stats["acceptance_rate"] = (
                stats["total_accepted_tokens"] / stats["total_draft_tokens"]
            )
        else:
            stats["acceptance_rate"] = 0.0

        return output_ids, stats


# ==============================================================================
# 训练函数（标准语言模型训练，用于投机采样中的草稿模型和目标模型）
# ==============================================================================

def train_standard_lm(
    model: SimpleTransformerLM,
    data: torch.Tensor,
    epochs: int = 5,
    lr: float = 1e-3,
) -> List[float]:
    """
    标准语言模型训练。

    损失函数 (Loss Function):
        使用标准的交叉熵损失 (Cross-Entropy Loss)。

        对于序列 [x_1, x_2, ..., x_T]，损失定义为:
            L = -1/T * Σ_{t=1}^{T-1} log P(x_{t+1} | x_1, ..., x_t)

        其中 P(x_{t+1} | x_1, ..., x_t) 是模型在给定前缀 x_1..x_t 条件下
        预测下一个 token x_{t+1} 的概率。

        直觉理解:
        - 交叉熵衡量模型预测分布与真实分布之间的"距离"
        - 最小化交叉熵等价于最大化模型对真实下一个 token 的预测概率
        - 这就是经典的"Next Token Prediction"范式

    参数:
        model: 要训练的语言模型
        data:  训练数据, shape = (num_samples, seq_len)
        epochs: 训练轮数
        lr:    学习率

    返回:
        losses: 每个 epoch 的平均损失
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 交叉熵损失函数
    # ignore_index=-100 表示忽略标签为 -100 的位置（用于 padding）
    criterion = nn.CrossEntropyLoss()

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for i in range(data.shape[0]):
            # 输入是 x_1..x_{T-1}，目标是 x_2..x_T（向右移一位）
            input_ids = data[i : i + 1, :-1]   # shape = (1, seq_len - 1)
            target_ids = data[i : i + 1, 1:]    # shape = (1, seq_len - 1)

            # 前向传播
            logits = model(input_ids)  # shape = (1, seq_len - 1, vocab_size)

            # 计算交叉熵损失
            # 需要 reshape:
            #   logits: (seq_len - 1, vocab_size) → 每个位置的预测
            #   target: (seq_len - 1,)            → 每个位置的真实 token
            loss = criterion(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1),
            )

            # 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / data.shape[0]
        losses.append(avg_loss)
        print(f"  [标准LM训练] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


# ==============================================================================
# 演示与测试
# ==============================================================================

def demo_speculative_decoding():
    """演示投机采样的完整流程。"""
    print("=" * 70)
    print("投机采样 (Speculative Decoding) 演示")
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

    # 创建草稿模型（小模型：2层，128维）和目标模型（大模型：4层，256维）
    print("\n1. 创建模型...")
    draft_model = SimpleTransformerLM(
        vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2
    ).to(device)
    target_model = SimpleTransformerLM(
        vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=4
    ).to(device)

    draft_params = sum(p.numel() for p in draft_model.parameters())
    target_params = sum(p.numel() for p in target_model.parameters())
    print(f"   草稿模型参数量: {draft_params:,}")
    print(f"   目标模型参数量: {target_params:,}")
    print(f"   参数比例: 1:{target_params / draft_params:.1f}")

    # 生成模拟训练数据
    print("\n2. 训练模型...")
    train_data = torch.randint(0, vocab_size, (50, seq_len), device=device)

    print("   训练草稿模型:")
    train_standard_lm(draft_model, train_data, epochs=3, lr=1e-3)

    print("   训练目标模型:")
    train_standard_lm(target_model, train_data, epochs=3, lr=1e-3)

    # 使用投机采样生成
    print("\n3. 使用投机采样生成...")
    decoder = SpeculativeDecoder(draft_model, target_model, K=4)

    prompt = torch.randint(0, vocab_size, (1, 5), device=device)
    print(f"   输入序列: {prompt[0].tolist()}")

    output, stats = decoder.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"   输出序列: {output[0].tolist()}")

    # 输出统计信息
    print("\n4. 性能统计:")
    print(f"   投机迭代次数:     {stats['num_iterations']}")
    print(f"   草稿 token 总数:  {stats['total_draft_tokens']}")
    print(f"   接受 token 总数:  {stats['total_accepted_tokens']}")
    print(f"   接受率:           {stats['acceptance_rate']:.2%}")
    print(f"   目标模型调用次数: {stats['target_model_calls']}")
    print(f"   草稿模型调用次数: {stats['draft_model_calls']}")

    # ---------------------------------------------------------------
    # 效率分析:
    # 假设目标模型单次前向传播时间为 T_target，草稿模型为 T_draft。
    #
    # 标准自回归解码生成 N 个 token:
    #   耗时 = N * T_target
    #
    # 投机采样解码生成 N 个 token (设每轮平均接受 α 个 token):
    #   迭代次数 ≈ N / α
    #   每轮耗时 ≈ K * T_draft + T_target
    #   总耗时 ≈ (N / α) * (K * T_draft + T_target)
    #
    # 加速比 ≈ N * T_target / [(N / α) * (K * T_draft + T_target)]
    #        = α * T_target / (K * T_draft + T_target)
    #
    # 当 T_draft << T_target 且 α 接近 K 时，加速比接近 α ≈ K
    # 典型加速比为 2x - 3x
    # ---------------------------------------------------------------
    if stats["num_iterations"] > 0:
        avg_tokens_per_iter = stats["total_accepted_tokens"] / stats["num_iterations"]
        print(f"   每轮平均生成 token: {avg_tokens_per_iter:.1f}")
        print(f"   (标准自回归每轮仅生成 1 个 token)")

    return stats


if __name__ == "__main__":
    demo_speculative_decoding()
