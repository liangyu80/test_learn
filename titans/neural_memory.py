"""
Titans 神经长期记忆模块 (Neural Long-Term Memory Module)

来源论文: "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024, Google Research)

核心思想:
    传统 Transformer 的注意力机制是"短期记忆"——只能关注当前上下文窗口内的 token。
    Titans 引入了一个"长期记忆"模块，它是一个小型神经网络 (MLP)，
    通过在推理时不断更新自身参数来记忆历史信息。

    类比人类记忆系统:
        - 注意力 (Attention) → 工作记忆 (短期): 容量有限，关注当前内容
        - 神经记忆 (NMM)    → 长期记忆:      容量大，存储历史抽象信息
        - 持久记忆 (PM)     → 先验知识:      任务无关的通用知识

    关键创新——"惊讶度"驱动的记忆更新:
        人类更容易记住"出乎意料"的事件。Titans 借鉴这一原理，
        用梯度的大小作为"惊讶度"——梯度越大说明当前输入越难预测，
        需要更多地记忆。

记忆更新数学公式:
    ========================================

    给定输入 x_t，计算 key 和 value:
        k_t = normalize(SiLU(x_t · W_K))
        v_t = SiLU(x_t · W_V)

    关联记忆损失 (Associative Memory Loss):
        ℓ(M_{t-1}; x_t) = ||M_{t-1}(k_t) - v_t||²₂
        (M 能否通过 key 正确回忆 value？如果不能，说明这个信息"令人惊讶")

    惊讶度 = 损失的梯度:
        ∇ℓ = ∂ℓ/∂θ_M  (θ_M 是记忆网络 M 的参数)

    带动量的记忆更新 (类比 SGD with Momentum + Weight Decay):
        S_t = η · S_{t-1} - θ · ∇ℓ(M_{t-1}; x_t)    (惊讶度累积)
        M_t = α · M_{t-1} + S_t                        (记忆更新)

    其中:
        S_t:  惊讶度动量 (Surprise Momentum)
        η:    动量衰减系数 (类似 SGD momentum, 通常 0.9)
        θ:    惊讶度缩放 (学习率, 通常 0.01~0.05)
        α:    记忆衰减 (Weight Decay, 遗忘机制, 通常 0.999)

    注意: W_K 和 W_V 初始化后冻结，只有 MLP 层的参数参与更新。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from typing import Dict, Optional, Tuple
import copy


class NeuralMemory(nn.Module):
    """
    神经长期记忆模块 (Neural Long-Term Memory Module)。

    架构: 一个 2 层 MLP，参数在推理过程中通过"惊讶度梯度"不断更新。

        输入 k → [Linear(d, 2d) → SiLU → Linear(2d, d)] → 输出 M(k)

    这个 MLP 本身就是"记忆"——它的参数编码了所有历史信息的抽象。

    与传统 KV Cache 的对比:
        ┌─────────────────┬──────────────────────┬──────────────────────┐
        │     维度         │     KV Cache         │     Neural Memory    │
        ├─────────────────┼──────────────────────┼──────────────────────┤
        │ 存储方式         │ 显式存储所有 KV 对   │ 隐式编码在 MLP 参数中│
        │ 内存增长         │ O(序列长度)          │ O(1) 固定大小        │
        │ 检索方式         │ 注意力 (线性扫描)    │ MLP 前向传播         │
        │ 更新方式         │ 追加新 KV 对         │ 梯度下降更新参数     │
        │ 容量限制         │ 上下文窗口长度       │ MLP 表达能力         │
        │ 遗忘机制         │ 滑动窗口截断         │ 自适应权重衰减       │
        └─────────────────┴──────────────────────┴──────────────────────┘

    参数:
        d_model:     输入/输出维度
        n_layers:    MLP 层数 (论文推荐 2 层)
        hidden_dim:  MLP 隐藏层维度 (通常 2 × d_model)
        alpha:       记忆衰减系数 (遗忘率, 越接近 1 遗忘越慢)
        eta:         惊讶度动量系数 (类似 SGD momentum)
        theta:       惊讶度缩放系数 (类似学习率)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 2,
        hidden_dim: int = 128,
        alpha: float = 0.999,
        eta: float = 0.9,
        theta: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha
        self.eta = eta
        self.theta = theta

        # ---- 记忆网络 (MLP) ----
        # 这个 MLP 的参数就是"长期记忆"的载体
        # 参数在推理过程中会被不断更新 (这是 Titans 的核心创新)
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(d_model, d_model))
        else:
            # 第一层: d_model → hidden_dim + SiLU
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
            ))
            # 中间层 (如果有)
            for _ in range(n_layers - 2):
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ))
            # 最后一层: hidden_dim → d_model (无激活)
            self.layers.append(nn.Linear(hidden_dim, d_model))

        # ---- Key 和 Value 投影 (冻结, 不参与记忆更新) ----
        # W_K: 将输入映射到 key 空间 (用于"写入"和"查询"记忆)
        # W_V: 将输入映射到 value 空间 (存储的目标信息)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)

        self.silu = nn.SiLU()

        # 惊讶度动量: 每个参数一个，初始为零
        self.surprise: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP 前向传播 (用于读取记忆)。

        参数:
            x: 查询向量, shape = (..., d_model)

        返回:
            记忆检索结果, shape = (..., d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        从长期记忆中检索信息。

        使用 functional_call 实现, 确保使用当前的参数值。

        参数:
            query: 查询向量 (已经过 Q 投影), shape = (..., d_model)

        返回:
            检索到的记忆内容, shape = (..., d_model)
        """
        return functional_call(self, dict(self.named_parameters()), query)

    def update(self, x: torch.Tensor) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        用新数据更新长期记忆 (核心算法)。

        完整流程:
            1. 计算 key = normalize(SiLU(x · W_K))
            2. 计算 value = SiLU(x · W_V)
            3. 将 key 送入 MLP, 得到 M(key)
            4. 计算关联记忆损失: ℓ = ||M(key) - value||²₂
            5. 反向传播计算梯度 ∇ℓ (只对 MLP 参数, 不更新 W_K/W_V)
            6. 更新惊讶度动量: S_t = η·S_{t-1} - θ·∇ℓ
            7. 更新记忆参数:   M_t = α·M_{t-1} + S_t

        参数:
            x: 新输入数据, shape = (N, d_model)

        返回:
            loss:           关联记忆损失 (用于监控)
            updated_params: 更新后的参数字典
        """
        # detach: 不让梯度流回主模型
        z = x.detach()

        # 1. 计算 key 和 value
        keys = F.normalize(self.silu(self.W_K(z)), dim=-1)   # (N, d_model)
        vals = self.silu(self.W_V(z))                         # (N, d_model)

        # 2. 将 key 送入 MLP (记忆检索)
        retrieved = keys
        for layer in self.layers:
            retrieved = layer(retrieved)

        # 3. 关联记忆损失: ||M(key) - value||²
        # 衡量记忆能否正确关联 key→value
        # 损失越大 → 这个信息越"令人惊讶"
        loss = ((retrieved - vals) ** 2).mean(dim=0).sum()

        # 4. 计算梯度 (惊讶度)
        grads = torch.autograd.grad(loss, self.parameters())

        # 5. 用动量更新记忆参数
        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            # 跳过 W_K 和 W_V (它们是冻结的)
            if name.startswith("W_K") or name.startswith("W_V"):
                updated_params[name] = param.data
                continue

            # 初始化惊讶度动量 (第一次更新时)
            if name not in self.surprise:
                self.surprise[name] = torch.zeros_like(grad)

            # 惊讶度动量更新:
            # S_t = η · S_{t-1} - θ · ∇ℓ
            # η (eta):   过去的惊讶度保留多少 (动量)
            # θ (theta): 当前惊讶度的权重 (学习率)
            self.surprise[name] = self.eta * self.surprise[name] - self.theta * grad

            # 记忆参数更新:
            # M_t = α · M_{t-1} + S_t
            # α (alpha): 记忆衰减 (遗忘机制)
            # α < 1: 旧记忆逐渐衰减, 为新记忆腾出空间
            updated_params[name] = self.alpha * param.data + self.surprise[name]

            # 就地更新参数
            param.data = updated_params[name]

        return loss.item(), updated_params

    def reset_memory(self):
        """重置记忆 (清除惊讶度动量, 重新初始化 MLP 参数)。"""
        self.surprise = {}
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
