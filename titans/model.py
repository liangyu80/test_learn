"""
Titans MAC (Memory As Context) 模型

来源论文: "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024)

MAC 架构概览:
    Titans 提出了三种将长期记忆与注意力结合的方式:
        1. MAC (Memory As Context): 将记忆输出作为注意力的额外上下文 ← 本文件实现
        2. MAG (Memory As Gate):    用记忆输出作为门控信号
        3. MAL (Memory As Layer):   将记忆和注意力串联堆叠

    MAC 是效果最好的变体 (特别是在长上下文任务上)。

MAC 单层工作流程:
    ┌──────────────────────────────────────────────────────────────┐
    │                    MAC Layer                                 │
    │                                                              │
    │  输入 x (当前上下文窗口)                                      │
    │  │                                                           │
    │  ├──→ [1] 用 Q 投影查询长期记忆: h = M(Q(x))               │
    │  │                                                           │
    │  ├──→ [2] 拼接三种记忆:                                      │
    │  │    context = [Persistent Memory ∥ h ∥ x]                 │
    │  │              (先验知识)    (长期记忆)  (短期/工作记忆)      │
    │  │                                                           │
    │  ├──→ [3] 注意力层处理拼接后的上下文                          │
    │  │    y = Attention(context)                                 │
    │  │                                                           │
    │  ├──→ [4] 用 y 更新长期记忆 (惊讶度驱动)                     │
    │  │    M_t = α·M_{t-1} + S_t                                 │
    │  │                                                           │
    │  └──→ [5] 门控输出: output = y ⊙ σ(M_new(Q(y)))            │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    三种记忆的角色:
        - 持久记忆 (Persistent Memory):
          可学习的参数矩阵，编码任务无关的通用知识。
          类似人类的"常识"——不需要从输入中学习。

        - 长期记忆 (Long-Term Memory, NMM):
          通过惊讶度梯度更新的 MLP，编码历史信息的抽象。
          类似人类的"经验"——从过去的事件中积累。

        - 短期记忆 (Short-Term Memory, Attention):
          标准注意力机制，关注当前上下文窗口。
          类似人类的"工作记忆"——处理当前任务。

模型参数设计:
    本实现为教学 demo，参数量约 200K-500K，可在 Mac CPU/MPS 上运行。
    论文中的完整模型在大规模数据上训练，参数量可达数十亿。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass

from neural_memory import NeuralMemory


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class TitanConfig:
    """
    Titans MAC 模型配置。

    参数设计考虑:
        - d_model=64, n_layers=2: 足够小以在 CPU 上快速运行
        - context_window=16: 短上下文窗口，强迫模型依赖长期记忆
        - pm_size=4: 少量持久记忆 token
    """
    input_dim: int = 1          # 输入特征维度 (时间序列: 1)
    output_dim: int = 1         # 输出特征维度
    d_model: int = 64           # 隐藏层维度
    n_heads: int = 2            # 注意力头数
    n_layers: int = 2           # MAC 层数
    context_window: int = 16    # 上下文窗口大小 (短期记忆范围)
    pm_size: int = 4            # 持久记忆 token 数量
    # NMM 参数
    nmm_layers: int = 2         # 记忆 MLP 层数
    nmm_hidden: int = 128       # 记忆 MLP 隐藏维度
    alpha: float = 0.999        # 记忆衰减 (遗忘率)
    eta: float = 0.9            # 惊讶度动量
    theta: float = 0.05         # 惊讶度缩放 (学习率)


# ==============================================================================
# MAC Layer (Memory As Context 单层)
# ==============================================================================

class MACLayer(nn.Module):
    """
    MAC (Memory As Context) 层。

    核心: 将长期记忆检索结果作为注意力的"额外上下文"拼接到输入中。

    参数:
        config: 模型配置
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # ---- 持久记忆 (Persistent Memory) ----
        # 可学习的参数，编码任务无关的通用知识
        # shape = (pm_size, d_model)
        self.persistent_memory = nn.Parameter(torch.randn(config.pm_size, d) * 0.02)

        # ---- Query 投影 (用于查询长期记忆) ----
        self.W_Q = nn.Linear(d, d)

        # ---- 神经长期记忆模块 (NMM) ----
        self.nmm = NeuralMemory(
            d_model=d,
            n_layers=config.nmm_layers,
            hidden_dim=config.nmm_hidden,
            alpha=config.alpha,
            eta=config.eta,
            theta=config.theta,
        )

        # ---- 注意力层 ----
        # 处理拼接后的 [PM || LTM || x]
        # 总上下文长度 = pm_size + context_window + context_window
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=d * 4,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
        )

        # ---- 输出映射 ----
        total_ctx = config.pm_size + 2 * config.context_window
        self.out_proj = nn.Linear(total_ctx * d, config.context_window * d)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MAC 层前向传播。

        参数:
            x: 输入序列, shape = (B, context_window, d_model)

        返回:
            输出, shape = (B, context_window, d_model)
        """
        B = x.shape[0]
        d = self.config.d_model
        C = self.config.context_window

        # ---- 步骤 1: 从长期记忆中检索信息 ----
        # 将输入 x 转换为查询向量
        queries = F.normalize(self.silu(self.W_Q(x.reshape(-1, d))), dim=-1)
        # 用 NMM 检索: shape = (B*C, d_model)
        nmm_output = self.nmm.retrieve(queries)
        nmm_output = nmm_output.view(B, C, d)  # (B, C, d)

        # ---- 步骤 2: 拼接三种记忆 ----
        # [Persistent Memory || Long-Term Memory || Short-Term (input)]
        pm = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)  # (B, pm_size, d)
        context = torch.cat([pm, nmm_output, x], dim=1)  # (B, pm+C+C, d)

        # ---- 步骤 3: 注意力处理 ----
        attn_out = self.silu(self.attn_layer(context))  # (B, total_ctx, d)
        y = self.out_proj(attn_out.reshape(B, -1))       # (B, C*d)
        y = y.view(-1, d)                                # (B*C, d)

        # ---- 步骤 4: 更新长期记忆 ----
        # 用注意力输出更新 NMM (惊讶度驱动)
        _, new_params = self.nmm.update(y)

        # ---- 步骤 5: 门控输出 ----
        # 从更新后的记忆中检索，作为门控信号
        gate_query = F.normalize(self.W_Q(y), dim=-1)
        gate_value = self.nmm.retrieve(gate_query)
        gated_output = y * self.sigmoid(gate_value)  # 门控: 用记忆控制信息流

        return gated_output.view(B, C, d)


# ==============================================================================
# Titans MAC 完整模型
# ==============================================================================

class TitanMAC(nn.Module):
    """
    完整的 Titans MAC 模型。

    架构:
        输入 → Embedding → [MAC Layer × N + Residual] → Output Projection

    核心特点:
        1. 上下文窗口化: 将长序列切分为固定大小的窗口 (context_window)
        2. 记忆跨窗口: NMM 在处理每个窗口时持续更新，信息跨窗口传递
        3. 三种记忆协同: PM (先验) + NMM (长期) + Attention (短期)

    处理长序列的方式:
        ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
        │ 窗口1  │→│ 窗口2  │→│ 窗口3  │→│ 窗口4  │
        └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
            │          │          │          │
            ▼          ▼          ▼          ▼
        ┌──────────────────────────────────────┐
        │       NMM (长期记忆, 持续更新)        │
        │   窗口1的信息流向窗口2,3,4...          │
        │   通过参数更新实现"无限上下文"          │
        └──────────────────────────────────────┘

    参数:
        config: 模型配置
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config

        # 输入嵌入
        self.embed = nn.Linear(config.input_dim, config.d_model)

        # MAC 层堆叠
        self.mac_layers = nn.ModuleList([
            MACLayer(config) for _ in range(config.n_layers)
        ])

        # 输出投影
        self.out_proj = nn.Linear(config.d_model * config.context_window, config.output_dim)

        self.silu = nn.SiLU()

    def process_window(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理单个上下文窗口。

        参数:
            x: 输入窗口, shape = (B, context_window, input_dim)

        返回:
            输出预测, shape = (B, output_dim)
        """
        B = x.shape[0]

        # 嵌入
        h = self.embed(x)  # (B, C, d_model)

        # 逐层 MAC 处理 + 残差连接
        for mac_layer in self.mac_layers:
            h = h + self.silu(mac_layer(h))

        # 输出投影
        output = self.out_proj(h.reshape(B, -1))  # (B, output_dim)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理完整序列 (自动切分为窗口)。

        参数:
            x: 输入序列, shape = (B, seq_len, input_dim)

        返回:
            预测序列, shape = (B, seq_len, output_dim)
        """
        B, T, D = x.shape
        C = self.config.context_window
        device = x.device

        # 确保输出 tensor 在正确的设备上
        outputs = torch.zeros(B, T, self.config.output_dim, device=device)

        # 滑动窗口处理
        for t in range(C - 1, T):
            # 取当前窗口: x[t-C+1 : t+1]
            window = x[:, t - C + 1: t + 1, :]  # (B, C, input_dim)
            pred = self.process_window(window)     # (B, output_dim)
            outputs[:, t, :] = pred

        return outputs

    def reset_memory(self):
        """重置所有层的长期记忆。"""
        for layer in self.mac_layers:
            layer.nmm.reset_memory()


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# 独立运行测试
# ==============================================================================

if __name__ == "__main__":
    config = TitanConfig()
    model = TitanMAC(config)

    print(f"Titans MAC 模型参数量: {count_parameters(model):,}")
    print(f"配置: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"context_window={config.context_window}")

    # 测试前向传播
    B, T = 2, 64
    x = torch.randn(B, T, config.input_dim)
    y = model(x)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")
