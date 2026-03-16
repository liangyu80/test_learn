"""
Flow Matching (Rectified Flow) —— 从零实现

核心思想:
    Flow Matching 是扩散模型的简化替代方案。核心区别:
    - DDPM: 学习预测噪声 ε_θ(x_t, t)，需要复杂的噪声调度(β)
    - Flow Matching: 学习预测速度场 v_θ(x_t, t)，直接学习从噪声到数据的"流"

    想象一下: 噪声粒子沿着某条路径"流向"数据点。
    Flow Matching 就是学习这些粒子在每个时刻的速度。

数学框架:
    插值路径 (前向过程):
        x_t = (1-t)·x_0 + t·ε,   t ∈ [0, 1],  ε ~ N(0, I)
        注意: t=0 是数据, t=1 是噪声 (与 DDPM 相反!)

    速度场 (Ground Truth):
        v(x_t, t) = dx_t/dt = ε - x_0  (常数! 不依赖 t)

    训练目标:
        L = E_{t, x_0, ε} [‖v_θ(x_t, t) - (ε - x_0)‖²]
        即: 让网络预测从数据到噪声的速度

    采样 (ODE 求解):
        从 x_1 ~ N(0, I) 出发
        使用 Euler 方法: x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
        逐步从 t=1 走到 t=0

    Rectified Flow 的关键改进:
        通过 "reflow" 操作使路径更直，从而减少采样步数:
        1. 训练初始 Flow Matching 模型
        2. 用模型生成 (x_0, x_1) 配对
        3. 在配对上重新训练 → 路径更直 → 更少步数

与 DDPM/DDIM 对比:
    ┌──────────────────┬─────────────────────┬─────────────────────┐
    │                  │   DDPM/DDIM          │   Flow Matching     │
    ├──────────────────┼─────────────────────┼─────────────────────┤
    │ 学习目标         │ 噪声 ε              │ 速度 v = ε - x_0    │
    │ 噪声调度         │ 需要 (β_start, β_end)│ 不需要!             │
    │ 前向过程         │ 复杂 (ᾱ_t 等)       │ 简单线性插值         │
    │ 时间范围         │ 离散 [0, T]         │ 连续 [0, 1]         │
    │ 采样方式         │ SDE/逆扩散          │ ODE (Euler/RK)      │
    │ 路径形状         │ 弯曲               │ 趋向直线             │
    │ 代码复杂度       │ 较高               │ 显著更低             │
    └──────────────────┴─────────────────────┴─────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class FlowMatchingConfig:
    """Flow Matching 配置。"""
    data_dim: int = 200           # 数据维度
    hidden_dim: int = 256         # 隐藏层维度
    time_emb_dim: int = 64        # 时间嵌入维度
    num_blocks: int = 3           # 残差块数量


# ==============================================================================
# 时间嵌入 (与 DDPM 相同)
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步嵌入 (复用 DDPM 的设计)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ==============================================================================
# 速度场网络 (Velocity Network)
# ==============================================================================

class ResBlock(nn.Module):
    """带时间条件的残差块。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.silu(self.norm1(x))
        h = self.linear1(h)
        h = h + self.time_proj(t_emb)
        h = F.silu(self.norm2(h))
        h = self.linear2(h)
        return h + residual


class VelocityNetwork(nn.Module):
    """
    速度场预测网络 v_θ(x_t, t)。

    与 DDPM 的噪声预测网络结构类似，但:
    - 输入 t 是连续值 [0, 1] 而非离散整数
    - 输出是速度场 v 而非噪声 ε
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        # 时间嵌入
        self.time_emb = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        # 输入投影
        self.input_proj = nn.Linear(config.data_dim, config.hidden_dim)
        # 残差块
        self.blocks = nn.ModuleList([
            ResBlock(config.hidden_dim) for _ in range(config.num_blocks)
        ])
        # 输出
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        预测速度场。

        Args:
            x_t: 插值状态, shape (batch_size, data_dim)
            t:   时间, shape (batch_size,), 值在 [0, 1]
        Returns:
            v: 预测的速度, shape (batch_size, data_dim)
        """
        # 时间嵌入 (乘以 1000 使数值范围与 DDPM 类似)
        t_emb = self.time_emb(t * 1000)
        t_emb = self.time_proj(t_emb)

        h = self.input_proj(x_t) + t_emb
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


# ==============================================================================
# Flow Matching 模型
# ==============================================================================

class FlowMatching(nn.Module):
    """
    Flow Matching (Rectified Flow) 完整模型。

    训练:
        1. 采样 x_0 ~ p_data, ε ~ N(0, I), t ~ U[0, 1]
        2. 计算 x_t = (1-t)·x_0 + t·ε
        3. Ground truth 速度: v = ε - x_0
        4. 损失: ‖v_θ(x_t, t) - v‖²

    采样:
        1. x_1 ~ N(0, I)
        2. for t = 1, 1-dt, 1-2dt, ..., 0:
              x_{t-dt} = x_t - dt · v_θ(x_t, t)
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        self.velocity_net = VelocityNetwork(config)

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        计算 Flow Matching 训练损失。

        对比 DDPM 的训练:
            DDPM:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  L = ‖ε - ε_θ(x_t, t)‖²
            FM:    x_t = (1-t)·x_0 + t·ε,               L = ‖(ε-x_0) - v_θ(x_t, t)‖²

        FM 的公式明显更简洁! 不需要计算 ᾱ_t。
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # 1. 采样时间 t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # 2. 采样噪声 ε ~ N(0, I)
        noise = torch.randn_like(x_0)

        # 3. 计算插值 x_t = (1-t)·x_0 + t·ε
        t_expand = t.unsqueeze(-1)  # (batch, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * noise

        # 4. Ground truth 速度: v = ε - x_0 (就是这么简单!)
        target_v = noise - x_0

        # 5. 网络预测速度
        predicted_v = self.velocity_net(x_t, t)

        # 6. MSE 损失
        loss = F.mse_loss(predicted_v, target_v)
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int = 50,
               device: torch.device = None,
               return_trajectory: bool = False) -> torch.Tensor:
        """
        使用 Euler 方法求解 ODE 进行采样。

        ODE: dx/dt = v_θ(x, t)
        从 t=1 (噪声) 积分到 t=0 (数据)

        Euler 方法: x_{t-Δt} = x_t - Δt · v_θ(x_t, t)

        Args:
            num_samples: 生成样本数
            num_steps:   Euler 步数 (越多越精确，但越慢)
            device:      设备
            return_trajectory: 是否返回轨迹
        """
        if device is None:
            device = next(self.parameters()).device

        self.velocity_net.eval()

        # 从标准正态噪声开始 (t=1)
        x = torch.randn(num_samples, self.config.data_dim, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        # 时间步: 从 1 到 0
        dt = 1.0 / num_steps
        timesteps = torch.linspace(1.0, dt, num_steps, device=device)

        for t_val in timesteps:
            t = torch.full((num_samples,), t_val.item(), device=device)
            v = self.velocity_net(x, t)
            # Euler 更新: x = x - dt * v (从 t 走向 t-dt)
            x = x - dt * v

            if return_trajectory:
                trajectory.append(x.clone())

        self.velocity_net.train()

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_midpoint(self, num_samples: int, num_steps: int = 25,
                        device: torch.device = None) -> torch.Tensor:
        """
        使用中点法 (Midpoint Method) 采样 —— 二阶 ODE 求解器。

        比 Euler 更精确，相同步数下质量更好:
            k1 = v_θ(x_t, t)
            k2 = v_θ(x_t - dt/2 · k1, t - dt/2)
            x_{t-dt} = x_t - dt · k2

        Args:
            num_samples: 生成样本数
            num_steps:   步数
            device:      设备
        """
        if device is None:
            device = next(self.parameters()).device

        self.velocity_net.eval()

        x = torch.randn(num_samples, self.config.data_dim, device=device)
        dt = 1.0 / num_steps
        timesteps = torch.linspace(1.0, dt, num_steps, device=device)

        for t_val in timesteps:
            t = torch.full((num_samples,), t_val.item(), device=device)
            t_mid = torch.full((num_samples,), t_val.item() - dt / 2, device=device)

            # 第一步: 用 Euler 走半步到中点
            k1 = self.velocity_net(x, t)
            x_mid = x - (dt / 2) * k1

            # 第二步: 在中点处评估速度，用它走完整步
            k2 = self.velocity_net(x_mid, t_mid)
            x = x - dt * k2

        self.velocity_net.train()
        return x


# ==============================================================================
# 训练函数
# ==============================================================================

def train_flow_matching(config: FlowMatchingConfig = None, epochs: int = 50,
                        batch_size: int = 128, lr: float = 1e-3,
                        device: torch.device = None) -> dict:
    """训练 Flow Matching 模型。"""
    if config is None:
        config = FlowMatchingConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Flow Matching] 设备: {device}")

    # 生成合成数据 (中心化到 [-1, 1])
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.data_dim, device)
    print(f"[Flow Matching] 合成数据: {data.shape}")

    # 初始化模型
    model = FlowMatching(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Flow Matching] 参数量: {num_params:,}")

    # 训练
    history = {'loss': []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i + batch_size]
            loss = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    # 生成样本
    model.eval()
    generated = model.sample(16, device=device)
    print(f"[Flow Matching] 生成样本: {generated.shape}, "
          f"范围 [{generated.min():.3f}, {generated.max():.3f}]")

    return {'model': model, 'history': history, 'generated_samples': generated}


def _generate_structured_data(num_samples: int, dim: int,
                              device: torch.device) -> torch.Tensor:
    """生成带结构的合成数据 ([-1, 1] 范围)。"""
    data = torch.zeros(num_samples, dim, device=device)
    segment = dim // 5
    for i in range(num_samples):
        mode = i % 5
        start = mode * segment
        end = start + segment
        data[i, start:end] = torch.rand(segment, device=device) * 0.8 + 0.2
        data[i] += torch.rand(dim, device=device) * 0.05
    return data * 2 - 1


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_flow_matching():
    """Flow Matching 演示。"""
    print("=" * 60)
    print("Flow Matching (Rectified Flow) 演示")
    print("=" * 60)

    config = FlowMatchingConfig(data_dim=200, hidden_dim=128, time_emb_dim=32)
    results = train_flow_matching(config=config, epochs=60, batch_size=64)

    model = results['model']
    device = next(model.parameters()).device

    # 对比 Euler vs Midpoint
    import time
    print("\n--- Euler vs Midpoint 采样对比 ---")
    for steps in [50, 20, 10]:
        t0 = time.time()
        s_euler = model.sample(32, num_steps=steps, device=device)
        t_euler = time.time() - t0

        t0 = time.time()
        s_mid = model.sample_midpoint(32, num_steps=steps, device=device)
        t_mid = time.time() - t0

        print(f"  {steps} 步 | Euler: std={s_euler.std():.3f} ({t_euler:.3f}s) | "
              f"Midpoint: std={s_mid.std():.3f} ({t_mid:.3f}s)")

    # 展示 ODE 轨迹
    print("\n--- ODE 轨迹 (从噪声到数据) ---")
    samples, traj = model.sample(4, num_steps=50, device=device, return_trajectory=True)
    for i in range(0, len(traj), max(1, len(traj) // 5)):
        x = traj[i]
        print(f"  t={1.0 - i/50:.2f}: std={x.std():.3f}, "
              f"范围 [{x.min():.3f}, {x.max():.3f}]")

    print("\n[Flow Matching 演示完成]")
    return results


if __name__ == '__main__':
    demo_flow_matching()
