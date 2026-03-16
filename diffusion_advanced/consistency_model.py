"""
Consistency Model (CM) —— 从零实现

核心思想:
    Consistency Model 的目标是学习一个函数 f_θ，使得 ODE 轨迹上
    任意一点都能直接映射到起点 x_0 (数据)。

    核心性质 (自一致性):
        f_θ(x_t, t) = f_θ(x_t', t')   对所有 t, t' 在同一 ODE 轨迹上

    这意味着: 无论你在轨迹的哪个位置，模型都能一步跳到 x_0!

    训练后只需 1-2 步采样即可生成高质量样本。

两种训练方式:
    1. Consistency Distillation (CD): 从预训练的扩散模型蒸馏
       - 需要 teacher 模型
       - 质量更好
    2. Consistency Training (CT): 从头训练
       - 不需要 teacher
       - 更独立

    本实现采用 Consistency Training (CT)。

训练过程 (CT):
    1. 采样 x_0 ~ p_data, t ~ [ε, T]
    2. 加噪得到 x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    3. 找相邻时刻 t' = t - Δt  (Δt 随训练逐渐增大)
    4. 加噪得到 x_{t'} = √ᾱ_{t'} · x_0 + √(1-ᾱ_{t'}) · ε
    5. 损失: ‖f_θ(x_t, t) - f_{θ⁻}(x_{t'}, t')‖²
       其中 θ⁻ 是 EMA 目标网络 (stop gradient)

    关键约束 (边界条件):
        f_θ(x_ε, ε) = x_ε  (在 t≈0 时是恒等映射)

架构概览:
    ┌─────────────┐     ┌─────────────┐
    │   x_t, t    │     │  x_{t'}, t' │
    │  (在线网络)  │     │ (EMA 目标网络)│
    │   f_θ(·)    │     │  f_{θ⁻}(·)  │
    ├─────────────┤     ├─────────────┤
    │   output_1  │     │  output_2   │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           └───── 损失 ────────┘
           ‖output_1 - output_2‖²

与 DDPM/DDIM/Flow Matching 对比:
    ┌──────────────┬──────────┬──────────┬──────────────┬──────────────┐
    │              │ DDPM     │ DDIM     │ Flow Matching│ Consistency  │
    ├──────────────┼──────────┼──────────┼──────────────┼──────────────┤
    │ 最少采样步数 │ T        │ ~50      │ ~10-50       │ 1-2 步!      │
    │ 训练目标     │ 预测噪声 │ (同DDPM) │ 预测速度     │ 自一致性     │
    │ 是否需蒸馏   │ 否       │ 否       │ 否           │ CT不需要     │
    │ 单步质量     │ 差       │ 中       │ 中           │ 好           │
    └──────────────┴──────────┴──────────┴──────────────┴──────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class ConsistencyConfig:
    """Consistency Model 配置。"""
    data_dim: int = 200           # 数据维度
    hidden_dim: int = 256         # 隐藏层维度
    time_emb_dim: int = 64        # 时间嵌入维度
    num_blocks: int = 3           # 残差块数量
    sigma_min: float = 0.002      # 最小噪声水平 ε
    sigma_max: float = 80.0       # 最大噪声水平 T
    rho: float = 7.0              # Karras 噪声调度的 ρ 参数
    ema_rate: float = 0.999       # EMA 衰减率


# ==============================================================================
# 噪声调度 (Karras schedule)
# ==============================================================================

def karras_schedule(n_steps: int, sigma_min: float, sigma_max: float,
                    rho: float, device: torch.device) -> torch.Tensor:
    """
    Karras 噪声调度 (用于 Consistency Model)。

    公式: σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

    这种调度在低噪声区域有更密集的采样点，
    因为低噪声区域的去噪更重要。

    Args:
        n_steps:   离散化步数
        sigma_min: 最小 σ
        sigma_max: 最大 σ
        rho:       调度参数
        device:    设备
    Returns:
        sigmas: 噪声水平序列, shape (n_steps,), 从大到小
    """
    ramp = torch.linspace(0, 1, n_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


# ==============================================================================
# 时间嵌入
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入。"""

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
# Consistency 网络
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


class ConsistencyNetwork(nn.Module):
    """
    Consistency Model 网络 f_θ(x, σ)。

    满足边界条件:
        f_θ(x, σ_min) = x  (在最小噪声时恒等映射)

    实现方式 (Skip Connection 参数化):
        F_θ(x, σ) 是原始网络输出
        c_skip(σ) 和 c_out(σ) 是依赖 σ 的缩放系数

        f_θ(x, σ) = c_skip(σ) · x + c_out(σ) · F_θ(x, σ)

        其中:
        c_skip(σ) = σ_data² / (σ² + σ_data²)
        c_out(σ)  = σ · σ_data / √(σ² + σ_data²)

        当 σ → σ_min ≈ 0 时:
        c_skip → 1, c_out → 0  → f = x (满足边界条件!)
    """

    def __init__(self, config: ConsistencyConfig):
        super().__init__()
        self.config = config
        self.sigma_data = 0.5  # 数据标准差的估计值

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

    def _skip_scaling(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 skip connection 的缩放系数。

        c_skip(σ) = σ_data² / (σ² + σ_data²)        → σ→0 时趋向 1
        c_out(σ)  = σ · σ_data / √(σ² + σ_data²)    → σ→0 时趋向 0
        """
        sigma_data_sq = self.sigma_data ** 2
        sigma_sq = sigma ** 2
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma_sq + sigma_data_sq)
        return c_skip.unsqueeze(-1), c_out.unsqueeze(-1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (满足边界条件)。

        Args:
            x:     输入 (加噪数据), shape (batch, data_dim)
            sigma: 噪声水平, shape (batch,)
        Returns:
            去噪输出, shape (batch, data_dim)
        """
        c_skip, c_out = self._skip_scaling(sigma)

        # 网络前向
        t_emb = self.time_emb(sigma)
        t_emb = self.time_proj(t_emb)

        h = self.input_proj(x) + t_emb
        for block in self.blocks:
            h = block(h, t_emb)
        F_out = self.output_proj(h)

        # Skip connection: f(x, σ) = c_skip · x + c_out · F(x, σ)
        return c_skip * x + c_out * F_out


# ==============================================================================
# Consistency Model 完整训练
# ==============================================================================

class ConsistencyModel:
    """
    Consistency Model (Consistency Training 版本)。

    训练步骤:
        1. 初始化在线网络 f_θ 和 EMA 目标网络 f_{θ⁻}
        2. 对每个 batch:
            a. 采样数据 x_0 和噪声 ε
            b. 选择相邻噪声水平 (σ_n, σ_{n+1})
            c. 加噪: x_{σ_n} = x_0 + σ_n · ε,  x_{σ_{n+1}} = x_0 + σ_{n+1} · ε
            d. 计算损失: ‖f_θ(x_{σ_{n+1}}, σ_{n+1}) - f_{θ⁻}(x_{σ_n}, σ_n)‖²
            e. 更新 θ (在线网络)
            f. EMA 更新 θ⁻
    """

    def __init__(self, config: ConsistencyConfig, device: torch.device):
        self.config = config
        self.device = device

        # 在线网络
        self.online_net = ConsistencyNetwork(config).to(device)
        # EMA 目标网络 (初始化为在线网络的副本)
        self.target_net = copy.deepcopy(self.online_net)
        # 冻结目标网络
        for p in self.target_net.parameters():
            p.requires_grad_(False)

    def update_target(self, ema_rate: float):
        """EMA 更新目标网络: θ⁻ = ema_rate · θ⁻ + (1 - ema_rate) · θ"""
        with torch.no_grad():
            for p_target, p_online in zip(self.target_net.parameters(),
                                          self.online_net.parameters()):
                p_target.data.lerp_(p_online.data, 1 - ema_rate)

    def compute_loss(self, x_0: torch.Tensor, n_discrete: int) -> torch.Tensor:
        """
        计算 Consistency Training 损失。

        Args:
            x_0:        真实数据
            n_discrete: 当前离散化步数 (训练中逐渐增大)
        """
        batch_size = x_0.size(0)
        config = self.config

        # 1. 计算噪声调度
        sigmas = karras_schedule(n_discrete, config.sigma_min, config.sigma_max,
                                 config.rho, self.device)

        # 2. 随机选择相邻的噪声水平对 (σ_n, σ_{n+1})
        indices = torch.randint(0, n_discrete - 1, (batch_size,), device=self.device)
        sigma_n = sigmas[indices]       # 较小的噪声 (更接近数据)
        sigma_n1 = sigmas[indices + 1]  # 较大的噪声 (如果 sigmas 从大到小则反之)

        # 确保 sigma_n1 > sigma_n (sigma_n 更小，更接近数据)
        # karras_schedule 返回从大到小，所以 indices 对应大噪声，indices+1 对应小噪声
        # 交换使 sigma_n1 是大噪声
        sigma_n, sigma_n1 = sigma_n1, sigma_n  # 现在 sigma_n 小, sigma_n1 大

        # 3. 采样噪声并加噪
        noise = torch.randn_like(x_0)
        x_sigma_n = x_0 + sigma_n.unsqueeze(-1) * noise       # 小噪声版本
        x_sigma_n1 = x_0 + sigma_n1.unsqueeze(-1) * noise     # 大噪声版本

        # 4. 在线网络处理大噪声版本
        online_out = self.online_net(x_sigma_n1, sigma_n1)

        # 5. 目标网络处理小噪声版本 (stop gradient)
        with torch.no_grad():
            target_out = self.target_net(x_sigma_n, sigma_n)

        # 6. 一致性损失: 两个输出应该相同!
        # 使用 Pseudo-Huber 损失 (比 MSE 更鲁棒)
        loss = pseudo_huber_loss(online_out, target_out)
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int = 1) -> torch.Tensor:
        """
        Consistency Model 采样。

        1 步采样 (核心优势!):
            x_T ~ N(0, σ_max² I)
            x_0 ≈ f_θ(x_T, σ_max)

        多步采样 (质量更好):
            x_T ~ N(0, σ_max² I)
            x̂_0 = f_θ(x_T, σ_max)
            x_{σ_i} = x̂_0 + σ_i · ε   (重新加少量噪声)
            x̂_0 = f_θ(x_{σ_i}, σ_i)    (再去噪)
            重复...
        """
        self.online_net.eval()
        config = self.config

        # 从大噪声开始
        x = torch.randn(num_samples, config.data_dim, device=self.device) * config.sigma_max

        if num_steps == 1:
            # 一步采样!
            sigma = torch.full((num_samples,), config.sigma_max, device=self.device)
            x = self.online_net(x, sigma)
        else:
            # 多步采样: 交替去噪和加噪
            sigmas = karras_schedule(num_steps + 1, config.sigma_min,
                                     config.sigma_max, config.rho, self.device)
            for i, sigma_val in enumerate(sigmas):
                sigma = torch.full((num_samples,), sigma_val.item(), device=self.device)
                x = self.online_net(x, sigma)

                if i < len(sigmas) - 1:
                    # 加少量噪声 (去噪-加噪循环)
                    next_sigma = sigmas[i + 1].item()
                    noise = torch.randn_like(x)
                    x = x + next_sigma * noise

        self.online_net.train()
        return x


def pseudo_huber_loss(x: torch.Tensor, y: torch.Tensor, c: float = 0.00054) -> torch.Tensor:
    """
    Pseudo-Huber 损失函数 (Consistency Model 论文推荐)。

    L(x, y) = √(‖x-y‖² + c²) - c

    比 MSE 更鲁棒:
    - 当差异小时，≈ ‖x-y‖²/(2c) (类似 MSE)
    - 当差异大时，≈ ‖x-y‖ - c (类似 L1，对异常值更鲁棒)
    """
    diff_sq = (x - y).pow(2).sum(dim=-1)
    return (torch.sqrt(diff_sq + c ** 2) - c).mean()


# ==============================================================================
# 训练函数
# ==============================================================================

def train_consistency_model(config: ConsistencyConfig = None, epochs: int = 60,
                            batch_size: int = 128, lr: float = 1e-4,
                            device: torch.device = None) -> dict:
    """
    训练 Consistency Model (CT 模式)。

    关键训练策略:
        - n_discrete (离散化步数) 随训练逐渐增大
          初始较小(2) → 最终较大(50)
          这样模型先学习粗粒度的一致性，再学习细粒度
        - EMA rate 也逐渐增大
    """
    if config is None:
        config = ConsistencyConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Consistency Model] 设备: {device}")

    # 数据
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.data_dim, device)
    print(f"[Consistency Model] 合成数据: {data.shape}")

    # 模型
    cm = ConsistencyModel(config, device)
    optimizer = torch.optim.Adam(cm.online_net.parameters(), lr=lr)
    num_params = sum(p.numel() for p in cm.online_net.parameters())
    print(f"[Consistency Model] 参数量: {num_params:,}")

    # 训练
    history = {'loss': []}
    n_min, n_max = 2, 50  # 离散化步数范围

    for epoch in range(epochs):
        cm.online_net.train()
        epoch_loss = 0.0
        num_batches = 0

        # 逐渐增大离散化步数 (课程学习)
        progress = epoch / max(epochs - 1, 1)
        n_discrete = int(n_min + progress * (n_max - n_min))
        n_discrete = max(n_discrete, 2)

        # EMA rate 随训练增大
        ema_rate = config.ema_rate + (1 - config.ema_rate) * progress * 0.5

        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i + batch_size]

            loss = cm.compute_loss(batch, n_discrete)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cm.online_net.parameters(), 1.0)
            optimizer.step()

            # EMA 更新目标网络
            cm.update_target(ema_rate)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | "
                  f"n_disc: {n_discrete} | ema: {ema_rate:.4f}")

    # 生成样本
    print("\n[Consistency Model] 采样测试:")
    for steps in [1, 2, 5]:
        samples = cm.sample(16, num_steps=steps)
        print(f"  {steps} 步采样: 范围 [{samples.min():.3f}, {samples.max():.3f}], "
              f"std={samples.std():.3f}")

    return {'model': cm, 'history': history}


def _generate_structured_data(num_samples: int, dim: int,
                              device: torch.device) -> torch.Tensor:
    """生成带结构的合成数据。"""
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

def demo_consistency_model():
    """Consistency Model 演示。"""
    print("=" * 60)
    print("Consistency Model 演示")
    print("=" * 60)

    config = ConsistencyConfig(
        data_dim=200,
        hidden_dim=128,
        time_emb_dim=32,
    )

    results = train_consistency_model(config=config, epochs=80, batch_size=64)

    cm = results['model']

    # 对比不同步数
    import time
    print("\n--- 不同步数的采样质量 ---")
    real_data = _generate_structured_data(500, config.data_dim, cm.device)

    for steps in [1, 2, 5, 10]:
        t0 = time.time()
        samples = cm.sample(200, num_steps=steps)
        elapsed = time.time() - t0

        mean_err = abs(samples.mean().item() - real_data.mean().item())
        std_err = abs(samples.std().item() - real_data.std().item())
        print(f"  {steps:2d} 步: mean_err={mean_err:.4f}, std_err={std_err:.4f}, "
              f"耗时={elapsed:.3f}s")

    print("\n  → Consistency Model 的核心优势: 1-2 步即可生成!")
    print("\n[Consistency Model 演示完成]")
    return results


if __name__ == '__main__':
    demo_consistency_model()
