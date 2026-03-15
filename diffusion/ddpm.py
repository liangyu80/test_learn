"""
去噪扩散概率模型 (Denoising Diffusion Probabilistic Model, DDPM) —— 从零实现

核心思想:
    扩散模型通过两个过程来生成数据:
    1. 前向过程 (加噪): 逐步向数据添加高斯噪声，直到变成纯噪声
    2. 反向过程 (去噪): 学习逐步去除噪声，从纯噪声恢复出数据

    这个过程类比于: 把一滴墨水滴入清水（前向），然后学习如何从浑浊的水中
    恢复出那滴墨水的形状（反向）。

数学框架:
    前向过程 (固定，不需要学习):
        q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

        任意时刻的封闭解:
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
        其中 ᾱ_t = ∏_{s=1}^{t} α_s,  α_t = 1 - β_t

    反向过程 (需要学习):
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    训练目标 (简化后):
        L_simple = E_{t, x_0, ε} [ ‖ε - ε_θ(x_t, t)‖² ]
        即: 模型预测加入的噪声 ε，与真实噪声的 MSE

架构概览:
    前向过程 (加噪):
    x_0 → x_1 → x_2 → ... → x_T  (纯噪声)
     ↓      ↓      ↓            ↓
    +ε₁   +ε₂   +ε₃          +ε_T

    反向过程 (去噪):
    x_T → x_{T-1} → ... → x_1 → x_0  (生成样本)
     ↓       ↓              ↓      ↓
    UNet    UNet           UNet   UNet  (噪声预测网络)

与 VAE/GAN 的关键区别:
    1. 不需要编码器 (vs VAE) 或判别器 (vs GAN)
    2. 训练非常稳定，目标函数简单 (MSE)
    3. 生成质量极高，但采样速度慢 (需要多步去噪)
    4. 有坚实的数学基础 (随机微分方程、得分匹配)
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
class DDPMConfig:
    """
    DDPM 配置。

    默认参数适用于简单的 1D/2D 数据演示。
    """
    data_dim: int = 200           # 数据维度 (与 VAE/GAN 对比实验一致)
    hidden_dim: int = 256         # 隐藏层维度
    time_emb_dim: int = 64        # 时间步嵌入维度
    num_timesteps: int = 200      # 扩散步数 T (越大噪声越细腻，但采样越慢)
    beta_start: float = 1e-4      # β 调度起始值
    beta_end: float = 0.02        # β 调度终止值
    beta_schedule: str = 'linear' # β 调度策略: 'linear' 或 'cosine'


# ==============================================================================
# 噪声调度 (Noise Schedule)
# ==============================================================================

class NoiseSchedule:
    """
    噪声调度器: 管理前向过程中各时刻的噪声参数。

    核心参数:
        β_t:  每步添加的噪声方差
        α_t:  1 - β_t
        ᾱ_t:  α_1 × α_2 × ... × α_t (累积乘积)

    ᾱ_t 的直觉:
        ᾱ_t 越大 → 该时刻保留的原始信号越多
        ᾱ_t 越小 → 该时刻噪声越大
        ᾱ_T ≈ 0   → 最终时刻几乎是纯噪声
    """

    def __init__(self, config: DDPMConfig, device: torch.device):
        self.num_timesteps = config.num_timesteps
        self.device = device

        # 计算 β 调度
        if config.beta_schedule == 'linear':
            # 线性调度: β 从 beta_start 线性增长到 beta_end
            self.betas = torch.linspace(
                config.beta_start, config.beta_end,
                config.num_timesteps, device=device
            )
        elif config.beta_schedule == 'cosine':
            # 余弦调度 (Improved DDPM 提出):
            # 在训练后期保留更多信号，生成质量更好
            steps = torch.linspace(0, 1, config.num_timesteps + 1, device=device)
            alphas_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]  # 归一化
            betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
            self.betas = betas.clamp(max=0.999)  # 防止数值问题
        else:
            raise ValueError(f"未知的 β 调度: {config.beta_schedule}")

        # 预计算所有需要的参数 (避免重复计算)
        self.alphas = 1.0 - self.betas                    # α_t
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)  # ᾱ_t
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}

        # 前向过程参数
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)            # √ᾱ_t
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)  # √(1-ᾱ_t)

        # 反向过程参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/√α_t
        # 后验方差: σ²_t = β_t × (1-ᾱ_{t-1}) / (1-ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向过程: 给 x_0 添加 t 步噪声。

        封闭解公式:
            x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε,  ε ~ N(0, I)

        这个公式允许我们直接从 x_0 跳到任意 x_t，
        不需要逐步添加噪声（训练时的关键加速）。

        Args:
            x_0: 原始数据, shape (batch_size, data_dim)
            t:   时间步, shape (batch_size,)
            noise: 可选的预生成噪声
        Returns:
            x_t:   加噪后的数据
            noise: 使用的噪声 (训练目标)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 提取对应时间步的参数, shape → (batch_size, 1)
        sqrt_ab = self.sqrt_alphas_bar[t].unsqueeze(-1)
        sqrt_1_ab = self.sqrt_one_minus_alphas_bar[t].unsqueeze(-1)

        # x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise
        return x_t, noise


# ==============================================================================
# 时间步嵌入 (Sinusoidal Time Embedding)
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦位置编码用于时间步嵌入。

    与 Transformer 的位置编码相同的原理:
    使用不同频率的正弦/余弦函数编码时间步信息。

    这样做的好处:
        1. 不同时间步有不同的嵌入向量
        2. 相近的时间步有相似的嵌入
        3. 可以泛化到训练时未见过的时间步
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步, shape (batch_size,)
        Returns:
            嵌入向量, shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        # 频率从低到高: exp(-log(10000) × i / half_dim)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        # 拼接 sin 和 cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ==============================================================================
# 噪声预测网络 (Noise Prediction Network)
# ==============================================================================

class NoisePredictor(nn.Module):
    """
    噪声预测网络 ε_θ(x_t, t)。

    输入: 加噪数据 x_t 和时间步 t
    输出: 预测的噪声 ε

    在实际应用中通常使用 U-Net，这里使用简化的 MLP 版本演示核心原理。

    关键设计:
        1. 时间步通过正弦嵌入编码后，与数据特征融合
        2. 使用残差连接加速训练
        3. 使用 GroupNorm (简化为 LayerNorm) 稳定训练
    """

    def __init__(self, config: DDPMConfig):
        super().__init__()

        # 时间步嵌入
        self.time_emb = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.hidden_dim),
            nn.SiLU(),  # Swish 激活函数 (扩散模型常用)
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # 数据编码
        self.input_proj = nn.Linear(config.data_dim, config.hidden_dim)

        # 主干网络: 带残差连接的 MLP
        self.blocks = nn.ModuleList([
            ResBlock(config.hidden_dim) for _ in range(3)
        ])

        # 输出层
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        预测加入的噪声。

        Args:
            x_t: 加噪数据, shape (batch_size, data_dim)
            t:   时间步, shape (batch_size,)
        Returns:
            predicted_noise: 预测的噪声, shape (batch_size, data_dim)
        """
        # 时间步嵌入
        t_emb = self.time_emb(t)        # (batch, time_emb_dim)
        t_emb = self.time_proj(t_emb)   # (batch, hidden_dim)

        # 数据编码
        h = self.input_proj(x_t)        # (batch, hidden_dim)

        # 融合时间信息并通过残差块
        h = h + t_emb  # 简单相加融合
        for block in self.blocks:
            h = block(h, t_emb)

        # 输出预测噪声
        return self.output_proj(h)


class ResBlock(nn.Module):
    """
    带时间条件的残差块。

    结构:
        输入 → LayerNorm → SiLU → Linear → (+时间嵌入) → LayerNorm → SiLU → Linear → (+残差)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     特征, shape (batch, dim)
            t_emb: 时间嵌入, shape (batch, dim)
        Returns:
            输出特征, shape (batch, dim)
        """
        residual = x
        h = self.norm1(x)
        h = F.silu(h)
        h = self.linear1(h)
        h = h + self.time_proj(t_emb)  # 注入时间信息
        h = self.norm2(h)
        h = F.silu(h)
        h = self.linear2(h)
        return h + residual  # 残差连接


# ==============================================================================
# DDPM 完整模型
# ==============================================================================

class DDPM(nn.Module):
    """
    去噪扩散概率模型 (DDPM)。

    整合噪声调度器和噪声预测网络，提供训练和采样接口。
    """

    def __init__(self, config: DDPMConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.noise_schedule = NoiseSchedule(config, device)
        self.noise_predictor = NoisePredictor(config)

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        计算训练损失。

        简化目标:
            L = E_{t, ε} [‖ε - ε_θ(x_t, t)‖²]

        步骤:
            1. 随机采样时间步 t ~ Uniform(0, T-1)
            2. 随机采样噪声 ε ~ N(0, I)
            3. 计算 x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε
            4. 用网络预测 ε̂ = ε_θ(x_t, t)
            5. 损失 = MSE(ε, ε̂)

        Args:
            x_0: 真实数据, shape (batch_size, data_dim)
        Returns:
            损失 (标量)
        """
        batch_size = x_0.size(0)

        # 1. 随机采样时间步
        t = torch.randint(0, self.config.num_timesteps, (batch_size,),
                          device=self.device)

        # 2. 前向加噪
        x_t, noise = self.noise_schedule.q_sample(x_0, t)

        # 3. 预测噪声
        predicted_noise = self.noise_predictor(x_t, t)

        # 4. MSE 损失
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int, return_trajectory: bool = False) -> torch.Tensor:
        """
        DDPM 采样: 从纯噪声逐步去噪生成数据。

        反向过程:
            x_T ~ N(0, I)
            for t = T-1, T-2, ..., 0:
                ε̂ = ε_θ(x_t, t)
                μ = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε̂)
                x_{t-1} = μ + σ_t × z,  z ~ N(0, I)  (t > 0)

        Args:
            num_samples:       生成样本数
            return_trajectory: 是否返回完整去噪轨迹
        Returns:
            生成的样本 (或完整轨迹)
        """
        self.noise_predictor.eval()
        ns = self.noise_schedule

        # 从纯噪声开始
        x_t = torch.randn(num_samples, self.config.data_dim, device=self.device)
        trajectory = [x_t.clone()] if return_trajectory else None

        for t_val in reversed(range(self.config.num_timesteps)):
            t = torch.full((num_samples,), t_val, device=self.device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.noise_predictor(x_t, t)

            # 计算均值 μ
            # μ = (1/√α_t) × (x_t - (β_t / √(1-ᾱ_t)) × ε̂)
            beta_t = ns.betas[t_val]
            sqrt_recip_alpha = ns.sqrt_recip_alphas[t_val]
            sqrt_one_minus_ab = ns.sqrt_one_minus_alphas_bar[t_val]

            mu = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_ab * predicted_noise)

            if t_val > 0:
                # 添加噪声 (最后一步不加)
                sigma = torch.sqrt(ns.posterior_variance[t_val])
                z = torch.randn_like(x_t)
                x_t = mu + sigma * z
            else:
                x_t = mu

            if return_trajectory and t_val % 20 == 0:
                trajectory.append(x_t.clone())

        self.noise_predictor.train()

        if return_trajectory:
            return x_t, trajectory
        return x_t


# ==============================================================================
# 训练函数
# ==============================================================================

def train_ddpm(config: DDPMConfig = None, epochs: int = 40,
               batch_size: int = 128, lr: float = 1e-3,
               device: torch.device = None) -> dict:
    """
    训练 DDPM 模型。

    Args:
        config:     DDPM 配置
        epochs:     训练轮数
        batch_size: 批大小
        lr:         学习率
        device:     计算设备
    Returns:
        包含训练历史的字典
    """
    if config is None:
        config = DDPMConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[DDPM] 设备: {device}")
    print(f"[DDPM] 扩散步数 T={config.num_timesteps}, β调度={config.beta_schedule}")

    # ---------- 生成合成数据 ----------
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.data_dim, device)
    print(f"[DDPM] 合成数据: {data.shape}, 范围 [{data.min():.2f}, {data.max():.2f}]")

    # ---------- 初始化模型 ----------
    model = DDPM(config, device).to(device)
    optimizer = torch.optim.Adam(model.noise_predictor.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.noise_predictor.parameters())
    print(f"[DDPM] 噪声预测网络参数量: {num_params:,}")

    # ---------- 训练循环 ----------
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
            # 梯度裁剪 (扩散模型训练常用)
            torch.nn.utils.clip_grad_norm_(model.noise_predictor.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    # ---------- 生成样本 ----------
    model.eval()
    generated = model.sample(16)
    print(f"[DDPM] 生成样本 shape: {generated.shape}, "
          f"范围 [{generated.min():.3f}, {generated.max():.3f}]")

    return {
        'model': model,
        'history': history,
        'generated_samples': generated,
    }


def _generate_structured_data(num_samples: int, dim: int,
                              device: torch.device) -> torch.Tensor:
    """
    生成带结构的合成数据（与 VAE/GAN 对比实验相同）。

    创建 5 种不同的"模式"——每种模式在不同位置有高激活值。
    注意: 扩散模型的输入不需要限制在 [0,1]，这里将数据中心化到 [-1,1]。
    """
    data = torch.zeros(num_samples, dim, device=device)
    segment = dim // 5

    for i in range(num_samples):
        mode = i % 5
        start = mode * segment
        end = start + segment
        data[i, start:end] = torch.rand(segment, device=device) * 0.8 + 0.2
        data[i] += torch.rand(dim, device=device) * 0.05

    # 中心化到 [-1, 1] (扩散模型的常见做法)
    data = data * 2 - 1
    return data


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_ddpm():
    """DDPM 演示: 训练并展示去噪过程。"""
    print("=" * 60)
    print("去噪扩散概率模型 (DDPM) 演示")
    print("=" * 60)

    config = DDPMConfig(
        data_dim=200,
        hidden_dim=128,
        time_emb_dim=32,
        num_timesteps=100,    # 演示用较少步数
        beta_schedule='linear',
    )

    results = train_ddpm(config=config, epochs=50, batch_size=64)

    # 展示训练收敛
    print("\n--- 训练收敛 ---")
    history = results['history']
    print(f"  初始损失: {history['loss'][0]:.6f}")
    print(f"  最终损失: {history['loss'][-1]:.6f}")
    print(f"  损失下降: {history['loss'][0] - history['loss'][-1]:.6f}")

    # 展示去噪轨迹
    print("\n--- 去噪轨迹 ---")
    model = results['model']
    model.eval()
    samples, trajectory = model.sample(4, return_trajectory=True)
    print(f"  轨迹步数: {len(trajectory)}")
    for i, x in enumerate(trajectory):
        print(f"  步骤 {i}: 范围 [{x.min():.3f}, {x.max():.3f}], "
              f"std={x.std():.3f}")

    # 前向加噪演示
    print("\n--- 前向加噪演示 ---")
    x_0 = _generate_structured_data(1, config.data_dim,
                                     next(model.parameters()).device)
    ns = model.noise_schedule
    for t_val in [0, 25, 50, 75, 99]:
        t = torch.tensor([t_val], device=x_0.device)
        x_t, _ = ns.q_sample(x_0, t)
        print(f"  t={t_val:3d}: 范围 [{x_t.min():.3f}, {x_t.max():.3f}], "
              f"std={x_t.std():.3f}, ᾱ_t={ns.alphas_bar[t_val]:.4f}")

    print("\n[DDPM 演示完成]")
    return results


if __name__ == '__main__':
    demo_ddpm()
