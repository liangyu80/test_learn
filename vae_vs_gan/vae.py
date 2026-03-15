"""
变分自编码器 (Variational Autoencoder, VAE) —— 从零实现

核心思想:
    VAE 是一种生成模型，通过学习数据的潜在分布来生成新样本。
    与普通 Autoencoder 不同，VAE 的编码器输出的是潜在空间的
    均值(μ)和方差(σ²)，而不是确定性的编码向量。

    训练目标是最大化证据下界 (ELBO):
        ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
             = 重构损失        + KL 散度正则化

架构概览:
    ┌─────────────┐
    │   输入 x    │
    ├─────────────┤
    │   编码器    │
    │  (Encoder)  │
    ├──────┬──────┤
    │  μ   │  σ²  │  ← 潜在分布参数
    ├──────┴──────┤
    │ 重参数化技巧 │  ← z = μ + σ·ε, ε~N(0,1)
    │(Reparameter)│
    ├─────────────┤
    │   解码器    │
    │  (Decoder)  │
    ├─────────────┤
    │ 重构输出 x̂  │
    └─────────────┘

与 GAN 的关键区别:
    1. VAE 有显式的概率模型，可以计算似然
    2. 训练稳定，不存在模式崩塌问题
    3. 生成样本可能较模糊（因为使用 MSE/BCE 重构损失）
    4. 潜在空间连续且有意义，可以做插值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class VAEConfig:
    """
    VAE 模型配置。

    默认使用 MNIST 尺寸 (28×28=784)，可调整为其他数据。
    """
    input_dim: int = 784          # 输入维度 (28*28 for MNIST)
    hidden_dim: int = 256         # 隐藏层维度
    latent_dim: int = 16          # 潜在空间维度
    img_channels: int = 1         # 图像通道数
    img_size: int = 28            # 图像尺寸


# ==============================================================================
# 编码器 (Encoder): x → (μ, log σ²)
# ==============================================================================

class Encoder(nn.Module):
    """
    编码器网络。

    将输入 x 映射到潜在分布的参数 μ 和 log σ²。
    使用 log σ² 而非 σ² 是为了数值稳定性（避免取 log 时出现负数）。
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        # 分别输出 μ 和 log σ²
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入数据, shape (batch_size, input_dim)
        Returns:
            mu:     潜在分布均值, shape (batch_size, latent_dim)
            logvar: 潜在分布对数方差, shape (batch_size, latent_dim)
        """
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ==============================================================================
# 解码器 (Decoder): z → x̂
# ==============================================================================

class Decoder(nn.Module):
    """
    解码器网络。

    将潜在变量 z 映射回数据空间，生成重构样本。
    最后使用 Sigmoid 将输出限制在 [0, 1]（对应像素值范围）。
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 潜在变量, shape (batch_size, latent_dim)
        Returns:
            x_recon: 重构输出, shape (batch_size, input_dim)
        """
        return self.net(z)


# ==============================================================================
# VAE 完整模型
# ==============================================================================

class VAE(nn.Module):
    """
    变分自编码器 (VAE)。

    训练过程:
        1. 编码器将输入 x 映射为 (μ, log σ²)
        2. 通过重参数化技巧采样 z = μ + σ·ε
        3. 解码器从 z 重构出 x̂
        4. 损失 = 重构损失 + KL 散度
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick)。

        核心公式: z = μ + σ · ε,  其中 ε ~ N(0, I)

        为什么需要这个技巧？
            直接从 N(μ, σ²) 采样的操作不可微分，无法反向传播。
            通过重参数化，将随机性转移到 ε 上，使得梯度可以
            通过 μ 和 σ 传播回编码器。

        Args:
            mu:     均值, shape (batch_size, latent_dim)
            logvar: 对数方差, shape (batch_size, latent_dim)
        Returns:
            z: 采样的潜在变量, shape (batch_size, latent_dim)
        """
        # std = exp(0.5 * log σ²) = exp(log σ) = σ
        std = torch.exp(0.5 * logvar)
        # ε ~ N(0, I)
        eps = torch.randn_like(std)
        # z = μ + σ · ε
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x: 输入, shape (batch_size, input_dim)
        Returns:
            x_recon: 重构输出
            mu:      潜在分布均值
            logvar:  潜在分布对数方差
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        从先验分布 p(z) = N(0, I) 采样并生成新数据。

        Args:
            num_samples: 生成样本数
            device:      设备
        Returns:
            生成的样本, shape (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    steps: int = 10) -> torch.Tensor:
        """
        在两个样本之间进行潜在空间插值。

        这是 VAE 相比 GAN 的独特优势——潜在空间是连续且有结构的。

        Args:
            x1, x2: 两个输入样本, shape (1, input_dim)
            steps:  插值步数
        Returns:
            插值结果, shape (steps, input_dim)
        """
        with torch.no_grad():
            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)
            # 线性插值
            alphas = torch.linspace(0, 1, steps, device=x1.device)
            interpolations = []
            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decoder(z)
                interpolations.append(x_interp)
            return torch.cat(interpolations, dim=0)


# ==============================================================================
# 损失函数
# ==============================================================================

def vae_loss(x_recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 损失函数 = 重构损失 + KL 散度。

    1. 重构损失 (Reconstruction Loss):
       使用 BCE (Binary Cross Entropy)，衡量重构质量。
       L_recon = -Σ [x·log(x̂) + (1-x)·log(1-x̂)]

    2. KL 散度 (KL Divergence):
       衡量学到的分布 q(z|x) 与先验 p(z)=N(0,I) 的距离。
       KL(N(μ,σ²) || N(0,1)) = -0.5 × Σ (1 + log σ² - μ² - σ²)

       这个公式有解析解，因为两个分布都是高斯分布。

    Args:
        x_recon: 重构输出, shape (batch_size, input_dim)
        x:       原始输入, shape (batch_size, input_dim)
        mu:      编码器输出的均值
        logvar:  编码器输出的对数方差
        kl_weight: KL 项权重 (β-VAE 中可调整)
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # 重构损失: 二元交叉熵 (逐元素求和，再取 batch 平均)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)

    # KL 散度: 解析公式
    # KL = -0.5 * Σ(1 + log σ² - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


# ==============================================================================
# 训练函数
# ==============================================================================

def train_vae(config: VAEConfig = None, epochs: int = 20,
              batch_size: int = 128, lr: float = 1e-3,
              device: torch.device = None) -> dict:
    """
    训练 VAE 模型。

    使用合成数据（随机二值图像）进行演示训练。

    Args:
        config:     VAE 配置
        epochs:     训练轮数
        batch_size: 批大小
        lr:         学习率
        device:     计算设备
    Returns:
        包含训练历史的字典
    """
    if config is None:
        config = VAEConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[VAE] 设备: {device}")

    # ---------- 生成合成数据 ----------
    # 使用带结构的合成数据: 5 种不同的"模式"(简单图案)
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.input_dim, device)
    print(f"[VAE] 合成数据: {data.shape}, 范围 [{data.min():.2f}, {data.max():.2f}]")

    # ---------- 初始化模型 ----------
    model = VAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[VAE] 模型参数量: {num_params:,}")

    # ---------- 训练循环 ----------
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(epochs):
        model.train()
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        num_batches = 0

        # 随机打乱数据
        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i + batch_size]

            # 前向传播
            x_recon, mu, logvar = model(batch)

            # 计算损失
            total_loss, recon_loss, kl_loss = vae_loss(x_recon, batch, mu, logvar)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1

        # 记录历史
        avg_total = epoch_total / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        history['total_loss'].append(avg_total)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Total: {avg_total:.2f} | "
                  f"Recon: {avg_recon:.2f} | "
                  f"KL: {avg_kl:.2f}")

    # ---------- 生成样本 ----------
    model.eval()
    generated = model.generate(16, device)
    print(f"[VAE] 生成样本 shape: {generated.shape}, "
          f"范围 [{generated.min():.3f}, {generated.max():.3f}]")

    return {
        'model': model,
        'history': history,
        'generated_samples': generated,
    }


def _generate_structured_data(num_samples: int, dim: int,
                              device: torch.device) -> torch.Tensor:
    """
    生成带结构的合成数据。

    创建 5 种不同的"模式"——每种模式在不同位置有高激活值。
    这样 VAE/GAN 需要学习多模态分布。
    """
    data = torch.zeros(num_samples, dim, device=device)
    segment = dim // 5  # 每种模式占据的区域

    for i in range(num_samples):
        mode = i % 5  # 循环分配模式
        start = mode * segment
        end = start + segment
        # 在对应区域放置随机高值
        data[i, start:end] = torch.rand(segment, device=device) * 0.8 + 0.2
        # 添加少量全局噪声
        data[i] += torch.rand(dim, device=device) * 0.05

    # 裁剪到 [0, 1]
    data = data.clamp(0, 1)
    return data


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_vae():
    """VAE 演示: 训练并展示结果。"""
    print("=" * 60)
    print("变分自编码器 (VAE) 演示")
    print("=" * 60)

    config = VAEConfig(
        input_dim=200,    # 使用较小维度加速演示
        hidden_dim=128,
        latent_dim=8,
    )

    results = train_vae(config=config, epochs=30, batch_size=64)

    # 展示训练收敛情况
    print("\n--- 训练收敛 ---")
    history = results['history']
    print(f"  初始损失: {history['total_loss'][0]:.2f}")
    print(f"  最终损失: {history['total_loss'][-1]:.2f}")
    print(f"  损失下降: {history['total_loss'][0] - history['total_loss'][-1]:.2f}")

    # 测试潜在空间插值
    print("\n--- 潜在空间插值 ---")
    model = results['model']
    device = next(model.parameters()).device

    x1 = _generate_structured_data(1, config.input_dim, device)
    x2 = _generate_structured_data(1, config.input_dim, device)
    # 让 x2 使用不同模式
    x2 = torch.zeros_like(x2)
    seg = config.input_dim // 5
    x2[0, 3*seg:4*seg] = torch.rand(seg, device=device) * 0.8 + 0.2

    interp = model.interpolate(x1, x2, steps=5)
    print(f"  插值样本数: {interp.shape[0]}")
    print(f"  插值范围: [{interp.min():.3f}, {interp.max():.3f}]")

    print("\n[VAE 演示完成]")
    return results


if __name__ == '__main__':
    demo_vae()
