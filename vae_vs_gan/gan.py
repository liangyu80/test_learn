"""
生成对抗网络 (Generative Adversarial Network, GAN) —— 从零实现

核心思想:
    GAN 由两个网络组成——生成器(G)和判别器(D)——进行博弈:
    - 生成器: 从随机噪声生成假样本，试图欺骗判别器
    - 判别器: 区分真实样本和生成的假样本

    训练目标 (极小极大博弈):
        min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

    直觉理解:
        D 想最大化这个目标 → 尽可能区分真假
        G 想最小化这个目标 → 让 D 分辨不出真假

架构概览:
    ┌──────────────────────────────────────────────┐
    │              生成器 (Generator)               │
    │  ┌────────┐    ┌────────┐    ┌────────────┐ │
    │  │ 噪声 z │ →  │ 全连接 │ →  │ 假样本 x̂  │ │
    │  │ ~N(0,I)│    │  网络  │    │ (Sigmoid)  │ │
    │  └────────┘    └────────┘    └────────────┘ │
    └──────────────────────────────────────────────┘
                                        ↓
    ┌──────────────────────────────────────────────┐
    │             判别器 (Discriminator)            │
    │  ┌────────────┐  ┌────────┐  ┌───────────┐  │
    │  │ 真样本 x   │→ │ 全连接 │→ │ P(real)   │  │
    │  │ 或假样本 x̂ │  │  网络  │  │ (Sigmoid) │  │
    │  └────────────┘  └────────┘  └───────────┘  │
    └──────────────────────────────────────────────┘

与 VAE 的关键区别:
    1. 没有显式的概率模型或似然计算
    2. 通过对抗训练隐式学习数据分布
    3. 生成样本通常更清晰锐利
    4. 训练不稳定，容易出现模式崩塌 (mode collapse)
    5. 潜在空间没有显式的结构化约束
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
class GANConfig:
    """
    GAN 模型配置。

    与 VAE 使用相同的数据维度，方便公平比较。
    """
    input_dim: int = 784          # 数据维度 (28*28 for MNIST)
    hidden_dim: int = 256         # 隐藏层维度
    latent_dim: int = 16          # 噪声维度 (与 VAE 潜在空间维度一致)
    img_channels: int = 1         # 图像通道数
    img_size: int = 28            # 图像尺寸


# ==============================================================================
# 生成器 (Generator): z → x̂
# ==============================================================================

class Generator(nn.Module):
    """
    生成器网络。

    将随机噪声 z 映射到数据空间，生成假样本。
    使用 BatchNorm 和 LeakyReLU 来稳定训练
    (这是 GAN 训练的常见技巧)。
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        self.net = nn.Sequential(
            # 第 1 层: latent_dim → hidden_dim
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.LeakyReLU(0.2),

            # 第 2 层: hidden_dim → hidden_dim
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.LeakyReLU(0.2),

            # 输出层: hidden_dim → input_dim
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 随机噪声, shape (batch_size, latent_dim)
        Returns:
            fake_data: 生成的假样本, shape (batch_size, input_dim)
        """
        return self.net(z)


# ==============================================================================
# 判别器 (Discriminator): x → P(real)
# ==============================================================================

class Discriminator(nn.Module):
    """
    判别器网络。

    判断输入数据是真实样本还是生成的假样本。
    输出一个 [0, 1] 的概率值。

    注意:
        - 使用 LeakyReLU 而非 ReLU（避免梯度消失）
        - 不使用 BatchNorm（判别器中 BN 有时会导致不稳定）
        - 使用 Dropout 防止判别器过强
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        self.net = nn.Sequential(
            # 第 1 层
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # 第 2 层
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # 输出层: → 1 (真假概率)
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入数据, shape (batch_size, input_dim)
        Returns:
            prob: 判定为真实数据的概率, shape (batch_size, 1)
        """
        return self.net(x)


# ==============================================================================
# GAN 完整模型
# ==============================================================================

class GAN(nn.Module):
    """
    生成对抗网络 (GAN)。

    封装生成器和判别器，提供统一的接口。

    训练策略 (交替训练):
        1. 固定 G，训练 D: 让 D 更好地区分真假
        2. 固定 D，训练 G: 让 G 更好地欺骗 D
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        从随机噪声生成新样本。

        Args:
            num_samples: 生成样本数
            device:      设备
        Returns:
            生成的样本, shape (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        with torch.no_grad():
            self.generator.eval()
            samples = self.generator(z)
            self.generator.train()
        return samples


# ==============================================================================
# 损失函数
# ==============================================================================

def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """
    判别器损失。

    目标: 最大化 E[log D(x)] + E[log(1 - D(G(z)))]
    等价于最小化: -E[log D(x)] - E[log(1 - D(G(z)))]

    即: D 对真实样本输出接近 1，对假样本输出接近 0。

    Args:
        d_real: 判别器对真实样本的输出, shape (batch_size, 1)
        d_fake: 判别器对假样本的输出, shape (batch_size, 1)
    Returns:
        判别器损失 (标量)
    """
    real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
    fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
    return real_loss + fake_loss


def generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    生成器损失。

    原始目标: min_G E[log(1 - D(G(z)))]
    实践中使用: max_G E[log D(G(z))]  (非饱和损失，梯度更大)
    等价于最小化: -E[log D(G(z))]

    即: G 希望 D 对假样本输出接近 1（被判定为真）。

    Args:
        d_fake: 判别器对假样本的输出, shape (batch_size, 1)
    Returns:
        生成器损失 (标量)
    """
    return F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))


# ==============================================================================
# 训练函数
# ==============================================================================

def train_gan(config: GANConfig = None, epochs: int = 20,
              batch_size: int = 128, lr_g: float = 2e-4, lr_d: float = 2e-4,
              d_steps: int = 1, device: torch.device = None) -> dict:
    """
    训练 GAN 模型。

    使用与 VAE 相同的合成数据进行公平比较。

    训练技巧:
        1. 使用 Adam 优化器，β1=0.5 (GAN 训练常用设置)
        2. 每个 epoch 交替训练 D 和 G
        3. 可选: 多步训练 D (d_steps > 1)

    Args:
        config:     GAN 配置
        epochs:     训练轮数
        batch_size: 批大小
        lr_g:       生成器学习率
        lr_d:       判别器学习率
        d_steps:    每次训练 G 前训练 D 的次数
        device:     计算设备
    Returns:
        包含训练历史的字典
    """
    if config is None:
        config = GANConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[GAN] 设备: {device}")

    # ---------- 生成合成数据 (与 VAE 相同) ----------
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.input_dim, device)
    print(f"[GAN] 合成数据: {data.shape}, 范围 [{data.min():.2f}, {data.max():.2f}]")

    # ---------- 初始化模型 ----------
    model = GAN(config).to(device)

    # 使用 Adam, β1=0.5 (GAN 常用设置，比默认 0.9 更稳定)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"[GAN] 生成器参数: {g_params:,} | 判别器参数: {d_params:,} | "
          f"总计: {g_params + d_params:,}")

    # ---------- 训练循环 ----------
    history = {'g_loss': [], 'd_loss': [], 'd_real_acc': [], 'd_fake_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        epoch_d_real_acc, epoch_d_fake_acc = 0.0, 0.0
        num_batches = 0

        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            real_batch = data_shuffled[i:i + batch_size]
            bs = real_batch.size(0)

            # ========== 训练判别器 ==========
            for _ in range(d_steps):
                # 生成假样本
                z = torch.randn(bs, config.latent_dim, device=device)
                fake_batch = model.generator(z).detach()  # detach: 不更新 G

                # 判别器前向
                d_real = model.discriminator(real_batch)
                d_fake = model.discriminator(fake_batch)

                # 计算损失并更新 D
                d_loss = discriminator_loss(d_real, d_fake)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # ========== 训练生成器 ==========
            z = torch.randn(bs, config.latent_dim, device=device)
            fake_batch = model.generator(z)
            d_fake = model.discriminator(fake_batch)

            g_loss = generator_loss(d_fake)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            # 记录统计
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real_acc += (d_real > 0.5).float().mean().item()
            epoch_d_fake_acc += (d_fake < 0.5).float().mean().item()
            num_batches += 1

        # 记录历史
        history['g_loss'].append(epoch_g_loss / num_batches)
        history['d_loss'].append(epoch_d_loss / num_batches)
        history['d_real_acc'].append(epoch_d_real_acc / num_batches)
        history['d_fake_acc'].append(epoch_d_fake_acc / num_batches)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"G_loss: {history['g_loss'][-1]:.3f} | "
                  f"D_loss: {history['d_loss'][-1]:.3f} | "
                  f"D_real: {history['d_real_acc'][-1]:.2%} | "
                  f"D_fake: {history['d_fake_acc'][-1]:.2%}")

    # ---------- 生成样本 ----------
    model.eval()
    generated = model.generate(16, device)
    print(f"[GAN] 生成样本 shape: {generated.shape}, "
          f"范围 [{generated.min():.3f}, {generated.max():.3f}]")

    return {
        'model': model,
        'history': history,
        'generated_samples': generated,
    }


def _generate_structured_data(num_samples: int, dim: int,
                              device: torch.device) -> torch.Tensor:
    """
    生成带结构的合成数据（与 vae.py 中相同）。

    创建 5 种不同的"模式"——每种模式在不同位置有高激活值。
    """
    data = torch.zeros(num_samples, dim, device=device)
    segment = dim // 5

    for i in range(num_samples):
        mode = i % 5
        start = mode * segment
        end = start + segment
        data[i, start:end] = torch.rand(segment, device=device) * 0.8 + 0.2
        data[i] += torch.rand(dim, device=device) * 0.05

    data = data.clamp(0, 1)
    return data


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_gan():
    """GAN 演示: 训练并展示结果。"""
    print("=" * 60)
    print("生成对抗网络 (GAN) 演示")
    print("=" * 60)

    config = GANConfig(
        input_dim=200,    # 与 VAE 演示相同维度
        hidden_dim=128,
        latent_dim=8,
    )

    results = train_gan(config=config, epochs=50, batch_size=64)

    # 展示训练情况
    print("\n--- 训练结果 ---")
    history = results['history']
    print(f"  最终 G 损失: {history['g_loss'][-1]:.3f}")
    print(f"  最终 D 损失: {history['d_loss'][-1]:.3f}")
    print(f"  D 真样本准确率: {history['d_real_acc'][-1]:.2%}")
    print(f"  D 假样本准确率: {history['d_fake_acc'][-1]:.2%}")

    # 理想状态: D 准确率接近 50% (无法区分真假)
    avg_d_acc = (history['d_real_acc'][-1] + history['d_fake_acc'][-1]) / 2
    print(f"\n  D 平均准确率: {avg_d_acc:.2%} (理想值: 50%)")
    if avg_d_acc < 0.7:
        print("  → 生成器已经能较好地欺骗判别器!")
    else:
        print("  → 判别器仍占优势，可能需要更多训练")

    print("\n[GAN 演示完成]")
    return results


if __name__ == '__main__':
    demo_gan()
