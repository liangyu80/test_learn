"""
VAE vs GAN 对比实验

本脚本在相同条件下训练 VAE 和 GAN，然后从多个维度对比:
    1. 训练稳定性 —— 损失曲线的波动程度
    2. 生成质量   —— 通过统计指标衡量
    3. 模式覆盖   —— 是否覆盖了所有数据模式
    4. 潜在空间   —— VAE 独有的插值能力
    5. 训练效率   —— 参数量和训练时间

理论对比:
    ┌──────────────┬─────────────────────┬─────────────────────┐
    │     维度     │        VAE          │        GAN          │
    ├──────────────┼─────────────────────┼─────────────────────┤
    │ 训练目标     │ 最大化 ELBO         │ 极小极大博弈        │
    │ 训练稳定性   │ ✅ 稳定             │ ❌ 不稳定           │
    │ 生成清晰度   │ ❌ 较模糊           │ ✅ 更清晰           │
    │ 模式崩塌     │ ✅ 不容易           │ ❌ 容易发生         │
    │ 似然估计     │ ✅ 可以计算         │ ❌ 无法直接计算     │
    │ 潜在空间     │ ✅ 连续有结构       │ ❌ 无显式结构       │
    │ 典型应用     │ 数据压缩/异常检测   │ 图像生成/超分辨率   │
    └──────────────┴─────────────────────┴─────────────────────┘
"""

import torch
import time
import math
from vae import VAE, VAEConfig, train_vae, vae_loss, _generate_structured_data as gen_data_vae
from gan import GAN, GANConfig, train_gan, _generate_structured_data as gen_data_gan


def compare_models():
    """在相同条件下训练并对比 VAE 和 GAN。"""
    print("=" * 70)
    print("  VAE vs GAN 对比实验")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 统一配置
    input_dim = 200
    hidden_dim = 128
    latent_dim = 8
    epochs = 30
    batch_size = 64

    # ====================================================================
    # 1. 训练 VAE
    # ====================================================================
    print("▶ 训练 VAE...")
    vae_config = VAEConfig(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    t0 = time.time()
    vae_results = train_vae(config=vae_config, epochs=epochs,
                            batch_size=batch_size, device=device)
    vae_time = time.time() - t0
    print(f"  VAE 训练耗时: {vae_time:.2f}s\n")

    # ====================================================================
    # 2. 训练 GAN
    # ====================================================================
    print("▶ 训练 GAN...")
    gan_config = GANConfig(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    t0 = time.time()
    gan_results = train_gan(config=gan_config, epochs=epochs,
                            batch_size=batch_size, device=device)
    gan_time = time.time() - t0
    print(f"  GAN 训练耗时: {gan_time:.2f}s\n")

    # ====================================================================
    # 3. 对比分析
    # ====================================================================
    print("=" * 70)
    print("  对比分析")
    print("=" * 70)

    # --- 3.1 参数量对比 ---
    print("\n📊 参数量对比:")
    vae_model = vae_results['model']
    gan_model = gan_results['model']

    vae_params = sum(p.numel() for p in vae_model.parameters())
    gan_params = sum(p.numel() for p in gan_model.parameters())
    print(f"  VAE: {vae_params:,} 参数 (编码器 + 解码器)")
    print(f"  GAN: {gan_params:,} 参数 (生成器 + 判别器)")
    print(f"  差异: GAN 多 {gan_params - vae_params:,} 参数 "
          f"(因为判别器比编码器输出层更简单，但输入层更大)")

    # --- 3.2 训练稳定性 ---
    print("\n📊 训练稳定性:")
    vae_losses = vae_results['history']['total_loss']
    gan_g_losses = gan_results['history']['g_loss']
    gan_d_losses = gan_results['history']['d_loss']

    vae_var = _compute_variance(vae_losses)
    gan_g_var = _compute_variance(gan_g_losses)
    gan_d_var = _compute_variance(gan_d_losses)

    print(f"  VAE 损失方差: {vae_var:.4f}")
    print(f"  GAN G损失方差: {gan_g_var:.4f}")
    print(f"  GAN D损失方差: {gan_d_var:.4f}")
    if vae_var < gan_g_var:
        print("  → VAE 训练更稳定 (预期中，因为 VAE 没有对抗训练)")
    else:
        print("  → GAN 训练也挺稳定")

    # --- 3.3 生成质量: 统计特征匹配 ---
    print("\n📊 生成质量 (统计特征匹配):")
    real_data = gen_data_vae(500, input_dim, device)

    vae_samples = vae_model.generate(500, device)
    gan_samples = gan_model.generate(500, device)

    # 计算均值和标准差的匹配程度
    real_mean, real_std = real_data.mean().item(), real_data.std().item()
    vae_mean, vae_std = vae_samples.mean().item(), vae_samples.std().item()
    gan_mean, gan_std = gan_samples.mean().item(), gan_samples.std().item()

    print(f"  真实数据: mean={real_mean:.4f}, std={real_std:.4f}")
    print(f"  VAE 生成: mean={vae_mean:.4f}, std={vae_std:.4f}")
    print(f"  GAN 生成: mean={gan_mean:.4f}, std={gan_std:.4f}")

    vae_mean_err = abs(vae_mean - real_mean)
    gan_mean_err = abs(gan_mean - real_mean)
    vae_std_err = abs(vae_std - real_std)
    gan_std_err = abs(gan_std - real_std)
    print(f"\n  均值误差: VAE={vae_mean_err:.4f}, GAN={gan_mean_err:.4f}")
    print(f"  标准差误差: VAE={vae_std_err:.4f}, GAN={gan_std_err:.4f}")

    # --- 3.4 模式覆盖 ---
    print("\n📊 模式覆盖 (Mode Coverage):")
    vae_modes = _check_mode_coverage(vae_samples, input_dim)
    gan_modes = _check_mode_coverage(gan_samples, input_dim)

    print(f"  数据模式总数: 5")
    print(f"  VAE 覆盖模式: {vae_modes}/5")
    print(f"  GAN 覆盖模式: {gan_modes}/5")
    if gan_modes < vae_modes:
        print("  → GAN 可能出现了模式崩塌 (mode collapse)!")
    elif gan_modes == vae_modes:
        print("  → 两者模式覆盖相当")
    else:
        print("  → GAN 模式覆盖更好")

    # --- 3.5 VAE 独特能力: 潜在空间插值 ---
    print("\n📊 潜在空间插值 (VAE 独特能力):")
    x1 = gen_data_vae(1, input_dim, device)
    x2 = torch.zeros(1, input_dim, device=device)
    seg = input_dim // 5
    x2[0, 3*seg:4*seg] = torch.rand(seg, device=device) * 0.8 + 0.2

    interp = vae_model.interpolate(x1, x2, steps=5)
    # 检查插值是否平滑: 相邻样本之间的 L2 距离应该大致相等
    distances = []
    for j in range(len(interp) - 1):
        d = (interp[j] - interp[j+1]).pow(2).sum().sqrt().item()
        distances.append(d)

    dist_var = _compute_variance(distances)
    print(f"  插值步间距离: {['%.3f' % d for d in distances]}")
    print(f"  距离方差: {dist_var:.4f} (越小越平滑)")
    print("  → VAE 的潜在空间允许有意义的插值，GAN 没有这个能力")

    # --- 3.6 训练效率 ---
    print("\n📊 训练效率:")
    print(f"  VAE 训练时间: {vae_time:.2f}s")
    print(f"  GAN 训练时间: {gan_time:.2f}s")
    if vae_time < gan_time:
        print(f"  → VAE 快 {gan_time/vae_time:.1f}x (GAN 需要交替训练 D 和 G)")
    else:
        print(f"  → GAN 快 {vae_time/gan_time:.1f}x")

    # ====================================================================
    # 4. 总结
    # ====================================================================
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print("""
    VAE 优势:
      ✅ 训练稳定，损失函数有明确的理论依据 (ELBO)
      ✅ 不存在模式崩塌问题，能覆盖数据的所有模式
      ✅ 潜在空间连续有结构，支持插值和属性编辑
      ✅ 可以计算近似似然，用于异常检测
      ✅ 训练更快（无需交替训练两个网络）

    GAN 优势:
      ✅ 生成样本通常更清晰锐利
      ✅ 不受重构损失的模糊化影响
      ✅ 理论上可以学习任意复杂的分布
      ✅ 在图像生成领域表现更优 (如 StyleGAN, BigGAN)

    选择建议:
      - 需要稳定训练和潜在空间结构 → VAE
      - 需要高质量图像生成 → GAN
      - 两者都要 → VAE-GAN 混合模型
    """)


def _compute_variance(values: list) -> float:
    """计算方差。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _check_mode_coverage(samples: torch.Tensor, dim: int) -> int:
    """
    检查生成样本覆盖了多少种数据模式。

    每种模式对应数据的不同区域有较高的激活值。
    如果某个区域的平均激活值高于阈值，认为该模式被覆盖。
    """
    segment = dim // 5
    covered = 0
    threshold = 0.15  # 激活阈值

    for mode in range(5):
        start = mode * segment
        end = start + segment
        # 检查生成样本在该区域的平均激活值
        region_mean = samples[:, start:end].mean().item()
        if region_mean > threshold:
            covered += 1

    return covered


if __name__ == '__main__':
    compare_models()
