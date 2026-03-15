"""
扩散模型训练与对比实验

本脚本完成三件事:
    1. 训练 DDPM 模型
    2. 对比 DDPM 和 DDIM 采样 (质量 vs 速度)
    3. 对比不同 β 调度策略 (linear vs cosine)
    4. 与 VAE/GAN 进行横向对比

实验设计:
    所有实验使用相同的合成数据 (5 种模式)，
    与 vae_vs_gan/ 项目保持一致，便于横向对比。
"""

import torch
import time
import sys
import os

# 添加父目录到路径，以便导入 vae_vs_gan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ddpm import DDPM, DDPMConfig, train_ddpm, _generate_structured_data
from ddim import DDIMSampler


def run_experiments():
    """运行完整的对比实验。"""
    print("=" * 70)
    print("  扩散模型 (Diffusion Model) 训练与对比实验")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    data_dim = 200

    # ====================================================================
    # 实验 1: DDPM 训练 (Linear β)
    # ====================================================================
    print("▶ 实验 1: DDPM 训练 (Linear β 调度)")
    print("-" * 50)

    config_linear = DDPMConfig(
        data_dim=data_dim,
        hidden_dim=128,
        time_emb_dim=32,
        num_timesteps=100,
        beta_schedule='linear',
    )

    t0 = time.time()
    results_linear = train_ddpm(config=config_linear, epochs=50,
                                batch_size=64, device=device)
    time_linear = time.time() - t0
    print(f"  训练耗时: {time_linear:.2f}s\n")

    # ====================================================================
    # 实验 2: DDPM 训练 (Cosine β)
    # ====================================================================
    print("▶ 实验 2: DDPM 训练 (Cosine β 调度)")
    print("-" * 50)

    config_cosine = DDPMConfig(
        data_dim=data_dim,
        hidden_dim=128,
        time_emb_dim=32,
        num_timesteps=100,
        beta_schedule='cosine',
    )

    t0 = time.time()
    results_cosine = train_ddpm(config=config_cosine, epochs=50,
                                batch_size=64, device=device)
    time_cosine = time.time() - t0
    print(f"  训练耗时: {time_cosine:.2f}s\n")

    # ====================================================================
    # 实验 3: DDPM vs DDIM 采样对比
    # ====================================================================
    print("=" * 70)
    print("  实验 3: DDPM vs DDIM 采样速度与质量对比")
    print("=" * 70)

    model = results_linear['model']
    model.eval()
    sampler = DDIMSampler(model)

    # 参考数据
    real_data = _generate_structured_data(500, data_dim, device)
    real_mean = real_data.mean().item()
    real_std = real_data.std().item()

    num_eval = 200
    print(f"\n  真实数据统计: mean={real_mean:.4f}, std={real_std:.4f}\n")

    # DDPM 采样
    t0 = time.time()
    ddpm_samples = model.sample(num_eval)
    ddpm_time = time.time() - t0

    # DDIM 不同步数
    results_table = [("DDPM (100步)", ddpm_time, ddpm_samples)]
    for steps in [50, 20, 10]:
        t0 = time.time()
        ddim_s = sampler.sample(num_eval, num_steps=steps, eta=0.0)
        t = time.time() - t0
        results_table.append((f"DDIM ({steps}步)", t, ddim_s))

    print(f"  {'方法':<16} {'耗时':>8} {'加速比':>8} "
          f"{'mean误差':>10} {'std误差':>10} {'模式覆盖':>8}")
    print("  " + "-" * 68)

    base_time = results_table[0][1]
    for name, t, samples in results_table:
        mean_err = abs(samples.mean().item() - real_mean)
        std_err = abs(samples.std().item() - real_std)
        modes = _check_mode_coverage(samples, data_dim)
        speedup = base_time / t if t > 0 else float('inf')
        print(f"  {name:<16} {t:>7.3f}s {speedup:>7.1f}x "
              f"{mean_err:>10.4f} {std_err:>10.4f} {modes:>6}/5")

    # ====================================================================
    # 实验 4: Linear vs Cosine β 调度对比
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("  实验 4: Linear vs Cosine β 调度对比")
    print("=" * 70)

    model_linear = results_linear['model']
    model_cosine = results_cosine['model']
    model_linear.eval()
    model_cosine.eval()

    # 比较训练损失
    loss_linear = results_linear['history']['loss']
    loss_cosine = results_cosine['history']['loss']

    print(f"\n  训练损失对比:")
    print(f"    Linear - 初始: {loss_linear[0]:.6f}, 最终: {loss_linear[-1]:.6f}")
    print(f"    Cosine - 初始: {loss_cosine[0]:.6f}, 最终: {loss_cosine[-1]:.6f}")

    # 比较生成质量
    samples_linear = model_linear.sample(num_eval)
    samples_cosine = model_cosine.sample(num_eval)

    modes_linear = _check_mode_coverage(samples_linear, data_dim)
    modes_cosine = _check_mode_coverage(samples_cosine, data_dim)

    print(f"\n  生成质量对比:")
    print(f"    Linear: mean={samples_linear.mean():.4f}, "
          f"std={samples_linear.std():.4f}, 模式覆盖={modes_linear}/5")
    print(f"    Cosine: mean={samples_cosine.mean():.4f}, "
          f"std={samples_cosine.std():.4f}, 模式覆盖={modes_cosine}/5")

    # 比较 ᾱ_t 的衰减曲线
    print(f"\n  ᾱ_t 衰减对比 (信号保留比例):")
    ns_l = model_linear.noise_schedule
    ns_c = model_cosine.noise_schedule
    for t in [0, 25, 50, 75, 99]:
        print(f"    t={t:3d}: Linear ᾱ_t={ns_l.alphas_bar[t]:.4f}, "
              f"Cosine ᾱ_t={ns_c.alphas_bar[t]:.4f}")
    print("    → Cosine 调度在后期保留更多信号，避免信息过早丢失")

    # ====================================================================
    # 实验 5: 与 VAE/GAN 横向对比
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("  实验 5: Diffusion vs VAE vs GAN 横向对比")
    print("=" * 70)

    # 尝试导入 VAE/GAN
    try:
        from vae_vs_gan.vae import train_vae, VAEConfig
        from vae_vs_gan.gan import train_gan, GANConfig

        print("\n▶ 训练 VAE...")
        vae_config = VAEConfig(input_dim=data_dim, hidden_dim=128, latent_dim=8)
        t0 = time.time()
        vae_results = train_vae(config=vae_config, epochs=30, batch_size=64, device=device)
        vae_time = time.time() - t0

        print("\n▶ 训练 GAN...")
        gan_config = GANConfig(input_dim=data_dim, hidden_dim=128, latent_dim=8)
        t0 = time.time()
        gan_results = train_gan(config=gan_config, epochs=30, batch_size=64, device=device)
        gan_time = time.time() - t0

        vae_samples = vae_results['model'].generate(num_eval, device)
        gan_samples = gan_results['model'].generate(num_eval, device)
        # GAN/VAE 数据在 [0,1]，转换到 [-1,1] 进行对比
        vae_samples_centered = vae_samples * 2 - 1
        gan_samples_centered = gan_samples * 2 - 1

        vae_params = sum(p.numel() for p in vae_results['model'].parameters())
        gan_params = sum(p.numel() for p in gan_results['model'].parameters())
        diff_params = sum(p.numel() for p in model_linear.noise_predictor.parameters())

        print(f"\n  {'指标':<16} {'Diffusion':>12} {'VAE':>12} {'GAN':>12}")
        print("  " + "-" * 54)
        print(f"  {'参数量':<16} {diff_params:>12,} {vae_params:>12,} {gan_params:>12,}")
        print(f"  {'训练时间(s)':<16} {time_linear:>12.2f} {vae_time:>12.2f} {gan_time:>12.2f}")
        print(f"  {'mean误差':<16} "
              f"{abs(ddpm_samples.mean().item() - real_mean):>12.4f} "
              f"{abs(vae_samples_centered.mean().item() - real_mean):>12.4f} "
              f"{abs(gan_samples_centered.mean().item() - real_mean):>12.4f}")
        print(f"  {'std误差':<16} "
              f"{abs(ddpm_samples.std().item() - real_std):>12.4f} "
              f"{abs(vae_samples_centered.std().item() - real_std):>12.4f} "
              f"{abs(gan_samples_centered.std().item() - real_std):>12.4f}")

        modes_diff = _check_mode_coverage(ddpm_samples, data_dim, centered=True)
        modes_vae = _check_mode_coverage(vae_samples_centered, data_dim, centered=True)
        modes_gan = _check_mode_coverage(gan_samples_centered, data_dim, centered=True)
        print(f"  {'模式覆盖':<16} {modes_diff:>10}/5 {modes_vae:>10}/5 {modes_gan:>10}/5")

    except ImportError:
        print("\n  [跳过] 未找到 vae_vs_gan 模块，跳过横向对比")
        print("  提示: 先运行 vae_vs_gan 项目或确保其在父目录中")

    # ====================================================================
    # 总结
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("  总结")
    print("=" * 70)
    print("""
    扩散模型 (Diffusion) 特点:
      ✅ 训练极其稳定 (简单 MSE 损失，无对抗训练)
      ✅ 生成质量高 (Stable Diffusion, DALL-E 2 等均基于此)
      ✅ 理论基础坚实 (SDE/ODE, 得分匹配)
      ✅ 不存在模式崩塌问题
      ❌ 采样速度慢 (需要多步迭代去噪)
      → DDIM 可加速 10-100x，质量仅略有下降

    β 调度策略:
      - Linear: 简单直接，噪声均匀增长
      - Cosine: 后期保留更多信号，通常生成质量更好

    三种生成模型对比:
      ┌──────────┬────────────┬────────────┬────────────┐
      │          │ Diffusion  │    VAE     │    GAN     │
      ├──────────┼────────────┼────────────┼────────────┤
      │ 训练稳定 │  ✅ 最稳定  │  ✅ 稳定   │  ❌ 不稳定 │
      │ 生成质量 │  ✅ 最高    │  ❌ 模糊   │  ✅ 清晰   │
      │ 采样速度 │  ❌ 最慢    │  ✅ 最快   │  ✅ 快     │
      │ 模式崩塌 │  ✅ 无      │  ✅ 无     │  ❌ 有     │
      │ 潜在空间 │  ❌ 无结构  │  ✅ 有结构 │  ❌ 无结构 │
      │ 代表应用 │ Stable Diff │  β-VAE    │ StyleGAN   │
      └──────────┴────────────┴────────────┴────────────┘
    """)


def _check_mode_coverage(samples: torch.Tensor, dim: int,
                          centered: bool = True) -> int:
    """
    检查生成样本的模式覆盖。

    Args:
        samples:  生成样本
        dim:      数据维度
        centered: 数据是否在 [-1,1] 范围
    """
    segment = dim // 5
    covered = 0
    threshold = -0.3 if centered else 0.15

    for mode in range(5):
        start = mode * segment
        end = start + segment
        region_mean = samples[:, start:end].mean().item()
        if region_mean > threshold:
            covered += 1

    return covered


if __name__ == '__main__':
    run_experiments()
