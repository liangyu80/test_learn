"""
去噪扩散隐式模型 (Denoising Diffusion Implicit Model, DDIM) —— 加速采样

核心思想:
    DDPM 采样需要 T 步 (如 1000 步)，非常慢。
    DDIM 发现可以将反向过程重新参数化为一个非马尔可夫过程，
    从而在子序列的时间步上采样，大幅减少步数。

    例如: T=1000 的 DDPM 采样需要 1000 步
         DDIM 可以只用 50 步甚至 10 步完成采样!

数学推导:
    DDIM 定义了一个更一般的采样公式:

    x_{t-1} = √ᾱ_{t-1} × x̂_0(x_t, t)
            + √(1-ᾱ_{t-1}-σ²_t) × ε_θ(x_t, t)
            + σ_t × ε

    其中:
    - x̂_0(x_t, t) = (x_t - √(1-ᾱ_t) × ε_θ(x_t, t)) / √ᾱ_t  (预测的 x_0)
    - σ_t 控制随机性:
        σ_t = 0:     完全确定性采样 (纯 DDIM)
        σ_t = σ_DDPM: 等价于 DDPM 采样

    DDIM 的关键洞察:
    这个公式对任意子序列 {τ_1, τ_2, ..., τ_S} 都成立 (S << T)!

DDPM vs DDIM:
    ┌──────────────────┬───────────────────┬──────────────────┐
    │      特性        │      DDPM         │      DDIM        │
    ├──────────────────┼───────────────────┼──────────────────┤
    │ 采样步数         │ T (如 1000)       │ S << T (如 50)   │
    │ 采样过程         │ 随机 (马尔可夫)   │ 可确定性         │
    │ 训练方式         │ 相同              │ 相同 (共用模型!) │
    │ 生成质量         │ 高                │ 少步时略低       │
    │ 采样速度         │ 慢                │ 快 10-100×       │
    │ 可复现性         │ 不可 (随机)       │ 可 (σ=0时确定)   │
    └──────────────────┴───────────────────┴──────────────────┘
"""

import torch
import torch.nn as nn
from typing import Optional, List
from ddpm import DDPM, DDPMConfig, NoiseSchedule


# ==============================================================================
# DDIM 采样器
# ==============================================================================

class DDIMSampler:
    """
    DDIM 采样器: 使用预训练的 DDPM 模型进行加速采样。

    重要: DDIM 不需要重新训练! 直接复用 DDPM 训练的噪声预测网络。
    """

    def __init__(self, ddpm_model: DDPM):
        """
        Args:
            ddpm_model: 已训练的 DDPM 模型
        """
        self.model = ddpm_model
        self.config = ddpm_model.config
        self.ns = ddpm_model.noise_schedule
        self.device = ddpm_model.device

    def _make_timestep_subsequence(self, num_steps: int) -> List[int]:
        """
        创建采样时间步子序列。

        从 [0, T-1] 中均匀选取 num_steps 个时间步。
        例如 T=100, num_steps=10 → [0, 10, 20, ..., 90]

        Args:
            num_steps: 采样步数
        Returns:
            时间步子序列 (从大到小排列)
        """
        T = self.config.num_timesteps
        # 均匀选取
        step_size = T // num_steps
        timesteps = list(range(0, T, step_size))[:num_steps]
        # 反转: 从大到小 (采样从噪声到数据)
        timesteps = list(reversed(timesteps))
        return timesteps

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int = 50,
               eta: float = 0.0, return_trajectory: bool = False) -> torch.Tensor:
        """
        DDIM 加速采样。

        Args:
            num_samples: 生成样本数
            num_steps:   采样步数 (远小于训练时的 T)
            eta:         随机性控制参数
                         η=0: 完全确定性 (纯 DDIM)
                         η=1: 等价于 DDPM
            return_trajectory: 是否返回轨迹
        Returns:
            生成的样本
        """
        self.model.noise_predictor.eval()

        # 获取采样子序列
        timesteps = self._make_timestep_subsequence(num_steps)

        # 从纯噪声开始
        x_t = torch.randn(num_samples, self.config.data_dim, device=self.device)
        trajectory = [x_t.clone()] if return_trajectory else None

        for i, t_val in enumerate(timesteps):
            t = torch.full((num_samples,), t_val, device=self.device, dtype=torch.long)

            # 预测噪声
            eps_pred = self.model.noise_predictor(x_t, t)

            # 当前时刻的 ᾱ_t
            alpha_bar_t = self.ns.alphas_bar[t_val]

            # 下一时刻的 ᾱ_{t-1} (最后一步为 1.0，即无噪声)
            if i + 1 < len(timesteps):
                alpha_bar_prev = self.ns.alphas_bar[timesteps[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # ---- DDIM 核心公式 ----

            # 1. 预测 x_0:
            #    x̂_0 = (x_t - √(1-ᾱ_t) × ε̂) / √ᾱ_t
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

            # 2. 计算 σ_t (控制随机性)
            #    σ_t = η × √((1-ᾱ_{t-1})/(1-ᾱ_t)) × √(1 - ᾱ_t/ᾱ_{t-1})
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

            # 3. "方向指向 x_t" 的分量
            #    dir = √(1 - ᾱ_{t-1} - σ²_t) × ε̂
            dir_xt = torch.sqrt(
                torch.clamp(1 - alpha_bar_prev - sigma_t ** 2, min=0)
            ) * eps_pred

            # 4. 最终采样
            #    x_{t-1} = √ᾱ_{t-1} × x̂_0 + dir + σ_t × z
            noise = torch.randn_like(x_t) if t_val > 0 else torch.zeros_like(x_t)
            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

            if return_trajectory:
                trajectory.append(x_t.clone())

        self.model.noise_predictor.train()

        if return_trajectory:
            return x_t, trajectory
        return x_t


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_ddim():
    """DDIM 演示: 使用 DDPM 模型进行加速采样。"""
    import time
    from ddpm import train_ddpm

    print("=" * 60)
    print("DDIM 加速采样演示")
    print("=" * 60)

    # 先训练一个 DDPM 模型
    config = DDPMConfig(
        data_dim=200,
        hidden_dim=128,
        time_emb_dim=32,
        num_timesteps=100,
        beta_schedule='linear',
    )

    print("\n▶ 第一步: 训练 DDPM 模型...")
    results = train_ddpm(config=config, epochs=50, batch_size=64)
    model = results['model']
    model.eval()

    # 创建 DDIM 采样器
    sampler = DDIMSampler(model)

    # ---- 对比不同采样步数 ----
    print("\n" + "=" * 60)
    print("  DDPM vs DDIM 采样对比")
    print("=" * 60)

    num_samples = 32

    # DDPM 采样 (全步)
    t0 = time.time()
    ddpm_samples = model.sample(num_samples)
    ddpm_time = time.time() - t0

    print(f"\n📊 DDPM 采样 ({config.num_timesteps} 步):")
    print(f"  耗时: {ddpm_time:.3f}s")
    print(f"  范围: [{ddpm_samples.min():.3f}, {ddpm_samples.max():.3f}]")
    print(f"  std:  {ddpm_samples.std():.3f}")

    # DDIM 不同步数
    for num_steps in [50, 20, 10, 5]:
        t0 = time.time()
        ddim_samples = sampler.sample(num_samples, num_steps=num_steps, eta=0.0)
        ddim_time = time.time() - t0

        speedup = ddpm_time / ddim_time if ddim_time > 0 else float('inf')

        print(f"\n📊 DDIM 采样 ({num_steps} 步, η=0):")
        print(f"  耗时: {ddim_time:.3f}s (加速 {speedup:.1f}x)")
        print(f"  范围: [{ddim_samples.min():.3f}, {ddim_samples.max():.3f}]")
        print(f"  std:  {ddim_samples.std():.3f}")

    # ---- 对比不同 eta ----
    print(f"\n\n📊 不同 η 值的影响 (20 步采样):")
    for eta in [0.0, 0.5, 1.0]:
        # 采样两次，检查结果是否一致 (η=0 应该一致)
        torch.manual_seed(42)
        s1 = sampler.sample(8, num_steps=20, eta=eta)
        torch.manual_seed(42)
        s2 = sampler.sample(8, num_steps=20, eta=eta)

        diff = (s1 - s2).abs().max().item()
        label = "确定性" if eta == 0 else "半随机" if eta == 0.5 else "全随机(≈DDPM)"
        print(f"  η={eta:.1f} ({label}): "
              f"两次采样最大差异={diff:.6f} "
              f"{'✅ 可复现' if diff < 1e-5 else '(有随机性)'}")

    print("\n[DDIM 演示完成]")


if __name__ == '__main__':
    demo_ddim()
