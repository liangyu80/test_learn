"""
Shortcut Model —— 从零实现 (One Step Diffusion via Shortcut Models)

核心思想:
    Shortcut Model 是一种单次训练就能实现 1 步生成的方法。
    核心创新: 网络不仅以 (x_t, t) 为条件，还额外输入**期望步长 d**，
    使得网络学会在任意步长下准确跳跃。

    关键洞察:
    - 标准扩散模型学的是"小步走" (速度场 v_θ(x_t, t))
    - Shortcut Model 学的是"跳远" (可以一步跳到终点)
    - 通过**自蒸馏**: 两个小步 = 一个大步

数学框架:
    标准 Flow Matching:
        v_θ(x_t, t)     → 速度场 (无穷小步长)

    Shortcut Model:
        s_θ(x_t, t, d)  → 步长为 d 的跳跃方向

        自一致性约束 (训练目标):
        s_θ(x_t, t, 2d) ≈ s_θ(x_t, t, d) + s_θ(x̂_{t-d}, t-d, d)

        其中 x̂_{t-d} = x_t - d · s_θ(x_t, t, d)

        直觉: 走两步 d 的结果应该和走一步 2d 的结果一致!

训练过程 (自蒸馏):
    阶段 0: 基础 Flow Matching 训练 (d → 0, 学习速度场)
    阶段 1: 自蒸馏
        1. 当前步长 d, 用模型走两步 d 得到结果
        2. 这就是步长 2d 的目标
        3. 训练模型在步长 2d 下匹配这个目标
        4. 步长翻倍: d → 2d → 4d → ... → 1 (一步到位!)

架构:
    ┌─────────────────────────────────┐
    │         Shortcut Network        │
    │   输入: (x_t, t, d)            │
    │                                 │
    │   t → 时间嵌入 ─┐              │
    │   d → 步长嵌入 ─┤→ 融合 → MLP  │
    │   x_t ──────────┘              │
    │                                 │
    │   输出: s_θ(x_t, t, d)         │
    └─────────────────────────────────┘

    采样 (1 步!):
        x_1 ~ N(0, I)
        x_0 = x_1 - 1.0 · s_θ(x_1, 1.0, 1.0)

与其他方法对比:
    ┌──────────────────┬────────────────────────────────────────┐
    │ 方法             │ 实现 1 步生成的方式                     │
    ├──────────────────┼────────────────────────────────────────┤
    │ Consistency Model│ EMA 目标网络 + 一致性损失              │
    │ 蒸馏方法         │ 需要预训练 teacher + 单独蒸馏阶段       │
    │ Shortcut Model   │ 单次训练中自蒸馏，步长渐进翻倍         │
    └──────────────────┴────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class ShortcutConfig:
    """Shortcut Model 配置。"""
    data_dim: int = 200
    hidden_dim: int = 256
    time_emb_dim: int = 64
    step_emb_dim: int = 32        # 步长嵌入维度
    num_blocks: int = 3


# ==============================================================================
# 网络组件
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    """正弦嵌入 (用于时间和步长)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    """带条件的残差块。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.cond_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.silu(self.norm1(x))
        h = self.linear1(h)
        h = h + self.cond_proj(cond)
        h = F.silu(self.norm2(h))
        h = self.linear2(h)
        return h + residual


# ==============================================================================
# Shortcut Network
# ==============================================================================

class ShortcutNetwork(nn.Module):
    """
    Shortcut 网络 s_θ(x_t, t, d)。

    与标准扩散模型的区别: 额外接受步长 d 作为条件。
    当 d → 0 时退化为标准速度场 (Flow Matching)。
    当 d = 1 时可以一步生成。
    """

    def __init__(self, config: ShortcutConfig):
        super().__init__()

        # 时间嵌入
        self.time_emb = SinusoidalEmbedding(config.time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # 步长嵌入 (Shortcut 的核心创新!)
        self.step_emb = SinusoidalEmbedding(config.step_emb_dim)
        self.step_proj = nn.Sequential(
            nn.Linear(config.step_emb_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # 条件融合
        self.cond_merge = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                d: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x_t: 加噪数据, shape (batch, data_dim)
            t:   时间, shape (batch,), 值在 [0, 1]
            d:   步长, shape (batch,), 值在 [0, 1]
        Returns:
            跳跃方向, shape (batch, data_dim)
        """
        # 编码时间和步长
        t_feat = self.time_proj(self.time_emb(t * 1000))
        d_feat = self.step_proj(self.step_emb(d * 1000))

        # 融合条件 (时间 + 步长)
        cond = self.cond_merge(torch.cat([t_feat, d_feat], dim=-1))

        # 主干网络
        h = self.input_proj(x_t) + cond
        for block in self.blocks:
            h = block(h, cond)

        return self.output_proj(h)


# ==============================================================================
# Shortcut Model 完整训练
# ==============================================================================

class ShortcutModel(nn.Module):
    """
    Shortcut Model 完整实现。

    训练分两个阶段:
        阶段 1 (基础): Flow Matching 训练 (d ≈ 0)
            与标准 Flow Matching 相同，学习无穷小速度场

        阶段 2 (自蒸馏): 步长翻倍训练
            对于步长 2d:
            target = s_θ(x_t, t, d) + s_θ(x̂, t-d, d)
            训练 s_θ(x_t, t, 2d) 去匹配 target
    """

    def __init__(self, config: ShortcutConfig):
        super().__init__()
        self.config = config
        self.network = ShortcutNetwork(config)

    def compute_flow_matching_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        阶段 1: 标准 Flow Matching 损失 (d ≈ 0)。

        当 d 很小时，shortcut 退化为速度场:
            s_θ(x_t, t, d≈0) ≈ v_θ(x_t, t) = ε - x_0
        """
        batch_size = x_0.size(0)
        device = x_0.device

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x_0)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * noise

        target_v = noise - x_0  # 标准 Flow Matching 目标

        # d ≈ 0 (使用很小的值)
        d = torch.full((batch_size,), 1e-3, device=device)
        predicted_v = self.network(x_t, t, d)

        return F.mse_loss(predicted_v, target_v)

    def compute_shortcut_loss(self, x_0: torch.Tensor,
                               step_size: float) -> torch.Tensor:
        """
        阶段 2: 自蒸馏损失。

        自一致性: 两个小步 = 一个大步
            s_θ(x_t, t, 2d) ≈ shortcut_from_two_steps

        步骤:
            1. 用当前模型走第一步 d: x̂_{t-d} = x_t - d · s_θ(x_t, t, d)
            2. 用当前模型走第二步 d: s_θ(x̂_{t-d}, t-d, d)
            3. 组合两步的结果作为 target
            4. 训练 s_θ(x_t, t, 2d) 匹配这个 target
        """
        batch_size = x_0.size(0)
        device = x_0.device

        d = step_size
        d2 = 2 * d  # 大步长

        # 采样时间 t (确保 t - 2d >= 0)
        t = torch.rand(batch_size, device=device) * (1.0 - d2) + d2

        # 构造 x_t
        noise = torch.randn_like(x_0)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * noise

        # ---- 用两个小步构造 target (stop gradient) ----
        with torch.no_grad():
            d_tensor = torch.full((batch_size,), d, device=device)

            # 第一步: x_t → x̂_{t-d}
            s1 = self.network(x_t, t, d_tensor)
            x_mid = x_t - d * s1

            # 第二步: x̂_{t-d} → 继续走 d
            t_mid = t - d
            s2 = self.network(x_mid, t_mid, d_tensor)

            # 两步合并的等效方向 (平均)
            target = (s1 + s2) / 2.0

        # ---- 训练大步长匹配两步结果 ----
        d2_tensor = torch.full((batch_size,), d2, device=device)
        predicted = self.network(x_t, t, d2_tensor)

        return F.mse_loss(predicted, target)

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int = 1,
               device: torch.device = None) -> torch.Tensor:
        """
        采样。

        1 步采样:
            x_1 ~ N(0, I)
            x_0 = x_1 - 1.0 · s_θ(x_1, 1.0, 1.0)

        N 步采样:
            步长 d = 1/N
            从 t=1 逐步走到 t=0
        """
        if device is None:
            device = next(self.parameters()).device

        self.network.eval()

        x = torch.randn(num_samples, self.config.data_dim, device=device)
        d = 1.0 / num_steps

        for step in range(num_steps):
            t_val = 1.0 - step * d
            t = torch.full((num_samples,), t_val, device=device)
            d_tensor = torch.full((num_samples,), d, device=device)

            s = self.network(x, t, d_tensor)
            x = x - d * s

        self.network.train()
        return x


# ==============================================================================
# 训练函数
# ==============================================================================

def train_shortcut_model(config: ShortcutConfig = None, epochs_fm: int = 40,
                         epochs_distill: int = 30, batch_size: int = 128,
                         lr: float = 1e-3, device: torch.device = None) -> dict:
    """
    训练 Shortcut Model。

    分两阶段:
        阶段 1: Flow Matching 基础训练
        阶段 2: 自蒸馏 (步长逐渐翻倍)
    """
    if config is None:
        config = ShortcutConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Shortcut Model] 设备: {device}")

    # 数据
    num_samples = 2000
    data = _generate_structured_data(num_samples, config.data_dim, device)
    print(f"[Shortcut Model] 合成数据: {data.shape}")

    # 模型
    model = ShortcutModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Shortcut Model] 参数量: {num_params:,}")

    history = {'fm_loss': [], 'distill_loss': []}

    # ================================================================
    # 阶段 1: Flow Matching 基础训练
    # ================================================================
    print(f"\n▶ 阶段 1: Flow Matching 基础训练 ({epochs_fm} epochs)")

    for epoch in range(epochs_fm):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i + batch_size]
            loss = model.compute_flow_matching_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['fm_loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs_fm} | FM Loss: {avg_loss:.6f}")

    # ================================================================
    # 阶段 2: 自蒸馏 (步长翻倍)
    # ================================================================
    print(f"\n▶ 阶段 2: 自蒸馏 ({epochs_distill} epochs)")

    # 步长从小到大: 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2
    step_sizes = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    epochs_per_step = max(1, epochs_distill // len(step_sizes))

    optimizer_distill = torch.optim.Adam(model.parameters(), lr=lr * 0.5)
    step_idx = 0

    for epoch in range(epochs_distill):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # 选择当前步长
        step_idx = min(epoch // epochs_per_step, len(step_sizes) - 1)
        current_step = step_sizes[step_idx]

        perm = torch.randperm(num_samples, device=device)
        data_shuffled = data[perm]

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i + batch_size]

            # 同时训练 FM 损失和蒸馏损失
            fm_loss = model.compute_flow_matching_loss(batch)
            distill_loss = model.compute_shortcut_loss(batch, current_step)
            loss = fm_loss + distill_loss

            optimizer_distill.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_distill.step()

            epoch_loss += distill_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['distill_loss'].append(avg_loss)

        if (epoch + 1) % max(1, epochs_per_step) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs_distill} | "
                  f"Step: {current_step:.4f} | Distill Loss: {avg_loss:.6f}")

    # ================================================================
    # 采样测试
    # ================================================================
    print("\n[Shortcut Model] 采样测试:")
    model.eval()
    for steps in [1, 2, 4, 8, 16]:
        samples = model.sample(32, num_steps=steps, device=device)
        print(f"  {steps:2d} 步: 范围 [{samples.min():.3f}, {samples.max():.3f}], "
              f"std={samples.std():.3f}")

    return {'model': model, 'history': history}


def _generate_structured_data(num_samples: int, dim: int,
                              device: torch.device) -> torch.Tensor:
    """生成合成数据 ([-1, 1])。"""
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

def demo_shortcut_model():
    """Shortcut Model 演示。"""
    print("=" * 60)
    print("Shortcut Model 演示 (One Step Diffusion)")
    print("=" * 60)

    config = ShortcutConfig(data_dim=200, hidden_dim=128, time_emb_dim=32)
    results = train_shortcut_model(config=config, epochs_fm=50,
                                   epochs_distill=35, batch_size=64)

    model = results['model']
    device = next(model.parameters()).device

    # 对比不同步数的质量
    import time
    print("\n--- 采样速度与质量 ---")
    real_data = _generate_structured_data(500, config.data_dim, device)

    for steps in [1, 2, 4, 8, 16, 32]:
        t0 = time.time()
        samples = model.sample(200, num_steps=steps, device=device)
        elapsed = time.time() - t0

        mean_err = abs(samples.mean().item() - real_data.mean().item())
        std_err = abs(samples.std().item() - real_data.std().item())
        print(f"  {steps:2d} 步: mean_err={mean_err:.4f}, std_err={std_err:.4f}, "
              f"耗时={elapsed:.3f}s")

    print("\n  → Shortcut Model 的核心优势: 单次训练实现 1 步生成!")
    print("\n[Shortcut Model 演示完成]")
    return results


if __name__ == '__main__':
    demo_shortcut_model()
