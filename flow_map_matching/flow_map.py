"""
Flow Map Matching (流映射匹配) —— 从零实现

来源论文:
    "Flow Map Matching" (Boffi, Albergo, Vanden-Eijnden, 2024)
    arXiv: 2406.07507

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心思想 (一句话):
    经典 Flow Matching 学习"瞬时速度场" v_t(x)，生成时需要用 ODE 求解器
    一步一步积分 (几十~上百次网络前向, NFE 很高)。
    Flow Map Matching 直接学习"两时刻传输映射" X_{s,t}(x)——概率流 ODE 的
    "解算子": 把 s 时刻的样本一步跳到 t 时刻。于是可以 1 步 / 少步生成。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 随机插值 (Stochastic Interpolant) —— 定义前向路径
   ────────────────────────────────────────────────
   本实现采用线性插值 (与仓库 diffusion_advanced/flow_matching.py 一致):
       x_t = (1 - t)·x_0 + t·ε,    t ∈ [0, 1],  ε ~ N(0, I)
       约定: t=0 是数据, t=1 是噪声
   条件速度 (沿单条路径, 是常数!):
       dx_t/dt = ε - x_0

   边缘速度场 (Flow Matching 学习的目标):
       v_t(x) = E[ ε - x_0 | x_t = x ]
   概率流 ODE:
       dX/dt = v_t(X)

2. 流映射 (Flow Map) 的定义
   ─────────────────────────
   流映射 X_{s,t} 是上面 ODE 的"解算子": 给定 s 时刻的点 x, 沿 ODE 演化到 t 时刻:
       d/dt X_{s,t}(x) = v_t( X_{s,t}(x) ),     X_{s,s}(x) = x    (初值=恒等)
   它满足两条关键性质:
       (a) 恒等边界:      X_{s,s}(x) = x
       (b) 半群 (semigroup): X_{t,u} ∘ X_{s,t} = X_{s,u}     ← 少步生成的理论依据
   生成样本 = X_{1,0}(噪声)  (从 t=1 一步跳到 t=0)

3. 参数化 (自动满足恒等边界)
   ─────────────────────────
       X_{s,t}(x) = x + (t - s)·g_φ(x, s, t)
   当 t → s 时, (t-s) → 0, 自动得到 X_{s,s}(x)=x。
   g_φ 可理解为"从 s 到 t 的平均速度"。
   (经典 Flow Matching 的 v_t(x) 是 g_φ 在 t→s 时的极限: v_s(x)=g_φ(x,s,s)。)

4. 训练目标: Lagrangian Flow Map Matching (LFMM, 拉格朗日形式)
   ──────────────────────────────────────────────────────────
   直接利用性质 d/dt X_{s,t}(x) = v_t( X_{s,t}(x) ):
       L_LFMM = E_{s,t,x_s} ‖ ∂_t X_{s,t}(x_s) - v_t( X_{s,t}(x_s) ) ‖²
   其中:
       - x_s 从 s 时刻的边缘分布采样 (即插值 x_s=(1-s)x_0+s·ε)
       - v_t 是一个已训练好的 (冻结的) 速度场 —— 因此这是一种"蒸馏"
       - ∂_t X_{s,t} 用前向模式自动微分 (JVP) 精确计算, 无需模拟 ODE
   注意: 无需在时间上"展开/rollout", 每步只需一次 JVP, 训练稳定高效。

   论文还给出 Eulerian (欧拉形式) 及 distillation-free (免蒸馏) 变体,
   本文件以最清晰的 LFMM 蒸馏形式演示核心机制 (见 README)。

5. 采样 (利用半群性质做少步生成)
   ───────────────────────────────
   选时间网格 1 = τ_0 > τ_1 > ... > τ_k = 0, 迭代:
       x ← X_{τ_i, τ_{i+1}}(x)
   k=1 即"一步生成": x = X_{1,0}(噪声)。

与经典 Flow Matching / Consistency Model 的关系:
    ┌────────────────┬──────────────────┬─────────────────────┬────────────────────┐
    │                │ Flow Matching    │ Consistency Model   │ Flow Map Matching  │
    ├────────────────┼──────────────────┼─────────────────────┼────────────────────┤
    │ 学习对象       │ 瞬时速度 v_t(x)  │ 单时刻映射 X_{t,0}  │ 两时刻映射 X_{s,t} │
    │ 生成 NFE       │ 高 (多步 ODE)    │ 1~2 步              │ 1~少步 (可调)      │
    │ 是否含前者     │ —                │ FMM 的特例(s→固定端)│ 一般化 (含二者)    │
    └────────────────┴──────────────────┴─────────────────────┴────────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ==============================================================================
# 玩具数据: 2D 八高斯环 (便于可视化 + 量化对比)
# ==============================================================================

def sample_eight_gaussians(n: int, device: torch.device,
                           radius: float = 4.0, std: float = 0.25) -> torch.Tensor:
    """
    从"八高斯环"分布采样 (2D 生成模型经典 toy)。

    8 个高斯的中心均匀分布在半径为 radius 的圆上, 每个高斯标准差为 std。
    多模态 + 有明确结构, 非常适合检验生成模型是否学到了整个分布 (而非塌缩)。
    """
    centers = torch.tensor(
        [[radius * math.cos(2 * math.pi * k / 8),
          radius * math.sin(2 * math.pi * k / 8)] for k in range(8)],
        device=device, dtype=torch.float32,
    )
    idx = torch.randint(0, 8, (n,), device=device)
    return centers[idx] + std * torch.randn(n, 2, device=device)


# ==============================================================================
# 时间嵌入 (傅里叶特征): 把标量 t∈[0,1] 映射为高维向量
# ==============================================================================

class FourierTimeEmbedding(nn.Module):
    """
    随机傅里叶时间嵌入。

    对标量时间 t, 输出 [sin(2π f_i t), cos(2π f_i t)]_i。
    高频特征让 MLP 能更好地区分不同时间, 提升对 t 的敏感度。
    """

    def __init__(self, dim: int = 64, scale: float = 10.0):
        super().__init__()
        assert dim % 2 == 0
        # 固定 (不训练) 的随机频率, 保证前向模式自动微分 (JVP) 干净
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, 1) -> (B, dim)"""
        proj = 2 * math.pi * t * self.freqs.unsqueeze(0)  # (B, dim/2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ==============================================================================
# 经典 Flow Matching: 速度场网络 v_θ(x, t)
# ==============================================================================

class VelocityNet(nn.Module):
    """
    速度场网络 v_θ(x, t) —— 经典 Flow Matching 学习的对象。

    输入: 状态 x (2D) + 时间 t
    输出: 瞬时速度 v (2D)
    """

    def __init__(self, data_dim: int = 2, hidden: int = 128,
                 time_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.time_emb = FourierTimeEmbedding(time_dim)
        layers = [nn.Linear(data_dim + time_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, D), t: (B, 1) -> (B, D)"""
        te = self.time_emb(t)
        return self.net(torch.cat([x, te], dim=-1))


def flow_matching_loss(model: VelocityNet, x0: torch.Tensor) -> torch.Tensor:
    """
    经典 Flow Matching 损失 (条件流匹配 / Rectified Flow):

        x_t = (1-t)·x_0 + t·ε
        目标速度 v* = ε - x_0
        L = E ‖ v_θ(x_t, t) - v* ‖²
    """
    b = x0.size(0)
    device = x0.device
    t = torch.rand(b, 1, device=device)
    eps = torch.randn_like(x0)
    x_t = (1 - t) * x0 + t * eps
    target_v = eps - x0
    pred_v = model(x_t, t)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def flow_matching_sample(model: VelocityNet, n: int, steps: int,
                         device: torch.device,
                         return_traj: bool = False) -> torch.Tensor:
    """
    经典 Flow Matching 采样: 用 Euler 法求解概率流 ODE。

    从 t=1 (噪声) 积分到 t=0 (数据):
        dx/dt = v_θ(x, t)
        x_{t-Δ} = x_t + v_θ(x_t, t)·(t_next - t)     (t_next < t, 故是减去)

    NFE (网络前向次数) = steps。steps 越少, 离散化误差越大。
    """
    model.eval()
    x = torch.randn(n, 2, device=device)  # t=1
    traj = [x.clone()]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_cur = ts[i]
        dt = ts[i + 1] - ts[i]  # 负数
        tt = torch.full((n, 1), t_cur.item(), device=device)
        v = model(x, tt)
        x = x + dt * v
        traj.append(x.clone())
    if return_traj:
        return x, traj
    return x


# ==============================================================================
# Flow Map Matching: 流映射网络 X_{s,t}(x) = x + (t-s)·g_φ(x, s, t)
# ==============================================================================

class FlowMapNet(nn.Module):
    """
    流映射网络。

    参数化:
        X_{s,t}(x) = x + (t - s)·g_φ(x, s, t)

    - g_φ 输入: 状态 x + 两个时间 (s, t) 的嵌入
    - 恒等边界 X_{s,s}=x 由 (t-s) 因子自动保证 (无需额外约束)
    """

    def __init__(self, data_dim: int = 2, hidden: int = 128,
                 time_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.emb_s = FourierTimeEmbedding(time_dim)
        self.emb_t = FourierTimeEmbedding(time_dim)
        layers = [nn.Linear(data_dim + 2 * time_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def g(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """平均速度场 g_φ(x, s, t)。"""
        h = torch.cat([x, self.emb_s(s), self.emb_t(t)], dim=-1)
        return self.net(h)

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """流映射 X_{s,t}(x) = x + (t-s)·g_φ(x,s,t)。"""
        return x + (t - s) * self.g(x, s, t)


def lagrangian_flow_map_loss(flow_map: FlowMapNet,
                             velocity: VelocityNet,
                             x0: torch.Tensor) -> torch.Tensor:
    """
    Lagrangian Flow Map Matching (LFMM) 损失 —— 论文核心目标 (蒸馏形式)。

        L = E_{s,t,x_s} ‖ ∂_t X_{s,t}(x_s) - v_t( X_{s,t}(x_s) ) ‖²

    实现要点:
        1. 采样 s, t ~ U(0,1) (允许 t<s 或 t>s, 覆盖任意方向)
        2. x_s 取自 s 时刻边缘分布: x_s = (1-s)x_0 + s·ε
        3. ∂_t X_{s,t}(x_s) 用【前向模式自动微分 (JVP)】精确计算:
           对函数 f(τ) = X_{s,τ}(x_s), 求方向导数 f'(t)·1 = ∂_t X。
           这样无需在时间上模拟/展开 ODE, 每步仅一次 JVP, 精确且高效。
        4. 目标速度 v_t 来自【冻结的】已训练速度场 (stop-grad) —— 即蒸馏。
    """
    b = x0.size(0)
    device = x0.device

    s = torch.rand(b, 1, device=device)
    t = torch.rand(b, 1, device=device)
    eps = torch.randn_like(x0)
    x_s = (1 - s) * x0 + s * eps  # s 时刻的样本

    # --- 用 JVP 计算 X 及 ∂_t X (对 t 求导, 切向量取 1) ---
    def map_of_t(t_in: torch.Tensor) -> torch.Tensor:
        return flow_map(x_s, s, t_in)

    tangent = torch.ones_like(t)
    X, dX_dt = torch.func.jvp(map_of_t, (t,), (tangent,))

    # --- 目标: 冻结速度场在映射输出处的值 ---
    with torch.no_grad():
        v_target = velocity(X, t)

    return F.mse_loss(dX_dt, v_target)


@torch.no_grad()
def flow_map_sample(flow_map: FlowMapNet, n: int, steps: int,
                    device: torch.device,
                    return_traj: bool = False) -> torch.Tensor:
    """
    流映射采样 (利用半群性质做少步生成)。

    时间网格 1=τ_0 > τ_1 > ... > τ_k=0, 迭代:
        x ← X_{τ_i, τ_{i+1}}(x)
    - steps=1: 一步生成 x = X_{1,0}(噪声)   ← 本方法的核心优势
    - steps>1: 少步 refine, 半群性质保证一致性

    NFE = steps。
    """
    flow_map.eval()
    x = torch.randn(n, 2, device=device)  # t=1
    traj = [x.clone()]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        s = torch.full((n, 1), ts[i].item(), device=device)
        t = torch.full((n, 1), ts[i + 1].item(), device=device)
        x = flow_map(x, s, t)
        traj.append(x.clone())
    if return_traj:
        return x, traj
    return x


# ==============================================================================
# 训练函数
# ==============================================================================

def train_flow_matching(steps: int = 4000, batch: int = 512, lr: float = 2e-3,
                        device: torch.device = None, log_every: int = 1000,
                        verbose: bool = True) -> VelocityNet:
    """训练经典 Flow Matching 速度场 v_θ。"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VelocityNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if verbose:
        print(f"[Flow Matching] 训练速度场 v_θ (steps={steps}, batch={batch})")
    for it in range(steps):
        model.train()
        x0 = sample_eight_gaussians(batch, device)
        loss = flow_matching_loss(model, x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ((it + 1) % log_every == 0 or it == 0):
            print(f"  step {it+1:5d}/{steps} | loss = {loss.item():.4f}")
    return model


def train_flow_map(velocity: VelocityNet, steps: int = 5000, batch: int = 512,
                   lr: float = 2e-3, device: torch.device = None,
                   log_every: int = 1000, verbose: bool = True) -> FlowMapNet:
    """
    用 LFMM 目标, 从已训练速度场蒸馏出流映射 X_{s,t}。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    velocity.eval()
    for p in velocity.parameters():
        p.requires_grad_(False)

    flow_map = FlowMapNet().to(device)
    opt = torch.optim.Adam(flow_map.parameters(), lr=lr)
    if verbose:
        print(f"[Flow Map Matching] LFMM 蒸馏流映射 X_(s,t) (steps={steps}, batch={batch})")
    for it in range(steps):
        flow_map.train()
        x0 = sample_eight_gaussians(batch, device)
        loss = lagrangian_flow_map_loss(flow_map, velocity, x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ((it + 1) % log_every == 0 or it == 0):
            print(f"  step {it+1:5d}/{steps} | LFMM loss = {loss.item():.6f}")
    return flow_map


# ==============================================================================
# 量化指标: 能量距离 (Energy Distance)
# ==============================================================================

def energy_distance(x: torch.Tensor, y: torch.Tensor, max_n: int = 2000) -> float:
    """
    能量距离 (Energy Distance) —— 两个样本集之间的无偏分布距离。

        D²(X, Y) = 2·E‖X-Y‖ - E‖X-X'‖ - E‖Y-Y'‖

    越接近 0 说明两个分布越接近。用于量化"生成样本"与"真实样本"的差异。
    """
    x = x[:max_n]; y = y[:max_n]

    def pdist_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(a, b).mean()

    dxy = pdist_mean(x, y)
    dxx = pdist_mean(x, x)
    dyy = pdist_mean(y, y)
    val = (2 * dxy - dxx - dyy).clamp(min=0.0)
    return val.sqrt().item()


# ==============================================================================
# 演示
# ==============================================================================

def demo_flow_map():
    """Flow Map Matching 演示: 训练 + 少步生成 + 与经典 FM 对比。"""
    print("=" * 66)
    print("  Flow Map Matching (流映射匹配) 演示")
    print("=" * 66)

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 1. 训练经典 Flow Matching 速度场
    velocity = train_flow_matching(steps=4000, device=device)

    # 2. 蒸馏出流映射
    print()
    flow_map = train_flow_map(velocity, steps=5000, device=device)

    # 3. 对比: 相同 NFE 下的样本质量 (能量距离越小越好)
    print("\n" + "-" * 66)
    print("  相同 NFE 下的生成质量对比 (能量距离 vs 真实分布, 越小越好)")
    print("-" * 66)
    real = sample_eight_gaussians(2000, device)
    print(f"  {'NFE':>4} | {'Flow Matching (Euler)':>22} | {'Flow Map (少步)':>18}")
    print(f"  {'-'*4}-+-{'-'*22}-+-{'-'*18}")
    for nfe in [1, 2, 4, 8, 16]:
        fm = flow_matching_sample(velocity, 2000, steps=nfe, device=device)
        fmap = flow_map_sample(flow_map, 2000, steps=nfe, device=device)
        ed_fm = energy_distance(fm, real)
        ed_map = energy_distance(fmap, real)
        print(f"  {nfe:>4} | {ed_fm:>22.4f} | {ed_map:>18.4f}")

    print("\n观察: NFE=1~2 时经典 Flow Matching 误差很大 (Euler 步长太粗),")
    print("      而 Flow Map 在 1 步就能给出接近真实分布的样本。")
    print("\n[Flow Map Matching 演示完成]")
    return velocity, flow_map


if __name__ == '__main__':
    demo_flow_map()
