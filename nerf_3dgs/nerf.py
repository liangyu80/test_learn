"""
神经辐射场 (Neural Radiance Fields, NeRF) —— 从零实现

核心思想:
    NeRF 用一个 MLP 网络来表示 3D 场景。给定空间中一个点 (x,y,z) 和
    观察方向 (θ,φ)，网络输出该点的颜色 (r,g,b) 和体密度 (σ)。
    通过体渲染 (Volume Rendering) 将 3D 信息投影到 2D 图像。

    直觉: NeRF 把整个 3D 场景"记忆"在网络权重里。

数学框架:
    场景表示:
        F_θ: (x, y, z, θ, φ) → (r, g, b, σ)

    体渲染 (Volume Rendering):
        对每条光线 r(t) = o + t·d (o=相机原点, d=方向):

        C(r) = ∫_{t_n}^{t_f} T(t) · σ(r(t)) · c(r(t), d) dt

        其中:
        T(t) = exp(-∫_{t_n}^{t} σ(r(s)) ds)   ← 透射率 (光线到达 t 的概率)

    离散化近似 (实际计算):
        Ĉ(r) = Σ_i T_i · α_i · c_i

        α_i = 1 - exp(-σ_i · δ_i)              ← 不透明度
        T_i = Π_{j<i} (1 - α_j)                ← 累积透射率
        δ_i = t_{i+1} - t_i                    ← 采样间距

    位置编码 (Positional Encoding):
        γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^{L-1}πp), cos(2^{L-1}πp)]

        为什么需要? MLP 难以学习高频细节。位置编码将低维输入映射到
        高维空间，让网络能表示精细的几何和纹理。

架构概览:
    ┌──────────────────────────────────────────────┐
    │              位置编码                         │
    │  (x,y,z) → γ(x,y,z)  [60维]               │
    │  (θ,φ)   → γ(θ,φ)    [24维]               │
    ├──────────────────────────────────────────────┤
    │              MLP 网络                        │
    │  γ(xyz) → [256]×4 → σ (密度)               │
    │                    ↓                         │
    │            [256] + γ(dir) → [128] → rgb     │
    │  (密度只依赖位置，颜色依赖位置+方向)         │
    ├──────────────────────────────────────────────┤
    │              体渲染                          │
    │  沿光线采样 N 个点 → 查询颜色和密度          │
    │  → 加权求和得到像素颜色                      │
    └──────────────────────────────────────────────┘

与 3DGS 的关键区别:
    1. NeRF 是隐式表示 (MLP 权重里)，3DGS 是显式表示 (高斯点云)
    2. NeRF 渲染需要逐光线采样，速度慢; 3DGS 用 rasterization，快得多
    3. NeRF 训练和渲染都依赖 MLP 前向，3DGS 只在优化时需要梯度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class NeRFConfig:
    """NeRF 配置。"""
    # 位置编码
    pos_enc_levels: int = 10      # 位置编码的频率层数 L (xyz)
    dir_enc_levels: int = 4       # 方向编码的频率层数 L (direction)
    # 网络
    hidden_dim: int = 256         # 隐藏层维度
    num_layers: int = 8           # 网络深度
    skip_layer: int = 4           # skip connection 位置 (NeRF 论文中第 4 层)
    # 渲染
    num_samples: int = 64         # 每条光线上的采样点数
    near: float = 2.0             # 近平面
    far: float = 6.0              # 远平面
    # 场景
    img_size: int = 32            # 图像尺寸 (演示用小尺寸)


# ==============================================================================
# 位置编码 (Positional Encoding)
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    NeRF 位置编码。

    将标量 p 映射为:
        γ(p) = [p, sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ...,
                sin(2^{L-1}πp), cos(2^{L-1}πp)]

    对 3D 坐标 (x,y,z): 输出维度 = 3 × (1 + 2L)
    对方向 (θ,φ):       输出维度 = 3 × (1 + 2L)  (方向也用 3D 单位向量)

    为什么需要位置编码?
        MLP 具有"频谱偏置"(spectral bias)，倾向于学习低频函数。
        高频的 sin/cos 编码帮助网络表示高频细节(如纹理边缘)。
    """

    def __init__(self, num_levels: int, include_input: bool = True):
        super().__init__()
        self.num_levels = num_levels
        self.include_input = include_input
        # 频率: 2^0, 2^1, ..., 2^{L-1}
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.arange(num_levels) * math.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入坐标, shape (..., D)
        Returns:
            编码后, shape (..., D × (1 + 2L))  若 include_input
        """
        encodings = []
        if self.include_input:
            encodings.append(x)

        for freq in self.freq_bands:
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))

        return torch.cat(encodings, dim=-1)

    def output_dim(self, input_dim: int) -> int:
        """计算编码后的输出维度。"""
        d = input_dim * 2 * self.num_levels
        if self.include_input:
            d += input_dim
        return d


# ==============================================================================
# NeRF 网络
# ==============================================================================

class NeRFNetwork(nn.Module):
    """
    NeRF MLP 网络。

    架构 (来自原论文):
        位置编码(xyz) → [Linear+ReLU]×4 → skip connection → [Linear+ReLU]×3
                                                                  ↓
                                                            → σ (密度, 无激活)
                                                            → feature [256]
                                                                  ↓
                                                    feature + 编码(dir) → [Linear+ReLU]
                                                                  ↓
                                                            → rgb (Sigmoid)

    关键设计:
        1. 密度 σ 只依赖位置 (x,y,z)，不依赖观察方向
        2. 颜色 rgb 依赖位置和方向 (实现视角相关效果，如高光)
        3. 第 4 层有 skip connection (类似 ResNet)
    """

    def __init__(self, config: NeRFConfig):
        super().__init__()
        self.config = config

        # 位置编码器
        self.pos_encoder = PositionalEncoding(config.pos_enc_levels)
        self.dir_encoder = PositionalEncoding(config.dir_enc_levels)

        pos_dim = self.pos_encoder.output_dim(3)   # 3D 位置
        dir_dim = self.dir_encoder.output_dim(3)    # 3D 方向
        D = config.hidden_dim

        # ---- 位置相关的 MLP (输出密度和特征) ----
        layers = []
        in_dim = pos_dim
        for i in range(config.num_layers):
            if i == config.skip_layer:
                in_dim += pos_dim  # skip connection
            layers.append(nn.Linear(in_dim, D))
            in_dim = D
        self.pos_layers = nn.ModuleList(layers)

        # 密度输出
        self.sigma_head = nn.Linear(D, 1)

        # ---- 方向相关的 MLP (输出颜色) ----
        self.feature_proj = nn.Linear(D, D)
        self.dir_layers = nn.Sequential(
            nn.Linear(D + dir_dim, D // 2),
            nn.ReLU(),
        )
        self.rgb_head = nn.Linear(D // 2, 3)

    def forward(self, pos: torch.Tensor,
                direction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pos:       3D 位置, shape (..., 3)
            direction: 观察方向, shape (..., 3), 应为单位向量
        Returns:
            rgb:   颜色, shape (..., 3), 范围 [0, 1]
            sigma: 体密度, shape (..., 1), 非负
        """
        # 位置编码
        pos_enc = self.pos_encoder(pos)
        dir_enc = self.dir_encoder(direction)

        # 位置 MLP
        h = pos_enc
        for i, layer in enumerate(self.pos_layers):
            if i == self.config.skip_layer:
                h = torch.cat([h, pos_enc], dim=-1)  # skip connection
            h = F.relu(layer(h))

        # 密度 (只依赖位置)
        sigma = F.relu(self.sigma_head(h))  # 非负密度

        # 颜色 (依赖位置 + 方向)
        feature = self.feature_proj(h)
        h = torch.cat([feature, dir_enc], dim=-1)
        h = self.dir_layers(h)
        rgb = torch.sigmoid(self.rgb_head(h))  # [0, 1]

        return rgb, sigma


# ==============================================================================
# 体渲染 (Volume Rendering)
# ==============================================================================

def volume_render(rgb: torch.Tensor, sigma: torch.Tensor,
                  t_vals: torch.Tensor, dirs: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    体渲染: 将沿光线的采样点颜色和密度合成为像素颜色。

    核心公式:
        δ_i = t_{i+1} - t_i                    (采样间距)
        α_i = 1 - exp(-σ_i · δ_i)              (不透明度)
        T_i = Π_{j<i} (1 - α_j)                (累积透射率)
        Ĉ(r) = Σ_i T_i · α_i · c_i            (最终颜色)

    直觉:
        - σ 大 → α 大 → 该点更不透明 → 对最终颜色贡献更大
        - T 表示光线穿过前面所有点后"还剩多少光"
        - 前面的点越不透明，后面的点贡献越小

    Args:
        rgb:    采样点颜色, shape (batch, num_samples, 3)
        sigma:  采样点密度, shape (batch, num_samples, 1)
        t_vals: 采样点在光线上的位置, shape (batch, num_samples)
        dirs:   光线方向, shape (batch, 3)
    Returns:
        color:   渲染的像素颜色, shape (batch, 3)
        depth:   渲染的深度图, shape (batch, 1)
        weights: 每个采样点的权重, shape (batch, num_samples)
    """
    sigma = sigma.squeeze(-1)  # (batch, num_samples)

    # 1. 计算相邻采样点的间距 δ
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (batch, num_samples - 1)
    # 最后一个间距设为一个大值 (无穷远)
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)

    # 乘以方向的范数 (考虑方向不一定是单位向量)
    deltas = deltas * dirs.norm(dim=-1, keepdim=True)

    # 2. 计算不透明度 α = 1 - exp(-σ · δ)
    alpha = 1.0 - torch.exp(-sigma * deltas)

    # 3. 计算累积透射率 T = Π_{j<i} (1 - α_j)
    # 使用 cumprod 实现
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]

    # 4. 权重 w_i = T_i · α_i
    weights = transmittance * alpha  # (batch, num_samples)

    # 5. 加权求和得到颜色
    color = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (batch, 3)

    # 6. 深度图 (加权平均深度)
    depth = (weights * t_vals).sum(dim=1, keepdim=True)  # (batch, 1)

    return color, depth, weights


# ==============================================================================
# 光线生成
# ==============================================================================

def generate_rays(img_size: int, focal: float,
                  cam_pos: torch.Tensor, look_at: torch.Tensor,
                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从相机位置生成图像平面上的光线。

    每个像素对应一条光线:
        r(t) = origin + t · direction

    Args:
        img_size: 图像尺寸 (H = W)
        focal:    焦距
        cam_pos:  相机位置, shape (3,)
        look_at:  注视点, shape (3,)
        device:   设备
    Returns:
        origins:    光线原点, shape (H*W, 3)
        directions: 光线方向, shape (H*W, 3)
    """
    # 构造像素网格
    i, j = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32, device=device),
        torch.arange(img_size, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # 像素坐标 → 相机坐标系方向
    # (将像素中心移到图像中心, 然后除以焦距)
    dirs_cam = torch.stack([
        (j - img_size / 2) / focal,
        -(i - img_size / 2) / focal,  # y 轴翻转
        -torch.ones_like(i),          # 相机看向 -z 方向
    ], dim=-1)  # (H, W, 3)

    # 构造相机到世界坐标的旋转矩阵
    forward = F.normalize(look_at - cam_pos, dim=0)
    right = F.normalize(torch.linalg.cross(forward, torch.tensor([0., 1., 0.], device=device)), dim=0)
    up = torch.linalg.cross(right, forward)
    rot = torch.stack([right, up, -forward], dim=-1)  # (3, 3)

    # 旋转方向到世界坐标
    dirs_world = (dirs_cam.reshape(-1, 3) @ rot.T)  # (H*W, 3)

    origins = cam_pos.unsqueeze(0).expand_as(dirs_world)
    return origins, dirs_world


# ==============================================================================
# NeRF 完整模型
# ==============================================================================

class NeRF(nn.Module):
    """
    Neural Radiance Fields 完整模型。

    整合: 位置编码 + MLP网络 + 光线采样 + 体渲染
    """

    def __init__(self, config: NeRFConfig):
        super().__init__()
        self.config = config
        self.network = NeRFNetwork(config)

    def render_rays(self, origins: torch.Tensor, directions: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染一批光线。

        步骤:
            1. 沿每条光线均匀采样 N 个点
            2. 查询每个点的颜色和密度
            3. 体渲染合成像素颜色

        Args:
            origins:    光线原点, shape (batch, 3)
            directions: 光线方向, shape (batch, 3)
        Returns:
            colors: 像素颜色, shape (batch, 3)
            depths: 深度图, shape (batch, 1)
        """
        config = self.config
        device = origins.device
        batch_size = origins.size(0)

        # 1. 均匀采样 + 随机扰动 (分层采样)
        t_vals = torch.linspace(config.near, config.far, config.num_samples,
                                device=device)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)  # (batch, N)

        if self.training:
            # 训练时添加随机扰动 (分层采样，避免规律性采样的伪影)
            noise = torch.rand_like(t_vals) * (config.far - config.near) / config.num_samples
            t_vals = t_vals + noise

        # 2. 计算采样点的 3D 坐标
        # pts = origin + t * direction
        pts = origins.unsqueeze(1) + t_vals.unsqueeze(-1) * directions.unsqueeze(1)
        # shape: (batch, num_samples, 3)

        # 扩展方向
        dirs_expanded = directions.unsqueeze(1).expand_as(pts)
        dirs_normalized = F.normalize(dirs_expanded, dim=-1)

        # 3. 查询网络
        # 展平后输入网络
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs_normalized.reshape(-1, 3)

        rgb_flat, sigma_flat = self.network(pts_flat, dirs_flat)

        rgb = rgb_flat.reshape(batch_size, config.num_samples, 3)
        sigma = sigma_flat.reshape(batch_size, config.num_samples, 1)

        # 4. 体渲染
        colors, depths, _ = volume_render(rgb, sigma, t_vals, directions)

        return colors, depths

    def render_image(self, cam_pos: torch.Tensor, look_at: torch.Tensor,
                     focal: float = None, batch_rays: int = 1024
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染完整图像。

        Args:
            cam_pos:    相机位置, shape (3,)
            look_at:    注视点, shape (3,)
            focal:      焦距 (默认 = img_size)
            batch_rays: 每批处理的光线数 (节省内存)
        Returns:
            image: 渲染图像, shape (H, W, 3)
            depth: 深度图, shape (H, W)
        """
        if focal is None:
            focal = float(self.config.img_size)

        device = next(self.parameters()).device
        origins, directions = generate_rays(
            self.config.img_size, focal, cam_pos, look_at, device
        )

        # 分批渲染 (避免 OOM)
        all_colors, all_depths = [], []
        for i in range(0, origins.size(0), batch_rays):
            o_batch = origins[i:i + batch_rays]
            d_batch = directions[i:i + batch_rays]
            colors, depths = self.render_rays(o_batch, d_batch)
            all_colors.append(colors)
            all_depths.append(depths)

        colors = torch.cat(all_colors, dim=0)
        depths = torch.cat(all_depths, dim=0)

        H = W = self.config.img_size
        image = colors.reshape(H, W, 3)
        depth = depths.reshape(H, W)

        return image, depth


# ==============================================================================
# 合成场景 (用于训练)
# ==============================================================================

def create_synthetic_scene(img_size: int, num_views: int,
                           device: torch.device) -> dict:
    """
    创建合成球体场景用于训练。

    场景: 一个彩色球体，位于原点
    生成多个视角的"真实"图像 (通过解析公式渲染)。

    光线-球体求交:
        球: ‖p‖² = r²
        光线: p = o + td
        代入: ‖o+td‖² = r²
        → t²(d·d) + 2t(o·d) + (o·o - r²) = 0
    """
    radius = 1.0
    views = []
    focal = float(img_size)

    for v in range(num_views):
        # 相机在球面上不同位置
        angle = 2 * math.pi * v / num_views
        elevation = 0.3 * math.sin(angle * 2)  # 稍有高低变化
        cam_pos = torch.tensor([
            4.0 * math.cos(angle),
            4.0 * elevation,
            4.0 * math.sin(angle)
        ], device=device)
        look_at = torch.zeros(3, device=device)

        origins, directions = generate_rays(img_size, focal, cam_pos, look_at, device)

        # 解析球体渲染
        image = _render_sphere(origins, directions, radius, device)
        image = image.reshape(img_size, img_size, 3)

        views.append({
            'cam_pos': cam_pos,
            'look_at': look_at,
            'image': image,
            'origins': origins,
            'directions': directions,
        })

    return {'views': views, 'radius': radius, 'focal': focal}


def _render_sphere(origins, directions, radius, device):
    """解析渲染球体 (光线追踪)。"""
    # 球心在原点
    a = (directions * directions).sum(dim=-1)
    b = 2.0 * (origins * directions).sum(dim=-1)
    c = (origins * origins).sum(dim=-1) - radius ** 2

    disc = b ** 2 - 4 * a * c
    hit = disc > 0

    # 计算交点
    t = (-b - torch.sqrt(disc.clamp(min=0))) / (2 * a + 1e-8)
    t = t.clamp(min=0)

    pts = origins + t.unsqueeze(-1) * directions
    normal = F.normalize(pts, dim=-1)

    # 简单着色: 法线颜色 + 光照
    light_dir = F.normalize(torch.tensor([1.0, 1.0, 1.0], device=device), dim=0)
    diffuse = (normal * light_dir).sum(dim=-1).clamp(min=0)

    # 颜色基于法线方向 (使球体有丰富颜色)
    base_color = (normal * 0.5 + 0.5)  # 映射到 [0,1]
    color = base_color * (0.3 + 0.7 * diffuse.unsqueeze(-1))

    # 未命中的光线 → 白色背景
    bg_color = torch.ones(origins.size(0), 3, device=device)
    color = torch.where(hit.unsqueeze(-1), color, bg_color)

    return color


# ==============================================================================
# 训练函数
# ==============================================================================

def train_nerf(config: NeRFConfig = None, epochs: int = 50,
               batch_rays: int = 512, lr: float = 5e-4,
               num_views: int = 8, device: torch.device = None) -> dict:
    """训练 NeRF 模型。"""
    if config is None:
        config = NeRFConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[NeRF] 设备: {device}")
    print(f"[NeRF] 图像: {config.img_size}×{config.img_size}, 采样点: {config.num_samples}/光线")

    # 创建合成场景
    scene = create_synthetic_scene(config.img_size, num_views, device)
    print(f"[NeRF] 训练视角: {num_views}")

    # 模型
    model = NeRF(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[NeRF] 参数量: {num_params:,}")

    # 训练
    history = {'loss': [], 'psnr': []}
    total_rays = config.img_size * config.img_size

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for view in scene['views']:
            origins = view['origins']
            directions = view['directions']
            target = view['image'].reshape(-1, 3)

            # 随机采样光线
            perm = torch.randperm(total_rays, device=device)[:batch_rays]
            o_batch = origins[perm]
            d_batch = directions[perm]
            t_batch = target[perm]

            colors, _ = model.render_rays(o_batch, d_batch)
            loss = F.mse_loss(colors, t_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        psnr = -10 * math.log10(avg_loss + 1e-8)
        history['loss'].append(avg_loss)
        history['psnr'].append(psnr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.6f} | PSNR: {psnr:.2f} dB")

    # 渲染测试视角
    model.eval()
    test_cam = torch.tensor([3.0, 2.0, 3.0], device=device)
    test_look = torch.zeros(3, device=device)
    with torch.no_grad():
        test_img, test_depth = model.render_image(test_cam, test_look,
                                                   focal=scene['focal'])
    print(f"[NeRF] 测试渲染: {test_img.shape}, "
          f"颜色范围 [{test_img.min():.3f}, {test_img.max():.3f}]")

    return {
        'model': model, 'history': history, 'scene': scene,
        'test_image': test_img, 'test_depth': test_depth,
    }


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_nerf():
    """NeRF 演示。"""
    print("=" * 60)
    print("神经辐射场 (NeRF) 演示")
    print("=" * 60)

    config = NeRFConfig(
        pos_enc_levels=6,
        dir_enc_levels=3,
        hidden_dim=128,
        num_layers=6,
        skip_layer=3,
        num_samples=32,
        img_size=16,          # 演示用小图
    )

    results = train_nerf(config=config, epochs=50, batch_rays=256,
                         num_views=6)

    print("\n--- 训练结果 ---")
    h = results['history']
    print(f"  初始 PSNR: {h['psnr'][0]:.2f} dB")
    print(f"  最终 PSNR: {h['psnr'][-1]:.2f} dB")
    print(f"  PSNR 提升: {h['psnr'][-1] - h['psnr'][0]:.2f} dB")

    print("\n[NeRF 演示完成]")
    return results


if __name__ == '__main__':
    demo_nerf()
