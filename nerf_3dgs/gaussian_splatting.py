"""
3D 高斯泼溅 (3D Gaussian Splatting, 3DGS) —— 从零实现

核心思想:
    3DGS 用一组 3D 高斯椭球体来显式表示场景。每个高斯有:
    - 位置 μ (3D 中心坐标)
    - 协方差 Σ (3D 椭球形状和朝向)
    - 不透明度 α (透明度)
    - 颜色 c (通过球谐函数表示，支持视角相关效果)

    渲染时，将 3D 高斯投影到 2D 图像平面 (splatting)，
    然后按深度排序，前到后 alpha 混合。

    这与 NeRF 完全不同:
    - NeRF: 隐式 (MLP) + 逐光线采样 → 慢
    - 3DGS: 显式 (点云) + 光栅化 → 快 (100-1000x)

数学框架:
    3D 高斯:
        G(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))

    协方差分解 (保证半正定):
        Σ = R S Sᵀ Rᵀ
        R = 旋转矩阵 (由四元数参数化)
        S = 缩放矩阵 (对角)

    2D 投影 (EWA Splatting):
        Σ' = J W Σ Wᵀ Jᵀ
        J = 雅可比矩阵 (透视投影的局部线性化)
        W = 相机外参

    Alpha 混合 (前到后):
        C = Σ_i c_i · α_i · G_i(x) · T_i
        T_i = Π_{j<i} (1 - α_j · G_j(x))

架构概览:
    ┌───────────────────────────────────────────────┐
    │  3D 高斯点云 (可优化参数)                      │
    │  每个点: {μ, Σ(q,s), α, color}                │
    ├───────────────────────────────────────────────┤
    │  投影到 2D                                     │
    │  3D Gaussian → 2D Gaussian (EWA Splatting)    │
    ├───────────────────────────────────────────────┤
    │  按深度排序                                    │
    ├───────────────────────────────────────────────┤
    │  逐像素 Alpha 混合                             │
    │  C(pixel) = Σ c_i · α_i · G_2d_i(pixel) · T_i│
    └───────────────────────────────────────────────┘

与 NeRF 的关键区别:
    ┌──────────────────┬─────────────────┬─────────────────┐
    │                  │     NeRF        │     3DGS        │
    ├──────────────────┼─────────────────┼─────────────────┤
    │ 场景表示         │ 隐式 (MLP)     │ 显式 (高斯点云) │
    │ 渲染方式         │ Volume Rendering│ Rasterization   │
    │ 渲染速度         │ 慢 (分钟级)    │ 快 (实时)       │
    │ 训练速度         │ 慢 (小时级)    │ 快 (分钟级)     │
    │ 内存使用         │ 低 (只有MLP)   │ 高 (百万高斯)   │
    │ 编辑能力         │ 难             │ 容易 (直接操作)  │
    │ 新视角合成       │ 高质量         │ 高质量          │
    │ 可微渲染         │ 是             │ 是              │
    └──────────────────┴─────────────────┴─────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List


# ==============================================================================
# 模型配置
# ==============================================================================

@dataclass
class GaussianSplattingConfig:
    """3DGS 配置。"""
    num_gaussians: int = 500       # 高斯数量
    img_size: int = 32             # 图像尺寸
    sh_degree: int = 0             # 球谐函数阶数 (0=常数颜色, 1=一阶, ...)
    near: float = 0.1              # 近平面
    far: float = 100.0             # 远平面
    # 自适应密度控制
    densify_interval: int = 10     # 密化间隔 (每 N 步)
    densify_grad_thresh: float = 0.01  # 梯度阈值
    opacity_reset_interval: int = 30   # 不透明度重置间隔


# ==============================================================================
# 四元数工具函数
# ==============================================================================

def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """
    四元数 → 旋转矩阵。

    四元数 q = (w, x, y, z) 编码 3D 旋转。
    相比欧拉角: 无万向锁问题，插值平滑。
    相比旋转矩阵: 只需 4 个参数 (vs 9)。

    Args:
        q: 四元数, shape (..., 4), [w, x, y, z]
    Returns:
        R: 旋转矩阵, shape (..., 3, 3)
    """
    q = F.normalize(q, dim=-1)  # 归一化
    w, x, y, z = q.unbind(dim=-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R


def build_covariance_3d(scales: torch.Tensor,
                        rotations: torch.Tensor) -> torch.Tensor:
    """
    从缩放和旋转构建 3D 协方差矩阵。

    Σ = R @ S @ Sᵀ @ Rᵀ

    这种分解保证 Σ 是半正定的 (协方差矩阵的要求)。

    Args:
        scales:    缩放, shape (N, 3), 正数
        rotations: 四元数, shape (N, 4)
    Returns:
        cov3d: 3D 协方差矩阵, shape (N, 3, 3)
    """
    R = quaternion_to_rotation(rotations)  # (N, 3, 3)
    S = torch.diag_embed(scales)           # (N, 3, 3)
    M = R @ S                              # (N, 3, 3)
    cov3d = M @ M.transpose(-1, -2)        # (N, 3, 3)
    return cov3d


# ==============================================================================
# 3DGS 可微渲染器
# ==============================================================================

def project_gaussians(means: torch.Tensor, cov3d: torch.Tensor,
                      cam_pos: torch.Tensor, look_at: torch.Tensor,
                      focal: float, img_size: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 3D 高斯投影到 2D 图像平面。

    步骤:
        1. 将 3D 中心变换到相机坐标系
        2. 投影到 2D 像素坐标
        3. 计算 2D 协方差 (EWA Splatting 近似)

    EWA (Elliptical Weighted Average) Splatting:
        Σ_2d = J @ W @ Σ_3d @ Wᵀ @ Jᵀ

        J = 透视投影的雅可比矩阵 = [[f/z, 0, -fx/z²],
                                      [0, f/z, -fy/z²]]
        W = 相机旋转矩阵 (世界→相机)

    Args:
        means:    3D 中心, shape (N, 3)
        cov3d:    3D 协方差, shape (N, 3, 3)
        cam_pos:  相机位置, shape (3,)
        look_at:  注视点, shape (3,)
        focal:    焦距
        img_size: 图像尺寸
    Returns:
        means_2d: 2D 中心 (像素坐标), shape (N, 2)
        cov_2d:   2D 协方差, shape (N, 2, 2)
        depths:   深度, shape (N,)
    """
    device = means.device
    N = means.size(0)

    # 构建相机坐标系
    forward = F.normalize(look_at - cam_pos, dim=0)
    right = F.normalize(torch.linalg.cross(forward, torch.tensor([0., 1., 0.], device=device)), dim=0)
    up = torch.linalg.cross(right, forward)
    W = torch.stack([right, up, -forward], dim=0)  # (3, 3) 世界→相机

    # 变换到相机坐标系
    means_cam = (means - cam_pos.unsqueeze(0)) @ W.T  # (N, 3)

    # 深度 (z 坐标, 注意相机看向 -z)
    depths = -means_cam[:, 2]  # (N,)

    # 投影到像素坐标
    x_cam = means_cam[:, 0] / (-means_cam[:, 2] + 1e-8)
    y_cam = means_cam[:, 1] / (-means_cam[:, 2] + 1e-8)

    means_2d = torch.stack([
        x_cam * focal + img_size / 2,
        -y_cam * focal + img_size / 2,
    ], dim=-1)  # (N, 2)

    # EWA Splatting: 计算 2D 协方差
    # 简化: 只取 3x3 协方差在相机坐标系下的前 2x2 子矩阵，除以 z²
    cov_cam = W.unsqueeze(0) @ cov3d @ W.T.unsqueeze(0)  # (N, 3, 3)
    z_sq = (depths ** 2).clamp(min=0.01).unsqueeze(-1).unsqueeze(-1)

    # J @ cov @ Jᵀ 的近似 (透视除法)
    cov_2d = cov_cam[:, :2, :2] * (focal ** 2) / z_sq  # (N, 2, 2)

    # 添加最小方差 (避免退化)
    cov_2d = cov_2d + 0.3 * torch.eye(2, device=device).unsqueeze(0)

    return means_2d, cov_2d, depths


def render_gaussians(means_2d: torch.Tensor, cov_2d: torch.Tensor,
                     depths: torch.Tensor, colors: torch.Tensor,
                     opacities: torch.Tensor, img_size: int
                     ) -> torch.Tensor:
    """
    渲染 2D 高斯到图像 (可微 alpha 混合)。

    步骤:
        1. 按深度排序 (前到后)
        2. 对每个像素，计算所有高斯的贡献
        3. Alpha 混合: C = Σ c_i · α_i · G_i · T_i

    Args:
        means_2d:  2D 中心, shape (N, 2)
        cov_2d:    2D 协方差, shape (N, 2, 2)
        depths:    深度, shape (N,)
        colors:    颜色, shape (N, 3)
        opacities: 不透明度, shape (N,)
        img_size:  图像尺寸
    Returns:
        image: 渲染图像, shape (H, W, 3)
    """
    device = means_2d.device
    N = means_2d.size(0)
    H = W = img_size

    # 1. 按深度排序
    sort_idx = depths.argsort()
    means_2d = means_2d[sort_idx]
    cov_2d = cov_2d[sort_idx]
    colors = colors[sort_idx]
    opacities = opacities[sort_idx]

    # 过滤掉在图像外或深度为负的高斯
    valid = (depths[sort_idx] > 0.1) & \
            (means_2d[:, 0] > -img_size) & (means_2d[:, 0] < 2 * img_size) & \
            (means_2d[:, 1] > -img_size) & (means_2d[:, 1] < 2 * img_size)

    means_2d = means_2d[valid]
    cov_2d = cov_2d[valid]
    colors = colors[valid]
    opacities = opacities[valid]
    N_valid = means_2d.size(0)

    if N_valid == 0:
        return torch.ones(H, W, 3, device=device)

    # 2. 构建像素网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    pixels = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)

    # 3. 计算每个高斯对每个像素的贡献
    # 为效率，只计算有效范围内的高斯
    image = torch.ones(H, W, 3, device=device)  # 白色背景
    T_accum = torch.ones(H, W, device=device)     # 累积透射率

    # 协方差矩阵求逆 (用于计算高斯值)
    # 对 2x2 矩阵: [[a,b],[c,d]]⁻¹ = 1/(ad-bc) [[d,-b],[-c,a]]
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]
    det = (a * d - b * b).clamp(min=1e-6)
    cov_inv = torch.stack([d, -b, -b, a], dim=-1).reshape(N_valid, 2, 2) / det.unsqueeze(-1).unsqueeze(-1)

    # 逐高斯渲染 (前到后 alpha 混合)
    for i in range(N_valid):
        mu = means_2d[i]  # (2,)
        diff = pixels - mu.unsqueeze(0).unsqueeze(0)  # (H, W, 2)

        # 计算高斯值: exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ))
        inv = cov_inv[i]  # (2, 2)
        mahal = (diff @ inv * diff).sum(dim=-1)  # (H, W)
        gauss_val = torch.exp(-0.5 * mahal)

        # alpha = 不透明度 × 高斯值
        alpha = (opacities[i] * gauss_val).clamp(max=0.99)  # (H, W)

        # 颜色贡献
        color_i = colors[i]  # (3,)
        image = image + (T_accum * alpha).unsqueeze(-1) * \
                (color_i.unsqueeze(0).unsqueeze(0) - image)

        # 更新透射率
        T_accum = T_accum * (1 - alpha)

        # 早停: 如果几乎所有像素都已饱和
        if T_accum.max() < 0.01:
            break

    return image.clamp(0, 1)


# ==============================================================================
# 3DGS 模型
# ==============================================================================

class GaussianSplatting(nn.Module):
    """
    3D Gaussian Splatting 完整模型。

    可优化参数 (每个高斯):
        - means:     3D 位置, shape (N, 3)
        - scales:    缩放 (log空间), shape (N, 3)
        - rotations: 四元数, shape (N, 4)
        - opacities: 不透明度 (logit空间), shape (N, 1)
        - colors:    RGB 颜色 (sigmoid前), shape (N, 3)
    """

    def __init__(self, config: GaussianSplattingConfig,
                 init_points: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        N = config.num_gaussians

        # 初始化高斯参数
        if init_points is not None:
            self.means = nn.Parameter(init_points.clone())
        else:
            # 随机初始化在单位球内
            self.means = nn.Parameter(torch.randn(N, 3) * 0.5)

        # log 缩放 (保证正数)
        self.log_scales = nn.Parameter(torch.full((N, 3), -2.0))

        # 四元数 (初始为单位四元数 = 无旋转)
        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0  # w = 1
        self.rotations = nn.Parameter(quats)

        # 不透明度 (logit 空间, sigmoid 后为实际不透明度)
        self.opacity_logit = nn.Parameter(torch.full((N, 1), 0.0))

        # 颜色 (sigmoid 前的值)
        self.color_pre = nn.Parameter(torch.randn(N, 3) * 0.5)

    @property
    def scales(self):
        """实际缩放 (正数)。"""
        return torch.exp(self.log_scales)

    @property
    def opacities(self):
        """实际不透明度 [0, 1]。"""
        return torch.sigmoid(self.opacity_logit).squeeze(-1)

    @property
    def colors(self):
        """实际颜色 [0, 1]。"""
        return torch.sigmoid(self.color_pre)

    def render(self, cam_pos: torch.Tensor, look_at: torch.Tensor,
               focal: float = None) -> torch.Tensor:
        """
        渲染一帧图像。

        步骤:
            1. 构建 3D 协方差矩阵
            2. 投影到 2D
            3. Alpha 混合渲染

        Args:
            cam_pos: 相机位置, shape (3,)
            look_at: 注视点, shape (3,)
            focal:   焦距
        Returns:
            image: 渲染图像, shape (H, W, 3)
        """
        if focal is None:
            focal = float(self.config.img_size)

        # 1. 构建 3D 协方差
        cov3d = build_covariance_3d(self.scales, self.rotations)

        # 2. 投影到 2D
        means_2d, cov_2d, depths = project_gaussians(
            self.means, cov3d, cam_pos, look_at,
            focal, self.config.img_size
        )

        # 3. 渲染
        image = render_gaussians(
            means_2d, cov_2d, depths,
            self.colors, self.opacities,
            self.config.img_size
        )

        return image

    def get_stats(self) -> dict:
        """获取高斯统计信息。"""
        return {
            'num_gaussians': self.means.size(0),
            'mean_opacity': self.opacities.mean().item(),
            'mean_scale': self.scales.mean().item(),
            'pos_range': [self.means.min().item(), self.means.max().item()],
        }


# ==============================================================================
# 训练函数
# ==============================================================================

def train_3dgs(config: GaussianSplattingConfig = None, epochs: int = 100,
               lr: float = 1e-2, num_views: int = 8,
               device: torch.device = None) -> dict:
    """训练 3DGS 模型。"""
    if config is None:
        config = GaussianSplattingConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[3DGS] 设备: {device}")
    print(f"[3DGS] 高斯数量: {config.num_gaussians}, 图像: {config.img_size}×{config.img_size}")

    # 创建合成场景 (复用 NeRF 的场景)
    from nerf import create_synthetic_scene
    scene = create_synthetic_scene(config.img_size, num_views, device)
    print(f"[3DGS] 训练视角: {num_views}")

    # 初始化: 在球面附近随机分布高斯
    init_points = torch.randn(config.num_gaussians, 3, device=device) * 0.8
    model = GaussianSplatting(config, init_points=init_points).to(device)

    # 不同参数使用不同学习率 (3DGS 论文的关键技巧)
    optimizer = torch.optim.Adam([
        {'params': [model.means], 'lr': lr * 0.1},       # 位置: 较小学习率
        {'params': [model.log_scales], 'lr': lr * 0.5},   # 缩放
        {'params': [model.rotations], 'lr': lr * 0.1},    # 旋转
        {'params': [model.opacity_logit], 'lr': lr},       # 不透明度
        {'params': [model.color_pre], 'lr': lr},           # 颜色
    ])

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[3DGS] 参数量: {num_params:,}")

    # 训练
    history = {'loss': [], 'psnr': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_views_trained = 0

        for view in scene['views']:
            cam_pos = view['cam_pos']
            look_at = view['look_at']
            target = view['image']  # (H, W, 3)

            # 渲染
            rendered = model.render(cam_pos, look_at, focal=scene['focal'])

            # L1 + SSIM-like 损失 (简化版)
            l1_loss = F.l1_loss(rendered, target)
            # 简化的结构相似性: 局部均值差
            mse_loss = F.mse_loss(rendered, target)
            loss = 0.8 * l1_loss + 0.2 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_views_trained += 1

        avg_loss = epoch_loss / num_views_trained
        psnr = -10 * math.log10(avg_loss + 1e-8)
        history['loss'].append(avg_loss)
        history['psnr'].append(psnr)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            stats = model.get_stats()
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.6f} | PSNR: {psnr:.2f} dB | "
                  f"Opacity: {stats['mean_opacity']:.3f}")

    # 测试渲染
    model.eval()
    test_cam = torch.tensor([3.0, 2.0, 3.0], device=device)
    test_look = torch.zeros(3, device=device)
    with torch.no_grad():
        test_img = model.render(test_cam, test_look, focal=scene['focal'])
    print(f"[3DGS] 测试渲染: {test_img.shape}, "
          f"颜色范围 [{test_img.min():.3f}, {test_img.max():.3f}]")

    return {
        'model': model, 'history': history, 'scene': scene,
        'test_image': test_img,
    }


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_3dgs():
    """3DGS 演示。"""
    print("=" * 60)
    print("3D 高斯泼溅 (3D Gaussian Splatting) 演示")
    print("=" * 60)

    config = GaussianSplattingConfig(
        num_gaussians=300,
        img_size=16,       # 演示用小图
    )

    results = train_3dgs(config=config, epochs=100, num_views=6)

    print("\n--- 训练结果 ---")
    h = results['history']
    print(f"  初始 PSNR: {h['psnr'][0]:.2f} dB")
    print(f"  最终 PSNR: {h['psnr'][-1]:.2f} dB")
    print(f"  PSNR 提升: {h['psnr'][-1] - h['psnr'][0]:.2f} dB")

    stats = results['model'].get_stats()
    print(f"\n--- 高斯统计 ---")
    print(f"  数量: {stats['num_gaussians']}")
    print(f"  平均不透明度: {stats['mean_opacity']:.3f}")
    print(f"  平均缩放: {stats['mean_scale']:.4f}")
    print(f"  位置范围: {stats['pos_range']}")

    print("\n[3DGS 演示完成]")
    return results


if __name__ == '__main__':
    demo_3dgs()
