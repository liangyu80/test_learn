"""
NeRF vs 3DGS 对比实验

在相同的合成场景上训练两种方法，从多个维度对比:
    1. 渲染质量 (PSNR)
    2. 训练速度
    3. 渲染速度
    4. 参数量/内存
    5. 可编辑性

理论对比:
    ┌──────────────┬──────────────────┬──────────────────┐
    │              │      NeRF        │      3DGS        │
    ├──────────────┼──────────────────┼──────────────────┤
    │ 场景表示     │ 隐式 (MLP)      │ 显式 (高斯点云)  │
    │ 渲染方式     │ Volume Rendering │ Rasterization    │
    │ 渲染速度     │ ❌ 慢           │ ✅ 实时          │
    │ 训练速度     │ ❌ 慢           │ ✅ 快            │
    │ 内存使用     │ ✅ 小           │ ❌ 大            │
    │ 编辑         │ ❌ 难           │ ✅ 易 (操作点云) │
    │ 紧凑表示     │ ✅ MLP 权重小   │ ❌ 百万高斯      │
    │ 泛化能力     │ ✅ 较好         │ ❌ 过拟合风险    │
    │ 代表应用     │ 新视角合成      │ 实时渲染/VR      │
    └──────────────┴──────────────────┴──────────────────┘
"""

import torch
import torch.nn.functional as F
import time
import math

from nerf import NeRF, NeRFConfig, train_nerf, create_synthetic_scene
from gaussian_splatting import GaussianSplatting, GaussianSplattingConfig, train_3dgs


def compare_methods():
    """NeRF vs 3DGS 对比实验。"""
    print("=" * 70)
    print("  NeRF vs 3D Gaussian Splatting 对比实验")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    img_size = 16
    num_views = 6

    # ====================================================================
    # 1. 训练 NeRF
    # ====================================================================
    print("▶ 训练 NeRF...")
    print("-" * 50)
    nerf_config = NeRFConfig(
        pos_enc_levels=6, dir_enc_levels=3,
        hidden_dim=128, num_layers=6, skip_layer=3,
        num_samples=32, img_size=img_size,
    )
    t0 = time.time()
    nerf_results = train_nerf(config=nerf_config, epochs=50, batch_rays=256,
                               num_views=num_views, device=device)
    nerf_train_time = time.time() - t0
    print()

    # ====================================================================
    # 2. 训练 3DGS
    # ====================================================================
    print("▶ 训练 3DGS...")
    print("-" * 50)
    gs_config = GaussianSplattingConfig(
        num_gaussians=300, img_size=img_size,
    )
    t0 = time.time()
    gs_results = train_3dgs(config=gs_config, epochs=100,
                             num_views=num_views, device=device)
    gs_train_time = time.time() - t0
    print()

    # ====================================================================
    # 3. 对比分析
    # ====================================================================
    print("=" * 70)
    print("  对比分析")
    print("=" * 70)

    nerf_model = nerf_results['model']
    gs_model = gs_results['model']

    # --- 参数量 ---
    nerf_params = sum(p.numel() for p in nerf_model.parameters())
    gs_params = sum(p.numel() for p in gs_model.parameters())

    print(f"\n📊 模型规模:")
    print(f"  NeRF: {nerf_params:,} 参数 (MLP 权重)")
    print(f"  3DGS: {gs_params:,} 参数 ({gs_config.num_gaussians} 个高斯)")
    print(f"  → NeRF 更紧凑" if nerf_params < gs_params else "  → 3DGS 更紧凑")

    # --- 训练时间 ---
    print(f"\n📊 训练时间:")
    print(f"  NeRF: {nerf_train_time:.2f}s")
    print(f"  3DGS: {gs_train_time:.2f}s")

    # --- 训练质量 ---
    print(f"\n📊 训练质量 (PSNR):")
    nerf_psnr = nerf_results['history']['psnr'][-1]
    gs_psnr = gs_results['history']['psnr'][-1]
    print(f"  NeRF: {nerf_psnr:.2f} dB")
    print(f"  3DGS: {gs_psnr:.2f} dB")

    # --- 渲染速度 ---
    print(f"\n📊 渲染速度 (单帧):")
    test_cam = torch.tensor([3.0, 2.0, 3.0], device=device)
    test_look = torch.zeros(3, device=device)
    focal = float(img_size)

    nerf_model.eval()
    gs_model.eval()

    # 预热
    with torch.no_grad():
        nerf_model.render_image(test_cam, test_look, focal=focal)
        gs_model.render(test_cam, test_look, focal=focal)

    # 计时
    n_runs = 5
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            nerf_model.render_image(test_cam, test_look, focal=focal)
    nerf_render_time = (time.time() - t0) / n_runs

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            gs_model.render(test_cam, test_look, focal=focal)
    gs_render_time = (time.time() - t0) / n_runs

    print(f"  NeRF: {nerf_render_time*1000:.1f} ms/帧")
    print(f"  3DGS: {gs_render_time*1000:.1f} ms/帧")
    if gs_render_time > 0:
        speedup = nerf_render_time / gs_render_time
        print(f"  → 3DGS 快 {speedup:.1f}x" if speedup > 1 else f"  → NeRF 快 {1/speedup:.1f}x")

    # --- 多视角一致性 ---
    print(f"\n📊 多视角渲染一致性:")
    angles = [0, 60, 120, 180, 240, 300]
    for angle_deg in [0, 90, 180]:
        angle = math.radians(angle_deg)
        cam = torch.tensor([
            4.0 * math.cos(angle), 1.0, 4.0 * math.sin(angle)
        ], device=device)

        with torch.no_grad():
            nerf_img, _ = nerf_model.render_image(cam, test_look, focal=focal)
            gs_img = gs_model.render(cam, test_look, focal=focal)

        print(f"  角度 {angle_deg:3d}° | "
              f"NeRF: mean={nerf_img.mean():.3f}, std={nerf_img.std():.3f} | "
              f"3DGS: mean={gs_img.mean():.3f}, std={gs_img.std():.3f}")

    # ====================================================================
    # 总结
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("  总结")
    print("=" * 70)
    print(f"""
    NeRF 优势:
      ✅ 紧凑表示 (只有 MLP 权重，{nerf_params:,} 参数)
      ✅ 连续表示，任意分辨率渲染
      ✅ 理论优雅 (体渲染 + 位置编码)
      ✅ 更好的泛化能力

    3DGS 优势:
      ✅ 渲染速度快 (光栅化 vs 体渲染)
      ✅ 训练速度快 (直接优化参数 vs MLP)
      ✅ 显式表示，易于编辑 (移动/删除/添加高斯)
      ✅ 支持实时渲染 (100+ FPS)

    选择建议:
      - 需要紧凑表示和理论优雅 → NeRF
      - 需要实时渲染和可编辑性 → 3DGS
      - 实际应用中 3DGS 已逐渐成为主流选择

    发展趋势 (2024-2025):
      - 3DGS 的各种改进: 反走样、压缩、动态场景
      - NeRF 与 3DGS 的融合: 用 NeRF 初始化 3DGS
      - 生成式 3D: 扩散模型 + 3DGS (DreamGaussian 等)
    """)


if __name__ == '__main__':
    compare_methods()
