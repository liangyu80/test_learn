"""
前沿扩散模型算法 —— 统一对比实验

对比 4 种前沿算法:
    1. Flow Matching (Rectified Flow) —— 学习速度场，无需噪声调度
    2. Consistency Model —— 自一致性约束，1-2 步生成
    3. Shortcut Model —— 自蒸馏，单次训练实现 1 步生成
    4. DiT / DyDiT —— Transformer 架构 + 动态计算优化
"""

import torch
import time
import sys
import os

from flow_matching import FlowMatching, FlowMatchingConfig, train_flow_matching
from consistency_model import ConsistencyModel, ConsistencyConfig, train_consistency_model
from shortcut_model import ShortcutModel, ShortcutConfig, train_shortcut_model
from dynamic_dit import DiffusionTransformer, DyDiTConfig, train_dit, _sample_euler


def run_all_experiments():
    """运行所有前沿扩散模型的对比实验。"""
    print("=" * 70)
    print("  前沿扩散模型算法对比实验")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    data_dim = 200
    results = {}

    # ====================================================================
    # 1. Flow Matching
    # ====================================================================
    print("▶ [1/5] Flow Matching (Rectified Flow)")
    print("-" * 50)
    fm_config = FlowMatchingConfig(data_dim=data_dim, hidden_dim=128, time_emb_dim=32)
    t0 = time.time()
    results['flow_matching'] = train_flow_matching(
        config=fm_config, epochs=50, batch_size=64, device=device)
    results['flow_matching']['time'] = time.time() - t0
    print()

    # ====================================================================
    # 2. Consistency Model
    # ====================================================================
    print("▶ [2/5] Consistency Model")
    print("-" * 50)
    cm_config = ConsistencyConfig(data_dim=data_dim, hidden_dim=128, time_emb_dim=32)
    t0 = time.time()
    results['consistency'] = train_consistency_model(
        config=cm_config, epochs=60, batch_size=64, device=device)
    results['consistency']['time'] = time.time() - t0
    print()

    # ====================================================================
    # 3. Shortcut Model
    # ====================================================================
    print("▶ [3/5] Shortcut Model")
    print("-" * 50)
    sc_config = ShortcutConfig(data_dim=data_dim, hidden_dim=128, time_emb_dim=32)
    t0 = time.time()
    results['shortcut'] = train_shortcut_model(
        config=sc_config, epochs_fm=40, epochs_distill=25,
        batch_size=64, device=device)
    results['shortcut']['time'] = time.time() - t0
    print()

    # ====================================================================
    # 4. DiT (标准)
    # ====================================================================
    print("▶ [4/5] DiT (标准 Diffusion Transformer)")
    print("-" * 50)
    dit_config = DyDiTConfig(data_dim=data_dim, num_tokens=25, token_dim=8,
                              d_model=64, n_heads=4, n_layers=3, d_ff=128,
                              time_emb_dim=32)
    t0 = time.time()
    results['dit'] = train_dit(config=dit_config, dynamic=False, epochs=50,
                                batch_size=64, device=device)
    results['dit']['time'] = time.time() - t0
    print()

    # ====================================================================
    # 5. DyDiT (动态)
    # ====================================================================
    print("▶ [5/5] DyDiT (Dynamic Diffusion Transformer)")
    print("-" * 50)
    t0 = time.time()
    results['dydit'] = train_dit(config=dit_config, dynamic=True, epochs=50,
                                  batch_size=64, device=device)
    results['dydit']['time'] = time.time() - t0
    print()

    # ====================================================================
    # 综合对比
    # ====================================================================
    print("=" * 70)
    print("  综合对比")
    print("=" * 70)

    # 参数量
    print("\n📊 参数量:")
    param_info = {
        'Flow Matching': sum(p.numel() for p in results['flow_matching']['model'].parameters()),
        'Consistency':   sum(p.numel() for p in results['consistency']['model'].online_net.parameters()),
        'Shortcut':      sum(p.numel() for p in results['shortcut']['model'].parameters()),
        'DiT':           sum(p.numel() for p in results['dit']['model'].parameters()),
        'DyDiT':         sum(p.numel() for p in results['dydit']['model'].parameters()),
    }
    for name, params in param_info.items():
        print(f"  {name:<16}: {params:>10,}")

    # 训练时间
    print("\n📊 训练时间:")
    for name, key in [('Flow Matching', 'flow_matching'), ('Consistency', 'consistency'),
                       ('Shortcut', 'shortcut'), ('DiT', 'dit'), ('DyDiT', 'dydit')]:
        print(f"  {name:<16}: {results[key]['time']:>8.2f}s")

    # 最终损失
    print("\n📊 最终训练损失:")
    for name, key in [('Flow Matching', 'flow_matching'), ('Consistency', 'consistency'),
                       ('DiT', 'dit'), ('DyDiT', 'dydit')]:
        loss = results[key]['history']['loss'][-1]
        print(f"  {name:<16}: {loss:.6f}")
    # Shortcut 分两阶段
    fm_loss = results['shortcut']['history']['fm_loss'][-1]
    dist_loss = results['shortcut']['history']['distill_loss'][-1]
    print(f"  {'Shortcut':<16}: FM={fm_loss:.6f}, Distill={dist_loss:.6f}")

    # 采样质量 (不同步数)
    print("\n📊 采样质量 (mean 误差, 越小越好):")
    from flow_matching import _generate_structured_data
    real_data = _generate_structured_data(500, data_dim, device)
    real_mean = real_data.mean().item()

    print(f"  {'方法':<16} {'1步':>8} {'2步':>8} {'10步':>8} {'50步':>8}")
    print("  " + "-" * 50)

    # Flow Matching
    fm_model = results['flow_matching']['model']
    fm_model.eval()
    errs = []
    for steps in [1, 2, 10, 50]:
        s = fm_model.sample(200, num_steps=steps, device=device)
        errs.append(abs(s.mean().item() - real_mean))
    print(f"  {'Flow Matching':<16} {errs[0]:>8.4f} {errs[1]:>8.4f} "
          f"{errs[2]:>8.4f} {errs[3]:>8.4f}")

    # Consistency Model
    cm = results['consistency']['model']
    errs = []
    for steps in [1, 2, 10, 50]:
        s = cm.sample(200, num_steps=min(steps, 10))
        errs.append(abs(s.mean().item() - real_mean))
    print(f"  {'Consistency':<16} {errs[0]:>8.4f} {errs[1]:>8.4f} "
          f"{errs[2]:>8.4f} {errs[3]:>8.4f}")

    # Shortcut Model
    sc_model = results['shortcut']['model']
    sc_model.eval()
    errs = []
    for steps in [1, 2, 10, 50]:
        s = sc_model.sample(200, num_steps=steps, device=device)
        errs.append(abs(s.mean().item() - real_mean))
    print(f"  {'Shortcut':<16} {errs[0]:>8.4f} {errs[1]:>8.4f} "
          f"{errs[2]:>8.4f} {errs[3]:>8.4f}")

    # DiT
    dit_model = results['dit']['model']
    dit_model.eval()
    errs = []
    for steps in [1, 2, 10, 50]:
        s = _sample_euler(dit_model, 200, data_dim, steps, device)
        errs.append(abs(s.mean().item() - real_mean))
    print(f"  {'DiT':<16} {errs[0]:>8.4f} {errs[1]:>8.4f} "
          f"{errs[2]:>8.4f} {errs[3]:>8.4f}")

    # DyDiT
    dydit_model = results['dydit']['model']
    dydit_model.eval()
    errs = []
    for steps in [1, 2, 10, 50]:
        s = _sample_euler(dydit_model, 200, data_dim, steps, device)
        errs.append(abs(s.mean().item() - real_mean))
    print(f"  {'DyDiT':<16} {errs[0]:>8.4f} {errs[1]:>8.4f} "
          f"{errs[2]:>8.4f} {errs[3]:>8.4f}")

    # 总结
    print(f"""
{'=' * 70}
  总结: 前沿扩散模型算法图谱
{'=' * 70}

    ┌─────────────────────────────────────────────────────────────┐
    │                   扩散模型发展路线                           │
    │                                                             │
    │  DDPM (2020) ─→ DDIM (2020) ─→ Flow Matching (2022)        │
    │       │              │               │                      │
    │       │              ↓               ↓                      │
    │       │     Consistency Model    Shortcut Model             │
    │       │         (2023)             (2024)                   │
    │       │                                                     │
    │       ↓                                                     │
    │   U-Net ─────────→ DiT (2023) ──→ DyDiT (2024)            │
    │  (SD 1/2)          (SD3, Sora)    (高效 DiT)               │
    └─────────────────────────────────────────────────────────────┘

    Flow Matching:      最简洁的框架，学习速度场，无需噪声调度
    Consistency Model:  1-2 步高质量生成，EMA 自一致性训练
    Shortcut Model:     单次训练 1 步生成，自蒸馏步长翻倍
    DiT / DyDiT:        Transformer 架构 + 动态计算优化
    """)


if __name__ == '__main__':
    run_all_experiments()
