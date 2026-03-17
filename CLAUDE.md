# CLAUDE.md

This file provides guidance for AI assistants working with the `test_learn` repository.

## Repository Overview

- **Repository**: `liangyu80/test_learn`
- **Language**: Python 3.8+
- **Primary Framework**: PyTorch

## Project Structure

```
test_learn/
├── CLAUDE.md                            # AI assistant guidance
├── speculative_vs_multitoken/           # LLM 加速技术对比项目
│   ├── README.md                        # 项目文档（中文）
│   ├── speculative_decoding.py          # 投机采样实现
│   ├── multitoken_prediction.py         # 多 Token 预测实现
│   └── compare.py                       # 对比实验脚本
├── gated_attention/                     # 门控注意力对比项目
│   ├── README.md                        # 项目文档（中文）
│   ├── model.py                         # GAU / Sigmoid-Gated / Standard 注意力
│   └── train.py                         # 对比训练脚本
├── titans/                              # Titans 长期记忆项目
│   ├── model.py                         # MAC 模型 (Memory As Context)
│   ├── neural_memory.py                 # 神经长期记忆模块 (NMM)
│   └── train.py                         # 训练演示脚本
├── vae_vs_gan/                          # VAE vs GAN 生成模型对比
│   ├── README.md                        # 项目文档（中文）
│   ├── vae.py                           # 变分自编码器 (VAE) 实现
│   ├── gan.py                           # 生成对抗网络 (GAN) 实现
│   └── compare.py                       # 对比实验脚本
├── diffusion/                           # 扩散模型 (DDPM & DDIM)
│   ├── README.md                        # 项目文档（中文）
│   ├── ddpm.py                          # DDPM 实现 (噪声调度 + 去噪网络)
│   ├── ddim.py                          # DDIM 加速采样器
│   └── train.py                         # 训练脚本 + 对比实验
├── diffusion_advanced/                  # 前沿扩散模型算法
│   ├── README.md                        # 项目文档（中文）
│   ├── flow_matching.py                 # Flow Matching (Rectified Flow)
│   ├── consistency_model.py             # Consistency Model (1-2步生成)
│   ├── shortcut_model.py                # Shortcut Model (自蒸馏1步生成)
│   ├── dynamic_dit.py                   # DiT + DyDiT (动态Transformer)
│   └── train.py                         # 统一对比实验
├── mamba_vs_transformer/                # Mamba vs Transformer 混合架构
│   ├── README.md                        # 项目文档（中文）
│   ├── mamba.py                         # Mamba S6 选择性状态空间模型
│   ├── transformer.py                   # 标准 Transformer (对比基线)
│   ├── hybrid.py                        # 混合模型 (Jamba/Alternate/Zamba)
│   └── compare.py                       # 统一对比实验
├── nerf_3dgs/                           # NeRF vs 3DGS 神经3D表示
│   ├── README.md                        # 项目文档（中文）
│   ├── nerf.py                          # NeRF 实现 (位置编码+MLP+体渲染)
│   ├── gaussian_splatting.py            # 3DGS 实现 (高斯点云+光栅化)
│   └── compare.py                       # 对比实验脚本
└── RL/                                  # 强化学习项目
    ├── ppo/                             # PPO-RLHF 训练 LLM
    │   ├── README.md                    # PPO 算法文档（中文）
    │   ├── model.py                     # 轻量 GPT 模型 (Actor/Critic/Reward)
    │   ├── ppo_trainer.py               # PPO 训练器 (GAE, Clip, KL)
    │   └── train.py                     # 主训练脚本 (SFT + PPO)
    ├── grpo/                            # GRPO 训练 LLM (无需 Critic)
    │   ├── README.md                    # GRPO 算法文档（中文）
    │   ├── model.py                     # 轻量 GPT 模型 (~1M 参数)
    │   ├── grpo_trainer.py              # GRPO 训练器 (组采样、相对优势)
    │   └── train.py                     # 主训练脚本 (SFT + GRPO)
    ├── rlvr/                            # RLVR 训练 LLM (可验证奖励)
    │   ├── README.md                    # RLVR 算法文档（中文）
    │   ├── model.py                     # 轻量 GPT 模型 (~1M 参数)
    │   ├── rlvr_trainer.py              # RLVR 训练器 (组采样、验证器、PPO-Clip)
    │   └── train.py                     # 主训练脚本 (SFT + RLVR)
    └── classic/                         # 经典表格型 RL (DP/MC/TD)
        ├── README.md                    # 项目文档（中文）
        ├── env.py                       # GridWorld 环境 (确定性/随机性)
        ├── dp.py                        # 策略迭代 & 价值迭代 (动态规划)
        ├── mc.py                        # 蒙特卡洛方法 (First/Every-Visit)
        ├── td.py                        # SARSA, Q-Learning, n-step TD, TD(λ)
        └── train.py                     # 统一对比实验
```

## Development Workflow

### Branching

- Feature branches should follow the naming convention: `claude/<description>-<id>`
- Develop on feature branches and merge into `main` via pull requests

### Commits

- Write clear, descriptive commit messages
- Keep commits focused on a single logical change

### Git Commands

```bash
# Push a branch
git push -u origin <branch-name>

# Fetch a specific branch
git fetch origin <branch-name>
```

## Code Conventions

- Language: Python, with Chinese comments for educational content
- Deep learning framework: PyTorch
- Code should include detailed inline comments, especially for key algorithms and loss functions
- Each module should have a standalone `demo_*()` function and be runnable as `__main__`

## Build & Run

```bash
# Install dependencies
pip install torch

# Run individual modules
python speculative_vs_multitoken/speculative_decoding.py
python speculative_vs_multitoken/multitoken_prediction.py

# Run comparison experiment
python speculative_vs_multitoken/compare.py

# Run VAE vs GAN comparison
cd vae_vs_gan && python compare.py

# Run PPO-RLHF training
cd RL/ppo && python train.py

# Run GRPO training
cd RL/grpo && python train.py

# Run Gated Attention comparison
cd gated_attention && python train.py

# Run Titans MAC training
cd titans && python train.py

# Run RLVR training
cd RL/rlvr && python train.py

# Run Diffusion model (DDPM + DDIM)
cd diffusion && python train.py

# Run Advanced Diffusion models
cd diffusion_advanced && python train.py

# Run Mamba vs Transformer comparison
cd mamba_vs_transformer && python compare.py

# Run NeRF vs 3DGS comparison
cd nerf_3dgs && python compare.py

# Run Classic Tabular RL comparison
cd RL/classic && python train.py
```

## Key Notes for AI Assistants

- Always read existing code before proposing changes
- Prefer editing existing files over creating new ones
- Keep changes minimal and focused on the task at hand
- Do not add unnecessary abstractions or over-engineer solutions
- Update this CLAUDE.md file when new conventions or tooling are introduced
