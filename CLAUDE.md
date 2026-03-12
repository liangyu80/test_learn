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
└── RL/                                  # 强化学习项目
    └── ppo/                             # PPO-RLHF 训练 LLM
        ├── README.md                    # PPO 算法文档（中文）
        ├── model.py                     # 轻量 GPT 模型 (Actor/Critic/Reward)
        ├── ppo_trainer.py               # PPO 训练器 (GAE, Clip, KL)
        └── train.py                     # 主训练脚本 (SFT + PPO)
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

# Run PPO-RLHF training
cd RL/ppo && python train.py
```

## Key Notes for AI Assistants

- Always read existing code before proposing changes
- Prefer editing existing files over creating new ones
- Keep changes minimal and focused on the task at hand
- Do not add unnecessary abstractions or over-engineer solutions
- Update this CLAUDE.md file when new conventions or tooling are introduced
