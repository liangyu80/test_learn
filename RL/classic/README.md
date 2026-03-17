# 经典表格型强化学习 (Classic Tabular RL)

在引入神经网络之前，先掌握 RL 的底层逻辑。本项目实现了强化学习的三大经典方法族：**动态规划**、**蒙特卡洛**、**时序差分**，全部使用表格型（tabular）方法在 GridWorld 环境上进行演示。

## 环境: GridWorld

```
┌───┬───┬───┬───┬───┐
│ S │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │ ▓ │   │ ▓ │   │
├───┼───┼───┼───┼───┤
│   │   │   │ ✖ │   │
├───┼───┼───┼───┼───┤
│   │ ▓ │ ✖ │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │ G │
└───┴───┴───┴───┴───┘
S=起点  G=终点(+1)  ▓=墙壁  ✖=陷阱(-1)
```

- **状态空间**: 网格坐标 (row, col)
- **动作空间**: 上(0)、下(1)、左(2)、右(3)
- **奖励**: 到达终点 +1, 掉入陷阱 -1, 每步 -0.01
- **支持确定性/随机性转移** (80/10/10 滑动)

## 算法总览

### 1. 动态规划 (DP) — `dp.py`

**前提**: 需要完整的环境模型 P(s'|s,a)

| 算法 | 核心思想 |
|------|----------|
| **策略迭代** (Policy Iteration) | 交替执行策略评估 + 策略改进，保证收敛到 π* |
| **价值迭代** (Value Iteration) | 直接迭代贝尔曼最优方程 V* = max_a Σ P·[R+γV*] |

```
策略迭代: π₀ → V^π₀ → π₁ → V^π₁ → ... → π* → V*
价值迭代: V₀ → V₁ → V₂ → ... → V* → π*
```

### 2. 蒙特卡洛方法 (MC) — `mc.py`

**核心**: 通过完整 episode 采样估计 Q(s,a)

| 算法 | 特点 |
|------|------|
| **First-Visit MC** | 只在 (s,a) 首次出现时更新，无偏估计 |
| **Every-Visit MC** | 每次出现都更新，数据利用率更高 |
| **MC Exploring Starts** | 随机初始化起点，不需要 ε-贪心 |

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...  (真实回报)
Q(s,a) ← Q(s,a) + (G - Q(s,a)) / N(s,a)        (增量平均)
```

### 3. 时序差分学习 (TD) — `td.py`

**核心**: 每步都能更新 (bootstrap)，不需要等 episode 结束

| 算法 | 类型 | 更新目标 |
|------|------|----------|
| **SARSA** | On-policy | r + γ·Q(s', a'), a'~ε-greedy |
| **Q-Learning** | Off-policy | r + γ·max_a' Q(s', a') |
| **Expected SARSA** | 混合 | r + γ·E_π[Q(s', ·)] |
| **n-step SARSA** | 多步 TD | r₁ + γr₂ + ... + γⁿQ(sₙ, aₙ) |
| **SARSA(λ)** | 资格迹 | TD error × 资格迹 e(s,a) |
| **Watkins's Q(λ)** | Off-policy 迹 | 截断迹的 Q-Learning |

```
TD(0):   V(s) ← V(s) + α·[r + γ·V(s') - V(s)]
MC:      V(s) ← V(s) + α·[G_t - V(s)]
n-step:  V(s) ← V(s) + α·[G_{t:t+n} - V(s)]    (桥接 TD 和 MC)
TD(λ):   等价于 λ-回报 = (1-λ)Σ λⁿ⁻¹ G_{t:t+n}  (所有 n 的加权平均)
```

## 方法对比

| 维度 | DP | MC | TD |
|------|-----|-----|-----|
| 需要模型 | ✅ 是 | ❌ 否 | ❌ 否 |
| 需要完整 episode | ❌ 否 | ✅ 是 | ❌ 否 |
| Bootstrap | ✅ 是 | ❌ 否 | ✅ 是 |
| 偏差 (Bias) | 无 | 无 | 有 (bootstrap) |
| 方差 (Variance) | 无 | 高 | 低 |
| 收敛速度 | 最快 | 慢 | 中等 |
| 适用场景 | 小规模已知 MDP | episodic 任务 | 通用 |

## 运行

```bash
# 运行单个模块
cd RL/classic
python env.py       # GridWorld 环境演示
python dp.py        # 动态规划演示
python mc.py        # 蒙特卡洛演示
python td.py        # 时序差分演示

# 运行统一对比实验
python train.py
```

## 学习路径

```
DP (已知模型)  →  MC (采样, 无 bootstrap)  →  TD (采样 + bootstrap)
     ↓                    ↓                          ↓
策略/价值迭代      First/Every-Visit MC       SARSA → Q-Learning
                                                     ↓
                                              n-step TD → TD(λ)
                                                     ↓
                                              (神经网络版本: DQN, A2C, PPO...)
```

## 参考

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd Edition
- David Silver, UCL RL Course Lectures 2-5
