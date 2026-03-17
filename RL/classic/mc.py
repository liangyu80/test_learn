"""
蒙特卡洛方法 (Monte Carlo Methods) —— 基于完整序列的策略学习

核心思想: 不需要环境模型, 通过采样完整 episode 来估计 Q(s,a)

┌─────────────────────────────────────────────────────────┐
│ 1. 生成完整 episode: s₀,a₀,r₁,s₁,a₁,r₂,...,s_T       │
│ 2. 对 episode 中每个 (s,a) 计算回报:                    │
│    G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...       │
│ 3. 更新 Q(s,a):                                         │
│    - First-Visit MC: 只在 (s,a) 首次出现时更新           │
│    - Every-Visit MC: 每次出现都更新                      │
│ 4. 策略改进: ε-贪心策略                                  │
│    π(s) = argmax_a Q(s,a) (概率 1-ε)                    │
│          = 随机动作        (概率 ε)                      │
└─────────────────────────────────────────────────────────┘

关键优势:
  - 无需环境模型 (model-free)
  - 无偏估计 (使用真实回报, 不用 bootstrap)
  - 可以用于非马尔可夫环境

关键劣势:
  - 必须等 episode 结束才能更新 (高方差, 低效率)
  - 仅适用于有终止状态的 episodic 任务
"""

import random
from typing import Dict, List, Tuple
from collections import defaultdict
from env import GridWorld, GridWorldConfig, ACTIONS, ACTION_NAMES


# ==============================================================================
# ε-贪心策略
# ==============================================================================

def epsilon_greedy_action(Q: Dict, state: Tuple[int, int],
                          epsilon: float) -> int:
    """
    ε-贪心策略选择动作。

    以 1-ε 的概率选择 Q 值最大的动作 (exploitation)
    以 ε 的概率随机选择动作 (exploration)
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        # 选择 Q 值最大的动作
        q_values = [Q.get((state, a), 0.0) for a in ACTIONS]
        max_q = max(q_values)
        # 如果有多个最大值, 随机选一个 (打破平局)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)


# ==============================================================================
# 生成 Episode
# ==============================================================================

def generate_episode(env: GridWorld, Q: Dict, epsilon: float,
                     max_steps: int = 200) -> List[Tuple]:
    """
    使用 ε-贪心策略生成一个完整 episode。

    Returns:
        episode: [(state, action, reward), ...] 的列表
    """
    episode = []
    state = env.reset()

    for _ in range(max_steps):
        action = epsilon_greedy_action(Q, state, epsilon)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break

    return episode


# ==============================================================================
# First-Visit MC (首次访问蒙特卡洛)
# ==============================================================================

def first_visit_mc(env: GridWorld, num_episodes: int = 5000,
                   gamma: float = 0.99, epsilon: float = 0.1,
                   epsilon_decay: float = 0.999,
                   verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    首次访问蒙特卡洛控制 (First-Visit MC Control)。

    对于每个 episode 中首次出现的 (s,a) 对:
      G_t = Σ_{k=0}^{T-t-1} γ^k · r_{t+k+1}
      Q(s,a) ← average(所有首次访问的 G_t)

    使用增量更新公式:
      N(s,a) += 1
      Q(s,a) += (G - Q(s,a)) / N(s,a)

    Returns:
        Q:      动作价值函数 {(state, action): value}
        policy: 贪心策略 {state: action}
    """
    if verbose:
        print("=" * 50)
        print("首次访问蒙特卡洛 (First-Visit MC)")
        print("=" * 50)

    Q = defaultdict(float)      # Q(s,a) 初始化为 0
    N = defaultdict(int)         # 访问计数 N(s,a)
    eps = epsilon

    for ep in range(num_episodes):
        # 生成完整 episode
        episode = generate_episode(env, Q, eps)

        # 从后向前计算回报 G
        G = 0.0
        visited = set()  # 记录已访问的 (s,a) 对

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            sa = (state, action)

            # 首次访问: 仅在 (s,a) 第一次出现时更新
            if sa not in visited:
                visited.add(sa)
                N[sa] += 1
                # 增量平均: Q(s,a) ← Q(s,a) + (G - Q(s,a)) / N(s,a)
                Q[sa] += (G - Q[sa]) / N[sa]

        # ε 衰减
        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 1000 == 0:
            avg_return = _evaluate_policy(env, Q, num_eval=20, gamma=gamma)
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg_return:.3f}")

    # 提取贪心策略
    policy = _extract_greedy_policy(Q, env)

    if verbose:
        print(f"\n✓ First-Visit MC 训练完成!")

    return dict(Q), policy


# ==============================================================================
# Every-Visit MC (每次访问蒙特卡洛)
# ==============================================================================

def every_visit_mc(env: GridWorld, num_episodes: int = 5000,
                   gamma: float = 0.99, epsilon: float = 0.1,
                   epsilon_decay: float = 0.999,
                   verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    每次访问蒙特卡洛控制 (Every-Visit MC Control)。

    与 First-Visit MC 唯一的区别:
    每次 (s,a) 出现都更新 (不管是不是首次)

    优点: 利用更多数据, 方差可能更低
    缺点: 估计有偏 (但渐近无偏)
    """
    if verbose:
        print("=" * 50)
        print("每次访问蒙特卡洛 (Every-Visit MC)")
        print("=" * 50)

    Q = defaultdict(float)
    N = defaultdict(int)
    eps = epsilon

    for ep in range(num_episodes):
        episode = generate_episode(env, Q, eps)

        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            sa = (state, action)

            # 每次访问都更新 (不检查是否首次)
            N[sa] += 1
            Q[sa] += (G - Q[sa]) / N[sa]

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 1000 == 0:
            avg_return = _evaluate_policy(env, Q, num_eval=20, gamma=gamma)
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg_return:.3f}")

    policy = _extract_greedy_policy(Q, env)

    if verbose:
        print(f"\n✓ Every-Visit MC 训练完成!")

    return dict(Q), policy


# ==============================================================================
# MC with Exploring Starts (探索性初始化)
# ==============================================================================

def mc_exploring_starts(env: GridWorld, num_episodes: int = 5000,
                        gamma: float = 0.99,
                        verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    带探索性初始化的蒙特卡洛 (MC with Exploring Starts)。

    解决探索问题的另一种方式: 每个 episode 的起始 (s₀, a₀) 均匀随机选取,
    确保所有 (s,a) 对都有机会被访问。之后使用纯贪心策略 (无需 ε-贪心)。

    注意: 需要能控制初始状态, 实际中不总是可行。
    """
    if verbose:
        print("=" * 50)
        print("探索性初始化蒙特卡洛 (MC Exploring Starts)")
        print("=" * 50)

    Q = defaultdict(float)
    N = defaultdict(int)
    # 初始策略: 随机
    policy = {s: random.choice(ACTIONS) for s in env.get_all_states()}

    all_states = env.get_all_states()

    for ep in range(num_episodes):
        # 随机选择初始状态和动作
        start_state = random.choice(all_states)
        start_action = random.choice(ACTIONS)

        # 从随机初始状态开始生成 episode
        env.state = start_state
        episode = [(start_state, start_action, None)]

        # 执行第一步
        next_state, reward, done = env.step(start_action)
        episode[0] = (start_state, start_action, reward)

        if not done:
            state = next_state
            for _ in range(200):
                # 第一步之后使用贪心策略
                action = policy.get(state, 0)
                next_state, reward, done = env.step(action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break

        # First-Visit MC 更新
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            sa = (state, action)

            if sa not in visited:
                visited.add(sa)
                N[sa] += 1
                Q[sa] += (G - Q[sa]) / N[sa]

                # 策略改进: 贪心
                q_vals = [Q.get((state, a), 0.0) for a in ACTIONS]
                policy[state] = ACTIONS[q_vals.index(max(q_vals))]

        if verbose and (ep + 1) % 1000 == 0:
            avg_return = _evaluate_policy(env, Q, num_eval=20, gamma=gamma)
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"平均回报={avg_return:.3f}")

    if verbose:
        print(f"\n✓ MC Exploring Starts 训练完成!")

    return dict(Q), policy


# ==============================================================================
# 辅助函数
# ==============================================================================

def _extract_greedy_policy(Q: Dict, env: GridWorld) -> Dict:
    """从 Q 函数提取贪心策略。"""
    policy = {}
    for s in env.get_all_states():
        q_values = [Q.get((s, a), 0.0) for a in ACTIONS]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        policy[s] = random.choice(best_actions)
    return policy


def _evaluate_policy(env: GridWorld, Q: Dict, num_eval: int = 20,
                     gamma: float = 0.99) -> float:
    """评估当前贪心策略的平均回报。"""
    total_return = 0.0
    for _ in range(num_eval):
        state = env.reset()
        G = 0.0
        discount = 1.0
        for _ in range(200):
            q_values = [Q.get((state, a), 0.0) for a in ACTIONS]
            action = ACTIONS[q_values.index(max(q_values))]
            next_state, reward, done = env.step(action)
            G += discount * reward
            discount *= gamma
            state = next_state
            if done:
                break
        total_return += G
    return total_return / num_eval


def q_to_v(Q: Dict, env: GridWorld) -> Dict:
    """从 Q(s,a) 提取 V(s) = max_a Q(s,a)。"""
    V = {}
    for s in env.get_all_states():
        q_vals = [Q.get((s, a), 0.0) for a in ACTIONS]
        V[s] = max(q_vals)
    return V


# ==============================================================================
# 演示
# ==============================================================================

def demo_mc():
    """蒙特卡洛方法演示。"""
    print("=" * 60)
    print("  蒙特卡洛方法 (MC) 演示")
    print("=" * 60)

    # 使用确定性环境 (MC 在确定性环境也能工作, 因为它是 model-free)
    config = GridWorldConfig(stochastic=False)
    env = GridWorld(config)
    print("\n地图:")
    print(env.render())

    # First-Visit MC
    print()
    Q_fv, policy_fv = first_visit_mc(
        env, num_episodes=5000, gamma=config.gamma, epsilon=0.2)
    V_fv = q_to_v(Q_fv, env)
    print("\nFirst-Visit MC - 最优策略:")
    print(env.render(policy=policy_fv))
    print("\nFirst-Visit MC - 价值函数:")
    print(env.render_values(V_fv))

    # Every-Visit MC
    print()
    Q_ev, policy_ev = every_visit_mc(
        env, num_episodes=5000, gamma=config.gamma, epsilon=0.2)
    V_ev = q_to_v(Q_ev, env)
    print("\nEvery-Visit MC - 最优策略:")
    print(env.render(policy=policy_ev))

    # MC Exploring Starts
    print()
    Q_es, policy_es = mc_exploring_starts(
        env, num_episodes=5000, gamma=config.gamma)
    V_es = q_to_v(Q_es, env)
    print("\nMC Exploring Starts - 最优策略:")
    print(env.render(policy=policy_es))

    # 对比
    print("\n" + "─" * 60)
    print("MC 方法对比:")
    print("─" * 60)
    avg_fv = _evaluate_policy(env, Q_fv, num_eval=50, gamma=config.gamma)
    avg_ev = _evaluate_policy(env, Q_ev, num_eval=50, gamma=config.gamma)
    avg_es = _evaluate_policy(env, Q_es, num_eval=50, gamma=config.gamma)
    print(f"  First-Visit MC:       平均回报 = {avg_fv:.3f}")
    print(f"  Every-Visit MC:       平均回报 = {avg_ev:.3f}")
    print(f"  MC Exploring Starts:  平均回报 = {avg_es:.3f}")

    print("\n[MC 演示完成]")

    return {
        'first_visit': {'Q': Q_fv, 'policy': policy_fv},
        'every_visit': {'Q': Q_ev, 'policy': policy_ev},
        'exploring_starts': {'Q': Q_es, 'policy': policy_es},
    }


if __name__ == '__main__':
    demo_mc()
