"""
时序差分学习 (Temporal Difference Learning)

核心思想: 结合 MC (采样) 和 DP (bootstrap) 的优势
  - 不需要完整 episode (每步都能更新)
  - 不需要环境模型 (model-free)

┌─────────────────────────────────────────────────────────┐
│ TD(0) 更新:                                             │
│   V(s) ← V(s) + α · [r + γ·V(s') - V(s)]              │
│                       ^^^^^^^^^^^^^^^^                  │
│                       TD target    TD error (δ)         │
│                                                         │
│ 对比:                                                    │
│   MC:   V(s) ← V(s) + α · [G_t - V(s)]    (真实回报)   │
│   TD:   V(s) ← V(s) + α · [r+γV(s') - V(s)] (bootstrap)│
│   DP:   V(s) = Σ P(s'|s,a)·[r+γV(s')]      (需要模型)  │
└─────────────────────────────────────────────────────────┘

本文件实现:
  1. SARSA        —— On-policy TD 控制
  2. Q-Learning   —— Off-policy TD 控制
  3. Expected SARSA—— SARSA 的低方差版本
  4. n-step TD    —— 多步 TD (连接 TD(0) 和 MC)
  5. TD(λ)        —— 资格迹 (eligibility traces) 方法
"""

import random
from typing import Dict, List, Tuple
from collections import defaultdict
from env import GridWorld, GridWorldConfig, ACTIONS, ACTION_NAMES


# ==============================================================================
# ε-贪心策略
# ==============================================================================

def epsilon_greedy(Q: Dict, state: Tuple[int, int], epsilon: float) -> int:
    """ε-贪心动作选择。"""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_values = [Q.get((state, a), 0.0) for a in ACTIONS]
    max_q = max(q_values)
    best = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
    return random.choice(best)


# ==============================================================================
# SARSA —— On-policy TD 控制
# ==============================================================================

def sarsa(env: GridWorld, num_episodes: int = 3000,
          gamma: float = 0.99, alpha: float = 0.1,
          epsilon: float = 0.1, epsilon_decay: float = 0.999,
          verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    SARSA (State-Action-Reward-State-Action)

    On-policy TD 控制:
      选择 a ~ π(s)     ← 用 ε-贪心选动作
      执行 a, 得到 r, s'
      选择 a' ~ π(s')   ← 用同一个 ε-贪心选下一步动作
      更新: Q(s,a) ← Q(s,a) + α · [r + γ·Q(s',a') - Q(s,a)]

    命名来源: 更新需要 5 个元素 (S, A, R, S', A')

    特点:
      - On-policy: 评估和改进的是同一个策略 (ε-greedy)
      - 因此学到的 Q 值反映了 ε-贪心的行为 (包含探索)
      - 在随机/危险环境中更保守 (会避开悬崖边)
    """
    if verbose:
        print("=" * 50)
        print("SARSA (On-policy TD)")
        print("=" * 50)

    Q = defaultdict(float)
    eps = epsilon

    for ep in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, eps)
        total_reward = 0

        for _ in range(200):
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                # 终止状态: Q(s',a') = 0
                Q[(state, action)] += alpha * (
                    reward - Q[(state, action)])
                break

            # 选择下一步动作 (SARSA 的关键: 用 ε-贪心)
            next_action = epsilon_greedy(Q, next_state, eps)

            # TD 更新: Q(s,a) ← Q(s,a) + α·[r + γ·Q(s',a') - Q(s,a)]
            td_target = reward + gamma * Q[(next_state, next_action)]
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            action = next_action

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = _evaluate(env, Q, gamma=gamma)
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg:.3f}")

    policy = _extract_policy(Q, env)
    if verbose:
        print(f"\n✓ SARSA 训练完成!")
    return dict(Q), policy


# ==============================================================================
# Q-Learning —— Off-policy TD 控制
# ==============================================================================

def q_learning(env: GridWorld, num_episodes: int = 3000,
               gamma: float = 0.99, alpha: float = 0.1,
               epsilon: float = 0.1, epsilon_decay: float = 0.999,
               verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Q-Learning (Watkins, 1989)

    Off-policy TD 控制:
      选择 a ~ ε-greedy(Q, s)    ← 行为策略 (behavior policy)
      执行 a, 得到 r, s'
      更新: Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a') - Q(s,a)]
                                          ^^^^^^^^^^^^^^^^^
                                          关键区别: 用 max (贪心)

    与 SARSA 的区别:
      - SARSA:      Q(s,a) ← ... + α·[r + γ·Q(s', a')]  a'~ε-greedy
      - Q-Learning: Q(s,a) ← ... + α·[r + γ·max Q(s',·)] 直接用 max

    特点:
      - Off-policy: 行为策略 (ε-greedy) ≠ 目标策略 (greedy)
      - 直接学习最优 Q* (不受探索策略影响)
      - 更激进 (不考虑探索的风险)
    """
    if verbose:
        print("=" * 50)
        print("Q-Learning (Off-policy TD)")
        print("=" * 50)

    Q = defaultdict(float)
    eps = epsilon

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(200):
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                Q[(state, action)] += alpha * (
                    reward - Q[(state, action)])
                break

            # Q-Learning 更新: 用 max_a' Q(s',a')
            max_q_next = max(Q.get((next_state, a), 0.0) for a in ACTIONS)
            td_target = reward + gamma * max_q_next
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = _evaluate(env, Q, gamma=gamma)
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg:.3f}")

    policy = _extract_policy(Q, env)
    if verbose:
        print(f"\n✓ Q-Learning 训练完成!")
    return dict(Q), policy


# ==============================================================================
# Expected SARSA
# ==============================================================================

def expected_sarsa(env: GridWorld, num_episodes: int = 3000,
                   gamma: float = 0.99, alpha: float = 0.1,
                   epsilon: float = 0.1, epsilon_decay: float = 0.999,
                   verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Expected SARSA

    使用 ε-贪心策略下的期望 Q 值来更新:
      Q(s,a) ← Q(s,a) + α · [r + γ·E_π[Q(s',a')] - Q(s,a)]

    其中:
      E_π[Q(s',a')] = Σ_a' π(a'|s') · Q(s',a')
                     = (1-ε)·max_a' Q(s',a') + (ε/|A|)·Σ_a' Q(s',a')

    优势: 比 SARSA 方差更低 (不依赖于采样的 a'), 性能介于 SARSA 和 Q-Learning 之间
    """
    if verbose:
        print("=" * 50)
        print("Expected SARSA")
        print("=" * 50)

    Q = defaultdict(float)
    eps = epsilon
    n_actions = len(ACTIONS)

    for ep in range(num_episodes):
        state = env.reset()

        for _ in range(200):
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, done = env.step(action)

            if done:
                Q[(state, action)] += alpha * (
                    reward - Q[(state, action)])
                break

            # 计算 E_π[Q(s', a')]
            q_next = [Q.get((next_state, a), 0.0) for a in ACTIONS]
            max_q = max(q_next)
            # ε-贪心期望: (1-ε)·max + ε/|A|·sum
            expected_q = (1 - eps) * max_q + eps * sum(q_next) / n_actions

            td_target = reward + gamma * expected_q
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state = next_state

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = _evaluate(env, Q, gamma=gamma)
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg:.3f}")

    policy = _extract_policy(Q, env)
    if verbose:
        print(f"\n✓ Expected SARSA 训练完成!")
    return dict(Q), policy


# ==============================================================================
# n-step TD (多步时序差分)
# ==============================================================================

def n_step_sarsa(env: GridWorld, n: int = 3, num_episodes: int = 3000,
                 gamma: float = 0.99, alpha: float = 0.1,
                 epsilon: float = 0.1, epsilon_decay: float = 0.999,
                 verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    n-step SARSA (多步 TD)

    连接 TD(0) 和 MC 的桥梁:
      n=1: 退化为 SARSA (TD(0))
      n=∞: 退化为 MC

    n-step 回报:
      G_t:t+n = r_{t+1} + γ·r_{t+2} + ... + γ^{n-1}·r_{t+n} + γ^n·Q(s_{t+n}, a_{t+n})

    更新:
      Q(s_t, a_t) ← Q(s_t, a_t) + α · [G_t:t+n - Q(s_t, a_t)]

    n 越大 → 方差越大, 偏差越小 (更接近 MC)
    n 越小 → 方差越小, 偏差越大 (更接近 TD)
    """
    if verbose:
        print("=" * 50)
        print(f"{n}-step SARSA (多步 TD)")
        print("=" * 50)

    Q = defaultdict(float)
    eps = epsilon

    for ep in range(num_episodes):
        # 存储完整的 episode 轨迹
        states = []
        actions = []
        rewards = [0.0]  # r_0 不使用, 占位

        state = env.reset()
        action = epsilon_greedy(Q, state, eps)
        states.append(state)
        actions.append(action)

        T = 10**9  # episode 终止时刻 (用大整数代替 inf)
        t = 0

        while True:
            if t < T:
                next_state, reward, done = env.step(actions[t])
                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t + 1
                else:
                    next_action = epsilon_greedy(Q, next_state, eps)
                    actions.append(next_action)

            # τ 是当前要更新的时刻
            tau = t - n + 1

            if tau >= 0:
                # 计算 n-step 回报 G
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]

                # 如果 episode 还没结束, 加上 bootstrap 项
                if tau + n < T:
                    G += (gamma ** n) * Q[(states[tau + n], actions[tau + n])]

                # 更新 Q
                sa = (states[tau], actions[tau])
                Q[sa] += alpha * (G - Q[sa])

            if tau == T - 1:
                break
            t += 1

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = _evaluate(env, Q, gamma=gamma)
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg:.3f}")

    policy = _extract_policy(Q, env)
    if verbose:
        print(f"\n✓ {n}-step SARSA 训练完成!")
    return dict(Q), policy


# ==============================================================================
# TD(λ) —— 资格迹方法
# ==============================================================================

def sarsa_lambda(env: GridWorld, lam: float = 0.8,
                 num_episodes: int = 3000,
                 gamma: float = 0.99, alpha: float = 0.1,
                 epsilon: float = 0.1, epsilon_decay: float = 0.999,
                 verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    SARSA(λ) —— 带资格迹的 SARSA

    资格迹 (Eligibility Trace) 的核心思想:
      - 维护一个 "迹" e(s,a), 记录每个 (s,a) 对的 "功劳/责任"
      - 最近访问的 (s,a) 迹值大, 久远的迹值小
      - TD error 按迹值大小分配给所有 (s,a)

    更新规则:
      δ = r + γ·Q(s',a') - Q(s,a)        # TD error
      e(s,a) ← e(s,a) + 1                # 累积迹 (accumulating trace)
      对所有 (s,a):
        Q(s,a) ← Q(s,a) + α·δ·e(s,a)    # 按迹值更新
        e(s,a) ← γ·λ·e(s,a)              # 迹衰减

    λ 的作用:
      λ=0: 退化为 SARSA (TD(0)), 只更新当前 (s,a)
      λ=1: 接近 MC, 更新整条轨迹
      0<λ<1: 折中, 近期 (s,a) 获得更大更新

    等价于:
      G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} · G_t:t+n  (λ-回报)
    """
    if verbose:
        print("=" * 50)
        print(f"SARSA(λ={lam}) (资格迹)")
        print("=" * 50)

    Q = defaultdict(float)
    eps = epsilon

    for ep in range(num_episodes):
        # 每个 episode 开始时清零资格迹
        E = defaultdict(float)  # 资格迹 e(s,a)

        state = env.reset()
        action = epsilon_greedy(Q, state, eps)

        for _ in range(200):
            next_state, reward, done = env.step(action)

            if done:
                # 终止: TD error
                delta = reward - Q[(state, action)]
                E[(state, action)] += 1  # 累积迹

                # 更新所有有迹的 (s,a) 对
                for sa in list(E.keys()):
                    Q[sa] += alpha * delta * E[sa]
                break

            next_action = epsilon_greedy(Q, next_state, eps)

            # TD error: δ = r + γ·Q(s',a') - Q(s,a)
            delta = (reward + gamma * Q[(next_state, next_action)]
                     - Q[(state, action)])

            # 累积迹: e(s,a) += 1
            E[(state, action)] += 1

            # 更新所有有迹的 (s,a) 对, 并衰减迹
            keys_to_delete = []
            for sa in list(E.keys()):
                Q[sa] += alpha * delta * E[sa]
                E[sa] *= gamma * lam
                # 清除过小的迹 (节省内存)
                if E[sa] < 1e-6:
                    keys_to_delete.append(sa)
            for sa in keys_to_delete:
                del E[sa]

            state = next_state
            action = next_action

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = _evaluate(env, Q, gamma=gamma)
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg:.3f}")

    policy = _extract_policy(Q, env)
    if verbose:
        print(f"\n✓ SARSA(λ={lam}) 训练完成!")
    return dict(Q), policy


# ==============================================================================
# Watkins's Q(λ) —— Off-policy 资格迹
# ==============================================================================

def q_lambda(env: GridWorld, lam: float = 0.8,
             num_episodes: int = 3000,
             gamma: float = 0.99, alpha: float = 0.1,
             epsilon: float = 0.1, epsilon_decay: float = 0.999,
             verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Watkins's Q(λ) —— Off-policy 资格迹

    与 SARSA(λ) 的区别:
      - 当选择的动作是贪心动作时, 正常累积迹
      - 当选择的动作是探索动作时, 截断迹 (清零)

    这保证了 off-policy 的正确性: 非贪心动作不应该传播 TD error
    """
    if verbose:
        print("=" * 50)
        print(f"Watkins's Q(λ={lam})")
        print("=" * 50)

    Q = defaultdict(float)
    eps = epsilon

    for ep in range(num_episodes):
        E = defaultdict(float)
        state = env.reset()
        action = epsilon_greedy(Q, state, eps)

        for _ in range(200):
            next_state, reward, done = env.step(action)

            # 贪心动作
            q_next = [Q.get((next_state, a), 0.0) for a in ACTIONS]
            greedy_action = ACTIONS[q_next.index(max(q_next))]

            if done:
                delta = reward - Q[(state, action)]
                E[(state, action)] += 1
                for sa in list(E.keys()):
                    Q[sa] += alpha * delta * E[sa]
                break

            next_action = epsilon_greedy(Q, next_state, eps)

            # TD error (使用 max, 类似 Q-Learning)
            delta = reward + gamma * max(q_next) - Q[(state, action)]
            E[(state, action)] += 1

            # 更新所有有迹的 (s,a)
            for sa in list(E.keys()):
                Q[sa] += alpha * delta * E[sa]

            # 如果下一步动作是贪心的 → 正常衰减; 否则 → 截断迹
            if next_action == greedy_action:
                for sa in list(E.keys()):
                    E[sa] *= gamma * lam
                    if E[sa] < 1e-6:
                        del E[sa]
            else:
                E.clear()  # 截断: 非贪心动作, 清零所有迹

            state = next_state
            action = next_action

        eps = max(0.01, eps * epsilon_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = _evaluate(env, Q, gamma=gamma)
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"ε={eps:.3f}, 平均回报={avg:.3f}")

    policy = _extract_policy(Q, env)
    if verbose:
        print(f"\n✓ Watkins's Q(λ={lam}) 训练完成!")
    return dict(Q), policy


# ==============================================================================
# 辅助函数
# ==============================================================================

def _extract_policy(Q: Dict, env: GridWorld) -> Dict:
    """从 Q 函数提取贪心策略。"""
    policy = {}
    for s in env.get_all_states():
        q_vals = [Q.get((s, a), 0.0) for a in ACTIONS]
        max_q = max(q_vals)
        best = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
        policy[s] = random.choice(best)
    return policy


def _evaluate(env: GridWorld, Q: Dict, num_eval: int = 20,
              gamma: float = 0.99) -> float:
    """评估贪心策略的平均回报。"""
    total = 0.0
    for _ in range(num_eval):
        state = env.reset()
        G = 0.0
        discount = 1.0
        for _ in range(200):
            q_vals = [Q.get((state, a), 0.0) for a in ACTIONS]
            action = ACTIONS[q_vals.index(max(q_vals))]
            next_state, reward, done = env.step(action)
            G += discount * reward
            discount *= gamma
            state = next_state
            if done:
                break
        total += G
    return total / num_eval


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

def demo_td():
    """TD 学习演示: SARSA, Q-Learning, Expected SARSA, n-step, TD(λ)"""
    print("=" * 60)
    print("  时序差分学习 (TD) 演示")
    print("=" * 60)

    config = GridWorldConfig(stochastic=False)
    env = GridWorld(config)
    print("\n地图:")
    print(env.render())

    results = {}

    # 1. SARSA
    print()
    Q_s, policy_s = sarsa(env, num_episodes=3000, gamma=config.gamma)
    print("\nSARSA - 最优策略:")
    print(env.render(policy=policy_s))
    results['SARSA'] = (Q_s, policy_s)

    # 2. Q-Learning
    print()
    Q_q, policy_q = q_learning(env, num_episodes=3000, gamma=config.gamma)
    print("\nQ-Learning - 最优策略:")
    print(env.render(policy=policy_q))
    results['Q-Learning'] = (Q_q, policy_q)

    # 3. Expected SARSA
    print()
    Q_e, policy_e = expected_sarsa(env, num_episodes=3000, gamma=config.gamma)
    print("\nExpected SARSA - 最优策略:")
    print(env.render(policy=policy_e))
    results['Expected SARSA'] = (Q_e, policy_e)

    # 4. n-step SARSA
    print()
    Q_n, policy_n = n_step_sarsa(env, n=3, num_episodes=3000, gamma=config.gamma)
    print(f"\n3-step SARSA - 最优策略:")
    print(env.render(policy=policy_n))
    results['3-step SARSA'] = (Q_n, policy_n)

    # 5. SARSA(λ)
    print()
    Q_l, policy_l = sarsa_lambda(env, lam=0.8, num_episodes=3000, gamma=config.gamma)
    print(f"\nSARSA(λ=0.8) - 最优策略:")
    print(env.render(policy=policy_l))
    results['SARSA(λ=0.8)'] = (Q_l, policy_l)

    # 6. Q(λ)
    print()
    Q_ql, policy_ql = q_lambda(env, lam=0.8, num_episodes=3000, gamma=config.gamma)
    print(f"\nQ(λ=0.8) - 最优策略:")
    print(env.render(policy=policy_ql))
    results['Q(λ=0.8)'] = (Q_ql, policy_ql)

    # 对比
    print("\n" + "─" * 60)
    print("TD 方法对比:")
    print("─" * 60)
    for name, (Q, _) in results.items():
        avg = _evaluate(env, Q, num_eval=50, gamma=config.gamma)
        print(f"  {name:20s}: 平均回报 = {avg:.3f}")

    print("\n[TD 演示完成]")
    return results


if __name__ == '__main__':
    demo_td()
