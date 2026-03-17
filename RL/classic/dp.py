"""
动态规划 (Dynamic Programming) —— 策略迭代 & 价值迭代

前提: 完整的 MDP 模型已知 (转移概率 P(s'|s,a) 和奖励 R)

两种算法:

1. 策略迭代 (Policy Iteration):
   ┌──────────────────────────────────────────────────────┐
   │  初始化随机策略 π₀                                   │
   │  重复:                                               │
   │    (1) 策略评估: 求解 V^π (贝尔曼期望方程)           │
   │        V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)        │
   │                  [R(s,a,s') + γ·V^π(s')]             │
   │    (2) 策略改进: 贪心更新 π                          │
   │        π'(s) = argmax_a Σ_{s'} P(s'|s,a)            │
   │                [R(s,a,s') + γ·V^π(s')]               │
   │  直到策略不再变化 (保证收敛到最优策略 π*)             │
   └──────────────────────────────────────────────────────┘

2. 价值迭代 (Value Iteration):
   ┌──────────────────────────────────────────────────────┐
   │  初始化 V₀(s) = 0                                   │
   │  重复 (贝尔曼最优方程):                               │
   │    V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)              │
   │                 [R(s,a,s') + γ·V_k(s')]              │
   │  直到收敛 (V 不再变化)                                │
   │  最后提取策略:                                        │
   │    π*(s) = argmax_a Σ_{s'} P(s'|s,a)                │
   │            [R(s,a,s') + γ·V*(s')]                    │
   └──────────────────────────────────────────────────────┘

关键区别:
  - 策略迭代: 每轮完整评估 V^π → 改进策略, 迭代次数少但每轮开销大
  - 价值迭代: 每轮只做一步贝尔曼最优更新, 迭代次数多但每轮开销小
  - 两者都保证收敛到最优策略 π*
"""

from typing import Dict, Tuple
from env import GridWorld, GridWorldConfig, ACTIONS, ACTION_NAMES, ACTION_ARROWS


# ==============================================================================
# 策略评估 (Policy Evaluation)
# ==============================================================================

def policy_evaluation(env: GridWorld, policy: Dict[Tuple[int, int], int],
                      gamma: float = 0.99, theta: float = 1e-6,
                      max_iters: int = 1000) -> Dict[Tuple[int, int], float]:
    """
    策略评估: 给定策略 π, 计算其状态价值函数 V^π。

    使用迭代法求解贝尔曼期望方程:
        V^π(s) = Σ_{s'} P(s'|s, π(s)) · [R + γ·V^π(s')]

    Args:
        env:       GridWorld 环境 (提供转移模型)
        policy:    确定性策略 {state: action}
        gamma:     折扣因子
        theta:     收敛阈值 (max|V_new - V_old| < theta 时停止)
        max_iters: 最大迭代次数
    Returns:
        V: 状态价值函数 {state: value}
    """
    # 初始化 V(s) = 0
    V = {s: 0.0 for s in env.states}

    for iteration in range(max_iters):
        delta = 0  # 本轮最大变化量

        for s in env.get_all_states():
            v_old = V[s]
            action = policy[s]

            # 贝尔曼期望方程: V(s) = Σ P(s'|s,a) · [r + γ·V(s')]
            v_new = 0.0
            for prob, next_state, reward, done in env.get_transitions(s, action):
                if done:
                    v_new += prob * reward
                else:
                    v_new += prob * (reward + gamma * V[next_state])

            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))

        # 收敛判断
        if delta < theta:
            break

    return V


# ==============================================================================
# 策略改进 (Policy Improvement)
# ==============================================================================

def policy_improvement(env: GridWorld, V: Dict[Tuple[int, int], float],
                       gamma: float = 0.99) -> Tuple[Dict, bool]:
    """
    策略改进: 根据 V^π 贪心更新策略。

    π'(s) = argmax_a Σ_{s'} P(s'|s,a) · [R + γ·V^π(s')]

    Returns:
        new_policy: 改进后的策略
        stable:     策略是否稳定 (未发生变化)
    """
    new_policy = {}
    stable = True

    for s in env.get_all_states():
        best_action = 0
        best_value = float('-inf')

        for a in ACTIONS:
            # 计算 Q(s, a) = Σ P(s'|s,a) · [r + γ·V(s')]
            q_sa = 0.0
            for prob, next_state, reward, done in env.get_transitions(s, a):
                if done:
                    q_sa += prob * reward
                else:
                    q_sa += prob * (reward + gamma * V[next_state])

            if q_sa > best_value:
                best_value = q_sa
                best_action = a

        new_policy[s] = best_action

    # 检查策略是否发生变化
    for s in env.get_all_states():
        if s in V and new_policy.get(s) != env._get_old_policy(s) if hasattr(env, '_get_old_policy') else False:
            stable = False
            break

    return new_policy, stable


# ==============================================================================
# 策略迭代 (Policy Iteration)
# ==============================================================================

def policy_iteration(env: GridWorld, gamma: float = 0.99,
                     verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    策略迭代算法。

    交替执行:
    1. 策略评估 (Policy Evaluation): V^π ← 求解贝尔曼期望方程
    2. 策略改进 (Policy Improvement): π ← greedy(V^π)
    直到策略收敛。

    Returns:
        policy: 最优策略 {state: action}
        V:      最优状态价值函数 {state: value}
    """
    if verbose:
        print("=" * 50)
        print("策略迭代 (Policy Iteration)")
        print("=" * 50)

    # 初始化: 随机策略 (全部选择 "右")
    policy = {s: 0 for s in env.get_all_states()}

    for iteration in range(100):
        # (1) 策略评估
        V = policy_evaluation(env, policy, gamma=gamma)

        # (2) 策略改进
        old_policy = dict(policy)
        new_policy = {}
        policy_changed = False

        for s in env.get_all_states():
            best_action = 0
            best_value = float('-inf')

            for a in ACTIONS:
                q_sa = 0.0
                for prob, ns, reward, done in env.get_transitions(s, a):
                    if done:
                        q_sa += prob * reward
                    else:
                        q_sa += prob * (reward + gamma * V[ns])

                if q_sa > best_value:
                    best_value = q_sa
                    best_action = a

            new_policy[s] = best_action
            if old_policy.get(s) != best_action:
                policy_changed = True

        policy = new_policy

        if verbose:
            print(f"  迭代 {iteration + 1}: "
                  f"策略{'变化' if policy_changed else '稳定'}, "
                  f"V 范围 [{min(V.values()):.3f}, {max(V.values()):.3f}]")

        # 策略稳定 → 收敛
        if not policy_changed:
            if verbose:
                print(f"\n✓ 策略迭代在第 {iteration + 1} 轮收敛!")
            break

    return policy, V


# ==============================================================================
# 价值迭代 (Value Iteration)
# ==============================================================================

def value_iteration(env: GridWorld, gamma: float = 0.99,
                    theta: float = 1e-6, verbose: bool = True
                    ) -> Tuple[Dict, Dict]:
    """
    价值迭代算法。

    直接迭代贝尔曼最优方程:
        V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) · [R + γ·V_k(s')]
    收敛后提取贪心策略。

    核心思想: 把策略评估和策略改进合并为一步
    (每次只做一步评估，但使用 max 而不是按策略求和)

    Returns:
        policy: 最优策略
        V:      最优状态价值函数
    """
    if verbose:
        print("=" * 50)
        print("价值迭代 (Value Iteration)")
        print("=" * 50)

    # 初始化 V(s) = 0
    V = {s: 0.0 for s in env.states}

    for iteration in range(1000):
        delta = 0

        for s in env.get_all_states():
            v_old = V[s]

            # 贝尔曼最优方程: V(s) = max_a Σ P(s'|s,a) · [r + γ·V(s')]
            best_value = float('-inf')
            for a in ACTIONS:
                q_sa = 0.0
                for prob, ns, reward, done in env.get_transitions(s, a):
                    if done:
                        q_sa += prob * reward
                    else:
                        q_sa += prob * (reward + gamma * V[ns])
                best_value = max(best_value, q_sa)

            V[s] = best_value
            delta = max(delta, abs(best_value - v_old))

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  迭代 {iteration + 1}: delta = {delta:.6f}")

        if delta < theta:
            if verbose:
                print(f"\n✓ 价值迭代在第 {iteration + 1} 轮收敛! (delta={delta:.2e})")
            break

    # 从 V* 提取最优策略
    policy = extract_policy(env, V, gamma)

    return policy, V


def extract_policy(env: GridWorld, V: Dict[Tuple[int, int], float],
                   gamma: float = 0.99) -> Dict[Tuple[int, int], int]:
    """
    从价值函数提取贪心策略。
    π*(s) = argmax_a Σ P(s'|s,a) · [R + γ·V*(s')]
    """
    policy = {}
    for s in env.get_all_states():
        best_action = 0
        best_value = float('-inf')

        for a in ACTIONS:
            q_sa = 0.0
            for prob, ns, reward, done in env.get_transitions(s, a):
                if done:
                    q_sa += prob * reward
                else:
                    q_sa += prob * (reward + gamma * V[ns])

            if q_sa > best_value:
                best_value = q_sa
                best_action = a

        policy[s] = best_action

    return policy


# ==============================================================================
# 演示
# ==============================================================================

def demo_dp():
    """动态规划演示: 策略迭代 vs 价值迭代。"""
    print("=" * 60)
    print("  动态规划 (DP) 演示: 策略迭代 vs 价值迭代")
    print("=" * 60)

    # ---- 确定性环境 ----
    print("\n" + "─" * 60)
    print("场景 1: 确定性环境 (动作 100% 生效)")
    print("─" * 60)

    config = GridWorldConfig(stochastic=False)
    env = GridWorld(config)
    print("\n地图:")
    print(env.render())

    # 策略迭代
    print()
    pi_policy, pi_V = policy_iteration(env, gamma=config.gamma)
    print("\n策略迭代 - 最优策略:")
    print(env.render(policy=pi_policy))
    print("\n策略迭代 - 价值函数:")
    print(env.render_values(pi_V))

    # 价值迭代
    print()
    vi_policy, vi_V = value_iteration(env, gamma=config.gamma)
    print("\n价值迭代 - 最优策略:")
    print(env.render(policy=vi_policy))
    print("\n价值迭代 - 价值函数:")
    print(env.render_values(vi_V))

    # 验证两种方法得到相同结果
    same_policy = all(pi_policy[s] == vi_policy[s] for s in env.get_all_states())
    v_diff = max(abs(pi_V[s] - vi_V[s]) for s in env.get_all_states())
    print(f"\n策略一致: {same_policy}, 价值最大差异: {v_diff:.2e}")

    # ---- 随机性环境 ----
    print("\n" + "─" * 60)
    print("场景 2: 随机性环境 (80% 按选择方向, 各 10% 侧滑)")
    print("─" * 60)

    config_stoch = GridWorldConfig(stochastic=True, slip_prob=0.1)
    env_stoch = GridWorld(config_stoch)

    print()
    pi_policy_s, pi_V_s = policy_iteration(env_stoch, gamma=config_stoch.gamma)
    print("\n策略迭代 (随机) - 最优策略:")
    print(env_stoch.render(policy=pi_policy_s))
    print("\n策略迭代 (随机) - 价值函数:")
    print(env_stoch.render_values(pi_V_s))

    print()
    vi_policy_s, vi_V_s = value_iteration(env_stoch, gamma=config_stoch.gamma)

    # 对比确定性 vs 随机性
    print("\n" + "─" * 60)
    print("确定性 vs 随机性 价值对比:")
    print("─" * 60)
    for s in sorted(env.get_all_states()):
        v_det = pi_V.get(s, 0)
        v_sto = pi_V_s.get(s, 0)
        diff = v_sto - v_det
        if abs(diff) > 0.01:
            print(f"  状态 {s}: 确定={v_det:+.3f}, 随机={v_sto:+.3f}, "
                  f"差异={diff:+.3f}")

    print("\n观察: 随机性环境中价值普遍更低 (因为有概率滑入非最优方向)")

    print("\n[DP 演示完成]")

    return {
        'det': {'policy': pi_policy, 'V': pi_V},
        'stoch': {'policy': pi_policy_s, 'V': pi_V_s},
    }


if __name__ == '__main__':
    demo_dp()
