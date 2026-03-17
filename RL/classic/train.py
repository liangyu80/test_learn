"""
经典表格型 RL 统一对比实验

对比以下方法在 GridWorld 上的表现:
  1. 动态规划: 策略迭代, 价值迭代 (需要模型)
  2. 蒙特卡洛: First-Visit MC, Every-Visit MC
  3. 时序差分: SARSA, Q-Learning, Expected SARSA, n-step SARSA, SARSA(λ), Q(λ)

评估指标:
  - 与最优策略的一致率 (DP 结果作为基准)
  - 平均回报
  - 收敛速度 (达到 95% 最优回报的 episode 数)
"""

import time
import random
from collections import defaultdict
from env import GridWorld, GridWorldConfig, ACTIONS

# 导入各算法模块
from dp import policy_iteration, value_iteration
from mc import first_visit_mc, every_visit_mc, q_to_v as mc_q_to_v
from td import (sarsa, q_learning, expected_sarsa, n_step_sarsa,
                sarsa_lambda, q_lambda, q_to_v as td_q_to_v)


def evaluate_policy(env: GridWorld, policy: dict, gamma: float = 0.99,
                    num_eval: int = 100) -> float:
    """评估策略的平均折扣回报。"""
    total = 0.0
    for _ in range(num_eval):
        state = env.reset()
        G = 0.0
        discount = 1.0
        for _ in range(200):
            action = policy.get(state, 0)
            next_state, reward, done = env.step(action)
            G += discount * reward
            discount *= gamma
            state = next_state
            if done:
                break
        total += G
    return total / num_eval


def policy_agreement(policy1: dict, policy2: dict, states: list) -> float:
    """计算两个策略在所有状态上的一致率。"""
    if not states:
        return 0.0
    agree = sum(1 for s in states if policy1.get(s) == policy2.get(s))
    return agree / len(states)


def run_comparison():
    """运行完整对比实验。"""
    random.seed(42)

    print("=" * 70)
    print("  经典表格型 RL 方法统一对比")
    print("=" * 70)

    # ====== 环境设置 ======
    config = GridWorldConfig(stochastic=False)
    env = GridWorld(config)
    gamma = config.gamma

    print("\n地图:")
    print(env.render())
    print(f"折扣因子 γ = {gamma}")

    all_states = env.get_all_states()
    results = {}

    # ====== 1. 动态规划 (基准) ======
    print("\n" + "=" * 70)
    print("  Part 1: 动态规划 (DP) — 最优解基准")
    print("=" * 70)

    t0 = time.time()
    pi_policy, pi_V = policy_iteration(env, gamma=gamma, verbose=False)
    t_pi = time.time() - t0

    t0 = time.time()
    vi_policy, vi_V = value_iteration(env, gamma=gamma, verbose=False)
    t_vi = time.time() - t0

    # DP 作为最优基准
    optimal_policy = pi_policy
    optimal_return = evaluate_policy(env, optimal_policy, gamma=gamma)

    print(f"\n策略迭代: 耗时 {t_pi:.4f}s, 回报 = {optimal_return:.3f}")
    print(f"价值迭代: 耗时 {t_vi:.4f}s, 回报 = {evaluate_policy(env, vi_policy, gamma=gamma):.3f}")
    print(f"策略一致率: {policy_agreement(pi_policy, vi_policy, all_states)*100:.1f}%")

    print("\n最优策略 (DP):")
    print(env.render(policy=optimal_policy))
    print("最优价值函数 (DP):")
    print(env.render_values(pi_V))

    results['策略迭代 (PI)'] = {'policy': pi_policy, 'time': t_pi}
    results['价值迭代 (VI)'] = {'policy': vi_policy, 'time': t_vi}

    # ====== 2. 蒙特卡洛方法 ======
    print("\n" + "=" * 70)
    print("  Part 2: 蒙特卡洛方法 (MC)")
    print("=" * 70)

    mc_configs = [
        ("First-Visit MC", lambda: first_visit_mc(
            env, num_episodes=5000, gamma=gamma, epsilon=0.2, verbose=False)),
        ("Every-Visit MC", lambda: every_visit_mc(
            env, num_episodes=5000, gamma=gamma, epsilon=0.2, verbose=False)),
    ]

    for name, run_fn in mc_configs:
        t0 = time.time()
        Q, policy = run_fn()
        elapsed = time.time() - t0
        avg_return = evaluate_policy(env, policy, gamma=gamma)
        agree = policy_agreement(policy, optimal_policy, all_states) * 100
        print(f"\n{name}:")
        print(f"  耗时: {elapsed:.2f}s, 平均回报: {avg_return:.3f}, "
              f"策略一致率: {agree:.1f}%")
        results[name] = {'policy': policy, 'time': elapsed, 'Q': Q}

    # ====== 3. 时序差分方法 ======
    print("\n" + "=" * 70)
    print("  Part 3: 时序差分学习 (TD)")
    print("=" * 70)

    td_configs = [
        ("SARSA", lambda: sarsa(
            env, num_episodes=3000, gamma=gamma, verbose=False)),
        ("Q-Learning", lambda: q_learning(
            env, num_episodes=3000, gamma=gamma, verbose=False)),
        ("Expected SARSA", lambda: expected_sarsa(
            env, num_episodes=3000, gamma=gamma, verbose=False)),
        ("3-step SARSA", lambda: n_step_sarsa(
            env, n=3, num_episodes=3000, gamma=gamma, verbose=False)),
        ("SARSA(λ=0.8)", lambda: sarsa_lambda(
            env, lam=0.8, num_episodes=3000, gamma=gamma, verbose=False)),
        ("Q(λ=0.8)", lambda: q_lambda(
            env, lam=0.8, num_episodes=3000, gamma=gamma, verbose=False)),
    ]

    for name, run_fn in td_configs:
        t0 = time.time()
        Q, policy = run_fn()
        elapsed = time.time() - t0
        avg_return = evaluate_policy(env, policy, gamma=gamma)
        agree = policy_agreement(policy, optimal_policy, all_states) * 100
        print(f"\n{name}:")
        print(f"  耗时: {elapsed:.2f}s, 平均回报: {avg_return:.3f}, "
              f"策略一致率: {agree:.1f}%")
        results[name] = {'policy': policy, 'time': elapsed, 'Q': Q}

    # ====== 4. 随机性环境对比 ======
    print("\n" + "=" * 70)
    print("  Part 4: 随机性环境 (滑动概率 10%)")
    print("=" * 70)

    config_s = GridWorldConfig(stochastic=True, slip_prob=0.1)
    env_s = GridWorld(config_s)

    # DP 基准 (随机)
    pi_s, V_s = policy_iteration(env_s, gamma=gamma, verbose=False)
    opt_return_s = evaluate_policy(env_s, pi_s, gamma=gamma)
    print(f"\nDP 最优回报 (随机): {opt_return_s:.3f}")

    stoch_methods = [
        ("SARSA", lambda: sarsa(
            env_s, num_episodes=5000, gamma=gamma, verbose=False)),
        ("Q-Learning", lambda: q_learning(
            env_s, num_episodes=5000, gamma=gamma, verbose=False)),
        ("SARSA(λ=0.8)", lambda: sarsa_lambda(
            env_s, lam=0.8, num_episodes=5000, gamma=gamma, verbose=False)),
    ]

    for name, run_fn in stoch_methods:
        Q, policy = run_fn()
        avg = evaluate_policy(env_s, policy, gamma=gamma)
        agree = policy_agreement(policy, pi_s, env_s.get_all_states()) * 100
        print(f"  {name}: 回报={avg:.3f}, 策略一致率={agree:.1f}%")

    # ====== 5. 总结 ======
    print("\n" + "=" * 70)
    print("  总结对比表")
    print("=" * 70)
    print(f"\n{'方法':<22s} {'类型':<12s} {'需要模型':>8s} {'回报':>8s} {'一致率':>8s} {'耗时':>8s}")
    print("─" * 70)

    summary = [
        ("策略迭代 (PI)", "DP", "是"),
        ("价值迭代 (VI)", "DP", "是"),
        ("First-Visit MC", "MC", "否"),
        ("Every-Visit MC", "MC", "否"),
        ("SARSA", "TD", "否"),
        ("Q-Learning", "TD", "否"),
        ("Expected SARSA", "TD", "否"),
        ("3-step SARSA", "TD", "否"),
        ("SARSA(λ=0.8)", "TD(λ)", "否"),
        ("Q(λ=0.8)", "TD(λ)", "否"),
    ]

    for name, category, needs_model in summary:
        if name in results:
            r = results[name]
            avg = evaluate_policy(env, r['policy'], gamma=gamma)
            agree = policy_agreement(r['policy'], optimal_policy, all_states) * 100
            print(f"  {name:<20s} {category:<12s} {needs_model:>8s} "
                  f"{avg:>+8.3f} {agree:>7.1f}% {r['time']:>7.3f}s")

    # ====== 关键观察 ======
    print("\n" + "=" * 70)
    print("  关键观察")
    print("=" * 70)
    print("""
  1. DP (策略/价值迭代):
     - 保证找到最优解, 但需要完整的环境模型
     - 计算速度最快 (不需要采样)

  2. MC (蒙特卡洛):
     - 无偏估计, 但方差较大
     - 需要完整 episode, 收敛较慢
     - 不依赖马尔可夫性质

  3. TD (时序差分):
     - 每步都能更新, 不需要等 episode 结束
     - Bootstrap 引入偏差, 但方差更低, 收敛更快
     - SARSA (on-policy) 更保守, Q-Learning (off-policy) 更激进
     - Expected SARSA 通常表现最好 (低方差 + off-policy 的优点)

  4. 多步 TD & TD(λ):
     - 桥接 TD(0) 和 MC, 通过 n 或 λ 控制 bias-variance 权衡
     - 通常 n=3~5 或 λ=0.7~0.9 效果较好
     - 资格迹实现更高效 (不需要存储 n 步轨迹)

  5. 确定性 vs 随机性环境:
     - 随机性使得所有方法的价值更低 (有概率滑入非最优方向)
     - On-policy (SARSA) 在随机环境中学到更保守但更安全的策略
""")

    print("[对比实验完成]")


if __name__ == '__main__':
    run_comparison()
