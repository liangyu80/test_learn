"""
GridWorld 环境 —— 经典强化学习表格型方法的试验场

环境描述:
    一个 N×N 的网格世界，智能体从起点 S 出发，到达终点 G 获得奖励。
    地图上有墙壁 (不可通过) 和陷阱 (负奖励)。

    示例 (5×5):
        ┌───┬───┬───┬───┬───┐
        │ S │   │   │   │   │
        ├───┼───┼───┼───┼───┤
        │   │ ▓ │   │ ▓ │   │
        ├───┼───┼───┼───┼───┤
        │   │   │   │   │   │
        ├───┼───┼───┼───┼───┤
        │   │ ▓ │ ✕ │   │   │
        ├───┼───┼───┼───┼───┤
        │   │   │   │   │ G │
        └───┴───┴───┴───┴───┘

        S = 起点, G = 终点 (+1), ▓ = 墙壁, ✕ = 陷阱 (-1)

    动作空间: {上, 下, 左, 右} = {0, 1, 2, 3}

    转移规则:
        - 确定性版本: 100% 按选择的方向移动
        - 随机性版本: 80% 按选择方向, 各 10% 侧滑到垂直方向

    奖励:
        - 到达终点 G: +1.0 (回合结束)
        - 掉入陷阱 ✕: -1.0 (回合结束)
        - 每步移动:  -0.01 (鼓励找最短路径)
        - 撞墙:      -0.01 (原地不动)

MDP 要素:
    S (状态空间): 所有合法格子 = {(r, c) | 0 ≤ r,c < N, 非墙壁}
    A (动作空间): {上=0, 下=1, 左=2, 右=3}
    P (转移概率): P(s'|s, a) — 确定性或随机性
    R (奖励函数): R(s, a, s')
    γ (折扣因子): 通常 0.99
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import random


# ==============================================================================
# 环境配置
# ==============================================================================

@dataclass
class GridWorldConfig:
    """GridWorld 配置。"""
    size: int = 5                   # 网格大小
    start: Tuple[int, int] = (0, 0)  # 起点
    goal: Tuple[int, int] = (4, 4)   # 终点
    walls: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 1), (1, 3), (3, 1)
    ])
    traps: List[Tuple[int, int]] = field(default_factory=lambda: [
        (3, 2)
    ])
    stochastic: bool = False        # 是否随机转移
    slip_prob: float = 0.1          # 侧滑概率 (每侧)
    step_reward: float = -0.01      # 每步惩罚
    goal_reward: float = 1.0        # 到达终点奖励
    trap_reward: float = -1.0       # 陷阱惩罚
    gamma: float = 0.99             # 折扣因子
    max_steps: int = 200            # 最大步数


# ==============================================================================
# 动作定义
# ==============================================================================

# 动作: 上=0, 下=1, 左=2, 右=3
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = ['上', '下', '左', '右']
ACTION_ARROWS = ['↑', '↓', '←', '→']

# 动作对应的行列偏移 (row_delta, col_delta)
ACTION_DELTAS = {
    0: (-1, 0),   # 上
    1: (1, 0),    # 下
    2: (0, -1),   # 左
    3: (0, 1),    # 右
}

# 每个动作的垂直方向 (用于随机侧滑)
ACTION_PERPENDICULAR = {
    0: [2, 3],  # 上 → 可能侧滑到 左/右
    1: [2, 3],  # 下 → 左/右
    2: [0, 1],  # 左 → 上/下
    3: [0, 1],  # 右 → 上/下
}


# ==============================================================================
# GridWorld 环境
# ==============================================================================

class GridWorld:
    """
    网格世界环境。

    提供两种接口:
    1. 交互式 (用于 MC/TD 方法):
       env.reset() → state
       env.step(action) → (next_state, reward, done)

    2. 模型已知 (用于 DP 方法):
       env.get_transitions(state, action) → [(prob, next_state, reward, done)]
       env.get_all_states() → [states]
    """

    def __init__(self, config: GridWorldConfig = None):
        if config is None:
            config = GridWorldConfig()
        self.config = config
        self.size = config.size

        # 构建地图
        self.walls = set(config.walls)
        self.traps = set(config.traps)
        self.goal = config.goal
        self.start = config.start

        # 所有合法状态
        self.states = []
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self.walls:
                    self.states.append((r, c))

        # 终止状态
        self.terminal_states = {config.goal} | self.traps

        # 状态到索引的映射 (用于表格方法)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.num_states = len(self.states)
        self.num_actions = len(ACTIONS)

        # 当前状态
        self.current_state = None
        self.steps = 0

    def reset(self) -> Tuple[int, int]:
        """重置环境，返回初始状态。"""
        self.current_state = self.start
        self.steps = 0
        return self.current_state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行动作，返回 (next_state, reward, done)。

        Args:
            action: 动作 (0=上, 1=下, 2=左, 3=右)
        Returns:
            next_state: 下一个状态
            reward:     获得的奖励
            done:       是否结束
        """
        assert self.current_state is not None, "请先调用 reset()"
        self.steps += 1

        # 随机性: 80% 按选择方向, 各 10% 侧滑
        if self.config.stochastic:
            rand = random.random()
            if rand < 1 - 2 * self.config.slip_prob:
                actual_action = action
            elif rand < 1 - self.config.slip_prob:
                actual_action = ACTION_PERPENDICULAR[action][0]
            else:
                actual_action = ACTION_PERPENDICULAR[action][1]
        else:
            actual_action = action

        next_state = self._move(self.current_state, actual_action)
        reward = self._get_reward(next_state)
        done = next_state in self.terminal_states or self.steps >= self.config.max_steps

        self.current_state = next_state
        return next_state, reward, done

    def _move(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """计算移动后的状态 (处理边界和墙壁)。"""
        dr, dc = ACTION_DELTAS[action]
        new_r = state[0] + dr
        new_c = state[1] + dc

        # 越界或撞墙 → 原地不动
        if (new_r < 0 or new_r >= self.size or
            new_c < 0 or new_c >= self.size or
            (new_r, new_c) in self.walls):
            return state

        return (new_r, new_c)

    def _get_reward(self, state: Tuple[int, int]) -> float:
        """获取到达某状态的奖励。"""
        if state == self.goal:
            return self.config.goal_reward
        elif state in self.traps:
            return self.config.trap_reward
        else:
            return self.config.step_reward

    # ==================================================================
    # 模型已知接口 (用于 DP 方法)
    # ==================================================================

    def get_transitions(self, state: Tuple[int, int], action: int
                        ) -> List[Tuple[float, Tuple[int, int], float, bool]]:
        """
        获取转移概率 P(s'|s, a)。

        返回: [(概率, 下一状态, 奖励, 是否终止), ...]

        DP 方法需要完整的转移模型，这是 DP 与 MC/TD 的关键区别:
        - DP: 需要 P(s'|s,a) (model-based)
        - MC/TD: 只需要与环境交互 (model-free)
        """
        if state in self.terminal_states:
            return [(1.0, state, 0.0, True)]

        transitions = []

        if self.config.stochastic:
            # 80% 按选择方向
            main_prob = 1 - 2 * self.config.slip_prob
            ns = self._move(state, action)
            transitions.append((main_prob, ns, self._get_reward(ns),
                                ns in self.terminal_states))
            # 各 10% 侧滑
            for perp_action in ACTION_PERPENDICULAR[action]:
                ns = self._move(state, perp_action)
                transitions.append((self.config.slip_prob, ns, self._get_reward(ns),
                                    ns in self.terminal_states))
        else:
            ns = self._move(state, action)
            transitions.append((1.0, ns, self._get_reward(ns),
                                ns in self.terminal_states))

        return transitions

    def get_all_states(self) -> List[Tuple[int, int]]:
        """获取所有非终止状态。"""
        return [s for s in self.states if s not in self.terminal_states]

    # ==================================================================
    # 可视化
    # ==================================================================

    def render(self, policy: Optional[Dict] = None,
               values: Optional[Dict] = None) -> str:
        """
        渲染网格世界。

        Args:
            policy: 策略字典 {state: action}
            values: 价值字典 {state: value}
        """
        lines = []
        sep = "┌" + "───┬" * (self.size - 1) + "───┐"
        lines.append(sep)

        for r in range(self.size):
            row = "│"
            for c in range(self.size):
                s = (r, c)
                if s in self.walls:
                    cell = " ▓ "
                elif s == self.goal:
                    cell = " G "
                elif s in self.traps:
                    cell = " ✕ "
                elif s == self.start:
                    if policy and s in policy:
                        cell = f" {ACTION_ARROWS[policy[s]]} "
                    else:
                        cell = " S "
                elif policy and s in policy:
                    cell = f" {ACTION_ARROWS[policy[s]]} "
                elif values and s in values:
                    v = values[s]
                    cell = f"{v:+.0f}" if abs(v) >= 1 else f".{int(abs(v)*10)}" if v >= 0 else "-.{0}".format(int(abs(v)*10))
                    cell = cell[:3].center(3)
                else:
                    cell = "   "
                row += cell + "│"
            lines.append(row)

            if r < self.size - 1:
                lines.append("├" + "───┼" * (self.size - 1) + "───┤")

        lines.append("└" + "───┴" * (self.size - 1) + "───┘")
        return "\n".join(lines)

    def render_policy(self, policy: Dict) -> str:
        """渲染策略 (用箭头表示)。"""
        return self.render(policy=policy)

    def render_values(self, values: Dict) -> str:
        """渲染价值函数。"""
        lines = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                s = (r, c)
                if s in self.walls:
                    row += "  ▓▓▓ "
                elif s == self.goal:
                    row += f" {self.config.goal_reward:+5.2f}"
                elif s in self.traps:
                    row += f" {self.config.trap_reward:+5.2f}"
                elif s in values:
                    row += f" {values[s]:+5.2f}"
                else:
                    row += "  ---  "
            lines.append(row)
        return "\n".join(lines)


# ==============================================================================
# 演示
# ==============================================================================

def demo_env():
    """环境演示。"""
    print("=" * 50)
    print("GridWorld 环境演示")
    print("=" * 50)

    env = GridWorld()
    print("\n网格世界地图:")
    print(env.render())
    print(f"\n状态数: {env.num_states}, 动作数: {env.num_actions}")
    print(f"起点: {env.start}, 终点: {env.goal}")
    print(f"墙壁: {env.config.walls}, 陷阱: {list(env.traps)}")

    # 随机游走
    print("\n--- 随机游走演示 ---")
    state = env.reset()
    total_reward = 0
    for step in range(20):
        action = random.choice(ACTIONS)
        next_state, reward, done = env.step(action)
        total_reward += reward
        if step < 5 or done:
            print(f"  步 {step+1}: {state} --{ACTION_NAMES[action]}--> "
                  f"{next_state}, r={reward:.2f}" + (" [结束]" if done else ""))
        state = next_state
        if done:
            break
    print(f"  总奖励: {total_reward:.2f}, 步数: {step+1}")

    print("\n[环境演示完成]")


if __name__ == '__main__':
    demo_env()
