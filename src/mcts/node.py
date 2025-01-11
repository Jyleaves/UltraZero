import numpy as np


class Node:
    def __init__(self, parent=None):
        # parent: Node or None for root
        self.parent = parent
        self.children_added = False
        # 子节点相关统计信息在expand时创建
        self.child_moves = None  # np.array of moves (int)
        self.P = None  # np.array, prior probability for each child
        self.N = None  # np.array, visit count
        self.W = None  # np.array, total value sum
        # 存储该节点状态价值（在扩展时从NN获得）
        self.value = 0.0

    def expand(self, moves, priors, value):
        # moves: 1D np.array of legal moves
        # priors: 1D np.array of priors for each legal move, filtered by moves
        # value: float, value predicted by NN for this node state
        num_children = len(moves)
        self.child_moves = moves
        self.P = priors
        self.N = np.zeros(num_children, dtype=np.float32)
        self.W = np.zeros(num_children, dtype=np.float32)
        self.value = value
        self.children_added = True

    def is_expanded(self):
        return self.children_added

    def select_child(self, c_puct):
        # 根据UCT公式选择子节点
        # U = c_puct * P * sqrt(sum(N)) / (1 + N_child)
        # Q = W/N
        sum_n = np.sum(self.N)
        inv_den = 1.0 / (self.N + 1e-8)  # 防止除零
        Q = self.W * inv_den
        U = c_puct * self.P * np.sqrt(sum_n + 1e-8) * inv_den
        scores = Q + U
        best_idx = np.argmax(scores)
        return best_idx

    def update(self, child_idx, value):
        # 回传更新
        self.N[child_idx] += 1
        self.W[child_idx] += value

    def get_most_visited_move(self):
        # 用于最终决策时选择访问次数最高的动作
        return self.child_moves[np.argmax(self.N)]

    def get_policy_distribution(self, temperature=1.0):
        # 将访问次数N作为策略分布π
        if temperature == 0:
            # 选最大者为1，其余为0
            dist = np.zeros_like(self.N)
            dist[np.argmax(self.N)] = 1.0
            return dist
        else:
            # N^(1/temp)归一化
            x = np.power(self.N, 1.0 / temperature)
            if np.sum(x) == 0:
                # 若无访问, 则均匀分布
                return np.ones_like(self.N) / len(self.N)
            return x / np.sum(x)
