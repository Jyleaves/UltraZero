import numpy as np
import random

from node import Node

# 假设有一个模型接口model.predict(state) -> (policy_logits, value)
# policy_logits：长度81的原始logits（未mask非法动作时）
# value：标量[-1,1]
# 在扩展节点时，需根据legal_moves对policy_logits进行softmax归一化
# 另外，支持在根节点添加Dirichlet噪声

class MCTS:
    def __init__(self, model=None, simulations=800, c_puct=1.0, dirichlet_alpha=0.03, root_dirichlet_frac=0.25):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_dirichlet_frac = root_dirichlet_frac

    def search(self, root_node, state, legal_moves, add_dirichlet=False):
        # root_node: 一个未expand的节点或已expand的根节点
        # state: 当前棋盘状态对象(具备复制和apply_move等方法)
        # legal_moves: root的合法动作列表
        #
        # 返回root节点中基于MCTS搜索后的策略分布π和访问后root节点
        # 内部会对root进行expand
        if not root_node.is_expanded():
            self._expand_node(root_node, state, legal_moves, is_root=True, add_dirichlet=add_dirichlet)

        for _ in range(self.simulations):
            # 对每次模拟都复制状态，进行从root到叶子的选择与扩展
            sim_state = state.copy()
            node = root_node
            node_stack = [node]
            moves_stack = []

            # 1. Selection
            while node.is_expanded():
                if len(node.N) == 0:  # 无子节点（无后续可走棋，可能游戏结束）
                    break
                child_idx = node.select_child(self.c_puct)
                move = node.child_moves[child_idx]
                sim_state.apply_move(move)
                node_stack.append(node)
                moves_stack.append((node, child_idx))

                # 下一个子节点（根据该child是否已expand）
                # 我们需要知道子节点对象，但为了保持轻量，节点只在expand时创建。
                # 这里使用技巧：不为每个子节点创建独立Node对象，只有在expand时访问叶子点再创建。
                # 优化：当没有预先分配子Node对象时，可以在expand时创建一个dummy节点。
                # 这里省略子节点缓存，可以在Node中加字典存子节点node实例，若需优化可采用数组。
                # 这里我们简单实现为延迟expand，无需实际子node对象存储，用当前node重复利用数据结构。
                # 注意：这可能使代码复杂，我们这里保持AlphaZero逻辑，每个node只represent一个状态点。
                # 实际中应为每个child建立新Node进行管理。我们在expand时为children创建Node列表。

                # 因为AlphaZero要求对树中每个状态有一个节点，这里在expand时对child都创建Node对象并存储在list中。
                # 所以需要node的children节点对象存储。
                # 补充：node中加一个children_nodes列表：
                if not hasattr(node, 'children_nodes'):
                    # 在expand时会添加
                    pass
                node = node.children_nodes[child_idx]

                if node is None:
                    # 未扩展的叶子节点到达，结束Selection
                    break

            # 如果node没有expand过，进行expand
            if node is not None and not node.is_expanded() and not sim_state.is_game_over():
                next_legal_moves = sim_state.get_legal_moves()
                self._expand_node(node, sim_state, next_legal_moves)

            # 2. Evaluate leaf（如果游戏结束或无子节点）
            leaf_value = 0.0
            if sim_state.is_game_over():
                current_player = sim_state.get_current_player()  # 获取当前玩家
                winner = sim_state.get_winner()  # 获取游戏结果
                leaf_value = self._terminal_value(winner, current_player)
            else:
                # 若刚expand过，则node.value已设置；
                leaf_value = node.value

            # 3. Backpropagate
            # 回溯时从最后到root，每经过一层翻转value符号
            # leaf_value是对最后一层node所在player的价值，对其父节点的player价值相反。
            for (nd, child_idx) in reversed(moves_stack):
                nd.update(child_idx, leaf_value)
                leaf_value = -leaf_value

        # 返回策略分布π
        # 若是最终决策阶段，可设temperature=0
        pi = root_node.get_policy_distribution(temperature=1.0)
        return pi, root_node

    def _expand_node(self, node, state, legal_moves, is_root=False, add_dirichlet=False):
        if self.model is not None:
            # 使用model评估当前状态
            # model输入state得到(policy_logits, value)
            # policy_logits为长度81，需对合法动作取子集并归一化为P
            board_tensor = state.get_feature_tensor()
            policy_logits, value = self.model.predict(board_tensor)
            # 只保留legal_moves对应的logits
            legal_logits = policy_logits[legal_moves]
            # softmax归一化
            max_logit = np.max(legal_logits)
            exp_ = np.exp(legal_logits - max_logit)
            priors = exp_ / (np.sum(exp_) + 1e-8)

            if is_root and add_dirichlet and len(priors) > 0:
                dir_noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
                priors = priors * (1.0 - self.root_dirichlet_frac) + dir_noise * self.root_dirichlet_frac

            node.expand(legal_moves, priors, value)
        else:
            # 如果模型不存在，使用随机走子来估计价值
            value = self._random_rollout(state)
            priors = np.ones(len(legal_moves)) / len(legal_moves)  # 均匀分布
            node.expand(legal_moves, priors, value)

        # 为子节点建立占位对象
        node.children_nodes = [None] * len(legal_moves)
        for i in range(len(legal_moves)):
            node.children_nodes[i] = Node(parent=node)

    def _random_rollout(self, state):
        """
        通过随机走子直到终局来估计当前状态的价值。
        :param state: 当前棋盘状态。
        :return: 从当前玩家视角的价值。
        """
        sim_state = state.copy()
        while not sim_state.is_game_over():
            legal_moves = sim_state.get_legal_moves()
            move = random.choice(legal_moves)
            sim_state.apply_move(move)

        current_player = state.get_current_player()
        winner = sim_state.get_winner()
        return self._terminal_value(winner, current_player)

    def _terminal_value(self, winner, current_player):
        """
        计算叶子节点的价值。
        :param winner: 游戏结果，1 表示 X 获胜，-1 表示 O 获胜，0 表示平局。
        :param current_player: 当前玩家，1 表示 X，-1 表示 O。
        :return: 从当前玩家视角的价值。
        """
        if winner == current_player:
            return 1.0  # 当前玩家获胜
        elif winner == -current_player:
            return -1.0  # 对手获胜
        else:
            return 0.0  # 平局

    def get_action_prob(self, root_node, temperature=1.0):
        return root_node.get_policy_distribution(temperature=temperature)
