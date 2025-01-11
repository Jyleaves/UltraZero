import numpy as np
import math

from ..mcts.mcts import MCTS
from ..mcts.node import Node
from utils import data_augmentations


class SelfPlay:
    def __init__(self, model, mcts_simulations=800, c_puct=1.0, dirichlet_alpha=0.03, root_dirichlet_frac=0.25,
                 initial_temperature=1.0, final_temperature=0.1, decay_rate=0.01, warmup_steps=10):
        # model: 已加载好参数的NN (UltraZeroModel)
        # MCTS参数
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_dirichlet_frac = root_dirichlet_frac

        # Temperature 参数
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.decay_rate = decay_rate  # 指数衰减速率
        self.warmup_steps = warmup_steps  # 温度衰减的步数范围
        self.temperature = initial_temperature
        self.step_count = 0  # 用于动态调整 temperature

    def update_temperature(self):
        """动态调整 temperature"""
        if self.step_count < self.warmup_steps:
            # 在 warmup_steps 内使用指数衰减
            self.temperature = self.final_temperature + (
                (self.initial_temperature - self.final_temperature) *
                math.exp(-self.decay_rate * self.step_count)
            )
        else:
            # 超过 warmup_steps 后，直接将 temperature 设置为 0
            self.temperature = 0.0
        self.step_count += 1

    def play_game(self, board, buffer=None, add_to_buffer=True, initial_add_dirichlet=True):
        # board: Board对象（从game中导入）
        # buffer: ReplayBuffer对象，用于存储(s,pi,z)
        # add_to_buffer: 是否将数据存入buffer
        # initial_add_dirichlet: 首次动作时在根节点添加Dirichlet噪声

        # 存储本局对局所有步骤的数据，用于在对局结束后更新z
        state_list = []
        pi_list = []
        player_list = []

        mcts = MCTS(self.model, simulations=self.mcts_simulations, c_puct=self.c_puct,
                    dirichlet_alpha=self.dirichlet_alpha, root_dirichlet_frac=self.root_dirichlet_frac)

        # 对局循环
        step_count = 0
        add_dirichlet = initial_add_dirichlet
        while not board.is_game_over():
            legal_moves = board.get_legal_moves()
            if len(legal_moves) == 0:
                # 无可走步，但未判定game_over，强制结束(平局)
                board.game_over = True
                board.winner = 0
                break

            root_node = Node(parent=None)
            # 使用MCTS搜索
            pi, root_node = mcts.search(root_node, board, np.array(legal_moves), add_dirichlet=add_dirichlet)
            add_dirichlet = False  # 仅第一步加噪声

            # 根据 temperature 对 pi 进行温度缩放
            if self.temperature == 0:
                # 选最大访问次数的动作
                action = legal_moves[np.argmax(pi)]
            else:
                # 对 pi 进行温度缩放
                pi = np.power(pi, 1.0 / self.temperature)
                pi = pi / np.sum(pi)  # 归一化
                # 按缩放后的 pi 分布随机采样动作
                action = np.random.choice(legal_moves, p=pi)

            # 存储状态数据
            # 获取当前状态特征(5,9,9)
            state_tensor = board.get_feature_tensor()
            state_list.append(state_tensor)
            pi_list.append(pi)
            player_list.append(board.get_current_player())

            # 执行动作
            board.apply_move(action)
            step_count += 1

            # 更新 temperature
            self.update_temperature()

        # 对局结束，计算最终结果z
        # winner: {1: X胜, -1:O胜, 0:平局}
        # z对每个state从该状态player视角：如果player==winner则z=1；平局z=0；否则z=-1
        winner = board.get_winner()
        for i, player in enumerate(player_list):
            if winner == player:
                z = 1.0
            elif winner == 0:
                z = 0.0
            else:
                z = -1.0

            if add_to_buffer and buffer is not None:
                # 对状态进行数据增强
                augmented_states = data_augmentations(state_list[i])
                augmented_pis = data_augmentations(pi_list[i].reshape(1, 9, 9))  # 将pi reshape为(1,9,9)以便增强

                # 将增强后的数据添加到buffer
                for aug_state, aug_pi in zip(augmented_states, augmented_pis):
                    buffer.add(aug_state, aug_pi.flatten(), z)  # 将aug_pi恢复为(81,)的形状

        return winner
