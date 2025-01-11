import numpy as np
import torch

from .metrics import compute_win_rate, compute_value_error
from ..mcts.mcts import MCTS
from ..mcts.node import Node


# 假设评估方式：
# 1. 利用给定模型对若干固定对局起点进行对战，直接使用policy头最大概率动作（不使用MCTS或使用较少次模拟）
# 2. 或使用两个模型对战统计胜率
# 3. 对一批已标注的state-value数据集来评估价值预测误差

class Evaluator:
    def __init__(self, model, num_games=100, use_mcts=False, mcts_simulations=200):
        self.model = model
        self.num_games = num_games
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations

    def evaluate_win_rate(self, opponent_model, board_cls):
        # 使用给定model与opponent_model对战num_games局，统计胜率
        # board_cls为创建Board实例的类名
        # use_mcts决定是否用MCTS搜索，否则直接用policy argmax落子
        # 返回(win_rate, draw_rate, loss_rate)

        results = []
        for i in range(self.num_games):
            board = board_cls()
            # 偶数局model先手(X=1)，奇数局opponent先手
            current_player_model = self.model if (i % 2 == 0) else opponent_model
            other_model = opponent_model if (i % 2 == 0) else self.model

            while not board.is_game_over():
                legal_moves = board.get_legal_moves()
                if not legal_moves:
                    # 无合法动作默认平局
                    board.game_over = True
                    board.winner = 0
                    break

                if self.use_mcts:
                    mcts = MCTS(current_player_model, simulations=self.mcts_simulations, c_puct=1.0)
                    root = Node(parent=None)
                    pi, root = mcts.search(root, board, np.array(legal_moves), add_dirichlet=False)
                    move = legal_moves[np.argmax(pi)]
                else:
                    # 直接用policy argmax
                    state_tensor = board.get_feature_tensor()
                    state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()
                    state_tensor = state_tensor.to(next(self.model.parameters()).device)
                    with torch.no_grad():
                        policy_logits, _ = current_player_model(state_tensor)
                    policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy().squeeze()
                    # 仅在legal_moves中选argmax
                    legal_probs = policy_probs[legal_moves]
                    chosen_move = legal_moves[np.argmax(legal_probs)]
                    move = chosen_move

                board.apply_move(move)
                # 切换玩家
                current_player_model, other_model = other_model, current_player_model

            # 最终winner与第一个玩家的关系
            # 第一个玩家是(i%2 == 0时self.model，否则opponent_model)
            # 如果i%2==0: self.model为X先手, winner=1则self.model赢, -1则opponent赢
            # 如果i%2==1: opponent_model为X先手, winner=1则opponent赢, self.model输

            if i % 2 == 0:
                # self.model先手
                if board.get_winner() == 1:
                    results.append(1)  # self.model win
                elif board.get_winner() == 0:
                    results.append(0)  # draw
                else:
                    results.append(-1)  # self.model loss
            else:
                # opponent_model先手
                if board.get_winner() == 1:
                    results.append(-1)  # self.model loss
                elif board.get_winner() == 0:
                    results.append(0)
                else:
                    results.append(1)  # self.model win

        win_rate, draw_rate, loss_rate = compute_win_rate(results)
        return win_rate, draw_rate, loss_rate

    def evaluate_value_error(self, dataset):
        # dataset包含(state, z_true)，这里z_true为真实结果或标注的价值
        # 测试模型对这些状态的价值预测误差
        pred_values = []
        true_values = []
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            for state, pi, z in dataset:
                # state:(5,9,9), z标量
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
                policy_logits, value = self.model(state_tensor)
                pred_values.append(value.item())
                true_values.append(z)
        mse = compute_value_error(np.array(pred_values), np.array(true_values))
        return mse
