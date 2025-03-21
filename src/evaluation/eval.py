import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from multiprocessing import Manager, Process

from .metrics import compute_win_rate, compute_value_error
from src.mcts.mcts import MCTS
from src.mcts.node import Node


# 假设评估方式：
# 1. 利用给定模型对若干固定对局起点进行对战，直接使用policy头最大概率动作（不使用MCTS或使用较少次模拟）
# 2. 或使用两个模型对战统计胜率
# 3. 对一批已标注的state-value数据集来评估价值预测误差

class Evaluator:
    def __init__(self, model, num_games=100, use_mcts=False, mcts_simulations=200, dirichlet_alpha=0.03, root_dirichlet_frac=0.25,):
        self.model = model
        self.num_games = num_games
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.root_dirichlet_frac = root_dirichlet_frac
        self.device_id = 0

    def evaluate_win_rate(self, opponent_model, board_cls, initial_add_dirichlet=True):
        # 使用给定model与opponent_model对战num_games局，统计胜率
        # board_cls为创建Board实例的类名
        # use_mcts决定是否用MCTS搜索，否则直接用policy argmax落子
        # 返回(win_rate, draw_rate, loss_rate)

        results = []
        for i in tqdm(range(self.num_games), desc="Evaluating Win Rate"):
            board = board_cls()
            # 偶数局model先手(X=1)，奇数局opponent先手
            current_player_model = self.model if (i % 2 == 0) else opponent_model
            other_model = opponent_model if (i % 2 == 0) else self.model
            add_dirichlet = initial_add_dirichlet

            while not board.is_game_over():
                legal_moves = board.get_legal_moves()
                if not legal_moves:
                    # 无合法动作默认平局
                    board.game_over = True
                    board.winner = 0
                    break

                if self.use_mcts:
                    mcts = MCTS(self.device_id, current_player_model, simulations=self.mcts_simulations, c_puct=1.0)
                    root = Node(parent=None)
                    pi, root = mcts.search(root, board, np.array(legal_moves), add_dirichlet=add_dirichlet)
                    move = legal_moves[np.argmax(pi)]
                else:
                    # 直接用policy argmax
                    state_tensor = board.get_feature_tensor()
                    state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float().to(self.device_id)
                    with torch.no_grad(), autocast('cuda'):
                        policy_logits, _ = current_player_model(state_tensor)
                    policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy().squeeze()
                    # 仅在legal_moves中选argmax
                    legal_probs = policy_probs[legal_moves]
                    chosen_move = legal_moves[np.argmax(legal_probs)]
                    move = chosen_move

                add_dirichlet = False

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


def evaluate_win_rate_parallel(
        model,
        opponent_model,
        board_cls,
        num_games,
        num_workers,
        use_mcts=False,
        mcts_simulations=200,
        device_ids=None,
):
    """
    多进程并行评估胜率。
    每个进程分配一定数量的游戏，最终统计总结果。
    """
    manager = Manager()
    results_dict = manager.dict()

    # 初始化结果字典
    for worker_id in range(num_workers):
        results_dict[worker_id] = {"wins": 0, "losses": 0, "draws": 0}

    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    def worker_function(worker_id, num_games, results_dict):
        evaluator = Evaluator(
            model=model,
            num_games=num_games,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
        )
        win_rate, draw_rate, loss_rate = evaluator.evaluate_win_rate(
            opponent_model=opponent_model,
            board_cls=board_cls,
        )
        results_dict[worker_id] = {
            "wins": win_rate * num_games,
            "losses": loss_rate * num_games,
            "draws": draw_rate * num_games,
        }

    processes = []
    for worker_id in range(num_workers):
        n = games_per_worker + (1 if worker_id < remainder else 0)
        if n == 0:
            continue
        p = Process(
            target=worker_function,
            args=(worker_id, n, results_dict),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 汇总结果
    total_wins = sum(results_dict[worker_id]["wins"] for worker_id in range(num_workers))
    total_losses = sum(results_dict[worker_id]["losses"] for worker_id in range(num_workers))
    total_draws = sum(results_dict[worker_id]["draws"] for worker_id in range(num_workers))

    total_games = total_wins + total_losses + total_draws
    return total_wins / total_games, total_draws / total_games, total_losses / total_games