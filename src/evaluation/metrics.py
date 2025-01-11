import numpy as np


def compute_win_rate(results):
    # results: list或np.array，每个元素为胜负结果 {1:win,0:draw,-1:loss}
    # win_rate = (#win) / (#win+#loss+#draw)
    # 这里计算时draw只计入总对局数分母
    results = np.array(results)
    total_games = len(results)
    if total_games == 0:
        return 0.0, 0.0, 0.0
    wins = np.sum(results == 1)
    losses = np.sum(results == -1)
    draws = np.sum(results == 0)
    win_rate = wins / total_games
    draw_rate = draws / total_games
    loss_rate = losses / total_games
    return win_rate, draw_rate, loss_rate


def compute_value_error(pred_values, true_values):
    # pred_values, true_values: np.array, shape=(N,)
    # 均方误差(MSE)或MAE都可
    pred_values = np.array(pred_values)
    true_values = np.array(true_values)
    mse = np.mean((pred_values - true_values) ** 2)
    return mse
