import numpy as np

# 为加速计算，将所有可能赢法（行、列、对角线）的坐标预先定义
# 小棋盘的赢法索引: 3x3共8条赢法 (行3, 列3, 对角2)
subboard_wins = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6]  # diagonals
]

# 大棋盘由9个子棋盘(0~8)组成，从board_state中抽取子棋盘时注意映射
# 全局索引(win lines for the big board)也是3x3格的类似8条赢法
bigboard_wins = subboard_wins


def check_subboard_winner(board_state, sub_id):
    # 检查指定子棋盘(sub_id)是否有胜利者
    # sub_id: 0~8
    # 根据board_state抽取对应3x3子棋盘
    big_r = sub_id // 3
    big_c = sub_id % 3
    sub = board_state[big_r * 3:big_r * 3 + 3, big_c * 3:big_c * 3 + 3].reshape(-1)
    for line in subboard_wins:
        vals = sub[line]
        if np.all(vals == 1):
            return 1
        elif np.all(vals == -1):
            return -1
    return 0


def check_game_winner(board_state):
    # 检查整盘结果，用9个子棋盘的胜负结果构成一个3x3大棋盘胜负阵列
    # 若大棋盘的某行/列/对角有3个相同的且不为0的子棋盘胜者标记则整盘结束
    big_board = np.zeros((3, 3), dtype=np.int8)
    for sub_id in range(9):
        w = check_subboard_winner(board_state, sub_id)
        if w != 0:
            big_board[sub_id // 3, sub_id % 3] = w
    # 检查big_board的胜负线
    for line in bigboard_wins:
        vals = [big_board[pos // 3, pos % 3] for pos in line]
        if all(v == 1 for v in vals):
            return 1
        elif all(v == -1 for v in vals):
            return -1
    return 0
