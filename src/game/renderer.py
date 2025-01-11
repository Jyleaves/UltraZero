import numpy as np


# 渲染棋盘为文本。为减小开销，不使用复杂花哨的渲染，只用简单文本表示。
# X = 'X', O = 'O', 空格 = '.'
# 使用分隔符区分大棋盘(3x3)之间的界限

def render_board(board_state):
    # board_state: shape=(9,9)
    # 转换为字符
    char_map = {0: '.', 1: 'X', -1: 'O'}
    lines = []
    for i in range(9):
        if i % 3 == 0 and i != 0:
            lines.append("------+-------+------")
        row_chars = []
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_chars.append("|")
            row_chars.append(char_map[board_state[i, j]])
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


def print_board(board_state):
    print(render_board(board_state))
