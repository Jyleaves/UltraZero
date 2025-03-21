import numpy as np
from .rules import check_game_winner, check_subboard_winner
from .renderer import print_board


# 棋盘编码方式：X = 1, O = -1, 空 = 0
# 整个盘面为9x9，分为3x3大格，每格3x3小格
# current_player: 1表示X先手, -1表示O后手
# forced_subboard: 当前必须下子的小棋盘编号，范围0~8（表示0~8个子棋盘），-1表示无强制约束

class Board:
    def __init__(self):
        # 状态初始化
        # board_state: shape=(9,9), int8
        # 索引：board_state[big_row*3 + sub_row, big_col*3 + sub_col]
        self.board_state = np.zeros((9, 9), dtype=np.int8)
        self.current_player = 1  # X先手
        self.forced_subboard = -1  # 无强制限制
        self.game_over = False
        self.winner = 0  # 1=X胜, -1=O胜, 0=未分胜负(或平局)
        self.last_move = None  # 最近一次落子的位置，初始为None

    def copy(self):
        # 快速复制状态以便MCTS模拟
        new_board = Board()
        new_board.board_state = self.board_state.copy()
        new_board.current_player = self.current_player
        new_board.forced_subboard = self.forced_subboard
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.last_move = self.last_move
        return new_board

    def get_legal_moves(self):
        # 返回所有合法动作列表，每个动作为单个int，表示0~80的格子索引
        # 若存在forced_subboard则只能在相应小棋盘内走棋
        if self.game_over:
            return []
        if self.forced_subboard != -1:
            # 有强制小棋盘
            moves = self._get_legal_moves_in_subboard(self.forced_subboard)
            if not moves:
                # 如果强制小棋盘无合法步，则允许全盘任意空格下子
                return self._get_all_legal_moves()
            return moves
        else:
            # 无强制小棋盘，任意空格
            return self._get_all_legal_moves()

    def _get_all_legal_moves(self):
        # 返回整个9x9中空格的索引(0~80)，但排除那些已经完成的小棋盘中的空格
        legal_moves = []
        for sub_id in range(9):  # 遍历所有9个小棋盘
            # 检查该小棋盘是否已经完成
            if check_subboard_winner(self.board_state, sub_id) != 0:
                continue  # 如果小棋盘已经完成，跳过该小棋盘
            # 获取该小棋盘中的空格
            big_r = sub_id // 3
            big_c = sub_id % 3
            row_start = big_r * 3
            col_start = big_c * 3
            sub_slice = self.board_state[row_start:row_start + 3, col_start:col_start + 3]
            empty_positions = np.where(sub_slice.reshape(-1) == 0)[0]
            # 将子棋盘内部0~8的索引映射回全盘0~80
            for pos in empty_positions:
                sub_r = pos // 3
                sub_c = pos % 3
                global_pos = (row_start + sub_r) * 9 + (col_start + sub_c)
                legal_moves.append(global_pos)
        return legal_moves

    def _get_legal_moves_in_subboard(self, sub_id):
        # sub_id: 0~8，对应3x3大格中第sub_id个子格子
        # 子格子行列：(sub_id // 3, sub_id % 3)
        big_r = sub_id // 3
        big_c = sub_id % 3
        row_start = big_r * 3
        col_start = big_c * 3
        sub_slice = self.board_state[row_start:row_start + 3, col_start:col_start + 3]

        # 检查该子棋盘是否已经完成（有胜者或已填满）
        if check_subboard_winner(self.board_state, sub_id) != 0 or np.all(sub_slice != 0):
            return []  # 如果子棋盘已经完成，返回空列表

        empty_positions = np.where(sub_slice.reshape(-1) == 0)[0]
        # 将子棋盘内部0~8的索引映射回全盘0~80
        # 全局格子编号： (row_start + sub_row)*9 + (col_start + sub_col)
        moves = []
        for pos in empty_positions:
            sub_r = pos // 3
            sub_c = pos % 3
            global_pos = (row_start + sub_r) * 9 + (col_start + sub_c)
            moves.append(global_pos)
        return moves

    def apply_move(self, move, show_board=False):
        # move: 0~80
        if self.game_over:
            return
        r = move // 9
        c = move % 9
        if self.board_state[r, c] != 0:
            raise ValueError("Illegal move: cell already occupied")

        # 更新棋盘状态
        self.board_state[r, c] = self.current_player

        # 更新 last_move
        self.last_move = move

        # 确定下一步的forced_subboard
        sub_r = r % 3
        sub_c = c % 3
        next_forced = sub_r * 3 + sub_c

        # 判断此次落子后该子棋盘是否已决出胜负
        # 若整盘结束，更新game_over和winner
        result = check_game_winner(self.board_state)
        if result != 0:
            self.game_over = True
            self.winner = result
        else:
            # 检查是否平局
            all_subboards_finished = True
            for sub_id in range(9):
                sub_winner = check_subboard_winner(self.board_state, sub_id)
                if sub_winner == 0 and np.any(
                        self.board_state[sub_id // 3 * 3:sub_id // 3 * 3 + 3, sub_id % 3 * 3:sub_id % 3 * 3 + 3] == 0):
                    all_subboards_finished = False
                    break
            if all_subboards_finished:
                self.game_over = True
                self.winner = 0  # 平局
            else:
                # 检查落子决定下个forced_subboard是否可用（该子棋盘是否全满或决出胜负）
                if check_subboard_winner(self.board_state, next_forced) != 0 or \
                        not self._get_legal_moves_in_subboard(next_forced):
                    # 如果下一个子棋盘已满或已结束，则无强制子棋盘限制
                    self.forced_subboard = -1
                else:
                    self.forced_subboard = next_forced

        if show_board:
            print()
            print_board(self.board_state)

        # 切换玩家
        self.current_player = -self.current_player

    def get_feature_tensor(self):
        """
        将当前棋盘状态转换为一个 (5, 9, 9) 的特征张量。

        通道定义:
            0: X棋子位置(1)，非X(0)
            1: O棋子位置(1)，非O(0)
            2: 合法落子位置(1)，非法落子位置(0)
            3: 当前执棋方(X=1,O=-1)，填满整张9x9该通道为1表示X,为-1表示O
            4: 当前落子位置(1)，其余为0（若无落子则全0）
        """
        # 获取当前棋盘状态
        state = self.get_state()  # (9,9) int8

        # 通道0: X棋子位置
        x_channel = (state == 1).astype(np.float32)

        # 通道1: O棋子位置
        o_channel = (state == -1).astype(np.float32)

        # 通道2: 合法落子位置
        legal_moves = self.get_legal_moves()
        legal_mask = np.zeros((9, 9), dtype=np.float32)
        for move in legal_moves:
            r, c = move // 9, move % 9
            legal_mask[r, c] = 1.0

        # 通道3: 当前执棋方
        current_player_channel = np.full((9, 9), self.get_current_player(), dtype=np.float32)

        # 通道4: 当前落子位置
        last_move_channel = np.zeros((9, 9), dtype=np.float32)
        if self.last_move is not None:
            r, c = self.last_move // 9, self.last_move % 9
            last_move_channel[r, c] = 1.0

        # 将五个通道堆叠成一个 (5, 9, 9) 的张量
        feature_tensor = np.stack([x_channel, o_channel, legal_mask, current_player_channel, last_move_channel], axis=0)

        return feature_tensor

    def get_current_player(self):
        return self.current_player

    def get_state(self):
        # 返回当前盘面状态拷贝
        return self.board_state.copy()

    def get_last_move(self):
        """返回最近一次落子的位置（0~80），若无落子则返回None"""
        return self.last_move

    def is_game_over(self):
        return self.game_over

    def get_winner(self):
        return self.winner

    def render(self):
        print()
        print_board(self.board_state)
