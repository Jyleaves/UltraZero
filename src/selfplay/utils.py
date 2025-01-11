import numpy as np


# 数据增强：翻转、旋转棋盘以增广训练数据，减少过拟合。
# 棋盘为(5,9,9)的输入（包含X棋子、O棋子、激活掩码、当前执棋方通道）
# 若只是增强状态，则对board_state(9,9)也可同理处理

def rotate_90(x):
    # x: (C,9,9)
    # 沿逆时针90度旋转
    # np.rot90默认逆时针旋转
    return np.rot90(x, k=1, axes=(1, 2))


def rotate_180(x):
    return np.rot90(x, k=2, axes=(1, 2))


def rotate_270(x):
    return np.rot90(x, k=3, axes=(1, 2))


def flip_horizontal(x):
    # 水平翻转
    return x[:, :, ::-1]


def flip_vertical(x):
    # 垂直翻转
    return x[:, ::-1, :]


def data_augmentations(x):
    # 给定一个状态张量x(C,9,9), 返回一组变换后的张量列表(包括原始)
    aug_list = [
        x,
        rotate_90(x),
        rotate_180(x),
        rotate_270(x),
        flip_horizontal(x),
        flip_vertical(x),
    ]
    return aug_list
