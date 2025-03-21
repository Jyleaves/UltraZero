# parallel_selfplay.py

import os
import time
from tqdm import tqdm
import numpy as np
import torch
import multiprocessing
from multiprocessing import Manager

from model.train import UltraZeroModel
from selfplay.selfplay import SelfPlay
from selfplay.buffer import ReplayBuffer
from game.board import Board


def selfplay_worker(
    worker_id,           # 进程编号
    gpu_id,              # 绑定的GPU编号
    num_games,           # 要跑多少局
    config,              # 超参数配置
    progress_dict,       # 多进程共享的进度信息
    buffer_dir,          # 保存对局数据的目录
):
    """
    在指定 GPU 上跑 num_games 局自对弈，每完成一局就更新 progress_dict。
    同时将产生的 (state, pi, z) 存到本地 buffer 文件，也可直接写进全局队列。
    """

    # 可选：只暴露单卡给该进程（如果需要的话）
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    try:
        # 初始化模型 (可以加载已有权重)
        model = UltraZeroModel(**config['model']).to(device)
        model.eval()

        # 初始化自对弈对象
        selfplay = SelfPlay(model, device, **config['mcts'])

        # 本进程的临时 ReplayBuffer
        local_buffer = ReplayBuffer(max_size=100000)

        # 初始化共享字典里本进程的统计
        progress_dict[worker_id] = {
            'finished_games': 0,
            'x_wins': 0,
            'o_wins': 0,
            'draws': 0
        }

        x_wins = 0
        o_wins = 0
        draws = 0

        # 正式开始跑游戏
        for i in range(num_games):
            print(f"[Worker {worker_id}] Starting game {i+1}/{num_games} on GPU {gpu_id}...")
            board = Board()

            winner = selfplay.play_game(
                board,
                buffer=local_buffer,
                add_to_buffer=True,
                initial_add_dirichlet=True
            )
            print(f"[Worker {worker_id}] Finished game {i+1}/{num_games}, winner={winner}")

            # 更新本地统计
            if winner == 1:
                x_wins += 1
            elif winner == -1:
                o_wins += 1
            else:
                draws += 1

            # 完成一局后，实时更新共享进度
            progress_dict[worker_id]['finished_games'] += 1
            progress_dict[worker_id]['x_wins'] = x_wins
            progress_dict[worker_id]['o_wins'] = o_wins
            progress_dict[worker_id]['draws'] = draws

        # 把本进程的对局数据保存成一个 npz 文件
        buffer_path = os.path.join(buffer_dir, f"replay_worker_{worker_id}.npz")
        local_buffer.save(buffer_path)

        # 进程结束前可做收尾
        print(f"[Worker {worker_id}] DONE. Games={num_games}  X={x_wins}, O={o_wins}, D={draws}")
        print(f"[Worker {worker_id}] Data saved to {buffer_path}")

    except Exception as e:
        # 捕获子进程可能的异常并打印，方便调试
        print(f"[Worker {worker_id}] Error: {e}")
        raise e
