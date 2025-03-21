import sys
import argparse
import os
import torch
import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Manager

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from model.train import UltraZeroModel, train_model, UltraZeroDataset
from game.board import Board
from selfplay.selfplay import SelfPlay
from selfplay.buffer import ReplayBuffer
from mcts.mcts import MCTS
from mcts.node import Node
from evaluation.eval import Evaluator
from config.hyperparams import load_config


def selfplay_worker(
    worker_id,           # 进程编号
    gpu_id,              # 绑定的GPU编号
    num_games,           # 要跑多少局
    config,              # 超参数配置
    model_path,
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
        # # 1) 初始化模型(空壳)
        # model = UltraZeroModel(**config['model'])
        #
        # # 2) 从主进程传来的模型文件加载权重到CPU，再放到指定GPU
        # if model_path:
        #     state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        #     model.load_state_dict(state_dict)
        # model.to(device)
        # model.eval()
        model = None

        # 初始化自对弈对象
        selfplay = SelfPlay(model, gpu_id, **config['mcts'])

        # 本进程的临时 ReplayBuffer
        local_buffer = ReplayBuffer(max_size=100000)

        # 初始化共享字典里本进程的统计
        progress_dict[f"{worker_id}_finished_games"] = 0
        progress_dict[f"{worker_id}_x_wins"] = 0
        progress_dict[f"{worker_id}_o_wins"] = 0
        progress_dict[f"{worker_id}_draws"] = 0

        x_wins = 0
        o_wins = 0
        draws = 0

        # 正式开始跑游戏
        for i in range(num_games):
            board = Board()

            winner = selfplay.play_game(
                board,
                buffer=local_buffer,
                add_to_buffer=True,
                initial_add_dirichlet=True
            )

            # 更新本地统计
            if winner == 1:
                x_wins += 1
            elif winner == -1:
                o_wins += 1
            else:
                draws += 1

            # 完成一局后，实时更新共享进度
            progress_dict[f"{worker_id}_finished_games"] += 1
            progress_dict[f"{worker_id}_x_wins"] = x_wins
            progress_dict[f"{worker_id}_o_wins"] = o_wins
            progress_dict[f"{worker_id}_draws"] = draws

        # 把本进程的对局数据保存成一个 npz 文件
        buffer_path = os.path.join(buffer_dir, f"replay_worker_{worker_id}.npz")
        local_buffer.save(buffer_path)

        # 进程结束前可做收尾
        print(f"[Worker {worker_id}] DONE. Games={num_games}  X={x_wins}, O={o_wins}, D={draws}")

    except Exception as e:
        # 捕获子进程可能的异常并打印，方便调试
        print(f"[Worker {worker_id}] Error: {e}")
        raise e


def run_parallel_selfplay(
    total_games,
    gpu_ids,
    config,
    buffer_dir,
    model_path,
    processes_per_gpu=8,  # 每个 GPU 上运行的进程数量
):
    """
    启动多个进程并行自对弈，实时显示总进度和 X/O/平局统计。
    当进度 100% 后，返回保存对局数据的文件列表。
    """
    manager = Manager()
    progress_dict = manager.dict()

    # 初始化 progress_dict
    total_processes = len(gpu_ids) * processes_per_gpu
    for i in range(total_processes):
        progress_dict[f"{i}_finished_games"] = 0
        progress_dict[f"{i}_x_wins"] = 0
        progress_dict[f"{i}_o_wins"] = 0
        progress_dict[f"{i}_draws"] = 0

    processes = []
    games_per_process = total_games // total_processes
    remainder = total_games % total_processes

    start_idx = 0
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        for process_idx in range(processes_per_gpu):
            n = games_per_process + (1 if start_idx < remainder else 0)
            if n == 0:
                continue
            p = multiprocessing.Process(
                target=selfplay_worker,
                args=(start_idx, gpu_id, n, config, model_path, progress_dict, buffer_dir)
            )
            processes.append(p)
            p.start()
            start_idx += 1

    # 在主进程里，用 tqdm 显示进度
    pbar = tqdm(total=total_games, desc="Parallel SelfPlay", ncols=80)
    last_finished = 0

    # 不断查询 progress_dict，更新进度条，直到完成
    while True:
        finished = 0
        x_wins_total = 0
        o_wins_total = 0
        draws_total = 0

        for worker_id in range(total_processes):
            finished += progress_dict[f"{worker_id}_finished_games"]
            x_wins_total += progress_dict[f"{worker_id}_x_wins"]
            o_wins_total += progress_dict[f"{worker_id}_o_wins"]
            draws_total += progress_dict[f"{worker_id}_draws"]

        # 更新进度条
        step = finished - last_finished
        if step > 0:
            pbar.update(step)
            last_finished = finished

        # 更新进度条的描述，实时显示 X/O/平局统计
        pbar.set_description(f"X={x_wins_total}, O={o_wins_total}, D={draws_total}")

        if finished >= total_games:
            break
        time.sleep(0.5)  # 控制刷新频率

    pbar.close()

    # 等待所有子进程结束
    for p in processes:
        p.join()

    # 收集生成的文件列表，过滤文件名，确保编号在 worker_id 范围内
    valid_worker_ids = set(range(total_processes))
    files = [
        os.path.join(buffer_dir, f)
        for f in os.listdir(buffer_dir)
        if f.startswith("replay_worker_") and int(f.split("_")[-1].split(".")[0]) in valid_worker_ids
    ]
    return files


def interactive_play(model, board_cls, device, mcts_simulations):
    """
    让人类与模型进行可交互对弈的示例函数。
    :param model: 已加载好权重的 UltraZeroModel
    :param board_cls: 棋盘类，用于创建新棋局
    :param device: torch.device
    """
    board = board_cls()  # 初始化一个空棋盘

    # 询问玩家执 X 还是 O
    while True:
        player_side = input("请选择你的棋子 (X 或 O): ").strip().upper()
        if player_side in ["X", "O"]:
            break
        print("输入不合法，请输入 'X' 或者 'O'。")

    # X 对应 winner=1，O 对应 winner=-1
    # 如果玩家选 X，则玩家先手；否则 AI 先手
    human_is_x = (player_side == "X")

    # 游戏循环
    while not board.is_game_over():
        # 先打印当前棋盘
        board.render()

        # 判断当前轮到谁： board.current_player 可能是 1 表示 X，下一个落子就是 X
        # 如果和 human_is_x 对应，则是人类回合，否则是 AI 回合
        if (board.current_player == 1 and human_is_x) or (board.current_player == -1 and not human_is_x):
            # 人类回合
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                print("无合法落子点，游戏结束！")
                break

            # 获取用户输入“大格 小格”，并转成 move
            while True:
                raw_input = input("请下棋，大格 小格 (0~8 之间，用空格分隔): ").strip()
                try:
                    big_sq, small_sq = map(int, raw_input.split())
                    if not (0 <= big_sq <= 8 and 0 <= small_sq <= 8):
                        raise ValueError
                    # 将大格、子格映射到 [0..80] 的线性坐标
                    move = (big_sq // 3)*27 + (big_sq % 3)*3 + (small_sq // 3)*9 + (small_sq % 3)
                    if move in legal_moves:
                        board.apply_move(move)
                        break
                    else:
                        print("非法落子，请检查你输入的大格和小格是否符合当前规则。")
                except ValueError:
                    print("输入格式不正确，请输入形如 '1 3' 这样的数字。")
        else:
            # AI 回合
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                print("无合法落子点，游戏结束！")
                break

            # 简单示例：这里直接用 policy argmax，下棋前不加 Dirichlet 噪声
            state_tensor = board.get_feature_tensor()
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                _, value = model(state_tensor)
                value_number = value.item()
            # 如果你想用 MCTS，可以按照 selfplay/eval 里的写法组建 MCTS 搜索
            mcts = MCTS(0, model, simulations=mcts_simulations, c_puct=1.0)
            root = Node(parent=None)
            pi, root = mcts.search(root, board, np.array(legal_moves), add_dirichlet=False)
            move = legal_moves[np.argmax(pi)]
            board.apply_move(move)
            print(f"\nAI 落子到位置: {move}, AI胜率: {(value_number + 1) / 2 * 100}%\n")
            state_tensor = board.get_feature_tensor()
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                _, value = model(state_tensor)
                value_number = value.item()
            print(f"\nAI 落子到位置: {move}, AI对手胜率: {(value_number + 1) / 2 * 100}%\n")

    # 对局结束后，打印最终棋盘和结果
    winner = board.get_winner()
    if winner == 1:
        print("X 胜利!")
    elif winner == -1:
        print("O 胜利!")
    else:
        print("平局!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='interactive', choices=['train', 'test', 'versus', 'parallel_selfplay', 'interactive'], help='Run mode')
    parser.add_argument('--config', type=str, default='config/hyperparams.yaml', help='Path to config file')
    parser.add_argument('--model_dir', type=str, default='../data/pretrained_models', help='Directory to save models')
    parser.add_argument('--replay_path', type=str, default='../data/replay_buffer')
    parser.add_argument('--games', type=int, default=10, help='Number of games for evaluation')
    parser.add_argument('--model_path', type=str, default='../data/pretrained_models/model_cycle_66.pt', help='Path to the model to load for interactive mode')
    parser.add_argument('--model1_path', type=str, default='../data/pretrained_models/model_cycle_1.pt',
                        help='Path to the first model for versus mode')
    parser.add_argument('--model2_path', type=str, default='../data/pretrained_models/model_cycle_16.pt',
                        help='Path to the second model for versus mode')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs to use for parallel selfplay, e.g. "0,1,2".')
    parser.add_argument('--ppg', type=int, default=6, help='Process per GPU.')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'train':
        # 创建模型保存目录
        os.makedirs(args.model_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化模型
        model = UltraZeroModel(**config['model']).to(device)

        # 检查是否存在最新的模型和轮次信息
        latest_cycle_file = os.path.join(args.model_dir, 'latest_cycle.txt')
        if os.path.exists(latest_cycle_file):
            with open(latest_cycle_file, 'r') as f:
                latest_cycle = int(f.read().strip())
            latest_model_path = os.path.join(args.model_dir, f'model_cycle_{latest_cycle}.pt')
            if os.path.exists(latest_model_path):
                model.load_state_dict(torch.load(latest_model_path, map_location=device, weights_only=True))
                print(f"Resuming training from cycle {latest_cycle + 1}")
        else:
            latest_cycle = -1  # 从第 0 轮开始

        # 初始化旧模型用于评估
        old_model = UltraZeroModel(**config['model']).to(device)
        old_model.load_state_dict(model.state_dict())

        # 获取配置中的循环次数和每次selfplay的游戏次数
        num_cycles = config['selfplay']['num_cycles']
        selfplay_games_per_cycle = config['selfplay']['games_per_cycle']

        for cycle in range(latest_cycle + 1, num_cycles):
            print(f"\nCycle {cycle + 1}/{num_cycles}\n")

            # Selfplay 阶段
            buffer = ReplayBuffer(max_size=500000)
            model.eval()
            selfplay = SelfPlay(model, device_id=0, **config['mcts'])

            # 初始化统计信息
            x_wins = 0
            o_wins = 0
            draws = 0

            # 使用 tqdm 包装 selfplay_games_per_cycle 的循环
            with tqdm(total=selfplay_games_per_cycle, desc="Selfplay Games (X wins: 0, O wins: 0, Draws: 0)",
                      unit="game") as pbar:
                for _ in range(selfplay_games_per_cycle):
                    board = Board()
                    winner = selfplay.play_game(board, buffer=buffer, add_to_buffer=True)

                    # 更新统计信息
                    if winner == 1:  # X 赢
                        x_wins += 1
                    elif winner == -1:  # O 赢
                        o_wins += 1
                    else:  # 平局
                        draws += 1

                    # 更新进度条描述
                    pbar.set_description(f"Selfplay Games (X wins: {x_wins}, O wins: {o_wins}, Draws: {draws})")
                    pbar.update(1)

            # 保存当前 cycle 的 buffer
            buffer_path = os.path.join(args.replay_path, f'buffer_cycle_{cycle}.npz')  # 生成具体的文件名
            buffer.save(buffer_path)
            print(f"\nSaved replay buffer to {buffer_path}\n")

            # 训练阶段
            data = buffer.get_data()  # 获取 buffer 中的数据
            dataset = UltraZeroDataset(data)
            model.train()
            train_model(model, dataset, gpu_ids=[0], **config['training'])

            # 保存当前轮次的模型
            model_path = os.path.join(args.model_dir, f'model_cycle_{cycle}.pt')
            torch.save(model.state_dict(), model_path)

            # 更新最新的轮次信息
            with open(latest_cycle_file, 'w') as f:
                f.write(str(cycle))

            # 每 n 次循环后进行一次评估
            if (cycle + 1) % config['evaluation']['eval_interval'] == 0:
                print("\nEvaluating models...\n")
                evaluator = Evaluator(
                    model=model,
                    num_games=config['evaluation']['num_games'],
                    use_mcts=config['evaluation']['use_mcts'],
                    mcts_simulations=config['evaluation']['mcts_simulations']
                )
                win_rate, draw_rate, loss_rate = evaluator.evaluate_win_rate(opponent_model=old_model, board_cls=Board)
                print(
                    f"\nWin rate against previous model: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, Loss rate: {loss_rate:.2f}\n")

                # 更新旧模型
                old_model.load_state_dict(model.state_dict())

    elif args.mode == 'test':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 评估模型胜率或价值误差
        model = UltraZeroModel(**config['model']).to(device)
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.eval()

        evaluator = Evaluator(model, num_games=100, use_mcts=True, mcts_simulations=500)
        # 与非神经网络MCTS对战测试胜率
        win_rate, draw_rate, loss_rate = evaluator.evaluate_win_rate(opponent_model=None, board_cls=Board)
        print(f"\nWin rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, Loss rate: {loss_rate:.2f}\n")

    elif args.mode == 'versus':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 检查是否提供了两个模型的路径
        if not args.model1_path or not args.model2_path:
            raise ValueError("Both --model1_path and --model2_path must be provided for versus mode")

        # 加载两个模型
        model1 = UltraZeroModel(**config['model']).to(device)
        model1.load_state_dict(torch.load(args.model1_path, map_location=device, weights_only=True))
        model1.eval()

        model2 = UltraZeroModel(**config['model']).to(device)
        model2.load_state_dict(torch.load(args.model2_path, map_location=device, weights_only=True))
        model2.eval()

        # 初始化评估器
        evaluator = Evaluator(
            model=model1,
            num_games=args.games,
            use_mcts=config['evaluation']['use_mcts'],
            mcts_simulations=config['evaluation']['mcts_simulations']
        )

        # 进行对弈测试
        win_rate, draw_rate, loss_rate = evaluator.evaluate_win_rate(opponent_model=model2, board_cls=Board)
        print(f"\nModel 1 Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, Loss rate: {loss_rate:.2f}\n")

    # ----------------------
    # 并行自对弈 (多进程 + 多卡)
    # ----------------------
    elif args.mode == 'parallel_selfplay':
        """
                这里可演示一个类似AlphaZero多轮循环：
                  每一轮：
                    a) 并行自对弈 (run_parallel_selfplay)
                    b) 合并数据 => 训练
                    c) 评估 => 下轮
                """
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        processes_per_gpu = args.ppg
        os.makedirs('./tmp_buffers', exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)

        model = UltraZeroModel(**config['model']).cuda()
        model_path = ""

        # 检查是否存在最新的模型和轮次信息
        latest_cycle_file = os.path.join(args.model_dir, 'latest_cycle.txt')
        if os.path.exists(latest_cycle_file):
            with open(latest_cycle_file, 'r') as f:
                latest_cycle = int(f.read().strip())
            latest_model_path = os.path.join(args.model_dir, f'model_cycle_{latest_cycle}.pt')
            if os.path.exists(latest_model_path):
                model_path = latest_model_path
                print(f"Resuming training from cycle {latest_cycle + 1}")
        else:
            latest_cycle = -1  # 从第 0 轮开始

        # 获取配置中的循环次数和每次selfplay的游戏次数
        num_cycles = config['selfplay']['num_cycles']

        for cycle in range(latest_cycle + 1, num_cycles):
            print(f"\n=== Cycle {cycle + 1}/{config['selfplay']['num_cycles']} ===\n")

            # a) 并行自对弈
            buffer_files = run_parallel_selfplay(
                total_games=config['selfplay']['games_per_cycle'],
                gpu_ids=gpu_ids,
                config=config,
                buffer_dir='./tmp_buffers',
                model_path=model_path,
                processes_per_gpu=processes_per_gpu,
            )

            # b) 合并数据并训练
            big_buffer = ReplayBuffer(max_size=500000)
            for bf in buffer_files:
                big_buffer.load_and_merge(bf)
            buffer_path = os.path.join(args.replay_path, f'buffer_cycle_{cycle}.npz')  # 生成具体的文件名
            big_buffer.save(buffer_path)

            data = big_buffer.get_data()
            dataset = UltraZeroDataset(data)
            train_model(model, dataset, gpu_ids, **config['training'])

            # 保存当前轮次的模型
            model_path = os.path.join(args.model_dir, f'model_cycle_{cycle}.pt')
            torch.save(model.state_dict(), model_path)

            # 更新最新的轮次信息
            with open(latest_cycle_file, 'w') as f:
                f.write(str(cycle))

        print("All cycles done!")
    elif args.mode == 'interactive':
        """
        新增的人机交互模式
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 如果用户没有提供 model_path，需要提示一下
        if not args.model_path or not os.path.exists(args.model_path):
            raise ValueError("必须提供一个可用的 --model_path 来进行人机对弈")

        # 1. 加载模型
        model = UltraZeroModel(**config['model']).to(device)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"已加载模型: {args.model_path}")

        # 2. 进入交互式对战
        interactive_play(model, Board, device, mcts_simulations=800)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
