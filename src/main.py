import argparse
import os
import torch
from tqdm import tqdm

from model.train import UltraZeroModel, train_model, UltraZeroDataset
from game.board import Board
from selfplay.selfplay import SelfPlay
from selfplay.buffer import ReplayBuffer
from evaluation.eval import Evaluator
from config.hyperparams import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], help='Run mode')
    parser.add_argument('--config', type=str, default='src/config/hyperparams.yaml', help='Path to config file')
    parser.add_argument('--model_dir', type=str, default='data/pretrained_models/', help='Directory to save models')
    parser.add_argument('--replay_path', type=str, default='data/replay_buffer/')
    parser.add_argument('--games', type=int, default=10, help='Number of games for selfplay or evaluation')
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        # 创建模型保存目录
        os.makedirs(args.model_dir, exist_ok=True)

        # 初始化模型
        model = UltraZeroModel(**config['model']).to(device)

        # 检查是否存在最新的模型和轮次信息
        latest_cycle_file = os.path.join(args.model_dir, 'latest_cycle.txt')
        if os.path.exists(latest_cycle_file):
            with open(latest_cycle_file, 'r') as f:
                latest_cycle = int(f.read().strip())
            latest_model_path = os.path.join(args.model_dir, f'model_cycle_{latest_cycle}.pt')
            if os.path.exists(latest_model_path):
                model.load_state_dict(torch.load(latest_model_path, map_location=device))
                print(f"Resuming training from cycle {latest_cycle + 1}")
        else:
            latest_cycle = -1  # 从第 0 轮开始

        # 初始化旧模型用于评估
        old_model = UltraZeroModel(**config['model']).to(device)
        old_model.load_state_dict(model.state_dict())

        # 获取配置中的循环次数和每次selfplay的游戏次数
        num_cycles = config['training']['num_cycles']
        selfplay_games_per_cycle = config['selfplay']['games_per_cycle']

        for cycle in range(latest_cycle + 1, num_cycles):
            print(f"\nCycle {cycle + 1}/{num_cycles}\n")

            # Selfplay 阶段
            buffer = ReplayBuffer(max_size=500000)
            selfplay = SelfPlay(model, **config['mcts'])

            # 使用 tqdm 包装 selfplay_games_per_cycle 的循环
            for _ in tqdm(range(selfplay_games_per_cycle), desc="Selfplay Games", unit="game"):
                board = Board()
                selfplay.play_game(board, buffer=buffer, add_to_buffer=True)

            # 保存当前 cycle 的 buffer
            buffer_path = os.path.join(args.buffer_path, f'buffer_cycle_{cycle}.npz')  # 生成具体的文件名
            buffer.save(buffer_path)
            print(f"\nSaved replay buffer to {buffer_path}\n")

            # 训练阶段
            data = buffer.get_data()  # 获取 buffer 中的数据
            dataset = UltraZeroDataset(data)
            train_model(model, dataset, **config['training'])

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
                print(f"\nWin rate against previous model: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, Loss rate: {loss_rate:.2f}\n")

                # 更新旧模型
                old_model.load_state_dict(model.state_dict())

    elif args.mode == 'test':
        # 评估模型胜率或价值误差
        model = UltraZeroModel(**config['model']).to(device)
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        evaluator = Evaluator(model, num_games=100, use_mcts=True, mcts_simulations=500)
        # 与非神经网络MCTS对战测试胜率
        win_rate, draw_rate, loss_rate = evaluator.evaluate_win_rate(opponent_model=None, board_cls=Board)
        print(f"\nWin rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, Loss rate: {loss_rate:.2f}\n")


if __name__ == "__main__":
    main()
