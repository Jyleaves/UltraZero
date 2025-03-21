# UltraZero

## 项目简介

UltraZero 是一个基于强化学习与蒙特卡洛树搜索（MCTS）的开源棋类游戏项目，受到 AlphaZero 算法的启发。项目实现了自对弈、模型训练、评估以及人机交互等功能，并支持多进程并行自对弈与多 GPU 加速，以提高训练效率。

## 特性

- **自对弈**：自动生成对局数据，支持并行化和多 GPU 运算。
- **蒙特卡洛树搜索 (MCTS)**：通过 MCTS 算法提高搜索效率和决策质量。
- **模型训练**：基于 PyTorch 的神经网络训练，不断迭代优化模型性能。
- **人机交互**：提供人机对战模式，用户可通过命令行与 AI 进行对弈。
- **多模式支持**：项目支持训练、测试、对战、并行自对弈等多种运行模式。

## 目录结构

```
UltraZero/
├── data/
│   ├── pretrained_models/
│   └── replay_buffer/
├── src/
│   ├── config/            # 配置文件（如 hyperparams.yaml）
│   ├── evaluation/        # 模型评估模块
│   ├── game/              # 棋盘和游戏逻辑实现
│   ├── mcts/              # 蒙特卡洛树搜索相关代码
│   ├── model/             # 神经网络模型及训练代码
│   ├── selfplay/          # 自对弈模块
│   ├── utils/             # 工具函数
│   ├── __init__.py
│   ├── main.py            # 项目的主要入口文件
│   └── parallel_selfplay.py
├── LICENSE                # Apache License 2.0
├── README.md
└── requirements.txt       # 依赖列表
```

## 安装与依赖

1. **克隆仓库：**
   ```bash
   git clone <仓库地址>
   cd UltraZero
   ```

2. **创建虚拟环境并激活（可选）：**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

项目支持以下几种运行模式（通过命令行参数 `--mode` 指定）：

- **train**：进行模型训练。
- **test**：测试模型性能。
- **versus**：对战模式，比较两个模型的表现。
- **parallel_selfplay**：并行自对弈，生成训练数据。
- **interactive**：人机交互模式，与 AI 对弈。

例如，运行人机对战模式：
```bash
python main.py --mode interactive --model_path <模型文件路径>
```

如果你拥有多块 GPU，可以利用多进程并行自对弈的方式来加速数据生成与训练。项目支持通过 `parallel_selfplay` 模式进行并行自对弈，并允许指定多个 GPU 以及每个 GPU 上运行的进程数量。

例如，假设你有 4 块 GPU，并希望每块 GPU 启动 8 个并行进程来生成自对弈数据，你可以使用如下命令：

```bash
python main.py --mode parallel_selfplay --gpu_ids 0,1,2,3 --ppg 8
```

参数说明：  
- `--gpu_ids`：指定参与训练的 GPU 编号，多个编号之间使用逗号分隔（允许仅有1个GPU）。  
- `--ppg`：指定每个 GPU 上的并行进程数。  
- 其他配置项（例如 selfplay 的对局数量、超参数等）请参考配置文件 `config/hyperparams.yaml` 并根据需要调整。

该模式下，程序会自动在各个 GPU 上分配进程，并在命令行实时显示自对弈过程中 X、O 和平局的统计信息。当生成的对局达到预设数量后，数据会自动合并，并进入训练阶段。

## 贡献

欢迎大家为 UltraZero 项目贡献代码和意见！如果你有任何问题或建议，请提交 Issue 或 Pull Request。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议，欢迎在遵守许可证条款的前提下自由使用与修改。
