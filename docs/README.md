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
├── config/              # 配置文件（如 hyperparams.yaml）
├── evaluation/          # 模型评估模块
├── game/                # 棋盘和游戏逻辑实现
├── mcts/                # 蒙特卡洛树搜索相关代码
├── model/               # 神经网络模型及训练代码
├── selfplay/            # 自对弈模块
├── main.py              # 项目的入口文件
└── README.md            # 项目说明文档
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
   > 确保项目中包含 `requirements.txt` 文件，列出所有必要的依赖库，如 `torch`、`numpy`、`tqdm` 等。

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

## 贡献

欢迎大家为 UltraZero 项目贡献代码和意见！如果你有任何问题或建议，请提交 Issue 或 Pull Request。

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议，欢迎广泛使用与修改。
