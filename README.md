# 四子棋 AlphaZero 强化学习求解

## 游戏场景

在一个 8x8 的棋盘上进行四子棋对弈。棋盘四角被划分为八角形禁区（不可落子），双方轮流在空位自由放置棋子（无重力下落），先在横、竖或对角线方向连成 4 子的一方获胜。

## 项目架构

```
├── game.py              # 棋盘逻辑与游戏规则（Board、Game 类）
├── policy_value_net.py  # 策略-价值网络（PyTorch 实现）
├── mcts_alphaZero.py    # AlphaZero 风格的蒙特卡洛树搜索
├── mcts_pure.py         # 纯 MCTS（无神经网络，用作训练评估基准对手）
├── train.py             # 训练管线（自我博弈 + 策略梯度更新）
├── human_play.py        # 命令行人机对战
├── play_gui.py          # 图形化人机对战界面（含评分与概率可视化）
├── best_policy.model    # 训练好的最优模型
├── saved_models/        # 模型与检查点保存目录
├── AlphaZero_Gomoku/    # 参考实现（五子棋 AlphaZero）
└── _deprecated/         # 早期废弃版本（A2C + 手工特征）
```

## 核心算法

采用 AlphaZero 框架：一个 CNN 同时输出**策略概率**（落子分布）和**局面价值**（胜负评估），配合 MCTS 进行自我博弈训练。

- **策略损失**：交叉熵，让网络策略逼近 MCTS 搜索结果
- **价值损失**：均方误差，让网络预测最终胜负
- **训练流程**：自我博弈采集数据 → 数据增强（旋转+翻转）→ 小批量梯度更新
- **评估方式**：定期与纯 MCTS 对手对战，胜率提升时保存最优模型

主要算法参考自 [Junxiao Song/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) 仓库。

## 使用方法

**图形化人机对战（推荐）：**
```bash
conda activate pytorch311
python play_gui.py
```

**命令行人机对战：**
```bash
python human_play.py
```

**训练新模型：**
```bash
python train.py
```

## 依赖

- Python 3.11+
- PyTorch
- NumPy
- tkinter
