import os
# 设置可用的显卡编号
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from rl_mcts import Trainer


def main():
    # ==================== 训练规模 ====================
    batch_size = 512      # 减小batch_size以便更频繁更新
    episodes = 2000       # 增加训练轮数
    max_steps = 50        # 棋盘最大步数
    device = 'cuda'       # 'cuda' 或 'cpu'，None 自动选择

    # ==================== 优化器参数 ====================
    lr = 1e-4             # 降低学习率，A2C通常用较小的lr
    baseline_beta = 0.1   # (仅用于非A2C模式)

    # ==================== A2C相关参数（新增）====================
    use_a2c = True        # 使用A2C训练（推荐）
    gamma = 0.99          # 折扣因子
    value_loss_coef = 0.5 # 价值损失系数
    max_grad_norm = 0.5   # 梯度裁剪
    entropy_coef = 0.01   # 策略熵正则系数，鼓励探索

    # ==================== 模型与结果保存 ====================
    model_dir = "saved_models"
    results_dir = "results"
    model_name = "run_a2c.pth"
    csv_name = "run_a2c.csv"
    json_name = "run_a2c.json"

    # ==================== 网络架构 ====================
    stem_kernel_size = 5
    block_kernel_size = 3
    channels = 64
    num_layers = 5
    activation = "relu"
    bias = True

    # ==================== MCTS参数 ====================
    # 建议：先用纯A2C训练（关闭MCTS），等模型有基本策略后再开启MCTS
    mcts_num_simulations = 0   # 0表示关闭MCTS，使用纯策略网络
    mcts_max_depth = 6         # (MCTS开启时使用)
    c_puct = 1.5
    rollout_per_leaf = 16      # (MCTS开启时使用，但新版本用价值网络替代)
    rollout_max_moves = 40
    sample_policy = True       # 关闭MCTS时按策略采样动作
    force_win_move = True      # 优先选择一步致胜动作（硬编码规则）

    # ==================== 训练配置 ====================
    half_self_play = False     # A2C模式下建议False，双方都参与训练
    eval_interval_episodes = 10  # 每隔多少集进行一次新旧模型评测
    eval_games = 50            # 增加评测对弈局数（原来只有2局太少）
    replace_threshold = 0.55   # 新模型胜率至少达到该阈值才替换旧模型
    eval_log_name = "eval_log_a2c.csv"

    trainer = Trainer(
        batch_size=batch_size,
        lr=lr,
        episodes=episodes,
        max_steps=max_steps,
        device=device,
        baseline_beta=baseline_beta,
        entropy_coef=entropy_coef,
        stem_kernel_size=stem_kernel_size,
        block_kernel_size=block_kernel_size,
        channels=channels,
        num_layers=num_layers,
        activation=activation,
        bias=bias,
        mcts_num_simulations=mcts_num_simulations,
        mcts_max_depth=mcts_max_depth,
        c_puct=c_puct,
        rollout_per_leaf=rollout_per_leaf,
        rollout_max_moves=rollout_max_moves,
        use_policy_sampling=sample_policy,
        force_win_move=force_win_move,
        eval_interval_episodes=eval_interval_episodes,
        eval_games=eval_games,
        replace_threshold=replace_threshold,
        eval_log_name=eval_log_name,
        # A2C参数
        gamma=gamma,
        value_loss_coef=value_loss_coef,
        max_grad_norm=max_grad_norm,
        use_a2c=use_a2c,
    )
    trainer.train(
        model_dir=model_dir,
        results_dir=results_dir,
        model_name=model_name,
        csv_name=csv_name,
        json_name=json_name,
        half_self_play=half_self_play,
    )


if __name__ == "__main__":
    main()
