import os
# 设置可用的显卡编号
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from rl_mcts import Trainer


def main():
    # ==================== 训练规模 ====================
    batch_size = 256      # 适中大小
    episodes = 3000       # 更多训练轮数
    max_steps = 50
    device = 'cuda'

    # ==================== 优化器参数 ====================
    lr = 3e-4             # 提高学习率
    baseline_beta = 0.1

    # ==================== A2C相关参数 ====================
    use_a2c = True
    gamma = 0.99
    value_loss_coef = 1.0   # 提高价值损失权重
    max_grad_norm = 0.5
    entropy_coef = 0.05     # 增加探索

    # ==================== 模型与结果保存 ====================
    model_dir = "saved_models"
    results_dir = "results"
    model_name = "run_a2c_v2.pth"
    csv_name = "run_a2c_v2.csv"
    json_name = "run_a2c_v2.json"

    # ==================== 网络架构 ====================
    stem_kernel_size = 5
    block_kernel_size = 3
    channels = 128          # 增加网络容量
    num_layers = 6
    activation = "relu"
    bias = True

    # ==================== MCTS参数 ====================
    mcts_num_simulations = 0   # 先不用MCTS
    mcts_max_depth = 6
    c_puct = 1.5
    rollout_per_leaf = 16
    rollout_max_moves = 40
    sample_policy = True
    force_win_move = True

    # ==================== 训练配置 ====================
    half_self_play = False
    eval_interval_episodes = 20  # 减少评测频率
    eval_games = 100             # 增加评测局数
    replace_threshold = 0.52     # 降低替换阈值
    eval_log_name = "eval_log_a2c_v2.csv"

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
