from rl_mcts import Trainer


def main():
    # 训练规模
    batch_size = 8
    episodes = 100
    max_steps = 40
    device = None  # 'cuda' 或 'cpu'，None 自动选择

    # 优化与基线
    lr = 1e-3
    baseline_beta = 0.1

    # 模型与结果保存
    model_dir = "saved_models"
    results_dir = "results"
    model_name = "run1.pth"
    csv_name = "run1.csv"
    json_name = "run1.json"

    # Agent 网络与MCTS超参
    stem_kernel_size = 5
    block_kernel_size = 3
    channels = 64
    num_layers = 4
    activation = "relu"  # 或 'relu'
    bias = True
    mcts_num_simulations = 256
    mcts_max_depth = 8
    c_puct = 1.5
    rollout_per_leaf = 32
    rollout_max_moves = 40

    trainer = Trainer(
        batch_size=batch_size,
        lr=lr,
        episodes=episodes,
        max_steps=max_steps,
        device=device,
        baseline_beta=baseline_beta,
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
    )
    trainer.train(
        model_dir=model_dir,
        results_dir=results_dir,
        model_name=model_name,
        csv_name=csv_name,
        json_name=json_name,
    )


if __name__ == "__main__":
    main()
