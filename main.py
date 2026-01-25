import os
# 设置可用的显卡编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from rl_mcts import Trainer


def main():
    # 训练规模
    batch_size = 2048
    episodes = 10000
    max_steps = 40
    device = 'cuda'  # 'cuda' 或 'cpu'，None 自动选择

    # 优化与基线
    lr = 3e-4
    baseline_beta = 0.1

    # 模型与结果保存
    model_dir = "saved_models"
    results_dir = "results"
    model_name = "run2.pth"
    csv_name = "run2.csv"
    json_name = "run2.json"

    # Agent 网络与MCTS超参
    stem_kernel_size = 5
    block_kernel_size = 3
    channels = 64
    num_layers = 5
    activation = "relu"  # 或 'relu'
    bias = True
    mcts_num_simulations = 0
    mcts_max_depth = 0
    c_puct = 1.5
    rollout_per_leaf = 0
    rollout_max_moves = 0
    sample_policy = True  # True 时按策略采样动作，关闭MCTS
    half_self_play = True  # True 时半监督：前半批训练先手，后半批训练后手
    force_win_move = True  # True 时优先选择一步致胜动作
    eval_interval_episodes = 20  # 每隔多少集进行一次新旧模型评测
    eval_games = 512  # 评测对弈局数
    replace_threshold = 0.55  # 新模型胜率至少达到该阈值才替换旧模型

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
        use_policy_sampling=sample_policy,
        force_win_move=force_win_move,
        eval_interval_episodes=eval_interval_episodes,
        eval_games=eval_games,
        replace_threshold=replace_threshold,
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
