import os
import csv
import torch
import torch.nn.functional as F
import json
from environment import BatchTracker
from agent import Agent


class Trainer:
    def __init__(
        self,
        batch_size=32,
        lr=1e-3,
        episodes=100,
        max_steps=40,
        device=None,
        baseline_beta=0.1,
        entropy_coef: float = 0.0,
        stem_kernel_size: int = 5,
        block_kernel_size: int = 3,
        channels: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        bias: bool = True,
        mcts_num_simulations: int = 256,
        mcts_max_depth: int = 8,
        c_puct: float = 1.5,
        rollout_per_leaf: int = 16,
        rollout_max_moves: int = 64,
        use_policy_sampling: bool = False,
        force_win_move: bool = True,
        eval_interval_episodes: int = 10,
        eval_games: int = 20,
        replace_threshold: float = 0.5,
        eval_log_name: str = "eval_log.csv",
        # A2C相关参数
        gamma: float = 0.99,  # 折扣因子
        value_loss_coef: float = 0.5,  # 价值损失系数
        max_grad_norm: float = 0.5,  # 梯度裁剪
        use_a2c: bool = True,  # 是否使用A2C训练
    ):
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.episodes = int(episodes)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = BatchTracker(self.batch_size, device=self.device, max_steps=max_steps)
        self.agent = Agent(
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
            use_policy_sampling=use_policy_sampling,
            force_win_move=force_win_move,
            device=self.device,
        )
        self.opponent = Agent(
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
            use_policy_sampling=use_policy_sampling,
            force_win_move=force_win_move,
            device=self.device,
        )
        self.opponent.load_state_dict(self.agent.state_dict())
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=self.lr)

        # A2C相关
        self.gamma = float(gamma)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.use_a2c = bool(use_a2c)

        # 保留旧的baseline参数（用于非A2C模式）
        self.baseline_off = 0.0
        self.baseline_def = 0.0
        self.baseline_beta = float(baseline_beta)
        self.entropy_coef = float(entropy_coef)
        self.eval_interval_episodes = int(eval_interval_episodes)
        self.eval_games = int(eval_games)
        self.replace_threshold = float(replace_threshold)
        # eval log init
        self.eval_log_name = eval_log_name
        self.results_dir = None

    def train(self, model_dir="saved_models", results_dir="results", model_name="model.pth", csv_name="results.csv", json_name="episodes.json", half_self_play: bool = False):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        csv_path = os.path.join(results_dir, csv_name)
        json_path = os.path.join(results_dir, json_name)
        eval_log_path = os.path.join(results_dir, self.eval_log_name)
        csv_exists = os.path.exists(csv_path)
        if not csv_exists:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if self.use_a2c:
                    writer.writerow(["episode", "off_win_rate", "draw_rate", "def_win_rate", "policy_loss", "value_loss", "entropy_loss", "total_loss"])
                else:
                    writer.writerow(["episode", "off_win_rate", "draw_rate", "def_win_rate", "loss"])
        if not os.path.exists(eval_log_path):
            with open(eval_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "new_total_win_rate", "old_total_win_rate", "new_first_win_rate", "new_second_win_rate", "old_first_win_rate", "old_second_win_rate", "eval_games"])
        episodes_out = []

        for ep in range(1, self.episodes + 1):
            print(f"episode:{ep}")
            self.env.reset()

            if self.use_a2c:
                # A2C训练模式
                losses = self._train_episode_a2c(half_self_play)
                policy_loss, value_loss, entropy_loss, total_loss = losses
                off_rate = float((self.env.winners == 1).sum().item()) / float(self.batch_size)
                draw_rate = float((self.env.winners == 0).sum().item()) / float(self.batch_size)
                def_rate = float((self.env.winners == -1).sum().item()) / float(self.batch_size)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([ep, off_rate, draw_rate, def_rate, policy_loss, value_loss, entropy_loss, total_loss])
                print(f"[episode={ep}] policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} entropy_loss={entropy_loss:.4f} total_loss={total_loss:.4f}")
            else:
                # 原始训练模式（保留向后兼容）
                loss, steps = self._train_episode_original(half_self_play)
                winners = self.env.winners.to(self.device).float()
                off_rate = float((winners == 1).sum().item()) / float(self.batch_size)
                draw_rate = float((winners == 0).sum().item()) / float(self.batch_size)
                def_rate = float((winners == -1).sum().item()) / float(self.batch_size)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([ep, off_rate, draw_rate, def_rate, float(loss.item())])
                print(f"[episode={ep}] steps={steps} loss={float(loss.item()):.6f}")

            # 记录episode信息
            episodes_out.append({
                "episode": ep,
                "winner": int(self.env.winners[0].item()),
                "steps": int(self.env.step_count[0].item()),
            })

            # 周期评测与替换
            if self.eval_interval_episodes > 0 and ep % self.eval_interval_episodes == 0:
                metrics = self._evaluate_agents(self.eval_games)
                with open(eval_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        ep,
                        metrics["new_total_win_rate"],
                        metrics["old_total_win_rate"],
                        metrics["new_first_win_rate"],
                        metrics["new_second_win_rate"],
                        metrics["old_first_win_rate"],
                        metrics["old_second_win_rate"],
                        int(self.eval_games),
                    ])
                if metrics["new_total_win_rate"] > metrics["old_total_win_rate"] and metrics["new_total_win_rate"] >= self.replace_threshold:
                    self.opponent.load_state_dict(self.agent.state_dict())
                    torch.save(self.agent.state_dict(), os.path.join(model_dir, model_name))

        torch.save(self.agent.state_dict(), os.path.join(model_dir, model_name))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": episodes_out}, f, ensure_ascii=False)
        return True

    def _train_episode_a2c(self, half_self_play: bool):
        """
        A2C训练一个episode
        核心思想：使用价值网络作为Critic，计算TD-error作为advantage
        """
        # 存储轨迹数据
        log_probs_list = []  # 每步的log概率
        values_list = []     # 每步的价值估计
        entropies_list = []  # 每步的熵
        sides_list = []      # 每步的玩家（1或-1）
        rewards_list = []    # 每步的即时奖励
        dones_list = []      # 每步是否结束

        steps = 0
        while True:
            state = self.env.state()
            side = self.env.turn().to(torch.int32)

            # 获取动作、策略、熵、价值
            actions, probs, entropy, value = self.agent(state, side, self.env)
            log_prob = torch.log(probs + 1e-9)

            # 存储当前步的数据
            log_probs_list.append(log_prob)
            values_list.append(value)
            entropies_list.append(entropy)
            sides_list.append(side.float())

            # 执行动作
            next_state, done, info = self.env.step(actions)

            # 计算即时奖励（游戏结束时才有非零奖励）
            winners = self.env.winners.to(self.device).float()
            # 从当前玩家视角的奖励
            reward = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                if done[i]:
                    # 当前玩家赢了 = +1, 输了 = -1, 平局 = 0
                    reward[i] = side[i].float() * winners[i]
            rewards_list.append(reward)
            dones_list.append(done.float())

            steps += 1
            if self.env.all_done() or steps >= self.env.max_steps:
                break

        # 计算returns和advantages
        T = len(log_probs_list)

        # 获取最后状态的价值估计（用于bootstrapping）
        if not self.env.all_done():
            with torch.no_grad():
                final_state = self.env.state()
                final_side = self.env.turn().to(torch.int32)
                _, _, _, final_value = self.agent(final_state, final_side, self.env)
        else:
            final_value = torch.zeros(self.batch_size, device=self.device)

        # 计算returns（从后向前）
        returns = []
        R = final_value
        for t in reversed(range(T)):
            # R = r_t + gamma * R * (1 - done_t)
            R = rewards_list[t] + self.gamma * R * (1 - dones_list[t])
            returns.insert(0, R)

        # 堆叠所有数据
        log_probs = torch.stack(log_probs_list, dim=0)  # [T, B]
        values = torch.stack(values_list, dim=0)        # [T, B]
        entropies = torch.stack(entropies_list, dim=0)  # [T, B]
        sides = torch.stack(sides_list, dim=0)          # [T, B]
        returns = torch.stack(returns, dim=0)           # [T, B]

        # 计算advantages（TD-error）
        # advantage = return - value（价值网络作为baseline）
        advantages = returns - values.detach()

        # 计算损失
        # 策略损失：-advantage * log_prob
        policy_loss = -(advantages * log_probs).mean()

        # 价值损失：MSE
        value_loss = F.mse_loss(values, returns)

        # 熵奖励（鼓励探索）
        entropy_loss = -entropies.mean()

        # 总损失
        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

        # 反向传播
        self.optim.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

        self.optim.step()

        return (
            float(policy_loss.item()),
            float(value_loss.item()),
            float(entropy_loss.item()),
            float(total_loss.item())
        )

    def _train_episode_original(self, half_self_play: bool):
        """原始训练方法（向后兼容）"""
        logs_off = [[] for _ in range(self.batch_size)]
        logs_def = [[] for _ in range(self.batch_size)]
        ents_off = [[] for _ in range(self.batch_size)]
        ents_def = [[] for _ in range(self.batch_size)]
        steps = 0

        while True:
            state = self.env.state()
            side = self.env.turn().to(torch.int32)
            actions, probs, entropy, value = self.agent(state, side, self.env)
            lp = torch.log(probs + 1e-9)
            for i in range(self.batch_size):
                if not bool(self.env.done[i].item()):
                    if int(side[i].item()) == 1:
                        logs_off[i].append(lp[i])
                        ents_off[i].append(entropy[i])
                    else:
                        logs_def[i].append(lp[i])
                        ents_def[i].append(entropy[i])
            self.env.step(actions)
            steps += 1
            if self.env.all_done() or steps >= self.env.max_steps:
                break

        winners = self.env.winners.to(self.device).float()
        r_off = winners
        r_def = -winners

        if half_self_play:
            half = self.batch_size // 2
            idx_off = torch.arange(0, half, device=self.device)
            mean_off = float(r_off[idx_off].mean().item()) if idx_off.numel() > 0 else 0.0
        else:
            mean_off = float(r_off.mean().item())
        self.baseline_off = (1 - self.baseline_beta) * self.baseline_off + self.baseline_beta * mean_off
        self.baseline_def = -self.baseline_off

        loss = torch.tensor(0.0, device=self.device)
        if half_self_play:
            half = self.batch_size // 2
            for i in range(0, half):
                if logs_off[i]:
                    s_off = torch.stack(logs_off[i]).sum()
                    adv_off = r_off[i] - self.baseline_off
                    loss = loss - adv_off * s_off
                    if self.entropy_coef != 0.0 and ents_off[i]:
                        h_off = torch.stack(ents_off[i]).sum()
                        loss = loss - self.entropy_coef * h_off
            for i in range(half, self.batch_size):
                if logs_def[i]:
                    s_def = torch.stack(logs_def[i]).sum()
                    adv_def = r_def[i] - self.baseline_def
                    loss = loss - adv_def * s_def
                    if self.entropy_coef != 0.0 and ents_def[i]:
                        h_def = torch.stack(ents_def[i]).sum()
                        loss = loss - self.entropy_coef * h_def
        else:
            for i in range(self.batch_size):
                if logs_off[i]:
                    s_off = torch.stack(logs_off[i]).sum()
                    adv_off = r_off[i] - self.baseline_off
                    loss = loss - adv_off * s_off
                    if self.entropy_coef != 0.0 and ents_off[i]:
                        h_off = torch.stack(ents_off[i]).sum()
                        loss = loss - self.entropy_coef * h_off
                if logs_def[i]:
                    s_def = torch.stack(logs_def[i]).sum()
                    adv_def = r_def[i] - self.baseline_def
                    loss = loss - adv_def * s_def
                    if self.entropy_coef != 0.0 and ents_def[i]:
                        h_def = torch.stack(ents_def[i]).sum()
                        loss = loss - self.entropy_coef * h_def

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss, steps

    @torch.no_grad()
    def _evaluate_agents(self, eval_games: int):
        """
        评测新模型(agent)和旧模型(opponent)的对战胜率
        修复：确保统计的游戏数量与实际运行的数量一致
        """
        total = int(eval_games)
        half = max(1, total // 2)  # new先手的局数
        second_half = total - half  # old先手的局数

        wins_new_total = 0
        wins_old_total = 0
        wins_new_first = 0
        wins_new_second = 0
        wins_old_first = 0
        wins_old_second = 0
        games_counted_new_first = 0  # 实际统计的new先手局数
        games_counted_new_second = 0  # 实际统计的new后手局数

        def play_batch(starter: str, n: int):
            nonlocal wins_new_total, wins_old_total, wins_new_first, wins_new_second
            nonlocal wins_old_first, wins_old_second, games_counted_new_first, games_counted_new_second

            remaining = n
            while remaining > 0:
                # 每轮最多运行 batch_size 局，但只统计需要的数量
                games_this_round = min(self.batch_size, remaining)
                self.env.reset()
                steps = 0
                while True:
                    state = self.env.state()
                    side = self.env.turn().to(torch.int32)
                    # 两个模型分别给出动作（只取actions，忽略其他返回值）
                    a_new = self.agent(state, side, self.env)[0]
                    a_old = self.opponent(state, side, self.env)[0]
                    # 按先手分配当前回合动作来源
                    if starter == "new":
                        cur = torch.where(side.view(-1) == 1, a_new, a_old)
                    else:
                        cur = torch.where(side.view(-1) == 1, a_old, a_new)
                    self.env.step(cur)
                    steps += 1
                    if self.env.all_done() or steps >= self.env.max_steps:
                        break

                w = self.env.winners
                # 只统计这轮需要的数量
                w_used = w[:games_this_round]

                if starter == "new":
                    wn_first = int((w_used == 1).sum().item())  # new先手赢
                    wo_second = int((w_used == -1).sum().item())  # old后手赢
                    w_draw = int((w_used == 0).sum().item())  # 平局
                    wins_new_total += wn_first
                    wins_old_total += wo_second
                    wins_new_first += wn_first
                    wins_old_second += wo_second
                    games_counted_new_first += games_this_round
                else:
                    wo_first = int((w_used == 1).sum().item())  # old先手赢
                    wn_second = int((w_used == -1).sum().item())  # new后手赢
                    w_draw = int((w_used == 0).sum().item())  # 平局
                    wins_old_total += wo_first
                    wins_new_total += wn_second
                    wins_old_first += wo_first
                    wins_new_second += wn_second
                    games_counted_new_second += games_this_round

                remaining -= games_this_round

        play_batch("new", half)
        play_batch("old", second_half)

        # 使用实际统计的局数作为分母（更安全）
        metrics = {
            "new_total_win_rate": wins_new_total / float(total) if total > 0 else 0.0,
            "old_total_win_rate": wins_old_total / float(total) if total > 0 else 0.0,
            "new_first_win_rate": wins_new_first / float(games_counted_new_first) if games_counted_new_first > 0 else 0.0,
            "new_second_win_rate": wins_new_second / float(games_counted_new_second) if games_counted_new_second > 0 else 0.0,
            "old_first_win_rate": wins_old_first / float(games_counted_new_second) if games_counted_new_second > 0 else 0.0,
            "old_second_win_rate": wins_old_second / float(games_counted_new_first) if games_counted_new_first > 0 else 0.0,
        }
        return metrics
