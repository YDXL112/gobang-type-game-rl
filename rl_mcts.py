import os
import csv
import torch
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
        self.baseline_off = 0.0
        self.baseline_def = 0.0
        self.baseline_beta = float(baseline_beta)
        self.eval_interval_episodes = int(eval_interval_episodes)
        self.eval_games = int(eval_games)
        self.replace_threshold = float(replace_threshold)

    def train(self, model_dir="saved_models", results_dir="results", model_name="model.pth", csv_name="results.csv", json_name="episodes.json", half_self_play: bool = False):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, csv_name)
        json_path = os.path.join(results_dir, json_name)
        csv_exists = os.path.exists(csv_path)
        if not csv_exists:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "off_win_rate", "draw_rate", "def_win_rate", "loss"])
        episodes_out = []
        for ep in range(1, self.episodes + 1):
            print(f"episode:{ep}")
            self.env.reset()
            logs_off = [[] for _ in range(self.batch_size)]
            logs_def = [[] for _ in range(self.batch_size)]
            ep_moves = []
            steps = 0
            while True:
                state = self.env.state()
                side = self.env.turn().to(torch.int32)
                actions, probs = self.agent(state, side, self.env)
                lp = torch.log(probs + 1e-9)
                for i in range(self.batch_size):
                    if not bool(self.env.done[i].item()):
                        if int(side[i].item()) == 1:
                            logs_off[i].append(lp[i])
                        else:
                            logs_def[i].append(lp[i])
                if not bool(self.env.done[0].item()):
                    s0 = int(side[0].item())
                    a0 = int(actions[0].item())
                    self.env.step(actions)
                    ep_moves.append({
                        "step": steps,
                        "side": s0,
                        "x": a0 // 8,
                        "y": a0 % 8,
                    })
                else:
                    self.env.step(actions)
                steps += 1
                print(f"step:{steps}")
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
                for i in range(half, self.batch_size):
                    if logs_def[i]:
                        s_def = torch.stack(logs_def[i]).sum()
                        adv_def = r_def[i] - self.baseline_def
                        loss = loss - adv_def * s_def
            else:
                for i in range(self.batch_size):
                    if logs_off[i]:
                        s_off = torch.stack(logs_off[i]).sum()
                        adv_off = r_off[i] - self.baseline_off
                        loss = loss - adv_off * s_off
                    if logs_def[i]:
                        s_def = torch.stack(logs_def[i]).sum()
                        adv_def = r_def[i] - self.baseline_def
                        loss = loss - adv_def * s_def
            self.optim.zero_grad()
            loss.backward()
            total_abs = 0.0
            total_cnt = 0
            max_abs = 0.0
            for p in self.agent.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    total_abs += float(g.abs().sum().item())
                    total_cnt += int(g.numel())
                    max_abs = max(max_abs, float(g.abs().max().item()))
            grad_mean_abs = (total_abs / total_cnt) if total_cnt > 0 else 0.0
            print(f"[episode={ep}] steps={steps} loss={float(loss.item()):.6f} grad_mean_abs={grad_mean_abs:.6f} grad_max_abs={max_abs:.6f}")
            self.optim.step()
            off_rate = float((winners == 1).sum().item()) / float(self.batch_size)
            draw_rate = float((winners == 0).sum().item()) / float(self.batch_size)
            def_rate = float((winners == -1).sum().item()) / float(self.batch_size)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ep, off_rate, draw_rate, def_rate, float(loss.item())])
            episodes_out.append({
                "episode": ep,
                "winner": int(self.env.winners[0].item()),
                "steps": len(ep_moves),
                "moves": ep_moves,
            })
            # 周期评测与替换
            if self.eval_interval_episodes > 0 and ep % self.eval_interval_episodes == 0:
                new_rate, old_rate = self._evaluate_agents(model_dir, results_dir, self.eval_games)
                if new_rate > old_rate and new_rate >= self.replace_threshold:
                    self.opponent.load_state_dict(self.agent.state_dict())
                    torch.save(self.agent.state_dict(), os.path.join(model_dir, model_name))
        torch.save(self.agent.state_dict(), os.path.join(model_dir, model_name))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": episodes_out}, f, ensure_ascii=False)
        return True

    @torch.no_grad()
    def _evaluate_agents(self, model_dir, results_dir, eval_games: int):
        total = int(eval_games)
        half = max(1, total // 2)
        wins_new = 0
        wins_old = 0
        def play_batch(starter: str, n: int):
            nonlocal wins_new, wins_old
            rounds = (n + self.batch_size - 1) // self.batch_size
            for r in range(rounds):
                self.env.reset()
                games_in_batch = min(self.batch_size, n - r * self.batch_size)
                steps = 0
                while True:
                    state = self.env.state()
                    side = self.env.turn().to(torch.int32)
                    # 两个模型分别给出动作
                    a_new, _ = self.agent(state, side, self.env)
                    a_old, _ = self.opponent(state, side, self.env)
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
                wins_new += int((w == 1).sum().item()) if starter == "new" else int((w == -1).sum().item())
                wins_old += int((w == 1).sum().item()) if starter == "old" else int((w == -1).sum().item())
        play_batch("new", half)
        play_batch("old", total - half)
        new_rate = wins_new / float(total)
        old_rate = wins_old / float(total)
        return new_rate, old_rate
