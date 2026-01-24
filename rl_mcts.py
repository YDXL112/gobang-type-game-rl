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
            device=self.device,
        )
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=self.lr)
        self.baseline_off = 0.0
        self.baseline_def = 0.0
        self.baseline_beta = float(baseline_beta)

    def train(self, model_dir="saved_models", results_dir="results", model_name="model.pth", csv_name="results.csv", json_name="episodes.json"):
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
                    p0 = float(probs[0].item())
                    b0_before = self.env.state()[0].detach().cpu().numpy().tolist()
                    self.env.step(actions)
                    b0_after = self.env.state()[0].detach().cpu().numpy().tolist()
                    ep_moves.append({
                        "step": steps,
                        "side": s0,
                        "action_idx": a0,
                        "prob": p0,
                        "x": a0 // 8,
                        "y": a0 % 8,
                        "board_before": b0_before,
                        "board_after": b0_after,
                    })
                else:
                    self.env.step(actions)
                steps += 1
                if self.env.all_done() or steps >= self.env.max_steps:
                    break
            winners = self.env.winners.to(self.device).float()
            r_off = winners
            r_def = -winners
            mean_off = float(r_off.mean().item())
            mean_def = float(r_def.mean().item())
            self.baseline_off = (1 - self.baseline_beta) * self.baseline_off + self.baseline_beta * mean_off
            self.baseline_def = (1 - self.baseline_beta) * self.baseline_def + self.baseline_beta * mean_def
            loss = torch.tensor(0.0, device=self.device)
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
        torch.save(self.agent.state_dict(), os.path.join(model_dir, model_name))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": episodes_out}, f, ensure_ascii=False)
        return True
