#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/8/03 19:00
# @Author: zhl

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from chess_board import ChessBoard

# -----------------------------------------------------------------------------
# 超参数
# -----------------------------------------------------------------------------
EPISODES            = 2000    # 总训练局数
BATCH_SIZE          = 4       # 每批自对弈局数
VISUALIZE_EVERY     = 100     # 可视化间隔
MAX_STEPS           = 40      # 单局最大步数
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOVE_ORDER_K        = 8       # 每节点仅搜索 policy top-K
DIRICHLET_ALPHA     = 0.3     # 根节点 Dirichlet 噪声参数
DIRICHLET_EPS       = 0.25    # 噪声混合系数
POOL_SNAPSHOT_FREQ  = 50      # 每多少集给对手池加入一次新快照

# -----------------------------------------------------------------------------
# 全局置换表
# -----------------------------------------------------------------------------
TT = {}

# -----------------------------------------------------------------------------
# 单步必胜/必输检测
# -----------------------------------------------------------------------------
def find_forced(env, player):
    mask = env.get_capable_move().flatten()
    losing = None
    for idx in np.where(mask)[0]:
        x, y = divmod(idx, 8)
        copy_env = ChessBoard()
        copy_env.board = env.board.copy()
        copy_env.player_to_go = env.player_to_go
        copy_env.move(player, x, y)
        res = copy_env.judge()
        if res == player:
            return True, idx
        if res == -player and losing is None:
            losing = idx
    return False, losing

# -----------------------------------------------------------------------------
# 网络定义：残差 + 先验热图 + Policy/Value
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.gelu  = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
    def forward(self, x):
        idt = x
        out = self.conv1(x); out = self.bn1(out); out = self.gelu(out)
        out = self.conv2(out); out = self.bn2(out)
        return self.gelu(out + idt)

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # stem + residual
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.res_tower = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32)
        )
        # policy head (带先验热图 bias)
        self.policy_conv = nn.Conv2d(32, 2, 1)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_gelu = nn.GELU()
        self.policy_flat = nn.Flatten()
        self.policy_fc   = nn.Linear(2*8*8, 64, bias=False)
        # trainable prior, 初值中心高、边角低
        prior_init = np.array([
            [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
            [0.05,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
            [0.05,0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.05],
            [0.05,0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.05],
            [0.05,0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.05],
            [0.05,0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.05],
            [0.05,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
            [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
        ], dtype=np.float32).flatten()
        self.prior = nn.Parameter(torch.from_numpy(prior_init), requires_grad=True)

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(1*8*8, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [B,1,8,8]
        x = self.stem(x)
        x = self.res_tower(x)
        # policy
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.policy_gelu(p)
        p = self.policy_flat(p)           # [B,64]
        p = self.policy_fc(p)             # [B,64]
        logits = p + self.prior.unsqueeze(0)
        pi = torch.softmax(logits, dim=-1)  # [B,64]
        # value
        v = self.value_head(x)            # [B,1]
        return pi, v

# -----------------------------------------------------------------------------
# 搜索：Negamax + α–β + TT + single-step + move ordering + root noise
# -----------------------------------------------------------------------------
def search_move(env, net, player, depth, α=-1e9, β=1e9, is_root=False):
    # 1) single-step forced
    forced_win, forced_idx = find_forced(env, player)
    if forced_win:
        return 1e6, forced_idx

    key = (player, env.board.tobytes(), depth)
    if key in TT:
        return TT[key]

    winner = env.judge()
    if winner != 0 or depth == 0:
        b = torch.tensor(env.get_state(), dtype=torch.float32, device=DEVICE)
        b = b.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, v = net(b)
        score = v.item() * player
        TT[key] = (score, None)
        return score, None

    # 2) policy 前向（一次）
    b = torch.tensor(env.get_state(), dtype=torch.float32, device=DEVICE)
    b = b.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pi, _ = net(b)   # [1,64]
    pi = pi.squeeze(0)

    # 3) root only: 加入 Dirichlet 噪声
    if is_root:
        noise = np.random.dirichlet([DIRICHLET_ALPHA]*64)
        noise_t = torch.from_numpy(noise).to(DEVICE)
        pi = (1 - DIRICHLET_EPS)*pi + DIRICHLET_EPS*noise_t

    # 4) mask & top-K
    mask = torch.from_numpy(env.get_capable_move().flatten()).bool().to(DEVICE)
    pi = pi * mask.float()
    if mask.sum() > MOVE_ORDER_K:
        topk = torch.topk(pi, k=MOVE_ORDER_K)
        cand = topk.indices.cpu().numpy().tolist()
    else:
        cand = np.where(mask.cpu().numpy())[0].tolist()

    best_score, best_move = -np.inf, None
    for idx in cand:
        x, y = divmod(idx, 8)
        copy_env = ChessBoard()
        copy_env.board        = env.board.copy()
        copy_env.player_to_go = env.player_to_go
        copy_env.move(player, x, y)

        sc, _ = search_move(copy_env, net, -player, depth-1, -β, -α, is_root=False)
        sc = -sc
        if sc > best_score:
            best_score, best_move = sc, idx
        α = max(α, sc)
        if α >= β:
            break

    TT[key] = (best_score, best_move)
    return best_score, best_move

# -----------------------------------------------------------------------------
# RL 智能体：带历史对手池
# -----------------------------------------------------------------------------
class RLAgent:
    def __init__(self):
        self.net = PolicyValueNet().to(DEVICE)
        self.optim = optim.Adam(self.net.parameters(), lr=1e-3)
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # 对手池：保存 net snapshots（PolicyValueNet 实例）
        self.opponent_pool = [copy.deepcopy(self.net)]

    def snapshot(self):
        # 加入快照
        self.opponent_pool.append(copy.deepcopy(self.net))

    def select_action(self, env, depth):
        player = env.get_turn()
        # single-step
        forced_win, forced_idx = find_forced(env, player)
        if forced_win:
            x, y = divmod(forced_idx, 8)
            return 1.0, x, y

        # ε-greedy
        if np.random.rand() < self.epsilon:
            moves = np.argwhere(env.get_capable_move())
            i = np.random.choice(len(moves))
            x, y = moves[i]
            return 1/len(moves), x, y

        # 从对手池随机选一个 net 作为对手网络
        opp_net = np.random.choice(self.opponent_pool)

        # 搜索（root noise=True）
        _, idx = search_move(env, opp_net, player, depth, is_root=True)
        if idx is None:
            return 1/64, 0, 0

        # 衰减 ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        x, y = divmod(idx, 8)
        return None, x, y

    def update(self, trajs):
        if not trajs:
            return 0.0
        states = np.stack([t[0] for t in trajs])
        states = torch.tensor(states, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        pis, _ = self.net(states)
        acts = torch.tensor([t[1] for t in trajs], device=DEVICE)
        rews = torch.tensor([t[2] for t in trajs], dtype=torch.float32, device=DEVICE)
        returns = rews.flip(0).cumsum(0).flip(0)
        returns = returns - returns.mean()
        logp = torch.log(pis.gather(1, acts.unsqueeze(1)).squeeze(1) + 1e-9)
        loss = -(logp * returns).sum()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

# -----------------------------------------------------------------------------
# 训练主循环
# -----------------------------------------------------------------------------
def train():
    off_agent = RLAgent()
    def_agent = RLAgent()
    losses_off, losses_def = [], []

    for ep in range(1, EPISODES+1):
        # 动态深度
        depth = 1 if ep<=100 else (2 if ep<=300 else 3)
        print(f"=== Episode {ep}/{EPISODES} | depth={depth} ===")

        # 每 POOL_SNAPSHOT_FREQ 集给对手池加入一次快照
        if ep % POOL_SNAPSHOT_FREQ == 0:
            def_agent.snapshot()
            off_agent.snapshot()

        batch_off, batch_def = [], []
        for _ in range(BATCH_SIZE):
            env = ChessBoard()
            player = 1
            traj_off, traj_def = [], []
            for t in range(MAX_STEPS):
                prob, x, y = (off_agent if player==1 else def_agent).select_action(env, depth)
                idx = x*8 + y
                if player==1:
                    traj_off.append((env.get_state().copy(), idx, None))
                else:
                    traj_def.append((env.get_state().copy(), idx, None))
                env.move(player, x, y)
                res = env.judge()
                if res != 0:
                    break
                player *= -1

            # 结算奖励
            if res==1:    rw={1:1,   -1:-1}
            elif res==-1: rw={1:-1,  -1:1}
            else:         rw={1:0,   -1:0}
            batch_off += [(s,a,rw[1]) for s,a,_ in traj_off]
            batch_def += [(s,a,rw[-1]) for s,a,_ in traj_def]

        loss1 = off_agent.update(batch_off)
        loss2 = def_agent.update(batch_def)
        losses_off.append(loss1); losses_def.append(loss2)
        print(f"Off_Loss={loss1:.3f}, Def_Loss={loss2:.3f}")

        if ep % VISUALIZE_EVERY == 0:
            print(f"--- Visualize Episode {ep} ---")
            visualize_game(off_agent, def_agent, depth)

    # 保存 & 绘图
    plt.figure(); plt.plot(losses_off,label='Off'); plt.plot(losses_def,label='Def')
    plt.legend(); plt.title("Training Loss"); plt.savefig("loss.png"); plt.show()
    torch.save(off_agent.net.state_dict(),  "off_net.pth")
    torch.save(def_agent.net.state_dict(), "def_net.pth")
    print("训练完成，模型已保存。")

# -----------------------------------------------------------------------------
# 可视化对弈
# -----------------------------------------------------------------------------
def visualize_game(off_agent, def_agent, depth):
    env = ChessBoard(); player, step = 1, 0
    plt.figure(figsize=(6,6))
    while True:
        bd = env.get_state()
        disp = np.where(bd==0.1, .5, bd)
        plt.imshow(disp, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"Step {step} ({'Off' if player==1 else 'Def'})")
        plt.pause(0.3)
        _, x, y = (off_agent if player==1 else def_agent).select_action(env, depth)
        env.move(player, x, y)
        step += 1
        if env.judge()!=0 or step>=MAX_STEPS:
            break
        player *= -1
    plt.show()

# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    start = time.time()
    train()
    print(f"Total time: {(time.time()-start)/60:.2f} 分钟")
