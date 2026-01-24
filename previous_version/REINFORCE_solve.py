#!/usr/bin/env python3
# -*- coding: utf-8
# @Time: 2025/6/18 18:55
# @Author: zhl

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from Agent import*
from chess_board import ChessBoard

# 超参
EPISODES = 2000
BATCH_SIZE = 32  # 批量评估大小
VISUALIZE_EVERY = 100  # 每10批次可视化一盘
MAX_STEPS = 40  # 最大步数限制
BOARD_SIZE = 8  # 棋盘大小
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)
loss1_to_draw = []
loss2_to_draw = []

class ParallelREINFORCE:
    def __init__(self, env_class):
        self.env_class = env_class
        self.Off = Off_Agent()
        self.Def = Def_Agent()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.clear_buffers()

    def clear_buffers(self):
        self.envs = [self.env_class() for _ in range(BATCH_SIZE)]
        self.states1 = [[] for _ in range(BATCH_SIZE)]
        self.actions1 = [[] for _ in range(BATCH_SIZE)]
        self.actionprobs1 = [[] for _ in range(BATCH_SIZE)]
        self.rewards1 = [[] for _ in range(BATCH_SIZE)]
        self.states2 = [[] for _ in range(BATCH_SIZE)]
        self.actions2 = [[] for _ in range(BATCH_SIZE)]
        self.actionprobs2 = [[] for _ in range(BATCH_SIZE)]
        self.rewards2 = [[] for _ in range(BATCH_SIZE)]

    def _process_env(self, idx, env):
        states1, actions1, actionprobs1, rewards1 = [], [], [], []
        states2, actions2, actionprobs2, rewards2 = [], [], [], []
        done = False
        player = 1
        step_count = 0

        while not done and step_count < MAX_STEPS:
            state = env.get_state()

            if player == 1:
                # 修改：获取动作概率张量
                prob, x, y, norm_prob = self.Off.action(env)
                action_idx = x + y * 8  # 将坐标转换为索引
                action_prob = norm_prob[action_idx].unsqueeze(0)  # 获取动作概率并保持张量维度
                states1.append(state)
                actions1.append((x, y))
                actionprobs1.append(action_prob)  # 存储为张量
                rewards1.append(-1)  # 默认惩罚浪费步数
            else:
                # 修改：获取动作概率张量
                prob, x, y, norm_prob = self.Def.action(env)
                action_idx = x + y * 8  # 将坐标转换为索引
                action_prob = norm_prob[action_idx].unsqueeze(0)  # 获取动作概率并保持张量维度
                states2.append(state)
                actions2.append((x, y))
                actionprobs2.append(action_prob)  # 存储为张量
                rewards2.append(step_count/5)  # 默认奖励拖延成功

            env.move(player, x, y)
            step_count += 1
            result = env.judge()

            if result != 0:  # 有玩家获胜
                if result == 1:  # 进攻方获胜
                    if player == 1:  # 进攻方走的最后一步
                        rewards1[-1] = 10
                    else:  # 防守方走的最后一步，但进攻方获胜
                        rewards1[-1] = 10
                    # 防守方惩罚
                    if rewards2:
                        rewards2[-1] = -10
                else:  # 防守方获胜
                    if player == -1:  # 防守方走的最后一步
                        rewards2[-1] = 10
                    else:  # 进攻方走的最后一步，但防守方获胜
                        rewards2[-1] = 10
                    # 进攻方惩罚
                    if rewards1:
                        rewards1[-1] = -10
                done = True
            elif step_count >= MAX_STEPS or (env.board != 0).all():  # 平局
                if rewards1:
                    rewards1[-1] = -5
                if rewards2:
                    rewards2[-1] = -5
                done = True
            else:  # 游戏继续
                player *= -1  # 换手

        return (states1, actions1, actionprobs1, rewards1,
                states2, actions2, actionprobs2, rewards2)

    def play_batch(self):
        futures = []
        for idx, env in enumerate(self.envs):
            # 深拷贝环境以避免状态共享
            env_copy = self.env_class()
            env_copy.board = np.copy(env.board)
            env_copy.player_to_go = env.player_to_go
            futures.append(self.executor.submit(self._process_env, idx, env_copy))

        for idx, future in enumerate(futures):
            (states1, actions1, actionprobs1, rewards1,
             states2, actions2, actionprobs2, rewards2) = future.result()

            self.states1[idx] = states1
            self.actions1[idx] = actions1
            self.actionprobs1[idx] = actionprobs1
            self.rewards1[idx] = rewards1

            self.states2[idx] = states2
            self.actions2[idx] = actions2
            self.actionprobs2[idx] = actionprobs2
            self.rewards2[idx] = rewards2

    def learn(self):
        gamma = 1.0
        total_loss1 = torch.tensor([0.0], device=DEVICE, requires_grad=True)
        total_loss2 = torch.tensor([0.0], device=DEVICE, requires_grad=True)

        # 处理进攻方数据
        for idx in range(BATCH_SIZE):
            if self.actionprobs1[idx] and self.rewards1[idx]:
                # 修改：确保actionprobs1是张量列表
                actionprobs1 = torch.cat(self.actionprobs1[idx]).to(DEVICE)  # 使用cat而非stack
                rewards1 = torch.tensor(self.rewards1[idx], dtype=torch.float32).to(DEVICE)

                # 计算折扣回报
                returns1 = torch.zeros_like(rewards1)
                R = torch.tensor(0.0).to(DEVICE)
                for t in reversed(range(len(rewards1))):
                    R = rewards1[t] + gamma * R
                    returns1[t] = R

                # 减去基线减少方差
                returns1 = returns1 - returns1.mean()

                # 计算损失
                loss1 = - (returns1 * actionprobs1).sum()
                total_loss1 = total_loss1 + loss1

        # 处理防守方数据
        for idx in range(BATCH_SIZE):
            if self.actionprobs2[idx] and self.rewards2[idx]:
                # 修改：确保actionprobs2是张量列表
                actionprobs2 = torch.cat(self.actionprobs2[idx]).to(DEVICE)  # 使用cat而非stack
                rewards2 = torch.tensor(self.rewards2[idx], dtype=torch.float32).to(DEVICE)

                # 计算折扣回报
                returns2 = torch.zeros_like(rewards2)
                R = torch.tensor(0.0).to(DEVICE)
                for t in reversed(range(len(rewards2))):
                    R = rewards2[t] + gamma * R
                    returns2[t] = R

                # 减去基线减少方差
                returns2 = returns2 - returns2.mean()

                # 计算损失
                loss2 = - (returns2 * actionprobs2).sum()
                total_loss2 = total_loss2 + loss2

        # 优化
        self.Off.reset_grad()
        if total_loss1.requires_grad:
            self.Off.upgrade(total_loss1)
            loss1_val = total_loss1.item()
        else:
            loss1_val = 0.0

        self.Def.reset_grad()
        if total_loss2.requires_grad:
            self.Def.upgrade(total_loss2)
            loss2_val = total_loss2.item()
        else:
            loss2_val = 0.0

        # 记录 loss
        loss1_to_draw.append(loss1_val)
        loss2_to_draw.append(loss2_val)

        return loss1_val, loss2_val

    def visualize(self):
        env = ChessBoard()
        env.reset()
        done = False
        player = 1
        step_count = 0

        plt.figure(figsize=(8, 8))

        while not done and step_count < MAX_STEPS:
            plt.clf()
            board_display = np.zeros((env.size, env.size))
            for i in range(env.size):
                for j in range(env.size):
                    if env.board[i, j] == 1:
                        board_display[i, j] = 1
                    elif env.board[i, j] == -1:
                        board_display[i, j] = -1
                    elif env.board[i, j] == 0.1:
                        board_display[i, j] = 0.5  # 不可落子区域

            plt.imshow(board_display, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Step {step_count}: {"Offensive" if player == 1 else "Defensive"}')
            plt.colorbar(ticks=[-1, 0, 1])
            plt.draw()
            plt.pause(0.1)

            if player == 1:
                prob, x, y, _ = self.Off.action(env)
                print(f"Offensive moves to ({x}, {y})")
            else:
                prob, x, y, _ = self.Def.action(env)
                print(f"Defensive moves to ({x}, {y})")

            env.move(player, x, y)
            step_count += 1

            result = env.judge()

            if result == 1:
                plt.clf()
                plt.imshow(board_display, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Offensive wins at step {step_count}!')
                plt.colorbar(ticks=[-1, 0, 1])
                plt.show()
                print("Offensive wins!")
                done = True
            elif result == -1:
                plt.clf()
                plt.imshow(board_display, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Defensive wins at step {step_count}!')
                plt.colorbar(ticks=[-1, 0, 1])
                plt.show()
                print("Defensive wins!")
                done = True
            elif step_count >= MAX_STEPS or (env.board != 0).all():
                plt.clf()
                plt.imshow(board_display, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Draw after {step_count} steps!')
                plt.colorbar(ticks=[-1, 0, 1])
                plt.show()
                print("Draw!")
                done = True
            else:
                player *= -1  # 换手


def main():
    start_time = time.time()
    reinforce = ParallelREINFORCE(ChessBoard)

    print(f"Using device: {DEVICE}")
    print(f"Training {BATCH_SIZE} games per batch...")

    for ep in range(1, EPISODES + 1):
        ep_start = time.time()

        # 并行游戏模拟
        reinforce.play_batch()

        # 学习
        loss1, loss2 = reinforce.learn()

        # 重置缓冲区
        reinforce.clear_buffers()

        ep_time = time.time() - ep_start
        print(f"Episode {ep}/{EPISODES} - Loss: Off={loss1:.2f}, Def={loss2:.2f} - Time: {ep_time:.2f}s")

        if ep % VISUALIZE_EVERY == 0:
            print(f"\n=== Visualizing Episode {ep} ===")
            reinforce.visualize()
            print("=" * 50 + "\n")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(loss1_to_draw, 'b-', label='Offensive Loss')
    plt.plot(loss2_to_draw, 'r-', label='Defensive Loss')
    plt.title('Training Loss over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    # 保存模型
    torch.save(reinforce.Off.brain.state_dict(), 'offensive_agent.pth')
    torch.save(reinforce.Def.brain.state_dict(), 'defensive_agent.pth')
    print("Models saved to offensive_agent.pth and defensive_agent.pth")


if __name__ == "__main__":
    main()