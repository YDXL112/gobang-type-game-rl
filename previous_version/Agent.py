#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/6/18 17:54
# @Author: zhl
from chess_board import*
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2),
            nn.Conv2d(16,1,5,padding=2),
        )
        self.mlp_net = nn.Sequential(
            nn.Linear(64,512),
            nn.GELU(),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256,40),
            nn.LayerNorm(40),
            nn.Softmax()
        )

    def forward(self,state):
        state = state.unsqueeze(0).unsqueeze(0)
        state = self.cnn_net(state)
        state = torch.flatten(state)
        return self.mlp_net(state)


def compute_layer(num):
    if num<20:
        if num<2:
            return 0,3+num
        elif num<6:
            return 1,2+num-2
        elif num<12:
            return 2,1+num-6
        else:
            return 3,num-12
    else:
        if num>37:
            return 7,3+num-38
        elif num>33:
            return 6,2+num-34
        elif num>27:
            return 5,1+num-28
        else:
            return 4,num-20



class Off_Agent():
    def __init__(self):
        self.brain = CNN()
        self.player=1
        self.optim = torch.optim.Adam(self.brain.parameters(), lr=0.001)
        # Epsilon-greedy探索参数
        self.epsilon = 0.5  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.99  # 探索率衰减系数

    def action(self, env):
        # 以epsilon概率进行随机探索
        if np.random.uniform(0, 1) < self.epsilon:
            # 随机选择一个合法动作
            capable_moves = np.argwhere(env.get_capable_move())
            if len(capable_moves) > 0:
                idx = np.random.choice(len(capable_moves))
                x, y = capable_moves[idx]
                # 计算这个随机动作的概率 (均匀分布)
                action_prob = 1.0 / len(capable_moves)
                # 创建一个全零的概率分布，只有选中的动作概率为1
                norm_prob = torch.zeros(64)
                norm_prob[x * 8 + y] = 1.0
                return action_prob, x, y, norm_prob
            else:
                # 如果没有合法动作，返回第一个位置（理论上不会发生）
                return 1.0, 0, 0, torch.ones(64) / 64

        # 以1-epsilon概率使用模型选择动作
        # 设置模型为评估模式
        self.brain.eval()

        state = env.get_state()
        state_tensors = [torch.from_numpy(arr).float() for arr in state]
        state = torch.stack(state_tensors)

        # 使用with torch.no_grad()上下文管理器减少内存消耗
        with torch.no_grad():
            prob = self.brain(state)

        capable_prob = prob.clone()
        prob_matrix = torch.zeros(size=(8, 8))

        # 填充概率矩阵
        for i in range(len(capable_prob)):
            prob_matrix[compute_layer(i)[0], compute_layer(i)[1]] = capable_prob[i]

        # 获取可用行动掩码
        np_mask = env.get_capable_move()
        torch_mask = torch.from_numpy(np_mask).bool()

        # 屏蔽不可用行动
        prob_matrix[~torch_mask] = 0  # 注意这里应该屏蔽不可用的位置，使用~torch_mask

        # 展平并处理概率
        flat = prob_matrix.flatten()

        # 数值稳定性处理
        min_prob = 1e-8
        flat = torch.clamp(flat, min_prob)  # 确保所有概率都大于最小值

        # 归一化处理，增加对零和的保护
        total = flat.sum()
        if total <= min_prob:  # 如果总和几乎为零，使用均匀分布
            norm_prob = torch.ones_like(flat) / len(flat)
        else:
            norm_prob = flat / total

        # 采样动作
        action_idx = torch.multinomial(norm_prob, 1).item()
        action_prob = norm_prob[action_idx]

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action_prob, action_idx // 8, action_idx % 8, norm_prob

    def upgrade(self,loss):
        loss.backward()
        self.optim.step()

    def reset_grad(self):
        self.optim.zero_grad()


class Def_Agent():
    def __init__(self):
        self.brain = CNN()
        self.player=-1
        self.optim = torch.optim.Adam(self.brain.parameters(), lr=0.001)
        # Epsilon-greedy探索参数
        self.epsilon = 0.5  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减系数

    def action(self, env):
        # 以epsilon概率进行随机探索
        if np.random.uniform(0, 1) < self.epsilon:
            # 随机选择一个合法动作
            capable_moves = np.argwhere(env.get_capable_move())
            if len(capable_moves) > 0:
                idx = np.random.choice(len(capable_moves))
                x, y = capable_moves[idx]
                # 计算这个随机动作的概率 (均匀分布)
                action_prob = 1.0 / len(capable_moves)
                # 创建一个全零的概率分布，只有选中的动作概率为1
                norm_prob = torch.zeros(64)
                norm_prob[x * 8 + y] = 1.0
                return action_prob, x, y, norm_prob
            else:
                # 如果没有合法动作，返回第一个位置（理论上不会发生）
                return 1.0, 0, 0, torch.ones(64) / 64

        # 以1-epsilon概率使用模型选择动作
        # 设置模型为评估模式
        self.brain.eval()

        state = env.get_state()
        state_tensors = [torch.from_numpy(arr).float() for arr in state]
        state = torch.stack(state_tensors)

        # 使用with torch.no_grad()上下文管理器减少内存消耗
        with torch.no_grad():
            prob = self.brain(state)

        capable_prob = prob.clone()
        prob_matrix = torch.zeros(size=(8, 8))

        # 填充概率矩阵
        for i in range(len(capable_prob)):
            prob_matrix[compute_layer(i)[0], compute_layer(i)[1]] = capable_prob[i]

        # 获取可用行动掩码
        np_mask = env.get_capable_move()
        torch_mask = torch.from_numpy(np_mask).bool()

        # 屏蔽不可用行动
        prob_matrix[~torch_mask] = 0  # 注意这里应该屏蔽不可用的位置，使用~torch_mask

        # 展平并处理概率
        flat = prob_matrix.flatten()

        # 数值稳定性处理
        min_prob = 1e-8
        flat = torch.clamp(flat, min_prob)  # 确保所有概率都大于最小值

        # 归一化处理，增加对零和的保护
        total = flat.sum()
        if total <= min_prob:  # 如果总和几乎为零，使用均匀分布
            norm_prob = torch.ones_like(flat) / len(flat)
        else:
            norm_prob = flat / total

        # 采样动作
        action_idx = torch.multinomial(norm_prob, 1).item()
        action_prob = norm_prob[action_idx]

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action_prob, action_idx // 8, action_idx % 8, norm_prob

    def upgrade(self,loss):
        loss.backward()
        self.optim.step()

    def reset_grad(self):
        self.optim.zero_grad()
