# -*- coding: utf-8 -*-
"""
AlphaZero 策略-价值网络的 PyTorch 实现。

输入: 4通道棋盘状态（己方棋子、对方棋子、上一步位置、当前玩家标识）
输出: (策略概率, 局面价值) 用于指导 MCTS 搜索
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def set_learning_rate(optimizer, lr):
    """动态设置学习率。"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """策略-价值网络模块。"""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # 共享卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 策略头卷积层
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # 价值头卷积层
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 共享层
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 策略输出
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 价值输出
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """策略-价值网络封装类。"""

    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2 正则化系数
        # 初始化网络与优化器
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            checkpoint = torch.load(model_file, map_location=self.device)
            if isinstance(checkpoint, dict) and 'net' in checkpoint:
                self.policy_value_net.load_state_dict(checkpoint['net'])
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                # 旧格式：仅包含网络参数
                self.policy_value_net.load_state_dict(checkpoint)

    def policy_value(self, state_batch):
        """
        批量推理。
        输入: 一批状态
        输出: 一批动作概率和局面价值
        """
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        单局面推理。
        输入: 棋盘对象
        输出: (动作, 概率) 元组列表 和 局面评分
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        state_tensor = torch.from_numpy(current_state).float().to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        value = value.cpu().numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """
        执行一步训练。
        损失 = 价值均方误差 + 策略交叉熵 + L2正则(由优化器处理)
        """
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)

        # 清零梯度
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 损失 = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # L2 正则化已通过优化器的 weight_decay 实现
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播与参数更新
        loss.backward()
        self.optimizer.step()
        # 计算策略熵（仅用于监控）
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """保存模型参数与优化器状态。"""
        checkpoint = {
            'net': self.policy_value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, model_file)
