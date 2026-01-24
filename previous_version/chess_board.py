#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/6/18 17:08
# @Author: zhl
import numpy as np


class ChessBoard:
    def __init__(self):
        self.size = 8  # 添加棋盘大小属性
        self.board = np.array([
            [0.1, 0.1, 0.1, 0, 0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0, 0, 0, 0, 0.1, 0.1],
            [0.1, 0, 0, 0, 0, 0, 0, 0.1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0, 0, 0, 0, 0, 0, 0.1],
            [0.1, 0.1, 0, 0, 0, 0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0, 0, 0.1, 0.1, 0.1]
        ])
        self.offensive = 1
        self.defensive = -1
        self.player_to_go = 1
        self.off_win_detect = np.array([1, 1, 1, 1])
        self.def_win_detect = np.array([-1, -1, -1, -1])

    def move(self, player, x, y):
        x = int(x)
        y = int(y)
        if player != self.player_to_go:
            raise ValueError("错误的玩家ID")
        elif self.board[x, y] != 0:
            raise ValueError("错误的落子位置")
        else:
            self.board[x, y] = player
            self.player_to_go = -self.player_to_go

    def reset(self):
        self.board = np.array([
            [0.1, 0.1, 0.1, 0, 0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0, 0, 0, 0, 0.1, 0.1],
            [0.1, 0, 0, 0, 0, 0, 0, 0.1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0, 0, 0, 0, 0, 0, 0.1],
            [0.1, 0.1, 0, 0, 0, 0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0, 0, 0.1, 0.1, 0.1]
        ])
        self.player_to_go = 1

    def judge(self):
        # 水平方向检查
        for i in range(self.size):
            for j in range(self.size - 3):
                segment = self.board[i, j:j + 4]
                if np.array_equal(segment, self.off_win_detect):
                    return 1
                elif np.array_equal(segment, self.def_win_detect):
                    return -1

        # 垂直方向检查
        for i in range(self.size - 3):
            for j in range(self.size):
                segment = self.board[i:i + 4, j]
                if np.array_equal(segment, self.off_win_detect):
                    return 1
                elif np.array_equal(segment, self.def_win_detect):
                    return -1

        # 主对角线方向检查（左上到右下）
        for i in range(self.size - 3):
            for j in range(self.size - 3):
                segment = np.array([
                    self.board[i, j],
                    self.board[i + 1, j + 1],
                    self.board[i + 2, j + 2],
                    self.board[i + 3, j + 3]
                ])
                if np.array_equal(segment, self.off_win_detect):
                    return 1
                elif np.array_equal(segment, self.def_win_detect):
                    return -1

        # 反对角线方向检查（右上到左下）
        for i in range(self.size - 3):
            for j in range(3, self.size):
                segment = np.array([
                    self.board[i, j],
                    self.board[i + 1, j - 1],
                    self.board[i + 2, j - 2],
                    self.board[i + 3, j - 3]
                ])
                if np.array_equal(segment, self.off_win_detect):
                    return 1
                elif np.array_equal(segment, self.def_win_detect):
                    return -1

        # 检查平局（棋盘填满）
        if (self.board != 0).all():
            return 0

        return 0  # 游戏继续

    def get_state(self):
        return self.board

    def get_turn(self):
        return self.player_to_go

    def get_capable_move(self):
        # 返回可落子位置的布尔掩码
        return (self.board == 0)

