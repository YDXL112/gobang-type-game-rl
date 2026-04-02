# -*- coding: utf-8 -*-
"""
四子棋棋盘与游戏逻辑（自由放置变体）。

规则：
- 自由放置：棋子可以放在任意空位上（无重力下落）
- 禁区：棋盘四角的八角形区域不可落子
- 胜负：先在横/竖/斜方向连成 n_in_row 子的一方获胜
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """四子棋棋盘。"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.n_in_row = int(kwargs.get('n_in_row', 4))
        # 禁区位置集合：元组 (row, col)
        self.restricted_positions = set()
        restricted = kwargs.get('restricted_positions', [])
        for pos in restricted:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                self.restricted_positions.add(tuple(pos))
        self.players = [1, 2]
        self.states = {}

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('棋盘宽高不能小于 {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        # 所有合法位置（排除禁区）
        self.availables = []
        for i in range(self.width * self.height):
            h, w = self.move_to_location(i)
            if (h, w) not in self.restricted_positions:
                self.availables.append(i)
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        一维索引 -> 二维坐标 (row, col)。
        棋盘布局示例 (width=3, height=3):
        6 7 8
        3 4 5
        0 1 2
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        返回当前玩家视角下的棋盘状态。
        形状: 4 x width x height
        通道 0: 当前玩家的棋子
        通道 1: 对手的棋子
        通道 2: 上一步落子位置
        通道 3: 当前玩家标识（先手为全1）
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 标记上一步落子位置
            if self.last_move != -1:
                square_state[2][self.last_move // self.width,
                                self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        """检查是否有玩家连成 n_in_row 子。"""
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(self.states.keys())
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # 水平方向
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # 垂直方向
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # 主对角线（左上到右下）
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # 副对角线（右上到左下）
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """检查游戏是否结束。"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """游戏服务器，管理对局流程。"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """以文本方式绘制棋盘。"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if (i, j) in board.restricted_positions:
                    print('#'.center(8), end='')
                elif p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """开始一局双人对战。"""
        if start_player not in (0, 1):
            raise Exception('start_player 应为 0 (player1先手) 或 1 (player2先手)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        使用 MCTS 玩家进行自我对弈，复用搜索树，
        并保存自博弈数据 (state, mcts_probs, z) 用于训练。
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # 保存数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行落子
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # 从每个状态当前玩家的视角计算胜负标签
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置 MCTS 根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
