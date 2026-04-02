# -*- coding: utf-8 -*-
"""
纯蒙特卡洛树搜索（不使用神经网络）。
通过随机模拟评估叶节点，用于训练时的基准对手。
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """随机策略，用于模拟阶段的快速落子。"""
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """均匀概率策略，返回等概率和零评分（纯 MCTS 不使用网络）。"""
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """
    MCTS 树节点。
    每个节点维护自身的价值 Q、先验概率 P 和基于访问次数调整的先验得分 u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 动作 -> 子节点 的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        根据策略函数输出的动作先验概率扩展子节点。
        action_priors: (动作, 先验概率) 元组列表
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        选择使 Q + u(P) 最大的子节点。
        返回: (动作, 子节点)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        根据叶节点评估值更新当前节点。
        leaf_value: 从当前玩家视角的子树评估值
        """
        self._n_visits += 1
        # Q 为所有访问值的滑动平均
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归地更新当前节点及其所有祖先。"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算并返回当前节点的选择得分。
        得分 = 叶节点评估 Q + 基于访问次数调整的先验 u
        c_puct: 控制 Q 值与先验概率 P 相对影响的参数
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """判断是否为叶节点（未扩展子节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """纯蒙特卡洛树搜索实现（不使用神经网络）。"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 接收棋盘状态，返回 (动作, 概率) 元组列表和
            当前局面评分的函数
        c_puct: 控制探索与利用权衡的参数
        n_playout: 每次决策的模拟次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        从根节点到叶节点执行一次模拟，获取叶节点评估值并回传更新。
        注意: state 会被原地修改，调用前需传入副本。
        """
        node = self._root
        while(1):
            if node.is_leaf():

                break
            # 贪心地选择下一步
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # 检查游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 通过随机模拟评估叶节点
        leaf_value = self._evaluate_rollout(state)
        # 回传更新路径上所有节点的访问次数和价值
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """
        使用随机策略模拟至游戏结束。
        返回: +1 当前玩家胜, -1 对手胜, 0 平局
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: 模拟达到步数上限")
        if winner == -1:  # 平局
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """
        执行所有模拟并返回访问次数最多的动作。
        state: 当前游戏状态
        返回: 选中的动作
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """
        在树中前进到 last_move 对应的子节点，保留已知子树信息。
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于纯 MCTS 的 AI 玩家（不使用神经网络）。"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: 棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)
