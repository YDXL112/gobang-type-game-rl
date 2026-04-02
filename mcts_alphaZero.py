# -*- coding: utf-8 -*-
"""
AlphaGo Zero 风格的蒙特卡洛树搜索（MCTS）。
使用策略-价值网络指导搜索并评估叶节点。
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


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
    """蒙特卡洛树搜索实现。"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 接收棋盘状态，返回 (动作, 概率) 元组列表和
            当前局面的评分 [-1, 1] 的函数
        c_puct: 控制探索与利用权衡的参数，值越大越依赖先验
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

        # 使用网络评估叶节点，输出动作概率和局面评分
        action_probs, leaf_value = self._policy(state)
        # 检查游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 终局状态使用真实胜负作为叶节点值
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 回传更新路径上所有节点的访问次数和价值
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        执行所有模拟，返回可用动作及其对应概率。
        state: 当前游戏状态
        temp: 温度参数，控制探索程度
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点子节点的访问次数计算动作概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

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
    """基于 MCTS 的 AI 玩家。"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # MCTS 返回的策略向量 pi
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 自博弈模式：添加狄利克雷噪声以促进探索
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点并复用搜索树
                self.mcts.update_with_move(move)
            else:
                # 对战模式：在默认 temp=1e-3 下几乎等价于选概率最大的动作
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: 棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)
