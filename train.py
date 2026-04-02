# -*- coding: utf-8 -*-
"""
四子棋 AlphaZero 训练管线。
包括自我博弈数据采集、数据增强、策略网络更新和定期评估。
"""

from __future__ import print_function
import os
import csv
import time
import random
import torch
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class TrainPipeline():
    def __init__(self, init_model=None, resume_checkpoint=None):
        # 棋盘与游戏参数
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 4
        # 禁区：八角形边角区域
        self.restricted_positions = [
            (0, 0), (0, 1), (0, 2), (0, 5), (0, 6), (0, 7),
            (1, 0), (1, 1),                    (1, 6), (1, 7),
            (2, 0),                             (2, 7),
            (5, 0),                             (5, 7),
            (6, 0), (6, 1),                    (6, 6), (6, 7),
            (7, 0), (7, 1), (7, 2), (7, 5), (7, 6), (7, 7),
        ]
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row,
                           restricted_positions=self.restricted_positions)
        self.game = Game(self.board)
        # 训练超参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 根据 KL 散度自适应调整学习率
        self.temp = 1.0  # 温度参数
        self.n_playout = 400  # 每步的 MCTS 模拟次数
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # 训练的 mini-batch 大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的训练步数
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # 纯 MCTS 评估对手的模拟次数
        self.pure_mcts_playout_num = 1000

        # --- 输出目录 ---
        self.save_dir = './saved_models'
        self.log_dir = './logs'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.train_log_path = os.path.join(self.log_dir, 'train_log.csv')
        self.eval_log_path = os.path.join(self.log_dir, 'eval_log.csv')

        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        # --- 断点续训 ---
        self.start_batch = 0
        if resume_checkpoint and os.path.isfile(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)

    # ------------------------------------------------------------------
    # 日志辅助
    # ------------------------------------------------------------------
    def _init_log_files(self):
        """如果日志文件不存在则创建并写入表头。"""
        if not os.path.exists(self.train_log_path):
            with open(self.train_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['batch', 'episode_len', 'loss', 'entropy',
                                 'kl', 'lr_multiplier',
                                 'explained_var_old', 'explained_var_new',
                                 'elapsed_sec'])
        if not os.path.exists(self.eval_log_path):
            with open(self.eval_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['batch', 'win', 'lose', 'tie',
                                 'win_ratio', 'best_win_ratio',
                                 'pure_mcts_playout', 'elapsed_sec'])

    def _log_train(self, batch, episode_len, loss, entropy,
                   kl, lr_mult, ev_old, ev_new, elapsed):
        with open(self.train_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                batch, episode_len,
                '{:.6f}'.format(loss) if loss is not None else '',
                '{:.6f}'.format(entropy) if entropy is not None else '',
                '{:.6f}'.format(kl) if kl is not None else '',
                '{:.4f}'.format(lr_mult),
                '{:.4f}'.format(ev_old) if ev_old is not None else '',
                '{:.4f}'.format(ev_new) if ev_new is not None else '',
                '{:.1f}'.format(elapsed)])

    def _log_eval(self, batch, win, lose, tie, win_ratio, best_ratio,
                  pure_playout, elapsed):
        with open(self.eval_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                batch, win, lose, tie,
                '{:.4f}'.format(win_ratio),
                '{:.4f}'.format(best_ratio),
                pure_playout,
                '{:.1f}'.format(elapsed)])

    # ------------------------------------------------------------------
    # 检查点辅助
    # ------------------------------------------------------------------
    def _save_checkpoint(self, batch_idx):
        ckpt = {
            'batch': batch_idx,
            'best_win_ratio': self.best_win_ratio,
            'lr_multiplier': self.lr_multiplier,
            'pure_mcts_playout_num': self.pure_mcts_playout_num,
            'net': self.policy_value_net.policy_value_net.state_dict(),
            'optimizer': self.policy_value_net.optimizer.state_dict(),
        }
        path = os.path.join(self.save_dir, 'checkpoint.pth')
        torch.save(ckpt, path)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.policy_value_net.device)
        self.policy_value_net.policy_value_net.load_state_dict(ckpt['net'])
        self.policy_value_net.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_batch = ckpt.get('batch', 0) + 1  # 从下一批继续
        self.best_win_ratio = ckpt.get('best_win_ratio', 0.0)
        self.lr_multiplier = ckpt.get('lr_multiplier', 1.0)
        self.pure_mcts_playout_num = ckpt.get('pure_mcts_playout_num', 1000)
        # 用恢复的策略函数重建 MCTS 玩家
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        print("已从检查点恢复: batch={}, best_win_ratio={:.3f}, "
              "lr_multiplier={:.3f}".format(
                  self.start_batch, self.best_win_ratio, self.lr_multiplier))

    # ------------------------------------------------------------------
    # 数据增强
    # ------------------------------------------------------------------
    def get_equi_data(self, play_data):
        """通过旋转和翻转扩充训练数据集。"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    # ------------------------------------------------------------------
    # 核心训练循环
    # ------------------------------------------------------------------
    def collect_selfplay_data(self, n_games=1):
        """通过自我博弈采集训练数据。"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """更新策略-价值网络参数。"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:
                break
        # 根据 KL 散度自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy, kl, explained_var_old, explained_var_new

    def policy_evaluate(self, n_games=10):
        """通过与纯 MCTS 对手对战来评估当前策略的强度。"""
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio, win_cnt

    def run(self):
        """运行训练管线。"""
        self._init_log_files()
        t_start = time.time()

        try:
            for i in range(self.start_batch, self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                elapsed = time.time() - t_start
                print("batch i:{}, episode_len:{}, elapsed:{:.0f}s".format(
                        i + 1, self.episode_len, elapsed))

                loss, entropy, kl, ev_old, ev_new = None, None, None, None, None
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, kl, ev_old, ev_new = self.policy_update()

                self._log_train(i + 1, self.episode_len,
                                loss, entropy, kl, self.lr_multiplier,
                                ev_old, ev_new, elapsed)

                # 定期评估与保存
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio, win_cnt = self.policy_evaluate()
                    elapsed = time.time() - t_start
                    self._log_eval(i + 1, win_cnt[1], win_cnt[2], win_cnt[-1],
                                   win_ratio, self.best_win_ratio,
                                   self.pure_mcts_playout_num, elapsed)
                    # 保存当前模型和检查点
                    self.policy_value_net.save_model(
                        os.path.join(self.save_dir, 'current_policy.model'))
                    self._save_checkpoint(i)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(
                            os.path.join(self.save_dir, 'best_policy.model'))
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

            # 训练结束，最终保存
            self.policy_value_net.save_model(
                os.path.join(self.save_dir, 'current_policy.model'))
            self._save_checkpoint(self.game_batch_num - 1)
            print("训练完成，总用时: {:.0f}s".format(time.time() - t_start))

        except KeyboardInterrupt:
            print('\n\r中断 — 正在保存检查点')
            self._save_checkpoint(max(i, 0))
            self.policy_value_net.save_model(
                os.path.join(self.save_dir, 'current_policy.model'))


if __name__ == '__main__':
    # 从头训练:
    training_pipeline = TrainPipeline()
    # 从检查点恢复:
    # training_pipeline = TrainPipeline(resume_checkpoint='./saved_models/checkpoint.pth')
    training_pipeline.run()
