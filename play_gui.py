# -*- coding: utf-8 -*-
"""
四子棋 AlphaZero 图形化人机对战界面。
功能：棋盘可视化、先手/后手选择、网络评分显示、落子概率热力图。
"""

import tkinter as tk
import numpy as np
import threading
from game import Board
from mcts_alphaZero import MCTS
from policy_value_net import PolicyValueNet

# ======================== 配置 ========================
BOARD_W = 8
BOARD_H = 8
N_IN_ROW = 4
RESTRICTED = [
    (0, 0), (0, 1), (0, 2), (0, 5), (0, 6), (0, 7),
    (1, 0), (1, 1),                    (1, 6), (1, 7),
    (2, 0),                             (2, 7),
    (5, 0),                             (5, 7),
    (6, 0), (6, 1),                    (6, 6), (6, 7),
    (7, 0), (7, 1), (7, 2), (7, 5), (7, 6), (7, 7),
]
MODEL_FILE = './best_policy.model'
AI_PLAYOUT = 400

CELL = 62
PAD = 24

# ======================== 颜色 ========================
C_BG       = '#0f172a'
C_PANEL    = '#1e293b'
C_BOARD    = '#0f3460'
C_GRID     = '#1e3a5f'
C_RESTR    = '#334155'
C_EMPTY    = '#1e3a5f'
C_P1       = '#ef4444'
C_P1_LT    = '#fca5a5'
C_P2       = '#3b82f6'
C_P2_LT    = '#93c5fd'
C_GOLD     = '#fbbf24'
C_TEXT     = '#f1f5f9'
C_TEXT_DIM = '#94a3b8'
C_WIN      = '#4ade80'
C_LOSE     = '#f87171'


class ConnectFourApp:
    """主应用类。"""

    def __init__(self, master):
        self.master = master
        master.title('四子棋 AlphaZero 人机对战')
        master.configure(bg=C_BG)
        master.resizable(False, False)

        # 状态变量
        self.board = None
        self.net = None
        self.human = None        # 玩家编号 (1 或 2)
        self.ai = None           # AI 编号
        self.active = False
        self.thinking = False
        self.probs = None        # (64,) 概率显示数组
        self.value = 0.0
        self.move_count = 0

        self._build_ui()
        self._load_model()
        self._draw_empty_boards()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # ---- 顶部控制栏 ----
        top = tk.Frame(self.master, bg=C_PANEL, padx=12, pady=8)
        top.pack(fill='x')

        tk.Label(top, text='四子棋 AlphaZero',
                 font=('Microsoft YaHei', 16, 'bold'),
                 fg=C_TEXT, bg=C_PANEL).pack(side='left')

        bkw = dict(font=('Microsoft YaHei', 10), relief='flat',
                   padx=14, pady=4, cursor='hand2',
                   activebackground='#475569')

        self.b_first = tk.Button(
            top, text='先手开始 (X)', bg=C_P1, fg='white',
            command=lambda: self._start_game(True), **bkw)
        self.b_first.pack(side='right', padx=3)

        self.b_second = tk.Button(
            top, text='后手开始 (O)', bg=C_P2, fg='white',
            command=lambda: self._start_game(False), **bkw)
        self.b_second.pack(side='right', padx=3)

        self.b_new = tk.Button(
            top, text='新游戏', bg='#475569', fg='white',
            command=self._new_game, **bkw)
        self.b_new.pack(side='right', padx=3)

        # ---- 主内容区 ----
        body = tk.Frame(self.master, bg=C_BG)
        body.pack(fill='both', expand=True, padx=10, pady=8)

        # -- 左侧: 棋盘 --
        left = tk.Frame(body, bg=C_BG)
        left.pack(side='left')

        tk.Label(left, text='棋盘 (点击落子)',
                 font=('Microsoft YaHei', 11, 'bold'),
                 fg=C_TEXT, bg=C_BG).pack(pady=(0, 4))

        cw = BOARD_W * CELL + 2 * PAD
        ch = BOARD_H * CELL + 2 * PAD
        self.cv_board = tk.Canvas(left, width=cw, height=ch,
                                  bg=C_BOARD, highlightthickness=0)
        self.cv_board.pack()
        self.cv_board.bind('<Button-1>', self._on_click)
        self.cv_board.bind('<Motion>', self._on_hover)
        self.cv_board.bind('<Leave>', lambda e: self.cv_board.delete('hover'))

        # -- 右侧: 热力图 + 评分 --
        right = tk.Frame(body, bg=C_BG)
        right.pack(side='left', padx=(14, 0))

        tk.Label(right, text='网络落子概率评估',
                 font=('Microsoft YaHei', 11, 'bold'),
                 fg=C_TEXT, bg=C_BG).pack(pady=(0, 4))

        self.cv_heat = tk.Canvas(right, width=cw, height=ch,
                                 bg=C_BOARD, highlightthickness=0)
        self.cv_heat.pack()

        # 评分条
        vbox = tk.Frame(right, bg=C_BG)
        vbox.pack(fill='x', pady=(10, 0), padx=4)

        bar_row = tk.Frame(vbox, bg=C_BG)
        bar_row.pack(fill='x')
        tk.Label(bar_row, text='AI 胜', font=('Microsoft YaHei', 9),
                 fg=C_LOSE, bg=C_BG).pack(side='left')
        self.cv_bar = tk.Canvas(bar_row, width=300, height=22,
                                bg='#334155', highlightthickness=0)
        self.cv_bar.pack(side='left', padx=6)
        tk.Label(bar_row, text='你 胜', font=('Microsoft YaHei', 9),
                 fg=C_WIN, bg=C_BG).pack(side='left')

        self.lbl_val = tk.Label(vbox, text='局面评分: --',
                                font=('Microsoft YaHei', 13, 'bold'),
                                fg=C_TEXT, bg=C_BG)
        self.lbl_val.pack(pady=(6, 0))

        # 手数计数器
        self.lbl_moves = tk.Label(vbox, text='',
                                  font=('Microsoft YaHei', 10),
                                  fg=C_TEXT_DIM, bg=C_BG)
        self.lbl_moves.pack(pady=(2, 0))

        # ---- 状态栏 ----
        sbar = tk.Frame(self.master, bg=C_PANEL, height=34)
        sbar.pack(fill='x', padx=10, pady=(0, 10))
        sbar.pack_propagate(False)
        self.lbl_status = tk.Label(
            sbar, text='请选择先手或后手开始游戏',
            font=('Microsoft YaHei', 10), fg=C_TEXT, bg=C_PANEL)
        self.lbl_status.pack(expand=True)

    # ---------------------------------------------------------------- 模型
    def _load_model(self):
        try:
            self.net = PolicyValueNet(BOARD_W, BOARD_H, model_file=MODEL_FILE)
            self.lbl_status.config(text='模型已加载，请选择先手或后手开始')
        except Exception as e:
            self.lbl_status.config(text='模型加载失败: {}'.format(e))

    # ----------------------------------------------------------- 游戏控制
    def _start_game(self, human_first):
        if self.net is None:
            return
        self.board = Board(width=BOARD_W, height=BOARD_H,
                           n_in_row=N_IN_ROW,
                           restricted_positions=RESTRICTED)
        self.board.init_board(start_player=0)
        self.human = 1 if human_first else 2
        self.ai = 2 if human_first else 1
        self.active = True
        self.thinking = False
        self.probs = None
        self.value = 0.0
        self.move_count = 0
        self._toggle_buttons(False)
        self._update_eval()
        self._refresh()

        if not human_first:
            self._run_ai()
        else:
            self.lbl_status.config(text='轮到你下棋 (X)')

    def _new_game(self):
        self.active = False
        self.thinking = False
        self.board = None
        self.probs = None
        self.value = 0.0
        self.move_count = 0
        self._toggle_buttons(True)
        self._draw_empty_boards()
        self.cv_bar.delete('all')
        self.cv_bar.create_rectangle(0, 0, 300, 22, fill='#334155', outline='')
        self.lbl_val.config(text='局面评分: --')
        self.lbl_moves.config(text='')
        self.lbl_status.config(text='请选择先手或后手开始游戏')

    def _toggle_buttons(self, on):
        st = 'normal' if on else 'disabled'
        self.b_first.config(state=st)
        self.b_second.config(state=st)

    # ---------------------------------------------------------- 评估
    def _update_eval(self):
        """获取网络原始策略与价值评估（瞬时完成）。"""
        if self.board is None:
            return
        act_iter, val = self.net.policy_value_fn(self.board)
        arr = np.zeros(BOARD_W * BOARD_H)
        for move, prob in act_iter:
            arr[move] = prob
        self.probs = arr
        self.value = val

    # ----------------------------------------------------------- 绘制
    def _rc2xy(self, r, c):
        """棋盘 (row,col) -> 画布中心坐标。row-0 为底部行。"""
        x = PAD + c * CELL + CELL // 2
        y = PAD + (BOARD_H - 1 - r) * CELL + CELL // 2
        return x, y

    def _xy2rc(self, mx, my):
        c = (mx - PAD) // CELL
        r = BOARD_H - 1 - (my - PAD) // CELL
        if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
            return int(r), int(c)
        return None

    def _draw_empty_boards(self):
        for cv in (self.cv_board, self.cv_heat):
            cv.delete('all')
            self._draw_grid(cv)

    def _draw_grid(self, cv):
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                x1 = PAD + c * CELL
                y1 = PAD + (BOARD_H - 1 - r) * CELL
                x2, y2 = x1 + CELL, y1 + CELL
                if (r, c) in RESTRICTED:
                    cv.create_rectangle(x1, y1, x2, y2,
                                        fill=C_RESTR, outline='#1e293b')
                else:
                    cv.create_rectangle(x1, y1, x2, y2,
                                        fill=C_BOARD, outline=C_GRID)
                    cx, cy = self._rc2xy(r, c)
                    rad = CELL // 2 - 6
                    cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                   fill=C_EMPTY, outline=C_GRID, tags='slot')
        # 列标签
        for c in range(BOARD_W):
            x = PAD + c * CELL + CELL // 2
            cv.create_text(x, PAD // 2, text=str(c),
                           fill=C_TEXT_DIM, font=('Arial', 8))
        # 行标签
        for r in range(BOARD_H):
            y = PAD + (BOARD_H - 1 - r) * CELL + CELL // 2
            cv.create_text(PAD // 2, y, text=str(r),
                           fill=C_TEXT_DIM, font=('Arial', 8))

    def _refresh(self):
        self._draw_pieces()
        self._draw_heatmap()
        self._draw_value_bar()
        self.lbl_moves.config(text='第 {} 手'.format(self.move_count))

    def _draw_pieces(self):
        self.cv_board.delete('piece')
        self.cv_board.delete('last_move')
        if self.board is None:
            return
        for move, player in self.board.states.items():
            r, c = move // BOARD_W, move % BOARD_W
            cx, cy = self._rc2xy(r, c)
            rad = CELL // 2 - 6
            fill = C_P1 if player == 1 else C_P2
            outl = C_P1_LT if player == 1 else C_P2_LT
            self.cv_board.create_oval(
                cx - rad, cy - rad, cx + rad, cy + rad,
                fill=fill, outline=outl, width=2, tags='piece')
            sym = 'X' if player == 1 else 'O'
            self.cv_board.create_text(
                cx, cy, text=sym, fill='white',
                font=('Arial', 15, 'bold'), tags='piece')

        # 最后一手高亮金色边框
        if self.board.last_move != -1:
            r = self.board.last_move // BOARD_W
            c = self.board.last_move % BOARD_W
            cx, cy = self._rc2xy(r, c)
            rad = CELL // 2 - 3
            self.cv_board.create_oval(
                cx - rad, cy - rad, cx + rad, cy + rad,
                outline=C_GOLD, width=3, tags='last_move')

    def _draw_heatmap(self):
        self.cv_heat.delete('heat')
        if self.board is None or self.probs is None:
            return
        pmax = max(self.probs.max(), 1e-9)
        for move in self.board.availables:
            r, c = move // BOARD_W, move % BOARD_W
            p = self.probs[move]
            t = p / pmax  # 归一化强度 0..1

            x1 = PAD + c * CELL + 2
            y1 = PAD + (BOARD_H - 1 - r) * CELL + 2
            x2, y2 = x1 + CELL - 4, y1 + CELL - 4
            cx, cy = self._rc2xy(r, c)

            color = self._heat_color(t)
            self.cv_heat.create_rectangle(
                x1, y1, x2, y2, fill=color, outline='', tags='heat')

            if p > 0.003:
                fg = 'white' if t > 0.35 else C_TEXT_DIM
                self.cv_heat.create_text(
                    cx, cy, text='{:.0%}'.format(p),
                    fill=fg, font=('Arial', 9, 'bold'), tags='heat')

    @staticmethod
    def _heat_color(t):
        """将归一化强度 [0,1] 映射为颜色：深蓝 -> 青 -> 黄 -> 红。"""
        if t < 0.33:
            s = t / 0.33
            r = int(30 + (6 - 30) * s)
            g = int(58 + (182 - 58) * s)
            b = int(95 + (212 - 95) * s)
        elif t < 0.66:
            s = (t - 0.33) / 0.33
            r = int(6 + (234 - 6) * s)
            g = int(182 + (179 - 182) * s)
            b = int(212 + (8 - 212) * s)
        else:
            s = (t - 0.66) / 0.34
            r = int(234 + (239 - 234) * s)
            g = int(179 + (68 - 179) * s)
            b = int(8 + (68 - 8) * s)
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def _draw_value_bar(self):
        self.cv_bar.delete('all')
        w, h = 300, 22
        mid = w // 2
        self.cv_bar.create_rectangle(0, 0, w, h, fill='#334155', outline='')
        self.cv_bar.create_line(mid, 0, mid, h, fill='#64748b', width=1)

        if self.board is not None:
            v = self.value
            # 转换为人类视角
            if self.board.get_current_player() != self.human:
                v = -v
        else:
            v = 0.0
        bw = int(min(abs(v), 1.0) * mid)
        if v >= 0:
            self.cv_bar.create_rectangle(mid, 0, mid + bw, h,
                                         fill=C_WIN, outline='')
        else:
            self.cv_bar.create_rectangle(mid - bw, 0, mid, h,
                                         fill=C_LOSE, outline='')

        sign = '+' if v >= 0 else ''
        self.lbl_val.config(text='局面评分: {}{:.3f}'.format(sign, v))

    # --------------------------------------------------------- 交互
    def _on_hover(self, event):
        """鼠标悬停时显示虚线预览。"""
        self.cv_board.delete('hover')
        if not self.active or self.thinking or self.board is None:
            return
        pos = self._xy2rc(event.x, event.y)
        if pos is None:
            return
        r, c = pos
        move = r * BOARD_W + c
        if move in self.board.availables:
            cx, cy = self._rc2xy(r, c)
            rad = CELL // 2 - 6
            fill = C_P1 if self.human == 1 else C_P2
            self.cv_board.create_oval(
                cx - rad, cy - rad, cx + rad, cy + rad,
                fill='', outline=fill, width=2, dash=(4, 4), tags='hover')

    def _on_click(self, event):
        """点击落子。"""
        if not self.active or self.thinking or self.board is None:
            return
        if self.board.get_current_player() != self.human:
            return
        pos = self._xy2rc(event.x, event.y)
        if pos is None:
            return
        r, c = pos
        move = r * BOARD_W + c
        if move not in self.board.availables:
            return

        # 执行人类落子
        self.board.do_move(move)
        self.move_count += 1
        self._update_eval()
        self._refresh()
        if self._check_end():
            return
        self._run_ai()

    def _run_ai(self):
        """在后台线程中运行 AI 的 MCTS 搜索。"""
        self.thinking = True
        self.lbl_status.config(text='AI 思考中...')
        self.master.update_idletasks()

        def worker():
            mcts = MCTS(self.net.policy_value_fn, c_puct=5, n_playout=AI_PLAYOUT)
            acts, probs = mcts.get_move_probs(self.board, temp=1e-3)

            full = np.zeros(BOARD_W * BOARD_H)
            for a, p in zip(acts, probs):
                full[a] = p
            move = acts[int(np.argmax(probs))]
            self.master.after(0, lambda: self._apply_ai(move, full))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_ai(self, move, mcts_probs):
        """在主线程中执行 AI 落子并更新界面。"""
        self.thinking = False
        self.board.do_move(move)
        self.move_count += 1

        # 显示 MCTS 搜索概率
        self.probs = mcts_probs
        # 更新网络对新局面的价值评估
        if self.board.availables:
            _, val = self.net.policy_value_fn(self.board)
            self.value = val
        self._refresh()

        if self._check_end():
            return
        sym = 'X' if self.human == 1 else 'O'
        self.lbl_status.config(text='轮到你下棋 ({})'.format(sym))

    def _check_end(self):
        """检查游戏是否结束并更新状态。"""
        end, winner = self.board.game_end()
        if not end:
            return False
        self.active = False
        self._toggle_buttons(True)
        if winner == -1:
            msg = '游戏结束 — 平局!'
        elif winner == self.human:
            msg = '游戏结束 — 你赢了!'
        else:
            msg = '游戏结束 — AI 赢了!'
        self.lbl_status.config(text=msg)
        self._refresh()
        return True


def main():
    root = tk.Tk()
    ConnectFourApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
