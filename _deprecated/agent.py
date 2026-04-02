import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import PolicyValueNet
# Agent：单权重自博弈；基于策略先验的MCTS与价值网络


class Agent(nn.Module):
    def __init__(
        self,
        stem_kernel_size: int = 5,
        block_kernel_size: int = 3,
        channels: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        bias: bool = True,
        mcts_num_simulations: int = 256,
        mcts_max_depth: int = 8,
        c_puct: float = 1.5,
        rollout_per_leaf: int = 16,
        rollout_max_moves: int = 64,
        use_policy_sampling: bool = False,
        force_win_move: bool = True,
        device=None,
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 策略-价值网络（输入28通道特征）
        self.net = PolicyValueNet(in_channels=28, stem_kernel_size=stem_kernel_size, block_kernel_size=block_kernel_size, channels=channels, num_layers=num_layers, activation=activation, bias=bias).to(self.device)
        self.mcts_num_simulations = int(mcts_num_simulations)
        self.mcts_max_depth = int(mcts_max_depth)
        self.c_puct = float(c_puct)
        self.rollout_per_leaf = int(rollout_per_leaf)
        self.rollout_max_moves = int(rollout_max_moves)
        self.use_policy_sampling = bool(use_policy_sampling)
        self.force_win_move = bool(force_win_move)

    def forward(self, state, side, batch_tracker):
        """
        前向推理：返回动作、策略概率、熵、以及价值估计
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device) if not isinstance(state, torch.Tensor) else state.to(self.device).float()
        if isinstance(side, torch.Tensor):
            side_vec = side.to(self.device).view(-1).float()
        else:
            side_vec = torch.full((state.shape[0],), float(int(side)), device=self.device)
        legal = batch_tracker.legal_mask().to(self.device)
        feats = self.extract_feature(state, side_vec)
        logits, value = self.net(feats)
        flat_logits = logits.flatten(1)
        mask = legal.view(legal.shape[0], -1)
        neg_inf = torch.finfo(flat_logits.dtype).min
        flat_logits = torch.where(mask, flat_logits, torch.full_like(flat_logits, neg_inf))
        pi = torch.softmax(flat_logits, dim=-1)

        # 获取棋盘状态
        b = state.clone()
        side_t = side_vec.view(-1, 1, 1)
        my = (b == side_t).float()
        op = (b == -side_t).float()
        empty = (b == 0.0).float()
        B = state.shape[0]

        # === 计算各种启发式分数 ===
        heuristic_scores = torch.zeros(B, 64, device=self.device)

        # 1. 一步制胜：最高优先级 (分数=100)
        my_win_mask = self._winning_mask(my, empty)
        my_win_flat = my_win_mask.view(B, -1).float()
        heuristic_scores += my_win_flat * 100.0

        # 2. 阻挡对手一步制胜：次高优先级 (分数=90)
        op_win_mask = self._winning_mask(op, empty)
        op_win_flat = op_win_mask.view(B, -1).float()
        heuristic_scores += op_win_flat * 90.0

        # 3. 形成活三：高优先级 (分数=50)
        my_live3 = self._get_live3_positions(my, op, empty)
        heuristic_scores += my_live3.view(B, -1) * 50.0

        # 4. 阻挡对手活三：较高优先级 (分数=45)
        op_live3 = self._get_live3_positions(op, my, empty)
        heuristic_scores += op_live3.view(B, -1) * 45.0

        # 5. 中心位置奖励 (分数=5-10)
        center_bonus = self._get_center_bonus(empty)  # [B, 64]
        heuristic_scores += center_bonus * 5.0

        # 6. 形成活二：低优先级 (分数=10)
        my_live2 = self._get_live2_positions(my, op, empty)
        heuristic_scores += my_live2.view(B, -1) * 10.0

        # 屏蔽非法位置
        heuristic_scores = torch.where(mask, heuristic_scores, torch.full_like(heuristic_scores, -1e9))

        # === 决定动作 ===
        if self.use_policy_sampling or self.mcts_num_simulations <= 0:
            # 混合策略：以一定概率选择启发式最佳，否则按网络策略采样
            # 这保证了探索的同时，也有足够强的baseline
            heuristic_actions = torch.argmax(heuristic_scores, dim=1)
            sample_actions = torch.multinomial(pi, 1).squeeze(1)

            # 如果有高分位置（>40分），90%概率选它；否则按网络采样
            max_scores = heuristic_scores.max(dim=1).values
            use_heuristic = (max_scores > 40.0) & (torch.rand(B, device=self.device) < 0.95)
            actions = torch.where(use_heuristic, heuristic_actions, sample_actions)
        else:
            counts = self.mcts_search_batched(batch_tracker, side_vec, pi, value)
            actions = torch.argmax(counts, dim=1)

        probs = pi[torch.arange(B, device=self.device), actions]
        entropy = -(pi * torch.log(pi + 1e-9)).sum(dim=1)
        value = value.squeeze(-1)
        return actions, probs, entropy, value

    def _get_live3_positions(self, me, op, empty):
        """获取能形成活三的位置"""
        # 复用_live_rush的结果
        _, live3, _ = self._live_rush(me, op)
        return live3

    def _get_live2_positions(self, me, op, empty):
        """获取能形成活二的位置"""
        live2, _, _ = self._live_rush(me, op)
        return live2

    def _get_center_bonus(self, empty):
        """中心位置奖励"""
        B = empty.shape[0]
        device = empty.device
        # 中心4x4区域有额外奖励
        center_mask = torch.zeros(8, 8, device=device)
        center_mask[2:6, 2:6] = 1.0
        # 最中心的2x2奖励更高
        center_mask[3:5, 3:5] = 2.0
        return (empty * center_mask.unsqueeze(0)).view(B, -1)

    def extract_feature(self, state, side):
        """
        构造28通道特征：
        原有19通道：
          - 我/对掩码 (2)
          - 连通度(1/2/3)*2 (6)
          - 活2/活3/冲3/双二/双三*2 (10)
          - legal_mask (1)
        新增9通道：
          - 当前玩家标识 (1)  <-- 新增！全1或全-1
          - 我的制胜位置 (1)
          - 对手制胜位置 (1)
          - 我的活三威胁 (1)
          - 对手活三威胁 (1)
          - 我的冲四位置 (1)
          - 对手冲四位置 (1)
          - 我的双三位置 (1)
          - 对手双三位置 (1)
        """
        b = state.clone()
        side_t = side.view(-1, 1, 1)
        my = (b == side_t).float()
        op = (b == -side_t).float()
        empty = (b == 0.0).float()

        feats = []
        # === 原有特征 (19通道) ===
        feats.append(my)  # 0: 我的棋子
        feats.append(op)  # 1: 对手棋子
        feats += self._connectivity_channels(my, [1, 2, 3])  # 2-4: 我的连通度
        feats += self._connectivity_channels(op, [1, 2, 3])  # 5-7: 对手连通度
        live2_my, live3_my, rush3_my = self._live_rush(my, op)
        live2_op, live3_op, rush3_op = self._live_rush(op, my)
        feats.append(live2_my)   # 8
        feats.append(live3_my)   # 9
        feats.append(rush3_my)   # 10
        feats.append(self._double_two(my, op))   # 11
        feats.append(self._double_three(my, op)) # 12
        feats.append(live2_op)   # 13
        feats.append(live3_op)   # 14
        feats.append(rush3_op)   # 15
        feats.append(self._double_two(op, my))   # 16
        feats.append(self._double_three(op, my)) # 17
        feats.append(empty)  # 18: 空位

        # === 新增特征 (9通道) ===
        # 19: 当前玩家标识（先手=全1，后手=全-1）
        side_channel = torch.full_like(my, fill_value=0.0)
        side_channel[:] = side.view(-1, 1, 1)
        feats.append(side_channel)

        # 20: 我的制胜位置（一步能赢）
        my_win = self._winning_mask(my, empty).float()
        feats.append(my_win)

        # 21: 对手制胜位置（必须阻挡！）
        op_win = self._winning_mask(op, empty).float()
        feats.append(op_win)

        # 22: 我的活三威胁位置
        my_live3_threat = self._threat_positions(my, op, empty, n=3)
        feats.append(my_live3_threat)

        # 23: 对手活三威胁位置
        op_live3_threat = self._threat_positions(op, my, empty, n=3)
        feats.append(op_live3_threat)

        # 24-27: 冲四和双三位置
        feats.append(self._rush_four_positions(my, op, empty))
        feats.append(self._rush_four_positions(op, my, empty))
        feats.append(self._double_three(my, op))
        feats.append(self._double_three(op, my))

        x = torch.stack(feats, dim=1)
        return x

    def _threat_positions(self, me, op, empty, n=3):
        """
        找出能形成n连威胁的位置
        对于n=3，找出放置一子后能形成活三的位置
        """
        B = me.shape[0]
        device = me.device
        threat = torch.zeros_like(me)

        # 检查每个空位：如果在这里放子，是否会形成n连且两端有空
        # 简化实现：检查 [空, 我, 我, 空] 这种模式中空位的位置
        x_me = me.unsqueeze(1)

        # 水平方向：检查 _XX_ 模式
        for pattern, pos_idx in [
            (torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.float32, device=device), [0, 3]),  # _11_
        ]:
            conv_result = F.conv2d(x_me, pattern)
            hits = (conv_result.squeeze(1) == float(n - 1))
            # 标记能形成威胁的位置
            for pos in pos_idx:
                k = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device=device)
                k[0, 0, 0, pos] = 1.0
                expanded = F.conv_transpose2d(hits.unsqueeze(1).float(), k, stride=1).squeeze(1)
                threat = torch.max(threat, expanded)

        # 垂直方向
        for pattern, pos_idx in [
            (torch.tensor([[[[0], [1], [1], [0]]]], dtype=torch.float32, device=device), [0, 3]),
        ]:
            conv_result = F.conv2d(x_me, pattern)
            hits = (conv_result.squeeze(1) == float(n - 1))
            for pos in pos_idx:
                k = torch.zeros(1, 1, 4, 1, dtype=torch.float32, device=device)
                k[0, 0, pos, 0] = 1.0
                expanded = F.conv_transpose2d(hits.unsqueeze(1).float(), k, stride=1).squeeze(1)
                threat = torch.max(threat, expanded)

        return threat

    def _rush_four_positions(self, me, op, empty):
        """
        找出冲四位置：已有三子连成一线，一端被堵，另一端空
        模式：X111_ 或 _111X（X=对手或边界，_=空）
        """
        B = me.shape[0]
        device = me.device
        rush4 = torch.zeros_like(me)

        x_me = me.unsqueeze(1)
        x_op = op.unsqueeze(1)
        x_empty = empty.unsqueeze(1)

        # 水平方向：检查 111_ 和 _111 模式
        # 冲四：三连且一端有对手或边界
        k3 = torch.ones((1, 1, 1, 3), device=device)
        sum3 = F.conv2d(x_me, k3).squeeze(1)  # 三连位置

        # 左端被堵，右端空
        left_blocked = F.pad(x_op, (1, 0), value=0.0)
        right_empty = F.pad(x_empty, (0, 1), value=0.0)

        # 简化：标记有三连的位置附近的空位
        hits = (sum3 == 3.0)
        # 扩展到原始位置
        expanded = F.conv_transpose2d(hits.unsqueeze(1).float(), k3, stride=1).squeeze(1)
        rush4 = torch.where((expanded > 0) & (empty > 0), torch.ones_like(rush4), rush4)

        # 垂直方向
        k3v = torch.ones((1, 1, 3, 1), device=device)
        sum3v = F.conv2d(x_me, k3v).squeeze(1)
        hits_v = (sum3v == 3.0)
        expanded_v = F.conv_transpose2d(hits_v.unsqueeze(1).float(), k3v, stride=1).squeeze(1)
        rush4 = torch.where((expanded_v > 0) & (empty > 0), torch.ones_like(rush4), rush4)

        return rush4

    def _winning_mask(self, me, empty):
        device = me.device
        x_me = me.unsqueeze(1)
        x_empty = empty.unsqueeze(1)
        win = torch.zeros_like(me, dtype=torch.bool)
        # horizontal
        kh = torch.ones((1, 1, 1, 4), device=device)
        sumH = F.conv2d(x_me, kh).squeeze(1) == 3
        pos_k = [torch.tensor([[[[1, 0, 0, 0]]]], dtype=torch.float32, device=device),
                 torch.tensor([[[[0, 1, 0, 0]]]], dtype=torch.float32, device=device),
                 torch.tensor([[[[0, 0, 1, 0]]]], dtype=torch.float32, device=device),
                 torch.tensor([[[[0, 0, 0, 1]]]], dtype=torch.float32, device=device)]
        for p in range(4):
            ep = F.conv2d(x_empty, (pos_k[p] > 0).float()).squeeze(1) == 1
            hits = sumH & ep
            seg = F.conv_transpose2d(hits.unsqueeze(1).float(), pos_k[p], stride=1).squeeze(1) > 0
            win = win | seg
        # vertical
        kv = torch.ones((1, 1, 4, 1), device=device)
        sumV = F.conv2d(x_me, kv).squeeze(1) == 3
        pos_v = [torch.tensor([[[[1], [0], [0], [0]]]], dtype=torch.float32, device=device),
                 torch.tensor([[[[0], [1], [0], [0]]]], dtype=torch.float32, device=device),
                 torch.tensor([[[[0], [0], [1], [0]]]], dtype=torch.float32, device=device),
                 torch.tensor([[[[0], [0], [0], [1]]]], dtype=torch.float32, device=device)]
        for p in range(4):
            ep = F.conv2d(x_empty, (pos_v[p] > 0).float()).squeeze(1) == 1
            hits = sumV & ep
            seg = F.conv_transpose2d(hits.unsqueeze(1).float(), pos_v[p], stride=1).squeeze(1) > 0
            win = win | seg
        # main diagonal
        kd = torch.zeros((1, 1, 4, 4), device=device)
        for t in range(4): kd[0, 0, t, t] = 1.0
        sumD = F.conv2d(x_me, kd).squeeze(1) == 3
        for t in range(4):
            kpos = torch.zeros((1, 1, 4, 4), device=device); kpos[0, 0, t, t] = 1.0
            ep = F.conv2d(x_empty, (kpos > 0).float()).squeeze(1) == 1
            hits = sumD & ep
            seg = F.conv_transpose2d(hits.unsqueeze(1).float(), kpos, stride=1).squeeze(1) > 0
            win = win | seg
        # anti diagonal
        ka = torch.zeros((1, 1, 4, 4), device=device)
        for t in range(4): ka[0, 0, t, 3 - t] = 1.0
        sumA = F.conv2d(x_me, ka).squeeze(1) == 3
        for t in range(4):
            kpos = torch.zeros((1, 1, 4, 4), device=device); kpos[0, 0, t, 3 - t] = 1.0
            ep = F.conv2d(x_empty, (kpos > 0).float()).squeeze(1) == 1
            hits = sumA & ep
            seg = F.conv_transpose2d(hits.unsqueeze(1).float(), kpos, stride=1).squeeze(1) > 0
            win = win | seg
        return win

    def _judge_batched(self, bd):
        x = bd.unsqueeze(1)
        kh = torch.ones((1, 1, 1, 4), device=self.device)
        kv = torch.ones((1, 1, 4, 1), device=self.device)
        kd = torch.zeros((1, 1, 4, 4), device=self.device)
        ka = torch.zeros((1, 1, 4, 4), device=self.device)
        kd[:, :, 0, 0] = 1.0; kd[:, :, 1, 1] = 1.0; kd[:, :, 2, 2] = 1.0; kd[:, :, 3, 3] = 1.0
        ka[:, :, 0, 3] = 1.0; ka[:, :, 1, 2] = 1.0; ka[:, :, 2, 1] = 1.0; ka[:, :, 3, 0] = 1.0
        sh = F.conv2d(x, kh); sv = F.conv2d(x, kv); sd = F.conv2d(x, kd); sa = F.conv2d(x, ka)
        p1_h = (sh == 4.0).flatten(1).any(1)
        p1_v = (sv == 4.0).flatten(1).any(1)
        p1_d = (sd == 4.0).flatten(1).any(1)
        p1_a = (sa == 4.0).flatten(1).any(1)
        p2_h = (sh == -4.0).flatten(1).any(1)
        p2_v = (sv == -4.0).flatten(1).any(1)
        p2_d = (sd == -4.0).flatten(1).any(1)
        p2_a = (sa == -4.0).flatten(1).any(1)
        win1 = p1_h | p1_v | p1_d | p1_a
        win2 = p2_h | p2_v | p2_d | p2_a
        full = (~(bd == 0.0)).view(bd.shape[0], -1).all(1)
        # winners: 1/-1/0
        winners = torch.where(win1, torch.ones(bd.shape[0], device=self.device), torch.zeros(bd.shape[0], device=self.device))
        winners = torch.where(win2 & (~win1), -torch.ones(bd.shape[0], device=self.device), winners)
        return winners

    @torch.no_grad()
    def _rollout_batched(self, boards, side_vec, max_steps):
        bd = boards.clone()
        cur = side_vec.clone()
        steps = 0
        done = torch.zeros(bd.shape[0], dtype=torch.bool, device=self.device)
        winners = torch.zeros(bd.shape[0], dtype=torch.float32, device=self.device)
        while steps < max_steps and (~done).any():
            legal = (bd == 0.0).view(bd.shape[0], -1)
            feats = self.extract_feature(bd, cur)
            logits = self.net(feats).flatten(1)
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(legal, logits, torch.full_like(logits, neg_inf))
            pi = torch.softmax(logits, dim=-1)
            # force win
            my = (bd == cur.view(-1, 1, 1)).float()
            empty = (bd == 0.0).float()
            win_mask = self._winning_mask(my, empty).view(bd.shape[0], -1)
            has_win = win_mask.any(1)
            sampled = torch.multinomial(pi, 1).squeeze(1)
            win_actions = torch.argmax(win_mask.float(), dim=1)
            actions = torch.where(has_win, win_actions, sampled)
            x = actions // 8; y = actions % 8
            idx = torch.arange(bd.shape[0], device=self.device)
            # apply where not done
            valid = ~done
            bd[valid, x[valid], y[valid]] = cur[valid]
            w = self._judge_batched(bd)
            newly_done = (w != 0) | (~(bd == 0.0).view(bd.shape[0], -1).any(1))
            winners = torch.where(newly_done & (~done), w.float(), winners)
            done = done | newly_done
            cur = torch.where(valid, -cur, cur)
            steps += 1
        # value from perspective cur at start: need original side_vec
        # Here side_vec is original mover; v = 1 if winners==side_vec; -1 if -side_vec; else 0
        v = torch.where(winners == side_vec, torch.ones_like(winners), torch.zeros_like(winners))
        v = torch.where(winners == -side_vec, -torch.ones_like(winners), v)
        return v

    def mcts_search_batched(self, batch_tracker, side_vec, prior_pi, root_value):
        """
        批量MCTS搜索，使用价值网络评估叶子节点（替代rollout）
        Args:
            batch_tracker: 环境实例
            side_vec: 当前玩家 [B]
            prior_pi: 策略先验 [B, 64]
            root_value: 根节点的价值估计 [B]
        Returns:
            N: 每个动作的访问次数 [B, 64]
        """
        B = side_vec.shape[0]
        board = batch_tracker.board.clone().to(self.device)
        P = prior_pi.clone()
        N = torch.zeros((B, 64), dtype=torch.float32, device=self.device)
        W = torch.zeros((B, 64), dtype=torch.float32, device=self.device)
        Q = torch.zeros((B, 64), dtype=torch.float32, device=self.device)

        for _ in range(self.mcts_num_simulations):
            legal = (board == 0.0).view(B, -1)
            # UCT公式
            uct = Q + self.c_puct * P * torch.sqrt(torch.clamp(N.sum(dim=1, keepdim=True), min=1.0)) / (1.0 + N)
            neg_inf = torch.finfo(uct.dtype).min
            uct = torch.where(legal, uct, torch.full_like(uct, neg_inf))
            a = torch.argmax(uct, dim=1)

            # 强制选择制胜动作
            my = (board == side_vec.view(-1, 1, 1)).float()
            empty = (board == 0.0).float()
            win_mask = self._winning_mask(my, empty).view(B, -1)
            has_win = win_mask.any(1)
            win_actions = torch.argmax(win_mask.float(), dim=1)
            a = torch.where(has_win, win_actions, a)

            x = a // 8
            y = a % 8
            idx = torch.arange(B, device=self.device)
            new_bd = board.clone()
            new_bd[idx, x, y] = side_vec[idx]

            # 检查是否终局
            w = self._judge_batched(new_bd)
            term = (w != 0)

            # 计算价值
            v = torch.zeros(B, dtype=torch.float32, device=self.device)
            v = torch.where(term & (w == side_vec), torch.ones_like(v), v)
            v = torch.where(term & (w == -side_vec), -torch.ones_like(v), v)

            # 非终局：使用价值网络评估（替代rollout）
            need_eval = ~term
            if need_eval.any():
                # 从对手视角评估，需要取反
                with torch.no_grad():
                    feats = self.extract_feature(new_bd[need_eval], -side_vec[need_eval])
                    _, leaf_value = self.net(feats)
                    # 价值是从对手视角的，所以取反
                    v[need_eval] = -leaf_value.squeeze(-1)

            # 更新统计
            N[idx, a] = N[idx, a] + 1.0
            W[idx, a] = W[idx, a] + v
            Q[idx, a] = W[idx, a] / N[idx, a]

        return N

    def _connectivity_channels(self, mask, Ns):
        B = mask.shape[0]
        device = mask.device
        out = []
        x = mask.unsqueeze(1)  # [B,1,8,8]
        for N in Ns:
            seg = torch.zeros_like(mask)
            # 水平
            kH = torch.ones((1, 1, 1, N), device=device)
            sumH = F.conv2d(x, kH)  # [B,1,8,8-N+1]
            hitsH = (sumH.squeeze(1) == float(N))
            padH = F.pad(mask, (1, 1, 0, 0))  # pad cols
            cols = torch.arange(0, 8 - N + 1, device=device)
            leftH = padH[:, :, cols]
            rightH = padH[:, :, cols + N + 1]
            hitsH = hitsH & (leftH == 0) & (rightH == 0)
            for t in range(N):
                seg[:, :, cols + t] = torch.where(hitsH, torch.ones_like(seg[:, :, cols + t]), seg[:, :, cols + t])
            # 垂直
            kV = torch.ones((1, 1, N, 1), device=device)
            sumV = F.conv2d(x, kV)  # [B,1,8-N+1,8]
            hitsV = (sumV.squeeze(1) == float(N))
            padV = F.pad(mask, (0, 0, 1, 1))  # pad rows
            rows = torch.arange(0, 8 - N + 1, device=device)
            topV = padV[:, rows, :]
            botV = padV[:, rows + N + 1, :]
            hitsV = hitsV & (topV == 0) & (botV == 0)
            for t in range(N):
                seg[:, rows + t, :] = torch.where(hitsV, torch.ones_like(seg[:, rows + t, :]), seg[:, rows + t, :])
            # 主对角线
            kD = torch.zeros((1, 1, N, N), device=device)
            for t in range(N):
                kD[0, 0, t, t] = 1.0
            sumD = F.conv2d(x, kD)  # [B,1,8-N+1,8-N+1]
            hitsD = (sumD.squeeze(1) == float(N))
            mp = F.pad(mask, (1, 1, 1, 1))  # [B,10,10]
            leftD = mp[:, 0 : 8 - N + 1, 0 : 8 - N + 1]
            rightD = mp[:, (N + 1) : (N + 1) + (8 - N + 1), (N + 1) : (N + 1) + (8 - N + 1)]
            hitsD = hitsD & (leftD == 0) & (rightD == 0)
            for t in range(N):
                seg[:, t : t + (8 - N + 1), t : t + (8 - N + 1)] = torch.where(
                    hitsD, torch.ones_like(seg[:, t : t + (8 - N + 1), t : t + (8 - N + 1)]), seg[:, t : t + (8 - N + 1), t : t + (8 - N + 1)]
                )
            # 反对角线（通过翻列处理为主对角）
            mf = torch.flip(mask, dims=[2])
            xf = mf.unsqueeze(1)
            sumAf = F.conv2d(xf, kD)  # 主对角于翻转
            hitsAf = (sumAf.squeeze(1) == float(N))
            mpf = F.pad(mf, (1, 1, 1, 1))
            leftAf = mpf[:, 0 : 8 - N + 1, 0 : 8 - N + 1]
            rightAf = mpf[:, (N + 1) : (N + 1) + (8 - N + 1), (N + 1) : (N + 1) + (8 - N + 1)]
            hitsAf = hitsAf & (leftAf == 0) & (rightAf == 0)
            segAf = torch.zeros_like(mf)
            for t in range(N):
                segAf[:, t : t + (8 - N + 1), t : t + (8 - N + 1)] = torch.where(
                    hitsAf,
                    torch.ones_like(segAf[:, t : t + (8 - N + 1), t : t + (8 - N + 1)]),
                    segAf[:, t : t + (8 - N + 1), t : t + (8 - N + 1)],
                )
            segA = torch.flip(segAf, dims=[2])
            m = torch.where((seg + segA) > 0, torch.ones_like(mask), torch.zeros_like(mask))
            out.append(m)
        return out

    def _live_rush(self, me, op):
        B = me.shape[0]
        device = me.device
        empty = ((me == 0) & (op == 0)).float()
        x_me = me.unsqueeze(1)
        x_empty = empty.unsqueeze(1)
        live2 = torch.zeros_like(me)
        live3 = torch.zeros_like(me)
        rush3 = torch.zeros_like(me)
        # horizontal live2
        k01 = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.float32, device=device)
        k23 = torch.tensor([[[[0, 0, 1, 1]]]], dtype=torch.float32, device=device)
        s01 = F.conv2d(x_me, (k01 > 0).float())
        s23_me = F.conv2d(x_me, (k23 > 0).float())
        s23_empty = F.conv2d(x_empty, (k23 > 0).float())
        hits_h_1100 = (s01.squeeze(1) == 2) & (s23_me.squeeze(1) == 0) & (s23_empty.squeeze(1) == 2)
        kexp_0110 = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.float32, device=device)
        seg_h_1100 = F.conv_transpose2d(hits_h_1100.unsqueeze(1).float(), kexp_0110, stride=1)
        k12 = torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.float32, device=device)
        k03 = torch.tensor([[[[1, 0, 0, 1]]]], dtype=torch.float32, device=device)
        s12 = F.conv2d(x_me, (k12 > 0).float())
        s03_me = F.conv2d(x_me, (k03 > 0).float())
        s03_empty = F.conv2d(x_empty, (k03 > 0).float())
        hits_h_0110 = (s12.squeeze(1) == 2) & (s03_me.squeeze(1) == 0) & (s03_empty.squeeze(1) == 2)
        kexp_0110_mid = torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.float32, device=device)
        seg_h_0110 = F.conv_transpose2d(hits_h_0110.unsqueeze(1).float(), kexp_0110_mid, stride=1)
        live2 = torch.where((seg_h_1100.squeeze(1) + seg_h_0110.squeeze(1)) > 0, torch.ones_like(live2), live2)
        # vertical live2
        k01_v = torch.tensor([[[[1], [1], [0], [0]]]], dtype=torch.float32, device=device)
        k23_v = torch.tensor([[[[0], [0], [1], [1]]]], dtype=torch.float32, device=device)
        s01_v = F.conv2d(x_me, (k01_v > 0).float())
        s23_me_v = F.conv2d(x_me, (k23_v > 0).float())
        s23_empty_v = F.conv2d(x_empty, (k23_v > 0).float())
        hits_v_1100 = (s01_v.squeeze(1) == 2) & (s23_me_v.squeeze(1) == 0) & (s23_empty_v.squeeze(1) == 2)
        kexp_v_0110 = k01_v
        seg_v_1100 = F.conv_transpose2d(hits_v_1100.unsqueeze(1).float(), kexp_v_0110, stride=1)
        k12_v = torch.tensor([[[[0], [1], [1], [0]]]], dtype=torch.float32, device=device)
        k03_v = torch.tensor([[[[1], [0], [0], [1]]]], dtype=torch.float32, device=device)
        s12_v = F.conv2d(x_me, (k12_v > 0).float())
        s03_me_v = F.conv2d(x_me, (k03_v > 0).float())
        s03_empty_v = F.conv2d(x_empty, (k03_v > 0).float())
        hits_v_0110 = (s12_v.squeeze(1) == 2) & (s03_me_v.squeeze(1) == 0) & (s03_empty_v.squeeze(1) == 2)
        seg_v_0110 = F.conv_transpose2d(hits_v_0110.unsqueeze(1).float(), k12_v, stride=1)
        live2 = torch.where((live2 + seg_v_1100.squeeze(1) + seg_v_0110.squeeze(1)) > 0, torch.ones_like(live2), live2)
        # diagonal live2
        kd4 = torch.zeros((1, 1, 4, 4), device=device); kd4[0, 0, 0, 0] = 1; kd4[0, 0, 1, 1] = 1; kd4[0, 0, 2, 2] = 0; kd4[0, 0, 3, 3] = 0
        kd4b = torch.zeros((1, 1, 4, 4), device=device); kd4b[0, 0, 0, 0] = 0; kd4b[0, 0, 1, 1] = 1; kd4b[0, 0, 2, 2] = 1; kd4b[0, 0, 3, 3] = 0
        s_d_me_0110 = F.conv2d(x_me, (kd4b > 0).float())
        s_d_me_1100 = F.conv2d(x_me, (kd4 > 0).float())
        kd4_tail = torch.zeros((1, 1, 4, 4), device=device); kd4_tail[0, 0, 2, 2] = 1; kd4_tail[0, 0, 3, 3] = 1
        kd4_head_tail = torch.zeros((1, 1, 4, 4), device=device); kd4_head_tail[0, 0, 0, 0] = 1; kd4_head_tail[0, 0, 3, 3] = 1
        s_d_empty_tail = F.conv2d(x_empty, (kd4_tail > 0).float())
        s_d_empty_head_tail = F.conv2d(x_empty, (kd4_head_tail > 0).float())
        hits_d_1100 = (s_d_me_1100.squeeze(1) == 2) & (s_d_empty_tail.squeeze(1) == 2)
        hits_d_0110 = (s_d_me_0110.squeeze(1) == 2) & (s_d_empty_head_tail.squeeze(1) == 2)
        seg_d_1100 = F.conv_transpose2d(hits_d_1100.unsqueeze(1).float(), kd4, stride=1).squeeze(1)
        seg_d_0110 = F.conv_transpose2d(hits_d_0110.unsqueeze(1).float(), kd4b, stride=1).squeeze(1)
        live2 = torch.where((live2 + seg_d_1100 + seg_d_0110) > 0, torch.ones_like(live2), live2)
        mf = torch.flip(me, dims=[2]); xf_me = mf.unsqueeze(1); empty_f = torch.flip(empty, dims=[2]); xf_empty = empty_f.unsqueeze(1)
        s_ad_me_0110 = F.conv2d(xf_me, (kd4b > 0).float())
        s_ad_me_1100 = F.conv2d(xf_me, (kd4 > 0).float())
        s_ad_empty_tail = F.conv2d(xf_empty, (kd4_tail > 0).float())
        s_ad_empty_head_tail = F.conv2d(xf_empty, (kd4_head_tail > 0).float())
        hits_ad_1100 = (s_ad_me_1100.squeeze(1) == 2) & (s_ad_empty_tail.squeeze(1) == 2)
        hits_ad_0110 = (s_ad_me_0110.squeeze(1) == 2) & (s_ad_empty_head_tail.squeeze(1) == 2)
        seg_ad_1100 = torch.flip(F.conv_transpose2d(hits_ad_1100.unsqueeze(1).float(), kd4, stride=1).squeeze(1), dims=[2])
        seg_ad_0110 = torch.flip(F.conv_transpose2d(hits_ad_0110.unsqueeze(1).float(), kd4b, stride=1).squeeze(1), dims=[2])
        live2 = torch.where((live2 + seg_ad_1100 + seg_ad_0110) > 0, torch.ones_like(live2), live2)
        # live3 and rush3 horizontal
        k01340 = torch.tensor([[[[0, 1, 1, 1, 0]]]], dtype=torch.float32, device=device)
        s01340_me = F.conv2d(x_me, (k01340 > 0).float())
        s01_empty = F.conv2d(x_empty, torch.tensor([[[[1, 0, 0, 0, 0]]]] , dtype=torch.float32, device=device))
        s40_empty = F.conv2d(x_empty, torch.tensor([[[[0, 0, 0, 0, 1]]]] , dtype=torch.float32, device=device))
        hits_h_live3 = (s01340_me.squeeze(1) == 3) & (s01_empty.squeeze(1) == 1) & (s40_empty.squeeze(1) == 1)
        kexp_h_live3 = torch.tensor([[[[0, 1, 1, 1, 0]]]], dtype=torch.float32, device=device)
        seg_h_live3 = F.conv_transpose2d(hits_h_live3.unsqueeze(1).float(), kexp_h_live3, stride=1).squeeze(1)
        left_block_h = torch.ones_like(hits_h_live3, dtype=torch.bool)
        right_block_h = left_block_h
        hits_h_rush3a = (F.conv2d(x_me, torch.tensor([[[[1, 1, 1, 0, 0]]]], dtype=torch.float32, device=device)).squeeze(1) == 3) & (F.conv2d(x_empty, torch.tensor([[[[0, 0, 0, 1, 0]]]], dtype=torch.float32, device=device)).squeeze(1) == 1) & left_block_h
        seg_h_rush3a = F.conv_transpose2d(hits_h_rush3a.unsqueeze(1).float(), torch.tensor([[[[1, 1, 1, 0, 0]]]], dtype=torch.float32, device=device), stride=1).squeeze(1)
        hits_h_rush3b = hits_h_live3 & (left_block_h ^ right_block_h)
        seg_h_rush3b = seg_h_live3
        rush3 = torch.where((seg_h_rush3a + seg_h_rush3b) > 0, torch.ones_like(rush3), rush3)
        live3 = torch.where((seg_h_live3) > 0, torch.ones_like(live3), live3)
        return live2, live3, rush3

    def _double_two(self, me, op):
        device = me.device
        empty = ((me == 0) & (op == 0)).float()
        x_me = me.unsqueeze(1)
        x_empty = empty.unsqueeze(1)
        k01 = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.float32, device=device)
        k12 = torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.float32, device=device)
        k23 = torch.tensor([[[[0, 0, 1, 1]]]], dtype=torch.float32, device=device)
        k03 = torch.tensor([[[[1, 0, 0, 1]]]], dtype=torch.float32, device=device)
        s01 = F.conv2d(x_me, (k01 > 0).float())
        s23_me = F.conv2d(x_me, (k23 > 0).float())
        s23_empty = F.conv2d(x_empty, (k23 > 0).float())
        hits_h_1100 = (s01.squeeze(1) == 2) & (s23_me.squeeze(1) == 0) & (s23_empty.squeeze(1) == 2)
        seg_h_1100 = F.conv_transpose2d(hits_h_1100.unsqueeze(1).float(), k01, stride=1).squeeze(1)
        s12 = F.conv2d(x_me, (k12 > 0).float())
        s03_me = F.conv2d(x_me, (k03 > 0).float())
        s03_empty = F.conv2d(x_empty, (k03 > 0).float())
        hits_h_0110 = (s12.squeeze(1) == 2) & (s03_me.squeeze(1) == 0) & (s03_empty.squeeze(1) == 2)
        seg_h_0110 = F.conv_transpose2d(hits_h_0110.unsqueeze(1).float(), k12, stride=1).squeeze(1)
        # vertical
        k01_v = torch.tensor([[[[1], [1], [0], [0]]]], dtype=torch.float32, device=device)
        k12_v = torch.tensor([[[[0], [1], [1], [0]]]], dtype=torch.float32, device=device)
        k23_v = torch.tensor([[[[0], [0], [1], [1]]]], dtype=torch.float32, device=device)
        k03_v = torch.tensor([[[[1], [0], [0], [1]]]], dtype=torch.float32, device=device)
        s01_v = F.conv2d(x_me, (k01_v > 0).float())
        s23_me_v = F.conv2d(x_me, (k23_v > 0).float())
        s23_empty_v = F.conv2d(x_empty, (k23_v > 0).float())
        hits_v_1100 = (s01_v.squeeze(1) == 2) & (s23_me_v.squeeze(1) == 0) & (s23_empty_v.squeeze(1) == 2)
        seg_v_1100 = F.conv_transpose2d(hits_v_1100.unsqueeze(1).float(), k01_v, stride=1).squeeze(1)
        s12_v = F.conv2d(x_me, (k12_v > 0).float())
        s03_me_v = F.conv2d(x_me, (k03_v > 0).float())
        s03_empty_v = F.conv2d(x_empty, (k03_v > 0).float())
        hits_v_0110 = (s12_v.squeeze(1) == 2) & (s03_me_v.squeeze(1) == 0) & (s03_empty_v.squeeze(1) == 2)
        seg_v_0110 = F.conv_transpose2d(hits_v_0110.unsqueeze(1).float(), k12_v, stride=1).squeeze(1)
        # diagonals
        kd4 = torch.zeros((1, 1, 4, 4), device=device); kd4[0, 0, 0, 0] = 1; kd4[0, 0, 1, 1] = 1
        kd4b = torch.zeros((1, 1, 4, 4), device=device); kd4b[0, 0, 1, 1] = 1; kd4b[0, 0, 2, 2] = 1
        tail = torch.zeros((1, 1, 4, 4), device=device); tail[0, 0, 2, 2] = 1; tail[0, 0, 3, 3] = 1
        headtail = torch.zeros((1, 1, 4, 4), device=device); headtail[0, 0, 0, 0] = 1; headtail[0, 0, 3, 3] = 1
        s_d_1100 = F.conv2d(x_me, (kd4 > 0).float()).squeeze(1)
        s_d_tail_me = F.conv2d(x_me, (tail > 0).float()).squeeze(1)
        s_d_tail_empty = F.conv2d(x_empty, (tail > 0).float()).squeeze(1)
        hits_d_1100 = (s_d_1100 == 2) & (s_d_tail_me == 0) & (s_d_tail_empty == 2)
        seg_d_1100 = F.conv_transpose2d(hits_d_1100.unsqueeze(1).float(), kd4, stride=1).squeeze(1)
        s_d_0110 = F.conv2d(x_me, (kd4b > 0).float()).squeeze(1)
        s_d_headtail_me = F.conv2d(x_me, (headtail > 0).float()).squeeze(1)
        s_d_headtail_empty = F.conv2d(x_empty, (headtail > 0).float()).squeeze(1)
        hits_d_0110 = (s_d_0110 == 2) & (s_d_headtail_me == 0) & (s_d_headtail_empty == 2)
        seg_d_0110 = F.conv_transpose2d(hits_d_0110.unsqueeze(1).float(), kd4b, stride=1).squeeze(1)
        mf = torch.flip(me, dims=[2]).unsqueeze(1)
        ef = torch.flip(empty, dims=[2]).unsqueeze(1)
        s_ad_1100 = F.conv2d(mf, (kd4 > 0).float()).squeeze(1)
        s_ad_tail_me = F.conv2d(mf, (tail > 0).float()).squeeze(1)
        s_ad_tail_empty = F.conv2d(ef, (tail > 0).float()).squeeze(1)
        hits_ad_1100 = (s_ad_1100 == 2) & (s_ad_tail_me == 0) & (s_ad_tail_empty == 2)
        seg_ad_1100 = torch.flip(F.conv_transpose2d(hits_ad_1100.unsqueeze(1).float(), kd4, stride=1).squeeze(1), dims=[2])
        s_ad_0110 = F.conv2d(mf, (kd4b > 0).float()).squeeze(1)
        s_ad_headtail_me = F.conv2d(mf, (headtail > 0).float()).squeeze(1)
        s_ad_headtail_empty = F.conv2d(ef, (headtail > 0).float()).squeeze(1)
        hits_ad_0110 = (s_ad_0110 == 2) & (s_ad_headtail_me == 0) & (s_ad_headtail_empty == 2)
        seg_ad_0110 = torch.flip(F.conv_transpose2d(hits_ad_0110.unsqueeze(1).float(), kd4b, stride=1).squeeze(1), dims=[2])
        counts = seg_h_1100 + seg_h_0110 + seg_v_1100 + seg_v_0110 + seg_d_1100 + seg_d_0110 + seg_ad_1100 + seg_ad_0110
        return (counts >= 2).float()

    def _double_three(self, me, op):
        B = me.shape[0]
        device = me.device
        empty_bool = (me == 0) & (op == 0)
        not_empty_bool = ~empty_bool
        empty = empty_bool.float()
        not_empty = not_empty_bool.float()
        x_me = me.unsqueeze(1)
        x_empty = empty.unsqueeze(1)
        x_not_empty = not_empty.unsqueeze(1)
        counts = torch.zeros_like(me)
        # Horizontal live3: [0,1,1,1,0] with both ends empty
        k_live3_h = torch.tensor([[[[0, 1, 1, 1, 0]]]], dtype=torch.float32, device=device)
        hit_me_h = F.conv2d(x_me, (k_live3_h > 0).float()).squeeze(1) == 3
        k_endL_h = torch.tensor([[[[1, 0, 0, 0, 0]]]], dtype=torch.float32, device=device)
        k_endR_h = torch.tensor([[[[0, 0, 0, 0, 1]]]], dtype=torch.float32, device=device)
        endL_empty_h = F.conv2d(x_empty, (k_endL_h > 0).float()).squeeze(1) == 1
        endR_empty_h = F.conv2d(x_empty, (k_endR_h > 0).float()).squeeze(1) == 1
        hits_live3_h = hit_me_h & endL_empty_h & endR_empty_h
        seg_live3_h = F.conv_transpose2d(hits_live3_h.unsqueeze(1).float(), k_live3_h, stride=1).squeeze(1)
        counts += seg_live3_h
        # Vertical live3
        k_live3_v = torch.tensor([[[[0], [1], [1], [1], [0]]]], dtype=torch.float32, device=device)
        hit_me_v = F.conv2d(x_me, (k_live3_v > 0).float()).squeeze(1) == 3
        k_endT_v = torch.tensor([[[[1], [0], [0], [0], [0]]]], dtype=torch.float32, device=device)
        k_endB_v = torch.tensor([[[[0], [0], [0], [0], [1]]]], dtype=torch.float32, device=device)
        endT_empty_v = F.conv2d(x_empty, (k_endT_v > 0).float()).squeeze(1) == 1
        endB_empty_v = F.conv2d(x_empty, (k_endB_v > 0).float()).squeeze(1) == 1
        hits_live3_v = hit_me_v & endT_empty_v & endB_empty_v
        seg_live3_v = F.conv_transpose2d(hits_live3_v.unsqueeze(1).float(), k_live3_v, stride=1).squeeze(1)
        counts += seg_live3_v
        # Diagonal live3
        k_live3_d = torch.zeros((1, 1, 5, 5), device=device); k_live3_d[0, 0, 1, 1] = 1; k_live3_d[0, 0, 2, 2] = 1; k_live3_d[0, 0, 3, 3] = 1
        hit_me_d = F.conv2d(x_me, (k_live3_d > 0).float()).squeeze(1) == 3
        k_end_d = torch.zeros((1, 1, 5, 5), device=device); k_end_d[0, 0, 0, 0] = 1; k_end_d[0, 0, 4, 4] = 1
        end_empty_d = F.conv2d(x_empty, (k_end_d > 0).float()).squeeze(1) == 2
        hits_live3_d = hit_me_d & end_empty_d
        seg_live3_d = F.conv_transpose2d(hits_live3_d.unsqueeze(1).float(), (k_live3_d > 0).float(), stride=1).squeeze(1)
        counts += seg_live3_d
        # Anti-diagonal live3
        k_live3_ad = torch.zeros((1, 1, 5, 5), device=device); k_live3_ad[0, 0, 1, 3] = 1; k_live3_ad[0, 0, 2, 2] = 1; k_live3_ad[0, 0, 3, 1] = 1
        hit_me_ad = F.conv2d(x_me, (k_live3_ad > 0).float()).squeeze(1) == 3
        k_end_ad = torch.zeros((1, 1, 5, 5), device=device); k_end_ad[0, 0, 0, 4] = 1; k_end_ad[0, 0, 4, 0] = 1
        end_empty_ad = F.conv2d(x_empty, (k_end_ad > 0).float()).squeeze(1) == 2
        hits_live3_ad = hit_me_ad & end_empty_ad
        seg_live3_ad = F.conv_transpose2d(hits_live3_ad.unsqueeze(1).float(), (k_live3_ad > 0).float(), stride=1).squeeze(1)
        counts += seg_live3_ad
        # Horizontal rush3: [1,1,1,0,0] with right end empty and left prev blocked OR symmetric
        k_rush3_hL = torch.tensor([[[[1, 1, 1, 0, 0]]]], dtype=torch.float32, device=device)
        hit_rush3_hL = F.conv2d(x_me, (k_rush3_hL > 0).float()).squeeze(1) == 3
        endR_empty_hL = F.conv2d(x_empty, (k_endR_h > 0).float()).squeeze(1) == 1
        not_empty_padL = F.pad(x_not_empty, (1, 0, 0, 0), value=1.0)
        prevL_block = F.conv2d(not_empty_padL, (k_endL_h > 0).float()).squeeze(1) == 1
        prevL_block = torch.ones_like(hit_rush3_hL, dtype=torch.bool)
        prevL_block[:, :, 1:] = not_empty_bool[:, :, 0:3]
        hits_rush3_hL = hit_rush3_hL & endR_empty_hL & prevL_block
        seg_rush3_hL = F.conv_transpose2d(hits_rush3_hL.unsqueeze(1).float(), k_rush3_hL, stride=1).squeeze(1)
        counts += seg_rush3_hL
        k_rush3_hR = torch.tensor([[[[0, 0, 1, 1, 1]]]], dtype=torch.float32, device=device)
        hit_rush3_hR = F.conv2d(x_me, (k_rush3_hR > 0).float()).squeeze(1) == 3
        endL_empty_hR = F.conv2d(x_empty, (k_endL_h > 0).float()).squeeze(1) == 1
        not_empty_padR = F.pad(x_not_empty, (0, 1, 0, 0), value=1.0)
        prevR_block = F.conv2d(not_empty_padR, (k_endR_h > 0).float()).squeeze(1) == 1
        prevR_block = torch.ones_like(hit_rush3_hR, dtype=torch.bool)
        prevR_block[:, :, 0:3] = not_empty_bool[:, :, 5:8]
        hits_rush3_hR = hit_rush3_hR & endL_empty_hR & prevR_block
        seg_rush3_hR = F.conv_transpose2d(hits_rush3_hR.unsqueeze(1).float(), k_rush3_hR, stride=1).squeeze(1)
        counts += seg_rush3_hR
        # Vertical rush3
        k_rush3_vT = torch.tensor([[[[1], [1], [1], [0], [0]]]], dtype=torch.float32, device=device)
        hit_rush3_vT = F.conv2d(x_me, (k_rush3_vT > 0).float()).squeeze(1) == 3
        endB_empty_vT = F.conv2d(x_empty, (k_endB_v > 0).float()).squeeze(1) == 1
        not_empty_padT = F.pad(x_not_empty, (0, 0, 1, 0), value=1.0)
        prevT_block = torch.ones_like(hit_rush3_vT, dtype=torch.bool)
        prevT_block[:, 0, :] = True
        prevT_block[:, 1, :] = not_empty_bool[:, 0, :]
        prevT_block[:, 2, :] = not_empty_bool[:, 1, :]
        prevT_block[:, 3, :] = not_empty_bool[:, 2, :]
        hits_rush3_vT = hit_rush3_vT & endB_empty_vT & prevT_block
        seg_rush3_vT = F.conv_transpose2d(hits_rush3_vT.unsqueeze(1).float(), k_rush3_vT, stride=1).squeeze(1)
        counts += seg_rush3_vT
        k_rush3_vB = torch.tensor([[[[0], [0], [1], [1], [1]]]], dtype=torch.float32, device=device)
        hit_rush3_vB = F.conv2d(x_me, (k_rush3_vB > 0).float()).squeeze(1) == 3
        endT_empty_vB = F.conv2d(x_empty, (k_endT_v > 0).float()).squeeze(1) == 1
        not_empty_padB = F.pad(x_not_empty, (0, 0, 0, 1), value=1.0)
        prevB_block = torch.ones_like(hit_rush3_vB, dtype=torch.bool)
        prevB_block[:, 0, :] = not_empty_bool[:, 5, :]
        prevB_block[:, 1, :] = not_empty_bool[:, 6, :]
        prevB_block[:, 2, :] = not_empty_bool[:, 7, :]
        prevB_block[:, 3, :] = True
        hits_rush3_vB = hit_rush3_vB & endT_empty_vB & prevB_block
        seg_rush3_vB = F.conv_transpose2d(hits_rush3_vB.unsqueeze(1).float(), k_rush3_vB, stride=1).squeeze(1)
        counts += seg_rush3_vB
        # Diagonal rush3 (approximate): use diagonal kernels with end empties and prev blocked via padding
        k_rush3_dL = torch.zeros((1, 1, 5, 5), device=device); k_rush3_dL[0, 0, 0, 0] = 1; k_rush3_dL[0, 0, 1, 1] = 1; k_rush3_dL[0, 0, 2, 2] = 1
        hit_rush3_dL = F.conv2d(x_me, (k_rush3_dL > 0).float()).squeeze(1) == 3
        end_dR_empty = F.conv2d(x_empty, (k_end_d > 0).float()).squeeze(1) >= 1
        prev_dL_block = torch.ones_like(hit_rush3_dL, dtype=torch.bool)
        prev_dL_block[:, 1:, 1:] = not_empty_bool[:, 0:3, 0:3]
        hits_rush3_dL = hit_rush3_dL & end_dR_empty & prev_dL_block
        seg_rush3_dL = F.conv_transpose2d(hits_rush3_dL.unsqueeze(1).float(), (k_rush3_dL > 0).float(), stride=1).squeeze(1)
        counts += seg_rush3_dL
        k_rush3_adL = torch.zeros((1, 1, 5, 5), device=device); k_rush3_adL[0, 0, 0, 4] = 1; k_rush3_adL[0, 0, 1, 3] = 1; k_rush3_adL[0, 0, 2, 2] = 1
        hit_rush3_adL = F.conv2d(x_me, (k_rush3_adL > 0).float()).squeeze(1) == 3
        end_ad_empty = F.conv2d(x_empty, (k_end_ad > 0).float()).squeeze(1) >= 1
        prev_ad_block = torch.ones_like(hit_rush3_adL, dtype=torch.bool)
        prev_ad_block[:, 1:, 0:3] = not_empty_bool[:, 0:3, 1:4]
        hits_rush3_adL = hit_rush3_adL & end_ad_empty & prev_ad_block
        seg_rush3_adL = F.conv_transpose2d(hits_rush3_adL.unsqueeze(1).float(), (k_rush3_adL > 0).float(), stride=1).squeeze(1)
        counts += seg_rush3_adL
        return (counts >= 2).float()

    def mcts_search(self, batch_tracker, side, idx, prior_pi):
        """
        非批处理版本的MCTS（当前未使用，保留向后兼容）
        注意：此方法尚未更新为使用价值网络，建议使用mcts_search_batched
        策略先验的MCTS：UCT选择、叶子随机rollout评估、回传更新N/W/Q
        """
        board = batch_tracker.board[idx].clone().to(self.device)
        turn = int(batch_tracker.player_to_go[idx].item())
        done = bool(batch_tracker.done[idx].item())
        max_steps = int(batch_tracker.max_steps)
        def legal_moves(bd):
            # 合法落子掩码（空位）
            return (bd == 0.0).view(-1)
        def judge(bd):
            # 胜负判定：四方向四连；满盘或未终继续返回0
            x = bd.unsqueeze(0).unsqueeze(0)
            kh = torch.ones((1, 1, 1, 4), device=self.device)
            kv = torch.ones((1, 1, 4, 1), device=self.device)
            kd = torch.zeros((1, 1, 4, 4), device=self.device)
            ka = torch.zeros((1, 1, 4, 4), device=self.device)
            kd[:, :, 0, 0] = 1.0; kd[:, :, 1, 1] = 1.0; kd[:, :, 2, 2] = 1.0; kd[:, :, 3, 3] = 1.0
            ka[:, :, 0, 3] = 1.0; ka[:, :, 1, 2] = 1.0; ka[:, :, 2, 1] = 1.0; ka[:, :, 3, 0] = 1.0
            sh = F.conv2d(x, kh); sv = F.conv2d(x, kv); sd = F.conv2d(x, kd); sa = F.conv2d(x, ka)
            p1_h = (sh == 4.0).flatten(1).any(1)
            p1_v = (sv == 4.0).flatten(1).any(1)
            p1_d = (sd == 4.0).flatten(1).any(1)
            p1_a = (sa == 4.0).flatten(1).any(1)
            win1 = p1_h | p1_v | p1_d | p1_a
            p2_h = (sh == -4.0).flatten(1).any(1)
            p2_v = (sv == -4.0).flatten(1).any(1)
            p2_d = (sd == -4.0).flatten(1).any(1)
            p2_a = (sa == -4.0).flatten(1).any(1)
            win2 = p2_h | p2_v | p2_d | p2_a
            if bool(win1.item()): return 1
            if bool(win2.item()): return -1
            if (~(bd == 0.0)).view(-1).all(): return 0
            return 0
        class Node:
            def __init__(self, bd, ply, device):
                self.bd = bd
                self.ply = ply
                self.device = device
                self.N = torch.zeros(64, dtype=torch.float32, device=device)
                self.W = torch.zeros(64, dtype=torch.float32, device=device)
                self.Q = torch.zeros(64, dtype=torch.float32, device=device)
                self.P = prior_pi.clone()
        root = Node(board, turn, self.device)
        root.P = prior_pi.clone()
        for _ in range(self.mcts_num_simulations):
            path = []
            node = root
            cur_side = side
            depth = 0
            while True:
                mask = legal_moves(node.bd)
                if mask.sum() == 0:
                    if path:
                        v = 0.0
                        for k in range(len(path) - 1, -1, -1):
                            n, act, s = path[k]
                            n.N[act] = n.N[act] + 1.0
                            n.W[act] = n.W[act] + (v if (len(path) - 1 - k) % 2 == 0 else -v)
                            n.Q[act] = n.W[act] / n.N[act]
                    break
                # UCT选择（屏蔽非法）
                uct = node.Q + self.c_puct * node.P * torch.sqrt(torch.sum(node.N)) / (1.0 + node.N)
                uct = torch.where(mask, uct, torch.full_like(uct, -1e9))
                a = int(torch.argmax(uct).item())
                path.append((node, a, cur_side))
                x = a // 8; y = a % 8
                new_bd = node.bd.clone()
                new_bd[x, y] = float(cur_side)
                winner = judge(new_bd)
                if winner != 0 or depth >= self.mcts_max_depth:
                    # 叶子评估：终局直接赋值，否则随机rollout均值
                    v = 0.0
                    if winner == cur_side: v = 1.0
                    elif winner == -cur_side: v = -1.0
                    else:
                        v = 0.0
                        res = 0
                        for _r in range(self.rollout_per_leaf):
                            rb = new_bd.clone()
                            rt = -cur_side
                            rm = 0
                            while rm < self.rollout_max_moves:
                                lm = legal_moves(rb)
                                if lm.sum() == 0: break
                                # 合法随机走子
                                ridx = torch.nonzero(lm, as_tuple=False).squeeze(1)
                                ra = int(ridx[torch.randint(0, ridx.numel(), (1,)).item()].item())
                                rx = ra // 8; ry = ra % 8
                                rb[rx, ry] = float(rt)
                                res = judge(rb)
                                if res != 0: break
                                rt = -rt
                                rm += 1
                            if res == cur_side: v += 1.0
                            elif res == -cur_side: v += -1.0
                            else: v += 0.0
                        v = v / max(1, self.rollout_per_leaf)
                    # 回传：交替视角符号
                    for k in range(len(path) - 1, -1, -1):
                        n, act, s = path[k]
                        n.N[act] = n.N[act] + 1.0
                        n.W[act] = n.W[act] + (v if (len(path) - 1 - k) % 2 == 0 else -v)
                        n.Q[act] = n.W[act] / n.N[act]
                    break
                else:
                    # 扩展：新节点的策略先验（掩码后softmax）
                    feats = self.extract_feature(new_bd.unsqueeze(0), torch.tensor([-cur_side], device=self.device, dtype=torch.float32))
                    with torch.no_grad():
                        logits = self.net(feats).flatten(1)
                    lmask = legal_moves(new_bd).unsqueeze(0)
                    neg_inf = torch.finfo(logits.dtype).min
                    logits = torch.where(lmask, logits, torch.full_like(logits, neg_inf))
                    node = Node(new_bd, -cur_side, self.device)
                    node.P = torch.softmax(logits, dim=-1).squeeze(0)
                    cur_side = -cur_side
                    depth += 1
        return root.N
