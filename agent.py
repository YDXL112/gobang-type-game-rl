import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import PolicyValueNet
# Agent：单权重自博弈；基于策略先验的MCTS与随机rollout


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
        device=None,
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 策略网络（输入固定18通道特征，输出[1,8,8] logits）
        self.net = PolicyValueNet(in_channels=18, stem_kernel_size=stem_kernel_size, block_kernel_size=block_kernel_size, channels=channels, num_layers=num_layers, activation=activation, bias=bias).to(self.device)
        self.mcts_num_simulations = int(mcts_num_simulations)
        self.mcts_max_depth = int(mcts_max_depth)
        self.c_puct = float(c_puct)
        self.rollout_per_leaf = int(rollout_per_leaf)
        self.rollout_max_moves = int(rollout_max_moves)

    def forward(self, state, side, batch_tracker):
        # 前向：特征→logits→掩码→softmax；用MCTS基于根节点N选动作，返回动作及其策略概率
        state = torch.tensor(state, dtype=torch.float32, device=self.device) if not isinstance(state, torch.Tensor) else state.to(self.device).float()
        if isinstance(side, torch.Tensor):
            side_vec = side.to(self.device).view(-1).float()
        else:
            side_vec = torch.full((state.shape[0],), float(int(side)), device=self.device)
        legal = batch_tracker.legal_mask().to(self.device)
        feats = self.extract_feature(state, side_vec)
        logits = self.net(feats)
        flat_logits = logits.flatten(1)
        mask = legal.view(legal.shape[0], -1)
        neg_inf = torch.finfo(flat_logits.dtype).min
        flat_logits = torch.where(mask, flat_logits, torch.full_like(flat_logits, neg_inf))
        pi = torch.softmax(flat_logits, dim=-1)
        actions = []
        probs = []
        for idx in range(state.shape[0]):
            # 对每盘运行MCTS，根节点依据访问计数N选动作
            counts = self.mcts_search(batch_tracker, int(side_vec[idx].item()), idx, pi[idx])
            if counts.sum() == 0:
                a = int(torch.argmax(pi[idx]).item())
            else:
                a = int(torch.argmax(counts).item())
            actions.append(a)
            probs.append(pi[idx, a])
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        probs = torch.stack(probs).to(self.device)
        return actions, probs

    def extract_feature(self, state, side):
        # 构造18通道特征：我/对掩码、连通度(1/2/3)*2、活2/活3/冲3/双二/双三*2
        b = state.clone()
        side_t = side.view(-1, 1, 1)
        my = (b == side_t).float()
        op = (b == -side_t).float()
        feats = []
        feats.append(my)
        feats.append(op)
        feats += self._connectivity_channels(my, [1, 2, 3])
        feats += self._connectivity_channels(op, [1, 2, 3])
        live2_my, live3_my, rush3_my = self._live_rush(my, op)
        live2_op, live3_op, rush3_op = self._live_rush(op, my)
        feats.append(live2_my)
        feats.append(live3_my)
        feats.append(rush3_my)
        feats.append(self._double_two(my, op))
        feats.append(self._double_three(my, op))
        feats.append(live2_op)
        feats.append(live3_op)
        feats.append(rush3_op)
        feats.append(self._double_two(op, my))
        feats.append(self._double_three(op, my))
        x = torch.stack(feats, dim=1)
        return x

    def _connectivity_channels(self, mask, Ns):
        # 四方向连续段长度恰为N的格置1（水平/垂直/主对角/反对角）
        B = mask.shape[0]
        out = []
        for N in Ns:
            m = torch.zeros_like(mask)
            for i in range(8):
                row = mask[:, i, :]
                for b in range(B):
                    r = row[b]
                    start = 0
                    while start < 8:
                        if r[start] == 1:
                            end = start
                            while end + 1 < 8 and r[end + 1] == 1:
                                end += 1
                            if end - start + 1 == N:
                                m[b, i, start : end + 1] = 1
                            start = end + 1
                        else:
                            start += 1
            for j in range(8):
                col = mask[:, :, j]
                for b in range(B):
                    c = col[b]
                    start = 0
                    while start < 8:
                        if c[start] == 1:
                            end = start
                            while end + 1 < 8 and c[end + 1] == 1:
                                end += 1
                            if end - start + 1 == N:
                                m[b, start : end + 1, j] = 1
                            start = end + 1
                        else:
                            start += 1
            for d in range(-7, 8):
                for b in range(B):
                    cells = []
                    for i in range(8):
                        j = i + d
                        if 0 <= j < 8:
                            cells.append((i, j))
                    k = 0
                    while k < len(cells):
                        i, j = cells[k]
                        if mask[b, i, j] == 1:
                            t = k
                            while t + 1 < len(cells) and mask[b, cells[t + 1][0], cells[t + 1][1]] == 1:
                                t += 1
                            if t - k + 1 == N:
                                for u in range(k, t + 1):
                                    ii, jj = cells[u]
                                    m[b, ii, jj] = 1
                            k = t + 1
                        else:
                            k += 1
            for d in range(-7, 8):
                for b in range(B):
                    cells = []
                    for i in range(8):
                        j = -i + d
                        if 0 <= j < 8:
                            cells.append((i, j))
                    k = 0
                    while k < len(cells):
                        i, j = cells[k]
                        if mask[b, i, j] == 1:
                            t = k
                            while t + 1 < len(cells) and mask[b, cells[t + 1][0], cells[t + 1][1]] == 1:
                                t += 1
                            if t - k + 1 == N:
                                for u in range(k, t + 1):
                                    ii, jj = cells[u]
                                    m[b, ii, jj] = 1
                            k = t + 1
                        else:
                            k += 1
            out.append(m)
        return out

    def _live_rush(self, me, op):
        B = me.shape[0]
        live2 = torch.zeros_like(me)
        live3 = torch.zeros_like(me)
        rush3 = torch.zeros_like(me)
        empty = (me == 0) & (op == 0)
        for i in range(8):
            for s in range(8 - 3):
                w = me[:, i, s : s + 4]
                cond1 = (w[:, 0] == 1) & (w[:, 1] == 1) & (w[:, 2] == 0) & (w[:, 3] == 0) & (empty[:, i, s + 2]) & (empty[:, i, s + 3])
                idx = torch.nonzero(cond1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    live2[idx, i, s : s + 2] = 1
                cond2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 0) & (empty[:, i, s]) & (empty[:, i, s + 3])
                idx = torch.nonzero(cond2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    live2[idx, i, s + 1 : s + 3] = 1
        for j in range(8):
            for s in range(8 - 3):
                w = me[:, s : s + 4, j]
                cond1 = (w[:, 0] == 1) & (w[:, 1] == 1) & (w[:, 2] == 0) & (w[:, 3] == 0) & (empty[:, s + 2, j]) & (empty[:, s + 3, j])
                idx = torch.nonzero(cond1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    live2[idx, s : s + 2, j] = 1
                cond2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 0) & (empty[:, s, j]) & (empty[:, s + 3, j])
                idx = torch.nonzero(cond2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    live2[idx, s + 1 : s + 3, j] = 1
        for d in range(-7, 8):
            cells = [(i, i + d) for i in range(8) if 0 <= i + d < 8]
            L = len(cells)
            for s in range(0, L - 3):
                i0, j0 = cells[s]
                w0 = torch.stack([me[:, cells[s + t][0], cells[s + t][1]] for t in range(4)], dim=1)
                cond1 = (w0[:, 0] == 1) & (w0[:, 1] == 1) & (w0[:, 2] == 0) & (w0[:, 3] == 0) & \
                        (empty[:, cells[s + 2][0], cells[s + 2][1]]) & (empty[:, cells[s + 3][0], cells[s + 3][1]])
                idx = torch.nonzero(cond1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(2):
                        ii, jj = cells[s + t]
                        live2[idx, ii, jj] = 1
                cond2 = (w0[:, 0] == 0) & (w0[:, 1] == 1) & (w0[:, 2] == 1) & (w0[:, 3] == 0) & \
                        (empty[:, cells[s][0], cells[s][1]]) & (empty[:, cells[s + 3][0], cells[s + 3][1]])
                idx = torch.nonzero(cond2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 3):
                        ii, jj = cells[s + t]
                        live2[idx, ii, jj] = 1
        for d in range(-7, 8):
            cells = [(i, -i + d) for i in range(8) if 0 <= -i + d < 8]
            L = len(cells)
            for s in range(0, L - 3):
                w0 = torch.stack([me[:, cells[s + t][0], cells[s + t][1]] for t in range(4)], dim=1)
                cond1 = (w0[:, 0] == 1) & (w0[:, 1] == 1) & (w0[:, 2] == 0) & (w0[:, 3] == 0) & \
                        (empty[:, cells[s + 2][0], cells[s + 2][1]]) & (empty[:, cells[s + 3][0], cells[s + 3][1]])
                idx = torch.nonzero(cond1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(2):
                        ii, jj = cells[s + t]
                        live2[idx, ii, jj] = 1
                cond2 = (w0[:, 0] == 0) & (w0[:, 1] == 1) & (w0[:, 2] == 1) & (w0[:, 3] == 0) & \
                        (empty[:, cells[s][0], cells[s][1]]) & (empty[:, cells[s + 3][0], cells[s + 3][1]])
                idx = torch.nonzero(cond2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 3):
                        ii, jj = cells[s + t]
                        live2[idx, ii, jj] = 1
        for i in range(8):
            for s in range(8 - 4):
                w = me[:, i, s : s + 5]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                       (empty[:, i, s]) & (empty[:, i, s + 4])
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    live3[idx, i, s + 1 : s + 4] = 1
                if s == 0:
                    left_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    left_block = (op[:, i, s - 1] == 1) | (~empty[:, i, s - 1])
                if s + 5 >= 8:
                    right_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    right_block = (op[:, i, s + 5] == 1) | (~empty[:, i, s + 5])
                cond_r1 = (w[:, 0] == 1) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 0) & (w[:, 4] == 0) & \
                          (empty[:, i, s + 3]) & left_block
                idx = torch.nonzero(cond_r1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    rush3[idx, i, s : s + 3] = 1
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                          (left_block ^ right_block)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    rush3[idx, i, s + 1 : s + 4] = 1
        for j in range(8):
            for s in range(8 - 4):
                w = me[:, s : s + 5, j]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                       (empty[:, s, j]) & (empty[:, s + 4, j])
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    live3[idx, s + 1 : s + 4, j] = 1
                if s == 0:
                    left_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    left_block = (op[:, s - 1, j] == 1) | (~empty[:, s - 1, j])
                if s + 5 >= 8:
                    right_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    right_block = (op[:, s + 5, j] == 1) | (~empty[:, s + 5, j])
                cond_r1 = (w[:, 0] == 1) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 0) & (w[:, 4] == 0) & \
                          (empty[:, s + 3, j]) & left_block
                idx = torch.nonzero(cond_r1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    rush3[idx, s : s + 3, j] = 1
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                          (left_block ^ right_block)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    rush3[idx, s + 1 : s + 4, j] = 1
        for d in range(-7, 8):
            cells = [(i, i + d) for i in range(8) if 0 <= i + d < 8]
            L = len(cells)
            for s in range(0, L - 4):
                w = torch.stack([me[:, cells[s + t][0], cells[s + t][1]] for t in range(5)], dim=1)
                e0 = empty[:, cells[s][0], cells[s][1]]
                e4 = empty[:, cells[s + 4][0], cells[s + 4][1]]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & e0 & e4
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        live3[idx, ii, jj] = 1
                lb = (s == 0) | ((op[:, cells[s - 1][0], cells[s - 1][1]] == 1) | (~empty[:, cells[s - 1][0], cells[s - 1][1]]))
                if s + 5 >= L:
                    rb = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    rb = (op[:, cells[s + 5][0], cells[s + 5][1]] == 1) | (~empty[:, cells[s + 5][0], cells[s + 5][1]])
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & (lb ^ rb)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        rush3[idx, ii, jj] = 1
        for d in range(-7, 8):
            cells = [(i, -i + d) for i in range(8) if 0 <= -i + d < 8]
            L = len(cells)
            for s in range(0, L - 4):
                w = torch.stack([me[:, cells[s + t][0], cells[s + t][1]] for t in range(5)], dim=1)
                e0 = empty[:, cells[s][0], cells[s][1]]
                e4 = empty[:, cells[s + 4][0], cells[s + 4][1]]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & e0 & e4
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        live3[idx, ii, jj] = 1
                lb = (s == 0) | ((op[:, cells[s - 1][0], cells[s - 1][1]] == 1) | (~empty[:, cells[s - 1][0], cells[s - 1][1]]))
                if s + 5 >= L:
                    rb = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    rb = (op[:, cells[s + 5][0], cells[s + 5][1]] == 1) | (~empty[:, cells[s + 5][0], cells[s + 5][1]])
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & (lb ^ rb)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        rush3[idx, ii, jj] = 1
        return live2, live3, rush3

    def _double_two(self, me, op):
        B = me.shape[0]
        empty = (me == 0) & (op == 0)
        cnt = torch.zeros_like(me)
        def mark_line(seq, coords, length):
            n = len(seq)
            for i in range(n - length + 1):
                yield i, seq[i : i + length], coords[i : i + length]
        for b in range(B):
            for i in range(8):
                seq_me = me[b, i, :].cpu().numpy().tolist()
                seq_empty = empty[b, i, :].cpu().numpy().tolist()
                coords = [(i, j) for j in range(8)]
                for k, window, wcoords in mark_line(seq_me, coords, 4):
                    if window == [1, 1, 0, 0] and seq_empty[k + 2] == 1 and seq_empty[k + 3] == 1:
                        for u in range(0, 2):
                            ii, jj = wcoords[u]
                            cnt[b, ii, jj] += 1
                    if window == [0, 1, 1, 0] and seq_empty[k] == 1 and seq_empty[k + 3] == 1:
                        for u in range(1, 3):
                            ii, jj = wcoords[u]
                            cnt[b, ii, jj] += 1
            for j in range(8):
                seq_me = me[b, :, j].cpu().numpy().tolist()
                seq_empty = empty[b, :, j].cpu().numpy().tolist()
                coords = [(i, j) for i in range(8)]
                for k, window, wcoords in mark_line(seq_me, coords, 4):
                    if window == [1, 1, 0, 0] and seq_empty[k + 2] == 1 and seq_empty[k + 3] == 1:
                        for u in range(0, 2):
                            ii, jj = wcoords[u]
                            cnt[b, ii, jj] += 1
                    if window == [0, 1, 1, 0] and seq_empty[k] == 1 and seq_empty[k + 3] == 1:
                        for u in range(1, 3):
                            ii, jj = wcoords[u]
                            cnt[b, ii, jj] += 1
            for d in range(-7, 8):
                cells = []
                for i in range(8):
                    j = i + d
                    if 0 <= j < 8:
                        cells.append((i, j))
                seq_me = [me[b, i, j].item() for i, j in cells]
                seq_empty = [empty[b, i, j].item() for i, j in cells]
                for k in range(0, len(cells) - 3):
                    window = seq_me[k : k + 4]
                    if window == [1, 1, 0, 0] and k + 3 < len(cells) and seq_empty[k + 2] == 1 and seq_empty[k + 3] == 1:
                        for u in range(0, 2):
                            ii, jj = cells[k + u]
                            cnt[b, ii, jj] += 1
                    if window == [0, 1, 1, 0] and seq_empty[k] == 1 and seq_empty[k + 3] == 1:
                        for u in range(1, 3):
                            ii, jj = cells[k + u]
                            cnt[b, ii, jj] += 1
            for d in range(-7, 8):
                cells = []
                for i in range(8):
                    j = -i + d
                    if 0 <= j < 8:
                        cells.append((i, j))
                seq_me = [me[b, i, j].item() for i, j in cells]
                seq_empty = [empty[b, i, j].item() for i, j in cells]
                for k in range(0, len(cells) - 3):
                    window = seq_me[k : k + 4]
                    if window == [1, 1, 0, 0] and k + 3 < len(cells) and seq_empty[k + 2] == 1 and seq_empty[k + 3] == 1:
                        for u in range(0, 2):
                            ii, jj = cells[k + u]
                            cnt[b, ii, jj] += 1
                    if window == [0, 1, 1, 0] and seq_empty[k] == 1 and seq_empty[k + 3] == 1:
                        for u in range(1, 3):
                            ii, jj = cells[k + u]
                            cnt[b, ii, jj] += 1
        return (cnt >= 2).float()

    def _double_three(self, me, op):
        B = me.shape[0]
        empty = (me == 0) & (op == 0)
        cnt = torch.zeros_like(me)
        for i in range(8):
            for s in range(8 - 4):
                w = me[:, i, s : s + 5]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                       (empty[:, i, s]) & (empty[:, i, s + 4])
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    cnt[idx, i, s + 1 : s + 4] += 1
                if s == 0:
                    left_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    left_block = (op[:, i, s - 1] == 1) | (~empty[:, i, s - 1])
                if s + 5 >= 8:
                    right_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    right_block = (op[:, i, s + 5] == 1) | (~empty[:, i, s + 5])
                cond_r1 = (w[:, 0] == 1) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 0) & (w[:, 4] == 0) & \
                          (empty[:, i, s + 3]) & left_block
                idx = torch.nonzero(cond_r1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    cnt[idx, i, s : s + 3] += 1
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                          (left_block ^ right_block)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    cnt[idx, i, s + 1 : s + 4] += 1
        for j in range(8):
            for s in range(8 - 4):
                w = me[:, s : s + 5, j]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                       (empty[:, s, j]) & (empty[:, s + 4, j])
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    cnt[idx, s + 1 : s + 4, j] += 1
                if s == 0:
                    left_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    left_block = (op[:, s - 1, j] == 1) | (~empty[:, s - 1, j])
                if s + 5 >= 8:
                    right_block = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    right_block = (op[:, s + 5, j] == 1) | (~empty[:, s + 5, j])
                cond_r1 = (w[:, 0] == 1) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 0) & (w[:, 4] == 0) & \
                          (empty[:, s + 3, j]) & left_block
                idx = torch.nonzero(cond_r1, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    cnt[idx, s : s + 3, j] += 1
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & \
                          (left_block ^ right_block)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    cnt[idx, s + 1 : s + 4, j] += 1
        for d in range(-7, 8):
            cells = [(i, i + d) for i in range(8) if 0 <= i + d < 8]
            L = len(cells)
            for s in range(0, L - 4):
                w = torch.stack([me[:, cells[s + t][0], cells[s + t][1]] for t in range(5)], dim=1)
                e0 = empty[:, cells[s][0], cells[s][1]]
                e4 = empty[:, cells[s + 4][0], cells[s + 4][1]]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & e0 & e4
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        cnt[idx, ii, jj] += 1
                lb = (s == 0) | ((op[:, cells[s - 1][0], cells[s - 1][1]] == 1) | (~empty[:, cells[s - 1][0], cells[s - 1][1]]))
                if s + 5 >= L:
                    rb = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    rb = (op[:, cells[s + 5][0], cells[s + 5][1]] == 1) | (~empty[:, cells[s + 5][0], cells[s + 5][1]])
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & (lb ^ rb)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        cnt[idx, ii, jj] += 1
        for d in range(-7, 8):
            cells = [(i, -i + d) for i in range(8) if 0 <= -i + d < 8]
            L = len(cells)
            for s in range(0, L - 4):
                w = torch.stack([me[:, cells[s + t][0], cells[s + t][1]] for t in range(5)], dim=1)
                e0 = empty[:, cells[s][0], cells[s][1]]
                e4 = empty[:, cells[s + 4][0], cells[s + 4][1]]
                cond = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & e0 & e4
                idx = torch.nonzero(cond, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        cnt[idx, ii, jj] += 1
                lb = (s == 0) | ((op[:, cells[s - 1][0], cells[s - 1][1]] == 1) | (~empty[:, cells[s - 1][0], cells[s - 1][1]]))
                if s + 5 >= L:
                    rb = torch.ones(B, dtype=torch.bool, device=me.device)
                else:
                    rb = (op[:, cells[s + 5][0], cells[s + 5][1]] == 1) | (~empty[:, cells[s + 5][0], cells[s + 5][1]])
                cond_r2 = (w[:, 0] == 0) & (w[:, 1] == 1) & (w[:, 2] == 1) & (w[:, 3] == 1) & (w[:, 4] == 0) & (lb ^ rb)
                idx = torch.nonzero(cond_r2, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for t in range(1, 4):
                        ii, jj = cells[s + t]
                        cnt[idx, ii, jj] += 1
        return (cnt >= 2).float()

    def mcts_search(self, batch_tracker, side, idx, prior_pi):
        # 策略先验的MCTS：UCT选择、叶子随机rollout评估、回传更新N/W/Q
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
