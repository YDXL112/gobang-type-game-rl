import torch
import torch.nn.functional as F


class BatchTracker:
    def __init__(self, batch_size: int, device=None, max_steps: int = 40):
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = int(max_steps)
        tpl = torch.tensor([
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1],
        ], dtype=torch.float32, device=self.device)
        self._template = tpl
        self.board = self._template.unsqueeze(0).repeat(self.batch_size, 1, 1).clone()
        self.player_to_go = torch.ones(self.batch_size, dtype=torch.int8, device=self.device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.winners = torch.zeros(self.batch_size, dtype=torch.int8, device=self.device)
        self.step_count = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)

    def reset(self, indices=None):
        if indices is None:
            self.board = self._template.unsqueeze(0).repeat(self.batch_size, 1, 1).clone()
            self.player_to_go.fill_(1)
            self.done.fill_(False)
            self.winners.fill_(0)
            self.step_count.fill_(0)
            return self.board
        idx = torch.tensor(indices, dtype=torch.int64, device=self.device)
        self.board[idx] = self._template.unsqueeze(0).repeat(idx.shape[0], 1, 1)
        self.player_to_go[idx] = 1
        self.done[idx] = False
        self.winners[idx] = 0
        self.step_count[idx] = 0
        return self.board

    def state(self):
        return self.board

    def turn(self):
        return self.player_to_go

    def legal_mask(self):
        return self.board == 0.0

    def illegal_mask(self):
        return self.board != 0.0

    def _judge_all(self):
        b = self.board.unsqueeze(1)
        kh = torch.ones((1, 1, 1, 4), dtype=torch.float32, device=self.device)
        kv = torch.ones((1, 1, 4, 1), dtype=torch.float32, device=self.device)
        kd = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=self.device)
        ka = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=self.device)
        kd[:, :, 0, 0] = 1.0
        kd[:, :, 1, 1] = 1.0
        kd[:, :, 2, 2] = 1.0
        kd[:, :, 3, 3] = 1.0
        ka[:, :, 0, 3] = 1.0
        ka[:, :, 1, 2] = 1.0
        ka[:, :, 2, 1] = 1.0
        ka[:, :, 3, 0] = 1.0
        sh = F.conv2d(b, kh)
        sv = F.conv2d(b, kv)
        sd = F.conv2d(b, kd)
        sa = F.conv2d(b, ka)
        p1_h = (sh == 4.0).flatten(1).any(dim=1)
        p1_v = (sv == 4.0).flatten(1).any(dim=1)
        p1_d = (sd == 4.0).flatten(1).any(dim=1)
        p1_a = (sa == 4.0).flatten(1).any(dim=1)
        win1 = p1_h | p1_v | p1_d | p1_a
        p2_h = (sh == -4.0).flatten(1).any(dim=1)
        p2_v = (sv == -4.0).flatten(1).any(dim=1)
        p2_d = (sd == -4.0).flatten(1).any(dim=1)
        p2_a = (sa == -4.0).flatten(1).any(dim=1)
        win2 = p2_h | p2_v | p2_d | p2_a
        winners = torch.zeros(self.batch_size, dtype=torch.int8, device=self.device)
        winners = torch.where(win1, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
        winners = torch.where(win2 & (~win1), torch.tensor(-1, dtype=torch.int8, device=self.device), winners)
        full = (~self.legal_mask()).view(self.batch_size, -1).all(dim=1)
        reached = self.step_count >= self.max_steps
        draw = (winners == 0) & (full | reached)
        self.winners = winners
        self.done = (winners != 0) | draw
        return self.winners

    def judge(self, indices=None):
        if indices is None:
            return self._judge_all()
        idx = torch.tensor(indices, dtype=torch.int64, device=self.device)
        sub = self.board[idx].unsqueeze(1)
        kh = torch.ones((1, 1, 1, 4), dtype=torch.float32, device=self.device)
        kv = torch.ones((1, 1, 4, 1), dtype=torch.float32, device=self.device)
        kd = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=self.device)
        ka = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=self.device)
        kd[:, :, 0, 0] = 1.0
        kd[:, :, 1, 1] = 1.0
        kd[:, :, 2, 2] = 1.0
        kd[:, :, 3, 3] = 1.0
        ka[:, :, 0, 3] = 1.0
        ka[:, :, 1, 2] = 1.0
        ka[:, :, 2, 1] = 1.0
        ka[:, :, 3, 0] = 1.0
        sh = F.conv2d(sub, kh)
        sv = F.conv2d(sub, kv)
        sd = F.conv2d(sub, kd)
        sa = F.conv2d(sub, ka)
        p1_h = (sh == 4.0).flatten(1).any(dim=1)
        p1_v = (sv == 4.0).flatten(1).any(dim=1)
        p1_d = (sd == 4.0).flatten(1).any(dim=1)
        p1_a = (sa == 4.0).flatten(1).any(dim=1)
        win1 = p1_h | p1_v | p1_d | p1_a
        p2_h = (sh == -4.0).flatten(1).any(dim=1)
        p2_v = (sv == -4.0).flatten(1).any(dim=1)
        p2_d = (sd == -4.0).flatten(1).any(dim=1)
        p2_a = (sa == -4.0).flatten(1).any(dim=1)
        win2 = p2_h | p2_v | p2_d | p2_a
        winners = torch.zeros(idx.shape[0], dtype=torch.int8, device=self.device)
        winners = torch.where(win1, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
        winners = torch.where(win2 & (~win1), torch.tensor(-1, dtype=torch.int8, device=self.device), winners)
        full = (~(self.board[idx] == 0.0)).view(idx.shape[0], -1).all(dim=1)
        reached = self.step_count[idx] >= self.max_steps
        draw = (winners == 0) & (full | reached)
        self.winners[idx] = winners
        self.done[idx] = (winners != 0) | draw
        return winners

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            act = actions.to(self.device)
        else:
            act = torch.tensor(actions, device=self.device)
        if act.ndim == 1:
            x = act // 8
            y = act % 8
        else:
            x = act[:, 0].long()
            y = act[:, 1].long()
        bidx = torch.arange(self.batch_size, device=self.device)
        legal_here = self.legal_mask()[bidx, x, y] & (~self.done)
        illegal_flags = (~legal_here)
        idx_b = bidx[legal_here]
        xs = x[legal_here]
        ys = y[legal_here]
        if idx_b.numel() > 0:
            self.board[idx_b, xs, ys] = self.player_to_go[legal_here].to(torch.float32)
            self.step_count[legal_here] = self.step_count[legal_here] + 1
        self._judge_all()
        self.player_to_go = torch.where(legal_here & (~self.done), -self.player_to_go, self.player_to_go)
        info = {
            "winners": self.winners.clone(),
            "illegal": illegal_flags.clone(),
        }
        return self.board.clone(), self.done.clone(), info

    def get_reward(self):
        rewards = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        rewards[self.done] = self.winners[self.done].to(torch.float32)
        return rewards

    def all_done(self):
        return bool(self.done.all().item())
