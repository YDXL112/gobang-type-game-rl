import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """
    策略-价值网络：同时输出策略logits和价值估计
    - 策略头：输出每个位置的logits [B, 1, 8, 8]
    - 价值头：输出局面价值标量 [B, 1]，通过tanh限制在[-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 16,
        stem_kernel_size: int = 5,
        block_kernel_size: int = 3,
        channels: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        bias: bool = True,
        value_hidden_channels: int = 32,  # 价值头隐藏层通道数
    ):
        super().__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, stem_kernel_size, padding=stem_kernel_size // 2, bias=bias),
            act,
        )
        blocks = []
        for _ in range(num_layers):
            blocks.append(nn.Conv2d(channels, channels, block_kernel_size, padding=block_kernel_size // 2, bias=bias))
            blocks.append(act)
        self.tower = nn.Sequential(*blocks) if blocks else nn.Identity()

        # 策略头：输出每个位置的logits
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=bias)

        # 价值头：评估当前局面优劣
        # 先用1x1 conv降维，然后全局池化，最后通过全连接层输出标量
        self.value_conv = nn.Conv2d(channels, value_hidden_channels, kernel_size=1, bias=bias)
        self.value_act = act
        self.value_fc = nn.Linear(value_hidden_channels * 8 * 8, 1)

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, 8, 8]
        Returns:
            policy_logits: 策略logits [B, 1, 8, 8]
            value: 局面价值 [B, 1]，范围[-1, 1]，正数表示当前玩家优势
        """
        x = self.stem(x)
        x = self.tower(x)

        # 策略输出
        policy_logits = self.policy_head(x)  # [B, 1, 8, 8]

        # 价值输出
        v = self.value_conv(x)  # [B, value_hidden_channels, 8, 8]
        v = self.value_act(v)
        v = v.view(v.size(0), -1)  # [B, value_hidden_channels * 64]
        v = self.value_fc(v)  # [B, 1]
        value = torch.tanh(v)  # 限制在[-1, 1]

        return policy_logits, value
