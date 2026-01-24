import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        stem_kernel_size: int = 5,
        block_kernel_size: int = 3,
        channels: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        bias: bool = True,
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
        self.head = nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)
        logits = self.head(x)
        return logits
