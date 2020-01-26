import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Base

WEIGHTS_URL = ""
WEIGHTS_PATH = ""


class UpsampleBlock(nn.Module):
    """
    Base PixelShuffle Upsample Block
    """
    def __init__(
        self,
        n_upsamples,
        channels,
        kernel_size
    ):

        super(UpsampleBlock, self).__init__()

        layers = []
        for _ in range(n_upsamples):
            layers.extend([
            nn.Conv2d(in_channels=channels, out_channels=channels * 2**2, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.PixelShuffle(2)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualBlock(nn.Module):
    """
    Base Residual Block
    """
    def __init__(
        self,
        channels,
        kernel_size,
        activation,
        res_scale
    ):

        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        )

    def forward(self, x):
        shortcut = x
        x = self.model(x) * self.res_scale
        x = x + shortcut
        return x


class EDSR(Base):
    def __init__(self, scale_factor, pretrained=False)
        super(EDSR, self).__init__()

        self.n_res_blocks = 32

        # Pre Residual Blocks
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(channels=256, kernel_size=3, activation=nn.ReLU, res_scale=0.1) \
            for _ in range(self.n_res_blocks)
            ]
        self.res_blocks.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        # Upsamples
        n_upsamples = 1 if scale_factor == 2 else 2
        self.upsample = UpsampleBlock(
            n_upsamples=n_upsamples,
            channels=256,
            kernel_size=3
        )

        # Output layer
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        
        if pretrained:
            self.load_pretrained(WEIGHTS_URL, WEIGHTS_PATH)

    def forward(self, x):
        x = self.head(x)
        shortcut = x
        x = self.res_blocks(x) + shortcut
        x = self.upsample(x)
        x = self.tail(x)
        return x


