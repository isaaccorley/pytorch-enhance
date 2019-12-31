import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base


class ResidualBlock(nn.Module):
    """
    Base Residual Block
    """
    def __init__(
        self,
        channels,
        kernel_size,
        activation
    ):

        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(num_features=channels)
        )

    def forward(self, x):
        x = self.model(x) + x
        return x


class UpsampleBlock(nn.Module):
    """
    Base PixelShuffle Upsample Block
    """
    def __init__(
        self,
        n_upsamples,
        channels,
        kernel_size,
        activation
    ):

        super(UpsampleBlock, self).__init__()

        layers = []
        for _ in range(n_upsamples):
            layers.extend([
            nn.Conv2d(in_channels=channels, out_channels=channels * 2**2, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.PixelShuffle(2),
            activation()
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class SRResNet(nn.Module, Base):
    """
    Super-Resolution Residual Neural Network
    https://arxiv.org/pdf/1609.04802v5.pdf
    """
    def __init__(self, scale_factor, n_res_blocks=16):

        super(SRResNet, self).__init__()

        self.loss = nn.MSELoss()

        # Pre Residual Blocks
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(channels=64, kernel_size=3, activation=nn.PReLU) \
            for _ in range(n_res_blocks)
            ]
        self.res_blocks = nn.Sequential(*self.res_blocks)

        # Post Residual Blocks
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64)
        )    

        # Upsamples
        n_upsamples = 1 if scale_factor ==2 else 2
        self.upsample = UpsampleBlock(
            n_upsamples=n_upsamples,
            channels=64,
            kernel_size=3,
            activation=nn.PReLU
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.normalize01(x)
        x = self.head(x)
        shortcut = x
        x = self.res_blocks(x)
        x = self.tail(x) + shortcut
        x = self.upsample(x)
        x = self.output(x)
        x = self.denormalize11(x)
        return x