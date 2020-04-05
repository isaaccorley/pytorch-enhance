import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base

WEIGHTS_URL = ""
WEIGHTS_PATH = ""


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


class SRResNet(Base):
    """Super-Resolution Residual Neural Network
        https://arxiv.org/pdf/1609.04802v5.pdf
        
    """
    def __init__(self, scale_factor, pretrained=False):

        super(SRResNet, self).__init__()

        self.n_res_blocks = 16
        
        # Pre Residual Blocks
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(channels=64, kernel_size=3, activation=nn.PReLU) \
            for _ in range(self.n_res_blocks)
            ]
        self.res_blocks = nn.Sequential(*self.res_blocks)
        self.res_blocks.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.res_blocks.append(nn.BatchNorm2d(num_features=64))

        # Upsamples
        n_upsamples = 1 if scale_factor == 2 else 2
        self.upsample = UpsampleBlock(
            n_upsamples=n_upsamples,
            channels=64,
            kernel_size=3,
            activation=nn.PReLU
        )

        # Output layer
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

        if pretrained:
            self.load_pretrained(WEIGHTS_URL, WEIGHTS_PATH)

    def forward(self, x):
        """Super-resolve Low-Resolution input tensor

        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor

        """
        x = self.head(x)
        shortcut = x
        x = self.res_blocks(x) + shortcut
        x = self.upsample(x)
        x = self.tail(x)
        return x