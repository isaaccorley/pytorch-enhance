import torch
import torch.nn as nn

from .base import BaseModel


class UpsampleBlock(nn.Module):
    """Base PixelShuffle Upsample Block

    """
    def __init__(
        self,
        n_upsamples: int,
        channels: int,
        kernel_size: int
    ):
        super().__init__()

        layers = []
        for _ in range(n_upsamples):
            layers.extend([
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels * 2**2,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2
                ),
                nn.PixelShuffle(2)
            ])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class ResidualBlock(nn.Module):
    """Base Residual Block

    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        res_scale: float,
        activation
    ):
        super().__init__()

        self.res_scale = res_scale

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
            activation(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.model(x) * self.res_scale
        x = x + shortcut
        return x


class EDSR(BaseModel):
    """Enhanced Deep Residual Networks for Single Image Super-Resolution
    https://arxiv.org/pdf/1707.02921v1.pdf

    Parameters
    ----------
    scale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.

    """
    def __init__(self, scale_factor: int):
        super().__init__()

        self.n_res_blocks = 32

        # Pre Residual Blocks
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(
                channels=256,
                kernel_size=3,
                res_scale=0.1,
                activation=nn.ReLU
            ) for _ in range(self.n_res_blocks)
        ]
        self.res_blocks.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
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
            nn.Conv2d(
                in_channels=256,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
