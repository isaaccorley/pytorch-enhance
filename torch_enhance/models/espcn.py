import torch
import torch.nn as nn

from .base import BaseModel


class ESPCN(BaseModel):
    """Efficient Sub-Pixel Convolutional Neural Network
    https://arxiv.org/pdf/1609.05158v2.pdf

    Parameters
    ----------
    scale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.
    pretrained : bool
        If True download and load pretrained weights

    """
    def __init__(self, scale_factor: int, channels: int = 3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=channels * scale_factor**2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(scale_factor),
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
        return self.model(x)
