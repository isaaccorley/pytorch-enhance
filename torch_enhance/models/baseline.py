import torch
import torch.nn as nn

from .base import BaseModel


class Bicubic(BaseModel):
    """Bicubic Interpolation Upsampling module

    Parameters
    ----------
    scale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.

    """

    def __init__(self, scale_factor: int, channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bicubic", align_corners=False
            )
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
