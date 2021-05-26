from typing import List

import torch
import torch.nn as nn

from .base import BaseModel
from .baseline import Bicubic


class DRCN(BaseModel):
    """Deeply-Recursive Convolutional Neural Network (DRCN)
    https://arxiv.org/abs/1511.04491

    Parameters
    ----------
    scale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.
    channels: int
        Number of input and output channels
    """

    def __init__(self, scale_factor: int, channels: int = 3, recursions: int = 16):
        super().__init__()

        self.recursions = recursions
        filters = 256
        kernel_size = 3
        padding=1

        self.upsample = Bicubic(scale_factor)
        self.embedding_network = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.inference_network = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.reconstruction_network = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Super-resolve Low-Resolution input tensor

        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        List[torch.Tensor]
            Super-Resolved image as tensor

        """
        x = self.upsample(x)
        skip = x
        x = self.embedding_network(x)

        outputs = []
        for _ in range(self.recursions):
            x = self.inference_network(x) + x
            outputs.append(self.reconstruction_network(x))

        x = torch.stack(outputs, dim=1)
        x = torch.sum(x, dim=1)
        outputs.append(x)

        return outputs

    @torch.no_grad()
    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve x and cast as image

        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor

        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = self.forward(x)[-1]
        x *= 255.0
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.squeeze(0)
        return x