import torch.nn as nn

from .base import Base


class Bicubic(Base):
    """
    Bicubic Interpolation Upsampling module
    """
    def __init__(self, scale_factor):
        super(Bicubic, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        return x