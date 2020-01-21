import torch.nn as nn

from .base import Base
from .baseline import Bicubic


class SRCNN(Base):
    """
    Super-Resolution Convolutional Neural Network
    https://arxiv.org/pdf/1501.00092v3.pdf
    """
    def __init__(self, scale_factor):
        super(SRCNN, self).__init__()

        self.loss = nn.MSELoss()

        self.upsample = Bicubic(scale_factor)
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.model(x)
        return x
