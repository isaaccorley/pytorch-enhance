import torch.nn as nn

from .base import Base
from .baseline import Bicubic


class SRCNN(nn.Module, Base):
    def __init__(self, scale_factor):
        super(SRCNN, self).__init__()

        self.upsample = Bicubic(scale_factor)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2,),
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.upsample(x)
        x = self.normalize01(x)
        x = self.model(x)
        x = self.denormalize01(x)
        return x
