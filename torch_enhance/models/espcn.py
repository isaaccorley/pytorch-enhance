
import torch.nn as nn

from .base import Base


class ESPCN(nn.Module, Base):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.normalize01(x)
        x = self.model(x)
        x = self.denormalize01(x)
        return x
