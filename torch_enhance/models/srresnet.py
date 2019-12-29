import torch.nn as nn

from .base import Base


class SRResnet(nn.Module, Base):
    def __init__(self, scale_factor, n_layers=20):
        super(SRResnet, self).__init__()

        # Initial layer
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]

        # Residual reconstruction
        for i in range(n_layers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())

        # Output reconstruction layer
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.normalize01(x)
        x = self.model(x) + x
        x = self.denormalize01(x)
        return x
