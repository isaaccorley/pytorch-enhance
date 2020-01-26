import torch.nn as nn

from .base import Base
from .baseline import Bicubic

WEIGHTS_URL = ""
WEIGHTS_PATH = ""


class VDSR(Base):
    """
    Very Deep Super Resolution
    https://arxiv.org/pdf/1511.04587.pdf
    """
    def __init__(self, scale_factor, pretrained=False):
        super(VDSR, self).__init__()

        self.n_layers = 20
        self.upsample = Bicubic(scale_factor)

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

        if pretrained:
            self.load_pretrained(WEIGHTS_URL, WEIGHTS_PATH)

    def forward(self, x):
        x = self.upsample(x)
        x = self.model(x) + x
        return x
