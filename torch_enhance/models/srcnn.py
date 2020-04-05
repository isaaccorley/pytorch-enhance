import torch.nn as nn

from .base import Base
from .baseline import Bicubic

WEIGHTS_URL = ""
WEIGHTS_PATH = ""

class SRCNN(Base):
    """Super-Resolution Convolutional Neural Network
        https://arxiv.org/pdf/1501.00092v3.pdf
        
    """
    def __init__(self, scale_factor, pretrained=False):
        super(SRCNN, self).__init__()

        self.upsample = Bicubic(scale_factor)
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2)
        )

        if pretrained:
            self.load_pretrained(WEIGHTS_URL, WEIGHTS_PATH)

    def forward(self, x):
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
        x = self.upsample(x)
        x = self.model(x)
        return x
