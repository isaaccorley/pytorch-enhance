import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


class VGG(nn.Module):
    """VGG/Perceptual Loss

    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """

    def __init__(self, conv_index: str = "22"):

        super().__init__()
        vgg_features = torchvision.models.vgg16(pretrained=True).features
        modules = [m for m in vgg_features]

        if conv_index == "22":
            vgg = nn.Sequential(*modules[:8])
        elif conv_index == "54":
            vgg = nn.Sequential(*modules[:35])

        vgg.requires_grad = False
        vgg.eval()

        self.vgg = vgg
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        sr = (sr - self.vgg_mean) / self.vgg_std
        hr = (hr - self.vgg_mean) / self.vgg_std
        vgg_sr = self.vgg(sr)

        with torch.no_grad():
            vgg_hr = self.vgg(hr)

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
