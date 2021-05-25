import os
import shutil

import torch
import torch.nn as nn
from torchvision.datasets.utils import download_and_extract_archive


MODELS_PATH = ".models"


class BaseModel(nn.Module):
    """Base Super-Resolution module"""

    def load_pretrained(self, weights_url: str, weights_path: str) -> None:
        """Download pretrained weights and load as state dict

        Parameters
        ----------
        weights_url : str
            Base URL to pretrained weights.
        weights_path : str
            Path to save pretrained weights.

        Returns
        -------
        None

        """
        base_file = os.path.basename(weights_path)

        if not os.path.exists(os.path.join(MODELS_PATH, base_file)):
            self.download(weights_url, weights_path)

        self.load_state_dict(torch.load(os.path.join(MODELS_PATH, base_file)))

    @staticmethod
    def download(url: str, weights_path: str) -> None:
        """Download pretrained weights

        Parameters
        ----------
        weights_path : str
            Path to save pretrained weights.

        Returns
        -------
        None

        """
        base_file = os.path.basename(weights_path)

        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)

        download_and_extract_archive(url, MODELS_PATH, remove_finished=True)
        shutil.copyfile(weights_path, os.path.join(MODELS_PATH, base_file))
        shutil.rmtree(os.path.dirname(weights_path))

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

        x = self.forward(x)
        x *= 255.0
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.squeeze(0)
        return x
