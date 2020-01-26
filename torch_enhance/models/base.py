import os
import shutil
import torch
import torch.nn as nn
import torchvision


MODELS_PATH = '.models'

class Base(nn.Module):
    """
    Base SR module containing common methods
    """

    def __init__(self):
        self.loss = nn.MSELoss()

    def load_pretrained(self, weights_url, weights_path):
        """ Download pretrained weights and load state dict """
        base_file = os.path.basename(weights_path)

        if not os.path.exists(os.path.join(MODELS_PATH, base_file)):
            self.download(weights_url, weights_path)

        self.load_state_dict(torch.load(os.path.join(MODELS_PATH, base_file)))

    @staticmethod
    def download(url, weights_path):
        """ Download pretrained weights """
        base_file = os.path.basename(weights_path)

        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)

        download_and_extract_archive(url, MODELS_PATH, remove_finished=True)
        shutil.copyfile(weights_path, os.path.join(MODELS_PATH, base_file))
        shutil.rmtree(os.path.dirname(weights_path))

    def normalize01(self, x):
        """ Normalize from [0, 255] -> [0, 1] """
        return x / 255

    def denormalize01(self, x):
        """ Normalize from [0, 1] -> [0, 255] """
        return x * 255

    def normalize11(self, x):
        """ Normalize from [0, 255] -> [-1, 1] """
        return x / 127.5 - 1

    def denormalize11(self, x):
        """ Normalize from [-1, 1] -> [0, 255] """
        return (x + 1) * 127.5

    @torch.no_grad()
    def enhance(self, x):
        """ Super-resolve x and cast as image """
        x = self.forward(x)
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.squeeze()
        return x