import os
from PIL import Image
from typing import List, Tuple

import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


BSDS300_URL = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
BSDS500_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
SET5_URL = "https://raw.github.com/IsaacCorley/pytorch-enhance/master/datasets/Set5.zip"
SET14_URL = "https://raw.github.com/IsaacCorley/pytorch-enhance/master/datasets/Set14.zip"
T91_URL = "https://raw.github.com/IsaacCorley/pytorch-enhance/master/datasets/T91.zip"
HISTORICAL_URL = "https://raw.github.com/IsaacCorley/pytorch-enhance/master/datasets/historical.zip"
DIV2K_TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_TEST_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


class SRDataset(torch.utils.data.Dataset):
    """Base Super Resolution Dataset Class

    """

    def __init__(self):
        super(SRDataset, self).__init__()

        self.base_dir = '.data'
        self.color_space = 'RGB'
        self.extensions = ['']
        self.lr_transform = None
        self.hr_transform = None

    def get_lr_transforms(self):
        """Returns HR to LR image transformations
        
        """
        return Compose(
            [
                Resize((self.image_size//self.scale_factor,
                        self.image_size//self.scale_factor),
                        Image.BICUBIC),
                ToTensor(),
            ]
        )

    def get_hr_transforms(self):
        """Returns HR image transformations
        
        """
        return Compose(
            [
                Resize((self.image_size, self.image_size), Image.BICUBIC),
                ToTensor(),
            ]
        )
        
    def get_files(self, root_dir: str) -> List[str]:
        """
        Returns  a list of valid image files in a directory

        Parameters
        ----------
        root_dir : str
            Path to directory of images.

        Returns
        -------
        List[str]
            List of valid images in `root_dir` directory.

        """

        return [
            os.path.join(root_dir, x)
            for x in os.listdir(root_dir)
            if self.is_valid_file(x)
        ]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns  a tuple of and lr and hr torch tensors

        Parameters
        ----------
        idx : int
            Index value to index the list of images

        Returns
        -------
        lr: torch.Tensor
            Low Resolution transformed indexed image.
        hr: torch.Tensor
            High Resolution transformed indexed image.

        """

        lr = self.load_img(self.file_names[idx])
        hr = lr.copy()
        if self.lr_transform:
            lr = self.lr_transform(lr)
        if self.hr_transform:
            hr = self.hr_transform(hr)

        return lr, hr

    def __len__(self) -> int:
        """ Return number of images in dataset

        Returns
        -------
        int
            Number of images in dataset file_names list

        """

        return len(self.file_names)

    def is_valid_file(self, file_path: str) -> bool:
        """
        Returns boolean if the given `file_path` has a valid image extension 

        Parameters
        ----------
        file_path : str
            Path to image file

        Returns
        -------
        bool
            True if `file_path` has a valid image extension otherwise False

        """

        return any(file_path.endswith(ext) for ext in self.extensions)

    def load_img(self, file_path) -> Image.Image:
        """
        Returns a PIL Image of the image located at `file_path`

        Parameters
        ----------
        file_path : str
            Path to image file to be loaded

        Returns
        -------
        PIL.Image.Image
            Loaded image as PIL Image

        """

        return Image.open(file_path).convert(self.color_space)
