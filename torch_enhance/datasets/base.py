import os
from typing import List, Tuple

import torch
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets.utils import (
    download_file_from_google_drive,
    extract_archive
)
from PIL import Image


DIV2K_TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_TEST_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
BSDS300_URL = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
BSDS500_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
BSDS100_URL = "1nu78kEKoSTti7ynh8pdxa7ae7TvZiNOy"
BSDS200_URL = "1N9cK1OScGrACUgCms0f2rFlUOHhgkW0l"
SET5_URL = "14g2glfOdkxzZ2RnQZR6jYU5CoClsxQRo"
SET14_URL = "1FSJqQVISh19onL1TUqPNor0uRyp8LlNb"
T91_URL = "1VSG1e5nvdV9UCUSYuaKecNFuk3OPUat4"
HISTORICAL_URL = "1sc14tdRslyZsfw1-LpoOCKF72kSWKedx"
MANGA109_URL = "1bEjcSRiT4V6vxjHjhr_jBPmAr3sGS_5l"
URBAN100_URL = "1svYMEyfc5mkpnW6JnkF0ZS_KetgEYgLR"
GENERAL100_URL = "1tD6XBLkV9Qteo2obMRcRueTRwie7Hqae"


class BaseDataset(torch.utils.data.Dataset):
    """Base Super Resolution Dataset Class

    """

    base_dir: str = ".data"
    color_space: str = "RGB"
    extensions: List[str] = [""]
    lr_transform: T.Compose = None
    hr_transform: T.Compose = None

    def get_lr_transforms(self):
        """Returns HR to LR image transformations

        """
        return Compose([
            Resize(size=(
                    self.image_size//self.scale_factor,
                    self.image_size//self.scale_factor
                ),
                interpolation=Image.BICUBIC
            ),
            ToTensor(),
        ])

    def get_hr_transforms(self):
        """Returns HR image transformations

        """
        return Compose([
            Resize((self.image_size, self.image_size), Image.BICUBIC),
            ToTensor(),
        ])

    def get_files(self, root_dir: str) -> List[str]:
        """Returns  a list of valid image files in a directory

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
        """Returns  a tuple of and lr and hr torch tensors

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
        """Return number of images in dataset

        Returns
        -------
        int
            Number of images in dataset file_names list

        """
        return len(self.file_names)

    def is_valid_file(self, file_path: str) -> bool:
        """Returns boolean if the given `file_path` has a valid image extension

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

    def load_img(self, file_path: str) -> Image.Image:
        """Returns a PIL Image of the image located at `file_path`

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

    def download_google_drive(self, data_dir: str, filename: str) -> None:
        """Download dataset

        Parameters
        ----------
        data_dir : str
            Path to base dataset directory
        filename : str
            Filename of google drive file being downloaded

        Returns
        -------
        None

        """
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if not os.path.exists(self.root_dir):

            download_file_from_google_drive(
                file_id=self.url,
                root=data_dir,
                filename=filename
            )
            extract_archive(
                from_path=os.path.join(data_dir, filename),
                to_path=data_dir,
                remove_finished=True
            )
