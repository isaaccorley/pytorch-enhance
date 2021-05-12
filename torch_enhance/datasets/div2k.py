import os
import shutil
from dataclasses import dataclass

import torchvision.transforms as T
from torchvision.datasets.utils import download_and_extract_archive

from .base import BSDS300_URL, BaseDataset


@dataclass()
class DIV2K(BaseDataset):

    scale_factor: int = 2
    image_size: int = 256
    color_space: str = "RGB"
    train: bool = True
    data_dir: str = ""
    lr_transforms: T.Compose = None
    hr_transforms: T.Compose = None

    def __post_init__(self):
        self.url = BSDS300_URL
        self.extensions = [".jpg"]

        if self.data_dir == "":
            self.data_dir = os.path.join(os.getcwd(), self.base_dir)

        self.root_dir = os.path.join(self.data_dir, "DIV2K")
        self.download(self.data_dir)
        self.set_dir = os.path.join(
            self.root_dir, 'train' if self.train else 'test'
        )
        self.file_names = self.get_files(self.set_dir)

        if self.lr_transforms is None:
            self.lr_transform = self.get_lr_transforms()
        if self.hr_transforms is None:
            self.hr_transform = self.get_hr_transforms()

    def download(self, data_dir: str) -> None:
        """Download dataset

        Parameters
        ----------
        data_dir : str
            Path to base dataset directory

        Returns
        -------
        None

        """
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

            download_and_extract_archive(
                self.url,
                data_dir,
                remove_finished=True
            )

            # Tidy up
            for d in ['train', 'val']:
                shutil.move(
                    src=os.path.join(self.root_dir, 'images', d),
                    dst=self.root_dir
                )

            for f in os.listdir(self.root_dir):
                if f not in ['train', 'test']:
                    path = os.path.join(self.root_dir, f)

                    if os.path.isdir(path):
                        _ = shutil.rmtree(path)
                    else:
                        _ = os.remove(path)