import os
from dataclasses import dataclass

import torchvision.transforms as T

from .base import BSDS100_URL, BaseDataset


@dataclass()
class BSDS100(BaseDataset):

    scale_factor: int = 2
    image_size: int = 256
    color_space: str = "RGB"
    data_dir: str = ""
    lr_transforms: T.Compose = None
    hr_transforms: T.Compose = None

    def __post_init__(self):

        self.url = BSDS100_URL
        self.extensions = [".png"]

        if self.data_dir == "":
            self.data_dir = os.path.join(os.getcwd(), self.base_dir)

        self.root_dir = os.path.join(self.data_dir, "BSDS100")
        self.download_google_drive(self.data_dir, filename="BSDS100.zip")
        self.file_names = self.get_files(self.root_dir)

        if self.lr_transforms is None:
            self.lr_transform = self.get_lr_transforms()
        if self.hr_transforms is None:
            self.hr_transform = self.get_hr_transforms()
