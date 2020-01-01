import os
import torch
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


BSDS300_URL = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
BSDS500_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
SET5_URL = 'https://raw.github.com/IsaacCorley/pytorch-enhance/master/data/Set5.zip'
SET14_URL = 'https://raw.github.com/IsaacCorley/pytorch-enhance/master/data/Set14.zip'
T91_URL = 'https://raw.github.com/IsaacCorley/pytorch-enhance/master/data/T91.zip'
HISTORICAL_URL = 'https://raw.github.com/IsaacCorley/pytorch-enhance/master/data/historical.zip'


class SRDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(SRDataset, self).__init__()

        self.root_dir = 'data'
        self.color_space = 'RGB'
        self.extensions = ['']
        self.lr_transform = None
        self.hr_transform = None

    def get_lr_transforms(self):
        return Compose(
            [
                Resize((self.image_size//self.scale_factor,
                        self.image_size//self.scale_factor),
                        Image.BICUBIC),
                ToTensor(),
            ]
        )

    def get_hr_transforms(self):
        return Compose(
            [
                Resize((self.image_size, self.image_size), Image.BICUBIC),
                ToTensor(),
            ]
        )
        
    def get_files(self, root_dir):
        return [
            os.path.join(root_dir, x)
            for x in os.listdir(root_dir)
            if self.is_valid_file(x)
        ]

    def __getitem__(self, idx):
        lr = self.load_img(self.file_names[idx])
        hr = lr.copy()
        if self.lr_transform:
            lr = self.lr_transform(lr)
        if self.hr_transform:
            hr = self.hr_transform(hr)

        return lr, hr

    def __len__(self):
        return len(self.file_names)

    def is_valid_file(self, file_name):
        return any(file_name.endswith(ext) for ext in self.extensions)

    def load_img(self, file_path):
        return Image.open(file_path).convert(self.color_space)
