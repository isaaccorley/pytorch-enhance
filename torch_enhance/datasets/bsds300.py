import os
import shutil

from .common import BSDS300_URL, SRDataset
from torchvision.datasets.utils import download_and_extract_archive

class BSDS300(SRDataset):

    url = BSDS300_URL
    extensions = ['.jpg']

    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        color_space='RGB',
        train=True,
        data_dir=None,
        lr_transforms=None,
        hr_transforms=None
    ):
        super(BSDS300, self).__init__()
        
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.color_space = color_space
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms

        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), self.base_dir)

        self.root_dir = os.path.join(data_dir, 'BSDS300')
        self.download(data_dir)
        self.set_dir = os.path.join(self.root_dir, 'train' if train else 'test')
        self.file_names = self.get_files(self.set_dir)

        if self.lr_transforms is None:
            self.lr_transform = self.get_lr_transforms()
        if self.hr_transforms is None:
            self.hr_transform = self.get_hr_transforms()

    def download(self, data_dir):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
            download_and_extract_archive(self.url, data_dir, remove_finished=True)

            # Tidy up
            for d in ['train', 'test']:
                shutil.move(src=os.path.join(self.root_dir, 'images', d), dst=self.root_dir)
            
            for f in os.listdir(self.root_dir):
                if f not in ['train', 'test']:
                    path = os.path.join(self.root_dir, f)
                    if os.path.isdir(path):
                        _ = shutil.rmtree(path)
                    else:
                        _ = os.remove(path)