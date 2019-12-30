import os
import shutil
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .common import BSDS300_URL, DatasetFolder
from .utils import download_and_extract_archive


class BSDS300(object):
    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        data_dir=os.path.join(os.getcwd(), 'data'),
        color_space='RGB'
    ):
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.root_dir = os.path.join(data_dir, 'BSDS300')
        self.color_space = color_space
        self.extensions = ['.jpg']
        self.url = BSDS300_URL

        self.crop_size = self.image_size - (self.image_size % self.scale_factor)

        self.download(data_dir)

        self.input_transform = Compose(
            [
                CenterCrop(self.crop_size),
                Resize(self.crop_size // self.scale_factor),
                ToTensor(),
            ]
        )

        self.target_transform = Compose([CenterCrop(self.crop_size), ToTensor()])

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

    def get_dataset(self, train=True):
        root_dir = os.path.join(self.root_dir, "train" if train else "test")
        return DatasetFolder(
            data_dir=root_dir,
            input_transform=self.input_transform,
            target_transform=self.target_transform,
            color_space=self.color_space,
            extensions=self.extensions,
        )
