import os
import shutil
import glob
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .common import T91_URL, DatasetFolder
from .utils import download_and_extract_archive


class T91(object):
    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        data_dir=os.path.join(os.getcwd(), 'data'),
        color_space='RGB'
    ):
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.root_dir = os.path.join(data_dir, 'T91')
        self.color_space = color_space
        self.extensions = ['.png']
        self.url = T91_URL

        self.__download(data_dir)

        self.lr_transform = Compose(
            [
                Resize((self.image_size//self.scale_factor,
                        self.image_size//self.scale_factor),
                        Image.BICUBIC),
                ToTensor(),
            ]
        )

        self.hr_transform = Compose(
            [
                Resize((self.image_size, self.image_size), Image.BICUBIC),
                ToTensor(),
            ]
        )

    def __download(self, data_dir):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

            download_and_extract_archive(self.url, data_dir, remove_finished=True)

    def get_dataset(self):
        return DatasetFolder(
            data_dir=self.root_dir,
            lr_transform=self.lr_transform,
            hr_transform=self.hr_transform,
            color_space=self.color_space,
            extensions=self.extensions,
        )
