import os
import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .common import DatasetFromFolder



class BSDS300(object):

    def __init__(self, scale_factor, image_size=256, dest=None, url=None):

        self.scale_factor = scale_factor
        self.image_size = image_size
        self.dest = dest
        self.url = url

        if self.dest is None:
            self.dest = os.path.join(os.getcwd(), 'data')

        if self.url is None:
            self.url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        self.crop_size = self.image_size - (self.image_size % self.scale_factor)

        self.root_dir = self.download_bsds300()

        self.input_transform = Compose([
            CenterCrop(self.crop_size),
            Resize(self.crop_size // self.scale_factor),
            ToTensor()
            ])

        self.target_transform = Compose([
            CenterCrop(self.crop_size),
            ToTensor()
            ])

    def download_bsds300(self):

        output_image_dir = os.path.join(self.dest, "BSDS300/images")

        if not os.path.exists(output_image_dir):
            os.makedirs(self.dest)
            print("downloading url ", self.url)

            data = urllib.request.urlopen(self.url)

            file_path = os.path.join(self.dest, os.path.basename(self.url))
            with open(file_path, "wb") as f:
                f.write(data.read())

            print("Extracting data")
            with tarfile.open(file_path) as tar:
                for item in tar:
                    tar.extract(item, self.dest)

            os.remove(file_path)

        return output_image_dir


    def get_dataset(self, train=True):
        data_dir = os.path.join(self.root_dir, 'train' if train else 'test')
        return DatasetFromFolder(
            data_dir,
            input_transform=self.input_transform,
            target_transform=self.target_transform
        )
