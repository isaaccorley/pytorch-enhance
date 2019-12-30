import os
import glob
import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .common import DatasetFolder


class BSDS300(object):
    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        data_dir=None,
        color_space="RGB"
    ):
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.data_dir = data_dir
        self.color_space = color_space
        self.extensions = ["jpg"]
        self.url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"

        if self.data_dir is None:
            self.data_dir = os.path.join(os.getcwd(), "data")

        self.crop_size = self.image_size - (self.image_size % self.scale_factor)

        self.root_dir = self.download()

        self.input_transform = Compose(
            [
                CenterCrop(self.crop_size),
                Resize(self.crop_size // self.scale_factor),
                ToTensor(),
            ]
        )

        self.target_transform = Compose([CenterCrop(self.crop_size), ToTensor()])

    def download(self):

        output_dir = os.path.join(self.data_dir, "BSDS300/images")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(output_dir):
            print("downloading url ", self.url)

            data = urllib.request.urlopen(self.url)

            file_path = os.path.join(self.data_dir, os.path.basename(self.url))
            with open(file_path, "wb") as f:
                f.write(data.read())

            print("Extracting data")
            with tarfile.open(file_path) as tar:
                for item in tar:
                    tar.extract(item, self.data_dir)

            os.remove(file_path)

            # Remove .txt files
            for f in glob.glob('data/BSDS300/*.txt'):
                os.remove(f)

        return output_dir

    def get_dataset(self, train=True):
        root_dir = os.path.join(self.root_dir, "train" if train else "test")
        return DatasetFolder(
            data_dir=root_dir,
            input_transform=self.input_transform,
            target_transform=self.target_transform,
            color_space=self.color_space,
            extensions=self.extensions,
        )
