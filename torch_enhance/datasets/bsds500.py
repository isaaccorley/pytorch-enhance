import os
import shutil
import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .common import DatasetFolder


class BSDS500(object):
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
        self.url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

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

        output_dir = os.path.join(self.data_dir, "BSDS500/images")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
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

            # Grab only the image sets
            image_dir = os.path.join(self.data_dir, "BSR/BSDS500/data/images/")

            for dataset in ['train', 'val', 'test']:
                shutil.copytree(
                    src=os.path.join(image_dir, dataset),
                    dst=os.path.join(output_dir, dataset)
                    )
                os.remove(os.path.join(output_dir, dataset, 'Thumbs.db'))
                
            shutil.rmtree(os.path.join(self.data_dir, 'BSR'))

        return output_dir

    def get_dataset(self, set_type='train'):

        assert set_type in ['train', 'val', 'test']
        root_dir = os.path.join(self.root_dir, set_type)
        return DatasetFolder(
            data_dir=root_dir,
            input_transform=self.input_transform,
            target_transform=self.target_transform,
            color_space=self.color_space,
            extensions=self.extensions,
        )
