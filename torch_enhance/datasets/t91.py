import os

from .common import T91_URL, SRDataset
from torchvision.datasets.utils import download_and_extract_archive


class T91(SRDataset):

    url = T91_URL
    extensions = ['.png']

    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        color_space='RGB',
        data_dir=None
    ):
        super(T91, self).__init__()
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.color_space = color_space

        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), self.base_dir)

        self.root_dir = os.path.join(data_dir, 'T91')
        self.download(data_dir)
        self.file_names = self.get_files(self.root_dir)

        self.lr_transform = self.get_lr_transforms()
        self.hr_transform = self.get_hr_transforms()

    def download(self, data_dir):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

            download_and_extract_archive(self.url, data_dir, remove_finished=True)
