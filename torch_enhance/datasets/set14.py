import os

from .common import SET14_URL, SRDataset
from torchvision.datasets.utils import download_and_extract_archive


class Set14(SRDataset):

    url = SET14_URL
    extensions = ['.png']

    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        color_space='RGB',
        data_dir=os.path.join(os.getcwd(), 'datasets')
    ):
        super(Set14, self).__init__()
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.color_space = color_space
        
        self.root_dir = os.path.join(data_dir, 'Set14')
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
