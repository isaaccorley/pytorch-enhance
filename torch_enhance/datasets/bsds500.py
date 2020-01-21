import os
import shutil

from .common import BSDS500_URL, SRDataset
from torchvision.datasets.utils import download_and_extract_archive


class BSDS500(SRDataset):

    url = BSDS500_URL
    extensions = ['.jpg']

    def __init__(
        self,
        scale_factor=2,
        image_size=256,
        color_space='RGB',
        set_type='train',
        data_dir=os.path.join(os.getcwd(), 'datasets')
    ):
        super(BSDS500, self).__init__()
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.color_space = color_space

        self.root_dir = os.path.join(data_dir, 'BSDS500')
        self.download(data_dir)
        self.set_dir = os.path.join(self.root_dir, set_type)
        self.file_names = self.get_files(self.set_dir)

        self.lr_transform = self.get_lr_transforms()
        self.hr_transform = self.get_hr_transforms()

    def download(self, data_dir):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
            download_and_extract_archive(self.url, data_dir, remove_finished=True)

            # Tidy up
            for d in ['train', 'val', 'test']:
                shutil.move(src=os.path.join(data_dir, 'BSR/BSDS500/data/images', d),
                            dst=self.root_dir)
                os.remove(os.path.join(self.root_dir, d, 'Thumbs.db'))
                
            shutil.rmtree(os.path.join(data_dir, 'BSR'))
