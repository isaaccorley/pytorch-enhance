import os

from .base import HISTORICAL_URL, BaseDataset


class Historical(BaseDataset):

    url = HISTORICAL_URL
    extensions = ['.png']

    def __init__(
        self,
        scale_factor: int = 2,
        image_size: int = 256,
        color_space: str = 'L',
        data_dir: str = '',
        lr_transforms=None,
        hr_transforms=None
    ):
        super(Historical, self).__init__()

        self.scale_factor = scale_factor
        self.image_size = image_size
        self.color_space = color_space
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms

        if data_dir == '':
            data_dir = os.path.join(os.getcwd(), self.base_dir)
            
        self.root_dir = os.path.join(data_dir, 'historical')
        self.download_google_drive(data_dir, filename="historical.zip")
        self.file_names = self.get_files(self.root_dir)

        if self.lr_transforms is None:
            self.lr_transform = self.get_lr_transforms()
        if self.hr_transforms is None:
            self.hr_transform = self.get_hr_transforms()
