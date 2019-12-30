import os
import torch
from PIL import Image


class DatasetFolder(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        input_transform=None,
        target_transform=None,
        color_space="RGB",
        extensions=[""],
    ):
        super(DatasetFolder, self).__init__()

        self.data_dir = data_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.color_space = color_space
        self.extensions = extensions

        self.file_names = [
            os.path.join(self.data_dir, x)
            for x in os.listdir(self.data_dir)
            if self.is_valid_file(x)
        ]

    def __getitem__(self, idx):
        x = load_img(self.file_names[idx])
        y = x.copy()
        if self.input_transform:
            x = self.input_transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.file_names)

    def is_image_file(self, file_name):
        return any(file_name.endswith(ext) for ext in self.extensions)

    def load_img(self, file_path):
        return Image.open(file_path).convert(self.color_space)
