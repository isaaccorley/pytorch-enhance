import torch
from torch.utils.data import DataLoader

from poutyne.framework import Model

from torch_enhance.datasets import BSDS300, Set14, Set5
from torch_enhance.models import SRCNN
from torch_enhance import metrics


scale_factor = 2
train_dataset = BSDS300(scale_factor=scale_factor)
val_dataset = Set14(scale_factor=scale_factor)
train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=2)

channels = 3 if train_dataset.color_space == "RGB" else 1
pytorch_network = SRCNN(scale_factor, channels)

model = Model(
    pytorch_network,
    "sgd",
    "mse"
)
model.fit_generator(
    train_dataloader,
    val_dataloader,
    epochs=1
)
