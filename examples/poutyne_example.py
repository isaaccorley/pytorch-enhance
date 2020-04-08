from poutyne.framework import Model
import torch
import numpy as np

from torch_enhance.datasets import BSDS300, Set14, Set5
from torch_enhance.models import SRCNN
from torch_enhance import metrics


scale_factor = 2
train_dataset = BSDS300(scale_factor=scale_factor, data_dir="../.data/")
val_dataset = Set14(scale_factor=scale_factor, data_dir="../.data/")
test_dataset = Set5(scale_factor=scale_factor, data_dir="../.data/")
pytorch_network = SRCNN(scale_factor)

model = Model(
    pytorch_network,
    "sgd",
    "mse",
    batch_metrics=["mse"],
    epoch_metrics=["mse"]
)
model.fit(
    train_x, train_y,
    validation_data=(valid_x, valid_y),
    epochs=5,
    batch_size=32
)