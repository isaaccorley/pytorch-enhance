import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch_enhance.datasets import BSDS300
from torch_enhance.models import SRCNN


class Enhance(pl.LightningModule):
    def __init__(self, model, dataset, scale_factor):
        super(Enhance, self).__init__()
        self.model = model
        self.dataset = dataset
        self.scale_factor = scale_factor
        self.loss = self.model.loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-3)]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset.get_dataset(train=True), batch_size=2)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset.get_dataset(train=False), batch_size=2)


scale_factor = 2
data = BSDS300(scale_factor)
model = SRCNN(scale_factor)
module = Enhance(model, data, scale_factor)
trainer = pl.Trainer()
trainer.fit(module)
