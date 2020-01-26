import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch_enhance.datasets import BSDS300, Set5
from torch_enhance.models import SRCNN
from torch_enhance import metrics


class Enhance(pl.LightningModule):
    def __init__(self, model, dataset, scale_factor, test_set=None):
        super(Enhance, self).__init__()
        self.model = model
        self.dataset = dataset
        self.scale_factor = scale_factor
        self.test_set = test_set
        self.loss = self.model.loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss(sr, hr)
        return {
            "loss": loss,
            "mse": metrics.mse(sr, hr),
            "psnr": metrics.psnr(sr, hr)
            }

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss(sr, hr)
        return {
            "val_loss": loss,
            "val_mse": metrics.mse(sr, hr),
            "val_psnr": metrics.psnr(sr, hr)
            }

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss(sr, hr)
        return {
            "test_loss": loss,
            "test_mse": metrics.mse(sr, hr),
            "test_psnr": metrics.psnr(sr, hr)
            }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['val_mse'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_mse': avg_mse,
            'val_psnr': avg_psnr
            }
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['test_psnr'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'test_loss': avg_loss,
            'test_mse': avg_mse,
            'test_psnr': avg_psnr
            }
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-3)]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset.get_dataset(train=True), batch_size=2)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset.get_dataset(train=False), batch_size=2)

    @pl.data_loader
    def test_dataloader(self):
        if self.test_set is None:
            return DataLoader(self.dataset.get_dataset(train=False), batch_size=2)
        else:
            return DataLoader(self.test_set.get_dataset(), batch_size=1)



if __name__ = '__main__':
    scale_factor = 2
    data = BSDS300(scale_factor)
    test_set = Set5(scale_factor)
    model = SRCNN(scale_factor)
    module = Enhance(model, data, scale_factor, test_set)
    trainer = pl.Trainer(max_nb_epochs=1, train_percent_check=0.1, val_percent_check=0.1)
    trainer.fit(module)
    trainer.test()