import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from torch_enhance.datasets import BSDS300, Set14, Set5
from torch_enhance.models import SRCNN
from torch_enhance import metrics


class Module(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")
        
        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)

        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")
        
        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        self.log("val_loss", loss)
        self.log("val_mae", mae)
        self.log("val_psnr", psnr)

        return loss

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")
        
        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        self.log("test_psnr", psnr)

        return loss


if __name__ == '__main__':
    
    scale_factor = 2

    # Setup dataloaders
    train_dataset = BSDS300(scale_factor=scale_factor)
    val_dataset = Set14(scale_factor=scale_factor)
    test_dataset = Set5(scale_factor=scale_factor)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Define model
    channels = 3 if train_dataset.color_space == "RGB" else 1
    model = SRCNN(scale_factor, channels)
    module = Module(model)

    trainer = pl.Trainer(max_epochs=5, gpus=1)
    trainer.fit(
        module,
        train_dataloader,
        val_dataloader
    )
    trainer.test(module, test_dataloader)
