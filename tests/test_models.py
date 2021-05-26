import itertools

import pytest
import torch
import torch.nn as nn
from torch_enhance import models

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 32
SCALE_FACTOR = [2, 3, 4]
CHANNELS = [1, 3]
BATCH_SIZE = [1, 2]
MODELS = [
    models.Bicubic, models.SRCNN, models.ESPCN,
    models.EDSR, models.VDSR, models.SRResNet
]
params = list(itertools.product(MODELS, SCALE_FACTOR, CHANNELS, BATCH_SIZE))


@pytest.mark.parametrize("module, scale_factor, channels, batch_size", params)
def test_model(module, scale_factor, channels, batch_size):

    # SRResNet only supports scale_factor 2 or 4
    if scale_factor == 3 and module in [models.SRResNet, models.EDSR]:
        return

    model = module(scale_factor, channels)
    model = model.to(DEVICE)

    lr = torch.ones(batch_size, channels, IMAGE_SIZE, IMAGE_SIZE)
    lr = lr.to(DTYPE)
    lr = lr.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (batch_size, channels, IMAGE_SIZE*scale_factor, IMAGE_SIZE*scale_factor)
    assert sr.dtype == torch.float32


@pytest.mark.parametrize("module, scale_factor, channels, batch_size", params)
def test_enhance(module, scale_factor, channels, batch_size):

    # SRResNet only supports scale_factor 2 or 4
    if scale_factor == 3 and module in [models.SRResNet, models.EDSR]:
        return

    model = module(scale_factor, channels)
    model = model.to(DEVICE)

    lr = torch.ones(batch_size, channels, IMAGE_SIZE, IMAGE_SIZE)
    lr = lr.to(DTYPE)
    lr = lr.to(DEVICE)
    sr = model.enhance(lr)

    if batch_size == 1:
        assert sr.shape == (channels, IMAGE_SIZE*scale_factor, IMAGE_SIZE*scale_factor)
    else:
        assert sr.shape == (batch_size, channels, IMAGE_SIZE*scale_factor, IMAGE_SIZE*scale_factor)
        
    assert sr.dtype == torch.torch.uint8
