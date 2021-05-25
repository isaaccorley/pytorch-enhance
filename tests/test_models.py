import pytest

import torch
from torch_enhance import models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE_FACTOR = 2
CHANNELS = 3
lr = torch.ones(1, CHANNELS, 32, 32)
lr = lr.to(torch.float32)
lr = lr.to(DEVICE)

def test_bicubic():
    model = models.Bicubic(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64)
    assert sr.dtype == torch.float32

def test_edsr():
    model = models.EDSR(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64)
    assert sr.dtype == torch.float32

def test_espcn():
    model = models.ESPCN(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64)
    assert sr.dtype == torch.float32

def test_srcnn():
    model = models.SRCNN(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64)
    assert sr.dtype == torch.float32

def test_srresnet():
    model = models.SRResNet(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64)
    assert sr.dtype == torch.float32

def test_vdsr():
    model = models.VDSR(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64)
    assert sr.dtype == torch.float32

def test_enhance():
    model = models.SRCNN(scale_factor=SCALE_FACTOR, channels=CHANNELS)
    model = model.to(DEVICE)
    sr = model.enhance(lr)
    assert sr.shape == (64, 64, 3)
    assert sr.dtype == torch.uint8

    sr = model.enhance(lr.squeeze(0))
    assert sr.shape == (64, 64, 3)