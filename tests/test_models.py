import pytest

import torch
from torch_enhance import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE_FACTOR = 2
x = torch.ones(1, 3, 32, 32)
x = x.to(torch.float32)
x = x.to(DEVICE)

def test_bicubic():
    model = models.Bicubic(scale_factor=SCALE_FACTOR)
    model = model.to(DEVICE)
    y_pred = model(x)
    assert y_pred.shape == (1, 3, 64, 64)
    assert y_pred.dtype == torch.float32

def test_edsr():
    model = models.EDSR(scale_factor=SCALE_FACTOR)
    model = model.to(DEVICE)
    y_pred = model(x)
    assert y_pred.shape == (1, 3, 64, 64)
    assert y_pred.dtype == torch.float32

def test_espcn():
    model = models.ESPCN(scale_factor=SCALE_FACTOR)
    model = model.to(DEVICE)
    y_pred = model(x)
    assert y_pred.shape == (1, 3, 64, 64)
    assert y_pred.dtype == torch.float32

def test_srcnn():
    model = models.SRCNN(scale_factor=SCALE_FACTOR)
    model = model.to(DEVICE)
    y_pred = model(x)
    assert y_pred.shape == (1, 3, 64, 64)
    assert y_pred.dtype == torch.float32

def test_srresnet():
    model = models.SRResNet(scale_factor=SCALE_FACTOR)
    model = model.to(DEVICE)
    y_pred = model(x)
    assert y_pred.shape == (1, 3, 64, 64)
    assert y_pred.dtype == torch.float32

def test_vdsr():
    model = models.VDSR(scale_factor=SCALE_FACTOR)
    model = model.to(DEVICE)
    y_pred = model(x)
    assert y_pred.shape == (1, 3, 64, 64)
    assert y_pred.dtype == torch.float32