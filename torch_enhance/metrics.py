import torch
import torch.nn as nn
import torch.nn.functional as F


def mse(y_pred, y_true):
    """ Mean squared error metric """
    return F.mse_loss(y_pred, y_true)

def mae(y_pred, y_true):
    """ Mean absolute error metric """
    return F.l1_loss(y_pred, y_true)

def psnr(y_pred, y_true):
    """ Peak-signal-noise ratio metric """
    10 * (1 / mse(y_pred, y_true)).log10()
