import torch
import torch.nn.functional as F


@torch.no_grad()
def mse(y_pred, y_true):
    """ Mean squared error metric """
    return F.mse_loss(y_pred, y_true)

@torch.no_grad()
def mae(y_pred, y_true):
    """ Mean absolute error metric """
    return F.l1_loss(y_pred, y_true)

@torch.no_grad()
def psnr(y_pred, y_true):
    """ Peak-signal-noise ratio metric """
    return 10 * (1 / mse(y_pred, y_true)).log10()
