import torch
import torch.nn.functional as F


@torch.no_grad()
def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean squared error (MSE) metric

    Parameters
    ----------
    y_pred : torch.Tensor
        Super-Resolved image tensor
    y_true : torch.Tensor
        High Resolution image tensor

    Returns
    -------
    torch.Tensor
        Mean squared error between y_true and y_pred

    """
    return F.mse_loss(y_pred, y_true)

@torch.no_grad()
def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean absolute error (MAE) metric

    Parameters
    ----------
    y_pred : torch.Tensor
        Super-Resolved image tensor
    y_true : torch.Tensor
        High Resolution image tensor

    Returns
    -------
    torch.Tensor
        Mean absolute error between y_true and y_pred

    """
    return F.l1_loss(y_pred, y_true)

@torch.no_grad()
def psnr(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Peak-signal-noise ratio (PSNR) metric

    Parameters
    ----------
    y_pred : torch.Tensor
        Super-Resolved image tensor
    y_true : torch.Tensor
        High Resolution image tensor

    Returns
    -------
    torch.Tensor
        Peak-signal-noise-ratio between y_true and y_pred

    """
    return 10 * (1 / mse(y_pred, y_true)).log10()
