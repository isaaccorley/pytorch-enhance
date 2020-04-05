import torch
import torchvision
from torchvision.utils import make_grid


def plot_compare(sr, hr, baseline, filename):
    """ Plot side by side comparison """
    sr, hr, baseline = sr.squeeze(), hr.squeeze(), baseline.squeeze()
    grid = torchvision.utils.make_grid([hr, baseline, sr])
    torchvision.utils.save_image(grid, filename)