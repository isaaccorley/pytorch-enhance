import torchvision


__all__ = ["plot_compare"]


def plot_compare(sr, hr, baseline, filename):
    """
    Plot Super-Resolution and High-Resolution image comparison
    """
    sr, hr, baseline = sr.squeeze(), hr.squeeze(), baseline.squeeze()
    grid = torchvision.utils.make_grid([hr, baseline, sr])
    torchvision.utils.save_image(grid, filename)
