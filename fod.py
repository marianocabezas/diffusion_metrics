import math
import numpy as np


def mean_squared_error(source, target, roi=None):
    if roi is not None:
        source = source[roi.astype(bool)]
        target = target[roi.astype(bool)]

    return np.sum((target - source) ** 2, axis=1)


def mean_absolute_error(source, target, roi=None):
    if roi is not None:
        source = source[roi.astype(bool)]
        target = target[roi.astype(bool)]

    return np.sum(np.abs(target - source), axis=1)


def psnr(source, target, roi=None):
    mse = mean_squared_error(source, target, roi)
    mse_mask = mse < 1.0e-10

    psnr_image = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    psnr_image[mse_mask] = 100
    return psnr_image