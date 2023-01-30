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

    psnr_image = 20 * np.log10(2 ** 12 / np.sqrt(mse))
    psnr_image[mse_mask] = 100
    return psnr_image


def fod_comparison(
    gt_fod, m_fods, roi=None
):
    mse_list = []
    mae_list = []
    psnr_list = []
    for m_fod_i in m_fods:
        mse_list.append(mean_squared_error(gt_fod, m_fod_i, roi))
        mae_list.append(mean_absolute_error(gt_fod, m_fod_i, roi))
        psnr_list.append(psnr(gt_fod, m_fod_i, roi))

    return mse_list, mae_list, psnr_list
