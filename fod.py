import math
import numpy as np


def mean_squared_error(target, source, roi=None):
    if roi is not None:
        source = source[roi.astype(bool)]
        target = target[roi.astype(bool)]

    return np.sum((target - source) ** 2, axis=1)


def mean_absolute_error(target, source, roi=None):
    if roi is not None:
        source = source[roi.astype(bool)]
        target = target[roi.astype(bool)]

    return np.sum(np.abs(target - source), axis=1)


def psnr(target, source, roi=None, pixel_max=1):
    mse = mean_squared_error(
        source, target, roi
    )
    mse_norm = mse / mse.size
    mse_mask = mse < 1.0e-10

    psnr_image = 20 * np.log10(pixel_max / np.sqrt(mse_norm))
    psnr_image[mse_mask] = 100
    return psnr_image


def angular_correlation(target, source, roi=None):
    if roi is None:
        pred_fod = source[..., 1:]
        gt_fod = target[..., 1:]
    else:
        pred_fod = source[roi.astype(bool), 1:]
        gt_fod = target[roi.astype(bool), 1:]

    numerator = np.sum(pred_fod * gt_fod, axis=-1)
    denominator = np.sqrt(
        np.sum(pred_fod ** 2, axis=-1)
    ) * np.sqrt(
        np.sum(gt_fod ** 2, axis=-1)
    )
    return numerator/denominator


def fod_comparison(
    target_fod, source_fods, roi=None
):
    mse_list = []
    mae_list = []
    psnr_list = []
    acc_list = []
    for m_fod_i in source_fods:
        mse_list.append(mean_squared_error(target_fod, m_fod_i, roi))
        mae_list.append(mean_absolute_error(target_fod, m_fod_i, roi))
        psnr_list.append(psnr(target_fod, m_fod_i, roi))
        acc_list.append(angular_correlation(target_fod, m_fod_i, roi))

    return mse_list, mae_list, psnr_list, acc_list
